import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import pickle
import io
import base64
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(title="Crop Disease AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load everything at startup ──
with open("data/processed/data_splits.pkl", "rb") as f:
    image_data = pickle.load(f)
IMAGE_CLASSES = image_data["classes"]

with open("data/treatments.json") as f:
    TREATMENTS = json.load(f)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet50 — keep full model for GradCAM
RESNET = models.resnet50(weights=None)
RESNET.fc = nn.Linear(2048, len(IMAGE_CLASSES))
RESNET.load_state_dict(torch.load("models/resnet50_finetuned.pth",
                                   map_location="cpu"))
RESNET.eval()

# Feature extractor (no final layer)
EMBEDDING_MODEL = nn.Sequential(*list(RESNET.children())[:-1])
EMBEDDING_MODEL.eval()

# GradCAM on last conv layer
GRAD_CAM = GradCAM(model=RESNET, target_layers=[RESNET.layer4[-1]])

# Fusion MLP
class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.network(x)

FUSION_MODEL = FusionMLP(input_dim=2057, num_classes=15)
FUSION_MODEL.load_state_dict(torch.load("models/fusion_mlp_best.pth",
                                         map_location="cpu"))
FUSION_MODEL.eval()

SCALER = joblib.load("models/tabular_scaler_v2.pkl")

print("All models loaded successfully!")

def image_to_base64(img_array):
    """Convert numpy image array to base64 string."""
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/")
def root():
    return {"message": "Crop Disease AI API is running!", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/classes")
def get_classes():
    return {"classes": IMAGE_CLASSES, "total": len(IMAGE_CLASSES)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    soil_pH: float = 6.5,
    nitrogen: float = 80.0,
    phosphorus: float = 60.0,
    potassium: float = 70.0,
    temperature: float = 28.0,
    humidity: float = 65.0,
    rainfall: float = 120.0,
    crop_age_days: float = 45.0,
    sunlight_hours: float = 7.0
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Load image
    contents = await file.read()
    import tempfile, os
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        orig_img = Image.open(tmp_path).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400,
                            detail="Could not read image. Please upload JPG or PNG.")
    finally:
        os.unlink(tmp_path)

    # Preprocess
    orig_resized = orig_img.resize((224, 224))
    orig_np = np.array(orig_resized) / 255.0
    img_tensor = TRANSFORM(orig_img).unsqueeze(0)

    # Get image embedding
    with torch.no_grad():
        embedding = EMBEDDING_MODEL(img_tensor)
        embedding = embedding.squeeze(-1).squeeze(-1).numpy()

    # Preprocess tabular
    tab_values = np.array([[soil_pH, nitrogen, phosphorus, potassium,
                            temperature, humidity, rainfall,
                            crop_age_days, sunlight_hours]])
    tab_scaled = SCALER.transform(tab_values)

    # Fuse and predict
    fused = np.concatenate([embedding, tab_scaled], axis=1)
    fused_tensor = torch.tensor(fused, dtype=torch.float32)

    with torch.no_grad():
        outputs = FUSION_MODEL(fused_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    disease = IMAGE_CLASSES[predicted.item()]
    confidence_pct = round(confidence.item() * 100, 2)

    # Generate GradCAM using ResNet prediction
    with torch.no_grad():
        resnet_out = RESNET(img_tensor)
        _, resnet_pred = resnet_out.max(1)

    grayscale_cam = GRAD_CAM(
        input_tensor=img_tensor,
        targets=[ClassifierOutputTarget(resnet_pred.item())]
    )[0]
    cam_overlay = show_cam_on_image(orig_np.astype(np.float32),
                                    grayscale_cam, use_rgb=True)
    gradcam_b64 = image_to_base64(cam_overlay)

    # Get treatment
    treatment = TREATMENTS.get(disease, {
        "severity": "Unknown",
        "description": "No information available.",
        "treatment": "Consult an agricultural expert.",
        "prevention": "Monitor crop regularly.",
        "urgency": "Consult expert"
    })

    return {
        "disease": disease,
        "confidence": confidence_pct,
        "severity": treatment["severity"],
        "description": treatment["description"],
        "treatment": treatment["treatment"],
        "prevention": treatment["prevention"],
        "urgency": treatment["urgency"],
        "gradcam_image": gradcam_b64
    }
