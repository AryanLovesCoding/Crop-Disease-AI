import numpy as np
import joblib
from PIL import Image
from torchvision import transforms

FEATURE_COLS = [
    "soil_pH", "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "rainfall",
    "crop_age_days", "sunlight_hours"
]

def get_inference_transform():
    """Image transform for inference — no augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    transform = get_inference_transform()
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def preprocess_tabular(features_dict,
                       scaler_path="models/tabular_scaler_v2.pkl"):
    """
    Preprocess tabular features for inference.
    
    Args:
        features_dict: dict with keys matching FEATURE_COLS
        scaler_path: path to fitted StandardScaler
    
    Returns:
        numpy array of shape (1, 9)
    """
    scaler = joblib.load(scaler_path)
    values = np.array([[features_dict[col] for col in FEATURE_COLS]])
    return scaler.transform(values)

def load_label_encoder(path="models/label_encoder_v2.pkl"):
    """Load the tabular label encoder."""
    return joblib.load(path)
