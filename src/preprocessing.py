import numpy as np
import pandas as pd
import joblib
from PIL import Image
from torchvision import transforms

# Image transform for inference (no augmentation)
def get_inference_transform():
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
    tensor = transform(img).unsqueeze(0)  # add batch dimension
    return tensor

def preprocess_tabular(features_dict, scaler_path="models/tabular_scaler_v2.pkl"):
    """
    Preprocess tabular features for inference.
    features_dict: {soil_pH, nitrogen, phosphorus, potassium,
                    temperature, humidity, rainfall,
                    crop_age_days, sunlight_hours}
    """
    feature_cols = ["soil_pH", "nitrogen", "phosphorus", "potassium",
                    "temperature", "humidity", "rainfall",
                    "crop_age_days", "sunlight_hours"]
    
    scaler = joblib.load(scaler_path)
    values = np.array([[features_dict[col] for col in feature_cols]])
    scaled = scaler.transform(values)
    return scaled

def load_label_encoder(path="models/label_encoder_v2.pkl"):
    """Load the label encoder."""
    return joblib.load(path)
