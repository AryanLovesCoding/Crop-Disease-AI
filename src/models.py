import torch
import torch.nn as nn
import numpy as np
import joblib
import pickle
import xgboost as xgb
from torchvision import models

def load_image_model(weights_path="models/resnet50_finetuned.pth", num_classes=15):
    """Load fine-tuned ResNet50 for feature extraction."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def get_embedding_model(resnet):
    """Strip the final layer to get a feature extractor."""
    embedding_model = nn.Sequential(*list(resnet.children())[:-1])
    embedding_model.eval()
    return embedding_model

def load_fusion_model(weights_path="models/fusion_mlp_best.pth",
                      input_dim=2057, num_classes=15):
    """Load the fusion MLP."""
    class FusionMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.network(x)

    model = FusionMLP(input_dim, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def load_xgboost_model(model_path="models/xgboost_tabular_v2.pkl"):
    """Load the XGBoost tabular model."""
    import joblib
    model = joblib.load(model_path)
    return model

def predict_fusion(image_tensor, tabular_array,
                   embedding_model, fusion_model, image_classes):
    """
    Run full fusion inference.
    Returns: predicted class name, confidence score
    """
    with torch.no_grad():
        # Get image embedding
        embedding = embedding_model(image_tensor)
        embedding = embedding.squeeze(-1).squeeze(-1).numpy()

        # Concatenate with tabular features
        fused = np.concatenate([embedding, tabular_array], axis=1)
        fused_tensor = torch.tensor(fused, dtype=torch.float32)

        # Run fusion model
        outputs = fusion_model(fused_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

        class_name = image_classes[predicted.item()]
        confidence_pct = round(confidence.item() * 100, 2)

    return class_name, confidence_pct
