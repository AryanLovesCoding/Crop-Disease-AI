import torch
import torch.nn as nn
import numpy as np
import joblib
from torchvision import models


class FusionMLP(nn.Module):
    """Multimodal fusion MLP combining image and tabular features."""
    def __init__(self, input_dim=2057, num_classes=15):
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


def load_image_model(weights_path="models/resnet50_balanced.pth",
                     num_classes=15):
    """Load fine-tuned ResNet50 classifier."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def get_embedding_model(resnet):
    """Strip final layer to get 2048-d feature extractor."""
    embedding_model = nn.Sequential(*list(resnet.children())[:-1])
    embedding_model.eval()
    return embedding_model


def load_fusion_model(weights_path="models/fusion_mlp_best.pth",
                      input_dim=2057, num_classes=15):
    """Load trained fusion MLP."""
    model = FusionMLP(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def load_xgboost_model(model_path="models/xgboost_tuned.pkl"):
    """Load tuned XGBoost tabular model."""
    return joblib.load(model_path)


def predict_fusion(image_tensor, tabular_array,
                   embedding_model, fusion_model, image_classes):
    """
    Run full fusion inference.

    Args:
        image_tensor:    preprocessed image tensor (1, 3, 224, 224)
        tabular_array:   scaled tabular features (1, 9)
        embedding_model: ResNet50 feature extractor
        fusion_model:    FusionMLP
        image_classes:   list of class names

    Returns:
        class_name (str), confidence_pct (float)
    """
    with torch.inference_mode():
        embedding = embedding_model(image_tensor)
        embedding = embedding.squeeze(-1).squeeze(-1).numpy()
        fused = np.concatenate([embedding, tabular_array], axis=1)
        fused_tensor = torch.tensor(fused, dtype=torch.float32)
        outputs = fusion_model(fused_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    class_name = image_classes[predicted.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    return class_name, confidence_pct
