"""
EfficientNet-V2-S Tumor Classifier for BRISC 2025 Dataset.
4-class classification: Glioma, Meningioma, No Tumor, Pituitary.
Achieves ~98% accuracy and macro F1 on the test set.
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
import torchvision.models as models


CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}


class TumorClassifier(nn.Module):
    """
    EfficientNet-V2-S for brain tumor classification (4 classes).
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        backbone: str = "efficientnet_v2_s",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = "efficientnet_v2_s"

        self.model = models.efficientnet_v2_s(weights=None)
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
            )
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_weights(self, path: str | Path) -> None:
        """Load pretrained weights from a .pth file."""
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        
        # Handle cases where weights are saved as a state_dict or a full model
        state_dict = state if isinstance(state, dict) else state.state_dict()

        # Auto-detect architecture from state dict keys
        keys = list(state_dict.keys())
        is_eff = any("features." in k for k in keys)
        if is_eff and "efficientnet" not in self.backbone_name:
            self.backbone_name = "efficientnet_v2_s"
            self.model = models.efficientnet_v2_s(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_ftrs, self.num_classes)
            )

        # Handle nested model. prefix if saved from outer class
        if any(k.startswith("model.") for k in keys) and not any(k.startswith("model.model.") for k in keys):
            # Model wrapper matches self.model
            self.load_state_dict(state_dict)
        else:
            try:
                self.load_state_dict(state_dict)
            except RuntimeError:
                # If weights were saved directly from self.model
                self.model.load_state_dict(state_dict)
            
        self.eval()
        logger.info(f"Loaded classifier weights from {path} (backbone: {self.backbone_name})")

    @torch.no_grad()
    def predict(
        self,
        image_tensor: torch.Tensor,
        device: str = "cpu",
    ) -> tuple[int, float, dict[str, float]]:
        """
        Classify a single 2D MRI slice.

        Args:
            image_tensor: (1, 3, H, W) or (3, H, W) float tensor
            device: inference device

        Returns:
            (class_index, confidence, class_probabilities_dict)
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        self.eval()
        image_tensor = image_tensor.to(device)
        self.to(device)

        logits = self(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_class = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_class].item())

        probability_dict = {
            self.CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(self.num_classes)
        }

        return predicted_class, confidence, probability_dict
