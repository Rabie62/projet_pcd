"""
ResNet-18 Tumor Classifier for BRISC 2025 Dataset.
4-class classification: Glioma, Meningioma, No Tumor, Pituitary.
Achieves ~98% accuracy on the test set.
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
    ResNet-18 for brain tumor classification (4 classes).
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load resnet18 (weights=None because we load our own)
        self.model = models.resnet18(weights=None)
        
        # Modify first conv layer if in_channels != 3
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
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
        if isinstance(state, dict):
            self.load_state_dict(state)
        else:
            self.load_state_dict(state.state_dict())
            
        self.eval()
        logger.info(f"Loaded classifier weights from {path}")

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
