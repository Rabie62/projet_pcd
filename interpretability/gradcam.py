"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Brain MRI Classification.
Generates visual explanations highlighting regions that influenced the model's decision.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GradCAM:
    """
    Grad-CAM for EfficientNet-V2-S TumorClassifier.
    Hooks into the final convolutional feature layer to extract feature maps and gradients.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: The TumorClassifier model instance
            target_layer: The convolutional layer to inspect. Defaults to model.model.features[-1].
        """
        self.model = model
        self.model.eval()

        # Find target layer if not provided
        if target_layer is None:
            if hasattr(model, "model") and hasattr(model.model, "features"):
                # EfficientNet features last block
                self.target_layer = model.model.features[-1]
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Target layer could not be determined automatically.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self.register_hooks()

    def register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: (1, 3, H, W) or (3, H, W) float tensor
            target_class: Target class index. If None, uses the model's top prediction.

        Returns:
            cam: 2D numpy array of shape (H, W) normalized to [0, 1]
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).clone().requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients during Grad-CAM backward pass.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Only features with positive influence

        # Resize CAM to match input image size (H, W)
        batch_size, channels, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def __del__(self):
        self.remove_hooks()
