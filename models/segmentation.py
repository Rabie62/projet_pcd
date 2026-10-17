"""
2D Vision Transformer Segmentation Model for BRISC Dataset.
Uses hierarchical Vision Transformer U-Net (mit_b2 / SegFormer) pre-trained on ImageNet,
specifically adapted from SwinUNETR.ipynb.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
import segmentation_models_pytorch as smp


class SegmentationModel:
    """
    Wrapper around 2D Pretrained Vision Transformer U-Net (SegFormer / mit_b2)
    for brain tumor segmentation.
    """

    def __init__(
        self,
        encoder_name: str = "mit_b2",
        in_channels: int = 3,   # RGB images from BRISC dataset
        out_channels: int = 2,  # Background vs Tumor
        encoder_weights: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.in_channels,
            classes=self.out_channels,
        ).to(self.device)

        self.model.eval()
        logger.info(
            f"SegmentationModel 2D ({self.encoder_name} U-Net) initialised on {self.device}"
        )

    def load_weights(self, path: str | Path) -> None:
        """Load pretrained weights from a .pth file."""
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"Loaded 2D segmentation weights from {path}")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run segmentation on a 2D image.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) float tensor

        Returns:
            (H, W) int64 tensor with class labels (0 or 1)
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)  # (1, 2, H, W)
        preds = torch.argmax(logits, dim=1)  # (1, H, W)
        return preds.squeeze(0)

    @torch.no_grad()
    def predict_probs(
        self, image_tensor: torch.Tensor, postprocess: bool = True, min_component_size: int = 30, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run segmentation and return both predictions and probabilities.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) float tensor
            postprocess: Whether to apply connected-component filtering to eliminate small noise
            min_component_size: Minimum pixel count for a valid tumor component

        Returns:
            (preds, probs) where:
              preds: (1, 1, H, W) int64 class labels
              probs:  (1, 2, H, W) float32 softmax probabilities
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)  # (1, 2, H, W)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1, keepdim=True)  # (1, 1, H, W)

        if postprocess:
            preds_np = preds.squeeze().cpu().numpy()
            cleaned_np = self.postprocess_mask(preds_np, min_size=min_component_size)
            preds = torch.from_numpy(cleaned_np).to(self.device).unsqueeze(0).unsqueeze(0)

        return preds, probs

    @staticmethod
    def postprocess_mask(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
        """
        Post-process binary segmentation mask using connected-component filtering.
        Removes isolated spurious noise regions smaller than min_size pixels.
        """
        import numpy as np
        from scipy.ndimage import label, binary_fill_holes

        if not np.any(mask == 1):
            return mask

        binary_mask = (mask == 1).astype(np.uint8)
        labeled_array, num_features = label(binary_mask)

        if num_features == 0:
            return mask

        cleaned = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            if np.sum(component) >= min_size:
                # Keep component and fill tiny interior holes
                filled = binary_fill_holes(component)
                cleaned[filled] = 1

        return cleaned


