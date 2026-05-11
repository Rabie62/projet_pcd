"""
2D Attention U-Net Segmentation Model for BRISC Dataset.
Specifically adapted from AttentionUnet 2D.ipynb.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from loguru import logger
from monai.networks.nets import AttentionUnet


class SegmentationModel:
    """
    Wrapper around MONAI 2D Attention U-Net for brain tumor segmentation.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,   # RGB images from BRISC dataset
        out_channels: int = 2,  # Background vs Tumor
        channels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        strides: tuple[int, ...] = (2, 2, 2, 2, 2),
        dropout: float = 0.3,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        ).to(self.device)

        self.model.eval()
        logger.info(
            f"SegmentationModel 2D (AttentionUnet) initialised on {self.device}"
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
    def predict_with_probabilities(
        self, image_tensor: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run segmentation and return both predictions and probabilities.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) float tensor

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
        return preds, probs

