"""
2D MRI Preprocessing for BRISC 2025 dataset.
Normalizes and prepares 2D T1-weighted RGB images for inference.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from loguru import logger


@dataclass
class PreprocessingResult:
    """Result of preprocessing a 2D MRI image."""

    tensor: torch.Tensor            
    original_shape: tuple[int, ...]  
    pixel_spacing: tuple[float, float] = (1.0, 1.0)  # mm/pixel (default)


class MRIPreprocessor:
    """
    Preprocessor for 2D T1-weighted MRI images (BRISC 2025).

    Pipeline:
      1. Channel-wise intensity normalisation (zero mean, unit variance)
      2. Convert to torch tensor
    """

    def __init__(self, target_size: tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def __call__(
        self,
        image: np.ndarray,
        pixel_spacing: tuple[float, float] = (1.0, 1.0),
    ) -> PreprocessingResult:
        """
        Preprocess a 2D MRI image.

        Args:
            image: (3, H, W) float32 array in [0, 1]
            pixel_spacing: physical pixel spacing in mm

        Returns:
            PreprocessingResult with normalised tensor
        """
        original_shape = image.shape[1:]  # (H, W)
        normalised = self.normalize_channels(image)
        tensor = torch.from_numpy(normalised).float()

        return PreprocessingResult(
            tensor=tensor,
            original_shape=original_shape,
            pixel_spacing=pixel_spacing,
        )

    @staticmethod
    def normalize_channels(image: np.ndarray) -> np.ndarray:
        """Channel-wise zero-mean, unit-variance normalisation."""
        result = np.zeros_like(image)
        for channel_idx in range(image.shape[0]):
            channel = image[channel_idx]
            nonzero_mask = channel > 0
            if nonzero_mask.any():
                mean = channel[nonzero_mask].mean()
                std = channel[nonzero_mask].std()
                if std > 1e-8:
                    result[channel_idx][nonzero_mask] = (channel[nonzero_mask] - mean) / std
                else:
                    result[channel_idx][nonzero_mask] = channel[nonzero_mask] - mean
            # Background remains 0 (initialised in result)
        return result
