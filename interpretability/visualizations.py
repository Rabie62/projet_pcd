"""
MRI visualization utilities for BRISC 2025 Dataset (2D).
Generates annotated image overlays and summaries for brain tumor segmentation.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# Standard BRISC Color Scheme (RGBA)
# 1: Tumor - Red
BRISC_COLORS = {
    1: (1.0, 0.0, 0.0, 0.6),   # Tumor - Red
}

BRISC_LABELS = {
    1: "Tumor",
}

class BRISCVisualizer:
    """
    Visualizer specifically optimized for 2D brain MRI analysis (BRISC).
    """

    def __init__(self, figsize: tuple[int, int] = (12, 5), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def generate_summary(
        self,
        mri_image: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
        pred_mask: Optional[np.ndarray] = None,
    ) -> bytes:
        """
        Generate a summary grid (Image, Ground Truth, Prediction).
        Adjusts to 2 or 3 panels based on seg_mask availability.
        
        Args:
            mri_image: (3, H, W) or (H, W) image
            seg_mask: (H, W) ground truth
            pred_mask: (H, W) model prediction
        """
        # Determine number of panels
        panels = []
        panels.append(("Original Image", mri_image, "image"))
        if seg_mask is not None:
            panels.append(("Ground Truth", seg_mask, "mask"))
        if pred_mask is not None:
            panels.append(("AI Prediction", pred_mask, "mask"))
            
        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), dpi=self.dpi)
        if n_panels == 1: axes = [axes]
        fig.patch.set_facecolor("black")
        
        for i, (title, data, dtype) in enumerate(panels):
            if dtype == "image":
                # Handle (3, H, W) -> (H, W, 3)
                if data.ndim == 3 and data.shape[0] == 3:
                    display_img = data.transpose(1, 2, 0)
                else:
                    display_img = data
                
                # Normalize for display if needed
                if display_img.max() > 1.0 or display_img.min() < 0.0:
                    display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
                
                axes[i].imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)
            else:
                # Mask overlay on original image
                if mri_image.ndim == 3 and mri_image.shape[0] == 3:
                    base_img = mri_image.transpose(1, 2, 0)
                else:
                    base_img = mri_image
                
                if base_img.max() > 1.0 or base_img.min() < 0.0:
                    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
                
                axes[i].imshow(base_img, cmap="gray" if base_img.ndim == 2 else None)
                overlay = self._create_segmentation_overlay(data)
                axes[i].imshow(overlay)
                
            axes[i].set_title(title, color="white", fontsize=12)
            axes[i].axis("off")
            
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="black")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _create_segmentation_overlay(self, mask_2d: np.ndarray, alpha_mult: float = 0.6) -> np.ndarray:
        """Helper to create an RGBA overlay from a segmentation slice."""
        overlay = np.zeros((*mask_2d.shape, 4))
        for label, color in BRISC_COLORS.items():
            pixels = (mask_2d == label)
            if np.any(pixels):
                c = list(color)
                c[3] = alpha_mult
                overlay[pixels] = c
        return overlay

    def save_visualization(self, data: bytes, path: str | Path):
        """Save bytes to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"Visualization saved to {path}")

