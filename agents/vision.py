"""
Vision Agent — handles the 2D MRI image analysis pipeline for BRISC 2025 Dataset.
Responsibilities:
  1. Preprocess the input MRI image (2D RGB)
  2. Run 2D Attention U-Net segmentation to detect tumor regions
  3. Extract quantitative tumor region statistics
  4. Classify the tumor type (Glioma, Meningioma, Pituitary, or No Tumor)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from loguru import logger
from config.settings import Settings, get_settings
from data.loader import BRISCPatient
from models.segmentation import SegmentationModel
from models.classifier import TumorClassifier
from monai.transforms import Resize, NormalizeIntensity


# ── Data classes ──────────────────────────────────────────────────────
@dataclass
class TumorRegion:
    """Detected tumor region information (2D)."""
    label: int
    label_name: str
    area_pixels: int
    area_mm2: float
    centroid: tuple[float, ...] = field(default_factory=tuple)
    bounding_box: tuple[tuple[int, int], ...] = field(default_factory=tuple)


@dataclass
class VisionResult:
    """Complete result from the Vision Agent."""
    patient_id: str
    tumor_detected: bool
    segmentation_mask: np.ndarray
    segmentation_probabilities: Optional[np.ndarray]
    tumor_regions: list[TumorRegion]
    tumor_class: Optional[str]
    tumor_class_confidence: float
    tumor_class_probabilities: dict[str, float]
    preprocessed_image: np.ndarray
    original_shape: tuple[int, ...]
    pixel_spacing: tuple[float, ...]
    ground_truth_mask: Optional[np.ndarray] = None
    errors: list[str] = field(default_factory=list)


# ── Vision Agent ──────────────────────────────────────────────────────
class VisionAgent:
    """
    Vision Agent for 2D brain MRI analysis (BRISC 2025).
    Pipeline: preprocess → 2D segmentation → extract regions → 2D classification
    """
    SEGMENTATION_LABELS = {
        0: "Background",
        1: "Tumor",
    }

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.segmentation_model: Optional[SegmentationModel] = None
        self.classifier: Optional[TumorClassifier] = None
        self.models_loaded = False

    def load_models(self) -> None:
        """Load segmentation and classification models (both 2D)."""
        device = self.settings.inference.device

        # --- Segmentation model (2D AttentionUnet) ---
        self.segmentation_model = SegmentationModel(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            device=device,
        )
        weights_path = self.settings.paths.models_dir / "segmentation.pth"
        if not weights_path.exists():
            logger.warning(f"2D Segmentation weights not found at {weights_path}")
        else:
            self.segmentation_model.load_weights(weights_path)

        # --- Classification model ---
        self.classifier = TumorClassifier(
            in_channels=3,
            num_classes=4,
        ).to(device if torch.cuda.is_available() else "cpu")

        weights_path = self.settings.paths.models_dir / "classification.pth"
        if not weights_path.exists():
            logger.warning(f"Classifier weights not found at {weights_path}")
        else:
            self.classifier.load_weights(weights_path)

        self.models_loaded = True
        logger.info("Vision Agent: 2D models loaded (BRISC)")

    def extract_tumor_regions(
        self,
        segmentation_mask: np.ndarray,
        pixel_spacing: tuple[float, ...],
    ) -> list[TumorRegion]:
        """Extract quantitative info from 2D segmentation mask."""
        unit_val = pixel_spacing[0] * pixel_spacing[1]

        tumor_indices = np.where(segmentation_mask == 1)
        total_pixels = len(tumor_indices[0])

        if total_pixels == 0:
            return []

        centroid = tuple(float(idx.mean()) for idx in tumor_indices)
        bbox = tuple((int(idx.min()), int(idx.max()) + 1) for idx in tumor_indices)
        area_mm2 = float(total_pixels * unit_val)

        region = TumorRegion(
            label=1,
            label_name=self.SEGMENTATION_LABELS[1],
            area_pixels=total_pixels,
            area_mm2=area_mm2,
            centroid=centroid,
            bounding_box=bbox,
        )

        return [region]

    def analyze(self, patient: BRISCPatient) -> VisionResult:
        """
        Run 2D analysis pipeline on BRISC patient data.
        Expects 2D image input.
        """
        if not self.models_loaded:
            self.load_models()

        errors = []
        logger.info(f"Analysing patient {patient.patient_id} (BRISC 2D Pipeline)")

        # 1. Preprocess — Assume 2D input
        img_array = patient.image

        # Convert to RGB (3, H, W) if needed
        if img_array.ndim == 2:  # Grayscale (H, W)
            img_array = np.stack([img_array] * 3, axis=0)
        elif img_array.ndim == 3 and img_array.shape[0] != 3:  # (H, W, C) -> (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
        elif img_array.ndim == 3 and img_array.shape[0] == 3:
            pass  # Already in correct format (C, H, W)
        else:
            errors.append(f"Unexpected image shape: {img_array.shape}")
            # Fallback: take first channel and convert to RGB
            if img_array.ndim == 3:
                img_array = img_array[0]
            img_array = np.stack([img_array] * 3, axis=0)

        # Resize to model input size
        resizer = Resize(spatial_size=(512, 512), mode="bilinear")
        img_resized = resizer(img_array)

        # Normalize
        normalizer = NormalizeIntensity(nonzero=False, channel_wise=True)
        input_tensor = normalizer(img_resized)

        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # 2. 2D Segmentation
        segmentation_mask = np.zeros((512, 512), dtype=np.int64)
        segmentation_probs = None

        if self.segmentation_model is not None:
            preds, probs = self.segmentation_model.predict_with_probabilities(input_tensor)
            segmentation_mask = preds.squeeze().cpu().numpy()
            segmentation_probs = probs.squeeze().cpu().numpy()

        # 3. Extract regions
        spacing = getattr(patient, 'pixel_spacing', (1.0, 1.0))
        tumor_regions = self.extract_tumor_regions(segmentation_mask, spacing)
        tumor_detected = len(tumor_regions) > 0

        # 4. 2D Classification
        tumor_class = None
        tumor_confidence = 0.0
        tumor_probabilities: dict[str, float] = {}

        if self.classifier is not None:
            class_idx, confidence, probs = self.classifier.predict(
                input_tensor, device=self.settings.inference.device
            )
            tumor_class = TumorClassifier.CLASS_NAMES[class_idx]
            tumor_confidence = confidence
            tumor_probabilities = probs

        return VisionResult(
            patient_id=patient.patient_id,
            tumor_detected=tumor_detected,
            segmentation_mask=segmentation_mask,
            segmentation_probabilities=segmentation_probs,
            tumor_regions=tumor_regions,
            tumor_class=tumor_class,
            tumor_class_confidence=tumor_confidence,
            tumor_class_probabilities=tumor_probabilities,
            preprocessed_image=img_resized.numpy() if isinstance(img_resized, torch.Tensor) else img_resized,
            original_shape=patient.image.shape,
            pixel_spacing=spacing,
            ground_truth_mask=patient.mask,
            errors=errors,
        )