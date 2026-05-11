"""
BRISC 2025 Data Loader — loads 2D T1-weighted MRI images (JPG)
with optional segmentation masks (PNG) and classification labels.

Supports two directory layouts:
  1. Classification: classification_task/train/{glioma,meningioma,no_tumor,pituitary}/*.jpg
  2. Segmentation:   segmentation_task/train/images/*.jpg  +  masks/*.png
  3. Flat directory:  *.jpg (+ optional *.png masks with matching basenames)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
from loguru import logger
from PIL import Image


# ── Label mappings ────────────────────────────────────────────────────

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
CLASS_FOLDERS = {"glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3}


# ── Patient dataclass ────────────────────────────────────────────────

@dataclass
class BRISCPatient:
    """Single BRISC 2025 patient/image sample."""

    patient_id: str
    image: np.ndarray           
    mask: Optional[np.ndarray]  
    class_label: Optional[int]  
    class_name: Optional[str]   
    image_path: Path
    image_size: tuple[int, ...] = (512, 512)
    pixel_spacing: tuple[float, ...] = (1.0, 1.0)


# ── Data loader ──────────────────────────────────────────────────────

class BRISCDataLoader:
    """
    Loads BRISC 2025 images (JPG) from a directory.
    """

    DEFAULT_SIZE = (512, 512)

    def __init__(
        self,
        data_dir: str | Path,
        image_size: tuple[int, int] = DEFAULT_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.patient_entries: list[dict] = []
        self.discover_images()

    # ── Discovery ─────────────────────────────────────────────────

    def discover_images(self) -> None:
        """Auto-detect directory layout and catalogue all images."""

        # 2. Try classification layout
        for folder_name, label in CLASS_FOLDERS.items():
            folder = self.data_dir / folder_name
            if folder.is_dir():
                for img_path in sorted(folder.glob("*.jpg")):
                    self.patient_entries.append({
                        "id": img_path.stem,
                        "image": img_path,
                        "mask": None,
                        "label": label,
                    })

        if self.patient_entries:
            logger.info(f"BRISC classification layout: {len(self.patient_entries)} images")
            return

        # 3. Flat directory of JPGs
        for img_path in sorted(self.data_dir.glob("*.jpg")):
            mask_path = self.data_dir / (img_path.stem + ".png")
            self.patient_entries.append({
                "id": img_path.stem,
                "image": img_path,
                "mask": mask_path if mask_path.exists() else None,
                "label": None,
            })

        logger.info(f"BRISC flat layout: {len(self.patient_entries)} images")

    # ── Public API ────────────────────────────────────────────────

    @property
    def num_patients(self) -> int:
        return len(self.patient_entries)

    def load_patient(self, patient_id: str) -> BRISCPatient:
        for entry in self.patient_entries:
            if entry["id"] == patient_id:
                return self.build_patient(entry)
        raise FileNotFoundError(f"Patient '{patient_id}' not found")

    def load_patient_from_path(self, image_path: str | Path) -> BRISCPatient:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check for mask in the same directory
        mask_path = image_path.parent / (image_path.stem + ".png")
        mask = None
        if mask_path.exists():
            try:
                mask = self.load_mask(mask_path)
                logger.info(f"Auto-loaded mask for {image_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e}")

        return self.build_patient({
            "id": image_path.name.split('.')[0],
            "image": image_path,
            "mask": mask,
            "label": None,
        })


    def iterate_patients(self, max_patients: Optional[int] = None) -> Iterator[BRISCPatient]:
        for i, entry in enumerate(self.patient_entries):
            if max_patients is not None and i >= max_patients:
                break
            yield self.build_patient(entry)

    # ── Loading Helpers ───────────────────────────────────────────

    def build_patient(self, entry: dict) -> BRISCPatient:
        image, spacing = self.load_image(entry["image"])
        mask = None # Masks for 3D usually separate NIfTIs
        label = entry["label"]
        class_name = CLASS_NAMES.get(label) if label is not None else None

        return BRISCPatient(
            patient_id=entry["id"],
            image=image,
            mask=mask,
            class_label=label,
            class_name=class_name,
            image_path=Path(entry["image"]),
            image_size=self.image_size,
            pixel_spacing=spacing
        )

    def load_image(self, path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
        """Load RGB image."""
        
        # Standard image
        img = Image.open(path).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return arr.transpose(2, 0, 1), (1.0, 1.0)

    def load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)
        arr = np.array(mask, dtype=np.float32)
        return (arr > 127).astype(np.float32)
