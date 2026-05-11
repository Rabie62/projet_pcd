"""
Central configuration for the Medical AI Agent.
All settings are configurable via environment variables.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and training configuration."""
    seg_in_channels: int = 3      # RGB images
    seg_out_channels: int = 2     # Background vs Tumor
    seg_channels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    seg_strides: tuple[int, ...] = (2, 2, 2, 2, 2)

    # Classifier
    clf_num_classes: int = 4
    clf_in_channels: int = 3

    # LLM (Dialogue Agent)
    llm_model_id: str = "google/gemma-3-1b-it"
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.3
    hf_token: Optional[str] = None


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-time parameters."""
    confidence_threshold: float = 0.85
    sliding_window_overlap: float = 0.5
    sw_batch_size: int = 1
    device: str = "cuda"
    tumor_min_area_mm2: float = 10.0


@dataclass(frozen=True)
class APIConfig:
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 500


@dataclass(frozen=True)
class PathConfig:
    """File system paths."""
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "brisc")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models" / "pretrained")
    logs_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    knowledge_dir: Path = field(default_factory=lambda: BASE_DIR / "knowledge")
    outputs_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    uploads_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "uploads")
    qdrant_storage_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "qdrant_storage")

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for p in [
            self.data_dir, self.models_dir, self.logs_dir,
            self.knowledge_dir, self.outputs_dir,
            self.uploads_dir, self.qdrant_storage_dir
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatabaseConfig:
    """MySQL Configuration."""
    host: str = field(default_factory=lambda: os.environ.get("MYSQL_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("MYSQL_PORT", "3306")))
    user: str = field(default_factory=lambda: os.environ.get("MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: os.environ.get("MYSQL_PASSWORD", ""))
    name: str = field(default_factory=lambda: os.environ.get("MYSQL_DATABASE", "medical_ai"))

    @property
    def url(self) -> Optional[str]:
        if not self.host:
            return None
        from urllib.parse import quote_plus
        encoded_pass = quote_plus(self.password)
        return f"mysql+pymysql://{self.user}:{encoded_pass}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class ICD11Config:
    """ICD-11 API configuration - Using the latest release (2026)."""
    client_id: Optional[str] = field(default_factory=lambda: os.environ.get("ICD11_CLIENT_ID"))
    client_secret: Optional[str] = field(default_factory=lambda: os.environ.get("ICD11_CLIENT_SECRET"))
    
    api_url: str = "https://id.who.int"
    token_url: str = "https://icdaccessmanagement.who.int/connect/token"
    
    # Latest ICD-11 Release
    release: str = "2026-01"
    
    # Alternative: Use this if you want to always fetch the latest available
    use_latest: bool = False


@dataclass(frozen=True)
class Settings:
    """Root settings container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    icd11: ICD11Config = field(default_factory=ICD11Config)

    def __post_init__(self) -> None:
        self.paths.ensure_dirs()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get or create singleton settings instance."""
    device = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

    inference = InferenceConfig(
        device=device,
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85")),
    )

    model = ModelConfig(
        llm_model_id=os.environ.get("LLM_MODEL_ID", "google/gemma-3-1b-it"),
        hf_token=os.environ.get("HF_TOKEN"),
    )

    return Settings(
        model=model,
        inference=inference,
        db=DatabaseConfig(),
        icd11=ICD11Config(),
    )