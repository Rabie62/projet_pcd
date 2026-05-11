"""
Safety Agent — monitors confidence, flags uncertain cases, and enforces
compliance with clinical safety standards.

Responsibilities:
  1. Confidence thresholding — flag low-confidence predictions
  2. Ambiguity detection — flag cases with similar top-2 probabilities
  3. Consistency checks — flag segmentation/classification disagreements
  4. Compliance notes — class-specific clinical guidance
  5. Audit logging — record every prediction for traceability
  6. Disclaimer enforcement — ensure disclaimers accompany all outputs

Adapted for BRISC 2025 (4-class: Glioma, Meningioma, Pituitary, No Tumor).
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from loguru import logger
from config.settings import Settings, get_settings
from agents.diagnostic import DiagnosticReport


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class SafetyCheck:
    """Result of a safety check on a diagnostic report."""
    passed: bool
    confidence_adequate: bool
    requires_human_review: bool
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """Single audit log entry for traceability."""
    timestamp: str
    patient_id: str
    action: str
    input_hash: str
    model_version: str
    results_summary: str
    confidence: float
    safety_check: dict
    flags: list[str]
    metadata: dict = field(default_factory=dict)


# ── Safety Agent ──────────────────────────────────────────────────────

class SafetyAgent:
    """
    Safety Agent — ensures all AI outputs meet clinical safety standards
    before being presented to doctors.
    """

    DISCLAIMER = (
        "⚠️ RESEARCH USE ONLY: This AI system is not a certified medical device. "
        "All findings are preliminary and must be reviewed and confirmed by a "
        "qualified radiologist or medical professional before any clinical "
        "decision-making. Do not use for diagnosis or treatment without "
        "appropriate clinical oversight."
    )

    MODEL_VERSION = "2.0.0-BRISC"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.guidelines = self.load_guidelines()
        self.audit_log: list[AuditEntry] = []
        logger.info("Safety Agent initialised")

    def load_guidelines(self) -> dict:
        """Load clinical guidelines from the knowledge base."""
        guidelines_path = self.settings.paths.knowledge_dir / "guidelines.json"
        if guidelines_path.exists():
            with open(guidelines_path, "r") as f:
                return json.load(f)
        logger.warning("No guidelines file found — using built-in defaults.")
        return self.default_guidelines()

    @staticmethod
    def default_guidelines() -> dict:
        """Return built-in default safety guidelines."""
        return {
            "confidence_minimum": 0.85,
            "tumor_area_concern_ratio": 0.15,
            "ambiguity_gap_threshold": 0.10,
            "mandatory_review_classifications": [],
            "who_grading": {
                "Glioma": "WHO Grade II–IV — requires further grading",
                "Meningioma": "WHO Grade I–III — typically benign",
                "Pituitary": "Pituitary adenoma — typically benign",
                "No Tumor": "No pathology detected",
            },
        }

    def check_safety(self, report: DiagnosticReport) -> SafetyCheck:
        """
        Run comprehensive safety checks on a diagnostic report.

        Checks performed:
          1. Confidence threshold
          2. Classification ambiguity (top-2 gap)
          3. Segmentation/classifier consistency
          4. Tumor area flags
          5. Class-specific compliance notes

        Returns:
            SafetyCheck with pass/fail status and detailed flags
        """
        flags = []
        warnings = []
        compliance_notes = []
        requires_review = False
        confidence_adequate = True

        confidence_threshold = self.guidelines.get(
            "confidence_minimum",
            self.settings.inference.confidence_threshold,
        )

        # --- No-tumor path ---
        if not report.tumor_detected:
            if report.classification and report.classification != "No Tumor":
                flags.append(
                    f"Segmentation found no tumor but classifier predicted "
                    f"'{report.classification}' — requires review."
                )
                requires_review = True

            compliance_notes.append(
                "No tumor detected — recommend clinical correlation."
            )
            return SafetyCheck(
                passed=not requires_review,
                confidence_adequate=True,
                requires_human_review=requires_review,
                flags=flags,
                compliance_notes=compliance_notes,
            )

        # --- 1. Confidence threshold ---
        if report.classification_confidence < confidence_threshold:
            confidence_adequate = False
            requires_review = True
            flags.append(
                f"Classification confidence ({report.classification_confidence:.1%}) "
                f"below threshold ({confidence_threshold:.1%}). "
                "MANDATORY HUMAN REVIEW REQUIRED."
            )

        # --- 2. Ambiguity check (close top-2 probabilities) ---
        ambiguity_gap = self.guidelines.get("ambiguity_gap_threshold", 0.10)
        if report.classification_probabilities:
            sorted_probabilities = sorted(
                report.classification_probabilities.values(), reverse=True
            )
            if (len(sorted_probabilities) >= 2
                    and (sorted_probabilities[0] - sorted_probabilities[1]) < ambiguity_gap):
                requires_review = True
                flags.append(
                    "Top two classifications have similar probabilities "
                    f"(gap < {ambiguity_gap:.0%}) — ambiguous result, review needed."
                )

        # --- 3. Mandatory review classifications ---
        mandatory_review_list = self.guidelines.get(
            "mandatory_review_classifications", []
        )
        if report.classification in mandatory_review_list:
            requires_review = True
            flags.append(
                f"Classification '{report.classification}' requires "
                "mandatory human review."
            )

        # --- 4. Tumor size warnings ---
        if report.tumor_features:
            tumor_features = report.tumor_features

            area_concern_ratio = self.guidelines.get(
                "tumor_area_concern_ratio", 0.15
            )
            if tumor_features.tumor_ratio > area_concern_ratio:
                warnings.append(
                    f"Large tumor area ({tumor_features.tumor_ratio:.1%} of image) — "
                    "urgent evaluation recommended."
                )

            if tumor_features.max_diameter_mm > 40:
                warnings.append(
                    f"Large tumor diameter ({tumor_features.max_diameter_mm:.1f} mm) — "
                    "urgent evaluation recommended."
                )

        # --- 5. Class-specific compliance notes ---
        if report.classification == "Glioma":
            compliance_notes.append(
                "Glioma detected — further WHO grading (II–IV) requires "
                "histopathological confirmation."
            )
            compliance_notes.append(
                "Consider advanced imaging for grade determination."
            )
        elif report.classification == "Meningioma":
            compliance_notes.append(
                "Meningioma detected — most are benign (WHO Grade I). "
                "Surgical consultation based on symptoms and size."
            )
        elif report.classification == "Pituitary":
            compliance_notes.append(
                "Pituitary tumor detected — endocrinological workup recommended."
            )

        compliance_notes.append(
            "All measurements are approximate and based on AI segmentation."
        )

        passed = confidence_adequate and not requires_review

        return SafetyCheck(
            passed=passed,
            confidence_adequate=confidence_adequate,
            requires_human_review=requires_review,
            flags=flags,
            warnings=warnings,
            compliance_notes=compliance_notes,
        )

    def log_audit(
        self,
        report: DiagnosticReport,
        safety_check: SafetyCheck,
        input_data_bytes: Optional[bytes] = None,
    ) -> AuditEntry:
        """Log a full audit entry for traceability and compliance."""
        input_hash = (
            hashlib.sha256(input_data_bytes).hexdigest()
            if input_data_bytes
            else "N/A"
        )

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            patient_id=report.patient_id,
            action="full_analysis",
            input_hash=input_hash,
            model_version=self.MODEL_VERSION,
            results_summary=report.clinical_summary,
            confidence=report.classification_confidence,
            safety_check=asdict(safety_check),
            flags=report.flags + safety_check.flags,
        )

        self.audit_log.append(entry)
        self.persist_audit_entry(entry)

        logger.info(
            f"Audit logged for patient {report.patient_id}: "
            f"passed={safety_check.passed}, "
            f"review_needed={safety_check.requires_human_review}"
        )

        return entry

    def persist_audit_entry(self, entry: AuditEntry) -> None:
        """Write an audit entry to disk as JSON-lines."""
        log_dir = self.settings.paths.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    def get_audit_log(self) -> list[AuditEntry]:
        """Return all audit entries from the current session."""
        return self.audit_log.copy()

    def get_disclaimer(self) -> str:
        """Return the standard medical disclaimer text."""
        return self.DISCLAIMER
