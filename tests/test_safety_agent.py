"""
Unit tests for the Safety Agent.
Tests confidence thresholding, ambiguity detection, tumor size warnings,
class-specific compliance notes, and audit logging.
"""
import pytest
from unittest.mock import MagicMock
from agents.safety import SafetyAgent, SafetyCheck, AuditEntry
from agents.diagnostic import DiagnosticReport, TumorFeatures


@pytest.fixture
def safety_agent(tmp_path):
    """Create a SafetyAgent with a temporary logs directory."""
    mock_settings = MagicMock()
    mock_settings.paths.logs_dir = tmp_path
    mock_settings.paths.knowledge_dir = tmp_path
    mock_settings.inference.confidence_threshold = 0.85
    agent = SafetyAgent(mock_settings)
    yield agent


def _make_report(
    tumor_detected=True,
    classification="Glioma",
    confidence=0.95,
    probabilities=None,
    tumor_features=None,
):
    if probabilities is None:
        probabilities = {"Glioma": 0.95, "Meningioma": 0.03, "Pituitary": 0.01, "No Tumor": 0.01}
    return DiagnosticReport(
        patient_id="test_001",
        timestamp="2025-01-01T00:00:00",
        tumor_detected=tumor_detected,
        tumor_features=tumor_features,
        classification=classification,
        classification_confidence=confidence,
        classification_probabilities=probabilities,
        who_grade="Grade II-IV",
        clinical_summary="Test summary",
        recommendations=[],
        flags=[],
    )


class TestSafetyAgentInit:
    def test_init_loads_default_guidelines_when_file_missing(self, safety_agent):
        assert safety_agent.guidelines["confidence_minimum"] == 0.85
        assert safety_agent.guidelines["tumor_area_concern_ratio"] == 0.15

    def test_disclaimer_present(self, safety_agent):
        assert "RESEARCH USE ONLY" in safety_agent.DISCLAIMER
        assert "not a certified medical device" in safety_agent.DISCLAIMER


class TestNoTumorPath:
    def test_no_tumor_clean_passes(self, safety_agent):
        report = _make_report(tumor_detected=False, classification="No Tumor", confidence=1.0)
        result = safety_agent.check_safety(report)
        assert result.passed is True
        assert result.requires_human_review is False
        assert len(result.flags) == 0

    def test_no_tumor_but_classifier_says_glioma_flags(self, safety_agent):
        report = _make_report(tumor_detected=False, classification="Glioma", confidence=0.6)
        result = safety_agent.check_safety(report)
        assert result.passed is False
        assert result.requires_human_review is True
        assert any("Segmentation found no tumor" in f for f in result.flags)


class TestConfidenceThreshold:
    def test_high_confidence_passes(self, safety_agent):
        report = _make_report(confidence=0.95)
        result = safety_agent.check_safety(report)
        assert result.confidence_adequate is True
        assert not any("below threshold" in f for f in result.flags)

    def test_low_confidence_flags_review(self, safety_agent):
        report = _make_report(confidence=0.5)
        result = safety_agent.check_safety(report)
        assert result.confidence_adequate is False
        assert result.requires_human_review is True
        assert any("below threshold" in f for f in result.flags)

    def test_confidence_at_boundary(self, safety_agent):
        report = _make_report(confidence=0.85)
        result = safety_agent.check_safety(report)
        assert result.confidence_adequate is True


class TestAmbiguityDetection:
    def test_ambiguous_top2_flags_review(self, safety_agent):
        report = _make_report(
            confidence=0.45,
            probabilities={"Glioma": 0.45, "Meningioma": 0.40, "Pituitary": 0.10, "No Tumor": 0.05},
        )
        result = safety_agent.check_safety(report)
        assert result.requires_human_review is True
        assert any("similar probabilities" in f for f in result.flags)

    def test_clear_winner_no_ambiguity(self, safety_agent):
        report = _make_report(
            confidence=0.95,
            probabilities={"Glioma": 0.95, "Meningioma": 0.03, "Pituitary": 0.01, "No Tumor": 0.01},
        )
        result = safety_agent.check_safety(report)
        assert not any("similar probabilities" in f for f in result.flags)


class TestTumorSizeWarnings:
    def test_large_tumor_ratio_warns(self, safety_agent):
        features = TumorFeatures(tumor_area_px=100000, tumor_area_mm2=1000.0, total_image_area_px=500000, tumor_ratio=0.20)
        report = _make_report(tumor_features=features)
        result = safety_agent.check_safety(report)
        assert any("Large tumor area" in w for w in result.warnings)

    def test_large_diameter_warns(self, safety_agent):
        features = TumorFeatures(max_diameter_mm=50.0)
        report = _make_report(tumor_features=features)
        result = safety_agent.check_safety(report)
        assert any("Large tumor diameter" in w for w in result.warnings)

    def test_small_tumor_no_size_warning(self, safety_agent):
        features = TumorFeatures(tumor_area_px=100, tumor_area_mm2=5.0, total_image_area_px=500000, tumor_ratio=0.001, max_diameter_mm=10.0)
        report = _make_report(tumor_features=features)
        result = safety_agent.check_safety(report)
        assert len(result.warnings) == 0


class TestClassSpecificCompliance:
    def test_glioma_compliance_notes(self, safety_agent):
        report = _make_report(classification="Glioma")
        result = safety_agent.check_safety(report)
        assert any("histopathological confirmation" in n.lower() for n in result.compliance_notes)

    def test_meningioma_compliance_notes(self, safety_agent):
        report = _make_report(classification="Meningioma")
        result = safety_agent.check_safety(report)
        assert any("benign" in n.lower() for n in result.compliance_notes)

    def test_pituitary_compliance_notes(self, safety_agent):
        report = _make_report(classification="Pituitary")
        result = safety_agent.check_safety(report)
        assert any("endocrinological" in n.lower() for n in result.compliance_notes)


class TestAuditLogging:
    def test_audit_entry_created(self, safety_agent, tmp_path):
        report = _make_report()
        check = safety_agent.check_safety(report)
        entry = safety_agent.log_audit(report, check)
        assert entry.patient_id == "test_001"
        assert entry.action == "full_analysis"
        assert len(safety_agent.audit_log) == 1

    def test_audit_persists_to_disk(self, safety_agent, tmp_path):
        report = _make_report()
        check = safety_agent.check_safety(report)
        safety_agent.log_audit(report, check)
        log_files = list(tmp_path.glob("audit_*.jsonl"))
        assert len(log_files) == 1
