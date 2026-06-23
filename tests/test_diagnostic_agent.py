"""
Unit tests for the Diagnostic Agent.
Tests feature extraction, location determination, and report generation logic.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from agents.diagnostic import DiagnosticAgent, DiagnosticReport, TumorFeatures
from agents.vision import VisionResult, TumorRegion


@pytest.fixture
def diagnostic_agent():
    """Create a DiagnosticAgent with LLM and ICD11 mocked."""
    with patch("knowledge.icd11.ICD11Client") as mock_icd:
        mock_icd.return_value.available = False
        agent = DiagnosticAgent(
            llm_report_enabled=False,
            rag_system=None,
            dialogue_agent=MagicMock(),  # placeholder, will mock generate_llm_summary
        )
        # Mock the LLM summary generation to avoid torch dependency
        agent.generate_llm_summary = MagicMock(return_value="Rapport de test généré.")
        yield agent


def _make_vision_result(
    tumor_detected=True,
    segmentation_mask=None,
    tumor_class="Glioma",
    confidence=0.92,
    probabilities=None,
):
    if segmentation_mask is None:
        mask = np.zeros((512, 512), dtype=np.int64)
        if tumor_detected:
            mask[200:250, 200:250] = 1
        segmentation_mask = mask
    if probabilities is None:
        probabilities = {"Glioma": 0.92, "Meningioma": 0.04, "Pituitary": 0.02, "No Tumor": 0.02}

    regions = []
    if tumor_detected:
        regions = [TumorRegion(label=1, label_name="Tumor", area_pixels=2500, area_mm2=2500.0, centroid=(225.0, 225.0))]

    return VisionResult(
        patient_id="test_001",
        tumor_detected=tumor_detected,
        segmentation_mask=segmentation_mask,
        segmentation_probabilities=None,
        tumor_regions=regions,
        tumor_class=tumor_class,
        tumor_class_confidence=confidence,
        tumor_class_probabilities=probabilities,
        preprocessed_image=np.random.rand(3, 512, 512).astype(np.float32),
        original_shape=(512, 512),
        pixel_spacing=(1.0, 1.0),
        ground_truth_mask=None,
        errors=[],
    )


class TestFeatureExtraction:
    def test_no_tumor_returns_empty_features(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=False)
        features = diagnostic_agent.extract_features(vision)
        assert features.tumor_area_px == 0
        assert features.tumor_area_mm2 == 0.0

    def test_tumor_area_extracted(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=True)
        features = diagnostic_agent.extract_features(vision)
        assert features.tumor_area_px > 0
        assert features.tumor_area_mm2 > 0

    def test_tumor_ratio_calculated(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=True)
        features = diagnostic_agent.extract_features(vision)
        assert 0 < features.tumor_ratio < 1

    def test_location_description_present(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=True)
        features = diagnostic_agent.extract_features(vision)
        assert len(features.location_description) > 0


class TestLocationDetermination:
    def test_quadrant_detection(self, diagnostic_agent):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:30, 10:30] = 1
        location = diagnostic_agent.determine_location(mask, (100, 100))
        assert location in ["postérieur gauche", "postérieur droit", "antérieur gauche", "antérieur droit", "région centrale"]

    def test_central_location(self, diagnostic_agent):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[40:60, 40:60] = 1
        location = diagnostic_agent.determine_location(mask, (100, 100))
        assert location in ["antérieur gauche", "postérieur droit", "région centrale"]


class TestReportGeneration:
    def test_no_tumor_report(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=False)
        report = diagnostic_agent.generate_report(vision)
        assert report.tumor_detected is False
        assert report.classification == "No Tumor"
        assert report.tumor_features is None

    def test_tumor_report_has_features(self, diagnostic_agent):
        vision = _make_vision_result(tumor_detected=True)
        report = diagnostic_agent.generate_report(vision)
        assert report.tumor_detected is True
        assert report.tumor_features is not None
        assert report.classification == "Glioma"

    def test_low_confidence_adds_flag(self, diagnostic_agent):
        vision = _make_vision_result(confidence=0.5)
        report = diagnostic_agent.generate_report(vision)
        assert any("Confiance IA modérée" in f for f in report.flags)

    def test_report_has_patient_id(self, diagnostic_agent):
        vision = _make_vision_result()
        report = diagnostic_agent.generate_report(vision)
        assert report.patient_id == "test_001"

    def test_report_has_timestamp(self, diagnostic_agent):
        vision = _make_vision_result()
        report = diagnostic_agent.generate_report(vision)
        assert len(report.timestamp) > 0

    def test_report_to_dict(self, diagnostic_agent):
        vision = _make_vision_result()
        report = diagnostic_agent.generate_report(vision)
        d = report.to_dict()
        assert "patient_id" in d
        assert "tumor_detected" in d
        assert "classification" in d
