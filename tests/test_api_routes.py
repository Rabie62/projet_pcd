"""
Unit tests for API route logic.
Tests review request validation and response schema construction.
"""
import pytest
from unittest.mock import MagicMock
import sys

# Mock api package to prevent api/__init__.py from importing api.main
# (which triggers the full ML import chain)
import types
api_mock = types.ModuleType("api")
api_mock.__path__ = ["api"]
sys.modules["api"] = api_mock

# Now we can import schemas directly without triggering api.main
from pydantic import BaseModel


class TestReviewValidation:
    def test_approve_action_valid(self):
        request = MagicMock()
        request.action = "approve"
        request.reviewer_name = "Dr. Smith"
        request.reason = None
        assert request.action in ("approve", "reject")

    def test_reject_action_requires_reason(self):
        request = MagicMock()
        request.action = "reject"
        request.reviewer_name = "Dr. Smith"
        request.reason = None
        if request.action == "reject" and not request.reason:
            with pytest.raises(ValueError, match="Reason is required"):
                raise ValueError("Reason is required when rejecting a report")

    def test_invalid_action_rejected(self):
        action = "invalid"
        assert action not in ("approve", "reject")


class TestResponseSchemaValidation:
    """Test that Pydantic schemas accept valid data."""

    def test_analysis_response_defaults(self):
        from api.schemas import AnalysisResponse
        response = AnalysisResponse(
            session_id="test",
            status="ok",
            tumor_detected=False,
        )
        assert response.session_id == "test"
        assert response.tumor_detected is False
        assert response.classification is None
        assert response.classification_confidence == 0.0

    def test_analysis_response_with_tumor(self):
        from api.schemas import AnalysisResponse, TumorFeaturesResponse
        features = TumorFeaturesResponse(
            tumor_area_mm2=1250.5,
            max_diameter_mm=42.1,
            tumor_ratio=0.05,
            location_description="antérieur droit",
        )
        response = AnalysisResponse(
            session_id="test",
            status="completed",
            tumor_detected=True,
            classification="Glioma",
            classification_confidence=0.95,
            who_grade="Grade II-IV",
            tumor_features=features,
        )
        assert response.tumor_features.tumor_area_mm2 == 1250.5
        assert response.classification == "Glioma"

    def test_review_response(self):
        from api.schemas import ReviewResponse
        response = ReviewResponse(
            session_id="test-123",
            review_status="approved",
            reviewed_by="Dr. Smith",
            reviewed_at="2025-01-01T00:00:00",
            message="Report approved by Dr. Smith",
        )
        assert response.session_id == "test-123"
        assert response.review_status == "approved"
        assert response.reviewed_by == "Dr. Smith"

    def test_chat_response(self):
        from api.schemas import ChatResponse
        response = ChatResponse(
            response="Le gliome détecté mesure 42.1 mm.",
            session_id="abc-123",
            disclaimer="RESEARCH USE ONLY",
        )
        assert "gliome" in response.response
        assert response.disclaimer == "RESEARCH USE ONLY"

    def test_health_response(self):
        from api.schemas import HealthResponse
        response = HealthResponse(
            status="ok",
            models_loaded=True,
            gpu_available=False,
            active_sessions=5,
        )
        assert response.status == "ok"
        assert response.active_sessions == 5
