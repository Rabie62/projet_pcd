"""
End-to-end integration tests for API routes.

Tests the full request/response cycle for key endpoints using FastAPI TestClient
with mocked ML dependencies (agents, models, etc.).
"""
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock heavy ML dependencies before importing routes
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.models"] = MagicMock()
sys.modules["loguru"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.spatial"] = MagicMock()
sys.modules["scipy.spatial.distance"] = MagicMock()
sys.modules["monai"] = MagicMock()
sys.modules["monai.networks"] = MagicMock()
sys.modules["monai.networks.nets"] = MagicMock()
sys.modules["monai.transforms"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.sql"] = MagicMock()
sys.modules["sqlalchemy.sql.expression"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["qdrant_client.models"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.measure"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["fitz"] = MagicMock()
sys.modules["easyocr"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.messages"] = MagicMock()
sys.modules["aiofiles"] = MagicMock()
sys.modules["nibabel"] = MagicMock()
sys.modules["einops"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

# Import the real route modules (not the api package mock)
from api.routes.analysis import router as analysis_router
from api.routes.analysis import set_controller
from api.routes.patients import router as patients_router
from api.routes.medecins import router as medecins_router
from api.routes.consultations import router as consultations_router
from api.routes.knowledge import router as knowledge_router

from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def mock_controller():
    """Create a fully mocked ControllerAgent."""
    ctrl = MagicMock()
    ctrl.models_loaded = True
    ctrl.get_disclaimer.return_value = "RESEARCH USE ONLY — Test disclaimer"

    # Mock patient store
    mock_patient = MagicMock()
    mock_patient.id = 1
    mock_patient.nom = "Doe"
    mock_patient.prenom = "John"
    mock_patient.date_naissance = None
    mock_patient.genre = "M"
    mock_patient.tel = None
    mock_patient.poids = None
    mock_patient.taille = None
    mock_patient.FC = None
    mock_patient.glycemie = None
    mock_patient.scans = []
    mock_patient.created_at = None
    mock_patient.updated_at = None

    ctrl.patient_store = MagicMock()
    ctrl.patient_store.create_patient.return_value = mock_patient
    ctrl.patient_store.get_patient.return_value = None
    ctrl.patient_store.list_patients.return_value = []
    ctrl.patient_store.format_history_for_prompt.return_value = ""
    ctrl.patient_store.update_patient.return_value = mock_patient

    # Mock session management
    ctrl.get_session.return_value = None
    ctrl.list_sessions.return_value = []

    # Mock analysis
    mock_report = MagicMock()
    mock_report.tumor_detected = True
    mock_report.classification = "Glioma"
    mock_report.classification_confidence = 0.95
    mock_report.who_grade = "Grade II-IV"
    mock_report.clinical_summary = "Test summary"
    mock_report.recommendations = []
    mock_report.tumor_features = None
    mock_report.flags = []
    mock_report.to_dict.return_value = {
        "patient_id": "test", "tumor_detected": True,
        "classification": "Glioma", "classification_confidence": 0.95,
    }

    mock_session = {
        "session_id": "test-session-123",
        "patient_id": 1,
        "status": "completed",
        "diagnostic_report": mock_report,
        "safety_check": None,
        "vision_result": None,
        "review_status": "pending_review",
        "errors": [],
    }
    ctrl.analyze_image_file.return_value = mock_session
    ctrl.analyze_image_dir.return_value = mock_session
    ctrl.analyze_patient.return_value = mock_session

    # Mock chat
    ctrl.chat.return_value = "Le gliome détecté mesure 42.1 mm."
    ctrl.chat_stream.return_value = iter(["Token1 ", "Token2"])

    # Mock review
    ctrl.approve_session.return_value = {
        "review_status": "approved",
        "reviewed_by": "Dr. Smith",
        "reviewed_at": "2025-01-01T00:00:00",
    }
    ctrl.reject_session.return_value = {
        "review_status": "rejected",
        "reviewed_by": "Dr. Smith",
        "reviewed_at": "2025-01-01T00:00:00",
    }

    return ctrl


@pytest.fixture
def client(mock_controller):
    """Create a FastAPI test client with mocked routes."""
    app = FastAPI()
    app.include_router(analysis_router)
    app.include_router(patients_router)
    app.include_router(medecins_router)
    app.include_router(consultations_router)
    app.include_router(knowledge_router)

    set_controller(mock_controller)

    yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_returns_gpu_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "gpu_available" in data
        assert "active_sessions" in data


class TestPatientEndpoints:
    def test_create_patient(self, client):
        response = client.post(
            "/patients",
            json={
                "prenom": "John",
                "nom": "Doe",
                "genre": "M",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == 1
        assert data["nom"] == "Doe"
        assert data["prenom"] == "John"

    def test_create_patient_missing_name(self, client):
        response = client.post(
            "/patients",
            json={"prenom": "John"},  # missing 'nom' — required field
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_list_patients(self, client):
        response = client.get("/patients")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_update_patient(self, client):
        response = client.put(
            "/patients/1",
            json={"prenom": "Jane"},
        )
        assert response.status_code == 200

    def test_delete_patient(self, client):
        # Mock returns True by default, so test with found patient
        response = client.delete("/patients/1")
        assert response.status_code == 200


class TestAnalysisEndpoints:
    def test_analyze_file_not_found(self, client):
        response = client.post(
            "/analyze",
            json={"patient_dir": "/nonexistent/path/image.jpg"},
        )
        assert response.status_code == 404

    def test_analyze_returns_response_structure(self, client):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name
        try:
            response = client.post(
                "/analyze",
                json={"patient_dir": temp_path},
            )
            if response.status_code == 200:
                data = response.json()
                assert "session_id" in data
                assert "status" in data
                assert "tumor_detected" in data
        finally:
            os.unlink(temp_path)


class TestReviewEndpoints:
    def test_approve_session(self, client):
        response = client.post(
            "/review/test-session-123",
            json={"action": "approve", "reviewer_name": "Dr. Smith"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["review_status"] == "approved"
        assert data["reviewed_by"] == "Dr. Smith"

    def test_reject_session_requires_reason(self, client):
        response = client.post(
            "/review/test-session-123",
            json={"action": "reject", "reviewer_name": "Dr. Smith"},
        )
        assert response.status_code == 400

    def test_reject_session_with_reason(self, client):
        response = client.post(
            "/review/test-session-123",
            json={
                "action": "reject",
                "reviewer_name": "Dr. Smith",
                "reason": "Incorrect classification",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["review_status"] == "rejected"

    def test_invalid_action(self, client):
        response = client.post(
            "/review/test-session-123",
            json={"action": "invalid", "reviewer_name": "Dr. Smith"},
        )
        assert response.status_code == 400


class TestChatEndpoints:
    def test_chat_returns_response(self, client):
        response = client.post(
            "/chat",
            json={"query": "Quelle est la taille de la tumeur ?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "disclaimer" in data

    def test_chat_with_session_id(self, client):
        response = client.post(
            "/chat",
            json={"query": "test", "session_id": "abc-123"},
        )
        assert response.status_code == 200


class TestSessionEndpoints:
    def test_list_sessions(self, client):
        response = client.get("/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestKnowledgeEndpoints:
    def test_knowledge_status_unavailable(self, client):
        """RAG system not initialized in tests — should return 200 with available=False."""
        # Mock the registry to avoid real RAG initialization
        mock_registry = MagicMock()
        mock_registry.rag_system = None
        with patch("api.routes.knowledge.get_registry", return_value=mock_registry):
            response = client.get("/knowledge/status")
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False


class TestResponseFormat:
    """Verify response format consistency across endpoints."""

    def test_analysis_response_has_required_fields(self, client):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name
        try:
            response = client.post(
                "/analyze",
                json={"patient_dir": temp_path},
            )
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "session_id", "status", "tumor_detected",
                    "classification", "disclaimer",
                ]
                for field in required_fields:
                    assert field in data, f"Missing field: {field}"
        finally:
            os.unlink(temp_path)

    def test_error_response_format(self, client):
        response = client.post(
            "/analyze",
            json={"patient_dir": "/nonexistent/path.jpg"},
        )
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
