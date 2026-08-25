"""
Unit tests for the Dialogue Agent session memory.
Tests per-session conversation isolation, history tracking, and report context binding.
"""
import pytest
from unittest.mock import MagicMock
from agents.dialogue import DialogueAgent
from agents.diagnostic import DiagnosticReport, TumorFeatures


@pytest.fixture
def mock_dialogue_agent():
    """Create a DialogueAgent with mocked tokenizer and model."""
    agent = DialogueAgent()
    agent.model_loaded = True
    agent.tokenizer = MagicMock()
    agent.tokenizer.pad_token_id = 0
    agent.tokenizer.eos_token_id = 1
    
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 3)
    agent.tokenizer.apply_chat_template = MagicMock(return_value={"input_ids": mock_input_ids})
    agent.tokenizer.decode = MagicMock(return_value="Réponse de test pour le médecin.")
    agent.model = MagicMock()
    agent.model.generate = MagicMock(return_value=[[1, 2, 3, 4, 5]])
    agent.model.device = "cpu"
    return agent


def _make_sample_report(patient_id="patient_101", classification="Glioma", tumor_detected=True):
    return DiagnosticReport(
        patient_id=patient_id,
        timestamp="2026-08-25T12:00:00Z",
        tumor_detected=tumor_detected,
        tumor_features=TumorFeatures(
            tumor_area_mm2=1250.0,
            max_diameter_mm=38.0,
            location_description="antérieur droit",
            tumor_ratio=0.08,
        ) if tumor_detected else None,
        classification=classification,
        classification_confidence=0.94,
        classification_probabilities={classification: 0.94},
        who_grade="Grade II-IV",
        clinical_summary="Masse suspecte détectée.",
        recommendations=["Consultation neurochirurgicale"],
        flags=[],
    )


class TestDialogueSessionMemory:
    def test_empty_history_for_new_session(self, mock_dialogue_agent):
        history = mock_dialogue_agent.get_session_history("session_abc")
        assert history == []

    def test_record_and_retrieve_exchange(self, mock_dialogue_agent):
        mock_dialogue_agent.add_session_exchange("session_1", "Quelle est la taille ?", "La surface est de 1250 mm².")
        history = mock_dialogue_agent.get_session_history("session_1")
        assert len(history) == 1
        assert history[0]["user"] == "Quelle est la taille ?"
        assert "1250 mm²" in history[0]["assistant"]

    def test_session_isolation(self, mock_dialogue_agent):
        mock_dialogue_agent.add_session_exchange("session_A", "Question A", "Réponse A")
        mock_dialogue_agent.add_session_exchange("session_B", "Question B", "Réponse B")

        hist_a = mock_dialogue_agent.get_session_history("session_A")
        hist_b = mock_dialogue_agent.get_session_history("session_B")

        assert len(hist_a) == 1
        assert hist_a[0]["user"] == "Question A"
        assert len(hist_b) == 1
        assert hist_b[0]["user"] == "Question B"

    def test_per_session_report_context(self, mock_dialogue_agent):
        report_a = _make_sample_report(patient_id="patient_A", classification="Glioma")
        report_b = _make_sample_report(patient_id="patient_B", classification="Meningioma")

        mock_dialogue_agent.set_report_context(report_a, session_id="session_A")
        mock_dialogue_agent.set_report_context(report_b, session_id="session_B")

        rep_a = mock_dialogue_agent.get_session_report("session_A")
        rep_b = mock_dialogue_agent.get_session_report("session_B")

        assert rep_a.patient_id == "patient_A"
        assert rep_a.classification == "Glioma"
        assert rep_b.patient_id == "patient_B"
        assert rep_b.classification == "Meningioma"

    def test_clear_session_history(self, mock_dialogue_agent):
        mock_dialogue_agent.add_session_exchange("session_1", "Q1", "R1")
        mock_dialogue_agent.add_session_exchange("session_2", "Q2", "R2")

        mock_dialogue_agent.clear_session_history("session_1")
        assert mock_dialogue_agent.get_session_history("session_1") == []
        assert len(mock_dialogue_agent.get_session_history("session_2")) == 1

    def test_chat_method_records_into_session_memory(self, mock_dialogue_agent):
        report = _make_sample_report(patient_id="patient_99", classification="Pituitary")
        mock_dialogue_agent.set_report_context(report, session_id="session_99")

        res = mock_dialogue_agent.chat("Où est située la lésion ?", session_id="session_99")
        assert res == "Réponse de test pour le médecin."

        history = mock_dialogue_agent.get_session_history("session_99")
        assert len(history) == 1
        assert history[0]["user"] == "Où est située la lésion ?"
        assert history[0]["assistant"] == "Réponse de test pour le médecin."
