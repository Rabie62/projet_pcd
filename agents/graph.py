"""
LangGraph — Medical AI pipeline as a state graph.

Nodes:
    vision      — MRI preprocessing, segmentation, classification
    diagnostic  — feature extraction, structured report generation
    safety      — compliance checks, audit logging

Edges:
    START → vision → (conditional) → diagnostic → safety → END
                   → safety (no tumor)
                   → END (error)
"""

from __future__ import annotations
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from langgraph.graph import StateGraph, END
from loguru import logger
from config.settings import Settings, get_settings
from agents.graph_state import MedicalGraphState
from agents.vision import VisionAgent
from agents.diagnostic import DiagnosticAgent, DiagnosticReport
from agents.safety import SafetyAgent
from agents.dialogue import DialogueAgent
from knowledge.rag import KnowledgeRAG

# ─── RAG system singleton ───
rag_system = None

# ─── Agent singleton instances (lazily initialised) ───

vision_agent: Optional[VisionAgent] = None
diagnostic_agent: Optional[DiagnosticAgent] = None
safety_agent: Optional[SafetyAgent] = None
dialogue_agent: Optional[DialogueAgent] = None


def get_agents(settings: Optional[Settings] = None):
    """Initialise and return agent singletons on first call."""
    global vision_agent, diagnostic_agent, safety_agent, dialogue_agent
    global rag_system

    if vision_agent is not None:
        return vision_agent, diagnostic_agent, safety_agent, dialogue_agent

    resolved_settings = settings or get_settings()
    vision_agent = VisionAgent(resolved_settings)

    # Initialise RAG system
    llm_report_enabled = (
        os.environ.get("LLM_REPORT_GENERATION", "1") == "1"
    )
    rag_enabled = os.environ.get("RAG_ENABLED", "1") == "1"

    if rag_enabled:
        rag_model = os.environ.get(
            "RAG_EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"
        )
        qdrant_persistent = (
            os.environ.get("QDRANT_PERSISTENT", "1") == "1"
        )
        rag_system = KnowledgeRAG(
            knowledge_dir=resolved_settings.paths.knowledge_dir,
            uploads_dir=resolved_settings.paths.uploads_dir,
            qdrant_storage_dir=resolved_settings.paths.qdrant_storage_dir,
            embedding_model=rag_model,
            persistent=qdrant_persistent,
        )
        n_indexed = rag_system.index_system_knowledge()
        total = rag_system.get_collection_count()
        logger.info(
            f"RAG system initialised: {n_indexed} new system chunks, "
            f"{total} total chunks in store"
        )

    # Initialise dialogue agent with RAG reference (rag_system is None if disabled)
    dialogue_agent = DialogueAgent(resolved_settings, rag_system=rag_system)

    # Initialise diagnostic agent with LLM + RAG references
    diagnostic_agent = DiagnosticAgent(
        llm_report_enabled=llm_report_enabled,
        rag_system=rag_system,
        dialogue_agent=dialogue_agent,
    )

    safety_agent = SafetyAgent(resolved_settings)

    return vision_agent, diagnostic_agent, safety_agent, dialogue_agent


def load_models(settings: Optional[Settings] = None) -> None:
    """Pre-load all models (vision and LLM)."""
    vision, _, _, dialogue = get_agents(settings)
    vision.load_models()
    dialogue.load_model()
    logger.info("All models (vision + LLM) loaded via graph.load_models()")


# ─── Node functions ───


def vision_node(state: MedicalGraphState) -> dict:
    """Run VisionAgent: preprocess, segment, classify."""
    vision, _, _, _ = get_agents()

    logger.info(f"[{state['session_id']}] Vision node executing...")
    state_update = {"status": "segmenting"}

    if not vision.models_loaded:
        vision.load_models()

    result = vision.analyze(state["patient"])
    state_update["vision_result"] = result
    state_update["status"] = "vision_complete"

    if result.errors:
        state_update["errors"] = state.get("errors", []) + result.errors

    return state_update


def diagnostic_node(state: MedicalGraphState) -> dict:
    """Run DiagnosticAgent: extract features, generate report."""
    _, diagnostic, _, _ = get_agents()

    logger.info(f"[{state['session_id']}] Diagnostic node executing...")
    state_update = {"status": "diagnosing"}

    report = diagnostic.generate_report(
        state["vision_result"],
        clinical_notes=state.get("clinical_notes", "") or "",
        patient_history_context=(
            state.get("patient_history_context", "") or ""
        ),
    )
    state_update["diagnostic_report"] = report
    state_update["status"] = "diagnostic_complete"

    return state_update


def safety_node(state: MedicalGraphState) -> dict:
    """Run SafetyAgent: compliance checks and audit logging."""
    _, _, safety, _ = get_agents()

    logger.info(f"[{state['session_id']}] Safety node executing...")
    state_update = {"status": "safety_check"}

    report = state.get("diagnostic_report")
    if report is None:
        vision_result = state.get("vision_result")
        report = DiagnosticReport(
            patient_id=state["patient_id"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            tumor_detected=False,
            tumor_features=None,
            classification=vision_result.tumor_class if vision_result else None,
            classification_confidence=vision_result.tumor_class_confidence if vision_result else 0.0,
            classification_probabilities=vision_result.tumor_class_probabilities if vision_result else {},
            who_grade=None,
            clinical_summary="No tumor detected. Recommend clinical correlation.",
            recommendations=["Clinical correlation recommended"],
            flags=[],
        )
        state_update["diagnostic_report"] = report

    check = safety.check_safety(report)
    safety.log_audit(report, check)
    state_update["safety_check"] = check

    # ── Human review gate ──
    # All tumor-positive cases require mandatory radiologist review.
    # No-tumor cases with safety flags also require review.
    if report.tumor_detected or check.requires_human_review:
        state_update["status"] = "pending_review"
        state_update["review_status"] = "pending_review"
        logger.info(
            f"[{state['session_id']}] Report held for mandatory "
            "radiologist review before finalization."
        )
    else:
        state_update["status"] = "completed"
        state_update["review_status"] = "approved"  # auto-approve no-tumor/clean

    return state_update


# ─── Conditional routing ───


def route_after_vision(state: MedicalGraphState) -> str:
    """Decide next node after vision completes."""
    if state.get("status") == "failed":
        return END

    vision_result = state.get("vision_result")
    if vision_result is None:
        return END

    return "diagnostic"


# ─── Build the graph ───


def build_graph() -> StateGraph:
    """Construct the medical analysis StateGraph."""
    graph = StateGraph(MedicalGraphState)

    graph.add_node("vision", vision_node)
    graph.add_node("diagnostic", diagnostic_node)
    graph.add_node("safety", safety_node)

    graph.set_entry_point("vision")

    graph.add_conditional_edges("vision", route_after_vision)
    graph.add_edge("diagnostic", "safety")
    graph.add_edge("safety", END)

    return graph


def compile_graph():
    """Build and compile the graph into a runnable app."""
    return build_graph().compile()


def create_initial_state(patient) -> MedicalGraphState:
    """Create the initial state for a graph invocation."""
    return MedicalGraphState(
        patient=patient,
        clinical_notes=None,
        patient_history_context=None,
        vision_result=None,
        diagnostic_report=None,
        safety_check=None,
        chat_response=None,
        review_status=None,
        reviewed_by=None,
        reviewed_at=None,
        rejection_reason=None,
        session_id=str(uuid.uuid4()),
        patient_id=patient.patient_id,
        status="pending",
        errors=[],
    )
