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

Agent management uses AgentRegistry for dependency injection instead of
global singletons, making the code thread-safe and testable.
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


class AgentRegistry:
    """
    Dependency injection container for agent instances.

    Replaces the previous global singleton pattern (get_agents() with mutable
    global variables) with an explicit, instance-based registry that is
    thread-safe and testable.

    Usage:
        registry = AgentRegistry.create(settings)
        registry.vision.load_models()
        result = registry.vision.analyze(patient)
    """

    def __init__(self):
        self.vision: Optional[VisionAgent] = None
        self.diagnostic: Optional[DiagnosticAgent] = None
        self.safety: Optional[SafetyAgent] = None
        self.dialogue: Optional[DialogueAgent] = None
        self.rag_system: Optional[KnowledgeRAG] = None

    @classmethod
    def create(cls, settings: Optional[Settings] = None) -> "AgentRegistry":
        """Factory method: create and initialize all agents."""
        logger.info("AgentRegistry.create() called")
        registry = cls()
        resolved_settings = settings or get_settings()

        registry.vision = VisionAgent(resolved_settings)

        # Initialize RAG system
        llm_report_enabled = os.environ.get("LLM_REPORT_GENERATION", "1") == "1"
        rag_enabled = os.environ.get("RAG_ENABLED", "1") == "1"

        if rag_enabled:
            rag_model = os.environ.get(
                "RAG_EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"
            )
            qdrant_persistent = os.environ.get("QDRANT_PERSISTENT", "1") == "1"
            registry.rag_system = KnowledgeRAG(
                knowledge_dir=resolved_settings.paths.knowledge_dir,
                uploads_dir=resolved_settings.paths.uploads_dir,
                qdrant_storage_dir=resolved_settings.paths.qdrant_storage_dir,
                embedding_model=rag_model,
                persistent=qdrant_persistent,
            )
            n_indexed = registry.rag_system.index_system_knowledge()
            total = registry.rag_system.get_collection_count()
            logger.info(
                f"RAG system initialised: {n_indexed} new system chunks, "
                f"{total} total chunks in store"
            )

        registry.dialogue = DialogueAgent(
            resolved_settings, rag_system=registry.rag_system
        )
        registry.diagnostic = DiagnosticAgent(
            llm_report_enabled=llm_report_enabled,
            rag_system=registry.rag_system,
            dialogue_agent=registry.dialogue,
        )
        registry.safety = SafetyAgent(resolved_settings)

        return registry

    def load_models(self) -> None:
        """Pre-load all vision and LLM models."""
        if self.vision:
            self.vision.load_models()
        if self.dialogue:
            self.dialogue.load_model()
        logger.info("All models (vision + LLM) loaded")


# ─── Module-level registry (lazy, replaced during app startup) ───
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get the current agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry.create()
    return _registry


def set_registry(registry: AgentRegistry) -> None:
    """Set the agent registry (called at app startup)."""
    global _registry
    _registry = registry


# ─── Node functions (use registry instead of global singletons) ───

def vision_node(state: MedicalGraphState) -> dict:
    """Run VisionAgent: preprocess, segment, classify."""
    registry = get_registry()
    vision = registry.vision

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
    registry = get_registry()
    diagnostic = registry.diagnostic

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
    registry = get_registry()
    safety = registry.safety

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

    if report.tumor_detected or check.requires_human_review:
        state_update["status"] = "pending_review"
        state_update["review_status"] = "pending_review"
        logger.info(
            f"[{state['session_id']}] Report held for mandatory "
            "radiologist review before finalization."
        )
    else:
        state_update["status"] = "completed"
        state_update["review_status"] = "approved"

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


# ─── Backward-compatible aliases ───

def get_agents(settings: Optional[Settings] = None):
    """
    Deprecated: Use get_registry() instead.

    Kept for backward compatibility during migration. Returns a tuple of
    (vision, diagnostic, safety, dialogue) agents.
    """
    import warnings
    warnings.warn(
        "get_agents() is deprecated, use get_registry() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    registry = get_registry()
    return (registry.vision, registry.diagnostic, registry.safety, registry.dialogue)


def load_models(settings: Optional[Settings] = None) -> None:
    """
    Backward-compatible alias for registry.load_models().

    Deprecated: Use registry.load_models() instead.
    """
    registry = get_registry()
    registry.load_models()
