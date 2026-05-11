from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger
from config.settings import Settings, get_settings
from agents.graph import compile_graph, create_initial_state, load_models, get_agents
from agents.graph_state import MedicalGraphState
from agents.safety import AuditEntry
from data.loader import BRISCDataLoader, BRISCPatient
from data.patient_store import PatientStore


class ControllerAgent:
    """
    Controller Agent — central orchestrator for the medical AI system.

    Uses LangGraph to manage the analysis pipeline:
        START → VisionAgent → DiagnosticAgent → SafetyAgent → END

    With conditional routing:
        - Tumor found → full pipeline
        - No tumor → skip diagnostic, go to safety
        - Error → END with error state
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.graph = compile_graph()
        self.sessions: dict[str, MedicalGraphState] = {}
        self.patient_store = PatientStore()
        self.models_loaded = False
        logger.info("Controller Agent initialised (LangGraph + PatientStore)")

    def load_models(self) -> None:
        """Pre-load all vision models at startup."""
        logger.info("Loading all models...")
        load_models(self.settings)
        self.models_loaded = True
        logger.info("All vision models loaded. Dialogue model loads on first use.")

    def analyze_patient(
        self,
        patient: BRISCPatient,
        patient_id: Optional[int] = None,
        clinical_notes: Optional[str] = None,
    ) -> MedicalGraphState:
        """
        Run the full analysis pipeline on a patient via LangGraph.

        Args:
            patient: loaded BRISC patient data
            patient_id: optional patient ID (int) for history lookup
            clinical_notes: optional clinical notes for this scan

        Returns:
            Final MedicalGraphState with all results
        """
        if not self.models_loaded:
            self.load_models()

        initial_state = create_initial_state(patient)
        session_id = initial_state["session_id"]
        initial_state["patient_id"] = patient_id

        # Inject patient history context if patient_id is provided
        if patient_id:
            history_context = self.patient_store.format_history_for_prompt(
                patient_id
            )
            if history_context:
                initial_state["patient_history_context"] = history_context
                logger.info(
                    f"[{session_id}] Patient history injected for ID {patient_id}"
                )

        if clinical_notes:
            initial_state["clinical_notes"] = clinical_notes

        logger.info(
            f"[{session_id}] Starting graph for patient {patient.patient_id}"
        )

        final_state = self.graph.invoke(initial_state)
        self.sessions[session_id] = final_state

        # Set dialogue context if a report was generated
        if final_state.get("diagnostic_report"):
            _, _, _, dialogue_agent = get_agents()
            dialogue_agent.set_report_context(final_state["diagnostic_report"])

        # Auto-link scan to patient record if patient_id provided
        if patient_id and final_state.get("diagnostic_report"):
            report = final_state["diagnostic_report"]
            features = report.tumor_features
            self.patient_store.link_scan(
                patient_id=patient_id,
                session_id=session_id,
                classification=report.classification,
                confidence=report.classification_confidence,
                tumor_area_mm2=(
                    features.tumor_area_mm2 if features else None
                ),
                max_diameter_mm=(
                    features.max_diameter_mm if features else None
                ),
                tumor_location=(
                    features.location_description if features else None
                ),
                clinical_summary=report.clinical_summary,
                review_status=final_state.get(
                    "review_status", "pending_review"
                ),
            )
            logger.info(
                f"[{session_id}] Scan auto-linked to patient ID {patient_id}"
            )

        status = final_state.get("status", "unknown")
        vision_result = final_state.get("vision_result")
        tumor_detected = (
            vision_result.tumor_detected if vision_result else "N/A"
        )

        logger.info(
            f"[{session_id}] Graph complete — "
            f"status: {status}, tumor: {tumor_detected}"
        )

        return final_state

    def analyze_image_dir(
        self,
        image_dir,
        patient_id: Optional[int] = None,
        clinical_notes: Optional[str] = None,
    ) -> MedicalGraphState:
        """
        Analyze a patient from a directory of images.

        Args:
            image_dir: path to directory containing BRISC images
            patient_id: optional patient ID (int) for history lookup
            clinical_notes: optional clinical notes for this scan

        Returns:
            MedicalGraphState
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        loader = BRISCDataLoader(image_dir)
        patients = list(loader.iterate_patients(max_patients=1))
        if not patients:
            raise FileNotFoundError(f"No images found in: {image_dir}")

        return self.analyze_patient(
            patients[0],
            patient_id=patient_id,
            clinical_notes=clinical_notes,
        )

    def analyze_image_file(
        self,
        image_path,
        patient_id: Optional[int] = None,
        clinical_notes: Optional[str] = None,
    ) -> MedicalGraphState:
        """
        Analyze a single image file.

        Args:
            image_path: path to a JPG image file
            patient_id: optional patient ID (int)

        Returns:
            MedicalGraphState
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        loader = BRISCDataLoader(image_path.parent)
        patient = loader.load_patient_from_path(image_path)
        return self.analyze_patient(
            patient,
            patient_id=patient_id,
            clinical_notes=clinical_notes,
        )

    def chat(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Handle a doctor chat query via the Dialogue Agent (Synchronous).

        Args:
            query: natural language question
            session_id: optional session ID to ground the conversation

        Returns:
            Natural language response from the dialogue LLM
        """
        _, _, _, dialogue_agent = get_agents()

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if session.get("diagnostic_report"):
                dialogue_agent.set_report_context(session["diagnostic_report"])

        return dialogue_agent.chat(query)

    def chat_stream(self, query: str, session_id: Optional[str] = None):
        """
        Handle a doctor chat query via the Dialogue Agent (Streaming).

        Args:
            query: natural language question
            session_id: optional session ID to ground the conversation

        Yields:
            Text chunks from the dialogue LLM
        """
        _, _, _, dialogue_agent = get_agents()

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if session.get("diagnostic_report"):
                dialogue_agent.set_report_context(session["diagnostic_report"])

        yield from dialogue_agent.chat_stream(query)

    def get_session(self, session_id: str) -> Optional[MedicalGraphState]:
        """Retrieve an analysis session by its ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all analysis sessions with summary info."""
        results = []
        for sid, session in self.sessions.items():
            report = session.get("diagnostic_report")
            results.append({
                "session_id": sid,
                "patient_id": session.get("patient_id", "unknown"),
                "status": session.get("status", "unknown"),
                "tumor_detected": report.tumor_detected if report else None,
                "review_status": session.get("review_status"),
            })
        return results

    def get_disclaimer(self) -> str:
        """Get the safety disclaimer text."""
        _, _, safety_agent, _ = get_agents()
        return safety_agent.get_disclaimer()

    # ── Human Review Gate ─────────────────────────────────────────────

    def approve_session(
        self, session_id: str, reviewer_name: str
    ) -> MedicalGraphState:
        """
        Approve a pending session — radiologist signs off on the report.

        Args:
            session_id: analysis session to approve
            reviewer_name: name/ID of the approving radiologist

        Returns:
            Updated session state

        Raises:
            KeyError: if session not found
            ValueError: if session is not in pending_review status
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        if session.get("review_status") != "pending_review":
            raise ValueError(
                f"Session {session_id} is not pending review "
                f"(current status: {session.get('review_status')})"
            )

        session["review_status"] = "approved"
        session["reviewed_by"] = reviewer_name
        session["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        session["status"] = "completed"

        # Log the approval in the audit trail
        _, _, safety_agent, _ = get_agents()
        report = session.get("diagnostic_report")
        safety_check = session.get("safety_check")
        if report and safety_check:
            entry = AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                patient_id=session.get("patient_id", "unknown"),
                action="radiologist_approval",
                input_hash="N/A",
                model_version=safety_agent.MODEL_VERSION,
                results_summary=f"Report approved by {reviewer_name}",
                confidence=report.classification_confidence,
                safety_check=asdict(safety_check),
                flags=[],
                metadata={"reviewer": reviewer_name},
            )
            safety_agent.audit_log.append(entry)
            safety_agent.persist_audit_entry(entry)

        logger.info(
            f"[{session_id}] Report APPROVED by {reviewer_name}"
        )

        # Sync review status to patient scan record
        self.patient_store.update_scan_review(
            session_id=session_id,
            review_status="approved",
            reviewed_by=reviewer_name,
        )

        return session

    def reject_session(
        self, session_id: str, reviewer_name: str, reason: str
    ) -> MedicalGraphState:
        """
        Reject a pending session — radiologist flags issues with the report.

        Args:
            session_id: analysis session to reject
            reviewer_name: name/ID of the rejecting radiologist
            reason: explanation for the rejection

        Returns:
            Updated session state

        Raises:
            KeyError: if session not found
            ValueError: if session is not in pending_review status
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        if session.get("review_status") != "pending_review":
            raise ValueError(
                f"Session {session_id} is not pending review "
                f"(current status: {session.get('review_status')})"
            )

        session["review_status"] = "rejected"
        session["reviewed_by"] = reviewer_name
        session["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        session["rejection_reason"] = reason
        session["status"] = "rejected"

        # Log the rejection in the audit trail
        _, _, safety_agent, _ = get_agents()
        report = session.get("diagnostic_report")
        safety_check = session.get("safety_check")
        if report and safety_check:
            entry = AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                patient_id=session.get("patient_id", "unknown"),
                action="radiologist_rejection",
                input_hash="N/A",
                model_version=safety_agent.MODEL_VERSION,
                results_summary=f"Report rejected by {reviewer_name}: {reason}",
                confidence=report.classification_confidence,
                safety_check=asdict(safety_check),
                flags=[f"REJECTED: {reason}"],
                metadata={"reviewer": reviewer_name, "reason": reason},
            )
            safety_agent.audit_log.append(entry)
            safety_agent.persist_audit_entry(entry)

        logger.info(
            f"[{session_id}] Report REJECTED by {reviewer_name}: {reason}"
        )

        # Sync review status to patient scan record
        self.patient_store.update_scan_review(
            session_id=session_id,
            review_status="rejected",
            reviewed_by=reviewer_name,
        )

        return session
