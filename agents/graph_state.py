"""
Graph State — typed state that flows through the LangGraph pipeline.
Uses TypedDict for LangGraph compatibility.
"""

from __future__ import annotations
from typing import Optional, TypedDict
from data.loader import BRISCPatient
from agents.vision import VisionResult
from agents.diagnostic import DiagnosticReport
from agents.safety import SafetyCheck


class MedicalGraphState(TypedDict):
    """Shared state passed between all graph nodes."""

    # Input
    patient: BRISCPatient
    clinical_notes: Optional[str]           
    patient_history_context: Optional[str]  

    # Agent outputs 
    vision_result: Optional[VisionResult]
    diagnostic_report: Optional[DiagnosticReport]
    safety_check: Optional[SafetyCheck]
    chat_response: Optional[str]

    # Human review gate 
    review_status: Optional[str]  # "pending_review" | "approved" | "rejected"
    reviewed_by: Optional[str]
    reviewed_at: Optional[str]
    rejection_reason: Optional[str]

    # Metadata
    session_id: str
    patient_id: str
    status: str
    errors: list[str]
