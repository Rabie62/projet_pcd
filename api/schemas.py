"""
Pydantic schemas for API request/response models.
Adapted for BRISC 2025 dataset.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    query: str = Field(..., description="Clinician's natural language query")
    session_id: Optional[str] = Field(
        None, description="Analysis session ID for context"
    )


class AnalyzeRequest(BaseModel):
    """Request for analyzing a directory of BRISC images."""
    patient_dir: str = Field(
        ..., description="Path to directory with BRISC 2025 JPG images"
    )
    patient_id: Optional[int] = Field(
        None,
        description="Patient ID — if provided, fetches history and auto-links scan",
    )
    clinical_notes: Optional[str] = Field(
        None,
        description="Clinical notes from the referring physician for this scan",
    )


class ReviewRequest(BaseModel):
    """Request for reviewing (approving/rejecting) an analysis session."""
    action: str = Field(
        ..., description="Review action: 'approve' or 'reject'"
    )
    reviewer_name: str = Field(
        ..., description="Name or ID of the reviewing radiologist"
    )
    reason: Optional[str] = Field(
        None, description="Reason for rejection (required if action is 'reject')"
    )


# ── Responses ─────────────────────────────────────────────────────────

class TumorRegionResponse(BaseModel):
    label: int
    label_name: str
    area_pixels: int
    area_mm2: float
    centroid: list[float]


class TumorFeaturesResponse(BaseModel):
    tumor_area_mm2: float
    max_diameter_mm: float
    tumor_ratio: float
    location_description: str


class SafetyCheckResponse(BaseModel):
    passed: bool
    confidence_adequate: bool
    requires_human_review: bool
    flags: list[str]
    warnings: list[str]
    compliance_notes: list[str]


class AnalysisResponse(BaseModel):
    """Full analysis result."""
    session_id: str
    patient_id: Optional[int] = None
    status: str
    tumor_detected: bool
    classification: Optional[str] = None
    classification_confidence: float = 0.0
    who_grade: Optional[str] = None
    tumor_features: Optional[TumorFeaturesResponse] = None
    tumor_regions: list[TumorRegionResponse] = []
    clinical_summary: str = ""
    recommendations: list[str] = []
    icd11_codes: list[str] = []
    safety_check: Optional[SafetyCheckResponse] = None
    review_status: Optional[str] = None
    disclaimer: str = ""
    errors: list[str] = []


class ReviewResponse(BaseModel):
    """Response after a review action."""
    session_id: str
    review_status: str
    reviewed_by: str
    reviewed_at: str
    message: str


class ChatResponse(BaseModel):
    """Chat response from the Dialogue Agent."""
    response: str
    session_id: Optional[str] = None
    disclaimer: str = ""


class HealthResponse(BaseModel):
    """System health check."""
    status: str
    models_loaded: bool
    gpu_available: bool
    active_sessions: int


class SessionListItem(BaseModel):
    session_id: str
    patient_id: Optional[int] = None
    status: str
    tumor_detected: Optional[bool] = None
    review_status: Optional[str] = None


# ── Knowledge Base / Document schemas ─────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document to the knowledge base."""
    document_id: str
    filename: str
    uploaded_by: str
    uploaded_at: str
    file_type: str
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    """Summary of an uploaded document."""
    document_id: str
    filename: str
    uploaded_by: str
    uploaded_at: str
    file_type: str
    chunk_count: int


class KnowledgeBaseStatus(BaseModel):
    """Status of the RAG knowledge base."""
    available: bool
    total_chunks: int
    uploaded_documents: int
    system_knowledge_indexed: bool


# ── Patient History schemas ─────────────────────────────────────────────

class PatientCreateRequest(BaseModel):
    """Request to register a new patient."""
    prenom: str = Field(..., description="Patient first name")
    nom: str = Field(..., description="Patient last name")
    date_naissance: Optional[str] = Field(None, description="Birth date (YYYY-MM-DD)")
    genre: Optional[str] = Field(None, description="M or F")
    tel: Optional[int] = Field(None, description="Phone number")
    poids: Optional[float] = Field(None, description="Weight in kg")
    taille: Optional[float] = Field(None, description="Height in cm")
    FC: Optional[int] = Field(None, description="Heart rate (bpm)")
    glycemie: Optional[float] = Field(None, description="Blood sugar (g/L)")


class PatientUpdateRequest(BaseModel):
    """Request to update an existing patient's information."""
    prenom: Optional[str] = None
    nom: Optional[str] = None
    date_naissance: Optional[str] = None
    genre: Optional[str] = None
    tel: Optional[int] = None
    poids: Optional[float] = None
    taille: Optional[float] = None
    FC: Optional[int] = None
    glycemie: Optional[float] = None


class ScanHistoryItem(BaseModel):
    """Summary of a single scan in a patient's history."""
    session_id: str
    scan_date: str
    classification: Optional[str] = None
    confidence: Optional[float] = None
    tumor_area_mm2: Optional[float] = None
    max_diameter_mm: Optional[float] = None
    tumor_location: Optional[str] = None
    review_status: Optional[str] = None
    reviewed_by: Optional[str] = None


class PatientResponse(BaseModel):
    """Full patient profile with scan history."""
    id: int
    prenom: str
    nom: str
    date_naissance: Optional[str] = None
    genre: Optional[str] = None
    tel: Optional[int] = None
    poids: Optional[float] = None
    taille: Optional[float] = None
    FC: Optional[int] = None
    glycemie: Optional[float] = None
    scan_count: int = 0
    scans: list[ScanHistoryItem] = []
    created_at: str = ""
    updated_at: Optional[str] = None


# ── Médecin schemas ─────────────────────────────────────────────────────

class MedecinCreateRequest(BaseModel):
    """Request to register a new doctor."""
    nom: str = Field(..., description="Doctor last name")
    prenom: str = Field(..., description="Doctor first name")
    specialite: Optional[str] = Field(None, description="Medical specialty")
    tel: Optional[str] = Field(None, description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    departement: Optional[str] = Field(None, description="Department / service")


class MedecinUpdateRequest(BaseModel):
    """Request to update an existing doctor's information."""
    nom: Optional[str] = None
    prenom: Optional[str] = None
    specialite: Optional[str] = None
    tel: Optional[str] = None
    email: Optional[str] = None
    departement: Optional[str] = None


class ConsultationListItem(BaseModel):
    """Summary of a single consultation."""
    id: int
    patient_id: int
    patient_nom: Optional[str] = None
    patient_prenom: Optional[str] = None
    medecin_id: int
    medecin_nom: Optional[str] = None
    medecin_prenom: Optional[str] = None
    scan_record_id: Optional[int] = None
    session_id: Optional[str] = None
    date_consultation: str
    motif: Optional[str] = None
    diagnostic: Optional[str] = None
    statut: Optional[str] = None
    created_at: str = ""


class ConsultationResponse(BaseModel):
    """Full consultation details."""
    id: int
    patient_id: int
    patient_nom: Optional[str] = None
    patient_prenom: Optional[str] = None
    medecin_id: int
    medecin_nom: Optional[str] = None
    medecin_prenom: Optional[str] = None
    medecin_specialite: Optional[str] = None
    scan_record_id: Optional[int] = None
    session_id: Optional[str] = None
    date_consultation: str
    motif: Optional[str] = None
    diagnostic: Optional[str] = None
    notes: Optional[str] = None
    rapport_genere: Optional[str] = None
    statut: Optional[str] = None
    created_at: str = ""
    updated_at: Optional[str] = None


class ConsultationCreateRequest(BaseModel):
    """Request to create a new consultation."""
    patient_id: int = Field(..., description="Patient ID")
    medecin_id: int = Field(..., description="Doctor ID")
    scan_record_id: Optional[int] = Field(None, description="Linked scan record ID")
    motif: Optional[str] = Field(None, description="Reason for consultation")
    diagnostic: Optional[str] = Field(None, description="Diagnosis")
    notes: Optional[str] = Field(None, description="Additional notes")
    rapport_genere: Optional[str] = Field(None, description="Generated report content")
    statut: Optional[str] = Field("en_cours", description="Consultation status")


class ConsultationUpdateRequest(BaseModel):
    """Request to update an existing consultation."""
    medecin_id: Optional[int] = None
    motif: Optional[str] = None
    diagnostic: Optional[str] = None
    notes: Optional[str] = None
    rapport_genere: Optional[str] = None
    statut: Optional[str] = None


class MedecinResponse(BaseModel):
    """Full doctor profile."""
    id: int
    nom: str
    prenom: str
    specialite: Optional[str] = None
    tel: Optional[str] = None
    email: Optional[str] = None
    departement: Optional[str] = None
    consultation_count: int = 0
    created_at: str = ""
    updated_at: Optional[str] = None


class MedecinListItem(BaseModel):
    """Summary of a doctor for list endpoints."""
    id: int
    nom: str
    prenom: str
    specialite: Optional[str] = None
    tel: Optional[str] = None
    email: Optional[str] = None
    departement: Optional[str] = None
    consultation_count: int = 0
    created_at: str = ""


class PatientListItem(BaseModel):
    """Summary of a patient for list endpoints."""
    id: int
    prenom: str
    nom: str
    genre: Optional[str] = None
    tel: Optional[int] = None
    age: Optional[int] = None
    poids: Optional[float] = None
    taille: Optional[float] = None
    FC: Optional[int] = None
    glycemie: Optional[float] = None
    scan_count: int = 0
    created_at: str = ""

