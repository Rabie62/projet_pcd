"""
FastAPI route definitions for the Medical AI Agent API.
Adapted for BRISC 2025 dataset (2D T1-weighted MRI images).
"""

from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    ChatRequest,
    ChatResponse,
    ConsultationCreateRequest,
    ConsultationListItem,
    ConsultationResponse,
    ConsultationUpdateRequest,
    DocumentListItem,
    DocumentUploadResponse,
    HealthResponse,
    KnowledgeBaseStatus,
    MedecinCreateRequest,
    MedecinListItem,
    MedecinResponse,
    MedecinUpdateRequest,
    PatientCreateRequest,
    PatientListItem,
    PatientResponse,
    PatientUpdateRequest,
    ReviewRequest,
    ReviewResponse,
    SafetyCheckResponse,
    ScanHistoryItem,
    SessionListItem,
    TumorFeaturesResponse,
    TumorRegionResponse,
)
from agents.controller import ControllerAgent
from agents import graph
from interpretability.visualizations import BRISCVisualizer

router = APIRouter()

# Singleton controller — initialized at startup
controller_instance: Optional[ControllerAgent] = None
visualizer_instance: Optional[BRISCVisualizer] = None


def get_controller() -> ControllerAgent:
    global controller_instance
    if controller_instance is None:
        raise HTTPException(503, "System not initialized. Wait for startup.")
    return controller_instance


def set_controller(controller: ControllerAgent) -> None:
    global controller_instance
    controller_instance = controller


def get_visualizer() -> BRISCVisualizer:
    global visualizer_instance
    if visualizer_instance is None:
        visualizer_instance = BRISCVisualizer()
    return visualizer_instance


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and model status."""
    ctrl = get_controller()
    return HealthResponse(
        status="ok",
        models_loaded=ctrl.models_loaded,
        gpu_available=torch.cuda.is_available(),
        active_sessions=len(ctrl.sessions),
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_patient(request: AnalyzeRequest):
    """
    Analyze a patient's MRI image (BRISC 2D).
    """
    ctrl = get_controller()
    patient_dir = Path(request.patient_dir)

    if not patient_dir.exists():
        raise HTTPException(404, f"Patient directory not found: {patient_dir}")

    try:
        if patient_dir.is_file():
            session = ctrl.analyze_image_file(
                patient_dir,
                patient_id=request.patient_id,
                clinical_notes=request.clinical_notes,
            )
        else:
            session = ctrl.analyze_image_dir(
                patient_dir,
                patient_id=request.patient_id,
                clinical_notes=request.clinical_notes,
            )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {e}")

    # Build response
    report = session.get("diagnostic_report")
    safety = session.get("safety_check")
    vision = session.get("vision_result")

    tumor_features = None
    tumor_regions = []

    if report and report.tumor_detected and report.tumor_features:
        f = report.tumor_features
        tumor_features = TumorFeaturesResponse(
            tumor_area_mm2=f.tumor_area_mm2 or 0.0,
            max_diameter_mm=f.max_diameter_mm or 0.0,
            tumor_ratio=f.tumor_ratio or 0.0,
            location_description=f.location_description or "Unknown",
        )

    if vision:
        tumor_regions = [
            TumorRegionResponse(
                label=r.label,
                label_name=r.label_name,
                area_pixels=r.area_pixels,
                area_mm2=r.area_mm2 or 0.0,
                centroid=list(r.centroid),
            )
            for r in vision.tumor_regions
        ]

    safety_response = None
    if safety:
        safety_response = SafetyCheckResponse(
            passed=safety.passed,
            confidence_adequate=safety.confidence_adequate,
            requires_human_review=safety.requires_human_review,
            flags=safety.flags,
            warnings=safety.warnings,
            compliance_notes=safety.compliance_notes,
        )

    return AnalysisResponse(
        session_id=session.get("session_id", ""),
        patient_id=str(session.get("patient_id", "")),
        status=session.get("status", "unknown"),
        tumor_detected=report.tumor_detected if report else False,
        classification=report.classification if report else None,
        classification_confidence=(
            report.classification_confidence if report else 0.0
        ),
        who_grade=report.who_grade if report else None,
        tumor_features=tumor_features,
        tumor_regions=tumor_regions,
        clinical_summary=report.clinical_summary if report else "",
        recommendations=report.recommendations if report else [],
        safety_check=safety_response,
        review_status=session.get("review_status"),
        disclaimer=ctrl.get_disclaimer(),
        errors=session.get("errors", []),
    )


@router.post("/analyze/upload", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_patient_upload(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    clinical_notes: Optional[str] = Form(None)
):
    """
    Analyze a patient's MRI image via direct upload.
    Supports JPG and PNG for BRISC.
    """
    valid_exts = ('.jpg', '.jpeg', '.png')
    if not (file.filename and file.filename.lower().endswith(valid_exts)):
        raise HTTPException(status_code=400, detail="Seuls les fichiers de type JPG ou PNG sont acceptés.")

    ctrl = get_controller()
    
    if patient_id:
        patient = ctrl.patient_store.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=400, detail=f"Patient ID {patient_id} not found.")
    
    # Create temporary directory to store the uploaded image
    temp_dir = tempfile.mkdtemp(prefix="agent_upload_")
    temp_dir_path = Path(temp_dir)
    
    temp_file_path = temp_dir_path / (file.filename or "image.jpg")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        session = ctrl.analyze_image_file(
            temp_file_path,
            patient_id=patient_id,
            clinical_notes=clinical_notes,
        )
    except Exception as e:
        logger.error(f"Analysis loop uploaded file failed: {e}")
        # Clean up on failure
        shutil.rmtree(temp_dir_path, ignore_errors=True)
        raise HTTPException(500, f"Analysis failed: {e}")
        
    shutil.rmtree(temp_dir_path, ignore_errors=True)

    # Build response
    report = session.get("diagnostic_report")
    safety = session.get("safety_check")
    vision = session.get("vision_result")

    tumor_features = None
    tumor_regions = []

    if report and report.tumor_detected and report.tumor_features:
        f = report.tumor_features
        tumor_features = TumorFeaturesResponse(
            tumor_area_mm2=f.tumor_area_mm2 or 0.0,
            max_diameter_mm=f.max_diameter_mm or 0.0,
            tumor_ratio=f.tumor_ratio or 0.0,
            location_description=f.location_description or "Unknown",
        )

    if vision:
        tumor_regions = [
            TumorRegionResponse(
                label=r.label,
                label_name=r.label_name,
                area_pixels=r.area_pixels,
                area_mm2=r.area_mm2 or 0.0,
                centroid=list(r.centroid),
            )
            for r in vision.tumor_regions
        ]

    safety_response = None
    if safety:
        safety_response = SafetyCheckResponse(
            passed=safety.passed,
            confidence_adequate=safety.confidence_adequate,
            requires_human_review=safety.requires_human_review,
            flags=safety.flags,
            warnings=safety.warnings,
            compliance_notes=safety.compliance_notes,
        )

    return AnalysisResponse(
        session_id=session.get("session_id", ""),
        patient_id=str(session.get("patient_id", "")),
        status=session.get("status", "unknown"),
        tumor_detected=report.tumor_detected if report else False,
        classification=report.classification if report else None,
        classification_confidence=(
            report.classification_confidence if report else 0.0
        ),
        who_grade=report.who_grade if report else None,
        tumor_features=tumor_features,
        tumor_regions=tumor_regions,
        clinical_summary=report.clinical_summary if report else "",
        recommendations=report.recommendations if report else [],
        safety_check=safety_response,
        review_status=session.get("review_status"),
        disclaimer=ctrl.get_disclaimer(),
        errors=session.get("errors", []),
    )



@router.post("/chat", response_model=ChatResponse, tags=["Dialogue"])
def chat(request: ChatRequest):
    """Chat with the Dialogue Agent about analysis results."""
    ctrl = get_controller()

    try:
        response = ctrl.chat(request.query, request.session_id)
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(500, f"Chat failed: {e}")

    return ChatResponse(
        response=response,
        session_id=request.session_id,
        disclaimer=ctrl.get_disclaimer(),
    )


@router.post("/chat/stream", tags=["Dialogue"])
def chat_stream(request: ChatRequest):
    """
    Stream a chat response with the Dialogue Agent.
    """
    ctrl = get_controller()

    def generate_tokens():
        try:
            for token in ctrl.chat_stream(request.query, request.session_id):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n[Streaming Error: {e}]"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream"
    )


@router.get(
    "/overlay/{session_id}",
    tags=["Visualization"],
    responses={200: {"content": {"image/png": {}}}},
)
def get_overlay(session_id: str):

    """Get 2D summary (Image, GT if available, AI Prediction)."""
    return get_summary(session_id)


@router.get(
    "/summary/{session_id}",
    tags=["Visualization"],
    responses={200: {"content": {"image/png": {}}}},
)
def get_summary(session_id: str):
    """Get 2D summary grid for BRISC sessions."""
    ctrl = get_controller()
    session = ctrl.get_session(session_id)

    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")

    vision_result = session.get("vision_result")
    if vision_result is None:
        raise HTTPException(400, "No results found")

    viz = get_visualizer()
    try:
        # Check if ground truth mask exists in the session/vision_result
        gt_mask = getattr(vision_result, 'ground_truth_mask', None)
        
        png = viz.generate_summary(
            vision_result.preprocessed_image,
            seg_mask=gt_mask,
            pred_mask=vision_result.segmentation_mask,
        )
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        raise HTTPException(500, str(e))

    return Response(content=png, media_type="image/png")



@router.get(
    "/report/{session_id}",
    tags=["Analysis"],
)
async def get_report(session_id: str):
    """Retrieve the full diagnostic report for an analysis session."""
    ctrl = get_controller()
    session = ctrl.get_session(session_id)

    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")

    report = session.get("diagnostic_report")
    if report is None:
        raise HTTPException(400, "No report available for this session")

    safety_check = session.get("safety_check")
    return {
        "report": report.to_dict(),
        "safety_check": {
            "passed": safety_check.passed,
            "requires_human_review": safety_check.requires_human_review,
            "flags": safety_check.flags,
            "warnings": safety_check.warnings,
        } if safety_check else None,
        "disclaimer": ctrl.get_disclaimer(),
    }


@router.get(
    "/sessions",
    response_model=list[SessionListItem],
    tags=["System"],
)
async def list_sessions():
    """List all analysis sessions."""
    ctrl = get_controller()
    sessions = ctrl.list_sessions()
    return [SessionListItem(**s) for s in sessions]


@router.post(
    "/review/{session_id}",
    response_model=ReviewResponse,
    tags=["Review"],
)
async def review_session(session_id: str, request: ReviewRequest):
    """
    Review an analysis session — approve or reject the AI-generated report.

    This endpoint enforces the mandatory human review gate required by
    EU AI Act Article 14. A radiologist must approve or reject every
    tumor-positive report before it can be considered finalized.

    - **approve**: Marks the report as clinically validated.
    - **reject**: Flags the report with the given reason and prevents finalization.
    """
    ctrl = get_controller()

    if request.action not in ("approve", "reject"):
        raise HTTPException(400, "Action must be 'approve' or 'reject'")

    if request.action == "reject" and not request.reason:
        raise HTTPException(400, "Reason is required when rejecting a report")

    try:
        if request.action == "approve":
            session = ctrl.approve_session(session_id, request.reviewer_name)
            message = f"Report approved by {request.reviewer_name}"
        else:
            session = ctrl.reject_session(
                session_id, request.reviewer_name, request.reason
            )
            message = (
                f"Report rejected by {request.reviewer_name}: {request.reason}"
            )
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(409, str(e))

    return ReviewResponse(
        session_id=session_id,
        review_status=session["review_status"],
        reviewed_by=session["reviewed_by"],
        reviewed_at=session["reviewed_at"],
        message=message,
    )


# ── Knowledge Base / Document endpoints ───────────────────────────────

def get_rag_system():
    """Get the RAG system instance from the graph module."""
    rag_system = graph.rag_system
    if rag_system is None or not rag_system.available:
        raise HTTPException(
            503, "RAG knowledge base is not available"
        )
    return rag_system


@router.get(
    "/knowledge/status",
    response_model=KnowledgeBaseStatus,
    tags=["Knowledge Base"],
)
async def knowledge_status():
    """Get the status of the RAG knowledge base."""
    try:
        rag_system = graph.rag_system
        if rag_system is None:
            return KnowledgeBaseStatus(
                available=False,
                total_chunks=0,
                uploaded_documents=0,
                system_knowledge_indexed=False,
            )
        return KnowledgeBaseStatus(
            available=rag_system.available,
            total_chunks=rag_system.get_collection_count(),
            uploaded_documents=len(rag_system.document_registry),
            system_knowledge_indexed=rag_system.is_source_indexed(
                "system:"
            ),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to get knowledge base status: {e}")


@router.post(
    "/knowledge/upload",
    response_model=DocumentUploadResponse,
    tags=["Knowledge Base"],
)
async def upload_document(
    file: UploadFile = File(...),
    uploaded_by: str = Query(
        ..., description="Name or ID of the uploading doctor"
    ),
):
    """
    Upload a clinical document to the RAG knowledge base.

    Supported formats: .txt, .md, .pdf

    The document will be:
    1. Saved to the uploads directory
    2. Parsed and split into semantic chunks
    3. Embedded and indexed into the Qdrant vector store
    4. Available for retrieval during report generation

    Doctors can upload clinical guidelines, research papers, case studies,
    or any relevant medical literature to improve report quality.
    """
    rag = get_rag_system()

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    # 10 MB limit
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10 MB)")

    try:
        record = rag.upload_document(
            file_content=content,
            filename=file.filename or "unnamed",
            uploaded_by=uploaded_by,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        uploaded_by=record.uploaded_by,
        uploaded_at=record.uploaded_at,
        file_type=record.file_type,
        chunk_count=record.chunk_count,
        message=(
            f"Document '{record.filename}' uploaded and indexed "
            f"({record.chunk_count} chunks)"
        ),
    )


@router.get(
    "/knowledge/documents",
    response_model=list[DocumentListItem],
    tags=["Knowledge Base"],
)
async def list_documents():
    """List all uploaded documents in the knowledge base."""
    rag = get_rag_system()
    documents = rag.list_documents()
    return [
        DocumentListItem(
            document_id=doc.document_id,
            filename=doc.filename,
            uploaded_by=doc.uploaded_by,
            uploaded_at=doc.uploaded_at,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
        )
        for doc in documents
    ]


@router.delete(
    "/knowledge/documents/{document_id}",
    tags=["Knowledge Base"],
)
async def delete_document(document_id: str):
    """
    Delete an uploaded document from the knowledge base.

    This removes the document file, its indexed chunks from Qdrant,
    and its registry entry. System knowledge cannot be deleted.
    """
    rag = get_rag_system()

    deleted = rag.delete_document(document_id)
    if not deleted:
        raise HTTPException(404, f"Document {document_id} not found")

    return {
        "document_id": document_id,
        "message": "Document deleted successfully",
    }


# ── Patient History endpoints ───────────────────────────────────────

from datetime import datetime

# ... (imports)

@router.post(
    "/patients",
    response_model=PatientResponse,
    tags=["Patient History"],
    status_code=201,
)
async def create_patient(request: PatientCreateRequest):
    """Register a new patient in the system."""
    ctrl = get_controller()
    
    dob = None
    if request.date_naissance:
        try:
            dob = datetime.strptime(request.date_naissance, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format for date_naissance. Use YYYY-MM-DD")

    try:
        patient = ctrl.patient_store.create_patient(
            prenom=request.prenom,
            nom=request.nom,
            date_naissance=dob,
            genre=request.genre,
            tel=request.tel,
            poids=request.poids,
            taille=request.taille,
            FC=request.FC,
            glycemie=request.glycemie,
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to create patient: {e}")

    return build_patient_response(patient)


@router.get(
    "/patients",
    response_model=list[PatientListItem],
    tags=["Patient History"],
)
async def list_patients():
    """List all registered patients."""
    ctrl = get_controller()
    patients = ctrl.patient_store.list_patients()
    return [
        PatientListItem(
            id=p.id,
            nom=p.nom,
            prenom=p.prenom,
            genre=p.genre,
            tel=p.tel,
            age=calculate_age(p.date_naissance),
            poids=p.poids,
            taille=p.taille,
            FC=p.FC,
            glycemie=p.glycemie,
            scan_count=len(p.scans),
            created_at=p.created_at.isoformat() if p.created_at else "",
        )
        for p in patients
    ]


@router.get(
    "/patients/{id}",
    response_model=PatientResponse,
    tags=["Patient History"],
)
async def get_patient(id: int):
    """Get a patient's full profile with scan history."""
    ctrl = get_controller()
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise HTTPException(404, f"Patient {id} not found")
    return build_patient_response(patient)


@router.put(
    "/patients/{id}",
    response_model=PatientResponse,
    tags=["Patient History"],
)
async def update_patient(id: int, request: PatientUpdateRequest):
    """Update a patient's information."""
    ctrl = get_controller()
    
    dob = None
    if request.date_naissance:
        try:
            dob = datetime.strptime(request.date_naissance, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format for date_naissance. Use YYYY-MM-DD")

    patient = ctrl.patient_store.update_patient(
        id=id,
        prenom=request.prenom,
        nom=request.nom,
        date_naissance=dob,
        genre=request.genre,
        tel=request.tel,
        poids=request.poids,
        taille=request.taille,
        FC=request.FC,
        glycemie=request.glycemie,
    )
    if patient is None:
        raise HTTPException(404, f"Patient {id} not found")
    return build_patient_response(patient)


@router.delete(
    "/patients/{id}",
    tags=["Patient History"],
)
async def delete_patient(id: int):
    """Delete a patient and all linked scan records."""
    ctrl = get_controller()
    deleted = ctrl.patient_store.delete_patient(id)
    if not deleted:
        raise HTTPException(404, f"Patient {id} not found")
    return {
        "id": id,
        "message": "Patient deleted successfully",
    }


@router.post(
    "/patients/{id}/link/{session_id}",
    tags=["Patient History"],
)
async def link_session_to_patient(id: int, session_id: str):
    """
    Manually link an existing analysis session to a patient.
    """
    ctrl = get_controller()

    # Verify patient exists
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise HTTPException(404, f"Patient {id} not found")

    # Verify session exists
    session = ctrl.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")

    report = session.get("diagnostic_report")
    features = report.tumor_features if report else None

    record = ctrl.patient_store.link_scan(
        patient_id=id,
        session_id=session_id,
        classification=report.classification if report else None,
        confidence=(
            report.classification_confidence if report else None
        ),
        tumor_area_mm2=(
            features.tumor_area_mm2 if features else None
        ),
        max_diameter_mm=(
            features.max_diameter_mm if features else None
        ),
        tumor_location=(
            features.location_description if features else None
        ),
        clinical_summary=(
            report.clinical_summary if report else None
        ),
        review_status=session.get("review_status", "pending_review"),
    )

    return {
        "patient_id": id,
        "session_id": session_id,
        "message": f"Session linked to patient {id}",
    }


@router.get(
    "/patients/{id}/history",
    tags=["Patient History"],
)
async def get_patient_history_text(id: int):
    """
    Get the formatted patient history text.
    """
    ctrl = get_controller()
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise HTTPException(404, f"Patient {id} not found")

    history_text = ctrl.patient_store.format_history_for_prompt(id)
    return {
        "patient_id": id,
        "history_context": history_text,
    }


# ── Helpers ───────────────────────────────────────────────────────

def calculate_age(dob: Optional[datetime]) -> Optional[int]:
    """Calculate age from date of birth."""
    if not dob:
        return None
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def build_patient_response(patient) -> PatientResponse:
    """Build a PatientResponse from a Patient ORM object."""
    return PatientResponse(
        id=patient.id,
        nom=patient.nom,
        prenom=patient.prenom,
        date_naissance=patient.date_naissance.strftime("%Y-%m-%d") if patient.date_naissance else None,
        genre=patient.genre,
        tel=patient.tel,
        poids=patient.poids,
        taille=patient.taille,
        FC=patient.FC,
        glycemie=patient.glycemie,
        scan_count=len(patient.scans),
        scans=[
            ScanHistoryItem(
                session_id=s.session_id,
                scan_date=s.scan_date.isoformat() if s.scan_date else "",
                classification=s.classification,
                confidence=s.confidence,
                tumor_area_mm2=s.tumor_area_mm2,
                max_diameter_mm=s.max_diameter_mm,
                tumor_location=s.tumor_location,
                review_status=s.review_status,
                reviewed_by=s.reviewed_by,
            )
            for s in patient.scans
        ],
        created_at=(
            patient.created_at.isoformat() if patient.created_at else ""
        ),
        updated_at=(
            patient.updated_at.isoformat() if patient.updated_at else None
        ),
    )


# ── Médecin Store singleton ─────────────────────────────────────────

from data.medecin_store import MedecinConsultationStore

_medecin_store: Optional[MedecinConsultationStore] = None


def get_medecin_store() -> MedecinConsultationStore:
    global _medecin_store
    if _medecin_store is None:
        _medecin_store = MedecinConsultationStore()
    return _medecin_store


# ── Médecin endpoints ─────────────────────────────────────────────

@router.post(
    "/medecins",
    response_model=MedecinResponse,
    tags=["Médecins"],
    status_code=201,
)
async def create_medecin(request: MedecinCreateRequest):
    """Register a new doctor in the system."""
    store = get_medecin_store()
    try:
        medecin = store.create_medecin(
            nom=request.nom,
            prenom=request.prenom,
            specialite=request.specialite,
            tel=request.tel,
            email=request.email,
            departement=request.departement,
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to create doctor: {e}")

    return build_medecin_response(medecin)


@router.get(
    "/medecins",
    response_model=list[MedecinListItem],
    tags=["Médecins"],
)
async def list_medecins():
    """List all registered doctors."""
    store = get_medecin_store()
    medecins = store.list_medecins()
    return [
        MedecinListItem(
            id=m.id,
            nom=m.nom,
            prenom=m.prenom,
            specialite=m.specialite,
            tel=m.tel,
            email=m.email,
            departement=m.departement,
            consultation_count=len(m.consultations),
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in medecins
    ]


@router.get(
    "/medecins/{id}",
    response_model=MedecinResponse,
    tags=["Médecins"],
)
async def get_medecin(id: int):
    """Get a doctor's full profile with consultation history."""
    store = get_medecin_store()
    medecin = store.get_medecin(id)
    if medecin is None:
        raise HTTPException(404, f"Médecin {id} not found")
    return build_medecin_response(medecin)


@router.put(
    "/medecins/{id}",
    response_model=MedecinResponse,
    tags=["Médecins"],
)
async def update_medecin(id: int, request: MedecinUpdateRequest):
    """Update a doctor's information."""
    store = get_medecin_store()
    medecin = store.update_medecin(
        id=id,
        nom=request.nom,
        prenom=request.prenom,
        specialite=request.specialite,
        tel=request.tel,
        email=request.email,
        departement=request.departement,
    )
    if medecin is None:
        raise HTTPException(404, f"Médecin {id} not found")
    return build_medecin_response(medecin)


@router.delete(
    "/medecins/{id}",
    tags=["Médecins"],
)
async def delete_medecin(id: int):
    """Delete a doctor and all linked consultation records."""
    store = get_medecin_store()
    deleted = store.delete_medecin(id)
    if not deleted:
        raise HTTPException(404, f"Médecin {id} not found")
    return {
        "id": id,
        "message": "Médecin deleted successfully",
    }


def build_medecin_response(medecin) -> MedecinResponse:
    """Build a MedecinResponse from a Medecin ORM object."""
    return MedecinResponse(
        id=medecin.id,
        nom=medecin.nom,
        prenom=medecin.prenom,
        specialite=medecin.specialite,
        tel=medecin.tel,
        email=medecin.email,
        departement=medecin.departement,
        consultation_count=len(medecin.consultations),
        created_at=(
            medecin.created_at.isoformat() if medecin.created_at else ""
        ),
        updated_at=(
            medecin.updated_at.isoformat() if medecin.updated_at else None
        ),
    )


# ── Consultation endpoints ───────────────────────────────────────

@router.post(
    "/consultations",
    response_model=ConsultationResponse,
    tags=["Consultations"],
    status_code=201,
)
async def create_consultation(request: ConsultationCreateRequest):
    """Create a new consultation linking a patient and a doctor."""
    store = get_medecin_store()
    try:
        consultation = store.create_consultation(
            patient_id=request.patient_id,
            medecin_id=request.medecin_id,
            scan_record_id=request.scan_record_id,
            motif=request.motif,
            diagnostic=request.diagnostic,
            notes=request.notes,
            rapport_genere=request.rapport_genere,
            statut=request.statut or "en_cours",
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to create consultation: {e}")

    if consultation is None:
        raise HTTPException(400, "Patient or Médecin not found, or scan does not belong to this patient.")

    # Re-fetch with names
    from data.database import Patient, Medecin, Consultation, ScanRecord
    with store.session_factory() as session:
        c = session.get(Consultation, consultation.id)
        patient = session.get(Patient, consultation.patient_id)
        medecin = session.get(Medecin, consultation.medecin_id)
        session.expunge_all()

    return build_consultation_response(
        consultation,
        patient_nom=patient.nom if patient else None,
        patient_prenom=patient.prenom if patient else None,
        medecin_nom=medecin.nom if medecin else None,
        medecin_prenom=medecin.prenom if medecin else None,
        medecin_specialite=medecin.specialite if medecin else None,
    )


@router.get(
    "/consultations",
    response_model=list[ConsultationListItem],
    tags=["Consultations"],
)
async def list_consultations(
    patient_id: Optional[int] = Query(None),
    medecin_id: Optional[int] = Query(None),
):
    """List all consultations, optionally filtered by patient or doctor."""
    store = get_medecin_store()
    results = store.list_all_consultations_with_names()

    if patient_id is not None:
        results = [r for r in results if r["consultation"].patient_id == patient_id]
    if medecin_id is not None:
        results = [r for r in results if r["consultation"].medecin_id == medecin_id]

    return [
        ConsultationListItem(
            id=r["consultation"].id,
            patient_id=r["consultation"].patient_id,
            patient_nom=r["patient_nom"],
            patient_prenom=r["patient_prenom"],
            medecin_id=r["consultation"].medecin_id,
            medecin_nom=r["medecin_nom"],
            medecin_prenom=r["medecin_prenom"],
            scan_record_id=r["consultation"].scan_record_id,
            session_id=r["session_id"],
            date_consultation=(
                r["consultation"].date_consultation.isoformat()
                if r["consultation"].date_consultation else ""
            ),
            motif=r["consultation"].motif,
            diagnostic=r["consultation"].diagnostic,
            statut=r["consultation"].statut,
            created_at=(
                r["consultation"].created_at.isoformat()
                if r["consultation"].created_at else ""
            ),
        )
        for r in results
    ]


@router.get(
    "/consultations/{id}",
    response_model=ConsultationResponse,
    tags=["Consultations"],
)
async def get_consultation(id: int):
    """Get a consultation's full details."""
    store = get_medecin_store()
    consultation = store.get_consultation(id)
    if consultation is None:
        raise HTTPException(404, f"Consultation {id} not found")

    with store.session_factory() as session:
        from data.database import Patient, Medecin, Consultation, ScanRecord
        patient = session.get(Patient, consultation.patient_id)
        medecin = session.get(Medecin, consultation.medecin_id)
        scan_record = session.get(ScanRecord, consultation.scan_record_id) if consultation.scan_record_id else None
        session.expunge_all()

    return build_consultation_response(
        consultation,
        patient_nom=patient.nom if patient else None,
        patient_prenom=patient.prenom if patient else None,
        medecin_nom=medecin.nom if medecin else None,
        medecin_prenom=medecin.prenom if medecin else None,
        medecin_specialite=medecin.specialite if medecin else None,
        session_id=scan_record.session_id if scan_record else None,
    )


@router.put(
    "/consultations/{id}",
    response_model=ConsultationResponse,
    tags=["Consultations"],
)
async def update_consultation(id: int, request: ConsultationUpdateRequest):
    """Update a consultation's information."""
    store = get_medecin_store()
    consultation = store.update_consultation(
        id=id,
        medecin_id=request.medecin_id,
        motif=request.motif,
        diagnostic=request.diagnostic,
        notes=request.notes,
        rapport_genere=request.rapport_genere,
        statut=request.statut,
    )
    if consultation is None:
        raise HTTPException(404, f"Consultation {id} not found")

    with store.session_factory() as session:
        from data.database import Patient, Medecin, Consultation, ScanRecord
        patient = session.get(Patient, consultation.patient_id)
        medecin = session.get(Medecin, consultation.medecin_id)
        scan_record = session.get(ScanRecord, consultation.scan_record_id) if consultation.scan_record_id else None
        session.expunge_all()

    return build_consultation_response(
        consultation,
        patient_nom=patient.nom if patient else None,
        patient_prenom=patient.prenom if patient else None,
        medecin_nom=medecin.nom if medecin else None,
        medecin_prenom=medecin.prenom if medecin else None,
        medecin_specialite=medecin.specialite if medecin else None,
        session_id=scan_record.session_id if scan_record else None,
    )


@router.delete(
    "/consultations/{id}",
    tags=["Consultations"],
)
async def delete_consultation(id: int):
    """Delete a consultation record."""
    store = get_medecin_store()
    deleted = store.delete_consultation(id)
    if not deleted:
        raise HTTPException(404, f"Consultation {id} not found")
    return {
        "id": id,
        "message": "Consultation deleted successfully",
    }


def build_consultation_response(
    consultation,
    patient_nom=None,
    patient_prenom=None,
    medecin_nom=None,
    medecin_prenom=None,
    medecin_specialite=None,
    session_id=None,
) -> ConsultationResponse:
    """Build a ConsultationResponse from a Consultation ORM object."""
    return ConsultationResponse(
        id=consultation.id,
        patient_id=consultation.patient_id,
        patient_nom=patient_nom,
        patient_prenom=patient_prenom,
        medecin_id=consultation.medecin_id,
        medecin_nom=medecin_nom,
        medecin_prenom=medecin_prenom,
        medecin_specialite=medecin_specialite,
        scan_record_id=consultation.scan_record_id,
        session_id=session_id,
        date_consultation=(
            consultation.date_consultation.isoformat()
            if consultation.date_consultation else ""
        ),
        motif=consultation.motif,
        diagnostic=consultation.diagnostic,
        notes=consultation.notes,
        rapport_genere=consultation.rapport_genere,
        statut=consultation.statut,
        created_at=(
            consultation.created_at.isoformat()
            if consultation.created_at else ""
        ),
        updated_at=(
            consultation.updated_at.isoformat()
            if consultation.updated_at else None
        ),
    )
