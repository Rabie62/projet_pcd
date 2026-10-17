"""
Consultation CRUD routes.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from api.errors import NotFoundError, ValidationError
from api.schemas import (
    ConsultationCreateRequest,
    ConsultationListItem,
    ConsultationResponse,
    ConsultationUpdateRequest,
    PaginatedResponse,
)
from api.errors import NotFoundError, ValidationError
from data.database import Patient, Medecin, Consultation, ScanRecord

router = APIRouter()

store = None


def get_store():
    global store
    if store is None:
        from data.medecin_store import MedecinConsultationStore
        store = MedecinConsultationStore()
    return store


def to_response(consultation, patient_nom=None, patient_prenom=None,
                medecin_nom=None, medecin_prenom=None,
                medecin_specialite=None, session_id=None) -> ConsultationResponse:
    return ConsultationResponse(
        id=consultation.id, patient_id=consultation.patient_id,
        patient_nom=patient_nom, patient_prenom=patient_prenom,
        medecin_id=consultation.medecin_id, medecin_nom=medecin_nom,
        medecin_prenom=medecin_prenom, medecin_specialite=medecin_specialite,
        scan_record_id=consultation.scan_record_id, session_id=session_id,
        date_consultation=consultation.date_consultation.isoformat() if consultation.date_consultation else "",
        motif=consultation.motif, diagnostic=consultation.diagnostic,
        notes=consultation.notes, rapport_genere=consultation.rapport_genere,
        statut=consultation.statut,
        created_at=consultation.created_at.isoformat() if consultation.created_at else "",
        updated_at=consultation.updated_at.isoformat() if consultation.updated_at else None,
    )


def fetch_named(consultation):
    st = get_store()
    with st.session_factory() as session:
        patient = session.get(Patient, consultation.patient_id)
        medecin = session.get(Medecin, consultation.medecin_id)
        scan_record = session.get(ScanRecord, consultation.scan_record_id) if consultation.scan_record_id else None
        session.expunge_all()
    return to_response(
        consultation,
        patient_nom=patient.nom if patient else None,
        patient_prenom=patient.prenom if patient else None,
        medecin_nom=medecin.nom if medecin else None,
        medecin_prenom=medecin.prenom if medecin else None,
        medecin_specialite=medecin.specialite if medecin else None,
        session_id=scan_record.session_id if scan_record else None,
    )


@router.post("/consultations", response_model=ConsultationResponse, tags=["Consultations"], status_code=201)
async def create_consultation(request: ConsultationCreateRequest):
    """Create a new consultation linking a patient and a doctor."""
    st = get_store()
    try:
        consultation = st.create_consultation(
            patient_id=request.patient_id, medecin_id=request.medecin_id,
            scan_record_id=request.scan_record_id, motif=request.motif,
            diagnostic=request.diagnostic, notes=request.notes,
            rapport_genere=request.rapport_genere, statut=request.statut or "en_cours",
        )
    except (OSError, RuntimeError) as e:
        raise  # global handler
    if consultation is None:
        raise HTTPException(400, "Patient or Médecin not found, or scan does not belong to this patient.")
    return fetch_named(consultation)


@router.get("/consultations", response_model=PaginatedResponse[ConsultationListItem], tags=["Consultations"])
async def list_consultations(
    patient_id: int | None = Query(None),
    medecin_id: int | None = Query(None),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all consultations, optionally filtered by patient or doctor, with pagination."""
    st = get_store()
    results = st.list_all_named()
    if patient_id is not None:
        results = [r for r in results if r["consultation"].patient_id == patient_id]
    if medecin_id is not None:
        results = [r for r in results if r["consultation"].medecin_id == medecin_id]
    total = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = results[start:end]
    items = [
        ConsultationListItem(
            id=r["consultation"].id, patient_id=r["consultation"].patient_id,
            patient_nom=r["patient_nom"], patient_prenom=r["patient_prenom"],
            medecin_id=r["consultation"].medecin_id, medecin_nom=r["medecin_nom"],
            medecin_prenom=r["medecin_prenom"], scan_record_id=r["consultation"].scan_record_id,
            session_id=r["session_id"],
            date_consultation=r["consultation"].date_consultation.isoformat() if r["consultation"].date_consultation else "",
            motif=r["consultation"].motif, diagnostic=r["consultation"].diagnostic,
            statut=r["consultation"].statut,
            created_at=r["consultation"].created_at.isoformat() if r["consultation"].created_at else "",
        )
        for r in page_items
    ]
    return PaginatedResponse.create(items=items, total=total, page=page, page_size=page_size)


@router.get("/consultations/{id}", response_model=ConsultationResponse, tags=["Consultations"])
async def get_consultation(id: int):
    """Get a consultation's full details."""
    st = get_store()
    consultation = st.get_consultation(id)
    if consultation is None:
        raise NotFoundError(f"Consultation {id} not found")
    return fetch_named(consultation)


@router.put("/consultations/{id}", response_model=ConsultationResponse, tags=["Consultations"])
async def update_consultation(id: int, request: ConsultationUpdateRequest):
    """Update a consultation's information."""
    st = get_store()
    consultation = st.update_consultation(
        id=id, medecin_id=request.medecin_id, motif=request.motif,
        diagnostic=request.diagnostic, notes=request.notes,
        rapport_genere=request.rapport_genere, statut=request.statut,
    )
    if consultation is None:
        raise NotFoundError(f"Consultation {id} not found")
    return fetch_named(consultation)


@router.delete("/consultations/{id}", tags=["Consultations"])
async def delete_consultation(id: int):
    """Delete a consultation record."""
    st = get_store()
    deleted = st.delete_consultation(id)
    if not deleted:
        raise NotFoundError(f"Consultation {id} not found")
    return {"id": id, "message": "Consultation deleted successfully"}
