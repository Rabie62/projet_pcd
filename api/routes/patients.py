"""
Patient CRUD routes.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from api.schemas import (
    PatientCreateRequest,
    PatientListItem,
    PatientResponse,
    PatientUpdateRequest,
    ScanHistoryItem,
    PaginatedResponse,
)
from api.errors import NotFoundError, ValidationError
from agents.controller import ControllerAgent

router = APIRouter()


def _get_controller() -> ControllerAgent:
    from api.routes.analysis import get_controller
    return get_controller()


def _calculate_age(dob: Optional[datetime]) -> Optional[int]:
    if not dob:
        return None
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _build_patient_response(patient) -> PatientResponse:
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
        created_at=patient.created_at.isoformat() if patient.created_at else "",
        updated_at=patient.updated_at.isoformat() if patient.updated_at else None,
    )


@router.post("/patients", response_model=PatientResponse, tags=["Patient History"], status_code=201)
async def create_patient(request: PatientCreateRequest):
    """Register a new patient in the system."""
    ctrl = _get_controller()
    dob = None
    if request.date_naissance:
        try:
            dob = datetime.strptime(request.date_naissance, "%Y-%m-%d")
        except ValueError:
            raise ValidationError("Invalid date format for date_naissance. Use YYYY-MM-DD")
    try:
        patient = ctrl.patient_store.create_patient(
            prenom=request.prenom, nom=request.nom,
            date_naissance=dob, genre=request.genre, tel=request.tel,
            poids=request.poids, taille=request.taille,
            FC=request.FC, glycemie=request.glycemie,
        )
    except (OSError, RuntimeError) as e:
        raise  # global handler will return 500
    return _build_patient_response(patient)


@router.get("/patients", response_model=PaginatedResponse[PatientListItem], tags=["Patient History"])
async def list_patients(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List registered patients with pagination."""
    ctrl = _get_controller()
    all_patients = ctrl.patient_store.list_patients()
    total = len(all_patients)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_patients[start:end]
    items = [
        PatientListItem(
            id=p.id, nom=p.nom, prenom=p.prenom, genre=p.genre, tel=p.tel,
            age=_calculate_age(p.date_naissance), poids=p.poids, taille=p.taille,
            FC=p.FC, glycemie=p.glycemie, scan_count=len(p.scans),
            created_at=p.created_at.isoformat() if p.created_at else "",
        )
        for p in page_items
    ]
    return PaginatedResponse.create(items=items, total=total, page=page, page_size=page_size)


@router.get("/patients/{id}", response_model=PatientResponse, tags=["Patient History"])
async def get_patient(id: int):
    """Get a patient's full profile with scan history."""
    ctrl = _get_controller()
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise NotFoundError(f"Patient {id} not found")
    return _build_patient_response(patient)


@router.put("/patients/{id}", response_model=PatientResponse, tags=["Patient History"])
async def update_patient(id: int, request: PatientUpdateRequest):
    """Update a patient's information."""
    ctrl = _get_controller()
    dob = None
    if request.date_naissance:
        try:
            dob = datetime.strptime(request.date_naissance, "%Y-%m-%d")
        except ValueError:
            raise ValidationError("Invalid date format for date_naissance. Use YYYY-MM-DD")
    patient = ctrl.patient_store.update_patient(
        id=id, prenom=request.prenom, nom=request.nom,
        date_naissance=dob, genre=request.genre, tel=request.tel,
        poids=request.poids, taille=request.taille,
        FC=request.FC, glycemie=request.glycemie,
    )
    if patient is None:
        raise NotFoundError(f"Patient {id} not found")
    return _build_patient_response(patient)


@router.delete("/patients/{id}", tags=["Patient History"])
async def delete_patient(id: int):
    """Delete a patient and all linked scan records."""
    ctrl = _get_controller()
    deleted = ctrl.patient_store.delete_patient(id)
    if not deleted:
        raise NotFoundError(f"Patient {id} not found")
    return {"id": id, "message": "Patient deleted successfully"}


@router.post("/patients/{id}/link/{session_id}", tags=["Patient History"])
async def link_session_to_patient(id: int, session_id: str):
    """Manually link an existing analysis session to a patient."""
    ctrl = _get_controller()
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise NotFoundError(f"Patient {id} not found")
    session = ctrl.get_session(session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    report = session.get("diagnostic_report")
    features = report.tumor_features if report else None
    ctrl.patient_store.link_scan(
        patient_id=id, session_id=session_id,
        classification=report.classification if report else None,
        confidence=report.classification_confidence if report else None,
        tumor_area_mm2=features.tumor_area_mm2 if features else None,
        max_diameter_mm=features.max_diameter_mm if features else None,
        tumor_location=features.location_description if features else None,
        clinical_summary=report.clinical_summary if report else None,
        review_status=session.get("review_status", "pending_review"),
    )
    return {"patient_id": id, "session_id": session_id, "message": f"Session linked to patient {id}"}


@router.get("/patients/{id}/history", tags=["Patient History"])
async def get_patient_history_text(id: int):
    """Get the formatted patient history text."""
    ctrl = _get_controller()
    patient = ctrl.patient_store.get_patient(id)
    if patient is None:
        raise NotFoundError(f"Patient {id} not found")
    history_text = ctrl.patient_store.format_history_for_prompt(id)
    return {"patient_id": id, "history_context": history_text}
