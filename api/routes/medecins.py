"""
Doctor (Médecin) CRUD routes.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from api.errors import NotFoundError, ValidationError
from loguru import logger
from api.schemas import (
    MedecinCreateRequest,
    MedecinListItem,
    MedecinResponse,
    MedecinUpdateRequest,
    PaginatedResponse,
)
from api.errors import NotFoundError, ValidationError
from agents.controller import ControllerAgent

router = APIRouter()

_medecin_store = None


def _get_medecin_store():
    global _medecin_store
    if _medecin_store is None:
        from data.medecin_store import MedecinConsultationStore
        _medecin_store = MedecinConsultationStore()
    return _medecin_store


def _build_medecin_response(medecin) -> MedecinResponse:
    return MedecinResponse(
        id=medecin.id, nom=medecin.nom, prenom=medecin.prenom,
        specialite=medecin.specialite, tel=medecin.tel, email=medecin.email,
        departement=medecin.departement, username=medecin.username,
        consultation_count=len(medecin.consultations),
        created_at=medecin.created_at.isoformat() if medecin.created_at else "",
        updated_at=medecin.updated_at.isoformat() if medecin.updated_at else None,
    )


@router.post("/medecins", response_model=MedecinResponse, tags=["Médecins"], status_code=201)
async def create_medecin(request: MedecinCreateRequest):
    """Register a new doctor in the system."""
    store = _get_medecin_store()
    try:
        medecin = store.create_medecin(
            nom=request.nom, prenom=request.prenom, specialite=request.specialite,
            tel=request.tel, email=request.email, departement=request.departement,
            username=request.username, password=request.password,
        )
    except (OSError, RuntimeError) as e:
        raise  # global handler
    return _build_medecin_response(medecin)


@router.get("/medecins", response_model=PaginatedResponse[MedecinListItem], tags=["Médecins"])
async def list_medecins(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List all registered doctors with pagination."""
    store = _get_medecin_store()
    all_medecins = store.list_medecins()
    total = len(all_medecins)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_medecins[start:end]
    items = [
        MedecinListItem(
            id=m.id, nom=m.nom, prenom=m.prenom, specialite=m.specialite,
            tel=m.tel, email=m.email, departement=m.departement,
            username=m.username, consultation_count=len(m.consultations),
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in page_items
    ]
    return PaginatedResponse.create(items=items, total=total, page=page, page_size=page_size)


@router.get("/medecins/{id}", response_model=MedecinResponse, tags=["Médecins"])
async def get_medecin(id: int):
    """Get a doctor's full profile with consultation history."""
    store = _get_medecin_store()
    medecin = store.get_medecin(id)
    if medecin is None:
        raise NotFoundError(f"Médecin {id} not found")
    return _build_medecin_response(medecin)


@router.put("/medecins/{id}", response_model=MedecinResponse, tags=["Médecins"])
async def update_medecin(id: int, request: MedecinUpdateRequest):
    """Update a doctor's information."""
    store = _get_medecin_store()
    medecin = store.update_medecin(
        id=id, nom=request.nom, prenom=request.prenom, specialite=request.specialite,
        tel=request.tel, email=request.email, departement=request.departement,
        username=request.username, password=request.password,
    )
    if medecin is None:
        raise NotFoundError(f"Médecin {id} not found")
    return _build_medecin_response(medecin)


@router.delete("/medecins/{id}", tags=["Médecins"])
async def delete_medecin(id: int):
    """Delete a doctor and all linked consultation records."""
    store = _get_medecin_store()
    deleted = store.delete_medecin(id)
    if not deleted:
        raise NotFoundError(f"Médecin {id} not found")
    return {"id": id, "message": "Médecin deleted successfully"}
