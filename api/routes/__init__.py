"""
API routes package.

Split from the monolithic routes.py into focused submodules:
  - analysis    : MRI analysis endpoints (analyze, upload, report, sessions, review, chat, visualization)
  - patients    : Patient CRUD endpoints
  - medecins    : Doctor CRUD endpoints
  - consultations: Consultation CRUD endpoints
  - knowledge   : RAG knowledge base endpoints
  - response_builder : Shared response construction helpers

All routes are prefixed with /api/v1 at the app level.
"""
from fastapi import APIRouter
from api.routes.analysis import router as analysis_router, set_controller
from api.routes.patients import router as patients_router
from api.routes.medecins import router as medecins_router
from api.routes.consultations import router as consultations_router
from api.routes.knowledge import router as knowledge_router

router = APIRouter()

# Include all route submodules with OpenAPI tags for documentation grouping
router.include_router(analysis_router, tags=["v1"])
router.include_router(patients_router, tags=["v1"])
router.include_router(medecins_router, tags=["v1"])
router.include_router(consultations_router, tags=["v1"])
router.include_router(knowledge_router, tags=["v1"])

__all__ = ["router", "set_controller"]
