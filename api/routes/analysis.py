"""
Analysis routes — MRI analysis, chat, review, visualization, and session endpoints.

Uses the shared build_analysis_response() helper to eliminate the ~160-line
duplication that previously existed between /analyze and /analyze/upload.

ML inference is offloaded to a thread pool executor to avoid blocking the
FastAPI event loop, enabling concurrent request handling.
"""
from __future__ import annotations
import asyncio
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
    SessionListItem,
)
from agents.controller import ControllerAgent
from agents import graph
from interpretability.visualizations import BRISCVisualizer
from api.routes.response_builder import build_analysis_response
from api.errors import NotFoundError, ValidationError, ConflictError, ServiceUnavailableError

router = APIRouter()

# Singleton controller — initialized at startup
controller_instance: Optional[ControllerAgent] = None
visualizer_instance: Optional[BRISCVisualizer] = None

# Thread pool for running blocking ML inference
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml-inference")


def get_controller() -> ControllerAgent:
    global controller_instance
    if controller_instance is None:
        raise ServiceUnavailableError("System not initialized. Wait for startup.")
    return controller_instance


def set_controller(controller: ControllerAgent) -> None:
    global controller_instance
    controller_instance = controller


def get_visualizer() -> BRISCVisualizer:
    global visualizer_instance
    if visualizer_instance is None:
        visualizer_instance = BRISCVisualizer()
    return visualizer_instance


# ── Health ──────────────────────────────────────────────────────────────

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


# ── Analysis (shared implementation) ────────────────────────────────────

def _run_analysis(
    image_path: Path,
    patient_id: Optional[int],
    clinical_notes: Optional[str],
) -> dict:
    """
    Core analysis logic shared by /analyze and /analyze/upload.

    Args:
        image_path: Path to the image file to analyze
        patient_id: Optional patient ID for history linking
        clinical_notes: Optional clinical notes

    Returns:
        Session state dict from the controller
    """
    ctrl = get_controller()
    return ctrl.analyze_image_file(
        image_path,
        patient_id=patient_id,
        clinical_notes=clinical_notes,
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_patient(request: AnalyzeRequest):
    """Analyze a patient's MRI image from a server-side path (BRISC 2D)."""
    ctrl = get_controller()
    patient_dir = Path(request.patient_dir)

    if not patient_dir.exists():
        raise NotFoundError(f"Patient directory not found: {patient_dir}")

    loop = asyncio.get_event_loop()
    try:
        if patient_dir.is_file():
            session = await loop.run_in_executor(
                _executor,
                lambda: ctrl.analyze_image_file(
                    patient_dir,
                    patient_id=request.patient_id,
                    clinical_notes=request.clinical_notes,
                ),
            )
        else:
            session = await loop.run_in_executor(
                _executor,
                lambda: ctrl.analyze_image_dir(
                    patient_dir,
                    patient_id=request.patient_id,
                    clinical_notes=request.clinical_notes,
                ),
            )
    except FileNotFoundError as e:
        raise NotFoundError(str(e))

    return build_analysis_response(session, ctrl)


@router.post("/analyze/upload", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_patient_upload(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    clinical_notes: Optional[str] = Form(None),
):
    """Analyze a patient's MRI image via direct upload. Supports JPG and PNG."""

    valid_exts = ('.jpg', '.jpeg', '.png')
    if not (file.filename and file.filename.lower().endswith(valid_exts)):
        raise ValidationError("Seuls les fichiers de type JPG ou PNG sont acceptés.")

    ctrl = get_controller()

    if patient_id:
        patient = ctrl.patient_store.get_patient(patient_id)
        if not patient:
            raise NotFoundError(f"Patient ID {patient_id} not found.")

    temp_dir = tempfile.mkdtemp(prefix="agent_upload_")
    temp_dir_path = Path(temp_dir)
    temp_file_path = temp_dir_path / (file.filename or "image.jpg")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(
            _executor,
            lambda: ctrl.analyze_image_file(
                temp_file_path,
                patient_id=patient_id,
                clinical_notes=clinical_notes,
            ),
        )
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Analysis of uploaded file failed: {e}")
        shutil.rmtree(temp_dir_path, ignore_errors=True)
        raise  # let the global handler return 500

    shutil.rmtree(temp_dir_path, ignore_errors=True)
    return build_analysis_response(session, ctrl)


# ── Chat ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Dialogue"])
async def chat(request: ChatRequest):
    """Chat with the Dialogue Agent about analysis results."""
    ctrl = get_controller()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        _executor,
        lambda: ctrl.chat(request.query, request.session_id),
    )
    return ChatResponse(
        response=response,
        session_id=request.session_id,
        disclaimer=ctrl.get_disclaimer(),
    )


@router.post("/chat/stream", tags=["Dialogue"])
def chat_stream(request: ChatRequest):
    """Stream a chat response with the Dialogue Agent."""
    ctrl = get_controller()

    def generate_tokens():
        try:
            for token in ctrl.chat_stream(request.query, request.session_id):
                yield token
        except (RuntimeError, OSError) as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n[Streaming Error: {e}]"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
    )


# ── Visualization ───────────────────────────────────────────────────────

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
        raise NotFoundError(f"Session {session_id} not found")

    vision_result = session.get("vision_result")
    if vision_result is None:
        raise ValidationError("No results found")

    viz = get_visualizer()
    gt_mask = getattr(vision_result, 'ground_truth_mask', None)
    png = viz.generate_summary(
        vision_result.preprocessed_image,
        seg_mask=gt_mask,
        pred_mask=vision_result.segmentation_mask,
    )
    return Response(content=png, media_type="image/png")


# ── Report ──────────────────────────────────────────────────────────────

@router.get("/report/{session_id}", tags=["Analysis"])
async def get_report(session_id: str):
    """Retrieve the full diagnostic report for an analysis session."""
    ctrl = get_controller()
    session = ctrl.get_session(session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")

    report = session.get("diagnostic_report")
    if report is None:
        raise ValidationError("No report available for this session")

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


# ── Sessions ────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[SessionListItem], tags=["System"])
async def list_sessions():
    """List all analysis sessions."""
    ctrl = get_controller()
    sessions = ctrl.list_sessions()
    return [SessionListItem(**s) for s in sessions]


# ── Review ──────────────────────────────────────────────────────────────

@router.post("/review/{session_id}", response_model=ReviewResponse, tags=["Review"])
async def review_session(session_id: str, request: ReviewRequest):
    """
    Review an analysis session — approve or reject the AI-generated report.

    This endpoint enforces the mandatory human review gate required by
    EU AI Act Article 14. A radiologist must approve or reject every
    tumor-positive report before it can be considered finalized.
    """
    ctrl = get_controller()

    if request.action not in ("approve", "reject"):
        raise ValidationError("Action must be 'approve' or 'reject'")

    if request.action == "reject" and not request.reason:
        raise ValidationError("Reason is required when rejecting a report")

    try:
        if request.action == "approve":
            session = ctrl.approve_session(session_id, request.reviewer_name)
            message = f"Report approved by {request.reviewer_name}"
        else:
            session = ctrl.reject_session(
                session_id, request.reviewer_name, request.reason
            )
            message = f"Report rejected by {request.reviewer_name}: {request.reason}"
    except KeyError:
        raise NotFoundError(f"Session {session_id} not found")
    except ValueError as e:
        raise ConflictError(str(e))

    return ReviewResponse(
        session_id=session_id,
        review_status=session["review_status"],
        reviewed_by=session["reviewed_by"],
        reviewed_at=session["reviewed_at"],
        message=message,
    )
