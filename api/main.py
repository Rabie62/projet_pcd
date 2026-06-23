"""
FastAPI application for the Medical AI Agent.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from data.db import init_database
from config.settings import get_settings
from agents.controller import ControllerAgent
from agents.graph import AgentRegistry, set_registry
from api.routes import router, set_controller
from api.errors import AppError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — loads models at startup."""
    settings = get_settings()
    logger.info("Starting Medical AI Agent API...")

    # Initialize the database (create tables if needed)
    init_database()

    # Initialize the agent registry first (creates RAG, agents, etc.)
    registry = AgentRegistry.create(settings)
    set_registry(registry)

    # Create controller — it will use the existing registry
    controller = ControllerAgent(settings)
    controller.load_models()  # Load all AI models at startup

    set_controller(controller)

    logger.info("API ready. Models loaded.")
    yield
    logger.info("Shutting down Medical AI Agent API.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Medical AI Agent — Brain Tumor Analysis",
        description=(
            "Multi-agent AI system for brain tumor detection, segmentation, "
            "classification, and clinical reporting from MRI scans.\n\n"
            "**RESEARCH USE ONLY — Not a certified medical device.**\n\n"
            "## API Versioning\n\n"
            "Current version: **v1**\n\n"
            "All endpoints are prefixed with `/api/v1`. "
            "When breaking changes are introduced, a new version prefix "
            "(`/api/v2`) will be added while maintaining backward compatibility "
            "with the previous version for at least 6 months."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ─────────────────────────────────────

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        """Handle all custom application errors with consistent JSON format."""
        logger.warning(f"AppError [{exc.error_code}]: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        """Catch-all for unexpected errors — log full traceback, return generic message."""
        logger.exception(f"Unhandled error at {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
                "details": {},
            },
        )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    return app


# For uvicorn: python -m uvicorn api.main:app --reload
app = create_app()
