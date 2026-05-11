"""
FastAPI application for the Medical AI Agent.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.settings import get_settings
from agents.controller import ControllerAgent
from api.routes import router, set_controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — loads models at startup."""
    settings = get_settings()
    logger.info("Starting Medical AI Agent API...")

    controller = ControllerAgent(settings)
    controller.load_models() # Load all AI models at startup
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
            "classification, and clinical reporting from MRI scans. "
            "**RESEARCH USE ONLY — Not a certified medical device.**"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    return app


# For uvicorn: python -m uvicorn api.main:app --reload
app = create_app()
