"""
Shared database module — single engine/session factory for the entire application.

All stores (PatientStore, MedecinConsultationStore) share one engine and session
factory, avoiding duplicate connection pools and ensuring transactional consistency.
"""

from __future__ import annotations
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
from config.settings import get_settings

_engine = None
_session_factory = None


def get_engine():
    """Get or create the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.db.url
        if not db_url:
            raise RuntimeError(
                "MySQL DATABASE_URL could not be constructed. "
                "Ensure MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and "
                "MYSQL_DATABASE are set in .env"
            )
        logger.info("Creating shared MySQL engine")
        _engine = create_engine(db_url, echo=False, pool_pre_ping=True)
    return _engine


def get_session_factory():
    """Get or create the shared session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory


def init_database():
    """
    Initialize the database: create tables if they don't exist.
    Must be called once at application startup.
    """
    from data.database import Base
    engine = get_engine()
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        settings = get_settings()
        logger.error(f"Cannot initialize database: {e}")
        raise RuntimeError(
            f"Failed to connect to MySQL at {settings.db.host}:"
            f"{settings.db.port}. Check that MySQL is running and "
            f"the database '{settings.db.name}' exists. "
            f"Original error: {e}"
        ) from e
    logger.info("Database initialized (shared engine).")
