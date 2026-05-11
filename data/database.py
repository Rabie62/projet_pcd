"""
SQLAlchemy database models and engine configuration.

Defines the Patient and ScanRecord ORM models backed by MySQL.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)
from loguru import logger
from config.settings import get_settings


# ── Base class ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────

class Patient(Base):
    """Patient profile with demographics and medical history."""

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(255), nullable=False)
    prenom = Column(String(255), nullable=False)
    date_naissance = Column(DateTime, nullable=True)
    genre = Column(String(10), nullable=True)  # M/F
    tel = Column(String(20), nullable=True)
    poids = Column(Float, nullable=True)       # kg
    taille = Column(Float, nullable=True)      # cm
    FC = Column(Integer, nullable=True)        # Heart rate
    glycemie = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    scans = relationship(
        "ScanRecord",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="ScanRecord.scan_date.desc()",
    )

    consultations = relationship(
        "Consultation",
        back_populates="patient",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Patient(id={self.id}, nom={self.nom!r}, prenom={self.prenom!r}, "
            f"scans={len(self.scans)})>"
        )


class ScanRecord(Base):
    """Record of a single MRI scan analysis linked to a patient."""

    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer, ForeignKey("patients.id"), nullable=False
    )
    session_id = Column(String(100), unique=True, nullable=False)
    scan_date = Column(DateTime, default=func.now())
    classification = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    tumor_area_mm2 = Column(Float, nullable=True)
    max_diameter_mm = Column(Float, nullable=True)
    tumor_location = Column(String(255), nullable=True)
    clinical_summary = Column(Text, nullable=True)
    review_status = Column(String(50), default="pending_review")
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    patient = relationship("Patient", back_populates="scans")

    def __repr__(self) -> str:
        return (
            f"<ScanRecord(id={self.id}, patient_id={self.patient_id}, "
            f"class={self.classification!r}, date={self.scan_date})>"
        )


class Medecin(Base):
    """Doctor profile with specialty and contact information."""

    __tablename__ = "medecins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(255), nullable=False)
    prenom = Column(String(255), nullable=False)
    specialite = Column(String(255), nullable=True)
    tel = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True)
    departement = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    consultations = relationship(
        "Consultation",
        back_populates="medecin",
        cascade="all, delete-orphan",
        order_by="Consultation.date_consultation.desc()",
    )

    def __repr__(self) -> str:
        return (
            f"<Medecin(id={self.id}, nom={self.nom!r}, prenom={self.prenom!r}, "
            f"specialite={self.specialite!r})>"
        )


class Consultation(Base):
    """Medical consultation record linking a patient, doctor, and optionally a scan."""

    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    medecin_id = Column(
        Integer, ForeignKey("medecins.id", ondelete="CASCADE"), nullable=False
    )
    scan_record_id = Column(
        Integer, ForeignKey("scan_records.id", ondelete="SET NULL"), nullable=True
    )
    date_consultation = Column(DateTime, default=func.now())
    motif = Column(String(500), nullable=True)
    diagnostic = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    rapport_genere = Column(Text, nullable=True)
    statut = Column(String(50), default="en_cours")  # en_cours, terminee, annulee
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    patient = relationship("Patient", back_populates="consultations")
    medecin = relationship("Medecin", back_populates="consultations")
    scan_record = relationship("ScanRecord")

    def __repr__(self) -> str:
        return (
            f"<Consultation(id={self.id}, patient_id={self.patient_id}, "
            f"medecin_id={self.medecin_id}, date={self.date_consultation})>"
        )


# ── Database engine setup ─────────────────────────────────────────────

def create_database_engine() -> tuple:
    """
    Create SQLAlchemy engine and session factory for MySQL.
    
    Raises:
        RuntimeError: If MySQL configuration is missing in environment.
    """
    settings = get_settings()
    db_url = settings.db.url
    
    if not db_url:
        logger.error("MySQL configuration missing in environment variables.")
        raise RuntimeError(
            "MySQL DATABASE_URL could not be constructed. "
            "Ensure MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and "
            "MYSQL_DATABASE are set in .env"
        )
    
    logger.info("Using MySQL SGBD engine")
    engine = create_engine(
        db_url,
        echo=False,
    )

    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    logger.info("Database initialized successfully (MySQL).")
    return engine, session_factory
