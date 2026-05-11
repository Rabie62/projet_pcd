"""
Patient History Store — CRUD operations and history formatting.

Provides all business logic for managing patient records and their
scan history. Uses SQLAlchemy sessions for transactional safety.
"""

from __future__ import annotations
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger
from data.database import (
    Patient,
    ScanRecord,
    create_database_engine,
)


class PatientStore:
    """
    Patient history store backed by MySQL via SQLAlchemy.

    Provides:
      - CRUD for patient profiles
      - Scan history linking and retrieval
      - History formatting for LLM prompt injection
    """

    def __init__(self):
        self.engine, self.session_factory = create_database_engine()
        logger.info("PatientStore initialised (MySQL)")

    # ── Patient CRUD ──────────────────────────────────────────────────

    def create_patient(
        self,
        prenom: str,
        nom: str,
        date_naissance: Optional[datetime] = None,
        genre: Optional[str] = None,
        tel: Optional[str] = None,
        poids: Optional[float] = None,
        taille: Optional[float] = None,
        FC: Optional[int] = None,
        glycemie: Optional[float] = None,
    ) -> Patient:
        """
        Create a new patient record.

        Args:
            nom: patient last name
            prenom: patient first name
            date_naissance: birth date
            genre: "M" or "F"
            tel: phone number
            poids: weight in kg
            taille: height in cm
            FC: heart rate
            glycemie: blood sugar

        Returns:
            Created Patient object
        """
        patient = Patient(
            nom=nom,
            prenom=prenom,
            date_naissance=date_naissance,
            genre=genre,
            tel=tel,
            poids=poids,
            taille=taille,
            FC=FC,
            glycemie=glycemie,
        )

        with self.session_factory() as session:
            session.add(patient)
            session.commit()
            session.refresh(patient)
            # Access relationship to eager-load before detachment
            _ = patient.scans
            # Detach from session so it can be used outside
            session.expunge(patient)

        logger.info(f"Patient created: ID {patient.id} ({nom} {prenom})")
        return patient

    def get_patient(self, id: int) -> Optional[Patient]:
        """Get a patient by ID, including scan history."""
        with self.session_factory() as session:
            patient = session.get(Patient, id)
            if patient is None:
                return None
            # Eager-load scans before detaching
            _ = patient.scans
            session.expunge(patient)
            return patient

    def update_patient(
        self,
        id: int,
        nom: Optional[str] = None,
        prenom: Optional[str] = None,
        date_naissance: Optional[datetime] = None,
        genre: Optional[str] = None,
        tel: Optional[str] = None,
        poids: Optional[float] = None,
        taille: Optional[float] = None,
        FC: Optional[int] = None,
        glycemie: Optional[float] = None,
    ) -> Optional[Patient]:
        """Update an existing patient's fields. Only non-None values are updated."""
        with self.session_factory() as session:
            patient = session.get(Patient, id)
            if patient is None:
                return None

            if nom is not None:
                patient.nom = nom
            if prenom is not None:
                patient.prenom = prenom
            if date_naissance is not None:
                patient.date_naissance = date_naissance
            if genre is not None:
                patient.genre = genre
            if tel is not None:
                patient.tel = tel
            if poids is not None:
                patient.poids = poids
            if taille is not None:
                patient.taille = taille
            if FC is not None:
                patient.FC = FC
            if glycemie is not None:
                patient.glycemie = glycemie

            patient.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(patient)
            _ = patient.scans
            session.expunge(patient)

            logger.info(f"Patient updated: ID {id}")
            return patient

    def delete_patient(self, id: int) -> bool:
        """Delete a patient and all linked scan records."""
        with self.session_factory() as session:
            patient = session.get(Patient, id)
            if patient is None:
                return False
            session.delete(patient)
            session.commit()

        logger.info(f"Patient deleted: ID {id}")
        return True

    def list_patients(self) -> list[Patient]:
        """List all patients with scan counts."""
        with self.session_factory() as session:
            patients = session.query(Patient).order_by(
                Patient.created_at.asc(),
                Patient.id.asc()
            ).all()
            # Eager-load scans for each patient
            for p in patients:
                _ = p.scans
            session.expunge_all()
            return patients

    # ── Scan History ──────────────────────────────────────────────────

    def link_scan(
        self,
        patient_id: int,
        session_id: str,
        classification: Optional[str] = None,
        confidence: Optional[float] = None,
        tumor_area_mm2: Optional[float] = None,
        max_diameter_mm: Optional[float] = None,
        tumor_location: Optional[str] = None,
        clinical_summary: Optional[str] = None,
        review_status: str = "pending_review",
    ) -> Optional[ScanRecord]:
        """
        Link an analysis session to a patient as a scan record.

        Args:
            patient_id: ID of the patient to link to
            session_id: analysis session ID
            classification: tumor classification result
            confidence: classification confidence
            tumor_area_mm2: tumor area in mm²
            max_diameter_mm: maximum tumor diameter in mm
            tumor_location: tumor location description
            clinical_summary: LLM-generated report summary
            review_status: review status for the scan

        Returns:
            Created ScanRecord or None if patient not found
        """
        with self.session_factory() as session:
            patient = session.get(Patient, patient_id)
            if patient is None:
                logger.warning(
                    f"Cannot link scan: patient {patient_id} not found"
                )
                return None

            # Check if session is already linked
            existing = session.query(ScanRecord).filter_by(
                session_id=session_id
            ).first()
            if existing:
                logger.warning(
                    f"Session {session_id} already linked to "
                    f"patient {existing.patient_id}"
                )
                return existing

            record = ScanRecord(
                patient_id=patient_id,
                session_id=session_id,
                scan_date=datetime.now(timezone.utc),
                classification=classification,
                confidence=confidence,
                tumor_area_mm2=tumor_area_mm2,
                max_diameter_mm=max_diameter_mm,
                tumor_location=tumor_location,
                clinical_summary=clinical_summary,
                review_status=review_status,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)

        logger.info(
            f"Scan linked: session {session_id} → patient ID {patient_id}"
        )
        return record

    def get_scan_history(self, patient_id: int) -> list[ScanRecord]:
        """Get all scans for a patient, ordered by date (newest first)."""
        with self.session_factory() as session:
            records = (
                session.query(ScanRecord)
                .filter_by(patient_id=patient_id)
                .order_by(ScanRecord.scan_date.desc())
                .all()
            )
            session.expunge_all()
            return records

    def update_scan_review(
        self,
        session_id: str,
        review_status: str,
        reviewed_by: str,
    ) -> Optional[ScanRecord]:
        """Update the review status of a scan record."""
        with self.session_factory() as session:
            record = session.query(ScanRecord).filter_by(
                session_id=session_id
            ).first()
            if record is None:
                return None

            record.review_status = review_status
            record.reviewed_by = reviewed_by
            record.reviewed_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    # ── History Formatting for LLM Prompt ─────────────────────────────

    def format_history_for_prompt(self, patient_id: int) -> str:
        """
        Format a patient's full history into a text block for LLM injection (BRISC).

        Returns an empty string if the patient is not found or has no history.
        """
        patient = self.get_patient(patient_id)
        if patient is None:
            return ""

        parts = ["PATIENT HISTORY:"]

        # Demographics
        demo = f"Patient: {patient.prenom} {patient.nom}"
        if patient.date_naissance:
            dob = patient.date_naissance.strftime("%Y-%m-%d")
            parts.append(f"Date de naissance: {dob}")
        if patient.genre:
            parts.append(f"Genre: {patient.genre}")
        if patient.tel:
            parts.append(f"Tel: {patient.tel}")
        if patient.poids:
            parts.append(f"Poids: {patient.poids} kg")
        if patient.taille:
            parts.append(f"Taille: {patient.taille} cm")
        if patient.FC:
            parts.append(f"FC: {patient.FC} bpm")
        if patient.glycemie:
            parts.append(f"Glycémie: {patient.glycemie} g/L")

        parts.insert(1, demo)

        # Prior scans
        scans = self.get_scan_history(patient_id)
        if scans:
            parts.append("\nPRIOR SCANS:")
            for i, scan in enumerate(scans, 1):
                date_str = scan.scan_date.strftime("%Y-%m-%d")
                scan_line = f"[{i}] {date_str}"
                if scan.classification:
                    scan_line += f" — {scan.classification}"
                if scan.confidence is not None:
                    scan_line += f" (conf: {scan.confidence:.2f})"
                if scan.max_diameter_mm is not None:
                    scan_line += f", {scan.max_diameter_mm:.1f}mm diameter"
                if scan.tumor_location:
                    scan_line += f", {scan.tumor_location}"
                parts.append(scan_line)

                # Review status
                if scan.review_status == "approved" and scan.reviewed_by:
                    parts.append(
                        f"    Status: Approved by {scan.reviewed_by}"
                    )
                elif scan.review_status == "rejected":
                    parts.append(f"    Status: Rejected")
                else:
                    parts.append(f"    Status: {scan.review_status}")

                # Clinical summary excerpt
                if scan.clinical_summary:
                    summary = scan.clinical_summary[:200]
                    if len(scan.clinical_summary) > 200:
                        summary += "..."
                    parts.append(f"    Summary: {summary}")

        return "\n".join(parts)
