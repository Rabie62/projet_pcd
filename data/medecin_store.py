"""
Doctor and Consultation Store — CRUD operations.

Provides business logic for managing doctor records and consultation records.
Uses SQLAlchemy sessions for transactional safety.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from data.database import (
    Patient,
    Medecin,
    Consultation,
    ScanRecord,
    create_database_engine,
)


class MedecinConsultationStore:
    """
    Store for doctors and consultations backed by MySQL via SQLAlchemy.
    """

    def __init__(self):
        self.engine, self.session_factory = create_database_engine()
        logger.info("MedecinConsultationStore initialised (MySQL)")

    # ── Médecin CRUD ──────────────────────────────────────────────────

    def create_medecin(
        self,
        nom: str,
        prenom: str,
        specialite: Optional[str] = None,
        tel: Optional[str] = None,
        email: Optional[str] = None,
        departement: Optional[str] = None,
    ) -> Medecin:
        """Create a new doctor record."""
        medecin = Medecin(
            nom=nom,
            prenom=prenom,
            specialite=specialite,
            tel=tel,
            email=email,
            departement=departement,
        )
        with self.session_factory() as session:
            session.add(medecin)
            session.commit()
            session.refresh(medecin)
            _ = medecin.consultations
            session.expunge(medecin)

        logger.info(f"Medecin created: ID {medecin.id} ({nom} {prenom})")
        return medecin

    def get_medecin(self, id: int) -> Optional[Medecin]:
        """Get a doctor by ID, including consultations."""
        with self.session_factory() as session:
            medecin = session.get(Medecin, id)
            if medecin is None:
                return None
            _ = medecin.consultations
            session.expunge(medecin)
            return medecin

    def update_medecin(
        self,
        id: int,
        nom: Optional[str] = None,
        prenom: Optional[str] = None,
        specialite: Optional[str] = None,
        tel: Optional[str] = None,
        email: Optional[str] = None,
        departement: Optional[str] = None,
    ) -> Optional[Medecin]:
        """Update an existing doctor's fields. Only non-None values are updated."""
        with self.session_factory() as session:
            medecin = session.get(Medecin, id)
            if medecin is None:
                return None

            if nom is not None:
                medecin.nom = nom
            if prenom is not None:
                medecin.prenom = prenom
            if specialite is not None:
                medecin.specialite = specialite
            if tel is not None:
                medecin.tel = tel
            if email is not None:
                medecin.email = email
            if departement is not None:
                medecin.departement = departement

            medecin.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(medecin)
            _ = medecin.consultations
            session.expunge(medecin)

            logger.info(f"Medecin updated: ID {id}")
            return medecin

    def delete_medecin(self, id: int) -> bool:
        """Delete a doctor and all linked consultation records."""
        with self.session_factory() as session:
            medecin = session.get(Medecin, id)
            if medecin is None:
                return False
            session.delete(medecin)
            session.commit()

        logger.info(f"Medecin deleted: ID {id}")
        return True

    def list_medecins(self) -> list[Medecin]:
        """List all doctors with consultation counts."""
        with self.session_factory() as session:
            medecins = session.query(Medecin).order_by(
                Medecin.created_at.asc(),
                Medecin.id.asc()
            ).all()
            for m in medecins:
                _ = m.consultations
            session.expunge_all()
            return medecins

    # ── Consultation CRUD ─────────────────────────────────────────────

    def create_consultation(
        self,
        patient_id: int,
        medecin_id: int,
        scan_record_id: Optional[int] = None,
        motif: Optional[str] = None,
        diagnostic: Optional[str] = None,
        notes: Optional[str] = None,
        rapport_genere: Optional[str] = None,
        statut: str = "en_cours",
    ) -> Optional[Consultation]:
        """Create a new consultation record."""
        with self.session_factory() as session:
            patient = session.get(Patient, patient_id)
            medecin = session.get(Medecin, medecin_id)
            if patient is None:
                logger.warning(f"Cannot create consultation: patient {patient_id} not found")
                return None
            if medecin is None:
                logger.warning(f"Cannot create consultation: medecin {medecin_id} not found")
                return None

            # Validate scan_record if provided
            if scan_record_id is not None:
                scan = session.get(ScanRecord, scan_record_id)
                if scan is None or scan.patient_id != patient_id:
                    logger.warning(f"Cannot link scan: scan {scan_record_id} not found or not for this patient")
                    return None

            consultation = Consultation(
                patient_id=patient_id,
                medecin_id=medecin_id,
                scan_record_id=scan_record_id,
                date_consultation=datetime.now(timezone.utc),
                motif=motif,
                diagnostic=diagnostic,
                notes=notes,
                rapport_genere=rapport_genere,
                statut=statut,
            )
            session.add(consultation)
            session.commit()
            session.refresh(consultation)
            session.expunge(consultation)

        logger.info(f"Consultation created: ID {consultation.id} — patient {patient_id}, medecin {medecin_id}")
        return consultation

    def get_consultation(self, id: int) -> Optional[Consultation]:
        """Get a consultation by ID with full details."""
        with self.session_factory() as session:
            consultation = session.get(Consultation, id)
            if consultation is None:
                return None
            session.expunge(consultation)
            return consultation

    def update_consultation(
        self,
        id: int,
        medecin_id: Optional[int] = None,
        motif: Optional[str] = None,
        diagnostic: Optional[str] = None,
        notes: Optional[str] = None,
        rapport_genere: Optional[str] = None,
        statut: Optional[str] = None,
    ) -> Optional[Consultation]:
        """Update an existing consultation."""
        with self.session_factory() as session:
            consultation = session.get(Consultation, id)
            if consultation is None:
                return None

            if medecin_id is not None:
                consultation.medecin_id = medecin_id
            if motif is not None:
                consultation.motif = motif
            if diagnostic is not None:
                consultation.diagnostic = diagnostic
            if notes is not None:
                consultation.notes = notes
            if rapport_genere is not None:
                consultation.rapport_genere = rapport_genere
            if statut is not None:
                consultation.statut = statut

            consultation.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(consultation)
            session.expunge(consultation)

            logger.info(f"Consultation updated: ID {id}")
            return consultation

    def delete_consultation(self, id: int) -> bool:
        """Delete a consultation record."""
        with self.session_factory() as session:
            consultation = session.get(Consultation, id)
            if consultation is None:
                return False
            session.delete(consultation)
            session.commit()

        logger.info(f"Consultation deleted: ID {id}")
        return True

    def list_consultations(
        self,
        patient_id: Optional[int] = None,
        medecin_id: Optional[int] = None,
    ) -> list[Consultation]:
        """List consultations, optionally filtered by patient or doctor."""
        with self.session_factory() as session:
            query = session.query(Consultation)
            if patient_id is not None:
                query = query.filter_by(patient_id=patient_id)
            if medecin_id is not None:
                query = query.filter_by(medecin_id=medecin_id)
            consultations = query.order_by(Consultation.date_consultation.desc()).all()
            session.expunge_all()
            return consultations

    def list_patient_consultations_with_names(self, patient_id: int) -> list[dict]:
        """List consultations for a patient with doctor and patient names."""
        with self.session_factory() as session:
            consultations = (
                session.query(Consultation, Medecin, Patient)
                .join(Medecin, Consultation.medecin_id == Medecin.id)
                .join(Patient, Consultation.patient_id == Patient.id)
                .filter(Consultation.patient_id == patient_id)
                .order_by(Consultation.date_consultation.desc())
                .all()
            )
            results = []
            for c, m, p in consultations:
                result = {
                    "consultation": c,
                    "medecin_nom": m.nom,
                    "medecin_prenom": m.prenom,
                    "medecin_specialite": m.specialite,
                    "patient_nom": p.nom,
                    "patient_prenom": p.prenom,
                    "session_id": c.scan_record.session_id if c.scan_record else None,
                }
                results.append(result)
            session.expunge_all()
            return results

    def list_all_consultations_with_names(self) -> list[dict]:
        """List all consultations with doctor and patient names."""
        with self.session_factory() as session:
            consultations = (
                session.query(Consultation, Medecin, Patient)
                .join(Medecin, Consultation.medecin_id == Medecin.id)
                .join(Patient, Consultation.patient_id == Patient.id)
                .order_by(Consultation.date_consultation.desc())
                .all()
            )
            results = []
            for c, m, p in consultations:
                result = {
                    "consultation": c,
                    "medecin_nom": m.nom,
                    "medecin_prenom": m.prenom,
                    "medecin_specialite": m.specialite,
                    "patient_nom": p.nom,
                    "patient_prenom": p.prenom,
                    "session_id": c.scan_record.session_id if c.scan_record else None,
                }
                results.append(result)
            session.expunge_all()
            return results
