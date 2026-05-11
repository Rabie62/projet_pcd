"""
Diagnostic Agent — synthesises vision outputs into structured medical insights.

Responsibilities:
  1. Extract quantitative tumor features (area, diameter, intensity)
  2. Determine approximate anatomical location
  3. Map classification to clinical grading
  4. Generate a structured diagnostic report

Adapted for the BRISC 2025 dataset (4 classes: Glioma, Meningioma, Pituitary, No Tumor).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import pdist
from agents.vision import VisionResult, TumorRegion

@dataclass
class TumorFeatures:
    """Quantitative tumor features extracted from 2D segmentation."""
    tumor_area_px: int = 0
    tumor_area_mm2: float = 0.0
    total_image_area_px: int = 0
    tumor_ratio: float = 0.0
    max_diameter_mm: float = 0.0
    max_diameter_px: float = 0.0
    centroid_mm: tuple[float, ...] = field(default_factory=tuple)
    location_description: str = ""
    t1_mean_intensity: float = 0.0
    t1_std_intensity: float = 0.0

@dataclass
class DiagnosticReport:
    """Structured diagnostic report."""
    patient_id: str
    timestamp: str
    tumor_detected: bool
    tumor_features: Optional[TumorFeatures]
    classification: Optional[str]
    classification_confidence: float
    classification_probabilities: dict[str, float]
    who_grade: Optional[str]
    clinical_summary: str
    recommendations: list[str]
    flags: list[str]
    icd11_codes: list[str] = field(default_factory=list)
    report_version: str = "2.1-BRISC"

    def to_dict(self) -> dict:
        return asdict(self)

class DiagnosticAgent:
    BRAIN_QUADRANTS = {
        (0, 0.5, 0, 0.5): "postérieur droit",
        (0, 0.5, 0.5, 1): "antérieur droit",
        (0.5, 1, 0, 0.5): "postérieur gauche",
        (0.5, 1, 0.5, 1): "antérieur gauche",
    }

    CLINICAL_GRADING = {
        "Glioma": "Grade OMS II–IV (Nécessite corrélation histopathologique)",
        "Meningioma": "Grade OMS I–III (Généralement bénin)",
        "Pituitary": "Adénome hypophysaire (Bénin)",
        "No Tumor": "Absence de processus expansif",
    }

    def __init__(self, llm_report_enabled: bool = True, rag_system=None, dialogue_agent=None):
        self.llm_report_enabled = llm_report_enabled
        self.rag_system = rag_system
        self.dialogue_agent = dialogue_agent
        from knowledge.icd11 import ICD11Client
        self.icd11 = ICD11Client()

    def extract_features(self, vision_result: VisionResult) -> TumorFeatures:
        features = TumorFeatures()
        if not vision_result.tumor_detected:
            return features

        segmentation = vision_result.segmentation_mask.squeeze()
        image = vision_result.preprocessed_image.squeeze()
        spacing = vision_result.pixel_spacing

        # Aggregate across all tumor regions
        total_area_px = 0
        total_area_mm2 = 0.0
        for region in vision_result.tumor_regions:
            total_area_px += region.area_pixels
            total_area_mm2 += region.area_mm2
        features.tumor_area_px = total_area_px
        features.tumor_area_mm2 = total_area_mm2
        
        features.total_image_area_px = segmentation.size
        features.tumor_ratio = features.tumor_area_px / features.total_image_area_px if features.total_image_area_px > 0 else 0

        tumor_mask = segmentation > 0
        if tumor_mask.any():
            tumor_coords = np.argwhere(tumor_mask)
            actual_spacing = np.array(spacing[:tumor_coords.shape[1]])
            tumor_coords_mm = tumor_coords.astype(float) * actual_spacing

            if len(tumor_coords_mm) > 1:
                sample = tumor_coords_mm[np.random.choice(len(tumor_coords_mm), min(5000, len(tumor_coords_mm)), replace=False)]
                features.max_diameter_mm = float(pdist(sample).max())

            centroid = tumor_coords.mean(axis=0)
            features.centroid_mm = tuple(centroid[i] * spacing[i] for i in range(min(len(centroid), len(spacing))))
            features.location_description = self.determine_location(tumor_mask, segmentation.shape)

            intensity_values = image[tumor_mask] if image.ndim == tumor_mask.ndim else image[:, tumor_mask].mean(axis=0)
            features.t1_mean_intensity = float(intensity_values.mean())
            features.t1_std_intensity = float(intensity_values.std())

        return features

    def determine_location(self, tumor_mask: np.ndarray, image_shape: tuple[int, ...]) -> str:
        tumor_coords = np.argwhere(tumor_mask)
        centroid = tumor_coords.mean(axis=0)
        norm_pos = [centroid[i] / image_shape[i] for i in range(2)]
        for bounds, name in self.BRAIN_QUADRANTS.items():
            if bounds[0] <= norm_pos[0] < bounds[1] and bounds[2] <= norm_pos[1] < bounds[3]:
                return name
        return "région centrale"

    def generate_report(self, vision_result: VisionResult, clinical_notes: str = "", patient_history_context: str = "") -> DiagnosticReport:
        features = self.extract_features(vision_result)
        flags = []

        if not vision_result.tumor_detected:
            return DiagnosticReport(
                patient_id=vision_result.patient_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tumor_detected=False,
                tumor_features=None,
                classification="No Tumor",
                classification_confidence=1.0,
                classification_probabilities={},
                who_grade=None,
                clinical_summary="## CONCLUSION\nExamen normal. Absence de masse tumorale intracrânienne décelable.",
                recommendations=[],
                flags=[],
            )

        clinical_grade = self.CLINICAL_GRADING.get(vision_result.tumor_class, "Non classifié")

        if vision_result.tumor_class_confidence < 0.85:
            flags.append("Confiance IA modérée")

        # Récupération codes CIM-11
        icd11_codes = []
        if self.icd11.available:
            results = self.icd11.search(f"{vision_result.tumor_class} brain tumor")
            for res in results[:5]:
                raw_id = res.get('id', '')
                segments = [s for s in raw_id.split('/') if s]
                entity_id = None
                for seg in reversed(segments):
                    if seg.replace('.', '').isdigit():
                        entity_id = seg
                        break
                if not entity_id:
                    continue
                code = self.icd11.get_mms_code(entity_id)
                if code:
                    icd11_codes.append(f"{code} ({res.get('title')})")

        # GÉNÉRATION DU RAPPORT (LLM avec contexte RAG)
        clinical_summary = self.generate_llm_summary(
            vision_result, features, clinical_grade, flags,
            clinical_notes, patient_history_context, icd11_codes
        )

        return DiagnosticReport(
            patient_id=vision_result.patient_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tumor_detected=True,
            tumor_features=features,
            classification=vision_result.tumor_class,
            classification_confidence=vision_result.tumor_class_confidence,
            classification_probabilities=vision_result.tumor_class_probabilities,
            who_grade=clinical_grade,
            clinical_summary=clinical_summary,
            recommendations=[],
            flags=flags,
            icd11_codes=icd11_codes,
        )

    # ── System prompt (optimized for Gemma-3 1B-IT — concise & directive) ──
    REPORT_SYSTEM_PROMPT = (
        "Vous etes un neuroradiologue senior. Redigez un rapport IRM structure "
        "et factuel en francais. Utilisez uniquement les donnees fournies.\n"
        "Regles :\n"
        "- Terminologie medicale rigoureuse\n"
        "- Pas de salutation ni introduction polie\n"
        "- Citez les mesures numeriques exactes\n"
        "- Si confiance < 85%, mentionnez un diagnostic differentiel\n"
        "- Integrez les codes CIM-11 et les connaissances RAG fournies\n"
        "- Terminez par : 'AVERTISSEMENT : Analyse assistee par IA. "
        "Confirmation clinique par un medecin qualifie requise.'"
    )

    # ── Report template (few-shot anchor for output format) ──
    REPORT_TEMPLATE = """## 1. SYNTHESE DES OBSERVATIONS
{clinical_notes_section}
## 2. ANALYSE MORPHOMETRIQUE ET LOCALISATION
{morphometry_section}
## 3. INTERPRETATION ET GRADATION OMS
{interpretation_section}
## 4. CONCLUSION ET RECOMMANDATIONS
{conclusion_section}
AVERTISSEMENT : Analyse assistee par IA. Confirmation clinique par un medecin qualifie requise."""

    REPORT_EXAMPLE = """## 1. SYNTHESE DES OBSERVATIONS
Patient sans antecedents notables. IRM cerebrale realisee pour exploration d'une lesion suspectee.
## 2. ANALYSE MORPHOMETRIQUE ET LOCALISATION
Processus expansif intraparenchymateux de la region anterieure droite.
Surface : 1247.3 mm2 | Diametre maximal : 42.1 mm | Ratio tumoral : 8.2%
## 3. INTERPRETATION ET GRADATION OMS
Classification : Gliome (Confiance IA : 93.2%)
Grade OMS estime : Grade II-IV (Necessite confirmation histopathologique)
## 4. CONCLUSION ET RECOMMANDATIONS
Diagnostique : Gliome suspecte. Consultation neurochirurgicale urgente recommandee.
IRM de perfusion souhaitee pour evaluation du grade.
CIM-11 : 2A00 (Gliome, non specifie)
AVERTISSEMENT : Analyse assistee par IA. Confirmation clinique par un medecin qualifie requise."""

    def generate_llm_summary(self, vision_result, features, clinical_grade, flags, clinical_notes, history, icd11) -> str:

        # ── RAG Context ──
        rag_context = ""
        rag_metrics = {}
        if self.rag_system and self.rag_system.available:
            chunks = self.rag_system.retrieve_for_findings(
                classification=vision_result.tumor_class,
                tumor_detected=vision_result.tumor_detected,
                confidence=vision_result.tumor_class_confidence,
            )
            if chunks:
                rag_context = self.rag_system.format_context(chunks, include_scores=False)
                rag_metrics = {
                    "chunks_retrieved": len(chunks),
                    "avg_score": sum(c.score for c in chunks) / len(chunks),
                    "sources": list(set(c.source for c in chunks)),
                }
                logger.info(
                    f"[RAG] Injected {len(chunks)} chunks into report "
                    f"(avg score: {rag_metrics['avg_score']:.3f})"
                )

        # ── Clinical notes & history ──
        notes_section = ""
        if clinical_notes:
            notes_section = f"Notes cliniques : {clinical_notes.strip()}\n"
        if history:
            notes_section += f"Antecedents : {history.strip()}\n"
        if not notes_section:
            notes_section = "Aucun antecedent ni note clinique fourni.\n"
        if flags:
            notes_section += f"**Alertes** : {'; '.join(flags)}\n"

        # ── Morphometry ──
        if vision_result.tumor_detected and features:
            morphometry = (
                f"Processus expansif detecte.\n"
                f"Surface : {features.tumor_area_mm2:.1f} mm2 | "
                f"Diametre maximal : {features.max_diameter_mm:.1f} mm | "
                f"Ratio tumoral : {features.tumor_ratio:.1%}\n"
                f"Localisation : {features.location_description}\n"
                f"Intensite T1 moyenne : {features.t1_mean_intensity:.1f} "
                f"(sigma : {features.t1_std_intensity:.1f})\n"
            )
            if features.max_diameter_mm > 40:
                morphometry += "**Volumineux** : diametre > 40 mm, evaluation urgente recommandee.\n"
        else:
            morphometry = "Aucun processus expansif detecte. Parenchyme cerebral d'aspect normal.\n"

        # ── Interpretation ──
        conf = vision_result.tumor_class_confidence
        interpretation = f"Classification : {vision_result.tumor_class} (Confiance IA : {conf:.1%})\n"
        interpretation += f"Grade OMS estime : {clinical_grade}\n"

        # Probabilities for top classes
        if vision_result.tumor_class_probabilities:
            sorted_probs = sorted(
                vision_result.tumor_class_probabilities.items(),
                key=lambda x: x[1], reverse=True
            )
            probs_str = " | ".join(f"{cls}={p:.1%}" for cls, p in sorted_probs)
            interpretation += f"Probabilites : {probs_str}\n"

        # Differential diagnosis if low confidence
        if conf < 0.85:
            interpretation += (
                "**Confiance IA moderee (< 85%)** : diagnostic differentiel a envisager. "
                "Correlation histopathologique forteement recommandee.\n"
            )

        # ── Conclusion ──
        conclusion_parts = []
        if vision_result.tumor_detected:
            conclusion_parts.append(f"Diagnostique : {vision_result.tumor_class} detecte.")
        else:
            conclusion_parts.append("Examen normal. Absence de processus expansif detecte.")

        if icd11:
            conclusion_parts.append("Codes CIM-11 : " + " | ".join(icd11))

        conclusion = "\n".join(conclusion_parts) + "\n"

        # ── RAG enrichment ──
        if rag_context:
            conclusion += (
                "\nReferences cliniques RAG (integrez les informations pertinentes "
                "ci-dessus dans le rapport) :\n" + rag_context + "\n"
            )

        # ── Build messages ──
        system_msg = (
            f"{self.REPORT_SYSTEM_PROMPT}\n\n"
            f"Structure du rapport (respectez ce format) :\n"
            f"{self.REPORT_TEMPLATE}\n\n"
            f"Exemple de rapport (calquez ce style) :\n"
            f"{self.REPORT_EXAMPLE}"
        )

        user_msg = (
            f"Redigez le rapport pour le patient {vision_result.patient_id}.\n\n"
            f"--- DONNEES STRUCTUREES ---\n"
            f"## 1. Contexte\n{notes_section}\n"
            f"## 2. Morphometrie\n{morphometry}\n"
            f"## 3. Classification\n{interpretation}\n"
            f"## 4. Elements de conclusion\n{conclusion}\n"
            f"--- FIN DONNEES ---\n\n"
            f"Redigez maintenant le rapport complet en respectant la structure."
        )
    
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        inputs = self.dialogue_agent.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(self.dialogue_agent.model.device)

        with torch.no_grad():
            outputs = self.dialogue_agent.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.25,
                do_sample=True,
                repetition_penalty=1.15,
            )

        summary = self.dialogue_agent.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Strip RAG metrics line if leaked into output
        summary = re.sub(r'\[RAG Metrics:.*?\]', '', summary).strip()

        return summary

