"""
Diagnostic Agent — synthesises vision outputs into structured medical insights.

Responsibilities:
  1. Extract quantitative tumor features (area, diameter, intensity)
  2. Determine approximate anatomical location
  3. Map classification to clinical grading
  4. Generate a structured radiology report (Indication / Technique / Résultats /
     Interprétation / Conclusion / Recommandations)

Adapted for the BRISC 2025 dataset (4 classes: Glioma, Meningioma, Pituitary, No Tumor).

Design principle: everything deterministic (header, measurements, fallback,
disclaimer) is produced in Python; the LLM only writes the
narrative text.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from loguru import logger
from scipy.spatial.distance import pdist
from agents.vision import VisionResult, TumorRegion

REPORT_VERSION = "2.2-BRISC"

DISCLAIMER = (
    "*Compte rendu généré avec assistance d'intelligence artificielle. "
    "Il ne constitue pas un diagnostic et doit être validé par un médecin "
    "avant toute décision clinique.*"
)

TECHNIQUE_TEXT = (
    "IRM cérébrale, séquence pondérée T1, analyse 2D. "
    "Injection de produit de contraste : non renseignée. "
    "Analyse assistée par IA (segmentation et classification automatiques)."
)


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
    report_version: str = REPORT_VERSION

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
        self.rag_system = None
        self.dialogue_agent = dialogue_agent

    # ──────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────────────────────────────────
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
        features.tumor_ratio = (
            features.tumor_area_px / features.total_image_area_px
            if features.total_image_area_px > 0 else 0
        )

        tumor_mask = segmentation > 0
        if tumor_mask.any():
            tumor_coords = np.argwhere(tumor_mask)
            actual_spacing = np.array(spacing[:tumor_coords.shape[1]])
            tumor_coords_mm = tumor_coords.astype(float) * actual_spacing

            if len(tumor_coords_mm) > 1:
                sample = tumor_coords_mm[
                    np.random.choice(len(tumor_coords_mm), min(5000, len(tumor_coords_mm)), replace=False)
                ]
                features.max_diameter_mm = float(pdist(sample).max())

            centroid = tumor_coords.mean(axis=0)
            features.centroid_mm = tuple(
                centroid[i] * spacing[i] for i in range(min(len(centroid), len(spacing)))
            )
            features.location_description = self.determine_location(tumor_mask, segmentation.shape)

            intensity_values = (
                image[tumor_mask] if image.ndim == tumor_mask.ndim
                else image[:, tumor_mask].mean(axis=0)
            )
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

    # ──────────────────────────────────────────────────────────────────────
    # Deterministic helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def build_header(patient_id: str) -> str:
        """Administrative header, generated in Python (never by the LLM)."""
        now = datetime.now(timezone.utc)
        return (
            "# COMPTE RENDU D'IRM CÉRÉBRALE\n"
            f"**Patient :** {patient_id} | "
            f"**Date :** {now:%d/%m/%Y %H:%M} UTC | "
            f"**Version :** {REPORT_VERSION}\n\n"
        )

    @staticmethod
    def build_recommendations(vision_result: VisionResult, features: Optional[TumorFeatures]) -> tuple[str, list[str]]:
        """Return (conclusion sentence, prioritised list of recommendations)."""
        if not vision_result.tumor_detected:
            return (
                "Examen sans processus expansif décelable (analyse 2D, séquence T1 seule).",
                ["Poursuite de la surveillance clinique habituelle."],
            )

        large = bool(features and features.max_diameter_mm > 40)
        conclusion = (
            f"Processus expansif compatible avec : {vision_result.tumor_class}"
            + (" (lésion volumineuse > 40 mm)." if large else ".")
        )
        recs = []
        if large:
            recs.append("Avis neurochirurgical urgent (lésion > 40 mm).")
        else:
            recs.append("Avis neurochirurgical spécialisé.")
        recs.append("IRM cérébrale complète avec injection (FLAIR, T2, diffusion, perfusion).")
        recs.append("Corrélation histopathologique indispensable pour confirmation du grade.")
        return conclusion, recs

    # ──────────────────────────────────────────────────────────────────────
    # Report generation
    # ──────────────────────────────────────────────────────────────────────
    def generate_report(self, vision_result: VisionResult, clinical_notes: str = "", patient_history_context: str = "") -> DiagnosticReport:
        features = self.extract_features(vision_result)
        flags = []

        if not vision_result.tumor_detected:
            conclusion, recs = self.build_recommendations(vision_result, None)
            summary = (
                self.build_header(vision_result.patient_id)
                + "## INDICATION\nNon renseignée.\n\n"
                + f"## TECHNIQUE\n{TECHNIQUE_TEXT}\n\n"
                + "## RÉSULTATS\nAbsence de processus expansif intracrânien décelable. "
                  "Parenchyme cérébral d'aspect normal sur les coupes analysées.\n\n"
                + "## INTERPRÉTATION\nAucune lésion tumorale détectée par l'analyse automatique. "
                  "Les limites de l'analyse (coupes 2D, séquence T1 seule) ne permettent pas "
                  "d'exclure une lésion non visible sur cette séquence.\n\n"
                + f"## CONCLUSION\n{conclusion}\n\n"
                + "## RECOMMANDATIONS\n" + "\n".join(f"- {r}" for r in recs)
                + f"\n\n{DISCLAIMER}"
            )
            return DiagnosticReport(
                patient_id=vision_result.patient_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tumor_detected=False,
                tumor_features=None,
                classification="No Tumor",
                classification_confidence=1.0,
                classification_probabilities={},
                who_grade=None,
                clinical_summary=summary,
                recommendations=recs,
                flags=[],
            )

        clinical_grade = self.CLINICAL_GRADING.get(vision_result.tumor_class, "Non classifié")

        if vision_result.tumor_class_confidence < 0.85:
            flags.append("Confiance IA modérée")

        # GÉNÉRATION DU RAPPORT (LLM)
        clinical_summary = self.generate_llm_summary(
            vision_result, features, clinical_grade, flags,
            clinical_notes, patient_history_context
        )

        _, recs = self.build_recommendations(vision_result, features)

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
            recommendations=recs,
            flags=flags,
        )

    # ── System prompt : compte rendu de radiologie hospitalier ──
    REPORT_SYSTEM_PROMPT = """Vous êtes médecin radiologue spécialisé en neuroradiologie. Vous rédigez le compte rendu \
d'une IRM cérébrale à partir des résultats d'un système d'analyse assistée par intelligence artificielle \
(segmentation et classification automatiques). Ce compte rendu sera versé au dossier médical et relu par un clinicien.

PRINCIPES MÉDICO-LÉGAUX
- Vous ne disposez que des données fournies. Toute information absente (antécédents, injection de produit de \
contraste, comparaison avec un examen antérieur, signes associés tels qu'œdème péri-lésionnel, effet de masse, \
engagement, hydrocéphalie, prise de contraste) est écrite « non renseigné » ou « non évaluable ». Ne rien déduire, \
ne rien extrapoler.
- Les mesures sont reprises à l'identique : valeurs, unités et pourcentages inchangés.
- Le diagnostic provient d'un algorithme. Utilisez un langage de probabilité (« compatible avec », « en faveur de », \
« évoque », « ne permet pas d'exclure »). Aucune affirmation de certitude. La confirmation histopathologique est \
requise pour tout grade OMS.
- Aucune prescription médicamenteuse ni schéma thérapeutique. Seuls des examens complémentaires, des avis \
spécialisés et un suivi peuvent être proposés.

CONDUITE DU RAISONNEMENT
- Si la confiance du modèle est inférieure à 85 % : indiquez qu'elle est modérée et formulez au moins un diagnostic \
différentiel pertinent, avec l'argument radiologique ou épidémiologique qui le soutient.
- Si l'absence de tumeur est rapportée : conclure à un examen sans processus expansif décelable, en précisant les \
limites de l'analyse (coupes 2D, séquence T1 seule).

STYLE ET FORMAT
- Français médical, registre d'un compte rendu hospitalier : phrases courtes, voix neutre, sans formule de politesse, \
sans introduction, sans commentaire sur votre propre démarche.
- Respectez exactement les sections de l'exemple ci-dessous, dans cet ordre, avec ces titres, en Markdown.
- CONCLUSION : synthèse en deux phrases maximum. RECOMMANDATIONS : liste à puces classée par priorité.
- Terminez par la mention réglementaire figurant dans l'exemple."""

    # Exemple FICTIF : toutes les valeurs sont volontairement inventées
    REPORT_EXAMPLE = """## INDICATION
Exploration d'une lésion cérébrale suspectée. Antécédents : non renseignés.

## TECHNIQUE
IRM cérébrale, séquence pondérée T1, analyse 2D. Injection de produit de contraste : non renseignée. \
Analyse assistée par IA (segmentation et classification automatiques).

## RÉSULTATS
**Lésion :** processus expansif de la région antérieure droite.
- Dimensions : surface 1247.3 mm2, diamètre maximal 42.1 mm, ratio tumoral 8.2 %
- Signal T1 : moyenne 118.4 (σ 21.7)
**Éléments associés :** œdème péri-lésionnel, effet de masse et prise de contraste non évaluables dans ces données.

## INTERPRÉTATION
Aspect compatible avec un gliome (confiance IA : 93.2 %). Grade OMS estimé II à IV. Les gliomes diffus de l'adulte \
sont les tumeurs primitives intra-axiales les plus fréquentes ; la gradation repose sur l'analyse histologique et \
moléculaire.

## CONCLUSION
Volumineux processus expansif antérieur droit (> 40 mm), compatible avec un gliome. Gradation non possible en imagerie T1 seule.

## RECOMMANDATIONS
- Avis neurochirurgical rapide, en raison de la taille de la lésion.
- IRM cérébrale complète avec injection (FLAIR, T2, diffusion, perfusion).
- Confirmation histopathologique et moléculaire.

*Compte rendu généré avec assistance d'intelligence artificielle. Il ne constitue pas un diagnostic et doit être \
validé par un médecin avant toute décision clinique.*"""

    # ──────────────────────────────────────────────────────────────────────
    # LLM summary
    # ──────────────────────────────────────────────────────────────────────
    def generate_llm_summary(self, vision_result, features, clinical_grade, flags, clinical_notes, history) -> str:

        header = self.build_header(vision_result.patient_id)

        # ── Clinical notes & history ──
        notes_section = ""
        if clinical_notes:
            notes_section = f"Notes cliniques : {clinical_notes.strip()}\n"
        if history:
            notes_section += f"Antécédents : {history.strip()}\n"
        if not notes_section:
            notes_section = "Non renseignée.\n"
        if flags:
            notes_section += f"**Alertes** : {'; '.join(flags)}\n"

        # ── Morphometry ──
        if vision_result.tumor_detected and features:
            morphometry = (
                f"**Lésion :** processus expansif, région {features.location_description}.\n"
                f"- Dimensions : surface {features.tumor_area_mm2:.1f} mm2, "
                f"diamètre maximal {features.max_diameter_mm:.1f} mm, "
                f"ratio tumoral {features.tumor_ratio:.1%}\n"
                f"- Signal T1 : moyenne {features.t1_mean_intensity:.1f} "
                f"(σ {features.t1_std_intensity:.1f})\n"
                f"**Éléments associés :** non renseignés dans les données.\n"
            )
            if features.max_diameter_mm > 40:
                morphometry += "Lésion volumineuse (> 40 mm).\n"
        else:
            morphometry = "Aucun processus expansif détecté. Parenchyme cérébral d'aspect normal.\n"

        # ── Interpretation ──
        conf = vision_result.tumor_class_confidence
        interpretation = f"Classification : {vision_result.tumor_class} (confiance IA : {conf:.1%})\n"
        interpretation += f"Grade OMS estimé : {clinical_grade}\n"

        if vision_result.tumor_class_probabilities:
            sorted_probs = sorted(
                vision_result.tumor_class_probabilities.items(),
                key=lambda x: x[1], reverse=True
            )
            probs_str = " | ".join(f"{cls}={p:.1%}" for cls, p in sorted_probs)
            interpretation += f"Probabilités : {probs_str}\n"

        if conf < 0.85:
            interpretation += (
                "**Confiance IA modérée (< 85 %)** : diagnostic différentiel à envisager. "
                "Corrélation histopathologique fortement recommandée.\n"
            )

        # ── Conclusion & recommendations (deterministic) ──
        conclusion_text, recs = self.build_recommendations(vision_result, features)
        recs_text = "\n".join(f"- {r}" for r in recs)
        clean_conclusion = f"Conclusion : {conclusion_text}\nRecommandations :\n{recs_text}\n"

        # ── Conclusion for LLM prompt ──
        llm_conclusion = clean_conclusion

        # ── Build messages ──
        system_msg = (
            f"{self.REPORT_SYSTEM_PROMPT}\n\n"
            f"Exemple de rapport (respectez cette structure et ce style) :\n"
            f"{self.REPORT_EXAMPLE}"
        )

        user_msg = (
            f"Rédigez le compte rendu pour le patient {vision_result.patient_id}.\n\n"
            f"--- DONNÉES STRUCTURÉES ---\n"
            f"## Indication\n{notes_section}\n"
            f"## Résultats\n{morphometry}\n"
            f"## Interprétation\n{interpretation}\n"
            f"## Éléments de conclusion\n{llm_conclusion}\n"
            f"--- FIN DONNÉES ---\n\n"
            f"Rédigez maintenant le compte rendu complet en respectant strictement la structure. "
            f"Les valeurs de l'exemple sont fictives : ne les réutilisez pas."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        fallback_summary = (
            header
            + f"## INDICATION\n{notes_section}\n"
            + f"## TECHNIQUE\n{TECHNIQUE_TEXT}\n\n"
            + f"## RÉSULTATS\n{morphometry}\n"
            + f"## INTERPRÉTATION\n{interpretation}\n"
            + f"## CONCLUSION\n{conclusion_text}\n\n"
            + f"## RECOMMANDATIONS\n{recs_text}\n\n"
            + DISCLAIMER
        ).strip()

        if not self.llm_report_enabled or not self.dialogue_agent:
            return fallback_summary

        try:
            if hasattr(self.dialogue_agent, "generate_chat"):
                summary = self.dialogue_agent.generate_chat(
                    messages=messages,
                    temperature=0.2,
                )
            else:
                summary = self.dialogue_agent.chat(user_msg)

            # Drop any preamble before the first section heading
            first_heading = re.search(r"^##\s", summary, flags=re.MULTILINE)
            if first_heading:
                summary = summary[first_heading.start():]

            # Guarantee the regulatory disclaimer, whatever the model did
            if "validé par un médecin" not in summary:
                summary = summary.rstrip() + "\n\n" + DISCLAIMER

            return header + summary
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}. Falling back to structured summary.")
            return fallback_summary