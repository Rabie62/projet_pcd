"""
Shared response builder helpers for API routes.

These functions extract the common response-building logic that was
duplicated between /analyze and /analyze/upload endpoints.
"""
from __future__ import annotations
from typing import Optional
from api.schemas import (
    AnalysisResponse,
    SafetyCheckResponse,
    TumorFeaturesResponse,
    TumorRegionResponse,
)


def build_analysis_response(
    session: dict,
    controller,
) -> AnalysisResponse:
    """
    Build an AnalysisResponse from a session state dict and controller.

    This is the shared builder used by both /analyze and /analyze/upload
    endpoints to ensure consistent response formatting.
    """
    report = session.get("diagnostic_report")
    safety = session.get("safety_check")
    vision = session.get("vision_result")

    tumor_features = None
    tumor_regions = []

    if report and report.tumor_detected and report.tumor_features:
        f = report.tumor_features
        tumor_features = TumorFeaturesResponse(
            tumor_area_mm2=f.tumor_area_mm2 or 0.0,
            max_diameter_mm=f.max_diameter_mm or 0.0,
            tumor_ratio=f.tumor_ratio or 0.0,
            location_description=f.location_description or "Unknown",
        )

    if vision:
        tumor_regions = [
            TumorRegionResponse(
                label=r.label,
                label_name=r.label_name,
                area_pixels=r.area_pixels,
                area_mm2=r.area_mm2 or 0.0,
                centroid=list(r.centroid),
            )
            for r in vision.tumor_regions
        ]

    safety_response = None
    if safety:
        safety_response = SafetyCheckResponse(
            passed=safety.passed,
            confidence_adequate=safety.confidence_adequate,
            requires_human_review=safety.requires_human_review,
            flags=safety.flags,
            warnings=safety.warnings,
            compliance_notes=safety.compliance_notes,
        )

    return AnalysisResponse(
        session_id=session.get("session_id", ""),
        patient_id=str(session.get("patient_id", "")),
        status=session.get("status", "unknown"),
        tumor_detected=report.tumor_detected if report else False,
        classification=report.classification if report else None,
        classification_confidence=(
            report.classification_confidence if report else 0.0
        ),
        who_grade=report.who_grade if report else None,
        tumor_features=tumor_features,
        tumor_regions=tumor_regions,
        clinical_summary=report.clinical_summary if report else "",
        recommendations=report.recommendations if report else [],
        safety_check=safety_response,
        review_status=session.get("review_status"),
        disclaimer=controller.get_disclaimer(),
        errors=session.get("errors", []),
    )
