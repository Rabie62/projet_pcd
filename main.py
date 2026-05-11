"""
Medical AI Agent — CLI entry point.
Supports running the API server, single-patient analysis, and batch processing.
Adapted for BRISC 2025 dataset (2D T1-weighted MRI images).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from loguru import logger
from config.settings import get_settings
from agents.controller import ControllerAgent
from data.loader import BRISCDataLoader
from interpretability.visualizations import BRISCVisualizer


def run_server(args):
    """Start the FastAPI server."""
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def run_analyze(args):
    """Analyze a single patient image or directory."""

    settings = get_settings()
    controller = ControllerAgent(settings)
    controller.load_models()

    patient_path = Path(args.patient_dir)
    if not patient_path.exists():
        logger.error(f"Path not found: {patient_path}")
        sys.exit(1)

    # Support both single image and directory
    if patient_path.is_file():
        result = controller.analyze_image_file(patient_path)
    else:
        result = controller.analyze_image_dir(patient_path)

    # Print report
    report = result.get("diagnostic_report")
    if report:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Patient Info
        patient_display = report.patient_id
        # If we have patient info in the result, use prenom nom
        if result.get("patient_info"):
            p = result["patient_info"]
            patient_display = f"{p.prenom} {p.nom} (ID: {p.id})"
            
        print(f"\n### PATIENT: {patient_display}")
        print(f"**TUMOR DETECTED:** {'YES' if report.tumor_detected else 'NO'}")

        if report.tumor_detected:
            print(f"\n#### 1. CLASSIFICATION")
            print(f"- **Type:** {report.classification}")
            print(f"- **WHO Grade:** {report.who_grade}")
            print(f"- **AI Confidence:** {report.classification_confidence:.1%}")
            
            if report.tumor_features:
                f = report.tumor_features
                print(f"\n#### 2. QUANTITATIVE MEASURES")
                print(f"- **Tumor Area:** {f.tumor_area_mm2:.1f} mm²")
                print(f"- **Max Diameter:** {f.max_diameter_mm:.1f} mm")
                print(f"- **Tumor-to-Image Ratio:** {f.tumor_ratio:.2%}")
                print(f"- **Anatomical Location:** {f.location_description}")

        print(f"\n#### 3. CLINICAL SUMMARY")
        # Ensure the summary itself is printed clearly
        summary_lines = report.clinical_summary.split("\n")
        for line in summary_lines:
            print(f"  {line}")

        safety_check = result.get("safety_check")
        if safety_check:
            status = "PASS" if safety_check.passed else "REVIEW REQUIRED"
            print(f"\n#### 4. SAFETY AUDIT: {status}")
            for flag in safety_check.flags:
                print(f"  - [!] {flag}")

        print(f"\n---\n*DISCLAIMER: {controller.get_disclaimer()}*")
        print("=" * 60)

    # Generate visualization
    vision_result = result.get("vision_result")
    if vision_result and args.output:
        output = Path(args.output)
        viz = BRISCVisualizer()
        
        # 2D summary (Image, GT if available, Prediction)
        png = viz.generate_summary(
            vision_result.preprocessed_image,
            seg_mask=getattr(vision_result, 'ground_truth_mask', None),
            pred_mask=vision_result.segmentation_mask,
        )
            
        viz.save_visualization(png, output)
        print(f"\n  Visualization saved to: {output}")


def run_batch(args):
    """Batch analyze all patients in a directory."""

    settings = get_settings()
    controller = ControllerAgent(settings)
    controller.load_models()

    loader = BRISCDataLoader(args.data_dir)
    logger.info(f"Found {loader.num_patients} images")

    results = []
    for patient in loader.iterate_patients(max_patients=args.max_patients):
        try:
            result = controller.analyze_patient(patient)
            report = result.get("diagnostic_report")
            results.append({
                "patient_id": patient.patient_id,
                "tumor": report.tumor_detected if report else None,
                "class": report.classification if report else None,
                "confidence": (
                    report.classification_confidence if report else 0
                ),
                "status": result.get("status", "unknown"),
            })
            logger.info(
                f"  {patient.patient_id}: "
                f"tumor={report.tumor_detected if report else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"  {patient.patient_id}: FAILED — {e}")
            results.append({
                "patient_id": patient.patient_id,
                "status": "failed",
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 40}")
    print(f"  Processed: {len(results)} images")
    tumors = sum(1 for r in results if r.get("tumor"))
    print(f"  Tumors found: {tumors}")
    print(f"{'=' * 40}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical AI Agent — Brain Tumor MRI Analysis (BRISC 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python main.py serve

  # Analyze a single image
  python main.py analyze --patient-dir data/brisc/sample_001.jpg

  # Analyze a directory of images
  python main.py analyze --patient-dir data/brisc/glioma/

  # Batch analyze all images
  python main.py batch --data-dir data/brisc --max-patients 10
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve = subparsers.add_parser("serve", help="Start the API server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    # analyze command
    analyze = subparsers.add_parser(
        "analyze", help="Analyze a single patient"
    )
    analyze.add_argument(
        "--patient-dir", required=True,
        help="Path to image file or directory with BRISC JPG images"
    )
    analyze.add_argument(
        "--output", "-o",
        help="Output path for overlay image (PNG)"
    )

    # batch command
    batch = subparsers.add_parser(
        "batch", help="Batch analyze patients"
    )
    batch.add_argument(
        "--data-dir", required=True,
        help="Path to BRISC data directory"
    )
    batch.add_argument(
        "--max-patients", type=int, default=None,
        help="Max images to process"
    )

    # Default to 'serve' if no command is provided
    if len(sys.argv) == 1:
        sys.argv.append("serve")

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "batch":
        run_batch(args)


if __name__ == "__main__":
    main()
