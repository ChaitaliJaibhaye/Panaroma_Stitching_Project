"""
cli.py
-------
Command-line interface for the Panorama Stitcher.

Supports stitching a single set of images, or batch-processing multiple
sub-folders (each treated as one panorama job) in one run.

Examples:
    # Stitch all images in a folder into one panorama:
    python cli.py --input sample_images/garden --output outputs/garden.png

    # Batch mode: each sub-folder of --batch-input becomes its own panorama.
    python cli.py --batch-input sample_images/ --output-dir outputs/

Author: Panorama Stitching Project
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import cv2

from ml_predictor import SuccessPredictor
from quality_metrics import build_quality_report
from stitcher import PanoramaStitcher, StitchingError
from utils import ImageValidationError, ensure_dir, validate_image_file

SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_images_from_folder(folder: str) -> List:
    """Load and validate all supported images from a folder, sorted by name.

    Args:
        folder: Path to a directory containing images.

    Returns:
        List of decoded BGR images.

    Raises:
        StitchingError: If the folder has fewer than 2 valid images.
    """
    paths: List[str] = []
    for pattern in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    paths = sorted(set(paths))

    images = []
    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        try:
            validate_image_file(os.path.basename(path), data)
        except ImageValidationError as exc:
            print(f"  Skipping '{path}': {exc}", file=sys.stderr)
            continue
        image = cv2.imread(path)
        if image is not None:
            images.append(image)

    if len(images) < 2:
        raise StitchingError(f"Folder '{folder}' needs at least 2 valid images.")

    return images


def build_stitcher(args: argparse.Namespace) -> PanoramaStitcher:
    """Instantiate a PanoramaStitcher from parsed CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured PanoramaStitcher instance.
    """
    return PanoramaStitcher(
        blend_method=args.blend,
        use_cylindrical=args.cylindrical,
        exposure_compensation=not args.no_exposure_comp,
        auto_order=not args.no_auto_order,
        max_dimension=args.max_dimension,
        debug=args.debug,
    )


def run_single(args: argparse.Namespace) -> int:
    """Run stitching for a single folder of images.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0 = success).
    """
    print(f"Loading images from '{args.input}'...")
    try:
        images = load_images_from_folder(args.input)
    except StitchingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(images)} images. Stitching...")
    stitcher = build_stitcher(args)
    try:
        result = stitcher.stitch(images)
    except StitchingError as exc:
        print(f"Stitching failed: {exc}", file=sys.stderr)
        return 1

    for msg in result.messages:
        print(f"  - {msg}")

    ensure_dir(os.path.dirname(args.output) or ".")
    cv2.imwrite(args.output, result.panorama)
    print(f"Panorama saved to '{args.output}'.")

    if args.report:
        _write_report(result, args.blend, args.cylindrical, args.output)

    return 0


def run_batch(args: argparse.Namespace) -> int:
    """Run stitching for every sub-folder inside a batch input directory.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0 = all jobs succeeded, 1 = at least one failed).
    """
    subfolders = sorted(
        f for f in glob.glob(os.path.join(args.batch_input, "*")) if os.path.isdir(f)
    )
    if not subfolders:
        print(f"No sub-folders found in '{args.batch_input}'.", file=sys.stderr)
        return 1

    ensure_dir(args.output_dir)
    overall_ok = True

    for folder in subfolders:
        job_name = os.path.basename(os.path.normpath(folder))
        print(f"\n=== Processing job: {job_name} ===")
        try:
            images = load_images_from_folder(folder)
        except StitchingError as exc:
            print(f"  Skipping job '{job_name}': {exc}", file=sys.stderr)
            overall_ok = False
            continue

        stitcher = build_stitcher(args)
        try:
            result = stitcher.stitch(images)
        except StitchingError as exc:
            print(f"  Stitching failed for '{job_name}': {exc}", file=sys.stderr)
            overall_ok = False
            continue

        for msg in result.messages:
            print(f"  - {msg}")

        output_path = os.path.join(args.output_dir, f"{job_name}.png")
        cv2.imwrite(output_path, result.panorama)
        print(f"  Saved: {output_path}")

        if args.report:
            _write_report(result, args.blend, args.cylindrical, output_path)

    return 0 if overall_ok else 1


def _write_report(result, blend_method: str, projection_enabled: bool, output_path: str) -> None:
    """Compute the quality/ML report for a stitch job and save it as JSON.

    Args:
        result: The `StitchResult` returned by `PanoramaStitcher.stitch`.
        blend_method: Blending method used ("multiband" or "feather").
        projection_enabled: Whether cylindrical projection was applied.
        output_path: Path the panorama image was saved to; the report is
            written alongside it with a ".report.json" suffix.
    """
    predictor = SuccessPredictor()
    prediction = predictor.predict(result.metrics)
    report = build_quality_report(
        metrics=result.metrics,
        processing_time_sec=result.processing_time_sec,
        panorama_shape=result.panorama.shape[:2],
        blend_method=blend_method,
        projection_enabled=projection_enabled,
        ml_success_probability=prediction.success_probability,
        ml_expected_quality=prediction.expected_quality,
    )
    report_path = os.path.splitext(output_path)[0] + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"  Quality report saved to '{report_path}' "
          f"({report.stars_display()} {report.rating_label}).")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Stitch overlapping images into a panorama from the command line."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Folder containing images for a single panorama job.")
    group.add_argument(
        "--batch-input",
        help="Folder whose sub-folders are each treated as a separate panorama job.",
    )

    parser.add_argument("--output", default="outputs/panorama.png", help="Output file path (single mode).")
    parser.add_argument("--output-dir", default="outputs/", help="Output directory (batch mode).")

    parser.add_argument("--blend", choices=["multiband", "feather"], default="multiband")
    parser.add_argument("--cylindrical", action="store_true", help="Enable cylindrical projection.")
    parser.add_argument("--no-exposure-comp", action="store_true", help="Disable exposure compensation.")
    parser.add_argument("--no-auto-order", action="store_true", help="Disable automatic image ordering.")
    parser.add_argument("--max-dimension", type=int, default=1600, help="Max working resolution in px.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also save a JSON quality/ML stitching report next to each output image.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input:
        exit_code = run_single(args)
    else:
        exit_code = run_batch(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
