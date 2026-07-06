"""
quality_metrics.py
-------------------
Post-stitch (and pre-stitch) quality measurement for the panorama
pipeline: keypoint/match statistics, RANSAC inlier ratios, overlap and
sharpness estimates, a single weighted "Confidence Score", and a
human-friendly star rating — all surfaced in the Streamlit quality
dashboard and exportable as a JSON stitching report.

Author: Panorama Stitching Project
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Weights for the weighted confidence score, per the project spec:
# 40% inlier ratio, 30% match count, 20% overlap, 10% sharpness.
WEIGHT_INLIER_RATIO = 0.40
WEIGHT_MATCH_COUNT = 0.30
WEIGHT_OVERLAP = 0.20
WEIGHT_SHARPNESS = 0.10

# Normalization caps used to map raw metrics onto a 0-100 scale before
# weighting. These are heuristic but reasonable for typical smartphone
# photos; tune here if results consistently feel too harsh/lenient.
MATCH_COUNT_NORMALIZATION_CAP = 200.0   # avg good matches considered "excellent" at/above this
SHARPNESS_NORMALIZATION_CAP = 500.0     # variance-of-Laplacian considered "sharp" at/above this

# Confidence score thresholds -> star rating / label.
RATING_THRESHOLDS: List[Tuple[float, int, str]] = [
    (90.0, 5, "Excellent"),
    (75.0, 4, "Good"),
    (55.0, 3, "Average"),
    (35.0, 2, "Poor"),
    (0.0, 1, "Failed"),
]


@dataclass
class PipelineMetrics:
    """Raw measurements gathered while running the stitching pipeline.

    These are the inputs to both the quality dashboard and the ML
    success predictor, so they are computed once and shared by both.
    """

    num_images: int
    detector_name: str
    avg_keypoints: float
    avg_good_matches: float
    avg_inlier_ratio: float
    total_ransac_inliers: int
    overlap_percent: float
    sharpness_score: float
    brightness_diff: float

    def as_feature_vector(self) -> List[float]:
        """Return the metrics as a fixed-order feature vector for the ML model.

        Order: [num_images, avg_keypoints, avg_good_matches,
        overlap_percent, sharpness_score, brightness_diff, avg_inlier_ratio]
        """
        return [
            float(self.num_images),
            float(self.avg_keypoints),
            float(self.avg_good_matches),
            float(self.overlap_percent),
            float(self.sharpness_score),
            float(self.brightness_diff),
            float(self.avg_inlier_ratio),
        ]


@dataclass
class QualityReport:
    """Complete quality dashboard payload for one stitching run."""

    metrics: PipelineMetrics
    processing_time_sec: float
    panorama_width: int
    panorama_height: int
    blend_method: str
    projection_enabled: bool
    confidence_score: float
    star_count: int
    rating_label: str
    ml_success_probability: Optional[float] = None
    ml_expected_quality: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def stars_display(self) -> str:
        """Return the star rating as a filled/empty star string, e.g. '★★★★☆'."""
        return "★" * self.star_count + "☆" * (5 - self.star_count)

    def to_dict(self) -> dict:
        """Serialize the full report to a plain dict (JSON-friendly)."""
        data = asdict(self)
        data["stars_display"] = self.stars_display()
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full report to a JSON string for the download button."""
        return json.dumps(self.to_dict(), indent=indent)


def _normalize(value: float, cap: float) -> float:
    """Clip-and-scale a raw metric to the 0-100 range using a fixed cap.

    Args:
        value: Raw metric value (assumed >= 0).
        cap: Value at/above which the normalized score saturates at 100.

    Returns:
        Normalized score in [0, 100].
    """
    if cap <= 0:
        return 0.0
    return float(np.clip((value / cap) * 100.0, 0.0, 100.0))


def compute_confidence_score(metrics: PipelineMetrics) -> float:
    """Compute the single weighted Confidence Score (0-100) for a stitch run.

    Weighting (per spec):
        40% inlier ratio
        30% number of matches
        20% overlap
        10% image sharpness

    Args:
        metrics: Raw pipeline metrics gathered during stitching.

    Returns:
        Confidence score in [0, 100].
    """
    inlier_score = np.clip(metrics.avg_inlier_ratio, 0.0, 1.0) * 100.0
    match_score = _normalize(metrics.avg_good_matches, MATCH_COUNT_NORMALIZATION_CAP)
    overlap_score = np.clip(metrics.overlap_percent, 0.0, 100.0)
    sharpness_score = _normalize(metrics.sharpness_score, SHARPNESS_NORMALIZATION_CAP)

    score = (
        WEIGHT_INLIER_RATIO * inlier_score
        + WEIGHT_MATCH_COUNT * match_score
        + WEIGHT_OVERLAP * overlap_score
        + WEIGHT_SHARPNESS * sharpness_score
    )
    return float(np.clip(score, 0.0, 100.0))


def star_rating(confidence_score: float) -> Tuple[int, str]:
    """Map a confidence score onto a (star_count, label) rating.

    Args:
        confidence_score: Score in [0, 100].

    Returns:
        Tuple of (number of filled stars 1-5, human-readable label).
    """
    for threshold, stars, label in RATING_THRESHOLDS:
        if confidence_score >= threshold:
            return stars, label
    return 1, "Failed"


def build_quality_report(
    metrics: PipelineMetrics,
    processing_time_sec: float,
    panorama_shape: Tuple[int, int],
    blend_method: str,
    projection_enabled: bool,
    ml_success_probability: Optional[float] = None,
    ml_expected_quality: Optional[str] = None,
) -> QualityReport:
    """Assemble the full quality dashboard report for one stitching run.

    Args:
        metrics: Raw pipeline metrics.
        processing_time_sec: Wall-clock seconds the pipeline took.
        panorama_shape: (height, width) of the final panorama.
        blend_method: "multiband" or "feather".
        projection_enabled: Whether cylindrical projection was used.
        ml_success_probability: Optional ML-predicted success probability.
        ml_expected_quality: Optional ML-predicted quality label.

    Returns:
        Populated QualityReport.
    """
    confidence = compute_confidence_score(metrics)
    stars, label = star_rating(confidence)

    warnings: List[str] = []
    if metrics.avg_inlier_ratio < 0.3:
        warnings.append(
            "Low RANSAC inlier ratio — the estimated alignment between "
            "images may be unreliable."
        )
    if metrics.overlap_percent < 15:
        warnings.append(
            "Low estimated overlap between images — make sure adjacent "
            "photos share at least 20-30% of the same scene."
        )
    if metrics.sharpness_score < 50:
        warnings.append(
            "Some images appear blurry, which can reduce match quality."
        )
    if metrics.brightness_diff > 40:
        warnings.append(
            "Large brightness differences detected between images; "
            "exposure compensation will attempt to correct this."
        )

    height, width = panorama_shape
    return QualityReport(
        metrics=metrics,
        processing_time_sec=processing_time_sec,
        panorama_width=width,
        panorama_height=height,
        blend_method=blend_method,
        projection_enabled=projection_enabled,
        confidence_score=confidence,
        star_count=stars,
        rating_label=label,
        ml_success_probability=ml_success_probability,
        ml_expected_quality=ml_expected_quality,
        warnings=warnings,
    )
