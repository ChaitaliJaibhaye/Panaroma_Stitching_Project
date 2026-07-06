"""
ml_predictor.py
----------------
Lightweight Machine Learning pipeline that predicts whether a panorama
stitch is likely to succeed *before* committing to the (potentially
slow) full warp + blend, based on cheap-to-compute pre-stitch features.

Since no labeled real-world dataset of "successful vs failed stitches"
is available, a synthetic dataset is generated from a set of domain
heuristics (more matches / higher inlier ratio / more overlap / sharper,
better-exposed images => more likely to succeed), with noise added so
the classifier has to learn a real decision boundary rather than a
trivial threshold. This mirrors a common, legitimate bootstrapping
strategy for early-stage ML products that lack labeled data yet.

The trained model is cached to disk (`models/panorama_success_model.pkl`)
so training only happens once per environment.


"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if sklearn is missing
    SKLEARN_AVAILABLE = False

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "panorama_success_model.pkl")

FEATURE_NAMES = [
    "num_images",
    "avg_keypoints",
    "avg_good_matches",
    "overlap_percent",
    "sharpness_score",
    "brightness_diff",
    "avg_inlier_ratio",
]

RANDOM_SEED = 42


@dataclass
class PredictionResult:
    """Result of an ML success prediction for one stitch job."""

    success_probability: float
    expected_quality: str
    low_confidence_warning: Optional[str] = None


def _quality_label_from_probability(probability: float) -> str:
    """Map a success probability onto a human-readable expected-quality label.

    Args:
        probability: Predicted probability of a successful stitch, in [0, 1].

    Returns:
        One of "Excellent", "Good", "Average", "Poor", "Failed".
    """
    pct = probability * 100.0
    if pct >= 90:
        return "Excellent"
    if pct >= 75:
        return "Good"
    if pct >= 55:
        return "Average"
    if pct >= 35:
        return "Poor"
    return "Failed"


def _generate_synthetic_dataset(n_samples: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (features, labels) dataset for training.

    Each sample's features are drawn from plausible ranges for the 7
    pre-stitch metrics, and the label ("1" = successful stitch) is
    assigned by a noisy scoring rule that rewards more matches, higher
    inlier ratios, more overlap, sharper images, small brightness
    differences, and (mildly) fewer images (more images -> more chances
    for one weak link to break the chain).

    Args:
        n_samples: Number of synthetic training samples to generate.

    Returns:
        Tuple of (feature matrix of shape (n_samples, 7), binary label vector).
    """
    rng = np.random.default_rng(RANDOM_SEED)

    num_images = rng.integers(2, 11, size=n_samples).astype(float)
    avg_keypoints = rng.uniform(50, 6000, size=n_samples)
    avg_good_matches = rng.uniform(0, 400, size=n_samples)
    overlap_percent = rng.uniform(0, 80, size=n_samples)
    sharpness_score = rng.uniform(0, 1200, size=n_samples)
    brightness_diff = rng.uniform(0, 100, size=n_samples)
    avg_inlier_ratio = rng.uniform(0, 1, size=n_samples)

    # Normalize each factor to roughly [0, 1] for a weighted heuristic score.
    match_norm = np.clip(avg_good_matches / 150.0, 0, 1)
    overlap_norm = np.clip(overlap_percent / 40.0, 0, 1)
    sharpness_norm = np.clip(sharpness_score / 400.0, 0, 1)
    brightness_penalty = np.clip(brightness_diff / 60.0, 0, 1)
    image_count_penalty = np.clip((num_images - 2) / 8.0, 0, 1) * 0.15

    heuristic_score = (
        0.40 * avg_inlier_ratio
        + 0.25 * match_norm
        + 0.20 * overlap_norm
        + 0.10 * sharpness_norm
        - 0.15 * brightness_penalty
        - image_count_penalty
    )

    noise = rng.normal(0, 0.12, size=n_samples)
    noisy_score = heuristic_score + noise

    labels = (noisy_score >= 0.45).astype(int)

    features = np.column_stack(
        [
            num_images,
            avg_keypoints,
            avg_good_matches,
            overlap_percent,
            sharpness_score,
            brightness_diff,
            avg_inlier_ratio,
        ]
    )
    return features, labels


class SuccessPredictor:
    """Predicts panorama stitch success probability from pre-stitch metrics.

    Usage:
        predictor = SuccessPredictor()
        result = predictor.predict(metrics)  # metrics: quality_metrics.PipelineMetrics
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """Load a cached model from disk, training and saving one if absent.

        Args:
            model_path: Path to the joblib-serialized RandomForestClassifier.
        """
        self.model_path = model_path
        self.model = None
        self.available = SKLEARN_AVAILABLE

        if not self.available:
            return

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = self._train_and_save()

    def _train_and_save(self):
        """Train a RandomForestClassifier on synthetic data and persist it.

        Returns:
            The trained ``RandomForestClassifier``.
        """
        features, labels = _generate_synthetic_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        test_accuracy = model.score(X_test, y_test)
        model.training_accuracy_ = float(model.score(X_train, y_train))
        model.holdout_accuracy_ = float(test_accuracy)

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(model, self.model_path)
        return model

    def predict(self, metrics) -> PredictionResult:
        """Predict success probability and expected quality for a stitch job.

        Args:
            metrics: A `quality_metrics.PipelineMetrics` instance (or any
                object exposing `.as_feature_vector()` in the same 7-value
                order used during training).

        Returns:
            PredictionResult with probability, expected quality label, and
            an optional low-confidence warning message.
        """
        if not self.available or self.model is None:
            return PredictionResult(
                success_probability=float("nan"),
                expected_quality="Unavailable",
                low_confidence_warning=(
                    "scikit-learn is not installed, so the ML success "
                    "predictor is disabled. Install scikit-learn and "
                    "joblib to enable it."
                ),
            )

        feature_vector = np.array([metrics.as_feature_vector()])
        probability = float(self.model.predict_proba(feature_vector)[0, 1])
        expected_quality = _quality_label_from_probability(probability)

        warning = None
        if probability < 0.55:
            warning = (
                f"Predicted success probability is only {probability:.0%}. "
                "Consider adding more overlap between photos, retaking "
                "blurry shots, or reducing brightness differences before "
                "stitching."
            )

        return PredictionResult(
            success_probability=probability,
            expected_quality=expected_quality,
            low_confidence_warning=warning,
        )


def get_predictor() -> SuccessPredictor:
    """Convenience factory returning a ready-to-use SuccessPredictor.

    Returns:
        A `SuccessPredictor` instance (trains + caches the model on first call).
    """
    return SuccessPredictor()
