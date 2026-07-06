"""
feature_matching.py
--------------------
Feature detection and matching utilities for panorama stitching.

Implements keypoint/descriptor extraction (SIFT preferred, ORB fallback)
and descriptor matching (FLANN for SIFT's float descriptors, BFMatcher
with Hamming distance for ORB's binary descriptors), followed by Lowe's
ratio test to discard ambiguous matches.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


class FeatureExtractionError(Exception):
    """Raised when no usable features can be extracted from an image."""


class FeatureMatchingError(Exception):
    """Raised when two images do not share enough reliable matches."""


@dataclass
class FeatureData:
    """Container for an image's detected keypoints and descriptors."""

    keypoints: Tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


class FeatureMatcher:
    """Detects and matches local features between images.

    Attempts to use SIFT (scale-invariant, high quality) and transparently
    falls back to ORB (fast, patent-free, binary descriptors) if SIFT is
    not available in the installed OpenCV build.

    Attributes:
        detector_name: Name of the detector actually in use ("SIFT" or "ORB").
        ratio_thresh: Threshold used for Lowe's ratio test.
        min_match_count: Minimum number of good matches required to trust
            a pair of images as overlapping.
    """

    def __init__(self, ratio_thresh: float = 0.75, min_match_count: int = 10, n_features: int = 4000):
        """Initialize the feature detector and matcher.

        Args:
            ratio_thresh: Lowe's ratio test threshold (lower = stricter).
            min_match_count: Minimum number of good matches to consider two
                images as a valid, overlapping pair.
            n_features: Maximum number of features to detect per image
                (relevant mainly for ORB).
        """
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count

        if hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.detector_name = "SIFT"
            self._is_binary_descriptor = False
        else:
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.detector_name = "ORB"
            self._is_binary_descriptor = True

        self.matcher = self._build_matcher()

    def _build_matcher(self):
        """Construct the appropriate matcher for the active descriptor type.

        Returns:
            A configured cv2.FlannBasedMatcher or cv2.BFMatcher instance.
        """
        if self._is_binary_descriptor:
            # ORB descriptors are binary -> Hamming distance, BFMatcher.
            return cv2.BFMatcher(cv2.NORM_HAMMING)

        # SIFT descriptors are float -> FLANN with KD-tree works well.
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def detect_and_compute(self, image: np.ndarray) -> FeatureData:
        """Detect keypoints and compute descriptors for a single image.

        Args:
            image: BGR input image.

        Returns:
            FeatureData with keypoints and descriptors.

        Raises:
            FeatureExtractionError: If no keypoints/descriptors are found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            raise FeatureExtractionError(
                "No distinctive features could be detected in one of the "
                "images. Try a higher-resolution or more textured image."
            )

        return FeatureData(keypoints=tuple(keypoints), descriptors=descriptors)

    def match(self, features_a: FeatureData, features_b: FeatureData) -> List[cv2.DMatch]:
        """Match descriptors between two images and apply Lowe's ratio test.

        Args:
            features_a: Features of the first (query) image.
            features_b: Features of the second (train) image.

        Returns:
            List of "good" matches that passed the ratio test, sorted by
            distance (best first).

        Raises:
            FeatureMatchingError: If fewer than `min_match_count` good
                matches remain after filtering.
        """
        if self._is_binary_descriptor:
            desc_a = features_a.descriptors
            desc_b = features_b.descriptors
        else:
            desc_a = features_a.descriptors.astype(np.float32)
            desc_b = features_b.descriptors.astype(np.float32)

        if len(desc_a) < 2 or len(desc_b) < 2:
            raise FeatureMatchingError("Not enough descriptors to perform matching.")

        raw_matches = self.matcher.knnMatch(desc_a, desc_b, k=2)

        good_matches: List[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        good_matches.sort(key=lambda m: m.distance)

        if len(good_matches) < self.min_match_count:
            raise FeatureMatchingError(
                f"Only {len(good_matches)} reliable matches found between "
                f"the image pair (need at least {self.min_match_count}). "
                "The images may not overlap enough."
            )

        return good_matches

    def draw_matches(
        self,
        image_a: np.ndarray,
        features_a: FeatureData,
        image_b: np.ndarray,
        features_b: FeatureData,
        matches: List[cv2.DMatch],
        max_matches: int = 60,
    ) -> np.ndarray:
        """Render a visualization of matched keypoints between two images.

        Args:
            image_a: First BGR image.
            features_a: Detected features of image_a.
            image_b: Second BGR image.
            features_b: Detected features of image_b.
            matches: Good matches to draw (already ratio-tested).
            max_matches: Cap on number of matches drawn, for clarity.

        Returns:
            BGR image showing both source images with lines connecting
            matched keypoints.
        """
        display_matches = matches[:max_matches]
        return cv2.drawMatches(
            image_a,
            features_a.keypoints,
            image_b,
            features_b.keypoints,
            display_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    def count_good_matches(self, features_a: FeatureData, features_b: FeatureData) -> int:
        """Count good matches between two feature sets without raising.

        Useful for building an overlap graph when auto-ordering images.

        Args:
            features_a: Features of the first image.
            features_b: Features of the second image.

        Returns:
            Number of good (ratio-tested) matches, or 0 if matching fails.
        """
        try:
            return len(self.match(features_a, features_b))
        except FeatureMatchingError:
            return 0
