"""
stitcher.py
------------
Core orchestration of the panorama stitching pipeline: automatic image
ordering, homography estimation (RANSAC), canvas computation, warping,
exposure compensation, blending, and final cropping.

Author: Panorama Stitching Project
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from blending import blend_pair
from cylindrical_projection import warp_batch
from feature_matching import (
    FeatureData,
    FeatureExtractionError,
    FeatureMatcher,
    FeatureMatchingError,
)
from quality_metrics import PipelineMetrics
from utils import (
    apply_gain,
    compute_mean_brightness,
    compute_overlap_mean_ratio,
    compute_sharpness,
    crop_black_borders,
    resize_keep_aspect,
    save_debug_image,
)


class StitchingError(Exception):
    """Raised for any unrecoverable failure during the stitching pipeline."""


@dataclass
class StitchResult:
    """Container for everything produced by a stitching run.

    Attributes:
        panorama: Final, cropped panorama image (BGR).
        order: Original input indices in the order they were stitched.
        match_debug_images: Debug visualizations of feature matches for
            each adjacent pair in the stitching order (empty if debug mode
            is off).
        warped_images: The individual images after warping onto the shared
            canvas, before blending (useful for debugging/visualization).
        messages: Human-readable status/info messages describing what the
            pipeline did (useful for a UI status log).
    """

    panorama: np.ndarray
    order: List[int]
    match_debug_images: List[np.ndarray] = field(default_factory=list)
    warped_images: List[np.ndarray] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    metrics: Optional[PipelineMetrics] = None
    processing_time_sec: float = 0.0
    focal_length_used: Optional[float] = None
    filenames_in_order: List[str] = field(default_factory=list)

    def order_description(self) -> str:
        """Return a human-readable "Image A -> Image B -> ..." order string.

        Uses `filenames_in_order` if populated (set by the caller from the
        original upload names), falling back to 1-based original indices.

        Returns:
            Arrow-joined description of the detected stitching order.
        """
        if self.filenames_in_order:
            return " → ".join(self.filenames_in_order)
        return " → ".join(f"Image {i + 1}" for i in self.order)


class PanoramaStitcher:
    """End-to-end panorama stitching pipeline.

    Usage:
        stitcher = PanoramaStitcher(blend_method="multiband", debug=True)
        result = stitcher.stitch([img1, img2, img3])
    """

    def __init__(
        self,
        blend_method: str = "multiband",
        use_cylindrical: bool = False,
        exposure_compensation: bool = True,
        auto_order: bool = True,
        ratio_thresh: float = 0.75,
        min_match_count: int = 10,
        max_dimension: int = 1600,
        debug: bool = False,
    ):
        """Configure the stitching pipeline.

        Args:
            blend_method: "multiband" or "feather".
            use_cylindrical: If True, project each image onto a cylinder
                before feature matching/warping. Helps for wide, rotation-
                only panoramas; can hurt for small (2-3 image) horizontal
                shifts, so it is off by default.
            exposure_compensation: If True, apply a simple brightness-gain
                correction between overlapping images before blending.
            auto_order: If True, automatically determine the best left-to-
                right stitching order using pairwise match counts. If
                False, images are stitched in the order given.
            ratio_thresh: Lowe's ratio test threshold.
            min_match_count: Minimum good matches required between an
                adjacent pair to trust the homography.
            max_dimension: Images are downsized so neither side exceeds
                this, for speed.
            debug: If True, generate and keep match visualizations.
        """
        self.blend_method = blend_method
        self.use_cylindrical = use_cylindrical
        self.exposure_compensation = exposure_compensation
        self.auto_order = auto_order
        self.max_dimension = max_dimension
        self.debug = debug

        self.matcher = FeatureMatcher(
            ratio_thresh=ratio_thresh, min_match_count=min_match_count
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stitch(self, images: List[np.ndarray]) -> StitchResult:
        """Run the full stitching pipeline on a list of images.

        Args:
            images: List of BGR images (at least 2) with overlapping
                content of the same scene.

        Returns:
            StitchResult containing the final panorama and debug artifacts.

        Raises:
            StitchingError: If fewer than 2 images are given, if features
                cannot be extracted, or if the images do not overlap enough
                to be stitched.
        """
        start_time = time.time()
        messages: List[str] = []

        if images is None or len(images) < 2:
            raise StitchingError("At least two images are required to build a panorama.")

        # Step 1: pre-process (resize, optional cylindrical projection).
        processed = [resize_keep_aspect(img, self.max_dimension) for img in images]
        focal_length_used: Optional[float] = None
        if self.use_cylindrical:
            processed, focal_length_used = warp_batch(processed)
            messages.append(
                f"Applied cylindrical projection to all images "
                f"(focal length ≈ {focal_length_used:.0f}px)."
            )

        messages.append(f"Using {self.matcher.detector_name} for feature detection.")

        # Step 2: feature extraction.
        features: List[FeatureData] = []
        for i, img in enumerate(processed):
            try:
                features.append(self.matcher.detect_and_compute(img))
            except FeatureExtractionError as exc:
                raise StitchingError(f"Image {i + 1}: {exc}") from exc

        # Step 3: determine stitching order.
        if self.auto_order and len(processed) > 2:
            order = self._auto_order(features)
            messages.append(f"Automatically determined stitching order: {order}")
        else:
            order = list(range(len(processed)))

        # Step 4: pairwise homography estimation between adjacent images
        # (in stitching order), plus optional match visualizations.
        pairwise_homographies: Dict[Tuple[int, int], np.ndarray] = {}
        match_debug_images: List[np.ndarray] = []

        good_match_counts: List[int] = []
        inlier_ratios: List[float] = []
        inlier_counts: List[int] = []

        for a, b in zip(order[:-1], order[1:]):
            try:
                matches = self.matcher.match(features[a], features[b])
            except FeatureMatchingError as exc:
                raise StitchingError(
                    f"Images {a + 1} and {b + 1} do not appear to overlap: {exc}"
                ) from exc

            H, inlier_ratio = self._estimate_homography(
                features[a], features[b], matches
            )
            if H is None:
                raise StitchingError(
                    f"Could not compute a reliable homography between images "
                    f"{a + 1} and {b + 1} (too few inliers after RANSAC)."
                )
            pairwise_homographies[(a, b)] = H
            good_match_counts.append(len(matches))
            inlier_ratios.append(inlier_ratio)
            inlier_counts.append(int(round(inlier_ratio * len(matches))))
            messages.append(
                f"Images {a + 1}->{b + 1}: {len(matches)} good matches, "
                f"{inlier_ratio:.0%} RANSAC inliers."
            )

            if self.debug:
                debug_img = self.matcher.draw_matches(
                    processed[a], features[a], processed[b], features[b], matches
                )
                match_debug_images.append(debug_img)
                save_debug_image(debug_img, f"matches_{a}_{b}")

        # Step 5: chain homographies to a common reference frame.
        reference_pos = len(order) // 2
        homographies_to_ref = self._chain_homographies(order, pairwise_homographies, reference_pos)

        # Step 6: compute the shared canvas size and warp every image onto it.
        canvas_size, translation = self._compute_canvas(processed, order, homographies_to_ref)
        warped_images = []
        for pos, idx in enumerate(order):
            H_final = translation @ homographies_to_ref[idx]
            warped = cv2.warpPerspective(processed[idx], H_final, canvas_size)
            warped_images.append(warped)

        # Estimate physical overlap (as % of a single frame's area) between
        # each adjacent pair, now that both are on the shared canvas.
        overlap_percents: List[float] = []
        for pos in range(len(warped_images) - 1):
            overlap_percents.append(
                self._estimate_overlap_percent(warped_images[pos], warped_images[pos + 1])
            )

        # Step 7: sequential exposure compensation + blending.
        panorama = warped_images[0]
        for next_img in warped_images[1:]:
            if self.exposure_compensation:
                next_img = self._compensate_exposure(panorama, next_img)
            panorama = blend_pair(panorama, next_img, method=self.blend_method)

        # Step 8: crop black borders left by warping.
        panorama = crop_black_borders(panorama)
        messages.append("Cropped black borders from the final panorama.")

        # Step 9: assemble quality/ML pipeline metrics.
        keypoint_counts = [len(f.keypoints) for f in features]
        sharpness_scores = [compute_sharpness(img) for img in processed]
        brightness_values = [compute_mean_brightness(img) for img in processed]
        brightness_diff = float(max(brightness_values) - min(brightness_values)) if brightness_values else 0.0

        metrics = PipelineMetrics(
            num_images=len(images),
            detector_name=self.matcher.detector_name,
            avg_keypoints=float(np.mean(keypoint_counts)) if keypoint_counts else 0.0,
            avg_good_matches=float(np.mean(good_match_counts)) if good_match_counts else 0.0,
            avg_inlier_ratio=float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
            total_ransac_inliers=int(sum(inlier_counts)),
            overlap_percent=float(np.mean(overlap_percents)) if overlap_percents else 0.0,
            sharpness_score=float(np.mean(sharpness_scores)) if sharpness_scores else 0.0,
            brightness_diff=brightness_diff,
        )

        processing_time = time.time() - start_time
        messages.append(f"Total processing time: {processing_time:.2f}s.")

        return StitchResult(
            panorama=panorama,
            order=order,
            match_debug_images=match_debug_images,
            warped_images=warped_images,
            messages=messages,
            metrics=metrics,
            processing_time_sec=processing_time,
            focal_length_used=focal_length_used,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _auto_order(self, features: List[FeatureData]) -> List[int]:
        """Determine a good left-to-right stitching order automatically.

        Builds a pairwise "overlap strength" matrix (number of good
        matches between every pair of images) and greedily grows a chain
        by always attaching the unused image with the strongest
        connection to either end of the current chain.

        Args:
            features: Detected features for every input image.

        Returns:
            List of original image indices in stitching order.
        """
        n = len(features)
        match_counts = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                count = self.matcher.count_good_matches(features[i], features[j])
                match_counts[i, j] = count
                match_counts[j, i] = count

        # Seed the chain with the strongest pair overall.
        i, j = np.unravel_index(np.argmax(match_counts), match_counts.shape)
        chain = [int(i), int(j)]
        used = set(chain)

        while len(used) < n:
            best_score = -1
            best_candidate = None
            best_side = None  # "front" or "back"

            for k in range(n):
                if k in used:
                    continue
                front_score = match_counts[chain[0], k]
                back_score = match_counts[chain[-1], k]
                if front_score > best_score:
                    best_score, best_candidate, best_side = front_score, k, "front"
                if back_score > best_score:
                    best_score, best_candidate, best_side = back_score, k, "back"

            if best_candidate is None or best_score <= 0:
                # No remaining connections found; append any leftover image
                # at the end rather than silently dropping it.
                leftover = next(k for k in range(n) if k not in used)
                chain.append(leftover)
                used.add(leftover)
                continue

            if best_side == "front":
                chain.insert(0, best_candidate)
            else:
                chain.append(best_candidate)
            used.add(best_candidate)

        return chain

    def _estimate_homography(
        self,
        features_a: FeatureData,
        features_b: FeatureData,
        matches: List[cv2.DMatch],
        ransac_reproj_thresh: float = 4.0,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Estimate the homography mapping image A's plane to image B's plane.

        Args:
            features_a: Features of the source image.
            features_b: Features of the destination image.
            matches: Good matches (query = A, train = B).
            ransac_reproj_thresh: RANSAC re-projection error threshold (px).

        Returns:
            Tuple of (homography matrix or None, inlier ratio in [0, 1]).
        """
        src_pts = np.float32(
            [features_a.keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [features_b.keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh
        )

        if H is None or mask is None:
            return None, 0.0

        inlier_ratio = float(mask.sum()) / float(len(mask))
        return H, inlier_ratio

    def _chain_homographies(
        self,
        order: List[int],
        pairwise: Dict[Tuple[int, int], np.ndarray],
        reference_pos: int,
    ) -> Dict[int, np.ndarray]:
        """Compose adjacent-pair homographies into a single reference frame.

        Args:
            order: Original image indices in stitching order.
            pairwise: Mapping (a, b) -> homography from image a's plane to
                image b's plane, for adjacent pairs in `order`.
            reference_pos: Position within `order` chosen as the reference
                (identity) frame -- typically the middle image, which
                minimizes cumulative distortion.

        Returns:
            Mapping from original image index -> 3x3 homography that maps
            that image's plane into the reference image's plane.
        """
        n = len(order)
        homographies: Dict[int, np.ndarray] = {order[reference_pos]: np.eye(3)}

        # Images to the left of the reference: compose forward.
        for pos in range(reference_pos - 1, -1, -1):
            a, b = order[pos], order[pos + 1]
            H_a_to_b = pairwise[(a, b)]
            homographies[a] = homographies[b] @ H_a_to_b

        # Images to the right of the reference: compose with inverses.
        for pos in range(reference_pos + 1, n):
            a, b = order[pos - 1], order[pos]
            H_a_to_b = pairwise[(a, b)]
            H_b_to_a = np.linalg.inv(H_a_to_b)
            homographies[b] = homographies[a] @ H_b_to_a

        return homographies

    def _compute_canvas(
        self,
        images: List[np.ndarray],
        order: List[int],
        homographies: Dict[int, np.ndarray],
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """Compute the output canvas size and a translation to keep it positive.

        Args:
            images: Pre-processed input images (indexed by original index).
            order: Stitching order (original indices).
            homographies: Per-image homography into the reference frame.

        Returns:
            Tuple of ((canvas_width, canvas_height), translation_matrix).
        """
        all_corners = []
        for idx in order:
            h, w = images[idx].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, homographies[idx])
            all_corners.append(transformed)

        all_corners = np.concatenate(all_corners, axis=0)
        x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

        translation = np.array(
            [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64
        )
        canvas_size = (x_max - x_min, y_max - y_min)
        return canvas_size, translation

    def _estimate_overlap_percent(self, canvas_a: np.ndarray, canvas_b: np.ndarray) -> float:
        """Estimate the overlap between two warped images as a percentage.

        Computed as the shared (non-black) content area between the two
        canvases divided by the average single-frame content area, giving
        an intuitive "roughly X% of each photo overlaps" figure for the
        quality dashboard and ML feature vector.

        Args:
            canvas_a: First image already warped onto the shared canvas.
            canvas_b: Second image already warped onto the shared canvas.

        Returns:
            Estimated overlap percentage in [0, 100].
        """
        gray_a = cv2.cvtColor(canvas_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(canvas_b, cv2.COLOR_BGR2GRAY)
        mask_a = (gray_a > 1).astype(np.uint8)
        mask_b = (gray_b > 1).astype(np.uint8)

        area_a = int(mask_a.sum())
        area_b = int(mask_b.sum())
        overlap_area = int(np.logical_and(mask_a, mask_b).sum())

        avg_area = (area_a + area_b) / 2.0
        if avg_area <= 0:
            return 0.0
        return float(np.clip((overlap_area / avg_area) * 100.0, 0.0, 100.0))

    def _compensate_exposure(self, base_canvas: np.ndarray, new_canvas: np.ndarray) -> np.ndarray:
        """Apply a simple brightness-gain correction to `new_canvas`.

        Args:
            base_canvas: The panorama built so far.
            new_canvas: The next warped image to be merged in.

        Returns:
            Gain-adjusted version of `new_canvas`.
        """
        gray_base = cv2.cvtColor(base_canvas, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY)
        mask_base = (gray_base > 1).astype(np.uint8) * 255
        mask_new = (gray_new > 1).astype(np.uint8) * 255

        gain = compute_overlap_mean_ratio(base_canvas, new_canvas, mask_base, mask_new)
        if abs(gain - 1.0) < 1e-3:
            return new_canvas
        return apply_gain(new_canvas, gain)
