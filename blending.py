"""
blending.py
------------
Blending strategies used to combine warped images on a shared panorama
canvas without visible seams: feather (distance-weighted) blending and
multi-band (Laplacian pyramid) blending.

Author: Panorama Stitching Project
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """Build a binary content mask (255 where pixel is non-black).

    Args:
        image: BGR image on a (possibly larger) canvas.

    Returns:
        uint8 mask, same height/width as image, values in {0, 255}.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return mask


def feather_blend(canvas_a: np.ndarray, canvas_b: np.ndarray) -> np.ndarray:
    """Blend two same-sized canvases using distance-transform feathering.

    Each pixel's blend weight is proportional to its distance from the
    edge of its own image's content region, so the seam between images
    fades smoothly rather than cutting sharply.

    Args:
        canvas_a: First image already warped onto the full panorama canvas
            (black where it has no content).
        canvas_b: Second image already warped onto the same canvas.

    Returns:
        Blended BGR image on the shared canvas.
    """
    mask_a = _binary_mask(canvas_a)
    mask_b = _binary_mask(canvas_b)

    # Distance transform: how far each content pixel is from the nearest
    # black (empty) pixel -> larger distance = more confident/central pixel.
    dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5).astype(np.float32)
    dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5).astype(np.float32)

    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)
    overlap = (mask_a > 0) & (mask_b > 0)

    weight_a = np.zeros(dist_a.shape, dtype=np.float32)
    weight_b = np.zeros(dist_b.shape, dtype=np.float32)

    weight_a[only_a] = 1.0
    weight_b[only_b] = 1.0

    total = dist_a[overlap] + dist_b[overlap]
    total[total == 0] = 1e-6
    weight_a[overlap] = dist_a[overlap] / total
    weight_b[overlap] = dist_b[overlap] / total

    weight_a_3 = cv2.merge([weight_a, weight_a, weight_a])
    weight_b_3 = cv2.merge([weight_b, weight_b, weight_b])

    blended = (
        canvas_a.astype(np.float32) * weight_a_3
        + canvas_b.astype(np.float32) * weight_b_3
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build a Gaussian pyramid for an image.

    Args:
        image: Source image (float32 recommended for blending math).
        levels: Number of pyramid levels to build (including the base).

    Returns:
        List of images from full resolution (index 0) to coarsest.
    """
    pyramid = [image]
    current = image
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """Build a Laplacian pyramid from a precomputed Gaussian pyramid.

    Args:
        gaussian_pyramid: Gaussian pyramid, full-res first.

    Returns:
        Laplacian pyramid, same length as input, coarsest level equals the
        smallest Gaussian level.
    """
    levels = len(gaussian_pyramid)
    laplacian_pyramid = []
    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = gaussian_pyramid[i].astype(np.float32) - expanded.astype(np.float32)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def multiband_blend(
    canvas_a: np.ndarray,
    canvas_b: np.ndarray,
    num_levels: int = 5,
) -> np.ndarray:
    """Blend two same-sized canvases using multi-band (Laplacian pyramid) blending.

    Multi-band blending combines images at multiple frequency bands: low
    frequencies (overall color/lighting) are blended smoothly over a wide
    transition, while high frequencies (detail/texture) are blended over a
    narrower transition. This avoids both hard seams and ghosting/blur.

    Args:
        canvas_a: First warped image on the shared panorama canvas.
        canvas_b: Second warped image on the shared panorama canvas.
        num_levels: Number of pyramid levels to use.

    Returns:
        Blended BGR image (uint8) on the shared canvas.
    """
    mask_a = _binary_mask(canvas_a)
    mask_b = _binary_mask(canvas_b)

    overlap = (mask_a > 0) & (mask_b > 0)
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)

    # Build a smooth 0..1 blend mask: 1 favors image A, 0 favors image B.
    # Inside the overlap, use the distance-transform ratio (same idea as
    # feather blending) so the transition follows the true overlap shape.
    dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5).astype(np.float32)
    dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5).astype(np.float32)

    blend_mask = np.zeros(mask_a.shape, dtype=np.float32)
    blend_mask[only_a] = 1.0
    blend_mask[only_b] = 0.0
    denom = dist_a[overlap] + dist_b[overlap]
    denom[denom == 0] = 1e-6
    blend_mask[overlap] = dist_a[overlap] / denom

    # Limit pyramid depth so it never exceeds what the image size supports.
    min_dim = min(canvas_a.shape[0], canvas_a.shape[1])
    max_levels = int(np.floor(np.log2(max(min_dim, 2))))
    levels = max(1, min(num_levels, max_levels))

    gp_a = _build_gaussian_pyramid(canvas_a.astype(np.float32), levels)
    gp_b = _build_gaussian_pyramid(canvas_b.astype(np.float32), levels)
    gp_mask = _build_gaussian_pyramid(blend_mask, levels)

    lp_a = _build_laplacian_pyramid(gp_a)
    lp_b = _build_laplacian_pyramid(gp_b)

    blended_pyramid = []
    for la, lb, gm in zip(lp_a, lp_b, gp_mask):
        gm_3 = cv2.merge([gm, gm, gm]) if la.ndim == 3 else gm
        blended_level = la * gm_3 + lb * (1.0 - gm_3)
        blended_pyramid.append(blended_level)

    # Collapse the pyramid back into a single image.
    result = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = result + blended_pyramid[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def blend_pair(canvas_a: np.ndarray, canvas_b: np.ndarray, method: str = "multiband") -> np.ndarray:
    """Blend two panorama-canvas images using the requested method.

    Args:
        canvas_a: First warped image on the shared canvas.
        canvas_b: Second warped image on the shared canvas.
        method: Either "multiband" or "feather".

    Returns:
        Blended BGR image.

    Raises:
        ValueError: If an unknown method is requested.
    """
    mask_a = _binary_mask(canvas_a)
    mask_b = _binary_mask(canvas_b)

    # Fast paths: no need to blend where only one image has content.
    if cv2.countNonZero(mask_a) == 0:
        return canvas_b
    if cv2.countNonZero(mask_b) == 0:
        return canvas_a

    if method == "feather":
        return feather_blend(canvas_a, canvas_b)
    if method == "multiband":
        return multiband_blend(canvas_a, canvas_b)

    raise ValueError(f"Unknown blending method: '{method}'")
