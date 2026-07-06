"""
cylindrical_projection.py
--------------------------
Cylindrical projection pre-processing step.

Projecting each input photo onto a cylindrical surface (as if the scene
were photographed from inside a cylinder) removes most of the
perspective distortion that accumulates when stitching many images shot
by rotating a camera around its optical center. This makes the
downstream homography estimates closer to pure translations, which in
turn produces straighter, more natural-looking wide panoramas.

This module was factored out of `utils.py` into its own file so the
projection step can be developed/tested/tuned independently, and so it
is obvious where to look when experimenting with focal-length estimation
or alternative projection models (e.g. spherical) in the future.

`utils.cylindrical_warp` remains as a thin backward-compatible wrapper
around `cylindrical_warp_image` for any external code still importing
it from there.

"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def estimate_focal_length(image: np.ndarray, assumed_fov_degrees: float = 60.0) -> float:
    """Estimate a reasonable focal length (in pixels) for cylindrical warping.

    True focal length normally comes from camera EXIF metadata or prior
    camera calibration, neither of which is reliably available for
    arbitrary uploaded photos. As a practical fallback, this estimates the
    focal length from the image width and an assumed horizontal field of
    view, which is a standard heuristic for smartphone-style photos and
    keeps the cylindrical warp visually stable.

    Args:
        image: Input BGR image.
        assumed_fov_degrees: Assumed horizontal field of view of the
            camera, in degrees. ~60 degrees is a reasonable default for
            typical smartphone main cameras.

    Returns:
        Estimated focal length in pixels. Falls back to the image width
        if estimation is not possible (e.g. degenerate FOV value).
    """
    height, width = image.shape[:2]
    try:
        fov_radians = np.deg2rad(assumed_fov_degrees)
        focal_length = (width / 2.0) / np.tan(fov_radians / 2.0)
        if not np.isfinite(focal_length) or focal_length <= 0:
            raise ValueError("Non-finite or non-positive focal length estimate.")
        return float(focal_length)
    except (ValueError, ZeroDivisionError):
        # Reasonable default: focal length approximately equal to image width.
        return float(width)


def cylindrical_warp_image(
    image: np.ndarray,
    focal_length: Optional[float] = None,
    crop_borders: bool = True,
) -> np.ndarray:
    """Project a planar image onto a cylindrical surface.

    Implements the standard cylindrical-coordinates re-projection used in
    classic panorama pipelines (e.g. Szeliski's "Image Alignment and
    Stitching"): for every destination pixel on the cylinder, the
    corresponding source pixel in the original planar image is looked up
    and copied via `cv2.remap`.

    Args:
        image: Input BGR image (planar / perspective photo).
        focal_length: Focal length in pixels. If None, it is estimated
            automatically via `estimate_focal_length`; if that estimation
            fails for any reason, the image width is used as a safe
            default.
        crop_borders: If True, crop the black borders left by the warp
            using `utils.crop_black_borders` (imported lazily to avoid a
            circular import).

    Returns:
        Cylindrically-warped BGR image.
    """
    if focal_length is None:
        focal_length = estimate_focal_length(image)
    if not focal_length or focal_length <= 0 or not np.isfinite(focal_length):
        focal_length = float(image.shape[1])  # width as last-resort default

    height, width = image.shape[:2]
    x_center = width / 2.0
    y_center = height / 2.0

    y_idx, x_idx = np.indices((height, width))

    theta = (x_idx - x_center) / focal_length
    h = (y_idx - y_center) / focal_length

    x_hat = np.sin(theta)
    y_hat = h
    z_hat = np.cos(theta)

    # Avoid division by (near) zero at extreme angles.
    z_hat = np.where(np.abs(z_hat) < 1e-6, 1e-6, z_hat)

    map_x = (focal_length * x_hat / z_hat + x_center).astype(np.float32)
    map_y = (focal_length * y_hat / z_hat + y_center).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if crop_borders:
        from utils import crop_black_borders  # local import avoids circular import

        warped = crop_black_borders(warped)

    return warped


# Backward/forward-compatible alias matching the exact name requested in
# the project spec (camelCase), delegating to the canonical snake_case
# implementation above.
def cylindricalWarpImage(image: np.ndarray, focal_length: Optional[float] = None) -> np.ndarray:  # noqa: N802
    """Alias for `cylindrical_warp_image` (camelCase, per spec naming).

    Args:
        image: Input BGR image.
        focal_length: Optional focal length in pixels; auto-estimated if omitted.

    Returns:
        Cylindrically-warped BGR image.
    """
    return cylindrical_warp_image(image, focal_length=focal_length)


def warp_batch(images: list, focal_length: Optional[float] = None):
    """Cylindrically warp a batch of images using one shared focal length.

    Using a single shared focal length (estimated from the first image
    unless one is given explicitly) keeps the projection consistent
    across an entire photo set. Estimating a different focal length per
    image would let each cylinder curve slightly differently, which can
    reintroduce misalignment that the projection is meant to remove.

    Args:
        images: List of BGR images.
        focal_length: Optional shared focal length in pixels. Estimated
            from the first image if not provided.

    Returns:
        Tuple of (list of warped images, focal length in pixels actually used).
    """
    if not images:
        return [], 0.0

    if focal_length is None:
        focal_length = estimate_focal_length(images[0])

    warped_images = [
        cylindrical_warp_image(img, focal_length=focal_length) for img in images
    ]
    return warped_images, float(focal_length)
