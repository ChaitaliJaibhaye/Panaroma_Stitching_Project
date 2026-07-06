"""
utils.py
--------
General-purpose helper functions used across the panorama stitching
pipeline: image validation/loading, black-border cropping, cylindrical
projection, simple exposure compensation, and debug-image saving.

"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Supported image extensions for upload validation.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Directory used to store intermediate debug images.
DEBUG_OUTPUT_DIR = "outputs/debug"


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def validate_image_file(filename: str, file_bytes: bytes) -> None:
    """Validate that an uploaded file is a supported, non-corrupted image.

    Args:
        filename: Original filename (used to check extension).
        file_bytes: Raw bytes of the uploaded file.

    Raises:
        ImageValidationError: If the extension is unsupported or the
            image data cannot be decoded (corrupted file).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Allowed types are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageValidationError(
            f"'{filename}' appears to be corrupted or is not a valid image."
        )


def bytes_to_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into a BGR OpenCV image.

    Args:
        file_bytes: Raw bytes of an image file.

    Returns:
        Decoded BGR image as a NumPy array.

    Raises:
        ImageValidationError: If decoding fails.
    """
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Failed to decode image bytes.")
    return image


def resize_keep_aspect(image: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """Resize an image so its largest dimension does not exceed a limit.

    Large images slow down feature detection/matching considerably, so
    resizing before processing keeps runtime reasonable without a large
    quality loss in the final panorama.

    Args:
        image: Input BGR image.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        Resized image (or the original if already small enough).
    """
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image

    scale = max_dimension / float(largest)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_black_borders(image: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Crop the black (empty) borders that result from perspective warping.

    Finds the largest bounding box that contains non-black pixels and
    crops the image to it. This removes the ragged black edges typical
    of a warped panorama canvas.

    Args:
        image: Panorama image, possibly with black borders.
        threshold: Pixel intensity below which a pixel is considered
            "black"/empty in every channel.

    Returns:
        Cropped image containing only useful (non-empty) pixels. If the
        image is entirely empty, the original image is returned.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = (gray > threshold).astype(np.uint8)

    if cv2.countNonZero(mask) == 0:
        return image

    # Find the largest all-content rectangle using contours on the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped = image[y : y + h, x : x + w]

    # Refine: shrink the rectangle inward until rows/cols are mostly non-black.
    # This helps remove thin black slivers left after the bounding-box crop.
    cropped = _tighten_crop(cropped, threshold=threshold)
    return cropped


def _tighten_crop(image: np.ndarray, threshold: int = 1, row_col_fill: float = 0.6) -> np.ndarray:
    """Iteratively trim rows/columns that are mostly black.

    Args:
        image: Image to trim.
        threshold: Black-pixel intensity threshold.
        row_col_fill: Minimum fraction of non-black pixels required for a
            row/column to be kept.

    Returns:
        Trimmed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = gray > threshold

    top, bottom = 0, mask.shape[0]
    left, right = 0, mask.shape[1]

    def row_ok(r: int) -> bool:
        return mask[r, left:right].mean() >= row_col_fill

    def col_ok(c: int) -> bool:
        return mask[top:bottom, c].mean() >= row_col_fill

    while top < bottom - 1 and not row_ok(top):
        top += 1
    while bottom > top + 1 and not row_ok(bottom - 1):
        bottom -= 1
    while left < right - 1 and not col_ok(left):
        left += 1
    while right > left + 1 and not col_ok(right - 1):
        right -= 1

    trimmed = image[top:bottom, left:right]
    return trimmed if trimmed.size > 0 else image


def cylindrical_warp(image: np.ndarray, focal_length: Optional[float] = None) -> np.ndarray:
    """Project an image onto a cylindrical surface.

    Deprecated: implementation now lives in `cylindrical_projection.py`
    (`cylindrical_warp_image` / `cylindricalWarpImage`), which also adds
    automatic focal-length estimation. This wrapper is kept so any
    existing code importing `cylindrical_warp` from `utils` continues to
    work unchanged.

    Args:
        image: Input BGR image.
        focal_length: Approximate focal length in pixels. If not
            provided, it is estimated automatically.

    Returns:
        Cylindrically-warped image, cropped to its non-black content.
    """
    from cylindrical_projection import cylindrical_warp_image  # local import avoids circular import

    return cylindrical_warp_image(image, focal_length=focal_length, crop_borders=True)


def compute_sharpness(image: np.ndarray) -> float:
    """Estimate image sharpness using the variance of the Laplacian.

    A well-focused, detailed image has high-frequency edges everywhere,
    which produces a high-variance Laplacian response. Blurry images
    produce a low-variance response. This is a standard, cheap blur
    proxy used both for the quality dashboard and as an ML feature.

    Args:
        image: Input BGR (or grayscale) image.

    Returns:
        Sharpness score (variance of Laplacian). Typical values range
        from near 0 (very blurry) to several hundred/thousand+ (sharp).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_mean_brightness(image: np.ndarray) -> float:
    """Compute the mean grayscale brightness of an image.

    Used as a cheap per-image exposure proxy for the quality dashboard
    and ML feature vector (brightness difference across the input set).

    Args:
        image: BGR (or grayscale) image.

    Returns:
        Mean pixel intensity in [0, 255].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray))


def compute_overlap_mean_ratio(
    img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
) -> float:
    """Compute the brightness gain ratio between two images in their overlap.

    Used for simple exposure compensation: the ratio tells us how much
    brighter/darker image2 is compared to image1 in the region where
    both images have content.

    Args:
        img1: First image (BGR), placed on the panorama canvas.
        img2: Second image (BGR), placed on the panorama canvas.
        mask1: Binary mask (uint8, 0/255) of where img1 has content.
        mask2: Binary mask (uint8, 0/255) of where img2 has content.

    Returns:
        Gain ratio to multiply into img2 so its brightness matches img1
        in the overlap region. Returns 1.0 if there is no overlap.
    """
    overlap = cv2.bitwise_and(mask1, mask2)
    if cv2.countNonZero(overlap) < 50:
        return 1.0

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mean1 = float(np.mean(gray1[overlap > 0]))
    mean2 = float(np.mean(gray2[overlap > 0]))

    if mean2 < 1e-3:
        return 1.0

    ratio = mean1 / mean2
    # Clamp to avoid extreme, unrealistic corrections.
    return float(np.clip(ratio, 0.5, 2.0))


def apply_gain(image: np.ndarray, gain: float) -> np.ndarray:
    """Multiply an image by a scalar gain and clip to a valid range.

    Args:
        image: Input BGR image.
        gain: Multiplicative brightness gain.

    Returns:
        Gain-adjusted image, dtype uint8.
    """
    adjusted = image.astype(np.float32) * gain
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def save_debug_image(image: np.ndarray, name: str, session_id: Optional[str] = None) -> str:
    """Save an intermediate image to the debug output directory.

    Args:
        image: Image to save.
        name: Descriptive name (without extension), e.g. "matches_0_1".
        session_id: Optional session identifier to group files from the
            same stitching run. A random one is generated if omitted.

    Returns:
        Path to the saved file.
    """
    ensure_dir(DEBUG_OUTPUT_DIR)
    session_id = session_id or uuid.uuid4().hex[:8]
    path = os.path.join(DEBUG_OUTPUT_DIR, f"{session_id}_{name}.jpg")
    cv2.imwrite(path, image)
    return path


def make_thumbnail(image: np.ndarray, max_size: int = 200) -> np.ndarray:
    """Create a small square-bounded thumbnail for UI preview.

    Args:
        image: Source BGR image.
        max_size: Maximum width/height of the thumbnail.

    Returns:
        Thumbnail image.
    """
    height, width = image.shape[:2]
    scale = max_size / float(max(height, width))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (OpenCV) image to RGB (for display in Streamlit/PIL).

    Args:
        image: BGR image.

    Returns:
        RGB image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_image_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode an OpenCV image to raw bytes for download.

    Args:
        image: BGR image to encode.
        ext: File extension/format, e.g. ".png" or ".jpg".

    Returns:
        Encoded image bytes.
    """
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return buffer.tobytes()
