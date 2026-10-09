"""Image processing, cropping, clamping, and color adjustment utilities."""
from __future__ import annotations

import cv2
import numpy as np

from .constants import GRAYSCALE_CHANNEL_DIFF_THRESHOLD


def is_grayscale_frame(frame_bgr: np.ndarray) -> bool:
    """Heuristic: a 3-channel image that's actually grayscale (common for old photos saved
    as BGR/RGB with all channels equal, or scanned B&W) has near-zero difference between its
    B/G/R channels across the whole image. Downsamples first -- only the mean matters, and a
    small sample is far cheaper than scanning a full-resolution frame."""
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    diff = (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)) / 3.0
    return float(diff.mean()) < GRAYSCALE_CHANNEL_DIFF_THRESHOLD


def crop_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop an arbitrary rectangle, clamped to frame bounds."""
    h, w = frame.shape[:2]
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    return frame[y1:y2, x1:x2]


def face_crop_bounds(
    box: tuple[int, int, int, int], frame_shape: tuple[int, int], padding_ratio: float = 0.1,
) -> tuple[int, int, int, int]:
    """Return a detector crop with scale-relative context, using exclusive bounds.

    The Caffe age model is sensitive to how much surrounding context occupies its fixed 227x227
    input. A fixed pixel margin makes that context dominate small faces and disappear around
    large faces, so keep the framing proportionate to the detected face instead.
    """
    x1, y1, x2, y2 = box
    frame_height, frame_width = frame_shape
    face_width, face_height = max(0, x2 - x1), max(0, y2 - y1)
    padding = max(1, round(max(face_width, face_height) * padding_ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(frame_width, x2 + padding),
        min(frame_height, y2 + padding),
    )


def _adjust_exposure(img: np.ndarray, stops: float) -> np.ndarray:
    """Apply exposure correction in f-stops (base 2 scaling)."""
    return img * (2.0 ** stops)


def _adjust_brightness(img: np.ndarray, amount: float) -> np.ndarray:
    """Apply additive brightness shift."""
    return img + amount


def _adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Apply contrast correction via the classic parametric formula."""
    c = amount * 2.55  # slider -100..100 -> classic contrast-correction-factor's -255..255
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    return factor * (img - 128.0) + 128.0


def _adjust_tone_region(img_bgr: np.ndarray, amount: float, region: str) -> np.ndarray:
    """Shift highlights or shadows via a luminance-weighted mask in HSV's V channel.
    Positive `amount` brightens highlights / lifts shadows (Lightroom convention)."""
    hsv = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2]
    if region == "highlights":
        mask = np.clip((v - 128.0) / 127.0, 0.0, 1.0)
    else:
        mask = np.clip((128.0 - v) / 128.0, 0.0, 1.0)
    hsv[..., 2] = np.clip(v + (amount / 100.0) * 50.0 * mask, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def _adjust_black_point(img: np.ndarray, amount: float) -> np.ndarray:
    """Shift and stretch the shadow point (rescales to compensate)."""
    bp = np.clip((amount / 100.0) * 60.0, -60.0, 250.0)
    return (img - bp) * (255.0 / max(255.0 - bp, 1.0))


def _adjust_saturation(img_bgr: np.ndarray, amount: float, vibrance: bool = False) -> np.ndarray:
    """Adjust saturation or skin-preserving vibrance in HSV color space."""
    hsv = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    if vibrance:
        # Boost low-saturation pixels more than already-saturated ones (protects skin tones).
        s = s + (amount / 100.0) * 60.0 * (1.0 - s / 255.0)
    else:
        s = s * (1.0 + amount / 100.0)
    hsv[..., 1] = np.clip(s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def _adjust_sharpness(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Classic unsharp mask -- small-radius blur subtracted back out to boost edge contrast."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
    return img_bgr + (amount / 100.0) * 1.5 * (img_bgr - blurred)


def _adjust_definition(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """'Clarity'-style local contrast: large-radius unsharp mask on the LAB lightness channel
    only, so it boosts midtone structure without shifting color."""
    lab = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=12.0)
    lab[..., 0] = np.clip(L + (amount / 100.0) * 1.2 * (L - blurred), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def _adjust_noise_reduction(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Edge-preserving denoise (bilateral filter); strength scales with the slider."""
    strength = amount / 100.0
    return cv2.bilateralFilter(np.clip(img_bgr, 0, 255).astype(np.uint8), d=5, sigmaColor=strength * 100, sigmaSpace=strength * 100).astype(np.float32)


def apply_image_adjustments(face_bgr: np.ndarray, adjustments: dict) -> np.ndarray:
    """Apply the Lightroom-style slider stack to one face crop, in a fixed pipeline order
    (denoise first so later steps don't amplify grain; sharpen last so it acts on the final
    tonal/color state). Any slider left at its default (0) is skipped entirely -- cheap when
    the panel is untouched, since this runs once per face per frame."""
    img = face_bgr.astype(np.float32)

    if adjustments.get("noise_reduction", 0):
        img = _adjust_noise_reduction(img, adjustments["noise_reduction"])
    if adjustments.get("exposure", 0):
        img = _adjust_exposure(img, adjustments["exposure"])
    if adjustments.get("black_point", 0):
        img = _adjust_black_point(img, adjustments["black_point"])
    if adjustments.get("shadows", 0):
        img = _adjust_tone_region(img, adjustments["shadows"], "shadows")
    if adjustments.get("highlights", 0):
        img = _adjust_tone_region(img, adjustments["highlights"], "highlights")
    if adjustments.get("contrast", 0):
        img = _adjust_contrast(img, adjustments["contrast"])
    if adjustments.get("brightness", 0):
        img = _adjust_brightness(img, adjustments["brightness"])
    if adjustments.get("saturation", 0):
        img = _adjust_saturation(img, adjustments["saturation"])
    if adjustments.get("vibrance", 0):
        img = _adjust_saturation(img, adjustments["vibrance"], vibrance=True)
    if adjustments.get("definition", 0):
        img = _adjust_definition(img, adjustments["definition"])
    if adjustments.get("sharpness", 0):
        img = _adjust_sharpness(img, adjustments["sharpness"])

    return np.clip(img, 0, 255).astype(np.uint8)

