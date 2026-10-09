"""Image transformations, colorization, alignment, and 3D reconstruction."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core import (
    FAIRFACE_LANDMARK_INDICES,
    FAIRFACE_REFERENCE_LANDMARKS,
    is_grayscale_frame,
)
from ._lock import _lock_for

try:
    from ..nets.deep3d_recon import landmarks_5pt_from_mediapipe, reconstruct_face_3d
except ImportError:
    landmarks_5pt_from_mediapipe = None
    reconstruct_face_3d = None

try:
    from ..nets.face_reaging_model import age_progress_face
except ImportError:
    age_progress_face = None


def colorize_frame(net: Any, frame_bgr: np.ndarray) -> np.ndarray:
    """Colorize a grayscale image using ECCV16 Lab colorization model."""
    scaled = frame_bgr.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    with _lock_for(net):
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (frame_bgr.shape[1], frame_bgr.shape[0]))

    L_full = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")


def maybe_colorize(models: Any, frame_bgr: np.ndarray, active_colorization: set) -> tuple[np.ndarray, bool]:
    """Auto-colorize frame if grayscale and colorization backend is active."""
    net = models.colorization_nets.get("eccv16")
    if net is None or "eccv16" not in active_colorization:
        return frame_bgr, False
    if not is_grayscale_frame(frame_bgr):
        return frame_bgr, False
    return colorize_frame(net, frame_bgr), True


def apply_intensity_transform(
    face_bgr: np.ndarray,
    method: str = "gamma",
    gamma: float = 0.7,
    r1: int = 70,
    s1: int = 0,
    r2: int = 140,
    s2: int = 255,
) -> np.ndarray:
    """Apply intensity transformation (negative, log, gamma, contrast_stretch)."""
    img = face_bgr.astype(np.float32)

    if method == "negative":
        return (255 - img).astype(np.uint8)

    if method == "log":
        c = 255.0 / np.log(1 + img.max()) if img.max() > 0 else 1.0
        return np.clip(c * np.log(1 + img), 0, 255).astype(np.uint8)

    if method == "gamma":
        return np.clip(255.0 * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)

    if method == "contrast_stretch":
        out = np.empty_like(img)
        low = img <= r1
        mid = (img > r1) & (img <= r2)
        high = img > r2
        out[low] = (s1 / r1) * img[low] if r1 else 0
        out[mid] = ((s2 - s1) / (r2 - r1)) * (img[mid] - r1) + s1
        out[high] = ((255 - s2) / (255 - r2)) * (img[high] - r2) + s2
        return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown intensity method: {method}")


def apply_enhance(face_bgr: np.ndarray, brightness: float = 10.0, contrast: float = 1.3) -> np.ndarray:
    """Enhance image with brightness/contrast adjustment and CLAHE/equalization."""
    adjusted = cv2.addWeighted(face_bgr, contrast, np.zeros_like(face_bgr), 0, brightness)
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    lum = cv2.equalizeHist(lum)
    return cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)


def apply_sharpen(face_bgr: np.ndarray, method: str = "laplacian") -> np.ndarray:
    """Sharpen image using Laplacian or high-boost filter."""
    if method == "high_boost":
        kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(face_bgr, -1, kernel)


def apply_color_correct(face_bgr: np.ndarray) -> np.ndarray:
    """Apply color correction using CLAHE on LAB luminance channel."""
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lum = clahe.apply(lum)
    return cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)


def apply_denoise(face_bgr: np.ndarray, method: str = "nlm") -> np.ndarray:
    """Denoise image using Gaussian, median, or Non-Local Means filter."""
    if method == "gaussian":
        return cv2.GaussianBlur(face_bgr, (5, 5), 1.5)
    if method == "median":
        return cv2.medianBlur(face_bgr, 5)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(face_bgr, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    raise ValueError(f"Unknown denoise method: {method}")


def apply_bilateral_filter(
    face_bgr: np.ndarray, diameter: int = 15, sigma_color: float = 75.0, sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral filter to image."""
    return cv2.bilateralFilter(face_bgr, diameter, sigma_color, sigma_space)


def apply_wavelet_denoise(face_bgr: np.ndarray, threshold: float = 0.05, wavelet: str = "db1") -> np.ndarray:
    """Apply 2D wavelet decomposition soft-threshold denoising."""
    import pywt

    channels = cv2.split(face_bgr.astype(np.float32))
    denoised_channels = []
    for channel in channels:
        coeffs = pywt.wavedec2(channel, wavelet, level=2)
        detail_coeffs = coeffs[1:]
        max_detail = max((np.abs(d).max() for level in detail_coeffs for d in level), default=1.0) or 1.0
        abs_threshold = threshold * max_detail
        thresholded = [coeffs[0]] + [
            tuple(pywt.threshold(d, abs_threshold, mode="soft") for d in level) for level in detail_coeffs
        ]
        reconstructed = pywt.waverec2(thresholded, wavelet)
        denoised_channels.append(reconstructed[:channel.shape[0], :channel.shape[1]])

    return np.clip(cv2.merge(denoised_channels), 0, 255).astype(np.uint8)


def apply_image_op(face_bgr: np.ndarray, op: str, **params: Any) -> np.ndarray:
    """Dispatch image operation by name to corresponding transformation function."""
    dispatch = {
        "intensity": apply_intensity_transform,
        "enhance": apply_enhance,
        "sharpen": apply_sharpen,
        "color_correct": apply_color_correct,
        "denoise": apply_denoise,
        "bilateral_filter": apply_bilateral_filter,
        "wavelet_denoise": apply_wavelet_denoise,
    }
    if op not in dispatch:
        raise ValueError(f"Unknown image op: {op}")
    return dispatch[op](face_bgr, **params)


def run_3d_reconstruction(models: Any, face_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reconstruct 3D face mesh using Deep3DFaceRecon."""
    if landmarks_5pt_from_mediapipe is None or reconstruct_face_3d is None:
        return None
    bundle = models.reconstruction_3d_nets.get("deep3d")
    landmarker = models.face_landmarks_nets.get("mediapipe")
    if bundle is None or landmarker is None:
        return None

    # Deferred: pipeline.landmarks imports this package, so a top-level import would be circular.
    from ..pipeline.landmarks import predict_face_landmarks_mediapipe

    recon_net, bfm_model, lm3d_template = bundle
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr)
    if points is None:
        return None

    h, w = face_bgr.shape[:2]
    landmarks_5pt = landmarks_5pt_from_mediapipe(points, w, h)
    return reconstruct_face_3d(recon_net, bfm_model, face_bgr, landmarks_5pt, lm3d_template)


def run_age_progression(models: Any, face_bgr: np.ndarray, source_age: float, target_age: float) -> np.ndarray | None:
    """Progress or regress face age using FRAN model."""
    if age_progress_face is None:
        return None
    net = models.age_progression_nets.get("franunet")
    if net is None:
        return None
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    aged_rgb = age_progress_face(net, face_rgb, source_age, target_age)
    return cv2.cvtColor(aged_rgb, cv2.COLOR_RGB2BGR)


def _margin_align(frame_bgr: np.ndarray, box: tuple[int, int, int, int], output_size: int, margin: float) -> np.ndarray:
    """Crop and center face box with margin context resized to output size."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = output_size / (max(w, h) * margin)
    m = np.array([
        [scale, 0, output_size / 2 - scale * cx],
        [0, scale, output_size / 2 - scale * cy],
    ], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, m, (output_size, output_size), borderValue=0.0)


def fairface_landmarks_from_mediapipe(
    points: list[tuple[float, float]], width: int, height: int,
) -> np.ndarray | None:
    """Convert MediaPipe landmarks to FairFace five-point pixel order."""
    if len(points) <= max(FAIRFACE_LANDMARK_INDICES) or width <= 0 or height <= 0:
        return None
    selected = np.asarray([points[index] for index in FAIRFACE_LANDMARK_INDICES], dtype=np.float32)
    if selected.shape != (5, 2) or not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
        return None
    selected *= np.array([width, height], dtype=np.float32)
    return selected


def align_face_with_landmarks(
    frame_bgr: np.ndarray, landmarks: np.ndarray, output_size: int,
) -> np.ndarray | None:
    """Align face to FairFace reference landmarks using similarity transform."""
    source = np.asarray(landmarks, dtype=np.float64)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        return None
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        return None
    target = (FAIRFACE_REFERENCE_LANDMARKS + 0.25) / 1.5 * output_size
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, :2] = np.column_stack((target[:, 0], -target[:, 1]))
    design[1::2, :2] = np.column_stack((target[:, 1], target[:, 0]))
    design[0::2, 2] = 1
    design[1::2, 3] = 1
    a, b, tx, ty = np.linalg.lstsq(design, source.reshape(-1), rcond=None)[0]
    if a * a + b * b < 1e-12:
        return None
    matrix = np.array([[a, -b, tx], [b, a, ty]])
    return cv2.warpAffine(
        frame_bgr, matrix, (output_size, output_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0.0,
    )


def _estimate_roll_angle(face_bgr: np.ndarray, eye_cascade: Any) -> float | None:
    """Estimate head roll angle in degrees from eye detections."""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    with _lock_for(eye_cascade):
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    return angle if abs(angle) <= 45 else None


def level_face_region(
    frame_bgr: np.ndarray, box: tuple[int, int, int, int], angle_deg: float | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (region, box) rotated level when the roll angle exceeds 3 degrees, else the inputs."""
    if angle_deg is not None and abs(angle_deg) > 3:
        return _rotate_region(frame_bgr, box, angle_deg)
    return frame_bgr, box


def _rotate_region(
    frame_bgr: np.ndarray, box: tuple[int, int, int, int], angle_deg: float, pad_factor: float = 0.8,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rotate image region around face bounding box center to level roll angle."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = int(pad_factor * max(w, h))
    fy, fx = frame_bgr.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(fx, x2 + pad), min(fy, y2 + pad)
    region = frame_bgr[ry1:ry2, rx1:rx2]
    local_box = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
    lcx, lcy = (local_box[0] + local_box[2]) / 2.0, (local_box[1] + local_box[3]) / 2.0
    m = cv2.getRotationMatrix2D((lcx, lcy), angle_deg, 1.0)
    rotated = cv2.warpAffine(region, m, (region.shape[1], region.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return rotated, local_box
