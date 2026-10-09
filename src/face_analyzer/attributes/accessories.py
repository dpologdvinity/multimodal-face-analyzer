"""Accessories, mask, glasses, eye color, and texture artifact prediction."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core import GLASSES_THRESHOLD, MASK_LABELS
from ..liveness import texture_artifact_score
from ._lock import _lock_for


def predict_texture_artifact_score(face_bgr: np.ndarray) -> float:
    """Score regular high-frequency texture in a face crop as a replay cue."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sample = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return texture_artifact_score(sample.tolist())


def predict_glasses_mobilenet(net: Any, face_bgr: np.ndarray) -> str:
    """Predict whether eyeglasses are present using MobileNet ONNX model."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    blob = face_rgb[np.newaxis, ...].astype(np.uint8)
    prob = float(net.run(["eyeglasses_prob"], {net.get_inputs()[0].name: blob})[0].flatten()[0])
    return "glasses" if prob >= GLASSES_THRESHOLD else "none"


def predict_mask_mobilenetv2(net: Any, face_bgr: np.ndarray) -> str:
    """Predict whether face mask is worn using MobileNetV2 model."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = face_rgb / 127.5 - 1.0
    with _lock_for(net):
        probs = net(face_norm[np.newaxis, ...], training=False).numpy().flatten()
    return MASK_LABELS[int(np.argmax(probs))]


def predict_eye_color_colorimetric(
    eye_cascade: Any, face_bgr: np.ndarray, landmarks: Any = None,
) -> str:
    """Predict eye color from iris crop or Haar cascade detection using HSV colorimetry."""
    samples = []
    if landmarks is not None and len(landmarks) >= 478:
        height, width = face_bgr.shape[:2]
        points = np.asarray(landmarks, dtype=np.float64)[:, :2] * [width, height]
        for index in (468, 473):
            iris = points[index:index + 5]
            if not np.isfinite(iris).all():
                continue
            cx, cy = iris[0]
            radius = float(np.mean(np.linalg.norm(iris[1:] - iris[0], axis=1)))
            if radius < 2 or not (0 <= cx < width and 0 <= cy < height):
                continue
            x1, x2 = max(0, int(cx - radius)), min(width, int(np.ceil(cx + radius)))
            y1, y2 = max(0, int(cy - radius)), min(height, int(np.ceil(cy + radius)))
            yy, xx = np.ogrid[y1:y2, x1:x2]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2 < (0.9 * radius) ** 2)
            mask &= (abs(xx - cx) > 0.35 * radius) & (abs(yy - cy) < 0.25 * radius)
            pixels = face_bgr[y1:y2, x1:x2][mask]
            if len(pixels) >= 4:
                samples.append(pixels)
    if samples:
        median = np.median(np.concatenate(samples), axis=0).astype(np.uint8)
        hue, saturation, value = cv2.cvtColor(median.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        if saturation >= 40 and 95 <= hue <= 130:
            return "blue"
        if value < 60:
            return "brown"
        if saturation < 40:
            return "grey"
        if 40 <= hue < 95:
            return "green"
        if 15 <= hue < 40 and saturation > 100:
            return "amber"
        return "brown" if hue < 15 or hue >= 170 else "hazel"

    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    with _lock_for(eye_cascade):
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    if len(eyes) == 0:
        return "unknown"

    ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
    cx1 = ex + int(ew * 0.3)
    cx2 = ex + int(ew * 0.7)
    cy1 = ey + int(eh * 0.3)
    cy2 = ey + int(eh * 0.7)
    iris_region = face_bgr[cy1:cy2, cx1:cx2]
    if iris_region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(iris_region, cv2.COLOR_BGR2HSV)
    med_h = float(np.median(hsv[..., 0]))
    med_s = float(np.median(hsv[..., 1]))
    med_v = float(np.median(hsv[..., 2]))

    if med_v < 60:
        return "brown"
    if med_s < 40:
        return "grey"
    if 95 <= med_h <= 130:
        return "blue"
    if 40 <= med_h < 95:
        return "green"
    if 15 <= med_h < 40 and med_s > 100:
        return "amber"
    if med_h < 15 or med_h >= 170:
        return "brown" if med_v < 130 else "hazel"
    return "hazel"
