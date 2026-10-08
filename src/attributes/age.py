"""Age prediction models, DEX expectation estimation, and preprocessing."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ._lock import _lock_for
from .race import fairface_age_label, fairface_probabilities

try:
    from src.core import AGE_LIST, DEX_MAX_AGE_SD, DEX_MEAN_VALUES
except ImportError:
    from core import AGE_LIST, DEX_MAX_AGE_SD, DEX_MEAN_VALUES

try:
    from src.nets.mivolo.inference_wrapper import MiVOLOInference
except ImportError:
    try:
        from nets.mivolo.inference_wrapper import MiVOLOInference
    except ImportError:
        MiVOLOInference = Any


def caffe_probabilities(net: Any, blob: np.ndarray) -> np.ndarray:
    """Extract class probabilities from Caffe model output."""
    with _lock_for(net):
        net.setInput(blob)
        return net.forward()[0].flatten()


def predict_age_caffe(net: Any, blob: np.ndarray) -> str:
    """Predict age bucket from a Caffe blob via argmax."""
    return AGE_LIST[int(caffe_probabilities(net, blob).argmax())]


def crop_face_dex(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Extract DEX 40% margin crop, replicating missing edge pixels."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = round((x2 - x1) * 0.4), round((y2 - y1) * 0.4)
    left, top, right, bottom = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
    height, width = frame.shape[:2]
    crop = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    return cv2.copyMakeBorder(
        crop, max(0, -top), max(0, bottom - height),
        max(0, -left), max(0, right - width), cv2.BORDER_REPLICATE,
    )


def dex_age_estimate(net: Any, face_bgr: np.ndarray) -> tuple[float, float] | None:
    """Decode DEX 101-class softmax over ages 0-100 into expected age and standard deviation."""
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (224, 224), DEX_MEAN_VALUES, swapRB=False, crop=False)
    with _lock_for(net):
        net.setInput(blob)
        probs = net.forward().flatten().astype(np.float64)
    if probs.size != 101 or not np.isfinite(probs).all() or (probs < 0).any():
        return None
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    probs /= total
    years = np.arange(101)
    age = float(probs @ years)
    spread = float(np.sqrt(probs @ ((years - age) ** 2)))
    return age, spread


def format_dex_age(estimate: tuple[float, float] | None) -> str:
    """Render a DEX estimate into a display string."""
    if estimate is None:
        return "unknown"
    age, spread = estimate
    if spread > DEX_MAX_AGE_SD:
        return f"uncertain (mean {age:.0f}, SD {spread:.0f})"
    return f"{age:.0f}"


def predict_age_dex(net: Any, face_bgr: np.ndarray) -> str:
    """Predict a continuous age with DEX, labelling wide distributions."""
    return format_dex_age(dex_age_estimate(net, face_bgr))


def mivolo_estimate(net: Any, face_bgr: np.ndarray) -> tuple[float, str]:
    """Estimate age and gender from a single MiVOLO forward pass."""
    with _lock_for(net):
        age, gender, _ = net.predict_face(face_bgr)
    return float(age), ("Male" if gender == "male" else "Female")


def mivolo_age_estimate(net: Any, face_bgr: np.ndarray) -> float:
    """Run MiVOLO age-only inference."""
    with _lock_for(net):
        return float(net.predict_age(face_bgr))


def predict_age_mivolo(net: Any, face_bgr: np.ndarray) -> str:
    """Predict age string using MiVOLO model."""
    age, _ = mivolo_estimate(net, face_bgr)
    return f"{int(round(age))}"


def predict_age_fairface(
    net: Any,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None = None,
) -> str:
    """Predict age bucket via FairFace using landmarks when available."""
    return fairface_age_label(fairface_probabilities(net, frame_bgr, box, "age_output", landmarks))
