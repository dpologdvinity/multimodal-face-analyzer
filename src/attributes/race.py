"""Race classification models and FairFace forward inference."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from ._lock import _lock_for
from .transformers import align_face_with_landmarks, _margin_align

try:
    from src.core import (
        RACE_CLOSE_MARGIN,
        RACE_LABELS_FAIRFACE,
        FAIRFACE_AGE_LABELS,
        RACE_LABELS_DEEPFACE,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
except ImportError:
    from core import (
        RACE_CLOSE_MARGIN,
        RACE_LABELS_FAIRFACE,
        FAIRFACE_AGE_LABELS,
        RACE_LABELS_DEEPFACE,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax."""
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def _format_race_label(probs: np.ndarray, labels: list[str]) -> str:
    """Format race prediction label, combining close top-2 predictions."""
    order = np.argsort(probs)[::-1]
    top1, top2 = order[0], order[1]
    if probs[top1] - probs[top2] < RACE_CLOSE_MARGIN:
        return f"{labels[top1]} ({probs[top1] * 100:.0f}%)/{labels[top2]} ({probs[top2] * 100:.0f}%)"
    return labels[top1]


def _fairface_forward(
    net: Any,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    output_name: str,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Run forward pass on FairFace model for specified output head."""
    aligned = align_face_with_landmarks(frame_bgr, landmarks, 224) if landmarks is not None else None
    if aligned is None:
        aligned = _margin_align(frame_bgr, box, 224, margin=1.5)
    face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    blob = face_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    with _lock_for(net):
        net.setInput(blob)
        return net.forward(output_name).flatten()


def fairface_probabilities(
    net: Any,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    output_name: str,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Compute softmax probabilities for FairFace head."""
    return _softmax(_fairface_forward(net, frame_bgr, box, output_name, landmarks))


def fairface_race_label(probs: np.ndarray) -> str:
    """Extract race prediction label from FairFace probabilities."""
    return _format_race_label(probs, RACE_LABELS_FAIRFACE)


def fairface_gender_label(probs: np.ndarray) -> str:
    """Extract gender label from FairFace probabilities."""
    return "Male" if np.argmax(probs) == 0 else "Female"


def fairface_age_label(probs: np.ndarray) -> str:
    """Extract age bucket label from FairFace probabilities."""
    return FAIRFACE_AGE_LABELS[int(np.argmax(probs))]


def predict_race_fairface(
    net: Any,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None = None,
) -> str:
    """Predict race using FairFace model."""
    return fairface_race_label(fairface_probabilities(net, frame_bgr, box, "race_output", landmarks))


def deepface_probabilities(net: Any, face_bgr: np.ndarray) -> np.ndarray:
    """Compute class probabilities from DeepFace head."""
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = face_rgb[np.newaxis, ...]
    with _lock_for(net):
        return net(blob, training=False).numpy().flatten()


def predict_race_deepface(net: Any, face_bgr: np.ndarray) -> str:
    """Predict race using DeepFace model."""
    return _format_race_label(deepface_probabilities(net, face_bgr), RACE_LABELS_DEEPFACE)
