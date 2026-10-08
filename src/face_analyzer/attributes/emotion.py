"""Emotion classification models and multimodal voice-face arousal fusion."""
from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from ._lock import _lock_for

try:
    import torch
except ImportError:
    torch = None

from ..core import (
    AUDIO_AROUSAL_LOUD_RMS,
    AUDIO_AROUSAL_QUIET_RMS,
    AUDIO_AROUSAL_WINDOW_SECONDS,
    EMOTION_HIGH_AROUSAL_LABELS,
    EMOTION_LABELS_DAN,
    EMOTION_LABELS_FERPLUS,
    EMOTION_LABELS_HSEMOTION,
    EMOTION_LABELS_MINI_XCEPTION,
    EMOTION_LOW_AROUSAL_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def predict_emotion_dan(net: Any, face_bgr: np.ndarray) -> str:
    """Classify facial expression into emotion category using DAN model."""
    if torch is None:
        raise RuntimeError("Torch is required for DAN emotion model.")
    face_rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = net(tensor)
    return EMOTION_LABELS_DAN[logits[0].argmax().item()]


def predict_emotion_mini_xception(net: Any, face_bgr: np.ndarray) -> str:
    """Classify facial expression using Mini-Xception model."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    face_norm = (face_gray / 255.0 - 0.5) * 2.0
    tensor = face_norm[np.newaxis, ..., np.newaxis]
    with _lock_for(net):
        probs = net(tensor, training=False).numpy().flatten()
    return EMOTION_LABELS_MINI_XCEPTION[int(np.argmax(probs))]


def predict_emotion_ferplus(net: Any, face_bgr: np.ndarray) -> str:
    """Classify facial expression using FERPlus ONNX model."""
    face_gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    blob = face_gray[np.newaxis, np.newaxis, ...]
    with _lock_for(net):
        net.setInput(blob)
        logits = net.forward().flatten()
    return EMOTION_LABELS_FERPLUS[int(np.argmax(logits))]


def predict_emotion_hsemotion(net: Any, face_bgr: np.ndarray) -> str:
    """Classify facial expression using HSEmotion ONNX model."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(face_rgb, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=False, crop=False)
    blob = (blob - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(1, 3, 1, 1)
    with _lock_for(net):
        net.setInput(blob.astype(np.float32))
        logits = net.forward().flatten()
    return EMOTION_LABELS_HSEMOTION[int(np.argmax(logits))]


def audio_frame_to_mono_float(samples: np.ndarray) -> np.ndarray:
    """Normalize raw audio samples to 1-D float32 array in [-1, 1]."""
    if np.issubdtype(samples.dtype, np.integer):
        max_value = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / max_value
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2 and samples.shape[0] <= 8:
        samples = samples.mean(axis=0)
    return samples.flatten()


def classify_voice_arousal(rms: float) -> str:
    """Bucket normalized RMS energy level into QUIET, SPEAKING, or LOUD."""
    if rms >= AUDIO_AROUSAL_LOUD_RMS:
        return "LOUD"
    if rms >= AUDIO_AROUSAL_QUIET_RMS:
        return "SPEAKING"
    return "QUIET"


def _emotion_arousal_category(emotion_label: str) -> str | None:
    """Map emotion label to high or low arousal category."""
    label = emotion_label.lower()
    if label in EMOTION_HIGH_AROUSAL_LABELS:
        return "high"
    if label in EMOTION_LOW_AROUSAL_LABELS:
        return "low"
    return None


def fuse_voice_and_emotion(voice_arousal: str, emotion_label: str) -> str | None:
    """Compare voice loudness against emotion label for arousal consistency."""
    emotion_category = _emotion_arousal_category(emotion_label)
    if emotion_category is None:
        return None
    voice_category = "low" if voice_arousal == "QUIET" else "high"
    return "consistent" if voice_category == emotion_category else "inconsistent"


class VoiceFaceFusion:
    """Thread-safe rolling audio buffer for multimodal voice-face fusion."""

    def __init__(self, window_seconds: float = AUDIO_AROUSAL_WINDOW_SECONDS):
        """Initialize rolling audio buffer with specified window duration in seconds."""
        self._lock = threading.Lock()
        self._window_seconds = window_seconds
        self._samples: list[np.ndarray] = []
        self._buffered_seconds = 0.0
        self._sample_rate: int | None = None
        self._latest_status: dict | None = None

    def ingest_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """Append audio samples to rolling window, evicting samples beyond window duration."""
        if samples.size == 0 or sample_rate <= 0:
            return
        with self._lock:
            self._sample_rate = sample_rate
            self._samples.append(samples)
            self._buffered_seconds += samples.size / sample_rate
            while self._buffered_seconds > self._window_seconds and len(self._samples) > 1:
                oldest = self._samples.pop(0)
                self._buffered_seconds -= oldest.size / self._sample_rate

    def current_arousal(self) -> str:
        """Return classified arousal level (QUIET, SPEAKING, LOUD) for buffered window."""
        with self._lock:
            if not self._samples:
                return "QUIET"
            window = np.concatenate(self._samples)
        rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
        return classify_voice_arousal(rms)

    def set_latest_status(self, status: dict) -> None:
        """Store latest fusion status dictionary."""
        with self._lock:
            self._latest_status = status

    def get_latest_status(self) -> dict | None:
        """Retrieve latest fusion status dictionary."""
        with self._lock:
            return self._latest_status

    def reset(self) -> None:
        """Reset rolling buffer and stored status."""
        with self._lock:
            self._samples = []
            self._buffered_seconds = 0.0
            self._latest_status = None
