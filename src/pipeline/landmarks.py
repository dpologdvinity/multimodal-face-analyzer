"""MediaPipe landmarker inference calls for one face crop or one whole frame."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from src.attributes import _lock_for
    from src.pipeline.drawing import MEDIAPIPE_SUPPORTED, _silence_native_logs
except ImportError:
    from attributes import _lock_for
    from pipeline.drawing import MEDIAPIPE_SUPPORTED, _silence_native_logs

if MEDIAPIPE_SUPPORTED:
    import mediapipe as mp


def _detect_face_landmarker(landmarker: Any, face_bgr: np.ndarray) -> Any:
    """Run MediaPipe FaceLandmarker on one face crop with thread-safe locking."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    with _lock_for(landmarker):
        with _silence_native_logs():
            return landmarker.detect(mp_image)


def detect_hand_landmarks_mediapipe(landmarker: Any, frame_bgr: np.ndarray) -> list[list[tuple[int, int]]]:
    """MediaPipe HandLandmarker, whole-frame (hands aren't tied to a detected face box)."""
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with _lock_for(landmarker):
        with _silence_native_logs():
            result = landmarker.detect(mp_image)
    return [
        [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand]
        for hand in result.hand_landmarks
    ]


__all__ = ["_detect_face_landmarker", "detect_hand_landmarks_mediapipe"]
