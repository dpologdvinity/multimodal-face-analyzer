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


def predict_face_landmarks_mediapipe(
    landmarker: Any, face_bgr: np.ndarray, result: Any = None
) -> list[tuple[float, float]] | None:
    """MediaPipe FaceLandmarker face-mesh points used by the landmark and gaze features."""
    result = result if result is not None else _detect_face_landmarker(landmarker, face_bgr)
    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y) for lm in result.face_landmarks[0]]


def predict_gaze_mediapipe(landmarker: Any, face_bgr: np.ndarray, result: Any = None) -> str:
    """Estimate coarse gaze direction from MediaPipe iris and eye landmarks."""
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 478:
        return "unknown"

    def center(indices: tuple[int, ...]) -> np.ndarray:
        return np.mean([points[index] for index in indices], axis=0)

    directions = []
    for iris, corners, vertical in (
        ((468, 469, 470, 471, 472), (33, 133), (159, 145)),
        ((473, 474, 475, 476, 477), (362, 263), (386, 374)),
    ):
        iris_center = center(iris)
        left_corner, right_corner = (points[index] for index in corners)
        eye_width = abs(right_corner[0] - left_corner[0])
        eye_height = abs(points[vertical[0]][1] - points[vertical[1]][1])
        if eye_width < 1e-6 or eye_height < 1e-6:
            continue
        horizontal = (iris_center[0] - min(left_corner[0], right_corner[0])) / eye_width
        vertical_position = (iris_center[1] - min(points[index][1] for index in vertical)) / eye_height
        directions.append((horizontal, vertical_position))

    if not directions:
        return "unknown"
    horizontal, vertical_position = np.mean(directions, axis=0)
    horizontal_label = "left" if horizontal < 0.38 else "right" if horizontal > 0.62 else "center"
    vertical_label = "up" if vertical_position < 0.35 else "down" if vertical_position > 0.65 else "level"
    return f"{horizontal_label}/{vertical_label}"


def predict_head_pose_mediapipe(landmarker: Any, face_bgr: np.ndarray, result: Any = None) -> str:
    """Estimate coarse yaw/pitch from stable MediaPipe face landmarks."""
    points = predict_face_landmarks_mediapipe(landmarker, face_bgr, result)
    if points is None or len(points) < 264:
        return "unknown"
    image_points = np.float32([points[i] for i in (1, 152, 33, 263, 61, 291)])
    h, w = face_bgr.shape[:2]
    image_points[:, 0] *= w
    image_points[:, 1] *= h
    model_points = np.float32([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ])
    focal = float(w)
    camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return "unknown"
    matrix, _ = cv2.Rodrigues(rotation)
    pitch = np.degrees(np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2])))
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return f"yaw={yaw:.0f}°, pitch={pitch:.0f}°"


__all__ = [
    "_detect_face_landmarker",
    "detect_hand_landmarks_mediapipe",
    "predict_face_landmarks_mediapipe",
    "predict_gaze_mediapipe",
    "predict_head_pose_mediapipe",
]
