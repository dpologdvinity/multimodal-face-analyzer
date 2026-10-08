"""Pipeline orchestration subpackage for facial analysis workflow and visualization."""
from __future__ import annotations

from .analyzer import AGGREGATE_FEATURES, aggregate_demographics, analyze_frame
from .cache import (
    _INFERENCE_EXECUTOR,
    _PREDICTION_CACHE,
    _cached_face_predict,
    _record_model_latency,
)
from .config import AnalysisConfig
from .drawing import (
    _silence_native_logs,
    draw_face_landmarks,
    draw_hand_landmarks,
    draw_outlined_text,
    draw_recognition_scan,
)
from .landmarks import (
    _detect_face_landmarker,
    detect_hand_landmarks_mediapipe,
    predict_face_landmarks_mediapipe,
    predict_gaze_mediapipe,
    predict_head_pose_mediapipe,
)
from .loader import load_models
from .tracker import FaceTracker, _box_iou

__all__ = [
    "AnalysisConfig",
    "load_models",
    "FaceTracker",
    "_box_iou",
    "analyze_frame",
    "aggregate_demographics",
    "draw_face_landmarks",
    "detect_hand_landmarks_mediapipe",
    "draw_hand_landmarks",
    "draw_outlined_text",
    "draw_recognition_scan",
    "predict_face_landmarks_mediapipe",
    "predict_gaze_mediapipe",
    "predict_head_pose_mediapipe",
    "_detect_face_landmarker",
    "_silence_native_logs",
    "_cached_face_predict",
    "_record_model_latency",
    "_PREDICTION_CACHE",
    "_INFERENCE_EXECUTOR",
    "AGGREGATE_FEATURES",
]
