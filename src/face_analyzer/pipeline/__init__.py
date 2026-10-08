"""Pipeline orchestration subpackage for facial analysis workflow and visualization."""
from __future__ import annotations

from .analyzer import AGGREGATE_FEATURES, aggregate_demographics, analyze_frame
from .config import AnalysisConfig
from .drawing import draw_recognition_scan
from .landmarks import predict_face_landmarks_mediapipe
from .loader import load_models
from .tracker import FaceTracker

__all__ = [
    "AnalysisConfig",
    "load_models",
    "FaceTracker",
    "analyze_frame",
    "aggregate_demographics",
    "draw_recognition_scan",
    "predict_face_landmarks_mediapipe",
    "AGGREGATE_FEATURES",
]
