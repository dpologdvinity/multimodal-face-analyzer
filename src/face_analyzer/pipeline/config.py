"""Analysis configuration dataclass for the facial analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisConfig:
    """Configuration options and feature flags for pipeline execution."""

    conf_threshold: float = 0.5
    active_age: set[str] = field(default_factory=set)
    active_gender: set[str] = field(default_factory=set)
    active_emotion: set[str] = field(default_factory=set)
    active_race: set[str] = field(default_factory=set)
    active_recognition: set[str] = field(default_factory=set)
    gallery: dict[str, Any] = field(default_factory=dict)
    active_glasses: set[str] = field(default_factory=set)
    active_mask: set[str] = field(default_factory=set)
    active_eye_color: set[str] = field(default_factory=set)
    active_face_landmarks: set[str] = field(default_factory=set)
    active_hands: set[str] = field(default_factory=set)
    active_gaze: set[str] = field(default_factory=set)
    global_adjustments: dict[str, Any] = field(default_factory=dict)
    face_adjustments: dict[str, Any] = field(default_factory=dict)
    face_detector: str = "yolo"
    metrics: dict[str, Any] | None = None
    tracker: Any | None = None
    liveness_tracker: Any | None = None
    active_liveness: set[str] | None = None


__all__ = ["AnalysisConfig"]
