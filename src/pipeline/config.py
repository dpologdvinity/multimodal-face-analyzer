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
    active_hair_color: set[str] = field(default_factory=set)
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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration options into a dictionary."""
        return {
            "conf_threshold": self.conf_threshold,
            "active_age": set(self.active_age),
            "active_gender": set(self.active_gender),
            "active_emotion": set(self.active_emotion),
            "active_race": set(self.active_race),
            "active_recognition": set(self.active_recognition),
            "gallery": self.gallery,
            "active_glasses": set(self.active_glasses),
            "active_mask": set(self.active_mask),
            "active_hair_color": set(self.active_hair_color),
            "active_eye_color": set(self.active_eye_color),
            "active_face_landmarks": set(self.active_face_landmarks),
            "active_hands": set(self.active_hands),
            "active_gaze": set(self.active_gaze),
            "global_adjustments": self.global_adjustments,
            "face_adjustments": self.face_adjustments,
            "face_detector": self.face_detector,
            "metrics": self.metrics,
            "tracker": self.tracker,
            "liveness_tracker": self.liveness_tracker,
            "active_liveness": set(self.active_liveness) if self.active_liveness is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisConfig:
        """Construct AnalysisConfig from a dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


__all__ = ["AnalysisConfig"]
