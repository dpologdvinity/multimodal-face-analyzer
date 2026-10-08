"""Core data types and structures for the multimodal face analyzer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Models:
    """Container holding loaded neural network models and session handles for face analysis."""
    face_net: Any = None
    age_nets: dict = field(default_factory=dict)
    gender_nets: dict = field(default_factory=dict)
    emotion_nets: dict = field(default_factory=dict)
    race_nets: dict = field(default_factory=dict)
    liveness_nets: dict = field(default_factory=dict)
    recognition_nets: dict = field(default_factory=dict)
    glasses_nets: dict = field(default_factory=dict)
    mask_nets: dict = field(default_factory=dict)
    hair_color_nets: dict = field(default_factory=dict)
    eye_color_nets: dict = field(default_factory=dict)
    colorization_nets: dict = field(default_factory=dict)
    face_landmarks_nets: dict = field(default_factory=dict)
    hand_nets: dict = field(default_factory=dict)
    reconstruction_3d_nets: dict = field(default_factory=dict)
    yolo_face_nets: dict = field(default_factory=dict)
    scrfd_face_nets: dict = field(default_factory=dict)
    retinaface_nets: dict = field(default_factory=dict)
    gaze_nets: dict = field(default_factory=dict)
    age_progression_nets: dict = field(default_factory=dict)

    @property
    def _feature_slots(self) -> list[tuple[str, dict]]:
        """List every named feature slot alongside its loaded-model dict, shared by the count and offline-list properties below."""
        return [
            ("AGE", self.age_nets), ("GENDER", self.gender_nets),
            ("EMOTION", self.emotion_nets),
            ("RACE", self.race_nets),
            ("LIVENESS", self.liveness_nets),
            ("GAZE", self.gaze_nets),
            ("RECOGNITION", self.recognition_nets),
            ("GLASSES", self.glasses_nets),
            ("MASK", self.mask_nets), ("HAIR_COLOR", self.hair_color_nets),
            ("EYE_COLOR", self.eye_color_nets), ("COLORIZATION", self.colorization_nets),
            ("FACE_LANDMARKS", self.face_landmarks_nets),
            ("HANDS", self.hand_nets), ("RECONSTRUCTION_3D", self.reconstruction_3d_nets),
            ("FACE_DETECTOR_YOLO", self.yolo_face_nets),
            ("FACE_DETECTOR_SCRFD", self.scrfd_face_nets),
            ("FACE_DETECTOR_RETINAFACE", self.retinaface_nets),
            ("AGE_PROGRESSION", self.age_progression_nets),
        ]

    @property
    def loaded_feature_count(self) -> int:
        """Count feature slots with at least one model loaded."""
        return sum(1 for _, nets in self._feature_slots if nets)

    @property
    def total_feature_count(self) -> int:
        """Count all known feature slots, loaded or not."""
        return len(self._feature_slots)

    @property
    def offline_features(self) -> list[str]:
        """Return a list of feature slot names that currently have no models loaded."""
        return [
            name for name, nets in self._feature_slots if not nets
        ]
