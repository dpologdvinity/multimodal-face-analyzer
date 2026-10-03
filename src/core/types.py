"""Core data types and structures for the multimodal face analyzer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def __iter__(self):
        return iter((self.x1, self.y1, self.x2, self.y2))

    def __getitem__(self, idx: int):
        return (self.x1, self.y1, self.x2, self.y2)[idx]

    def __len__(self) -> int:
        return 4


@dataclass
class Detection:
    box: BoundingBox
    confidence: float = 1.0
    landmarks: Any = None
    class_id: int = 0
    label: str = "face"

    def __post_init__(self):
        if isinstance(self.box, (tuple, list)):
            self.box = BoundingBox(*self.box)


@dataclass
class FaceResult:
    idx: int = 0
    box: tuple[int, int, int, int] | BoundingBox = (0, 0, 0, 0)
    image: np.ndarray | None = None
    track_id: int | None = None
    age: list = field(default_factory=list)
    gender: list = field(default_factory=list)
    race: list = field(default_factory=list)
    emotion: list = field(default_factory=list)
    headline: dict = field(default_factory=dict)
    gaze: list = field(default_factory=list)
    eye_contact: list = field(default_factory=list)
    head_pose: list = field(default_factory=list)
    identity: list = field(default_factory=list)
    glasses: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    hair_color: list = field(default_factory=list)
    eye_color: list = field(default_factory=list)
    liveness: list = field(default_factory=list)
    liveness_status: str | None = None
    blink_count: int | None = None
    blink_rate: float | None = None
    texture_score: float | None = None
    texture_artifact: str | None = None
    embedding: list | None = None
    raw_columns: dict = field(default_factory=dict)
    model_results: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.box, BoundingBox):
            d["box"] = self.box.to_tuple()
        return d

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class Models:
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
        return [
            name for name, nets in self._feature_slots if not nets
        ]
