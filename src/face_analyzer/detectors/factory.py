"""Face detector factory and unified dispatch."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.types import Models
from .retinaface import detect_faces_retinaface
from .scrfd import detect_faces_scrfd
from .ssd import detect_faces_ssd
from .yolo import detect_faces_yolo

# Sidebar display order of the detector keys.
FACE_DETECTOR_DISPLAY_ORDER = ("yolo", "ssd", "scrfd", "retinaface")
# Used when the requested detector is not loaded: SSD first, as before it became optional.
_FALLBACK_ORDER = ("ssd", "retinaface", "yolo", "scrfd")
_NET_SLOTS = {"yolo": "yolo_face_nets", "scrfd": "scrfd_face_nets", "retinaface": "retinaface_nets"}


def _is_models_container(obj: Any) -> bool:
    """Return True if obj represents a Models container rather than a single net."""
    if isinstance(obj, Models):
        return True
    if isinstance(getattr(obj, "yolo_face_nets", None), dict):
        return True
    if isinstance(getattr(obj, "scrfd_face_nets", None), dict):
        return True
    if isinstance(getattr(obj, "retinaface_nets", None), dict):
        return True
    return False


def face_detector_net(models: Any, face_detector: str) -> Any:
    """Return the loaded net for a detector key, or None when it is not loaded."""
    if face_detector == "ssd":
        return getattr(models, "face_net", None)
    return (getattr(models, _NET_SLOTS.get(face_detector, ""), None) or {}).get(face_detector)


def available_face_detectors(models: Any) -> list[str]:
    """Return the keys of the loaded face detectors in sidebar display order."""
    return [key for key in FACE_DETECTOR_DISPLAY_ORDER if face_detector_net(models, key) is not None]


def resolve_face_detector(models: Any, face_detector: str) -> str | None:
    """Return the detector that will actually run: the requested one if loaded, else a fallback."""
    if face_detector_net(models, face_detector) is not None:
        return face_detector
    return next((key for key in _FALLBACK_ORDER if face_detector_net(models, key) is not None), None)


def detect_faces(
    models: Any,
    frame: np.ndarray | None = None,
    conf_threshold: float = 0.5,
    face_detector: str = "yolo",
) -> list[list[int]]:
    """Detect faces with the configured detector, falling back to SSD, then any loaded detector."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    # Support legacy bare net passed as first argument: detect_faces(net, frame, conf_threshold)
    if not _is_models_container(models):
        return detect_faces_ssd(models, frame, conf_threshold=conf_threshold) if models is not None else []

    resolved = resolve_face_detector(models, face_detector)
    if resolved is None:
        return []
    net = face_detector_net(models, resolved)
    if resolved == "yolo":
        return detect_faces_yolo(net, frame, conf_threshold=conf_threshold)
    if resolved == "scrfd":
        return detect_faces_scrfd(net, frame, conf_threshold=conf_threshold)
    if resolved == "retinaface":
        return detect_faces_retinaface(net, frame, conf_threshold=conf_threshold)
    return detect_faces_ssd(net, frame, conf_threshold=conf_threshold)
