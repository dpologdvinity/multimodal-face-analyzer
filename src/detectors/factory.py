"""Face detector factory and unified dispatch."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from src.core.types import Models
except ImportError:
    from core.types import Models

from .retinaface import detect_faces_retinaface
from .scrfd import detect_faces_scrfd
from .ssd import detect_faces_ssd
from .yolo import detect_faces_yolo


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


def detect_faces(
    models: Any,
    frame: np.ndarray | None = None,
    conf_threshold: float = 0.5,
    face_detector: str = "yolo",
) -> list[list[int]]:
    """Detect faces by dispatching to the configured detector backend with SSD fallback."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    # Support legacy bare net passed as first argument: detect_faces(net, frame, conf_threshold)
    if not _is_models_container(models):
        return detect_faces_ssd(models, frame, conf_threshold=conf_threshold)

    if face_detector == "yolo":
        yolo_net = getattr(models, "yolo_face_nets", {}).get("yolo")
        if yolo_net is not None:
            return detect_faces_yolo(yolo_net, frame, conf_threshold=conf_threshold)
    elif face_detector == "scrfd":
        scrfd_net = getattr(models, "scrfd_face_nets", {}).get("scrfd")
        if scrfd_net is not None:
            return detect_faces_scrfd(scrfd_net, frame, conf_threshold=conf_threshold)
    elif face_detector == "retinaface":
        retinaface_net = getattr(models, "retinaface_nets", {}).get("retinaface")
        if retinaface_net is not None:
            return detect_faces_retinaface(retinaface_net, frame, conf_threshold=conf_threshold)

    # Fallback to SSD detector
    ssd_net = getattr(models, "face_net", None)
    if ssd_net is not None:
        return detect_faces_ssd(ssd_net, frame, conf_threshold=conf_threshold)
    return []
