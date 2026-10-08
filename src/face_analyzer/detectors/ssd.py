"""SSD face detection backend using OpenCV DNN."""
from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np

_NET_LOCKS: dict[int, Any] = {}
_NET_LOCKS_GUARD = threading.Lock()


def _lock_for(net: Any) -> Any:
    """Return a lock keyed by net identity for thread-safe cv2.dnn forward calls."""
    if net is None or isinstance(net, bool):
        return nullcontext()
    key = id(net)
    lock = _NET_LOCKS.get(key)
    if lock is None:
        with _NET_LOCKS_GUARD:
            lock = _NET_LOCKS.setdefault(key, threading.Lock())
    return lock


def detect_faces_ssd(net: Any, frame: np.ndarray | None, conf_threshold: float = 0.5) -> list[list[int]]:
    """Detect faces using OpenCV DNN SSD and return [x1, y1, x2, y2] bounding boxes."""
    if net is None or frame is None or getattr(frame, "size", 0) == 0:
        return []
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    with _lock_for(net):
        net.setInput(blob)
        detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes
