"""SCRFD face detection backend using ONNX Runtime."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

try:
    from src.core.constants import (
        SCRFD_FACE_INPUT_SIZE,
        SCRFD_FACE_STRIDES,
        SCRFD_FACE_NUM_ANCHORS,
        SCRFD_FACE_NMS_THRESHOLD,
    )
except ImportError:
    from core.constants import (
        SCRFD_FACE_INPUT_SIZE,
        SCRFD_FACE_STRIDES,
        SCRFD_FACE_NUM_ANCHORS,
        SCRFD_FACE_NMS_THRESHOLD,
    )


def _scrfd_distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode SCRFD distance regression offsets into [x1, y1, x2, y2] bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def detect_faces_scrfd(session: Any, frame: np.ndarray | None, conf_threshold: float = 0.5) -> list[list[int]]:
    """Detect faces using SCRFD ONNX session and return [x1, y1, x2, y2] bounding boxes."""
    if session is None or frame is None or getattr(frame, "size", 0) == 0:
        return []
    frame_h, frame_w = frame.shape[:2]
    scale = SCRFD_FACE_INPUT_SIZE / max(frame_h, frame_w)
    resized = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((SCRFD_FACE_INPUT_SIZE, SCRFD_FACE_INPUT_SIZE, 3), dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized

    blob = cv2.dnn.blobFromImage(
        padded, 1.0 / 128, (SCRFD_FACE_INPUT_SIZE, SCRFD_FACE_INPUT_SIZE), (127.5, 127.5, 127.5), swapRB=True
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    all_boxes, all_scores = [], []
    for idx, stride in enumerate(SCRFD_FACE_STRIDES):
        scores = outputs[idx]
        bbox_preds = outputs[3 + idx] * stride
        fm_size = SCRFD_FACE_INPUT_SIZE // stride
        anchor_centers = np.stack(np.mgrid[:fm_size, :fm_size][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape(-1, 2)
        anchor_centers = np.repeat(anchor_centers, SCRFD_FACE_NUM_ANCHORS, axis=0)

        mask = scores[:, 0] > conf_threshold
        if not np.any(mask):
            continue
        all_boxes.append(_scrfd_distance2bbox(anchor_centers[mask], bbox_preds[mask]))
        all_scores.append(scores[mask, 0])

    if not all_scores:
        return []

    boxes = np.concatenate(all_boxes, axis=0) / scale
    scores = np.concatenate(all_scores, axis=0)

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, SCRFD_FACE_NMS_THRESHOLD)
    if len(keep) == 0:
        return []
    boxes = boxes[np.array(keep).flatten()]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)

    return boxes.astype(int).tolist()
