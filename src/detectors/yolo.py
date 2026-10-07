"""YOLOv8-Face detection backend using ONNX Runtime."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from src.core.constants import (
        YOLO_FACE_INPUT_SIZE,
        YOLO_FACE_IOU_THRESHOLD,
        YOLO_FACE_STRIDES,
    )
except ImportError:
    from core.constants import (
        YOLO_FACE_INPUT_SIZE,
        YOLO_FACE_IOU_THRESHOLD,
        YOLO_FACE_STRIDES,
    )


def _yolo_letterbox(
    image: np.ndarray, target_size: int = YOLO_FACE_INPUT_SIZE
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio and pad to a square target size."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_size - new_w) / 2, (target_size - new_h) / 2
    top, bottom = int(dh), int(target_size - new_h - int(dh))
    left, right = int(dw), int(target_size - new_w - int(dw))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (dw, dh)


def _yolo_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def detect_faces_yolo(session: Any, frame: np.ndarray | None, conf_threshold: float = 0.5) -> list[list[int]]:
    """Detect faces using YOLOv8-Face ONNX session and return [x1, y1, x2, y2] bounding boxes."""
    if session is None or frame is None or getattr(frame, "size", 0) == 0:
        return []
    letterboxed, scale, (dw, dh) = _yolo_letterbox(frame)
    blob = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    all_boxes, all_scores = [], []
    for pred, stride in zip(outputs, YOLO_FACE_STRIDES, strict=False):
        _, channels, h, w = pred.shape
        pred = pred.reshape(1, channels, -1).transpose(0, 2, 1)[0]  # (H*W, 80)

        grid_y, grid_x = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()

        bbox_pred = pred[:, :64].reshape(-1, 4, 16)
        bbox_dist = _yolo_softmax(bbox_pred, axis=-1) @ np.arange(16)
        cls_conf = 1 / (1 + np.exp(-pred[:, 64]))  # sigmoid

        x1 = (grid_x - bbox_dist[:, 0]) * stride
        y1 = (grid_y - bbox_dist[:, 1]) * stride
        x2 = (grid_x + bbox_dist[:, 2]) * stride
        y2 = (grid_y + bbox_dist[:, 3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
        all_scores.append(cls_conf)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    mask = scores >= conf_threshold
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return []

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, YOLO_FACE_IOU_THRESHOLD)
    if len(keep) == 0:
        return []
    boxes = boxes[np.array(keep).flatten()]

    # Undo the letterbox padding/scale to map back to frame's own coordinates.
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= scale
    frame_h, frame_w = frame.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)

    return boxes.astype(int).tolist()
