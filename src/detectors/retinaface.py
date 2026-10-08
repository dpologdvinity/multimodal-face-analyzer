"""RetinaFace detection backend using ONNX Runtime."""
from __future__ import annotations

import functools
import itertools
from math import ceil
from typing import Any

import cv2
import numpy as np

try:
    from src.core.constants import (
        RETINAFACE_INPUT_HEIGHT,
        RETINAFACE_INPUT_WIDTH,
        RETINAFACE_MEAN,
        RETINAFACE_MIN_SIZES,
        RETINAFACE_NMS_THRESHOLD,
        RETINAFACE_STEPS,
        RETINAFACE_VARIANCE,
    )
except ImportError:
    from core.constants import (
        RETINAFACE_INPUT_HEIGHT,
        RETINAFACE_INPUT_WIDTH,
        RETINAFACE_MEAN,
        RETINAFACE_MIN_SIZES,
        RETINAFACE_NMS_THRESHOLD,
        RETINAFACE_STEPS,
        RETINAFACE_VARIANCE,
    )


def _retinaface_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@functools.lru_cache(maxsize=1)
def _retinaface_priors() -> np.ndarray:
    """Compute and cache normalized anchor prior boxes for RetinaFace."""
    feature_maps = [(ceil(RETINAFACE_INPUT_HEIGHT / s), ceil(RETINAFACE_INPUT_WIDTH / s)) for s in RETINAFACE_STEPS]
    anchors = []
    for k, (fm_h, fm_w) in enumerate(feature_maps):
        for i, j in itertools.product(range(fm_h), range(fm_w)):
            for min_size in RETINAFACE_MIN_SIZES[k]:
                s_kx = min_size / RETINAFACE_INPUT_WIDTH
                s_ky = min_size / RETINAFACE_INPUT_HEIGHT
                cx = (j + 0.5) * RETINAFACE_STEPS[k] / RETINAFACE_INPUT_WIDTH
                cy = (i + 0.5) * RETINAFACE_STEPS[k] / RETINAFACE_INPUT_HEIGHT
                anchors.append([cx, cy, s_kx, s_ky])
    return np.array(anchors, dtype=np.float32)


def _retinaface_decode(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Decode RetinaFace localization predictions against anchor priors."""
    boxes = np.concatenate(
        [
            priors[:, :2] + loc[:, :2] * RETINAFACE_VARIANCE[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * RETINAFACE_VARIANCE[1]),
        ],
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def detect_faces_retinaface(session: Any, frame: np.ndarray | None, conf_threshold: float = 0.5) -> list[list[int]]:
    """Detect faces using RetinaFace ONNX session and return [x1, y1, x2, y2] bounding boxes."""
    if session is None or frame is None or getattr(frame, "size", 0) == 0:
        return []
    frame_h, frame_w = frame.shape[:2]
    scale = min(RETINAFACE_INPUT_HEIGHT / frame_h, RETINAFACE_INPUT_WIDTH / frame_w)
    new_h, new_w = int(frame_h * scale), int(frame_w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((RETINAFACE_INPUT_HEIGHT, RETINAFACE_INPUT_WIDTH, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = resized.astype(np.float32)
    canvas -= RETINAFACE_MEAN
    blob = canvas[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    loc, conf, _landm = session.run(None, {input_name: blob})
    loc, conf = loc[0], conf[0]

    boxes = _retinaface_decode(loc, _retinaface_priors())
    boxes[:, 0::2] *= RETINAFACE_INPUT_WIDTH
    boxes[:, 1::2] *= RETINAFACE_INPUT_HEIGHT
    scores = _retinaface_softmax(conf, axis=-1)[:, 1]

    mask = scores > conf_threshold
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return []

    nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, RETINAFACE_NMS_THRESHOLD)
    if len(keep) == 0:
        return []
    boxes = boxes[np.array(keep).flatten()] / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)

    return boxes.astype(int).tolist()
