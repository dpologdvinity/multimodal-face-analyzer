"""Gender classification models across Caffe, MiVOLO, FairFace, and DeepFace."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import GENDER_LIST
from .age import caffe_probabilities, mivolo_estimate
from .race import deepface_probabilities, fairface_gender_label, fairface_probabilities


def predict_gender_caffe(net: Any, blob: np.ndarray) -> str:
    """Predict gender from a Caffe blob via argmax."""
    return GENDER_LIST[int(caffe_probabilities(net, blob).argmax())]


def predict_gender_mivolo(net: Any, face_bgr: np.ndarray) -> str:
    """Predict gender with MiVOLO model."""
    _, gender = mivolo_estimate(net, face_bgr)
    return gender


def predict_gender_fairface(
    net: Any,
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    landmarks: np.ndarray | None = None,
) -> str:
    """Predict gender via FairFace using landmarks when available."""
    return fairface_gender_label(fairface_probabilities(net, frame_bgr, box, "gender_output", landmarks))


def predict_gender_deepface(net: Any, face_bgr: np.ndarray) -> str:
    """Predict gender via DeepFace VGGFace model."""
    return "Woman" if np.argmax(deepface_probabilities(net, face_bgr)) == 0 else "Man"
