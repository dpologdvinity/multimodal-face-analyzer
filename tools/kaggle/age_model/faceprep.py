"""Face preprocessing shared by the age model's Kaggle kernels, the evals and the parity test.

The age model is trained on exactly the input the app gives its FairFace-style backends: the
SSD detector (largest in-frame box, as the held-out eval keeps), the roll leveling and MediaPipe
landmarks of the app's own per-face preparation, then the landmark-aligned (or margin-cropped)
224x224 face with ImageNet normalization. ``face_record`` runs that app code once per image and
keeps the few numbers that determine the crop (box, roll angle, landmarks), so training can
rebuild the identical crop with ``aligned_face`` from the original image on every epoch.

Torch-free on purpose: the prep kernel, the eval tools and the tests import it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_heldout import largest_in_frame  # noqa: E402  (tools/ is not a package)

from face_analyzer.attributes import (  # noqa: E402
    fairface_aligned_face,
    imagenet_blob,
    level_face_region,
)

DETECTION_CONF = 0.5
# Every model-selection variable the loader reads; empty means "not selected" (model_selection.py).
# Only the face landmarker stays selected; the SSD detector and eye cascade always load.
_SELECTION_ENV = (
    "AGE_MODEL", "GENDER_MODEL", "EMOTION_MODEL", "RACE_MODEL", "LIVENESS_MODEL",
    "RECOGNITION_MODEL", "GLASSES_MODEL", "MASK_MODEL", "COLORIZATION_MODEL", "HAND_MODEL",
    "RECONSTRUCTION_3D_MODEL", "AGE_PROGRESSION_MODEL", "YOLO_FACE_MODEL", "SCRFD_FACE_MODEL",
    "RETINAFACE_MODEL", "HAIR_COLOR_MODEL",
)


def load_prep_models():
    """Load only what face preparation needs: SSD, the eye cascade and the face landmarker."""
    from face_analyzer.inference import load_models

    wanted = {**dict.fromkeys(_SELECTION_ENV, ""), "FACE_LANDMARKS_MODEL": "mediapipe"}
    saved = {name: os.environ.get(name) for name in wanted}
    os.environ.update(wanted)
    try:
        return load_models()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prep_config():
    """Return the AnalysisConfig preparation runs under: SSD at the app's default threshold."""
    from face_analyzer.inference import AnalysisConfig

    return AnalysisConfig(conf_threshold=DETECTION_CONF, face_detector="ssd")


def face_record(models, frame: np.ndarray) -> tuple[dict, object] | None:
    """Run the app's detection and per-face preparation; return (crop record, _FaceInputs).

    None when the detector finds no box overlapping the frame or the face crop is empty.
    """
    from face_analyzer.pipeline import stages

    config = prep_config()
    boxes, _ = stages._detect(models, frame, config)
    box = largest_in_frame(boxes, *frame.shape[:2])
    if box is None:
        return None
    angle: dict = {}
    original = stages._cached_face_predict

    def recording(feature, model_key, face_bgr, predict_fn, *args):
        value = original(feature, model_key, face_bgr, predict_fn, *args)
        if feature == "roll_angle":
            angle["value"] = value
        return value

    stages._cached_face_predict = recording
    try:
        inputs = stages._prepare_face(
            models, frame, tuple(box), None, config, stages._frame_inputs(models, config))
    finally:
        stages._cached_face_predict = original
    if inputs is None:
        return None
    landmarks = inputs.fairface_landmarks
    record = {
        "box": [int(v) for v in box],
        "angle": None if angle.get("value") is None else float(angle["value"]),
        "landmarks": None if landmarks is None else np.asarray(landmarks, np.float32),
    }
    return record, inputs


def aligned_face(frame: np.ndarray, record: dict) -> np.ndarray:
    """Rebuild the app's 224x224 BGR aligned face from the original image and a crop record."""
    crop_frame, crop_box = level_face_region(frame, tuple(record["box"]), record["angle"])
    landmarks = record["landmarks"]
    if landmarks is not None:
        landmarks = np.asarray(landmarks, np.float32)
        if not np.isfinite(landmarks).all():
            landmarks = None
    return fairface_aligned_face(crop_frame, crop_box, landmarks)


def input_blob(frame: np.ndarray, record: dict) -> np.ndarray:
    """Return the 1x3x224x224 normalized blob the app feeds the age model for this record."""
    return imagenet_blob(aligned_face(frame, record))


def app_aligned_face(inputs) -> np.ndarray:
    """Return the aligned face the app itself builds from a _FaceInputs (the parity reference)."""
    return fairface_aligned_face(inputs.crop_frame, inputs.crop_box, inputs.fairface_landmarks)


def age_probabilities(net, inputs) -> np.ndarray:
    """Run a cv2.dnn age model on the app's aligned face for a _FaceInputs; return bucket probabilities."""
    net.setInput(imagenet_blob(app_aligned_face(inputs)))
    logits = net.forward().reshape(-1).astype(np.float64)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()
