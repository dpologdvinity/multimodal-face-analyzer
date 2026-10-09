"""Parity: the age model's training preprocessing must rebuild the app's exact input tensor."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_analyzer.core import constants as C
from tests._models import require_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "kaggle" / "age_model"))

import faceprep  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "crew_portrait.jpg"


class _RecordingNet:
    """Stand-in for a cv2.dnn net that keeps every blob the app feeds it."""

    def __init__(self):
        self.blobs = []

    def setInput(self, blob):  # noqa: N802 (cv2.dnn API name)
        self.blobs.append(np.array(blob, copy=True))

    def forward(self, _name=None):
        return np.zeros((1, 9), np.float32)


@pytest.fixture(scope="module")
def models():
    """Load the preparation models (the face landmarker too, when mediapipe is installed)."""
    for path in (C.FACE_PROTO, C.FACE_MODEL, C.EYE_CASCADE_FILE):
        require_model(path)
    return faceprep.load_prep_models()


def _app_blobs(models, frame: np.ndarray) -> list[np.ndarray]:
    """Run analyze_frame with a recording FairFace age net; return the blobs it received."""
    from face_analyzer.inference import AnalysisConfig, analyze_frame

    net = _RecordingNet()
    models.age_nets["fairface"] = net
    try:
        analyze_frame(models, frame, AnalysisConfig(
            conf_threshold=faceprep.DETECTION_CONF, face_detector="ssd", active_age={"fairface"}))
    finally:
        del models.age_nets["fairface"]
    return net.blobs


def _crop_around_largest_face(models, frame: np.ndarray) -> np.ndarray:
    """Cut a FairFace-like single-face image (face plus generous context) out of the group photo."""
    from face_analyzer.pipeline import stages

    boxes = stages._detect(models, frame, faceprep.prep_config())
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    pad = int(1.25 * max(x2 - x1, y2 - y1))
    return frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()


def test_training_rebuild_matches_app_input(models):
    frame = _crop_around_largest_face(models, cv2.imread(str(FIXTURE)))
    app = _app_blobs(models, frame)
    result = faceprep.face_record(models, frame)
    assert result is not None and len(app) >= 1
    record, _ = result
    training = faceprep.input_blob(cv2.imdecode(cv2.imencode(".png", frame)[1], cv2.IMREAD_COLOR), record)
    assert training.shape == (1, 3, 224, 224) and training.dtype == np.float32
    # analyze_frame runs every detected face; the largest in-frame box is the one training keeps.
    assert any(np.array_equal(training, blob) for blob in app)


def test_rebuild_with_roll_and_landmarks_matches_app_crop(models):
    frame = _crop_around_largest_face(models, cv2.imread(str(FIXTURE)))
    rotated = cv2.warpAffine(frame, cv2.getRotationMatrix2D(
        (frame.shape[1] / 2, frame.shape[0] / 2), -10, 1.0), frame.shape[1::-1])
    for image in (frame, rotated):
        result = faceprep.face_record(models, image)
        assert result is not None
        record, inputs = result
        np.testing.assert_array_equal(faceprep.aligned_face(image, record), faceprep.app_aligned_face(inputs))
    # The tilted copy must exercise the roll-leveling branch, or this test proves less than it claims.
    assert record["angle"] is not None and abs(record["angle"]) > 3


def test_rebuild_ignores_missing_landmarks():
    frame = np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    record = {"box": [50, 50, 150, 150], "angle": None, "landmarks": np.full((5, 2), np.nan, np.float32)}
    expected = faceprep.fairface_aligned_face(frame, (50, 50, 150, 150), None)
    np.testing.assert_array_equal(faceprep.aligned_face(frame, record), expected)
