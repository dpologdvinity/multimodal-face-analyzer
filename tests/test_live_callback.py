"""Tests for the live webcam frame callback, exercised without WebRTC or Streamlit."""
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from src.core.types import Models
from src.pipeline.config import AnalysisConfig
from src.ui import live


def _frame() -> av.VideoFrame:
    """Build a small black BGR video frame."""
    return av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format="bgr24")


@pytest.fixture(autouse=True)
def _clean_live_state():
    """Reset the module-level live state shared by every callback."""
    live.LIVE_STATE.update(faces=[], error=None, updated=0.0)
    live.LIVE_METRICS.clear()
    yield


def _models() -> Models:
    """Return a Models whose face detector is a stand-in (detection is patched in tests)."""
    return Models(face_net=MagicMock())


def test_normal_frame_returns_same_size_frame_and_records_metrics():
    """A normal frame comes back as a same-size VideoFrame and records one metrics sample."""
    config = AnalysisConfig(face_detector="ssd")
    callback = live.make_video_frame_callback(_models, lambda: config)
    with patch("src.pipeline.analyzer.detect_faces", return_value=[]):
        out = callback(_frame())
    assert isinstance(out, av.VideoFrame)
    assert (out.width, out.height) == (320, 240)
    assert live.LIVE_STATE["error"] is None
    assert len(live.LIVE_METRICS) == 1


def test_frame_skip_runs_classifiers_only_every_nth_frame():
    """Skipped frames blank only the slow classifiers and never mutate the shared config."""
    config = AnalysisConfig(
        face_detector="ssd", active_age={"caffe"}, active_gender={"caffe"},
        active_emotion={"hsemotion"}, active_race={"fairface"}, active_recognition={"x"},
        active_glasses={"x"}, active_mask={"x"}, active_hair_color={"x"}, active_eye_color={"x"},
        active_gaze={"mediapipe"}, active_face_landmarks={"mediapipe"}, active_hands={"mediapipe"},
        active_liveness={"mediapipe"},
    )
    seen = []

    def spy(models, img, cfg):
        seen.append(cfg)
        return img, [], False, False

    callback = live.make_video_frame_callback(_models, lambda: config, frame_skip=2)
    with patch("src.ui.live.analyze_frame", side_effect=spy):
        outputs = [callback(_frame()), callback(_frame())]
    assert all(isinstance(out, av.VideoFrame) for out in outputs)
    assert all((out.width, out.height) == (320, 240) for out in outputs)

    skipped, ran = seen
    for field in ("active_age", "active_gender", "active_emotion", "active_race", "active_recognition",
                  "active_glasses", "active_mask", "active_hair_color", "active_eye_color", "active_gaze"):
        assert getattr(skipped, field) == set(), field
        assert getattr(ran, field) == getattr(config, field), field
    # Overlays and liveness stay on every frame; only the classifiers are throttled.
    for cfg in seen:
        assert cfg.active_face_landmarks == {"mediapipe"}
        assert cfg.active_hands == {"mediapipe"}
        assert cfg.active_liveness == {"mediapipe"}
    # The shared config must not be mutated by throttling or per-frame metrics.
    assert config.active_age == {"caffe"}
    assert config.metrics is None


def test_frame_skip_path_survives_the_real_pipeline():
    """Both the skipped and the classifier frame run through the real analyze_frame cleanly."""
    config = AnalysisConfig(face_detector="ssd", active_age={"caffe"})
    callback = live.make_video_frame_callback(_models, lambda: config, frame_skip=2)
    with patch("src.pipeline.analyzer.detect_faces", return_value=[]):
        skipped = callback(_frame())
        ran = callback(_frame())
    assert (skipped.width, skipped.height) == (ran.width, ran.height) == (320, 240)
    assert live.LIVE_STATE["error"] is None


def test_inference_failure_returns_raw_frame_and_records_error():
    """An inference error passes the raw frame through and is surfaced in LIVE_STATE."""
    callback = live.make_video_frame_callback(_models, AnalysisConfig)
    frame = _frame()
    with patch("src.ui.live.analyze_frame", side_effect=RuntimeError("boom")):
        out = callback(frame)
    assert out is frame
    assert live.LIVE_STATE["error"] == "RuntimeError: boom"
