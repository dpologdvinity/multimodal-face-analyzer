"""Tests for the live webcam frame callback, exercised without WebRTC or Streamlit."""
import threading
from collections import deque
from unittest.mock import MagicMock, patch

import av
import numpy as np

from face_analyzer.core.types import Models
from face_analyzer.pipeline.config import AnalysisConfig
from face_analyzer.ui import live


def _frame() -> av.VideoFrame:
    """Build a small black BGR video frame."""
    return av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format="bgr24")


def _session_state() -> dict:
    """Build one session's live state, as src/face_analyzer/app.py does on every script run."""
    return {
        "live_state": {"faces": [], "error": None, "updated": 0.0},
        "live_state_lock": threading.Lock(),
        "live_metrics": deque(maxlen=120),
        "live_metrics_lock": threading.Lock(),
    }


def _models() -> Models:
    """Return a Models whose face detector is a stand-in (detection is patched in tests)."""
    return Models(face_net=MagicMock())


def test_normal_frame_returns_same_size_frame_and_records_metrics():
    """A normal frame comes back as a same-size VideoFrame and records one metrics sample."""
    config = AnalysisConfig(face_detector="ssd")
    state = _session_state()
    callback = live.make_video_frame_callback(_models, lambda: config, **state)
    with patch("face_analyzer.pipeline.stages.detect_faces", return_value=[]):
        out = callback(_frame())
    assert isinstance(out, av.VideoFrame)
    assert (out.width, out.height) == (320, 240)
    assert state["live_state"]["error"] is None
    assert len(state["live_metrics"]) == 1


def test_frame_skip_runs_classifiers_only_every_nth_frame():
    """Skipped frames blank only the slow classifiers and never mutate the shared config."""
    config = AnalysisConfig(
        face_detector="ssd", active_age={"caffe"}, active_gender={"caffe"},
        active_emotion={"hsemotion"}, active_race={"fairface"}, active_recognition={"x"},
        active_glasses={"x"}, active_mask={"x"}, active_eye_color={"x"},
        active_gaze={"mediapipe"}, active_face_landmarks={"mediapipe"}, active_hands={"mediapipe"},
        active_liveness={"mediapipe"},
    )
    seen = []

    def spy(models, img, cfg):
        seen.append(cfg)
        return img, [], False, False

    callback = live.make_video_frame_callback(_models, lambda: config, frame_skip=2, **_session_state())
    with patch("face_analyzer.ui.live.analyze_frame", side_effect=spy):
        outputs = [callback(_frame()), callback(_frame())]
    assert all(isinstance(out, av.VideoFrame) for out in outputs)
    assert all((out.width, out.height) == (320, 240) for out in outputs)

    skipped, ran = seen
    for field in ("active_age", "active_gender", "active_emotion", "active_race", "active_recognition",
                  "active_glasses", "active_mask", "active_eye_color", "active_gaze"):
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
    state = _session_state()
    callback = live.make_video_frame_callback(_models, lambda: config, frame_skip=2, **state)
    with patch("face_analyzer.pipeline.stages.detect_faces", return_value=[]):
        skipped = callback(_frame())
        ran = callback(_frame())
    assert (skipped.width, skipped.height) == (ran.width, ran.height) == (320, 240)
    assert state["live_state"]["error"] is None


def test_inference_failure_returns_raw_frame_and_records_error():
    """An inference error passes the raw frame through and is surfaced in the live state."""
    state = _session_state()
    callback = live.make_video_frame_callback(_models, AnalysisConfig, **state)
    frame = _frame()
    with patch("face_analyzer.ui.live.analyze_frame", side_effect=RuntimeError("boom")):
        out = callback(frame)
    assert out is frame
    assert state["live_state"]["error"] == "RuntimeError: boom"


def test_callbacks_with_separate_state_never_share_it():
    """Two sessions' callbacks keep their faces, errors and metrics apart (no module singletons)."""
    face = {"idx": 1, "model_results": [{"Feature": "IDENTITY", "Model": "x", "Output": "Alice"}]}

    def fake_analyze(models, img, cfg):
        if cfg.face_detector == "fail":
            raise RuntimeError("boom")
        return img, [face], True, False

    first, second = _session_state(), _session_state()
    first_callback = live.make_video_frame_callback(
        _models, lambda: AnalysisConfig(face_detector="ssd"), **first,
    )
    second_callback = live.make_video_frame_callback(
        _models, lambda: AnalysisConfig(face_detector="fail"), **second,
    )
    with patch("face_analyzer.ui.live.analyze_frame", side_effect=fake_analyze):
        first_callback(_frame())
        second_callback(_frame())

    assert first["live_state"]["faces"] == [face]
    assert first["live_state"]["error"] is None
    assert len(first["live_metrics"]) == 1
    assert second["live_state"]["faces"] == []
    assert second["live_state"]["error"] == "RuntimeError: boom"
    assert len(second["live_metrics"]) == 0
    assert not any(name.startswith("LIVE_") for name in vars(live))


def test_new_live_session_returns_fresh_state_matching_callback_kwargs():
    """Each script run gets its own live state and locks, never shared objects."""
    first, second = live.new_live_session(), live.new_live_session()
    assert first.keys() == _session_state().keys()
    for key in first:
        assert first[key] is not second[key]
    live.make_video_frame_callback(_models, AnalysisConfig, **first)
