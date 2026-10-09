"""Demo mode (FACE_ANALYZER_DEMO=1) hides persistence and live mode, and bounds image sizes."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from face_analyzer import inference
from face_analyzer.core.types import Models
from face_analyzer.demo import DEMO_MAX_IMAGE_SIDE, demo_manifest_entries
from face_analyzer.gallery import database, eigenfaces, search

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "src" / "face_analyzer" / "app.py")
PERSISTENCE_BUTTONS = {"Save face", "Search", "Scan all faces for recognition", "Enroll"}


def _forbidden(*_args, **_kwargs):
    """Stand in for any function that reads or writes saved faces or galleries."""
    raise AssertionError("demo mode touched a saved-face or gallery path")


def _fake_models() -> Models:
    """Return models with a detector and, deliberately, a recognition backend the demo must not expose."""
    return Models(retinaface_nets={"retinaface": object()}, age_nets={"fairface": object()},
                  recognition_nets={"lbph": True})


def _fake_analyze_frame(models, frame, config):
    """Return one face with an embedding, so every per-face action would normally render."""
    face = {
        "idx": 1, "box": [10, 10, 60, 60], "image": np.full((50, 50, 3), 128, np.uint8),
        "embedding": [0.5, 0.5, 0.5, 0.5], "raw_columns": {"age_fairface": "20-29"},
        "model_results": [{"Feature": "AGE", "Model": "fairface", "Output": "20-29"}],
    }
    return frame.copy(), [face], True, False


@pytest.fixture
def stubbed_app(monkeypatch):
    """Stub model loading and inference, and make every persistence function raise."""
    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(inference, "load_models", _fake_models)
    monkeypatch.setattr(inference, "analyze_frame", _fake_analyze_frame)
    for module in (inference, database, search, eigenfaces):
        for name in ("save_face", "save_gallery", "load_gallery", "enroll_lbph_face", "train_lbph_recognizer",
                     "match_face_eigenfaces", "match_faces_eigenfaces_batch", "build_gallery_from_directory"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, _forbidden)
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _upload_one_face(at: AppTest) -> AppTest:
    """Upload a small PNG through the image uploader and rerun."""
    png = cv2.imencode(".png", np.full((120, 120, 3), 200, np.uint8))[1].tobytes()
    at.file_uploader[0].set_value(("face.png", png, "image/png"))
    return at.run()


def test_demo_mode_hides_persistence_controls_and_live_mode(stubbed_app, monkeypatch):
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and not at.error
    assert any("never stored" in info.value for info in at.info)
    assert any("Live webcam mode runs locally only" in caption.value for caption in at.caption)
    assert not at.get("button_group"), "the Snapshot/Live capture switch must be hidden"
    sidebar_text = " ".join(md.value for md in at.sidebar.markdown)
    assert "Gallery" not in sidebar_text and "Identity search" not in sidebar_text

    at = _upload_one_face(at)

    assert not at.exception and not at.error
    assert any(metric.label == "Faces detected" for metric in at.metric)
    assert not PERSISTENCE_BUTTONS & {button.label for button in at.button}
    assert not [text for text in at.text_input if text.label in {"Enroll as", "Search directory (optional)"}]


def test_full_mode_still_shows_persistence_controls(stubbed_app, monkeypatch):
    monkeypatch.delenv("FACE_ANALYZER_DEMO", raising=False)
    monkeypatch.setattr(inference, "load_gallery", dict)
    at = _upload_one_face(AppTest.from_file(APP, default_timeout=60).run())
    assert not at.exception
    assert PERSISTENCE_BUTTONS <= {button.label for button in at.button}
    assert at.get("button_group")


def test_demo_manifest_entries_are_permissive_mirror_files_only():
    entries = json.loads((ROOT / "models" / "manifest.json").read_text())["files"]
    demo = demo_manifest_entries(entries, include_mediapipe=False)
    paths = {entry["path"] for entry in demo}
    assert {entry["source"] for entry in demo} == {"hf"}
    assert {"retinaface_mobilenet0.25.onnx", "fairface_7class.onnx", "emotion_ferplus.onnx",
            "haarcascade_eye.xml", "colorization_release_v2.caffemodel"} <= paths
    assert "opencv_face_detector_uint8.pb" not in paths
    assert "face_landmarker.task" not in paths
    assert "face_landmarker.task" in {e["path"] for e in demo_manifest_entries(entries, include_mediapipe=True)}


def test_decode_downscales_to_the_demo_side_and_rejects_oversized_headers():
    image = np.zeros((900, 3200, 3), np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    assert max(inference.decode_image_bytes(encoded, DEMO_MAX_IMAGE_SIDE).shape[:2]) == DEMO_MAX_IMAGE_SIDE
    with pytest.raises(ValueError, match="too large"):
        inference.decode_image_bytes(encoded, DEMO_MAX_IMAGE_SIDE, max_pixels=1_000_000)
