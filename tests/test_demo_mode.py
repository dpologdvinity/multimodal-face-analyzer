"""Demo mode (FACE_ANALYZER_DEMO=1) hides persistence and live mode, and bounds image sizes."""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from face_analyzer import demo as demo_module
from face_analyzer import inference
from face_analyzer.core.types import Models
from face_analyzer.demo import (
    DEMO_MAX_IMAGE_PIXELS,
    DEMO_MAX_IMAGE_SIDE,
    DemoBusyError,
    analysis_slot,
    demo_manifest_entries,
)
from face_analyzer.gallery import database, eigenfaces, search
from face_analyzer.pipeline import cache

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
    error_details = st.get_option("client.showErrorDetails")
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
    # Demo mode sets this process-wide option; restore it for the other app tests.
    st.set_option("client.showErrorDetails", error_details)


def _upload_one_face(at: AppTest, data: bytes | None = None) -> AppTest:
    """Upload a small PNG (or the given bytes, named face.png) through the image uploader and rerun."""
    png = data if data is not None else cv2.imencode(".png", np.full((120, 120, 3), 200, np.uint8))[1].tobytes()
    at.file_uploader[0].set_value(("face.png", png, "image/png"))
    return at.run()


def _tiny_hdr() -> bytes:
    """Return a 4x4 Radiance HDR image, which cv2 decodes as float32 and Pillow cannot read."""
    return cv2.imencode(".hdr", np.ones((4, 4, 3), np.float32))[1].tobytes()


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
    assert st.get_option("client.showErrorDetails") == "none"
    assert any("FairFace (Kärkkäinen and Joo, CC BY 4.0)" in caption.value for caption in at.caption)


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


def test_demo_mode_rejects_a_renamed_hdr_before_decoding(stubbed_app, monkeypatch):
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    at = _upload_one_face(AppTest.from_file(APP, default_timeout=60).run(), _tiny_hdr())
    assert not at.exception
    assert any("face.png" in error.value for error in at.error)
    assert not at.metric, "the file must not reach analysis"


def test_pixel_cap_rejects_unreadable_and_non_photo_formats():
    with pytest.raises(ValueError):
        inference.decode_image_bytes(_tiny_hdr(), DEMO_MAX_IMAGE_SIDE, DEMO_MAX_IMAGE_PIXELS)
    bmp = cv2.imencode(".bmp", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="not a JPEG, PNG or WebP"):
        inference.decode_image_bytes(bmp, DEMO_MAX_IMAGE_SIDE, DEMO_MAX_IMAGE_PIXELS)
    with pytest.raises(ValueError):
        inference.decode_image_bytes(b"not an image", DEMO_MAX_IMAGE_SIDE, DEMO_MAX_IMAGE_PIXELS)
    # Without a cap (full mode) cv2 still decodes whatever it supports.
    assert inference.decode_image_bytes(bmp).shape == (8, 8, 3)


def test_analysis_slot_waits_for_the_other_analysis_then_runs():
    waits = []
    demo_module._ANALYSIS_SLOT.acquire()
    threading.Timer(0.3, demo_module._ANALYSIS_SLOT.release).start()
    with analysis_slot(lambda: waits.append(time.monotonic())):
        assert waits, "on_wait must be called while another analysis holds the slot"
    with analysis_slot(lambda: waits.append("unexpected")):
        pass
    assert len(waits) == 1


def test_analysis_slot_times_out_with_a_busy_error(monkeypatch):
    monkeypatch.setattr(demo_module, "DEMO_ANALYSIS_WAIT_SECONDS", 0.1)
    demo_module._ANALYSIS_SLOT.acquire()
    try:
        with pytest.raises(DemoBusyError), analysis_slot(lambda: None):
            pass
    finally:
        demo_module._ANALYSIS_SLOT.release()


def test_demo_upload_queues_behind_another_analysis(stubbed_app, monkeypatch):
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    monkeypatch.setattr(demo_module, "DEMO_ANALYSIS_WAIT_SECONDS", 0.1)
    demo_module._ANALYSIS_SLOT.acquire()
    try:
        at = _upload_one_face(at)
    finally:
        demo_module._ANALYSIS_SLOT.release()
    assert not at.exception
    assert any("busy" in warning.value for warning in at.warning)
    assert not at.metric

    at = at.run()
    assert any(metric.label == "Faces detected" for metric in at.metric)


@pytest.mark.parametrize("writer", [
    lambda: database.save_face(np.zeros((8, 8, 3), np.uint8), {}),
    lambda: search.save_gallery({}),
    lambda: search.enroll_lbph_face("someone", np.zeros((8, 8, 3), np.uint8)),
])
def test_face_writers_refuse_in_demo_mode(writer, monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    for module, name in ((database, "FACES_DIR"), (database, "EIGEN_DIR"), (database, "FACES_DB_FILE"),
                         (search, "GALLERY_FILE"), (search, "LBPH_GALLERY_DIR")):
        monkeypatch.setattr(module, name, tmp_path / name)
    with pytest.raises(PermissionError, match="demo mode"):
        writer()
    assert not any(tmp_path.iterdir())


def test_prediction_cache_is_skipped_in_demo_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(cache, "_PREDICTION_CACHE", type(cache._PREDICTION_CACHE)())
    face = np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    for _ in range(2):
        cache._cached_face_predict("age", "demo-test", face, lambda: calls.append(1) or "x")
    assert len(calls) == 2 and not cache._PREDICTION_CACHE
    monkeypatch.delenv("FACE_ANALYZER_DEMO")
    for _ in range(2):
        cache._cached_face_predict("age", "demo-test", face, lambda: calls.append(1) or "x")
    assert len(calls) == 3 and len(cache._PREDICTION_CACHE) == 1


def test_prediction_cache_survives_concurrent_eviction(monkeypatch):
    monkeypatch.delenv("FACE_ANALYZER_DEMO", raising=False)
    monkeypatch.setattr(cache, "_PREDICTION_CACHE", type(cache._PREDICTION_CACHE)())
    monkeypatch.setattr(cache, "PREDICTION_CACHE_MAX_SIZE", 2)
    faces = [np.full((2, 2, 3), value, np.uint8) for value in range(6)]
    errors = []

    def hammer():
        try:
            for _ in range(300):
                for face in faces:
                    cache._cached_face_predict("age", "race-test", face, lambda: "x")
        except Exception as exc:  # noqa: BLE001 -- any error from the unlocked race fails the test
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(cache._PREDICTION_CACHE) <= 2
