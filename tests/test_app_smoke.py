"""Smoke tests that execute the Streamlit script headlessly via AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from face_analyzer.core.constants import MODEL_DIR
from tests._models import require_all_models

APP = str(Path(__file__).resolve().parents[1] / "src" / "face_analyzer" / "app.py")


@pytest.fixture(autouse=True)
def _real_models_present():
    """Skip when models are git-lfs pointers: load_models() would fail to parse them."""
    require_all_models(MODEL_DIR)


def _run() -> AppTest:
    """Run the app script once and return the finished AppTest."""
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_boots_without_exception():
    assert not _run().exception


def test_app_renders_upload_and_webcam_tabs():
    labels = [t.label.upper() for t in _run().tabs]
    assert any("UPLOAD" in label for label in labels)
    assert any("WEBCAM" in label or "LIVE" in label for label in labels)


def test_sliders_present_and_adjustable_without_exception():
    at = _run()
    assert len(at.slider) >= 1
    first = at.slider[0]
    at = first.set_value(first.min).run()
    assert not at.exception


def test_sidebar_toggling_a_model_checkbox_keeps_app_alive():
    at = _run()
    assert at.sidebar.checkbox
    box = at.sidebar.checkbox[0]
    at = box.set_value(not box.value).run()
    assert not at.exception



def test_sidebar_renders_model_checkboxes():
    assert len(_run().sidebar.checkbox) >= 1


def test_changing_confidence_threshold_reruns_without_exception():
    at = _run()
    slider = at.slider[0]
    assert slider.label == "Confidence threshold"
    at = slider.set_value(0.5).run()
    assert not at.exception
