"""Smoke tests that execute the Streamlit script headlessly via AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


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

