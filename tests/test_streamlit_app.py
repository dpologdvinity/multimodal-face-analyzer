"""The streamlit_app.py demo entrypoint boots in demo mode when the demo weights are present."""
import importlib.util
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from face_analyzer.core.constants import MODEL_DIR
from face_analyzer.demo import demo_manifest_entries
from tests._models import require_model

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = str(ROOT / "streamlit_app.py")


@pytest.fixture
def demo_weights_present():
    """Skip unless every demo weight file is already present, so the boot never downloads."""
    pytest.importorskip("onnxruntime")
    entries = json.loads((ROOT / "models" / "manifest.json").read_text())["files"]
    include_mediapipe = importlib.util.find_spec("mediapipe") is not None
    for entry in demo_manifest_entries(entries, include_mediapipe=include_mediapipe):
        require_model(MODEL_DIR / entry["path"])


@pytest.fixture
def fresh_caches(monkeypatch):
    """Isolate the entrypoint's demo-mode model cache and environment from the other app tests."""
    # The entrypoint sets FACE_ANALYZER_DEMO itself; registering it here lets monkeypatch undo it.
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def test_entrypoint_boots_in_demo_mode(demo_weights_present, fresh_caches):
    at = AppTest.from_file(ENTRYPOINT, default_timeout=120).run()

    assert not at.exception
    assert not at.error
    assert any("never stored" in info.value for info in at.info)
    assert not at.get("button_group")
    labels = {checkbox.label for checkbox in at.sidebar.checkbox}
    assert {"FairFace", "FERPlus"} <= labels
