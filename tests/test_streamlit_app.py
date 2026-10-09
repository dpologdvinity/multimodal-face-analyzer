"""The streamlit_app.py demo entrypoint boots in demo mode when the demo weights are present."""
import importlib.util
import json
import os
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from face_analyzer.core.constants import MODEL_DIR
from face_analyzer.demo import demo_manifest_entries
from tests._models import require_model

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = str(ROOT / "streamlit_app.py")

# Imported under another name, so its main() (which runs only as __main__) stays unexecuted.
_SPEC = importlib.util.spec_from_file_location("streamlit_app_under_test", ENTRYPOINT)
streamlit_app = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(streamlit_app)


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
    error_details = st.get_option("client.showErrorDetails")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()
    st.set_option("client.showErrorDetails", error_details)


def test_entrypoint_boots_in_demo_mode(demo_weights_present, fresh_caches):
    at = AppTest.from_file(ENTRYPOINT, default_timeout=120).run()

    assert not at.exception
    assert not at.error
    assert any("never stored" in info.value for info in at.info)
    assert not at.get("button_group")
    labels = {checkbox.label for checkbox in at.sidebar.checkbox}
    assert {"FairFace", "FERPlus"} <= labels


def _fresh_state() -> types.ModuleType:
    state = types.ModuleType("reload_state_under_test")
    state.lock = threading.Lock()
    state.fingerprint = None
    return state


def test_a_source_change_reloads_exactly_once_across_concurrent_runs(tmp_path):
    source = tmp_path / "face_analyzer"
    source.mkdir()
    module = source / "demo.py"
    module.write_text("SLOT = 1\n")
    state, purges = _fresh_state(), []

    def purge():
        purges.append(1)
        return True

    def run():
        return streamlit_app.reload_if_source_changed(state, streamlit_app.source_fingerprint(source), purge)

    assert [run(), run()] == [False, False]
    stat = module.stat()
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(run())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False] * 7 + [True]
    assert len(purges) == 1
    assert run() is False and len(purges) == 1


def test_purge_waits_for_a_running_analysis_and_keeps_its_semaphore():
    import face_analyzer.demo as old_demo

    slot = old_demo._ANALYSIS_SLOT
    # patch.dict restores sys.modules afterwards, so the purge never leaks into other tests.
    with patch.dict(sys.modules), patch.object(st.cache_resource, "clear") as clear_resource, \
            patch.object(st.cache_data, "clear") as clear_data:
        slot.acquire()
        try:
            assert streamlit_app.purge_package() is False
            assert sys.modules["face_analyzer.demo"] is old_demo
        finally:
            slot.release()

        assert streamlit_app.purge_package() is True
        new_demo = sys.modules["face_analyzer.demo"]
        assert new_demo is not old_demo
        assert new_demo._ANALYSIS_SLOT is slot
        assert not any(name.startswith("face_analyzer.") and name != "face_analyzer.demo"
                       for name in sys.modules)
        clear_resource.assert_called_once()
        clear_data.assert_called_once()
    assert sys.modules["face_analyzer.demo"] is old_demo
