"""Public demo entry point: turn on demo mode, fetch the demo weights once, then run the app.

streamlit_app.py is Streamlit Community Cloud's default entrypoint name; see docs/deploy.md.
"""

import importlib.util
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Import the package straight from src/ rather than installing it: a non-editable install would
# move it into site-packages, where BASE_DIR (and so models/) no longer points at this checkout.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Set before any face_analyzer import, so every check of the flag sees it.
os.environ["FACE_ANALYZER_DEMO"] = "1"

import streamlit as st  # noqa: E402

from face_analyzer.demo import demo_manifest_entries  # noqa: E402


def _load_fetcher():
    """Load tools/fetch_models.py, which stays a standalone, standard-library-only script."""
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "tools" / "fetch_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_models = _load_fetcher()


@st.cache_resource(show_spinner="First start: downloading the demo model weights (about 255 MB)...")
def ensure_demo_weights() -> int:
    """Download and verify the demo's weight files once per process; return how many were checked.

    A failure raises, and Streamlit does not cache exceptions, so the next page load retries.
    """
    include_mediapipe = importlib.util.find_spec("mediapipe") is not None
    entries = demo_manifest_entries(fetch_models.load_manifest(), include_mediapipe=include_mediapipe)
    model_dir = fetch_models.default_model_dir()
    for entry in entries:
        fetch_models.fetch_entry(entry, model_dir)
    return len(entries)


try:
    ensure_demo_weights()
except fetch_models.FetchError as exc:
    st.error(f"Could not download the demo model weights: {exc}. Reload the page to retry.")
    st.stop()

runpy.run_path(str(ROOT / "src" / "face_analyzer" / "app.py"), run_name="__main__")
