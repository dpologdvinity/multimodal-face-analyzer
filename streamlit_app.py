"""Public demo entry point: turn on demo mode, fetch the demo weights once, then run the app.

streamlit_app.py is Streamlit Community Cloud's default entrypoint name; see docs/deploy.md.
"""

import importlib.util
import os
import runpy
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Import the package straight from src/ rather than installing it: a non-editable install would
# move it into site-packages, where BASE_DIR (and so models/) no longer points at this checkout.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

PACKAGE = "face_analyzer"
SOURCE_DIR = ROOT / "src" / PACKAGE
# Streamlit re-executes only this script, so reload state lives in sys.modules, which outlives
# every run. The name is outside the package, so purging the package never drops it; demo.py
# keeps its analysis semaphore there too (ANALYSIS_SLOT_HOLDER).
_STATE_MODULE = "_face_analyzer_reload_state"


def source_fingerprint(source_dir: Path = SOURCE_DIR) -> int:
    """Hash the path and modification time of every package source file."""
    return hash(tuple(sorted((str(path), path.stat().st_mtime_ns) for path in source_dir.rglob("*.py"))))


def reload_state() -> types.ModuleType:
    """Return the process-wide reload state: a lock and the fingerprint of the imported source."""
    # Each setdefault is atomic, so concurrent first runs share one module and one lock, whether
    # this or demo.py's semaphore created the module.
    state = sys.modules.setdefault(_STATE_MODULE, types.ModuleType(_STATE_MODULE))
    vars(state).setdefault("lock", threading.Lock())
    vars(state).setdefault("fingerprint", None)
    return state


def purge_package() -> bool:
    """Drop the package's modules and Streamlit's caches; False if an analysis is running.

    Holding the analysis slot keeps a model reload from stacking on a running analysis. The slot
    itself lives on the reload state module, so the re-imported demo.py reuses it.
    """
    slot = vars(reload_state()).get("analysis_slot")
    if slot is not None and not slot.acquire(blocking=False):
        return False
    try:
        for name in [name for name in sys.modules if name == PACKAGE or name.startswith(f"{PACKAGE}.")]:
            del sys.modules[name]
        st.cache_resource.clear()
        st.cache_data.clear()
    finally:
        if slot is not None:
            slot.release()
    return True


def reload_if_source_changed(
    state: types.ModuleType, fingerprint: int, purge: Callable[[], bool] = purge_package,
) -> bool:
    """Purge the package once when its source changed since this process imported it.

    The first call only records the fingerprint. A purge refused because an analysis is
    running leaves the old fingerprint, so a later run retries it.
    """
    with state.lock:
        if state.fingerprint is None:
            state.fingerprint = fingerprint
            return False
        if fingerprint == state.fingerprint or not purge():
            return False
        state.fingerprint = fingerprint
        return True


def _load_fetcher():
    """Load tools/fetch_models.py, which stays a standalone, standard-library-only script."""
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "tools" / "fetch_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Reload changed package source, fetch the demo weights, then run the app."""
    # A redeploy pulls new source into the running process, whose imported modules would
    # otherwise stay on the old code until a manual reboot.
    try:
        reload_if_source_changed(reload_state(), source_fingerprint())
    except OSError:
        pass  # a file vanished mid-pull; the next run sees the finished checkout

    # Set before any face_analyzer import, so every check of the flag sees it.
    os.environ["FACE_ANALYZER_DEMO"] = "1"

    # Visitors see a generic error instead of tracebacks with server paths; the log keeps details.
    st.set_option("client.showErrorDetails", "none")

    from face_analyzer.demo import demo_manifest_entries

    fetch_models = _load_fetcher()

    @st.cache_resource(show_spinner="First start: downloading the demo model weights (about 250 MB)...")
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
    except (fetch_models.FetchError, OSError) as exc:
        # Details (URLs, server paths) go to the log only, never to the visitor.
        print(f"demo weight fetch failed: {exc!r}", file=sys.stderr)
        st.error("Could not download the demo model weights. Reload the page in a minute to retry.")
        st.stop()

    runpy.run_path(str(ROOT / "src" / "face_analyzer" / "app.py"), run_name="__main__")


# Streamlit runs this file as __main__; tests import it under another name to reach the helpers.
if __name__ == "__main__":
    main()
