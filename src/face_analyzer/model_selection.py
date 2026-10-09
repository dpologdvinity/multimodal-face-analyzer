"""Runtime filtering for model backends selected by the native installer or demo mode."""

import os

from .demo import demo_model_allowed


def native_model_selected(environment_name: str, model_name: str) -> bool:
    """Return whether a native installer selection (and demo mode) allows a model backend.

    An unset variable preserves normal model discovery for direct launches and Docker
    images. An empty variable intentionally means that the feature was not selected.
    Demo mode additionally limits every feature to its DEMO_MODELS backends.
    """
    if not demo_model_allowed(environment_name, model_name):
        return False
    selected = os.environ.get(environment_name)
    if selected is None:
        return True
    return model_name in {name for name in selected.split(",") if name}
