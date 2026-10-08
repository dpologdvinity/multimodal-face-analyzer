"""Runtime filtering for model backends selected by the native installer."""

import os


def native_model_selected(environment_name: str, model_name: str) -> bool:
    """Return whether a native installer selection allows a model backend.

    An unset variable preserves normal model discovery for direct launches and Docker
    images. An empty variable intentionally means that the feature was not selected.
    """
    selected = os.environ.get(environment_name)
    if selected is None:
        return True
    return model_name in {name for name in selected.split(",") if name}
