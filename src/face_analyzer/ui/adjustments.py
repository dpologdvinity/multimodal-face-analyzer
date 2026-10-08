"""Image adjustment sliders shared by the batch-wide and per-face result views."""
from __future__ import annotations

import streamlit as st

from .. import inference


def reset_adjustments(prefixes: tuple[str, ...]) -> None:
    """Reset image adjustment sliders to their default values."""
    for state_key in list(st.session_state):
        if not any(state_key.startswith(f"{prefix}_") for prefix in prefixes):
            continue
        for adj_key, (_, _, default) in inference.IMAGE_ADJUSTMENT_RANGES.items():
            if state_key.endswith(f"_{adj_key}"):
                st.session_state[state_key] = default
                break


def adjustment_sliders(caption: str, key_prefix: str, column_count: int = 2) -> dict:
    """Render brightness/contrast/saturation sliders and return their current values."""
    st.caption(caption)
    if st.button("Reset these sliders", key=f"{key_prefix}_reset"):
        reset_adjustments((key_prefix,))
    values = {}
    columns = st.columns(column_count)
    for index, (adj_key, (adj_min, adj_max, adj_default)) in enumerate(inference.IMAGE_ADJUSTMENT_RANGES.items()):
        with columns[index % column_count]:
            values[adj_key] = st.slider(
                adj_key.replace("_", " ").title(), adj_min, adj_max, adj_default, key=f"{key_prefix}_{adj_key}"
            )
    return values

