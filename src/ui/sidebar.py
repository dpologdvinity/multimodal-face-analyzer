"""Sidebar controls: model selection, gallery, identity search, and the control panel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

try:
    from src import inference
except ImportError:
    import inference


@dataclass
class SidebarState:
    """Every value the sidebar controls produce for one script run."""

    conf_threshold: float
    face_detector: str
    active_age: set[str] = field(default_factory=set)
    active_gender: set[str] = field(default_factory=set)
    active_race: set[str] = field(default_factory=set)
    active_emotion: set[str] = field(default_factory=set)
    active_glasses: set[str] = field(default_factory=set)
    active_mask: set[str] = field(default_factory=set)
    active_hair_color: set[str] = field(default_factory=set)
    active_eye_color: set[str] = field(default_factory=set)
    active_recognition: set[str] = field(default_factory=set)
    active_liveness: set[str] = field(default_factory=set)
    active_gaze: set[str] = field(default_factory=set)
    active_colorization: set[str] = field(default_factory=set)
    active_face_landmarks: set[str] = field(default_factory=set)
    active_hands: set[str] = field(default_factory=set)
    search_gallery: dict[str, Any] = field(default_factory=dict)
    enable_crowd_count: bool = False

    def to_config(self, **extra: Any) -> inference.AnalysisConfig:
        """Build the pipeline config from these selections plus per-call extras (gallery, trackers, ...)."""
        return inference.AnalysisConfig(
            conf_threshold=self.conf_threshold,
            active_age=self.active_age, active_gender=self.active_gender, active_emotion=self.active_emotion,
            active_race=self.active_race, active_recognition=self.active_recognition,
            active_glasses=self.active_glasses, active_mask=self.active_mask,
            active_hair_color=self.active_hair_color, active_eye_color=self.active_eye_color,
            active_face_landmarks=self.active_face_landmarks, active_hands=self.active_hands,
            active_gaze=self.active_gaze,
            face_detector=self.face_detector,
            active_liveness=self.active_liveness,
            **extra,
        )


def _reset_adjustments(prefixes: tuple[str, ...]) -> None:
    """Reset image adjustment sliders to their default values."""
    for state_key in list(st.session_state):
        if not any(state_key.startswith(f"{prefix}_") for prefix in prefixes):
            continue
        for adj_key, (_, _, default) in inference.IMAGE_ADJUSTMENT_RANGES.items():
            if state_key.endswith(f"_{adj_key}"):
                st.session_state[state_key] = default
                break


def _adjustment_sliders(caption: str, key_prefix: str, column_count: int = 2) -> dict:
    """Render brightness/contrast/saturation sliders and return their current values."""
    st.caption(caption)
    if st.button("Reset these sliders", key=f"{key_prefix}_reset"):
        _reset_adjustments((key_prefix,))
    values = {}
    columns = st.columns(column_count)
    for index, (adj_key, (adj_min, adj_max, adj_default)) in enumerate(inference.IMAGE_ADJUSTMENT_RANGES.items()):
        with columns[index % column_count]:
            values[adj_key] = st.slider(
                adj_key.replace("_", " ").title(), adj_min, adj_max, adj_default, key=f"{key_prefix}_{adj_key}"
            )
    return values
