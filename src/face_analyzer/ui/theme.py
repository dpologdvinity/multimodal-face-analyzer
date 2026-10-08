"""Theme picker, per-theme accents, and the app stylesheet."""
from __future__ import annotations

import functools
from pathlib import Path

import streamlit as st

THEME_NAMES = [
    "Optical Bench", "Light cyberpunk", "Amber Terminal", "Synthwave", "Phosphor Green",
    "Brutalist", "Corporate Slate", "Midnight Enterprise",
]
# The default theme ("Optical Bench") has no marker: its palette is the stylesheet's :root.
THEME_MARKER_CLASSES = {
    "Light cyberpunk": "light-theme",
    "Amber Terminal": "amber-theme",
    "Synthwave": "synthwave-theme",
    "Phosphor Green": "phosphor-theme",
    "Brutalist": "brutalist-theme",
    "Corporate Slate": "corporate-theme",
    "Midnight Enterprise": "midnight-theme",
}
THEME_ACCENTS = {
    "Optical Bench": "#4fb8ac",
    "Light cyberpunk": "#087a52",
    "Amber Terminal": "#ffb02e",
    "Synthwave": "#ff2fb8",
    "Phosphor Green": "#6bffa0",
    "Brutalist": "#0a0a0a",
    "Corporate Slate": "#2f5aa8",
    "Midnight Enterprise": "#4f8ff0",
}

_STYLESHEET = Path(__file__).with_name("theme.css")


def select_theme() -> str:
    """Render the sidebar theme picker and return the selected theme name."""
    return st.sidebar.selectbox("Theme", THEME_NAMES, key="theme")


@functools.cache
def theme_css() -> str:
    """Return the stylesheet wrapped in the <style> block passed to st.markdown."""
    # The surrounding whitespace reproduces the former inline triple-quoted literal exactly.
    return "\n    <style>\n" + _STYLESHEET.read_text(encoding="utf-8") + "    </style>\n    "


def inject_css() -> None:
    """Inject the stylesheet; every theme's palette is in it, keyed off a marker element."""
    st.markdown(theme_css(), unsafe_allow_html=True)


def apply_theme(name: str) -> None:
    """Emit the marker element whose class the stylesheet's body:has(...) rules select on."""
    if name in THEME_MARKER_CLASSES:
        st.markdown(f'<div class="{THEME_MARKER_CLASSES[name]}"></div>', unsafe_allow_html=True)
