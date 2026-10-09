"""Theme picker, theme tokens, and the app stylesheet."""
from __future__ import annotations

import functools
import re
from pathlib import Path

import streamlit as st

DEFAULT_THEME = "Umbra"
THEME_NAMES = ["Umbra", "Lumen", "Safelight", "Cyanotype"]
THEME_HINTS = {
    "Umbra": "dark",
    "Lumen": "light",
    "Safelight": "dark amber",
    "Cyanotype": "light blue",
}
THEME_MARKER_CLASSES = {name: f"theme-{name.lower()}" for name in THEME_NAMES}
# Sessions that predate the redesign may still hold an old name; the three themes that were
# reworked rather than dropped map to their successors, everything else to the default.
_FORMER_THEME_NAMES = {
    "Optical Bench": "Umbra",
    "Amber Terminal": "Safelight",
    "Corporate Slate": "Cyanotype",
}

_STYLESHEET = Path(__file__).with_name("theme.css")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE = re.compile(r"([^{};]+)\{([^{}]*)\}")
_COLOR_TOKEN = re.compile(r"--([\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;")


def resolve_theme(name: object) -> str:
    """Return a known theme name for name, falling back to the default theme."""
    if name in THEME_NAMES:
        return name
    return _FORMER_THEME_NAMES.get(name, DEFAULT_THEME) if isinstance(name, str) else DEFAULT_THEME


def theme_label(name: str) -> str:
    """Return the picker label for a theme, e.g. "Umbra (dark)"."""
    return f"{name} ({THEME_HINTS[name]})"


def select_theme() -> str:
    """Render the sidebar theme picker and return the selected theme name."""
    if "theme" in st.session_state:
        st.session_state["theme"] = resolve_theme(st.session_state["theme"])
    return st.sidebar.selectbox("Theme", THEME_NAMES, format_func=theme_label, key="theme")


@functools.cache
def stylesheet() -> str:
    """Return the raw stylesheet text."""
    return _STYLESHEET.read_text(encoding="utf-8")


@functools.cache
def theme_tokens() -> dict[str, dict[str, str]]:
    """Return each theme's #rrggbb colour tokens, read from its block in the stylesheet."""
    tokens: dict[str, dict[str, str]] = {}
    for selector, body in _RULE.findall(_COMMENT.sub("", stylesheet())):
        parts = {part.strip() for part in selector.split(",")}
        for name, marker in THEME_MARKER_CLASSES.items():
            if f"body:has(.{marker})" in parts and parts <= {":root", f"body:has(.{marker})"}:
                tokens[name] = dict(_COLOR_TOKEN.findall(body))
    # Themes inherit any token they leave unset from the :root block (the default theme).
    return {name: {**tokens[DEFAULT_THEME], **tokens[name]} for name in THEME_NAMES}


def theme_accent(name: str) -> str:
    """Return a theme's accent colour; unknown names get the default theme's."""
    return theme_tokens()[resolve_theme(name)]["accent"]


@functools.cache
def theme_css() -> str:
    """Return the stylesheet wrapped in the <style> block passed to st.markdown."""
    return f"<style>\n{stylesheet()}</style>"


def inject_css() -> None:
    """Inject the stylesheet; every theme's tokens are in it, keyed off a marker element."""
    st.markdown(theme_css(), unsafe_allow_html=True)


def apply_theme(name: str) -> None:
    """Emit the marker element whose class the stylesheet's body:has(...) rules select on."""
    marker = THEME_MARKER_CLASSES[resolve_theme(name)]
    st.markdown(f'<div class="{marker}"></div>', unsafe_allow_html=True)
