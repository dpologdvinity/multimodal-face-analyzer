"""Theme tokens, WCAG AA contrast, rendering, and fallback for unknown theme names."""
import pytest
from streamlit.testing.v1 import AppTest

from face_analyzer.ui.theme import (
    DEFAULT_THEME,
    THEME_MARKER_CLASSES,
    THEME_NAMES,
    box_style,
    stylesheet,
    theme_accent,
    theme_label,
    theme_tokens,
)

REQUIRED_TOKENS = {
    "bg", "surface", "surface-raised", "line", "line-strong",
    "text", "muted", "accent", "accent-ink", "ok", "warn", "alert", "box", "box-outline",
}
SURFACES = ("bg", "surface", "surface-raised")
REMOVED_THEMES = ("Brutalist", "Light cyberpunk", "Synthwave", "Phosphor Green", "Midnight Enterprise")


def _luminance(hex_color: str) -> float:
    """Return the WCAG relative luminance of a #rrggbb colour."""
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """Return the WCAG contrast ratio between two #rrggbb colours."""
    light, dark = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_contrast_helper_matches_known_ratios():
    assert contrast("#000000", "#ffffff") == pytest.approx(21)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


def test_exactly_the_four_themes_with_dark_first():
    assert THEME_NAMES == ["Umbra", "Lumen", "Safelight", "Cyanotype"]
    assert DEFAULT_THEME == THEME_NAMES[0]
    assert [theme_label(name) for name in THEME_NAMES] == [
        "Umbra (dark)", "Lumen (light)", "Safelight (dark amber)", "Cyanotype (light blue)",
    ]


@pytest.mark.parametrize("name", THEME_NAMES)
def test_every_theme_defines_every_colour_token(name):
    assert REQUIRED_TOKENS <= theme_tokens()[name].keys()
    assert f"body:has(.{THEME_MARKER_CLASSES[name]})" in stylesheet()


@pytest.mark.parametrize("name", THEME_NAMES)
@pytest.mark.parametrize("foreground", ["text", "muted", "accent", "ok", "warn", "alert"])
def test_text_tokens_meet_aa_on_every_surface(name, foreground):
    # Accent and status colours are also used as text (tabs, card headers, code), so they need
    # the 4.5:1 body-text ratio, not just the 3:1 one for UI components.
    tokens = theme_tokens()[name]
    for surface in SURFACES:
        ratio = contrast(tokens[foreground], tokens[surface])
        assert ratio >= 4.5, f"{name}: {foreground} on {surface} is {ratio:.2f}:1"


@pytest.mark.parametrize("name", THEME_NAMES)
def test_text_on_accent_fills_meets_aa(name):
    tokens = theme_tokens()[name]
    assert contrast(tokens["accent-ink"], tokens["accent"]) >= 4.5


@pytest.mark.parametrize("name", THEME_NAMES)
def test_control_borders_and_accent_meet_ui_contrast(name):
    tokens = theme_tokens()[name]
    for surface in ("bg", "surface"):
        assert contrast(tokens["line-strong"], tokens[surface]) >= 3, f"{name}: line-strong on {surface}"
        assert contrast(tokens["accent"], tokens[surface]) >= 3, f"{name}: accent on {surface}"


@pytest.mark.parametrize("name", THEME_NAMES)
def test_face_box_stands_out_from_its_rim(name):
    # The rim keeps the box visible where the box colour matches the photo underneath.
    tokens = theme_tokens()[name]
    assert contrast(tokens["box"], tokens["box-outline"]) >= 3


def test_box_style_converts_theme_tokens_to_bgr():
    tokens = theme_tokens()["Safelight"]
    assert tokens["box"] == "#ffb547"
    assert box_style("Safelight") == {
        "box_color": (0x47, 0xb5, 0xff), "box_outline": (0x02, 0x14, 0x22), "label_color": (0x47, 0xb5, 0xff),
    }
    assert box_style("Brutalist") == box_style(DEFAULT_THEME)


def test_removed_themes_are_gone_from_the_stylesheet():
    css = stylesheet().lower()
    for marker in ("light-theme", "amber-theme", "synthwave", "phosphor", "brutalist", "corporate-theme", "midnight"):
        assert marker not in css


def _theme_script():
    """Render only the theme picker, stylesheet and marker, as app.py does."""
    from face_analyzer.ui.theme import apply_theme, inject_css, select_theme

    apply_theme(select_theme())
    inject_css()


def _render(theme=None) -> AppTest:
    at = AppTest.from_function(_theme_script)
    if theme is not None:
        at.session_state["theme"] = theme
    return at.run()


def _markers(at: AppTest) -> list[str]:
    return [cls for el in at.markdown for cls in THEME_MARKER_CLASSES.values() if f'class="{cls}"' in el.value]


@pytest.mark.parametrize("name", THEME_NAMES)
def test_each_theme_renders_its_marker_and_stylesheet(name):
    at = _render(name)
    assert not at.exception
    assert at.sidebar.selectbox[0].value == name
    assert _markers(at) == [THEME_MARKER_CLASSES[name]]
    assert any(el.value.startswith("<style>") for el in at.markdown)


def test_new_sessions_start_on_the_dark_default():
    at = _render()
    assert at.sidebar.selectbox[0].value == DEFAULT_THEME
    assert _markers(at) == [THEME_MARKER_CLASSES[DEFAULT_THEME]]


@pytest.mark.parametrize("stale", [*REMOVED_THEMES, "Not a theme", 3])
def test_a_stale_session_theme_falls_back_to_dark(stale):
    at = _render(stale)
    assert not at.exception
    assert at.sidebar.selectbox[0].value == DEFAULT_THEME
    assert _markers(at) == [THEME_MARKER_CLASSES[DEFAULT_THEME]]


@pytest.mark.parametrize(("former", "successor"), [
    ("Optical Bench", "Umbra"), ("Amber Terminal", "Safelight"), ("Corporate Slate", "Cyanotype"),
])
def test_reworked_themes_map_to_their_successors(former, successor):
    assert _render(former).sidebar.selectbox[0].value == successor


def test_unknown_theme_accent_is_the_default_accent():
    assert theme_accent("Brutalist") == theme_tokens()[DEFAULT_THEME]["accent"]
    assert theme_accent("Lumen") == theme_tokens()["Lumen"]["accent"]
