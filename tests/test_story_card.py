"""Story cards and the results summary: headline parsing, escaping, and colour rotation."""
import numpy as np
from PIL import Image

from face_analyzer.ui.results import (
    _headline_rows,
    _summary_html,
    face_story_html,
    share_card_png,
)


def _face(idx=1, results=None):
    return {
        "idx": idx,
        "image": np.zeros((40, 30, 3), np.uint8),
        "model_results": results if results is not None else [
            {"Feature": "AGE", "Model": "best", "Output": "64 (fairface)"},
            {"Feature": "AGE", "Model": "dex", "Output": "47"},
            {"Feature": "EMOTION", "Model": "fused", "Output": "happy"},
            {"Feature": "GENDER", "Model": "fairface", "Output": "Female"},
            {"Feature": "AGE", "Model": "fairface", "Output": "uncertain (mean 40, SD 13)"},
        ],
    }


def test_headline_rows_take_the_headline_and_split_out_its_model():
    rows = _headline_rows(_face())
    assert rows == [("Age", "64", "fairface"), ("Emotion", "happy", "fused"), ("Gender", "Female", "fairface")]


def test_bracketed_notes_are_not_mistaken_for_a_model():
    face = _face(results=[{"Feature": "AGE", "Model": "best", "Output": "uncertain (mean 40, SD 13)"}])
    assert _headline_rows(face) == [("Age", "uncertain (mean 40, SD 13)", "best")]


def test_story_card_leads_with_age_and_lists_every_model_on_request():
    html = face_story_html(_face(), 0)
    assert 'class="story-card pop-1"' in html
    assert '<div class="story-big">64</div>' in html
    assert '<div class="story-model">age · fairface</div>' in html
    assert '<span class="story-num">01</span>' in html
    assert "<b>happy</b><span>emotion · fused</span>" in html
    assert "<details>" in html and "dex" in html


def test_story_card_colours_rotate_and_values_are_escaped():
    face = _face(results=[{"Feature": "EMOTION", "Model": "ferplus", "Output": "<b>x</b>"}])
    html = face_story_html(face, 6)
    assert "pop-2" in html
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<details>" not in html


def test_story_card_without_any_model_says_so():
    assert "No reading" in face_story_html(_face(results=[]), 0)


def test_summary_counts_faces_and_names_the_common_mood():
    html = _summary_html("a<b>.jpg", [_face(1), _face(2)], 3, "RetinaFace", note=True)
    assert '<span class="wrap-big">2</span>' in html
    assert "faces found. Top mood: happy." in html
    assert "a&lt;b&gt;<wbr>.jpg" in html
    assert "RetinaFace detector" in html and "biometric proof" in html
    assert '"wrap-caption">face found.<' in _summary_html("x.jpg", [_face(results=[])], 0, "SSD", note=False)


def test_age_ranges_use_an_en_dash_and_lead_the_card():
    face = _face(results=[
        {"Feature": "EMOTION", "Model": "ferplus", "Output": "happiness"},
        {"Feature": "AGE", "Model": "fairface", "Output": "40-49"},
    ])
    html = face_story_html(face, 0)
    assert '<div class="story-big">40\u201349</div>' in html


def test_summary_note_only_when_asked():
    assert "biometric proof" not in _summary_html("x.jpg", [_face()], 1, "SSD", note=False)


def test_share_card_is_a_portrait_png_in_the_card_fill():
    import io
    png = share_card_png(_face(), 2, "Night")
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.size == (1080, 1350)
    assert image.convert("RGB").getpixel((5, 5)) == (0x4b, 0x3c, 0xff)
