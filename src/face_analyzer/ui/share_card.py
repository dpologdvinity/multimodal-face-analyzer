"""Shareable per-face story card, drawn server-side as a 1080x1350 PNG in the app's typeface."""
from __future__ import annotations

import functools
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARD_SIZE = (1080, 1350)
_FONT = Path(__file__).with_name("fonts") / "BricolageGrotesque.ttf"
_MARGIN = 72
_CREDIT = "multimodel-face-analyzer.streamlit.app"


@functools.cache
def _font(size: int, weight: int = 800, width: int = 100) -> ImageFont.FreeTypeFont:
    """Return Bricolage Grotesque at a size, weight and width (axes: opsz, wght, wdth)."""
    font = ImageFont.truetype(str(_FONT), size)
    font.set_variation_by_axes([min(max(size, 12), 96), weight, width])
    return font


def _rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #rrggbb to an RGB tuple."""
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _fit(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, weight: int = 800, width: int = 75) -> ImageFont.FreeTypeFont:
    """Return the largest font (down from start) that keeps text within max_width."""
    size = start
    while size > 24 and draw.textlength(text, font=_font(size, weight, width)) > max_width:
        size -= 6
    return _font(size, weight, width)


def render_share_card(
    face_rgb: np.ndarray, label: str, rows: list[tuple[str, str, str]], fill: str, ink: str,
) -> bytes:
    """Draw a face card: crop, headline value, the other answers with their models, and a credit line.

    rows are (feature, value, model) with the headline first, as face_story_html shows them.
    """
    fill_rgb, ink_rgb = _rgb(fill), _rgb(ink)
    card = Image.new("RGB", CARD_SIZE, fill_rgb)
    draw = ImageDraw.Draw(card)
    inner = CARD_SIZE[0] - 2 * _MARGIN

    pill = _font(40, 800)
    pill_w = draw.textlength(label, font=pill) + 48
    draw.rounded_rectangle((_MARGIN, _MARGIN, _MARGIN + pill_w, _MARGIN + 68), radius=34, fill=ink_rgb)
    draw.text((_MARGIN + 24, _MARGIN + 34), label, font=pill, fill=fill_rgb, anchor="lm")

    diameter = 300
    crop = cv2.resize(face_rgb, (diameter, diameter), interpolation=cv2.INTER_AREA)
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
    left = CARD_SIZE[0] - _MARGIN - diameter
    draw.ellipse((left - 10, _MARGIN - 10, left + diameter + 10, _MARGIN + diameter + 10), fill=ink_rgb)
    card.paste(Image.fromarray(crop), (left, _MARGIN), mask)

    y = _MARGIN + diameter + 50
    if rows:
        feature, value, model = rows[0]
        big = _fit(draw, value, inner, 300)
        draw.text((_MARGIN - 6, y), value, font=big, fill=ink_rgb, anchor="la")
        y += big.size + 10
        credit = f"{feature.lower()} · {model}" if model else feature.lower()
        draw.text((_MARGIN, y), credit, font=_font(36, 500), fill=ink_rgb, anchor="la")
        y += 90
    for feature, value, model in rows[1:5]:
        draw.line((_MARGIN, y, CARD_SIZE[0] - _MARGIN, y), fill=ink_rgb, width=4)
        draw.text((_MARGIN, y + 46), value, font=_fit(draw, value, inner - 320, 44, 700, 100), fill=ink_rgb, anchor="lm")
        note = f"{feature.lower()} · {model}" if model else feature.lower()
        draw.text((CARD_SIZE[0] - _MARGIN, y + 46), note, font=_font(30, 500), fill=ink_rgb, anchor="rm")
        y += 92
    draw.text((_MARGIN, CARD_SIZE[1] - _MARGIN), _CREDIT, font=_font(30, 600), fill=ink_rgb, anchor="ls")

    buffer = io.BytesIO()
    card.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
