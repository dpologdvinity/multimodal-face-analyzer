"""HUD overlays, landmark visualizations, and annotated rendering utilities."""
from __future__ import annotations

import contextlib
import os
import sys

import cv2
import numpy as np

from ..core.constants import HAND_CONNECTIONS


@contextlib.contextmanager
def _silence_native_logs():
    """Redirect the process's real stderr fd during noisy native-lib calls."""
    try:
        fd = sys.stderr.fileno()
        saved_fd = os.dup(fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except (OSError, AttributeError):
        yield
        return

    try:
        os.dup2(devnull_fd, fd)
        yield
    finally:
        try:
            os.dup2(saved_fd, fd)
            os.close(devnull_fd)
            os.close(saved_fd)
        except OSError:
            pass

try:
    # Imported only to probe availability; the import is noisy, hence the silencing.
    with _silence_native_logs():
        import mediapipe  # noqa: F401
    MEDIAPIPE_SUPPORTED = True
except ImportError:
    MEDIAPIPE_SUPPORTED = False


def draw_face_landmarks(
    frame: np.ndarray,
    points_normalized: list[tuple[float, float]],
    box: tuple[int, int, int, int],
) -> None:
    """Draw face mesh points directly onto frame, scaled into box's pixel extent."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    for nx, ny in points_normalized:
        cv2.circle(frame, (x1 + int(nx * w), y1 + int(ny * h)), 1, (255, 0, 255), thickness=-1, lineType=cv2.LINE_AA)


def draw_hand_landmarks(frame: np.ndarray, hands: list[list[tuple[int, int]]]) -> None:
    """Draw each hand's skeleton (joints + connecting bones) directly onto frame."""
    for hand in hands:
        for point_a, point_b in HAND_CONNECTIONS:
            cv2.line(frame, hand[point_a], hand[point_b], (0, 255, 0), 2, cv2.LINE_AA)
        for point in hand:
            cv2.circle(frame, point, 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)


def draw_outlined_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw text with a black outline so it stays readable over any background."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    frame_h, frame_w = frame.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    while text_w > frame_w and scale > 0.3:
        scale -= 0.1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = org
    x = max(0, min(x, frame_w - text_w))
    y = max(text_h, min(y, frame_h - baseline))
    org = (x, y)

    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_recognition_scan(
    frame: np.ndarray,
    faces: list[tuple[tuple[int, int, int, int], bool]],
) -> None:
    """Draw the recognition scan results with green/red boxes and labels onto the frame."""
    for (x1, y1, x2, y2), recognized in faces:
        color = (0, 255, 0) if recognized else (0, 0, 255)
        box_thickness = int(round(frame.shape[0] / 150)) or 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, 8)
        draw_outlined_text(frame, "Recognized" if recognized else "Unrecognized", (x1, max(20, y1 - 10)), color)


__all__ = [
    "_silence_native_logs",
    "draw_face_landmarks",
    "draw_hand_landmarks",
    "draw_outlined_text",
    "draw_recognition_scan",
]
