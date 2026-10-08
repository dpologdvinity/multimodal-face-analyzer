"""Base face detector interface and protocols."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseFaceDetector(Protocol):
    """Protocol defining the interface for face detector backends."""

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list[list[int]]:
        """Detect faces in a frame and return [x1, y1, x2, y2] bounding boxes."""
        ...
