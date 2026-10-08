"""Per-face prediction memoization, the shared inference thread pool and latency metrics."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, (os.cpu_count() or 4)), thread_name_prefix="inference"
)

PREDICTION_CACHE_MAX_SIZE = 2048
_PREDICTION_CACHE: OrderedDict[tuple, object] = OrderedDict()


def _cached_face_predict(feature: str, model_key: str, face_bgr: np.ndarray, predict_fn: Any, *args: Any) -> Any:
    """Memoize a predict_*(net, face, ...) call on (feature, model_key, hash(image bytes))."""
    cache_key = (
        feature,
        model_key,
        face_bgr.shape,
        hashlib.blake2b(face_bgr.tobytes(), digest_size=16).digest(),
    )
    cached = _PREDICTION_CACHE.get(cache_key)
    if cached is not None or cache_key in _PREDICTION_CACHE:
        _PREDICTION_CACHE.move_to_end(cache_key)
        return cached
    value = predict_fn(*args)
    _PREDICTION_CACHE[cache_key] = value
    if len(_PREDICTION_CACHE) > PREDICTION_CACHE_MAX_SIZE:
        _PREDICTION_CACHE.popitem(last=False)
    return value


def _record_model_latency(metrics: dict | None, feature: str, model: str, started: float) -> None:
    """Append per-model inference latency (ms) to metrics dict for performance monitoring."""
    if metrics is None:
        return
    metrics.setdefault("model_latency_ms", {}).setdefault(f"{feature}/{model}", []).append(
        (time.perf_counter() - started) * 1000
    )


__all__ = [
    "_INFERENCE_EXECUTOR",
    "_PREDICTION_CACHE",
    "PREDICTION_CACHE_MAX_SIZE",
    "_cached_face_predict",
    "_record_model_latency",
]
