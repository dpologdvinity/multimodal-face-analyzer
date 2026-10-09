"""Per-face prediction memoization, the shared inference thread pool and latency metrics."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..core.constants import PREDICTION_CACHE_MAX_SIZE
from ..demo import is_demo_mode

_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, (os.cpu_count() or 4)), thread_name_prefix="inference"
)

_PREDICTION_CACHE: OrderedDict[tuple, object] = OrderedDict()
# Inference threads from several sessions share the cache; an unlocked get/move_to_end can race
# with another thread's eviction and raise KeyError.
_PREDICTION_CACHE_LOCK = threading.Lock()
_MISSING = object()


def _cached_face_predict(feature: str, model_key: str, face_bgr: np.ndarray, predict_fn: Any, *args: Any) -> Any:
    """Memoize a predict_*(net, face, ...) call on (feature, model_key, hash(image bytes)).

    Demo mode skips the cache, so nothing derived from a visitor's image outlives the request.
    """
    if is_demo_mode():
        return predict_fn(*args)
    cache_key = (
        feature,
        model_key,
        face_bgr.shape,
        hashlib.blake2b(face_bgr.tobytes(), digest_size=16).digest(),
    )
    with _PREDICTION_CACHE_LOCK:
        cached = _PREDICTION_CACHE.get(cache_key, _MISSING)
        if cached is not _MISSING:
            _PREDICTION_CACHE.move_to_end(cache_key)
            return cached
    value = predict_fn(*args)
    with _PREDICTION_CACHE_LOCK:
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
