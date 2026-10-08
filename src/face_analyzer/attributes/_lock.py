"""Thread-safe locking mechanism for shared model instances."""
from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

_NET_LOCKS: dict[int, threading.Lock] = {}
_NET_LOCKS_GUARD = threading.Lock()


def _lock_for(net: Any) -> Any:
    """Return a lock keyed by net identity for thread-safe cv2.dnn forward calls."""
    if net is None or isinstance(net, bool):
        return nullcontext()
    key = id(net)
    lock = _NET_LOCKS.get(key)
    if lock is None:
        with _NET_LOCKS_GUARD:
            lock = _NET_LOCKS.setdefault(key, threading.Lock())
    return lock
