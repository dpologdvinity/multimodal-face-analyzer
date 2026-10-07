"""Helpers for tests that need real model weights from the (git-lfs) models directory."""
from pathlib import Path

import pytest

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def require_model(path: Path) -> None:
    """Skip the calling test when a model file is missing or is an unpulled LFS pointer."""
    if not path.is_file():
        pytest.skip(f"model file missing: {path.name}")
    with path.open("rb") as fh:
        if fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            pytest.skip(f"model file is an LFS pointer: {path.name}")
