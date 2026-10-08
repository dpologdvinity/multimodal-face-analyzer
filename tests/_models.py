"""Skip helpers for tests that need git-lfs model weights or untracked local data."""
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


def require_local_data(path: Path) -> None:
    """Skip the calling test when an untracked local data file (e.g. a benchmark photo) is absent."""
    if not path.is_file():
        pytest.skip(f"local data file missing (untracked): {path.name}")
