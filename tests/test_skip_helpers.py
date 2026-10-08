"""Verify the skip helpers used by tests that need untracked or git-lfs data."""
import pytest

from tests._models import require_local_data, require_model

_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n"


def _skip_reason(excinfo: pytest.ExceptionInfo) -> str:
    """Return the reason text carried by a captured pytest skip."""
    return str(excinfo.value)


def test_require_local_data_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception) as excinfo:
        require_local_data(tmp_path / "joe-biden.jpg")
    assert "joe-biden.jpg" in _skip_reason(excinfo)


def test_require_local_data_passes_when_present(tmp_path):
    present = tmp_path / "photo.jpg"
    present.write_bytes(b"\xff\xd8")
    require_local_data(present)


def test_require_model_skips_on_lfs_pointer(tmp_path):
    pointer = tmp_path / "age_net.caffemodel"
    pointer.write_bytes(_POINTER)
    with pytest.raises(pytest.skip.Exception):
        require_model(pointer)
