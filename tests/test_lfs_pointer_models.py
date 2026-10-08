"""load_models() must treat unpulled git-lfs pointer files as missing models, not crash."""
from pathlib import Path

from face_analyzer.core.constants import FACE_MODEL, FACE_PROTO, MODEL_DIR
from face_analyzer.pipeline import loader
from tests._models import require_model

_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
    b"size 1234\n"
)


# The weight suffixes git-lfs tracked before the weights moved out of git; an old no-LFS
# checkout has pointer files at these paths, while prototxt/pbtxt/json/npy/xml/mat stayed real.
_LFS_SUFFIXES = {".h5", ".pth", ".pt", ".caffemodel", ".safetensors", ".onnx", ".task"}


def _point_loader_at_pointer_files(monkeypatch, tmp_path: Path) -> None:
    """Redirect every LFS-tracked model path in the loader to a pointer file under tmp_path."""
    # The loader binds its path constants at import time, so they are patched on the
    # loader module itself rather than via FACE_ANALYZER_MODEL_DIR.
    for name, value in vars(loader).copy().items():
        if not isinstance(value, Path) or value.suffix not in _LFS_SUFFIXES:
            continue
        pointer = tmp_path / value.relative_to(MODEL_DIR)
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_bytes(_POINTER)
        monkeypatch.setattr(loader, name, pointer)


def test_lfs_pointer_weights_are_absent_not_fatal(monkeypatch, tmp_path):
    require_model(FACE_PROTO)
    require_model(FACE_MODEL)
    _point_loader_at_pointer_files(monkeypatch, tmp_path)

    models = loader.load_models()

    assert models.face_net is not None
    assert "fairface" not in models.age_nets
    assert "caffe" not in models.age_nets
    assert "dex" not in models.age_nets
    assert "caffe" not in models.gender_nets
    assert "fairface" not in models.race_nets
    assert not models.emotion_nets
    assert not models.glasses_nets
    assert not models.colorization_nets
    assert not models.yolo_face_nets
    assert "EMOTION" in models.offline_features
