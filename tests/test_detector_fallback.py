"""load_models() needs at least one face detector; SSD is optional and is the preferred fallback."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_analyzer.core.constants import MODEL_DIR
from face_analyzer.core.types import Models
from face_analyzer.detectors import (
    available_face_detectors,
    detect_faces,
    resolve_face_detector,
)
from face_analyzer.pipeline import loader


def _point_loader_at_empty_dir(monkeypatch, tmp_path: Path) -> None:
    """Redirect every model path the loader knows to a missing file under tmp_path."""
    # The loader binds its path constants at import time, so they are patched on the module.
    for name, value in vars(loader).copy().items():
        if isinstance(value, Path) and value.is_relative_to(MODEL_DIR):
            monkeypatch.setattr(loader, name, tmp_path / value.relative_to(MODEL_DIR))
    monkeypatch.setattr(loader, "MODEL_DIR", tmp_path)


def _fake_onnxruntime(monkeypatch) -> MagicMock:
    """Stand in for onnxruntime so the test runs without it and without real weights."""
    runtime = MagicMock()
    monkeypatch.setattr(loader, "ONNXRUNTIME_SUPPORTED", True)
    monkeypatch.setattr(loader, "onnxruntime", runtime, raising=False)
    return runtime


def test_only_retinaface_present_loads_without_ssd(monkeypatch, tmp_path):
    _point_loader_at_empty_dir(monkeypatch, tmp_path)
    runtime = _fake_onnxruntime(monkeypatch)
    loader.RETINAFACE_MODEL.write_bytes(b"onnx")

    models = loader.load_models()

    assert models.face_net is None
    assert models.retinaface_nets == {"retinaface": runtime.InferenceSession.return_value}
    assert available_face_detectors(models) == ["retinaface"]
    # Every requested detector, SSD included, resolves to the one that is loaded.
    for requested in ("yolo", "ssd", "scrfd", "retinaface"):
        assert resolve_face_detector(models, requested) == "retinaface"


def test_no_detector_present_raises_the_missing_detector_error(monkeypatch, tmp_path):
    _point_loader_at_empty_dir(monkeypatch, tmp_path)
    _fake_onnxruntime(monkeypatch)

    with pytest.raises(FileNotFoundError, match=r"Missing face detector file\(s\) in .*opencv_face_detector"):
        loader.load_models()


def test_demo_mode_never_loads_ssd_or_non_demo_backends(monkeypatch, tmp_path):
    _point_loader_at_empty_dir(monkeypatch, tmp_path)
    _fake_onnxruntime(monkeypatch)
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    # Not real weights: cv2 would raise if demo mode tried to read any of them.
    for path in (loader.FACE_PROTO, loader.FACE_MODEL, loader.YOLO_FACE_MODEL, loader.SCRFD_FACE_MODEL,
                 loader.RETINAFACE_MODEL, loader.AGE_PROTO, loader.AGE_MODEL):
        path.write_bytes(b"not a model")

    models = loader.load_models()

    assert models.face_net is None
    assert set(models.retinaface_nets) == {"retinaface"}
    assert not models.yolo_face_nets and not models.scrfd_face_nets
    assert not models.age_nets
    assert not models.recognition_nets


def test_demo_mode_without_retinaface_raises(monkeypatch, tmp_path):
    _point_loader_at_empty_dir(monkeypatch, tmp_path)
    _fake_onnxruntime(monkeypatch)
    monkeypatch.setenv("FACE_ANALYZER_DEMO", "1")
    loader.FACE_PROTO.write_bytes(b"not a model")
    loader.FACE_MODEL.write_bytes(b"not a model")

    with pytest.raises(FileNotFoundError, match="Demo mode needs the RetinaFace detector.*onnxruntime"):
        loader.load_models()


def test_fallback_prefers_ssd_over_other_loaded_detectors():
    models = Models(face_net=MagicMock(), retinaface_nets={"retinaface": MagicMock()}, scrfd_face_nets={"scrfd": MagicMock()})
    assert resolve_face_detector(models, "yolo") == "ssd"
    assert resolve_face_detector(models, "scrfd") == "scrfd"
    assert available_face_detectors(models) == ["ssd", "scrfd", "retinaface"]


def test_detect_faces_falls_back_to_retinaface_without_ssd():
    retinaface = MagicMock()
    models = Models(face_net=None, retinaface_nets={"retinaface": retinaface})
    frame = np.zeros((32, 32, 3), np.uint8)
    with patch("face_analyzer.detectors.factory.detect_faces_retinaface", return_value=[[1, 2, 3, 4]]) as dispatch:
        assert detect_faces(models, frame, conf_threshold=0.4, face_detector="ssd") == [[1, 2, 3, 4]]
    dispatch.assert_called_once_with(retinaface, frame, conf_threshold=0.4)


def test_bare_none_detector_returns_no_faces():
    assert detect_faces(None, np.zeros((32, 32, 3), np.uint8)) == []


def test_ort_thread_cap_applies_only_when_set(monkeypatch, tmp_path):
    runtime = _fake_onnxruntime(monkeypatch)
    model = tmp_path / "model.onnx"

    monkeypatch.delenv("FACE_ANALYZER_ORT_THREADS", raising=False)
    loader._ort_session(model)
    assert runtime.InferenceSession.call_args.kwargs["sess_options"] is None

    monkeypatch.setenv("FACE_ANALYZER_ORT_THREADS", "2")
    loader._ort_session(model)
    options = runtime.SessionOptions.return_value
    assert runtime.InferenceSession.call_args.kwargs["sess_options"] is options
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (2, 1)
