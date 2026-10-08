"""Offline tests for models/manifest.json and tools/fetch_models.py."""
import hashlib
import importlib.util
import io
import json
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from face_analyzer.core import constants as C

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("fetch_models", ROOT / "tools" / "fetch_models.py")
fetch_models = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_models)

MANIFEST = ROOT / "models" / "manifest.json"
HF_PREFIX = "https://huggingface.co/kaitlynbassford/face-analyzer-weights/resolve/main/"
DOCKER_ARGS = {
    "AGE_MODEL", "GENDER_MODEL", "EMOTION_MODEL", "RACE_MODEL", "FACE_LANDMARKS_MODEL",
    "LIVENESS_MODEL", "RECOGNITION_MODEL", "GLASSES_MODEL", "MASK_MODEL", "COLORIZATION_MODEL",
    "HAND_MODEL", "RECONSTRUCTION_3D_MODEL", "YOLO_FACE_MODEL", "SCRFD_FACE_MODEL",
    "RETINAFACE_MODEL", "AGE_PROGRESSION_MODEL",
}


def _entry(path: str, payload: bytes, source: str = "upstream", url: str = "https://example.invalid/x", **extra):
    """Build a manifest entry describing payload."""
    entry = {
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "features": ["test"],
        "models": {"AGE_MODEL": "test"},
        "license": "MIT",
        "source": source,
    }
    if source == "manual":
        entry["instructions"] = "Put the file at {dest}."
    else:
        entry["url"] = url
    entry.update(extra)
    return entry


@pytest.fixture
def http_root(tmp_path):
    """Serve tmp_path/served over a local HTTP server and yield (directory, base_url)."""
    served = tmp_path / "served"
    served.mkdir()
    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield served, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _real_manifest() -> list[dict]:
    return fetch_models.load_manifest(MANIFEST)


def test_manifest_entries_validate_and_are_unique():
    entries = _real_manifest()
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    for entry in entries:
        assert set(entry["models"]) <= DOCKER_ARGS, entry["path"]
        if entry["source"] == "hf":
            assert entry["url"] == HF_PREFIX + entry["path"]
        if entry["source"] != "manual":
            assert "instructions" not in entry


def test_every_model_path_load_models_uses_has_a_manifest_entry():
    """Derive the expected files from the path constants, so new models cannot be forgotten."""
    model_dir = C.MODEL_DIR.resolve()
    expected = set()
    for value in vars(C).values():
        if isinstance(value, Path) and value.suffix and model_dir in value.resolve().parents:
            expected.add(value.resolve().relative_to(model_dir).as_posix())
    assert expected, "no model paths found in core/constants.py"
    assert expected <= {entry["path"] for entry in _real_manifest()}


def test_files_kept_in_git_are_permissive_text_or_the_required_detector():
    """Only the always-required SSD detector may stay in git without being re-hostable."""
    for entry in _real_manifest():
        if entry["in_git"] and entry["source"] != "hf":
            assert entry["required"], entry["path"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"source": "ftp"}, "source"),
        ({"sha256": "abc"}, "sha256"),
        ({"path": "../escape.onnx"}, "path"),
        ({"url": "http://insecure.invalid/x"}, "url"),
        ({"size": "12"}, "size"),
    ],
)
def test_validate_entry_rejects_bad_fields(mutation, message):
    entry = _entry("a.onnx", b"x")
    entry.update(mutation)
    with pytest.raises(ValueError, match=message):
        fetch_models.validate_entry(entry)


def test_validate_entry_requires_instructions_for_manual():
    entry = _entry("a.pth", b"x", source="manual")
    del entry["instructions"]
    with pytest.raises(ValueError, match="instructions"):
        fetch_models.validate_entry(entry)


def test_select_entries_matches_arg_pairs_bare_keys_and_required():
    entries = [
        _entry("ssd.pb", b"1", models={}, required=True),
        _entry("fairface.onnx", b"2", models={"AGE_MODEL": "fairface", "RACE_MODEL": "fairface"}),
        _entry("caffe_age.caffemodel", b"3", models={"AGE_MODEL": "caffe"}),
        _entry("caffe_gender.caffemodel", b"4", models={"GENDER_MODEL": "caffe"}),
        _entry("ferplus.onnx", b"5", models={"EMOTION_MODEL": "ferplus"}),
    ]
    picked = fetch_models.select_entries(entries, ["AGE_MODEL=caffe", "RACE_MODEL=fairface", "GENDER_MODEL="], False)
    assert [e["path"] for e in picked] == ["ssd.pb", "fairface.onnx", "caffe_age.caffemodel"]
    picked = fetch_models.select_entries(entries, ["ferplus"], False)
    assert [e["path"] for e in picked] == ["ssd.pb", "ferplus.onnx"]
    assert fetch_models.select_entries(entries, [], True) == entries


@pytest.mark.parametrize("source", ["hf", "upstream"])
def test_download_verifies_and_moves_into_place(tmp_path, http_root, source):
    served, base = http_root
    payload = b"weights" * 1000
    (served / "w.onnx").write_bytes(payload)
    model_dir = tmp_path / "models"

    status = fetch_models.fetch_entry(_entry("sub/w.onnx", payload, source=source, url=f"{base}/w.onnx"), model_dir)

    assert status == "downloaded"
    assert (model_dir / "sub" / "w.onnx").read_bytes() == payload
    assert not list((model_dir / "sub").glob("*.part"))


def test_sha256_mismatch_fails_and_leaves_nothing_behind(tmp_path, http_root):
    served, base = http_root
    (served / "w.onnx").write_bytes(b"tampered")
    model_dir = tmp_path / "models"

    with pytest.raises(fetch_models.FetchError, match="sha256 mismatch"):
        fetch_models.fetch_entry(_entry("w.onnx", b"expected", url=f"{base}/w.onnx"), model_dir)
    assert list(model_dir.iterdir()) == []


def test_main_exits_nonzero_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"tampered"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [_entry("w.onnx", b"expected")]}))

    code = fetch_models.main(["--all", "--manifest", str(manifest), "--model-dir", str(tmp_path / "m")])

    assert code == 1


def test_existing_file_with_right_hash_is_skipped(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "w.onnx").write_bytes(b"good")
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no download expected"))

    assert fetch_models.fetch_entry(_entry("w.onnx", b"good"), model_dir) == "ok"


def test_existing_file_with_wrong_hash_needs_force(tmp_path, http_root):
    served, base = http_root
    (served / "w.onnx").write_bytes(b"good")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "w.onnx").write_bytes(b"stale")
    entry = _entry("w.onnx", b"good", url=f"{base}/w.onnx")

    with pytest.raises(fetch_models.FetchError, match="--force"):
        fetch_models.fetch_entry(entry, model_dir)
    assert fetch_models.fetch_entry(entry, model_dir, force=True) == "downloaded"
    assert (model_dir / "w.onnx").read_bytes() == b"good"


def test_partial_download_resumes_with_range_request(tmp_path, monkeypatch):
    payload = b"0123456789" * 50
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "w.onnx.part").write_bytes(payload[:120])
    seen = {}

    class _Response(io.BytesIO):
        status = 206

    def fake_urlopen(request, timeout):
        seen["range"] = request.get_header("Range")
        return _Response(payload[120:])

    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", fake_urlopen)

    assert fetch_models.fetch_entry(_entry("w.onnx", payload), model_dir) == "downloaded"
    assert seen["range"] == "bytes=120-"
    assert (model_dir / "w.onnx").read_bytes() == payload


def test_archive_member_is_extracted_and_verified(tmp_path, http_root):
    served, base = http_root
    payload = b"detector-weights"
    with zipfile.ZipFile(served / "pack.zip", "w") as archive:
        archive.writestr("pack/det.onnx", payload)
        archive.writestr("pack/other.onnx", b"unused")
    model_dir = tmp_path / "models"
    entry = _entry("det.onnx", payload, url=f"{base}/pack.zip", archive_member="pack/det.onnx")

    assert fetch_models.fetch_entry(entry, model_dir) == "downloaded"
    assert sorted(p.name for p in model_dir.iterdir()) == ["det.onnx"]
    assert (model_dir / "det.onnx").read_bytes() == payload


def test_manual_entry_prints_instructions_without_failing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no download expected"))
    model_dir = tmp_path / "models"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [_entry("byo.pth", b"secret", source="manual")]}))

    code = fetch_models.main(["--keys", "AGE_MODEL=test", "--manifest", str(manifest), "--model-dir", str(model_dir)])

    err = capsys.readouterr().err
    assert code == 0
    assert "bring your own: byo.pth" in err
    assert f"Put the file at {model_dir / 'byo.pth'}." in err


def test_manual_entry_with_different_file_only_warns(tmp_path, capsys):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "byo.pth").write_bytes(b"my own conversion")

    assert fetch_models.fetch_entry(_entry("byo.pth", b"secret", source="manual"), model_dir) == "ok"
    assert "differs" in capsys.readouterr().err


def test_model_dir_defaults_to_face_analyzer_model_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_ANALYZER_MODEL_DIR", str(tmp_path))
    assert fetch_models.default_model_dir() == tmp_path
    monkeypatch.delenv("FACE_ANALYZER_MODEL_DIR")
    assert fetch_models.default_model_dir() == ROOT / "models"
