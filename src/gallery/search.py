"""Identity embedding, gallery matching, LBPH face recognition, and gallery building."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from src.attributes._lock import _lock_for
except ImportError:
    try:
        from attributes._lock import _lock_for
    except ImportError:
        from contextlib import nullcontext

        def _lock_for(net: Any) -> Any:
            """Fallback no-op lock."""
            return nullcontext()

try:
    from src.detectors import detect_faces
except ImportError:
    from detectors import detect_faces

try:
    from src.core.constants import (
        GALLERY_FILE,
        IMAGE_FILE_EXTENSIONS,
        LBPH_CONFIDENCE_THRESHOLD,
        LBPH_FACE_SIZE,
        LBPH_GALLERY_DIR,
        MAX_UPLOAD_DIMENSION,
        RECOGNITION_COSINE_THRESHOLD,
    )
except ImportError:
    from core.constants import (
        GALLERY_FILE,
        IMAGE_FILE_EXTENSIONS,
        LBPH_CONFIDENCE_THRESHOLD,
        LBPH_FACE_SIZE,
        LBPH_GALLERY_DIR,
        MAX_UPLOAD_DIMENSION,
        RECOGNITION_COSINE_THRESHOLD,
    )

_LBPH_CACHE_LOCK = threading.Lock()
_LBPH_CACHE_SIGNATURE: str | None = None
_LBPH_CACHE_RESULT: Any = None


def compute_face_embedding(net: Any, face_bgr: np.ndarray) -> np.ndarray:
    """Compute normalized VGGFace embedding for identity recognition (cosine distance)."""
    face_resized = cv2.resize(face_bgr, (224, 224)).astype(np.float32)
    with _lock_for(net):
        emb = net(face_resized[np.newaxis, ...], training=False).numpy().flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def match_face_identity(embedding: np.ndarray, gallery: dict) -> tuple[str, float] | None:
    """Find best cosine similarity match against enrolled identity gallery."""
    best_name, best_sim = None, -1.0
    for name, gal_emb in gallery.items():
        sim = float(np.dot(embedding, gal_emb))
        if sim > best_sim:
            best_name, best_sim = name, sim
    return (best_name, best_sim) if best_sim >= RECOGNITION_COSINE_THRESHOLD else None


def load_gallery() -> dict[str, np.ndarray]:
    """Load enrolled face embeddings from gallery/known_faces.json."""
    if not GALLERY_FILE.exists():
        return {}
    raw = json.loads(GALLERY_FILE.read_text())
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def save_gallery(gallery: dict) -> None:
    """Persist enrolled face embeddings to gallery/known_faces.json."""
    GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_FILE.write_text(json.dumps({name: vec.tolist() for name, vec in gallery.items()}))


def _lbph_preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Grayscale and resize to standard LBPH face dimensions."""
    return cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), LBPH_FACE_SIZE)


def validate_lbph_name(name: str) -> str:
    """Return a safe gallery directory name or raise for path traversal input."""
    name = name.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Enrollment name must be a non-empty name without path separators.")
    return name


def enroll_lbph_face(name: str, face_bgr: np.ndarray) -> None:
    """Save preprocessed face crop under gallery/lbph/<name>/ for LBPH training."""
    name = validate_lbph_name(name)
    person_dir = LBPH_GALLERY_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(person_dir.glob("*.png")))
    cv2.imwrite(str(person_dir / f"{existing:04d}.png"), _lbph_preprocess(face_bgr))


def _lbph_gallery_signature() -> str | None:
    """Compute a SHA256 hash of the LBPH gallery structure to detect changes."""
    if not LBPH_GALLERY_DIR.is_dir():
        return None
    entries = []
    for path in sorted(LBPH_GALLERY_DIR.glob("*/*.png")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append(f"{path.relative_to(LBPH_GALLERY_DIR)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def train_lbph_recognizer():
    """Train an LBPHFaceRecognizer fresh from gallery/lbph/ directory."""
    global _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT
    if not hasattr(cv2, "face"):
        return None

    signature = _lbph_gallery_signature()
    with _LBPH_CACHE_LOCK:
        if signature == _LBPH_CACHE_SIGNATURE:
            return _LBPH_CACHE_RESULT
        if signature is None:
            _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
            return None

    label_names = sorted(p.name for p in LBPH_GALLERY_DIR.iterdir() if p.is_dir())
    if not label_names:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
        return None

    features, labels = [], []
    for label, name in enumerate(label_names):
        for path in (LBPH_GALLERY_DIR / name).glob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features.append(img)
                labels.append(label)
    if not features:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, None
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(features, np.array(labels))
    with _LBPH_CACHE_LOCK:
        _LBPH_CACHE_SIGNATURE, _LBPH_CACHE_RESULT = signature, (recognizer, label_names)
        return _LBPH_CACHE_RESULT


def decode_image_bytes(file_bytes: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decode uploaded image bytes into BGR ndarray, downscaled if larger than max dimension."""
    encoded = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("The uploaded file is empty or could not be read.")
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("The uploaded file is not a valid supported image.")
    height, width = frame.shape[:2]
    longer_side = max(height, width)
    if longer_side > MAX_UPLOAD_DIMENSION:
        scale = MAX_UPLOAD_DIMENSION / longer_side
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return frame


def predict_identity_lbph(recognizer: Any, label_names: list[str], face_bgr: np.ndarray) -> tuple[str, float] | None:
    """Predict enrolled identity via LBPH recognizer if distance is below threshold."""
    with _lock_for(recognizer):
        label, confidence = recognizer.predict(_lbph_preprocess(face_bgr))
    return (label_names[label], confidence) if confidence < LBPH_CONFIDENCE_THRESHOLD else None


def build_gallery_from_directory(face_net: Any, recognition_net: Any, directory: str | Path) -> dict[str, np.ndarray]:
    """Scan directory for face images, detect faces, and build name-to-embedding gallery dict."""
    directory = Path(directory)
    gallery: dict[str, np.ndarray] = {}
    if not directory.is_dir():
        return gallery

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        boxes = detect_faces(face_net, image, conf_threshold=0.7)
        if not boxes:
            continue
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        h, w = image.shape[:2]
        y1c, y2c = max(0, y1 - 20), min(y2 + 20, h - 1)
        x1c, x2c = max(0, x1 - 20), min(x2 + 20, w - 1)
        face = image[y1c:y2c, x1c:x2c]
        if face.size == 0:
            continue
        gallery[path.stem.replace("_", " ")] = compute_face_embedding(recognition_net, face)

    return gallery
