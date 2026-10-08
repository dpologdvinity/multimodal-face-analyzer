"""Eigenfaces PCA training and projection for face matching."""
from __future__ import annotations

import cv2
import numpy as np

try:
    from src.core.constants import (
        EIGEN_DIR,
        EIGEN_FACE_SIZE,
        EIGENFACE_DISTANCE_THRESHOLD,
        IMAGE_FILE_EXTENSIONS,
    )
except ImportError:
    from core.constants import (
        EIGEN_DIR,
        EIGEN_FACE_SIZE,
        EIGENFACE_DISTANCE_THRESHOLD,
        IMAGE_FILE_EXTENSIONS,
    )


def _crop_and_resize_for_eigenfaces(face_bgr: np.ndarray) -> np.ndarray:
    """Preprocess for eigenfaces: grayscale, center-square crop, resize to standard size."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    side = min(h, w)
    cy, cx = h // 2, w // 2
    zoomed = gray[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
    return cv2.resize(zoomed, EIGEN_FACE_SIZE)


def _load_eigen_images() -> tuple[list[int], np.ndarray]:
    """Load all saved eigenfaces from eigen/ into memory as flattened vectors."""
    ids: list[int] = []
    vectors = []
    if EIGEN_DIR.is_dir():
        for path in sorted(EIGEN_DIR.iterdir()):
            if path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
                continue
            try:
                face_id = int(path.stem)
            except ValueError:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[::-1] != EIGEN_FACE_SIZE:
                img = cv2.resize(img, EIGEN_FACE_SIZE)
            ids.append(face_id)
            vectors.append(img.flatten().astype(np.float64))
    dim = EIGEN_FACE_SIZE[0] * EIGEN_FACE_SIZE[1]
    return ids, (np.array(vectors) if vectors else np.empty((0, dim)))


def _train_eigenfaces(k: int = 15) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray] | None:
    """Train PCA-based face subspace from saved eigenfaces (faces ever clicked SAVE on)."""
    ids, data = _load_eigen_images()
    if len(ids) < 2:
        return None

    mean_face = data.mean(axis=0)
    A = data - mean_face

    cov_small = A @ A.T
    eigvals, eigvecs_small = np.linalg.eigh(cov_small)
    order = np.argsort(eigvals)[::-1][:k]
    eigvecs_small = eigvecs_small[:, order]

    eigenfaces = A.T @ eigvecs_small
    norms = np.linalg.norm(eigenfaces, axis=0)
    norms[norms == 0] = 1.0
    eigenfaces = eigenfaces / norms

    weights = A @ eigenfaces
    return ids, mean_face, eigenfaces, weights


def _match_against_trained(
    face_bgr: np.ndarray,
    ids: list[int],
    mean_face: np.ndarray,
    eigenfaces: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, float] | None:
    """Project a query face into eigenspace and find the closest training sample (L2 distance)."""
    query = _crop_and_resize_for_eigenfaces(face_bgr).flatten().astype(np.float64) - mean_face
    query_weights = query @ eigenfaces
    distances = np.linalg.norm(weights - query_weights, axis=1)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    return (ids[best_idx], best_dist) if best_dist <= EIGENFACE_DISTANCE_THRESHOLD else None


def match_face_eigenfaces(face_bgr: np.ndarray, k: int = 15) -> tuple[int, float] | None:
    """Match one face against eigen/ via fresh PCA training."""
    trained = _train_eigenfaces(k)
    if trained is None:
        return None
    ids, mean_face, eigenfaces, weights = trained
    return _match_against_trained(face_bgr, ids, mean_face, eigenfaces, weights)


def match_faces_eigenfaces_batch(faces_bgr: list[np.ndarray], k: int = 15) -> list[tuple[int, float] | None]:
    """Match several faces from the same image against eigen/ in one PCA training pass."""
    trained = _train_eigenfaces(k)
    if trained is None:
        return [None] * len(faces_bgr)
    ids, mean_face, eigenfaces, weights = trained
    return [_match_against_trained(face_bgr, ids, mean_face, eigenfaces, weights) for face_bgr in faces_bgr]
