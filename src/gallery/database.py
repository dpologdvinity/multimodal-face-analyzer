"""SQLite persistence and image file management for classified faces."""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from src.core.constants import EIGEN_DIR, FACES_DB_FILE, FACES_DIR
except ImportError:
    from core.constants import EIGEN_DIR, FACES_DB_FILE, FACES_DIR

from .eigenfaces import _crop_and_resize_for_eigenfaces

_DEFAULT_FACES_DIR = FACES_DIR
_DEFAULT_EIGEN_DIR = EIGEN_DIR
_DEFAULT_FACES_DB_FILE = FACES_DB_FILE
_ORIGINAL_RANDINT = random.randint
_ORIGINAL_IMWRITE = cv2.imwrite


def _resolve_persistence_context():
    """Resolve active directory paths, random generator, and cv2 for test patching compatibility."""
    inf = sys.modules.get("src.inference") or sys.modules.get("inference")

    if FACES_DIR != _DEFAULT_FACES_DIR:
        active_faces_dir = FACES_DIR
    elif inf and hasattr(inf, "FACES_DIR") and inf.FACES_DIR != _DEFAULT_FACES_DIR:
        active_faces_dir = inf.FACES_DIR
    else:
        active_faces_dir = FACES_DIR

    if EIGEN_DIR != _DEFAULT_EIGEN_DIR:
        active_eigen_dir = EIGEN_DIR
    elif inf and hasattr(inf, "EIGEN_DIR") and inf.EIGEN_DIR != _DEFAULT_EIGEN_DIR:
        active_eigen_dir = inf.EIGEN_DIR
    else:
        active_eigen_dir = EIGEN_DIR

    if FACES_DB_FILE != _DEFAULT_FACES_DB_FILE:
        active_db_file = FACES_DB_FILE
    elif inf and hasattr(inf, "FACES_DB_FILE") and inf.FACES_DB_FILE != _DEFAULT_FACES_DB_FILE:
        active_db_file = inf.FACES_DB_FILE
    else:
        active_db_file = FACES_DB_FILE

    if random.randint is not _ORIGINAL_RANDINT:
        active_random = random
    elif inf and hasattr(inf, "random") and inf.random.randint is not _ORIGINAL_RANDINT:
        active_random = inf.random
    else:
        active_random = random

    if cv2.imwrite is not _ORIGINAL_IMWRITE:
        active_cv2 = cv2
    elif inf and hasattr(inf, "cv2") and inf.cv2.imwrite is not _ORIGINAL_IMWRITE:
        active_cv2 = inf.cv2
    else:
        active_cv2 = cv2

    return active_faces_dir, active_eigen_dir, active_db_file, active_random, active_cv2


def save_face(face_bgr: np.ndarray, raw_columns: dict[str, str]) -> int:
    """Save one classified face into the database with sparse columns and image artifacts."""
    faces_dir, eigen_dir, db_file, rnd, c_v2 = _resolve_persistence_context()

    faces_dir.mkdir(parents=True, exist_ok=True)
    eigen_dir.mkdir(parents=True, exist_ok=True)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    face_id = None
    conn = sqlite3.connect(str(db_file))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY)")
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}
        for col in raw_columns:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE faces ADD COLUMN {col} TEXT")
                existing_cols.add(col)

        for _ in range(20):
            candidate = rnd.randint(1, 999_999)
            if any(path.exists() for path in (
                faces_dir / f"{candidate}.jpg",
                eigen_dir / f"{candidate}.jpg",
            )):
                continue
            cols = ["id"] + list(raw_columns.keys())
            placeholders = ", ".join("?" for _ in cols)
            try:
                conn.execute(
                    f"INSERT INTO faces ({', '.join(cols)}) VALUES ({placeholders})",
                    [candidate] + list(raw_columns.values()),
                )
                face_id = candidate
                break
            except sqlite3.IntegrityError:
                continue
        if face_id is None:
            raise RuntimeError("Could not generate a unique face id after 20 attempts")

        face_path = faces_dir / f"{face_id}.jpg"
        eigen_path = eigen_dir / f"{face_id}.jpg"
        if not c_v2.imwrite(str(face_path), face_bgr):
            raise OSError(f"Could not write saved face image: {face_path}")
        if not c_v2.imwrite(str(eigen_path), _crop_and_resize_for_eigenfaces(face_bgr)):
            raise OSError(f"Could not write eigenface image: {eigen_path}")
        conn.commit()
    except Exception:
        conn.rollback()
        if face_id is not None:
            for path in (faces_dir / f"{face_id}.jpg", eigen_dir / f"{face_id}.jpg"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise
    finally:
        conn.close()

    return face_id
