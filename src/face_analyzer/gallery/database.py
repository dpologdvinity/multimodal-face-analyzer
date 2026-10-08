"""SQLite persistence and image file management for classified faces."""
from __future__ import annotations

import random
import sqlite3

import cv2
import numpy as np

from ..core.constants import EIGEN_DIR, FACES_DB_FILE, FACES_DIR
from .eigenfaces import _crop_and_resize_for_eigenfaces


def save_face(face_bgr: np.ndarray, raw_columns: dict[str, str]) -> int:
    """Save one classified face into the database with sparse columns and image artifacts."""
    # Read at call time (not bound as defaults) so the storage paths can be redirected
    # by patching this module's attributes.
    faces_dir, eigen_dir, db_file = FACES_DIR, EIGEN_DIR, FACES_DB_FILE

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
            candidate = random.randint(1, 999_999)
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
        if not cv2.imwrite(str(face_path), face_bgr):
            raise OSError(f"Could not write saved face image: {face_path}")
        if not cv2.imwrite(str(eigen_path), _crop_and_resize_for_eigenfaces(face_bgr)):
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
