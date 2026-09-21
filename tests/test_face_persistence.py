import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src import inference


class FacePersistenceTests(unittest.TestCase):
    def test_success_writes_both_files_before_returning_id(self):
        """Verify save_face writes image files and DB row, returning consistent ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_file = root / "db" / "faces.db"
            faces_dir = root / "faces"
            eigen_dir = root / "eigen"
            face = np.zeros((32, 32, 3), dtype=np.uint8)

            # Mock module-level paths to isolate test to temp directories
            with mock.patch.object(inference, "FACES_DB_FILE", db_file), \
                 mock.patch.object(inference, "FACES_DIR", faces_dir), \
                 mock.patch.object(inference, "EIGEN_DIR", eigen_dir), \
                 mock.patch.object(inference.random, "randint", return_value=123):
                face_id = inference.save_face(face, {"age_caffe": "25-32"})

            self.assertEqual(face_id, 123)
            self.assertTrue((faces_dir / "123.jpg").is_file())
            self.assertTrue((eigen_dir / "123.jpg").is_file())
            with sqlite3.connect(db_file) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM faces WHERE id = 123").fetchone()[0], 1)

    def test_image_write_failure_does_not_leave_database_row_or_partial_files(self):
        """Verify save_face rolls back DB and cleans up partial files on write failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_file = root / "db" / "faces.db"
            faces_dir = root / "faces"
            eigen_dir = root / "eigen"
            face = np.zeros((32, 32, 3), dtype=np.uint8)

            # Mock module-level paths and imwrite to simulate second file write failure
            with mock.patch.object(inference, "FACES_DB_FILE", db_file), \
                 mock.patch.object(inference, "FACES_DIR", faces_dir), \
                 mock.patch.object(inference, "EIGEN_DIR", eigen_dir), \
                 mock.patch.object(inference.random, "randint", return_value=123), \
                 mock.patch.object(inference.cv2, "imwrite", side_effect=[True, False]):
                with self.assertRaises(OSError):
                    inference.save_face(face, {"age_caffe": "25-32"})

            self.assertFalse((faces_dir / "123.jpg").exists())
            self.assertFalse((eigen_dir / "123.jpg").exists())
            with sqlite3.connect(db_file) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
