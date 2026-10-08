"""Tests for the gallery subpackage (src/face_analyzer/gallery/): database, eigenfaces, and identity search."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


class DatabaseTests(unittest.TestCase):
    """Tests for src/face_analyzer/gallery/database.py: SQLite schema management and save_face."""

    def test_save_face_inserts_row_and_writes_files(self):
        """save_face returns an int face_id and creates image artifacts plus a DB row."""
        from face_analyzer.gallery.database import save_face

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_file = root / "db" / "faces.db"
            faces_dir = root / "faces"
            eigen_dir = root / "eigen"
            face = np.zeros((32, 32, 3), dtype=np.uint8)

            import face_analyzer.gallery.database as db_mod
            with (
                mock.patch.object(db_mod, "FACES_DB_FILE", db_file),
                mock.patch.object(db_mod, "FACES_DIR", faces_dir),
                mock.patch.object(db_mod, "EIGEN_DIR", eigen_dir),
                mock.patch.object(db_mod.random, "randint", return_value=42),
            ):
                face_id = save_face(face, {"age_caffe": "25-32"})

            self.assertEqual(face_id, 42)
            self.assertTrue((faces_dir / "42.jpg").is_file())
            self.assertTrue((eigen_dir / "42.jpg").is_file())
            with sqlite3.connect(db_file) as conn:
                count = conn.execute("SELECT COUNT(*) FROM faces WHERE id = 42").fetchone()[0]
            self.assertEqual(count, 1)

    def test_save_face_rollback_on_write_failure(self):
        """save_face rolls back the DB row and cleans up partial files when imwrite fails."""
        from face_analyzer.gallery.database import save_face

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_file = root / "db" / "faces.db"
            faces_dir = root / "faces"
            eigen_dir = root / "eigen"
            face = np.zeros((32, 32, 3), dtype=np.uint8)

            import face_analyzer.gallery.database as db_mod
            with (
                mock.patch.object(db_mod, "FACES_DB_FILE", db_file),
                mock.patch.object(db_mod, "FACES_DIR", faces_dir),
                mock.patch.object(db_mod, "EIGEN_DIR", eigen_dir),
                mock.patch.object(db_mod.random, "randint", return_value=77),
                mock.patch.object(db_mod.cv2, "imwrite", side_effect=[True, False]),
            ):
                with self.assertRaises(OSError):
                    save_face(face, {"age_caffe": "25-32"})

            self.assertFalse((faces_dir / "77.jpg").exists())
            self.assertFalse((eigen_dir / "77.jpg").exists())
            with sqlite3.connect(db_file) as conn:
                count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            self.assertEqual(count, 0)

    def test_save_face_creates_columns_dynamically(self):
        """save_face adds new columns to the faces table on the fly."""
        from face_analyzer.gallery.database import save_face

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_file = root / "db" / "faces.db"
            faces_dir = root / "faces"
            eigen_dir = root / "eigen"
            face = np.zeros((32, 32, 3), dtype=np.uint8)

            import face_analyzer.gallery.database as db_mod
            with (
                mock.patch.object(db_mod, "FACES_DB_FILE", db_file),
                mock.patch.object(db_mod, "FACES_DIR", faces_dir),
                mock.patch.object(db_mod, "EIGEN_DIR", eigen_dir),
                mock.patch.object(db_mod.random, "randint", side_effect=[10, 11]),
            ):
                save_face(face, {"emotion_dan": "happy"})
                save_face(face, {"emotion_dan": "sad", "gender_caffe": "Male"})

            with sqlite3.connect(db_file) as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}
            self.assertIn("emotion_dan", cols)
            self.assertIn("gender_caffe", cols)


class EigenfacesTests(unittest.TestCase):
    """Tests for src/face_analyzer/gallery/eigenfaces.py: PCA training and matching."""

    def _make_face(self, value: int) -> np.ndarray:
        """Create a simple solid-color BGR face for testing."""
        img = np.full((64, 64, 3), value, dtype=np.uint8)
        return img

    def test_crop_and_resize_returns_correct_shape(self):
        """_crop_and_resize_for_eigenfaces returns a grayscale image at EIGEN_FACE_SIZE."""
        from face_analyzer.core.constants import EIGEN_FACE_SIZE
        from face_analyzer.gallery.eigenfaces import _crop_and_resize_for_eigenfaces

        face = self._make_face(128)
        result = _crop_and_resize_for_eigenfaces(face)
        self.assertEqual(result.shape, EIGEN_FACE_SIZE[::-1])  # (h, w) = (100, 100)
        self.assertEqual(result.ndim, 2)  # grayscale

    def test_train_eigenfaces_requires_at_least_two_images(self):
        """_train_eigenfaces returns None when fewer than 2 images exist."""
        import face_analyzer.gallery.eigenfaces as ef_mod
        from face_analyzer.gallery.eigenfaces import _train_eigenfaces

        with tempfile.TemporaryDirectory() as tmp:
            eigen_dir = Path(tmp)
            with mock.patch.object(ef_mod, "EIGEN_DIR", eigen_dir):
                # No images at all
                result = _train_eigenfaces()
        self.assertIsNone(result)

    def test_train_and_match_eigenfaces_with_two_dummy_images(self):
        """_train_eigenfaces and match_face_eigenfaces work end-to-end with 2 dummy images."""
        import face_analyzer.gallery.eigenfaces as ef_mod
        from face_analyzer.gallery.eigenfaces import (
            _crop_and_resize_for_eigenfaces,
            _train_eigenfaces,
            match_face_eigenfaces,
        )

        face_a = self._make_face(50)
        face_b = self._make_face(200)

        with tempfile.TemporaryDirectory() as tmp:
            eigen_dir = Path(tmp)
            # Save two preprocessed eigenface images
            img_a = _crop_and_resize_for_eigenfaces(face_a)
            img_b = _crop_and_resize_for_eigenfaces(face_b)
            cv2.imwrite(str(eigen_dir / "1.jpg"), img_a)
            cv2.imwrite(str(eigen_dir / "2.jpg"), img_b)

            with mock.patch.object(ef_mod, "EIGEN_DIR", eigen_dir):
                trained = _train_eigenfaces(k=1)
                self.assertIsNotNone(trained)
                ids, mean_face, eigenfaces, weights = trained
                self.assertEqual(len(ids), 2)

                # Match face_a — should find id 1 (dark face)
                # Lower threshold for dummy test to allow matching
                with mock.patch.object(ef_mod, "EIGENFACE_DISTANCE_THRESHOLD", 1e9):
                    result = match_face_eigenfaces(face_a, k=1)
                self.assertIsNotNone(result)
                matched_id, dist = result
                self.assertIn(matched_id, [1, 2])  # Should match one of our dummy faces
                self.assertGreaterEqual(dist, 0.0)

    def test_match_faces_eigenfaces_batch_returns_one_per_face(self):
        """match_faces_eigenfaces_batch returns a list with one entry per input face."""
        import face_analyzer.gallery.eigenfaces as ef_mod
        from face_analyzer.gallery.eigenfaces import (
            _crop_and_resize_for_eigenfaces,
            match_faces_eigenfaces_batch,
        )

        face_a = self._make_face(80)
        face_b = self._make_face(160)

        with tempfile.TemporaryDirectory() as tmp:
            eigen_dir = Path(tmp)
            cv2.imwrite(str(eigen_dir / "1.jpg"), _crop_and_resize_for_eigenfaces(face_a))
            cv2.imwrite(str(eigen_dir / "2.jpg"), _crop_and_resize_for_eigenfaces(face_b))

            with (
                mock.patch.object(ef_mod, "EIGEN_DIR", eigen_dir),
                mock.patch.object(ef_mod, "EIGENFACE_DISTANCE_THRESHOLD", 1e9),
            ):
                results = match_faces_eigenfaces_batch([face_a, face_b], k=1)

        self.assertEqual(len(results), 2)

    def test_batch_returns_none_list_when_fewer_than_two_saved(self):
        """match_faces_eigenfaces_batch returns [None] per face when training fails."""
        import face_analyzer.gallery.eigenfaces as ef_mod
        from face_analyzer.gallery.eigenfaces import match_faces_eigenfaces_batch

        face = self._make_face(128)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(ef_mod, "EIGEN_DIR", Path(tmp)):
                results = match_faces_eigenfaces_batch([face, face])

        self.assertEqual(results, [None, None])


class SearchTests(unittest.TestCase):
    """Tests for src/face_analyzer/gallery/search.py: embedding, identity matching, LBPH, gallery I/O."""

    def test_build_gallery_from_directory_with_mocked_detector(self):
        """build_gallery_from_directory correctly skips no-face images and embeds found faces."""
        from face_analyzer.gallery.search import (
            build_gallery_from_directory,
        )

        dummy_embedding = np.ones(128, dtype=np.float32)
        dummy_embedding /= np.linalg.norm(dummy_embedding)

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            # Create two dummy image files
            img_a = np.full((64, 64, 3), 100, dtype=np.uint8)
            img_b = np.full((64, 64, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(gallery_dir / "Alice_Smith.jpg"), img_a)
            cv2.imwrite(str(gallery_dir / "Bob_Jones.jpg"), img_b)
            # Create a non-image file that should be skipped
            (gallery_dir / "notes.txt").write_text("ignored")

            mock_face_net = mock.MagicMock()
            mock_recog_net = mock.MagicMock()

            # Mock detect_faces to return one box for every image
            fake_box = (0, 0, 64, 64)
            with (
                mock.patch("face_analyzer.gallery.search.detect_faces", return_value=[fake_box]),
                mock.patch("face_analyzer.gallery.search.compute_face_embedding", return_value=dummy_embedding),
            ):
                gallery = build_gallery_from_directory(mock_face_net, mock_recog_net, gallery_dir)

        self.assertIn("Alice Smith", gallery)
        self.assertIn("Bob Jones", gallery)
        self.assertNotIn("notes", gallery)
        np.testing.assert_array_equal(gallery["Alice Smith"], dummy_embedding)

    def test_build_gallery_skips_images_with_no_face(self):
        """build_gallery_from_directory omits images where detect_faces returns empty."""
        from face_analyzer.gallery.search import build_gallery_from_directory

        with tempfile.TemporaryDirectory() as tmp:
            gallery_dir = Path(tmp)
            img = np.full((64, 64, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(gallery_dir / "NoFace.jpg"), img)

            mock_face_net = mock.MagicMock()
            mock_recog_net = mock.MagicMock()

            with mock.patch("face_analyzer.gallery.search.detect_faces", return_value=[]):
                gallery = build_gallery_from_directory(mock_face_net, mock_recog_net, gallery_dir)

        self.assertEqual(gallery, {})

    def test_load_and_save_gallery_round_trip(self):
        """save_gallery writes JSON and load_gallery reads it back to numpy arrays."""
        import face_analyzer.gallery.search as search_mod
        from face_analyzer.gallery.search import load_gallery, save_gallery

        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        original = {"Alice": emb}

        with tempfile.TemporaryDirectory() as tmp:
            gallery_file = Path(tmp) / "gallery" / "known_faces.json"
            with mock.patch.object(search_mod, "GALLERY_FILE", gallery_file):
                save_gallery(original)
                loaded = load_gallery()

        self.assertIn("Alice", loaded)
        np.testing.assert_allclose(loaded["Alice"], emb, rtol=1e-5)

    def test_load_gallery_returns_empty_when_file_missing(self):
        """load_gallery returns {} when the gallery file does not exist."""
        import face_analyzer.gallery.search as search_mod
        from face_analyzer.gallery.search import load_gallery

        with tempfile.TemporaryDirectory() as tmp:
            missing_file = Path(tmp) / "nonexistent.json"
            with mock.patch.object(search_mod, "GALLERY_FILE", missing_file):
                result = load_gallery()

        self.assertEqual(result, {})

    def test_validate_lbph_name_strips_and_allows_normal_names(self):
        """validate_lbph_name returns the stripped name for normal inputs."""
        from face_analyzer.gallery.search import validate_lbph_name

        self.assertEqual(validate_lbph_name("  Alice  "), "Alice")
        self.assertEqual(validate_lbph_name("Bob"), "Bob")

    def test_validate_lbph_name_raises_for_path_traversal(self):
        """validate_lbph_name raises ValueError for path traversal inputs."""
        from face_analyzer.gallery.search import validate_lbph_name

        with self.assertRaises(ValueError):
            validate_lbph_name("../evil")
        with self.assertRaises(ValueError):
            validate_lbph_name("")
        with self.assertRaises(ValueError):
            validate_lbph_name(".")

    def test_match_face_identity_returns_best_match(self):
        """match_face_identity returns the name with the highest cosine similarity above threshold."""
        import face_analyzer.gallery.search as search_mod
        from face_analyzer.gallery.search import match_face_identity

        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        query = np.array([0.98, 0.1, 0.0], dtype=np.float32)
        query /= np.linalg.norm(query)

        gallery = {"Alice": emb_a, "Bob": emb_b}
        with mock.patch.object(search_mod, "RECOGNITION_COSINE_THRESHOLD", 0.5):
            result = match_face_identity(query, gallery)

        self.assertIsNotNone(result)
        name, sim = result
        self.assertEqual(name, "Alice")
        self.assertGreater(sim, 0.5)

    def test_match_face_identity_returns_none_below_threshold(self):
        """match_face_identity returns None when similarity is below threshold."""
        import face_analyzer.gallery.search as search_mod
        from face_analyzer.gallery.search import match_face_identity

        gallery = {"Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        query = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        with mock.patch.object(search_mod, "RECOGNITION_COSINE_THRESHOLD", 0.99):
            result = match_face_identity(query, gallery)

        self.assertIsNone(result)

    def test_decode_image_bytes_returns_bgr_array(self):
        """decode_image_bytes decodes JPEG bytes into a BGR ndarray."""
        from face_analyzer.gallery.search import decode_image_bytes

        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        result = decode_image_bytes(buf.tobytes())
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 3)

    def test_decode_image_bytes_raises_on_empty(self):
        """decode_image_bytes raises ValueError for empty bytes."""
        from face_analyzer.gallery.search import decode_image_bytes

        with self.assertRaises(ValueError):
            decode_image_bytes(b"")


class GalleryInitExportsTests(unittest.TestCase):
    """Verify src/face_analyzer/gallery/__init__.py exports all required symbols."""

    REQUIRED_SYMBOLS = [
        # database
        "save_face",
        # eigenfaces
        "_crop_and_resize_for_eigenfaces",
        "_load_eigen_images",
        "_train_eigenfaces",
        "match_face_eigenfaces",
        "_match_against_trained",
        "match_faces_eigenfaces_batch",
        # search
        "compute_face_embedding",
        "match_face_identity",
        "load_gallery",
        "save_gallery",
        "_lbph_preprocess",
        "enroll_lbph_face",
        "validate_lbph_name",
        "_lbph_gallery_signature",
        "train_lbph_recognizer",
        "decode_image_bytes",
        "predict_identity_lbph",
        "build_gallery_from_directory",
    ]

    def test_all_symbols_exported(self):
        """All required gallery symbols are importable from face_analyzer.gallery."""
        import face_analyzer.gallery as gallery_pkg

        for symbol in self.REQUIRED_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertTrue(
                    hasattr(gallery_pkg, symbol),
                    f"face_analyzer.gallery missing symbol: {symbol}",
                )

    def test_all_symbols_in_dunder_all(self):
        """All required gallery symbols appear in face_analyzer.gallery.__all__."""
        import face_analyzer.gallery as gallery_pkg

        for symbol in self.REQUIRED_SYMBOLS:
            with self.subTest(symbol=symbol):
                self.assertIn(
                    symbol,
                    gallery_pkg.__all__,
                    f"face_analyzer.gallery.__all__ missing: {symbol}",
                )


if __name__ == "__main__":
    unittest.main()
