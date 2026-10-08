"""Unit tests for modular face detector subpackage (src/face_analyzer/detectors/)."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from face_analyzer.core.types import Models
from face_analyzer.detectors import (
    detect_faces,
    detect_faces_retinaface,
    detect_faces_scrfd,
    detect_faces_ssd,
    detect_faces_yolo,
)


class TestDetectorsModular(unittest.TestCase):
    """Test suite for modular face detection backends and factory dispatch."""

    def setUp(self):
        """Set up dummy frame and mock models."""
        self.frame = np.zeros((300, 300, 3), dtype=np.uint8)
        self.empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self.mock_ssd = MagicMock()
        self.mock_yolo = MagicMock()
        self.mock_scrfd = MagicMock()
        self.mock_retinaface = MagicMock()

    def test_empty_frame_handling_across_all_detectors(self):
        """Verify all detector functions return empty lists on empty or None frames."""
        for empty_input in (None, self.empty_frame):
            self.assertEqual(detect_faces_ssd(self.mock_ssd, empty_input), [])
            self.assertEqual(detect_faces_yolo(self.mock_yolo, empty_input), [])
            self.assertEqual(detect_faces_scrfd(self.mock_scrfd, empty_input), [])
            self.assertEqual(detect_faces_retinaface(self.mock_retinaface, empty_input), [])

            models = Models(
                face_net=self.mock_ssd,
                yolo_face_nets={"yolo": self.mock_yolo},
                scrfd_face_nets={"scrfd": self.mock_scrfd},
                retinaface_nets={"retinaface": self.mock_retinaface},
            )
            self.assertEqual(detect_faces(models, empty_input, face_detector="yolo"), [])
            self.assertEqual(detect_faces(models, empty_input, face_detector="scrfd"), [])
            self.assertEqual(detect_faces(models, empty_input, face_detector="retinaface"), [])
            self.assertEqual(detect_faces(models, empty_input, face_detector="ssd"), [])

    def test_factory_dispatch_yolo(self):
        """Verify detect_faces dispatches to YOLO when yolo backend is requested and available."""
        models = Models(face_net=self.mock_ssd, yolo_face_nets={"yolo": self.mock_yolo})
        expected_boxes = [[10, 20, 100, 120]]

        with patch("face_analyzer.detectors.factory.detect_faces_yolo", return_value=expected_boxes) as mock_dispatch:
            boxes = detect_faces(models, self.frame, conf_threshold=0.6, face_detector="yolo")
            self.assertEqual(boxes, expected_boxes)
            mock_dispatch.assert_called_once_with(self.mock_yolo, self.frame, conf_threshold=0.6)

    def test_factory_dispatch_scrfd(self):
        """Verify detect_faces dispatches to SCRFD when scrfd backend is requested and available."""
        models = Models(face_net=self.mock_ssd, scrfd_face_nets={"scrfd": self.mock_scrfd})
        expected_boxes = [[15, 25, 105, 125]]

        with patch("face_analyzer.detectors.factory.detect_faces_scrfd", return_value=expected_boxes) as mock_dispatch:
            boxes = detect_faces(models, self.frame, conf_threshold=0.55, face_detector="scrfd")
            self.assertEqual(boxes, expected_boxes)
            mock_dispatch.assert_called_once_with(self.mock_scrfd, self.frame, conf_threshold=0.55)

    def test_factory_dispatch_retinaface(self):
        """Verify detect_faces dispatches to RetinaFace when requested and available."""
        models = Models(face_net=self.mock_ssd, retinaface_nets={"retinaface": self.mock_retinaface})
        expected_boxes = [[5, 10, 95, 110]]

        with patch("face_analyzer.detectors.factory.detect_faces_retinaface", return_value=expected_boxes) as mock_dispatch:
            boxes = detect_faces(models, self.frame, conf_threshold=0.45, face_detector="retinaface")
            self.assertEqual(boxes, expected_boxes)
            mock_dispatch.assert_called_once_with(self.mock_retinaface, self.frame, conf_threshold=0.45)

    def test_factory_dispatch_ssd(self):
        """Verify detect_faces dispatches directly to SSD when ssd backend is requested."""
        models = Models(face_net=self.mock_ssd)
        expected_boxes = [[30, 40, 130, 140]]

        with patch("face_analyzer.detectors.factory.detect_faces_ssd", return_value=expected_boxes) as mock_dispatch:
            boxes = detect_faces(models, self.frame, conf_threshold=0.7, face_detector="ssd")
            self.assertEqual(boxes, expected_boxes)
            mock_dispatch.assert_called_once_with(self.mock_ssd, self.frame, conf_threshold=0.7)

    def test_factory_fallback_to_ssd_when_models_absent(self):
        """Verify fallback from yolo, scrfd, retinaface to ssd when specific model is not loaded."""
        models = Models(face_net=self.mock_ssd)
        expected_boxes = [[20, 20, 80, 80]]

        for requested_backend in ("yolo", "scrfd", "retinaface"):
            with patch("face_analyzer.detectors.factory.detect_faces_ssd", return_value=expected_boxes) as mock_dispatch:
                boxes = detect_faces(models, self.frame, conf_threshold=0.5, face_detector=requested_backend)
                self.assertEqual(boxes, expected_boxes, f"Failed fallback for {requested_backend}")
                mock_dispatch.assert_called_once_with(self.mock_ssd, self.frame, conf_threshold=0.5)

    def test_factory_fallback_when_both_backend_and_ssd_absent(self):
        """Verify detect_faces returns empty list gracefully when neither backend nor SSD is available."""
        models = Models(face_net=None)
        boxes = detect_faces(models, self.frame, conf_threshold=0.5, face_detector="yolo")
        self.assertEqual(boxes, [])

    def test_factory_bare_net_dispatch(self):
        """Verify detect_faces supports legacy callers passing a bare net."""
        expected_boxes = [[10, 10, 50, 50]]
        with patch("face_analyzer.detectors.factory.detect_faces_ssd", return_value=expected_boxes) as mock_dispatch:
            boxes = detect_faces(self.mock_ssd, self.frame, conf_threshold=0.7)
            self.assertEqual(boxes, expected_boxes)
            mock_dispatch.assert_called_once_with(self.mock_ssd, self.frame, conf_threshold=0.7)

    def test_inference_reexports_detectors(self):
        """Verify face_analyzer.inference re-exports all detector public interfaces."""
        from face_analyzer import inference
        self.assertTrue(callable(getattr(inference, "detect_faces_ssd", None)))
        self.assertTrue(callable(getattr(inference, "detect_faces_yolo", None)))
        self.assertTrue(callable(getattr(inference, "detect_faces_scrfd", None)))
        self.assertTrue(callable(getattr(inference, "detect_faces_retinaface", None)))
        self.assertTrue(callable(getattr(inference, "detect_faces", None)))


if __name__ == "__main__":
    unittest.main()
