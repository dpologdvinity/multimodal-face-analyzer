"""Unit tests for the modular pipeline subpackage (src/face_analyzer/pipeline)."""
import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from face_analyzer.core.types import Models
from face_analyzer.pipeline.analyzer import aggregate_demographics, analyze_frame
from face_analyzer.pipeline.config import AnalysisConfig
from face_analyzer.pipeline.drawing import (
    draw_face_landmarks,
    draw_hand_landmarks,
    draw_outlined_text,
    draw_recognition_scan,
)
from face_analyzer.pipeline.tracker import FaceTracker, _box_iou


class TestAnalysisConfig(unittest.TestCase):
    """Verify AnalysisConfig dataclass defaults."""

    def test_default_config(self):
        """Test default values of AnalysisConfig."""
        config = AnalysisConfig()
        self.assertEqual(config.conf_threshold, 0.5)
        self.assertEqual(config.face_detector, "yolo")
        self.assertEqual(config.active_age, set())
        self.assertEqual(config.active_gender, set())
        self.assertEqual(config.active_emotion, set())
        self.assertEqual(config.active_race, set())
        self.assertEqual(config.active_recognition, set())
        self.assertEqual(config.gallery, {})
        self.assertEqual(config.global_adjustments, {})
        self.assertEqual(config.face_adjustments, {})
        self.assertIsNone(config.metrics)
        self.assertIsNone(config.tracker)
        self.assertIsNone(config.liveness_tracker)


class TestFaceTracker(unittest.TestCase):
    """Verify FaceTracker multi-face tracking and IoU assignment."""

    def test_box_iou_disjoint_and_overlap(self):
        """Verify IoU calculation for intersecting and non-intersecting boxes."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        box3 = (0, 0, 10, 10)
        box4 = (5, 0, 15, 10)
        self.assertEqual(_box_iou(box1, box2), 0.0)
        self.assertEqual(_box_iou(box1, box3), 1.0)
        self.assertAlmostEqual(_box_iou(box1, box4), 50.0 / 150.0)

    def test_tracker_assigns_stable_ids(self):
        """Verify tracks persist stable IDs across consecutive frames."""
        tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=2)
        # Frame 1: two faces
        boxes_f1 = [(10, 10, 50, 50), (100, 100, 150, 150)]
        ids_f1 = tracker.update(boxes_f1)
        self.assertEqual(ids_f1, [1, 2])

        # Frame 2: faces move slightly
        boxes_f2 = [(12, 12, 52, 52), (102, 98, 152, 148)]
        ids_f2 = tracker.update(boxes_f2)
        self.assertEqual(ids_f2, [1, 2])

    def test_tracker_handles_disappearance_and_reset(self):
        """Verify tracker drops stale tracks and resets numbering."""
        tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=1)
        tracker.update([(10, 10, 50, 50)])
        # Miss 1 frame
        tracker.update([])
        # Miss 2nd frame -> track 1 should expire
        tracker.update([])
        # New face appears -> assigned ID 2
        ids = tracker.update([(10, 10, 50, 50)])
        self.assertEqual(ids, [2])

        # Reset -> numbering restarts from 1
        tracker.reset()
        ids_after_reset = tracker.update([(10, 10, 50, 50)])
        self.assertEqual(ids_after_reset, [1])


class TestDrawingUtilities(unittest.TestCase):
    """Verify HUD and landmark drawing utilities without crashing."""

    def test_draw_outlined_text(self):
        """Verify draw_outlined_text modifies the image buffer safely."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        draw_outlined_text(img, "Test Face", (10, 30), (0, 255, 0))
        self.assertTrue(np.any(img > 0))

    def test_draw_face_landmarks(self):
        """Verify draw_face_landmarks plots normalized coordinates inside box."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        points = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)]
        box = (10, 10, 90, 90)
        draw_face_landmarks(img, points, box)
        self.assertTrue(np.any(img > 0))

    def test_draw_hand_landmarks(self):
        """Verify draw_hand_landmarks plots hand skeleton."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        hands = [[(10, 10)] * 21]
        draw_hand_landmarks(img, hands)
        self.assertTrue(np.any(img > 0))

    def test_draw_recognition_scan(self):
        """Verify draw_recognition_scan draws recognized and unrecognized boxes."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = [((10, 10, 40, 40), True), ((50, 50, 80, 80), False)]
        draw_recognition_scan(img, faces)
        self.assertTrue(np.any(img > 0))


class TestDemographicsAggregation(unittest.TestCase):
    """Verify aggregate_demographics summarizes face metadata."""

    def test_empty_demographics(self):
        """Verify empty list produces empty dict."""
        res = aggregate_demographics([])
        self.assertEqual(res, {})

    def test_aggregation_counts(self):
        """Verify aggregation counts categorized faces."""
        cropped_faces = [
            {
                "raw_columns": {
                    "age_caffe": "25-32",
                    "gender_caffe": "Male",
                    "race_fairface": "White",
                }
            },
            {
                "raw_columns": {
                    "age_caffe": "25-32",
                    "gender_caffe": "Female",
                    "race_fairface": "White",
                }
            },
        ]
        res = aggregate_demographics(cropped_faces)
        self.assertEqual(res["age"]["caffe"]["25-32"], 2)
        self.assertEqual(res["gender"]["caffe"]["Male"], 1)
        self.assertEqual(res["gender"]["caffe"]["Female"], 1)
        self.assertEqual(res["race"]["fairface"]["White"], 2)


class TestAnalyzeFrameModular(unittest.TestCase):
    """Verify analyze_frame execution with a consolidated AnalysisConfig."""

    def test_signature_is_models_frame_config(self):
        """Verify analyze_frame takes exactly the models, the frame and one config."""
        self.assertEqual(list(inspect.signature(analyze_frame).parameters), ["models", "frame", "config"])

    def test_analyze_frame_empty_no_faces(self):
        """Verify analyze_frame returns empty face list when no faces detected."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        models = Models(face_net=MagicMock())
        with patch("face_analyzer.pipeline.stages.detect_faces", return_value=[]):
            annotated, faces, has_faces, hands = analyze_frame(models, frame, AnalysisConfig())
            self.assertEqual(len(faces), 0)
            self.assertFalse(has_faces)
            self.assertFalse(hands)
            self.assertEqual(annotated.shape, frame.shape)

    def test_analyze_frame_honours_config_detector_and_threshold(self):
        """Verify the config's detector and confidence threshold reach face detection."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        models = Models(face_net=MagicMock())
        config = AnalysisConfig(conf_threshold=0.6, face_detector="ssd")
        with patch("face_analyzer.pipeline.stages.detect_faces", return_value=[]) as detect:
            analyze_frame(models, frame, config)
        self.assertEqual(detect.call_args.args[-1], 0.6)


if __name__ == "__main__":
    unittest.main()
