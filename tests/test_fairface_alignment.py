import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from face_analyzer import inference
from face_analyzer.attributes import level_face_points, level_face_region
from face_analyzer.attributes.race import _fairface_forward

# dlib get_face_chip_details: outer/inner right eye, outer/inner left eye, nose.
REFERENCE = (np.array([
    [0.8595674595992, 0.2134981538014], [0.6460604764104, 0.2289674387677],
    [0.1205750620789, 0.2137274526848], [0.3340850613712, 0.2290642403242],
    [0.4901123135679, 0.6277975316475],
]) + 0.25) / 1.5 * 224


class RecordingNet:
    def setInput(self, blob):
        self.blob = blob.copy()

    def forward(self, name):
        return np.arange(9, dtype=np.float32)[None, :]


class FairFaceAlignmentTests(unittest.TestCase):
    def setUp(self):
        # Create synthetic frame with pixel values encoding position for alignment verification
        y, x = np.mgrid[:224, :224]
        self.frame = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)

    def test_dlib_reference_preserves_pixels_and_padding(self):
        """Verify landmark alignment with dlib reference points preserves frame content."""
        aligned = inference.align_face_with_landmarks(self.frame, REFERENCE, 224)
        np.testing.assert_allclose(aligned[1:-1, 1:-1], self.frame[1:-1, 1:-1], atol=1)

    def test_recovers_rotated_translated_scaled_face(self):
        """Recover face alignment from rotated, translated, and scaled landmarks."""
        transform = cv2.getRotationMatrix2D((112, 112), 20, 1.2)
        transform[:, 2] += (35, 30)
        moved = cv2.warpAffine(self.frame, transform, (320, 320))
        points = np.column_stack((REFERENCE, np.ones(5))) @ transform.T
        aligned = inference.align_face_with_landmarks(moved, points, 224)
        np.testing.assert_allclose(aligned[25:-25, 25:-25], self.frame[25:-25, 25:-25], atol=2)

    def test_mediapipe_uses_four_eye_corners_and_nose(self):
        """Convert mediapipe landmark indices to dlib-format points."""
        points = [(0.5, 0.5)] * 468
        # Place reference points at specific mediapipe indices (eye corners, nose)
        for index, point in zip((263, 362, 33, 133, 1), REFERENCE / 224, strict=False):
            points[index] = tuple(point)
        actual = inference.fairface_landmarks_from_mediapipe(points, 224, 224)
        np.testing.assert_allclose(actual, REFERENCE, atol=1e-4)

    def test_invalid_landmarks_use_original_bbox_input(self):
        """Fall back to bounding box when landmarks are invalid (wrong shape, NaN, or degenerate)."""
        net = RecordingNet()
        box = (40, 40, 180, 200)
        _fairface_forward(net, self.frame, box, "age_output")
        expected = net.blob.copy()
        # Test multiple types of invalid landmarks: wrong count, NaN values, collinear points
        for bad in (np.zeros((4, 2)), np.zeros((5, 2)), np.full((5, 2), np.nan),
                    np.column_stack((np.arange(5), np.arange(5)))):
            with self.subTest(points=bad):
                self.assertIsNone(inference.align_face_with_landmarks(self.frame, bad, 224))
                _fairface_forward(net, self.frame, box, "age_output", bad)
                np.testing.assert_array_equal(net.blob, expected)

    def test_analyze_frame_translates_crop_landmarks_before_inference(self):
        """Verify landmarks detected in full frame are correctly translated to cropped face coords."""
        frame = np.tile(self.frame, (2, 2, 1))
        box = (80, 90, 240, 280)
        x1, y1, x2, y2 = inference.face_crop_bounds(box, frame.shape[:2])
        points = [(0.5, 0.5)] * 468
        # Place reference points at specific mediapipe indices
        for index, point in zip((263, 362, 33, 133, 1), REFERENCE / 224, strict=False):
            points[index] = tuple(point)
        result = SimpleNamespace(face_landmarks=[
            [SimpleNamespace(x=x, y=y) for x, y in points]
        ])
        net = RecordingNet()
        # Compute expected input blob as if we had manually translated landmarks
        landmarks = REFERENCE / 224 * (x2-x1, y2-y1) + (x1, y1)
        _fairface_forward(net, frame, box, "age_output", landmarks)
        expected = net.blob.copy()
        models = inference.Models(face_net=None, age_nets={"fairface": net},
                                  face_landmarks_nets={"mediapipe": object()})
        # Mock face detection and landmark detection; analyze_frame should produce same blob
        with patch("face_analyzer.pipeline.stages.detect_faces", return_value=[box]) as detect, \
             patch("face_analyzer.pipeline.stages._detect_face_landmarker", return_value=result) as landmarker:
            output = inference.analyze_frame(
                models, frame, inference.AnalysisConfig(active_age={"fairface"}),
            )
        detect.assert_called()
        landmarker.assert_called()
        self.assertEqual(output[1][0]["raw_columns"]["age_fairface"], "70+")
        # Float32 landmark roundoff can move interpolation by one uint8 level.
        np.testing.assert_allclose(net.blob, expected, atol=1 / (255 * 0.224))

    def test_detector_eye_centers_and_nose_align_like_the_reference(self):
        """Align on detector eye centers and nose as on the five-point reference they summarize."""
        mouth = [[90.0, 170.0], [134.0, 170.0]]
        points = np.array([REFERENCE[2:4].mean(axis=0), REFERENCE[0:2].mean(axis=0), REFERENCE[4], *mouth])
        landmarks = inference.fairface_landmarks_from_detector(points)
        aligned = inference.align_face_with_landmarks(self.frame, landmarks, 224)
        np.testing.assert_allclose(aligned[1:-1, 1:-1], self.frame[1:-1, 1:-1], atol=1)

    def test_detector_landmarks_with_swapped_or_missing_eyes_are_rejected(self):
        """Reject detector points a similarity transform would align upside down, or that are invalid."""
        points = np.array([[150, 70], [70, 70], [112, 120], [80, 160], [140, 160]], dtype=np.float32)
        self.assertIsNone(inference.fairface_landmarks_from_detector(points))
        self.assertIsNone(inference.fairface_landmarks_from_detector(np.full((5, 2), np.nan)))
        self.assertIsNone(inference.fairface_landmarks_from_detector(None))

    def test_level_face_points_follow_the_leveled_region(self):
        """Map a frame point to where level_face_region's rotated region shows it."""
        frame = np.zeros((300, 300, 3), np.uint8)
        point = np.array([[130.0, 120.0]], np.float32)
        cv2.circle(frame, (130, 120), 3, (255, 255, 255), -1)
        box = (100, 100, 200, 200)
        region, _ = level_face_region(frame, box, 12.0)
        x, y = np.rint(level_face_points(frame.shape, box, 12.0, point)[0]).astype(int)
        self.assertEqual(region[y, x].tolist(), [255, 255, 255])
        np.testing.assert_array_equal(level_face_points(frame.shape, box, 2.0, point), point)

    def test_analyze_frame_aligns_fairface_on_detector_landmarks_without_mediapipe(self):
        """Feed FairFace the face aligned on RetinaFace's landmarks when MediaPipe is absent."""
        frame = np.tile(self.frame, (2, 2, 1))
        box = (80, 90, 240, 280)
        points = np.array([[130, 150], [190, 150], [160, 185], [135, 220], [185, 220]], dtype=np.float32)
        net = RecordingNet()
        _fairface_forward(net, frame, box, "age_output", inference.fairface_landmarks_from_detector(points))
        expected = net.blob.copy()
        models = inference.Models(face_net=None, retinaface_nets={"retinaface": object()},
                                  age_nets={"fairface": net})
        with patch("face_analyzer.pipeline.stages.detect_faces_retinaface_landmarks",
                   return_value=([list(box)], [points])) as detect:
            inference.analyze_frame(
                models, frame, inference.AnalysisConfig(active_age={"fairface"}, face_detector="retinaface"),
            )
        detect.assert_called()
        np.testing.assert_array_equal(net.blob, expected)


if __name__ == "__main__":
    unittest.main()
