import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src import inference


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
        y, x = np.mgrid[:224, :224]
        self.frame = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)

    def test_dlib_reference_preserves_pixels_and_padding(self):
        aligned = inference.align_face_with_landmarks(self.frame, REFERENCE, 224)
        np.testing.assert_allclose(aligned[1:-1, 1:-1], self.frame[1:-1, 1:-1], atol=1)

    def test_recovers_rotated_translated_scaled_face(self):
        transform = cv2.getRotationMatrix2D((112, 112), 20, 1.2)
        transform[:, 2] += (35, 30)
        moved = cv2.warpAffine(self.frame, transform, (320, 320))
        points = np.column_stack((REFERENCE, np.ones(5))) @ transform.T
        aligned = inference.align_face_with_landmarks(moved, points, 224)
        np.testing.assert_allclose(aligned[25:-25, 25:-25], self.frame[25:-25, 25:-25], atol=2)

    def test_mediapipe_uses_four_eye_corners_and_nose(self):
        points = [(0.5, 0.5)] * 468
        for index, point in zip((263, 362, 33, 133, 1), REFERENCE / 224):
            points[index] = tuple(point)
        actual = inference.fairface_landmarks_from_mediapipe(points, 224, 224)
        np.testing.assert_allclose(actual, REFERENCE, atol=1e-4)

    def test_invalid_landmarks_use_original_bbox_input(self):
        net = RecordingNet()
        box = (40, 40, 180, 200)
        inference._fairface_forward(net, self.frame, box, "age_output")
        expected = net.blob.copy()
        for bad in (np.zeros((4, 2)), np.zeros((5, 2)), np.full((5, 2), np.nan),
                    np.column_stack((np.arange(5), np.arange(5)))):
            with self.subTest(points=bad):
                self.assertIsNone(inference.align_face_with_landmarks(self.frame, bad, 224))
                inference._fairface_forward(net, self.frame, box, "age_output", bad)
                np.testing.assert_array_equal(net.blob, expected)

    def test_analyze_frame_translates_crop_landmarks_before_inference(self):
        frame = np.tile(self.frame, (2, 2, 1))
        box = (80, 90, 240, 280)
        x1, y1, x2, y2 = inference.face_crop_bounds(box, frame.shape[:2])
        points = [(0.5, 0.5)] * 468
        for index, point in zip((263, 362, 33, 133, 1), REFERENCE / 224):
            points[index] = tuple(point)
        result = SimpleNamespace(face_landmarks=[
            [SimpleNamespace(x=x, y=y) for x, y in points]
        ])
        net = RecordingNet()
        landmarks = REFERENCE / 224 * (x2-x1, y2-y1) + (x1, y1)
        inference._fairface_forward(net, frame, box, "age_output", landmarks)
        expected = net.blob.copy()
        models = inference.Models(face_net=None, age_nets={"fairface": net},
                                  face_landmarks_nets={"mediapipe": object()})
        with patch.object(inference, "detect_faces", return_value=[box]), \
             patch.object(inference, "_detect_face_landmarker", return_value=result):
            output = inference.analyze_frame(
                models, frame, 0.5, active_age={"fairface"}, active_gender=set(),
                active_emotion=set(), active_race=set(), active_recognition=set(), gallery={},
                active_glasses=set(), active_mask=set(), active_hair_color=set(), active_eye_color=set(),
                active_face_landmarks=set(), active_hands=set(), active_gaze=set(),
                global_adjustments={}, face_adjustments={},
            )
        self.assertEqual(output[1][0]["raw_columns"]["age_fairface"], "70+")
        # Float32 landmark roundoff can move interpolation by one uint8 level.
        np.testing.assert_allclose(net.blob, expected, atol=1 / (255 * 0.224))


if __name__ == "__main__":
    unittest.main()
