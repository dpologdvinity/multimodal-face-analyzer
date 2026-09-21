import unittest

import numpy as np

from src.inference import align_face_with_landmarks


class FairFaceAlignmentTests(unittest.TestCase):
    def test_landmark_alignment_returns_target_sized_image(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        landmarks = np.array(
            [[60, 60], [120, 60], [90, 90], [68, 120], [112, 120]],
            dtype=np.float32,
        )

        aligned = align_face_with_landmarks(frame, landmarks, 224)

        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, (224, 224, 3))

    def test_invalid_landmarks_fall_back_to_bbox_alignment(self):
        frame = np.full((180, 180, 3), 127, dtype=np.uint8)

        aligned = align_face_with_landmarks(frame, np.zeros((4, 2), dtype=np.float32), 224)

        self.assertIsNone(aligned)


if __name__ == "__main__":
    unittest.main()
