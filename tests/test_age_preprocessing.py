import unittest

from src.inference import face_crop_bounds


class AgePreprocessingTests(unittest.TestCase):
    def test_face_crop_padding_scales_with_detection_size(self):
        """Verify padding scales proportionally with detection box size."""
        small = face_crop_bounds((40, 30, 80, 70), (200, 200), padding_ratio=0.1)
        large = face_crop_bounds((40, 30, 140, 130), (300, 300), padding_ratio=0.1)

        self.assertEqual(small, (36, 26, 84, 74))
        self.assertEqual(large, (30, 20, 150, 140))

    def test_face_crop_bounds_are_clamped_to_frame(self):
        """Clamp padded crop bounds to frame edges."""
        self.assertEqual(
            face_crop_bounds((2, 3, 20, 25), (30, 30), padding_ratio=0.5),
            (0, 0, 30, 30),
        )


if __name__ == "__main__":
    unittest.main()
