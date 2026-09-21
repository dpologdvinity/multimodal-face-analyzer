import unittest

import numpy as np

from src.inference import body_crop_bounds, predict_age_mivolo, predict_gender_mivolo


class _FakeMiVOLO:
    def __init__(self):
        self.calls = []

    def predict_face_with_body(self, face, body):
        self.calls.append((face, body))
        return 31.4, "male", 0.91


class MiVOLOBodyContextTests(unittest.TestCase):
    def test_body_crop_expands_around_face_and_clamps_to_frame(self):
        self.assertEqual(
            body_crop_bounds((40, 30, 80, 70), (200, 200)),
            (10, 10, 110, 190),
        )

    def test_mivolo_predictors_use_face_and_body_inputs(self):
        net = _FakeMiVOLO()
        face = np.zeros((40, 40, 3), dtype=np.uint8)
        body = np.ones((180, 100, 3), dtype=np.uint8)

        self.assertEqual(predict_age_mivolo(net, face, body), "31")
        self.assertEqual(predict_gender_mivolo(net, face, body), "Male")
        self.assertEqual(len(net.calls), 2)
        self.assertIs(net.calls[0][0], face)
        self.assertIs(net.calls[0][1], body)


if __name__ == "__main__":
    unittest.main()
