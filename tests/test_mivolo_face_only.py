import unittest

import numpy as np

from src.inference import mivolo_estimate, predict_age_mivolo, predict_gender_mivolo


class FakeMiVOLO:
    """Stand-in exposing both entry points, so a body-branch call would be visible if made."""

    def __init__(self):
        self.face_calls = []
        self.body_calls = []

    def predict_face(self, face):
        self.face_calls.append(face)
        return 31.4, "male", 0.99

    def predict_face_with_body(self, face, body):
        self.body_calls.append((face, body))
        return 31.4, "male", 0.99


class MiVOLOFaceOnlyTests(unittest.TestCase):
    """The bundled checkpoint is loaded with use_persons=False, so its second input branch must
    stay the zero tensor predict_face supplies -- a guessed person crop measured worse."""

    def test_estimate_uses_the_face_only_entry_point(self):
        net = FakeMiVOLO()
        face = np.zeros((60, 60, 3), dtype=np.uint8)
        self.assertEqual(mivolo_estimate(net, face), (31.4, "Male"))
        self.assertEqual(len(net.face_calls), 1)
        self.assertIs(net.face_calls[0], face)
        self.assertEqual(net.body_calls, [])

    def test_age_and_gender_wrappers_share_the_one_entry_point(self):
        net = FakeMiVOLO()
        face = np.zeros((60, 60, 3), dtype=np.uint8)
        self.assertEqual(predict_age_mivolo(net, face), "31")
        self.assertEqual(predict_gender_mivolo(net, face), "Male")
        self.assertEqual(len(net.face_calls), 2)
        self.assertEqual(net.body_calls, [])


if __name__ == "__main__":
    unittest.main()
