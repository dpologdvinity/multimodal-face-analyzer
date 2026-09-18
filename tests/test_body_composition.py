import unittest
from types import SimpleNamespace

from src.body_composition import (
    BODY_COMPOSITION_MODEL_OPTIONS,
    estimate_face_composition,
)


def _landmark_result(points):
    return SimpleNamespace(face_landmarks=[
        [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]
    ])


class BodyCompositionTests(unittest.TestCase):
    def test_face_geometry_backend_is_the_single_supported_option(self):
        self.assertEqual(BODY_COMPOSITION_MODEL_OPTIONS, ["face_geometry"])

    def test_face_geometry_returns_relative_proxy_for_sufficient_landmarks(self):
        points = [
            (0.20, 0.10), (0.80, 0.10),
            (0.10, 0.50), (0.90, 0.50),
            (0.24, 0.80), (0.76, 0.80),
            (0.50, 0.95),
        ]

        output = estimate_face_composition(_landmark_result(points))

        self.assertIn("facial adiposity proxy", output)
        self.assertRegex(output, r"index=\d+/100")
        self.assertIn("research only", output)

    def test_face_geometry_rejects_missing_landmarks(self):
        result = SimpleNamespace(face_landmarks=[])

        output = estimate_face_composition(result)

        self.assertEqual(output, "insufficient landmarks")

    def test_face_geometry_index_increases_with_lower_face_width(self):
        narrow = _landmark_result([
            (0.20, 0.10), (0.80, 0.10),
            (0.10, 0.50), (0.90, 0.50),
            (0.35, 0.80), (0.65, 0.80),
            (0.50, 0.95),
        ])
        wide = _landmark_result([
            (0.20, 0.10), (0.80, 0.10),
            (0.10, 0.50), (0.90, 0.50),
            (0.20, 0.80), (0.80, 0.80),
            (0.50, 0.95),
        ])

        narrow_index = int(estimate_face_composition(narrow).split("index=")[1].split("/")[0])
        wide_index = int(estimate_face_composition(wide).split("index=")[1].split("/")[0])

        self.assertGreater(wide_index, narrow_index)


if __name__ == "__main__":
    unittest.main()
