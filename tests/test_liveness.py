import unittest
from types import SimpleNamespace

from src.liveness import (
    LivenessTracker,
    assess_static_liveness,
    blink_score_from_landmarker,
    texture_artifact_score,
)


class LivenessTests(unittest.TestCase):
    def test_texture_score_flags_regular_high_frequency_pattern(self):
        smooth = [[128] * 32 for _ in range(32)]
        checkerboard = [
            [0 if (x + y) % 2 else 255 for x in range(32)]
            for y in range(32)
        ]

        self.assertLess(texture_artifact_score(smooth), 0.01)
        self.assertGreater(texture_artifact_score(checkerboard), 0.7)

    def test_tracker_counts_blink_transitions_and_reports_rate(self):
        tracker = LivenessTracker()

        tracker.update(4, 0.1, 0.1, now=0.0)
        tracker.update(4, 0.9, 0.1, now=1.0)
        tracker.update(4, 0.1, 0.1, now=2.0)
        result = tracker.update(4, 0.9, 0.1, now=60.0)

        self.assertEqual(result.status, "LIVE")
        self.assertEqual(result.blink_count, 2)
        self.assertAlmostEqual(result.blink_rate, 2.0)

    def test_texture_cue_overrides_blink_result_as_spoof(self):
        tracker = LivenessTracker()

        tracker.update(4, 0.1, 0.1, now=0.0)
        result = tracker.update(4, 0.9, 0.9, now=1.0)

        self.assertEqual(result.status, "SUSPECTED SPOOF")
        self.assertTrue(result.texture_artifact)

    def test_tracker_does_not_count_initially_closed_eyes_as_blink(self):
        tracker = LivenessTracker()

        result = tracker.update(4, 0.9, 0.1, now=0.0)

        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertEqual(result.blink_count, 0)

    def test_blink_score_averages_both_eye_blendshapes(self):
        result = SimpleNamespace(face_blendshapes=[[
            SimpleNamespace(category_name="eyeBlinkLeft", score=0.8),
            SimpleNamespace(category_name="eyeBlinkRight", score=0.6),
        ]])

        self.assertAlmostEqual(blink_score_from_landmarker(result), 0.7)

    def test_static_liveness_without_texture_evidence_is_inconclusive(self):
        result = assess_static_liveness(0.1)

        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertIsNone(result.blink_rate)
        self.assertIn("texture=0.10", result.summary)


if __name__ == "__main__":
    unittest.main()
