import unittest

import numpy as np

from face_analyzer import inference


class SelectAgeTests(unittest.TestCase):
    """Age names its most reliable model rather than blending -- see AGE_MODEL_RELIABILITY."""

    def test_needs_at_least_two_models(self):
        self.assertIsNone(inference.select_age({"mivolo": 30.0}))
        self.assertIsNone(inference.select_age({}))

    def test_picks_the_most_reliable_model_present(self):
        self.assertEqual(
            inference.select_age({"caffe": 5.0, "mivolo": 30.0, "fairface": 34.5}),
            ("30", "mivolo"),
        )

    def test_falls_back_down_the_ranking(self):
        self.assertEqual(inference.select_age({"caffe": 5.0, "dex": 41.2}), ("41", "dex"))

    def test_ignores_unusable_values(self):
        self.assertIsNone(inference.select_age({"mivolo": float("nan"), "dex": 200.0}))
        self.assertIsNone(inference.select_age({"mivolo": 30.0, "dex": float("inf")}))

    def test_unusable_top_model_hands_off_to_the_next(self):
        self.assertEqual(
            inference.select_age({"mivolo": float("nan"), "fairface": 24.5, "dex": 26.0}),
            ("24", "fairface"),
        )


class SelectGenderTests(unittest.TestCase):
    """Gender names its most reliable model rather than blending -- see GENDER_MODEL_RELIABILITY."""

    def test_picks_mivolo_over_fairface(self):
        self.assertEqual(
            inference.select_gender({"fairface": "Female", "mivolo": "Male"}), ("Male", "mivolo"),
        )

    def test_needs_at_least_two_models(self):
        self.assertIsNone(inference.select_gender({"mivolo": "Male"}))


class CanonicalRaceTests(unittest.TestCase):
    def test_fairface_asian_classes_collapse_into_one_key(self):
        probs = np.zeros(len(inference.RACE_LABELS_FAIRFACE))
        probs[inference.RACE_LABELS_FAIRFACE.index("East Asian")] = 0.4
        probs[inference.RACE_LABELS_FAIRFACE.index("Southeast Asian")] = 0.6
        canonical = inference.canonical_race_probabilities(probs, inference.RACE_LABELS_FAIRFACE)
        self.assertAlmostEqual(canonical["asian"], 1.0)

    def test_display_names_never_contain_the_top_two_separator(self):
        for label in inference.RACE_CANONICAL_LABELS.values():
            self.assertNotIn("/", label)


class SelectRaceTests(unittest.TestCase):
    """Race names its most reliable model rather than blending -- see RACE_MODEL_RELIABILITY."""

    def test_picks_fairface_over_deepface(self):
        self.assertEqual(
            inference.select_race({"deepface": "black", "fairface": "White"}), ("White", "fairface"),
        )

    def test_needs_at_least_two_models(self):
        self.assertIsNone(inference.select_race({"deepface": "black"}))


class FuseEmotionTests(unittest.TestCase):
    def test_votes_across_differing_spellings_of_one_state(self):
        # ferplus says "happiness", dan says "happy" -- one state, two spellings, two votes.
        self.assertEqual(
            inference.fuse_emotion({"ferplus": "happiness", "dan": "happy", "mini_xception": "sad"}),
            "happy",
        )

    def test_accurate_models_outweigh_an_inaccurate_one(self):
        self.assertEqual(
            inference.fuse_emotion({"dan": "neutral", "mini_xception": "sad"}), "neutral",
        )

    def test_needs_at_least_two_models(self):
        self.assertIsNone(inference.fuse_emotion({"dan": "happy"}))


class WithHeadlineTests(unittest.TestCase):
    def test_combined_answer_leads_the_model_pairs(self):
        self.assertEqual(
            inference.with_headline([("dan", "happy"), ("ferplus", "happiness")], "happy"),
            [("fused", "happy"), ("dan", "happy"), ("ferplus", "happiness")],
        )

    def test_age_uses_its_own_row_name(self):
        self.assertEqual(
            inference.with_headline([("mivolo", "32")], "32 (mivolo)", inference.BEST_MODEL_KEY),
            [("best", "32 (mivolo)"), ("mivolo", "32")],
        )

    def test_pairs_are_unchanged_without_a_combined_answer(self):
        self.assertEqual(inference.with_headline([("mivolo", "32")], None), [("mivolo", "32")])


if __name__ == "__main__":
    unittest.main()
