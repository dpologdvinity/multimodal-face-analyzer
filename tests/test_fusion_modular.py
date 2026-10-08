"""Tests for the src/face_analyzer/fusion subpackage (ranking.py + ensembles.py).

Verifies that all exported symbols exist and behave correctly when imported
directly from face_analyzer.fusion rather than through face_analyzer.inference -- i.e. that
the subpackage is a self-contained, importable unit.
"""
from __future__ import annotations

import unittest

import numpy as np

from face_analyzer.fusion import (
    BEST_MODEL_KEY,
    FUSED_MODEL_KEY,
    RACE_CANONICAL_LABELS,
    RACE_LABELS_FAIRFACE,
    _format_results,
    _gather_face_results,
    _sanitize_column_name,
    canonical_race_probabilities,
    fuse_emotion,
    fuse_gender,
    select_age,
    select_race,
    with_headline,
)


class SelectAgeTests(unittest.TestCase):
    """select_age: picks the most reliable model rather than blending."""

    def test_needs_at_least_two_models(self):
        self.assertIsNone(select_age({"mivolo": 30.0}))
        self.assertIsNone(select_age({}))

    def test_picks_the_most_reliable_model_present(self):
        self.assertEqual(
            select_age({"caffe": 5.0, "mivolo": 30.0, "fairface": 34.5}),
            ("30", "mivolo"),
        )

    def test_falls_back_down_the_ranking(self):
        self.assertEqual(select_age({"caffe": 5.0, "dex": 41.2}), ("41", "dex"))

    def test_ignores_unusable_values(self):
        self.assertIsNone(select_age({"mivolo": float("nan"), "dex": 200.0}))
        self.assertIsNone(select_age({"mivolo": 30.0, "dex": float("inf")}))

    def test_unusable_top_model_hands_off_to_next(self):
        self.assertEqual(
            select_age({"mivolo": float("nan"), "fairface": 24.5, "dex": 26.0}),
            ("24", "fairface"),
        )


class FuseGenderTests(unittest.TestCase):
    """fuse_gender: majority rule by weighted mean of P(Male)."""

    def test_weights_more_accurate_model_higher(self):
        # mivolo (weight 3.0) confidently says Male; caffe (weight 0.5) says Female.
        self.assertEqual(fuse_gender({"mivolo": 1.0, "caffe": 0.0}), "Male")
        self.assertEqual(fuse_gender({"mivolo": 0.0, "caffe": 1.0}), "Female")

    def test_needs_at_least_two_models(self):
        self.assertIsNone(fuse_gender({"mivolo": 1.0}))

    def test_tie_resolved_to_male_at_boundary(self):
        # Exactly 0.5 maps to Male.
        result = fuse_gender({"fairface": 0.5, "caffe": 0.5})
        self.assertEqual(result, "Male")


class SelectRaceTests(unittest.TestCase):
    """select_race: picks the most reliable race model rather than blending."""

    def test_picks_fairface_when_both_models_present(self):
        self.assertEqual(
            select_race({"deepface": "black", "fairface": "White (46%)/East Asian (41%)"}),
            ("White (46%)/East Asian (41%)", "fairface"),
        )

    def test_needs_at_least_two_models(self):
        self.assertIsNone(select_race({"fairface": "White"}))
        self.assertIsNone(select_race({"fairface": "White", "deepface": ""}))
        self.assertIsNone(select_race({}))

    def test_unknown_model_ranks_last(self):
        self.assertEqual(select_race({"newnet": "Indian", "deepface": "white"}), ("white", "deepface"))


class FuseEmotionTests(unittest.TestCase):
    """fuse_emotion: weighted majority vote over canonical labels."""

    def test_votes_across_differing_spellings_of_one_state(self):
        # ferplus says "happiness", dan says "happy" -- same canonical state.
        self.assertEqual(
            fuse_emotion({"ferplus": "happiness", "dan": "happy", "mini_xception": "sad"}),
            "happy",
        )

    def test_accurate_models_outweigh_inaccurate_one(self):
        self.assertEqual(
            fuse_emotion({"dan": "neutral", "mini_xception": "sad"}), "neutral",
        )

    def test_needs_at_least_two_models(self):
        self.assertIsNone(fuse_emotion({"dan": "happy"}))


class GatherFaceResultsTests(unittest.TestCase):
    """_gather_face_results: flatten (feature, model_key, value) to {col: value}."""

    def test_returns_dict_with_sanitized_column_names(self):
        pairs_by_feature = {
            "age": [("mivolo", "28"), ("caffe", "(25-32)")],
            "gender": [("mivolo", "Male")],
        }
        result = _gather_face_results(pairs_by_feature)
        self.assertIsInstance(result, dict)
        self.assertIn("age_mivolo", result)
        self.assertIn("age_caffe", result)
        self.assertIn("gender_mivolo", result)
        self.assertEqual(result["age_mivolo"], "28")
        self.assertEqual(result["gender_mivolo"], "Male")

    def test_shape_matches_total_model_outputs(self):
        pairs_by_feature = {
            "race": [("fairface", "White"), ("deepface", "White")],
            "emotion": [("dan", "happy"), ("ferplus", "happiness"), ("mini_xception", "sad")],
        }
        result = _gather_face_results(pairs_by_feature)
        self.assertEqual(len(result), 5)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_gather_face_results({}), {})


class SanitizeColumnNameTests(unittest.TestCase):
    """_sanitize_column_name: safe SQLite column identifier."""

    def test_basic_names(self):
        self.assertEqual(_sanitize_column_name("age", "mivolo"), "age_mivolo")

    def test_lowercases_and_replaces_spaces(self):
        self.assertEqual(_sanitize_column_name("Hair Color", "colorimetric"), "hair_color_colorimetric")

    def test_replaces_hyphens(self):
        self.assertEqual(_sanitize_column_name("eye-color", "colorimetric"), "eye_color_colorimetric")


class FormatResultsTests(unittest.TestCase):
    """_format_results: plain values without model-name prefix."""

    def test_returns_values_only(self):
        pairs = [("mivolo", "28"), ("caffe", "(25-32)")]
        self.assertEqual(_format_results(pairs), ["28", "(25-32)"])

    def test_empty_pairs(self):
        self.assertEqual(_format_results([]), [])


class WithHeadlineTests(unittest.TestCase):
    """with_headline: prepends headline combined answer to model pairs."""

    def test_combined_answer_leads_the_model_pairs(self):
        self.assertEqual(
            with_headline([("dan", "happy"), ("ferplus", "happiness")], "happy"),
            [(FUSED_MODEL_KEY, "happy"), ("dan", "happy"), ("ferplus", "happiness")],
        )

    def test_age_uses_its_own_row_name(self):
        self.assertEqual(
            with_headline([("mivolo", "32")], "32 (mivolo)", BEST_MODEL_KEY),
            [(BEST_MODEL_KEY, "32 (mivolo)"), ("mivolo", "32")],
        )

    def test_pairs_are_unchanged_without_a_combined_answer(self):
        self.assertEqual(with_headline([("mivolo", "32")], None), [("mivolo", "32")])


class CanonicalRaceProbabilitiesTests(unittest.TestCase):
    """canonical_race_probabilities: map backend class probs onto canonical categories."""

    def test_reexpresses_probabilities_across_canonical_keys(self):
        probs = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
        canonical = canonical_race_probabilities(probs, RACE_LABELS_FAIRFACE)
        self.assertAlmostEqual(sum(canonical.values()), 1.0, places=5)
        for key in RACE_CANONICAL_LABELS:
            self.assertIn(key, canonical)

    def test_handles_zero_sum_safely(self):
        probs = np.zeros(len(RACE_LABELS_FAIRFACE))
        canonical = canonical_race_probabilities(probs, RACE_LABELS_FAIRFACE)
        self.assertEqual(canonical["white"], 0.0)


if __name__ == "__main__":
    unittest.main()
