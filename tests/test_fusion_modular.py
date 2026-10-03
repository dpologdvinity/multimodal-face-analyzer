"""Tests for the src/fusion subpackage (ranking.py + ensembles.py).

Verifies that all exported symbols exist and behave correctly when imported
directly from src.fusion rather than through src.inference -- i.e. that
the subpackage is a self-contained, importable unit.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.fusion import (
    select_age,
    with_headline,
    _weighted_median,
    _format_results,
    _sanitize_column_name,
    _gather_face_results,
    fuse_gender,
    canonical_race_probabilities,
    fuse_race,
    fuse_emotion,
    _softmax,
)
from src.fusion import RACE_CANONICAL_LABELS, RACE_LABELS_FAIRFACE


class WeightedMedianTests(unittest.TestCase):
    """_weighted_median: value where cumulative weight first reaches half."""

    def test_picks_value_where_cumulative_weight_reaches_half(self):
        self.assertEqual(_weighted_median([10.0, 20.0, 30.0], [1.0, 5.0, 1.0]), 20.0)

    def test_heavier_model_outranks_two_lighter_ones(self):
        self.assertEqual(_weighted_median([10.0, 50.0, 12.0], [1.0, 9.0, 1.0]), 50.0)


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


class FuseRaceTests(unittest.TestCase):
    """fuse_race: weighted blend across canonical race distributions."""

    def test_blends_both_backends(self):
        fused = fuse_race({
            "fairface": {"white": 0.9, "black": 0.1},
            "deepface": {"white": 0.7, "black": 0.3},
        })
        self.assertEqual(fused, "White")

    def test_shows_close_runner_up(self):
        fused = fuse_race({
            "fairface": {"white": 0.52, "black": 0.48},
            "deepface": {"white": 0.5, "black": 0.5},
        })
        self.assertEqual(fused, "White (51%)/Black (49%)")

    def test_needs_at_least_two_models(self):
        self.assertIsNone(fuse_race({"fairface": {"white": 1.0}}))


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


class SoftmaxTests(unittest.TestCase):
    """_softmax: numerically stable softmax."""

    def test_outputs_sum_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=6)

    def test_larger_input_gets_higher_probability(self):
        x = np.array([0.0, 10.0])
        result = _softmax(x)
        self.assertGreater(result[1], result[0])

    def test_numerical_stability_with_large_values(self):
        x = np.array([1000.0, 1001.0])
        result = _softmax(x)
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
