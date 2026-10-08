"""Pin the held-out eval's label mappings, sampling, statistics and fusion-weight fitting."""
import importlib.util
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from face_analyzer.core.constants import (
    AGE_LIST,
    GENDER_FUSION_WEIGHTS,
    RACE_CANONICAL_LABELS,
    RACE_FUSION_WEIGHTS,
    RACE_LABELS_DEEPFACE,
    RACE_LABELS_FAIRFACE,
)

# tools/ is a script directory, not a package; load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "eval_heldout", Path(__file__).resolve().parents[1] / "tools" / "eval_heldout.py"
)
eval_heldout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_heldout)


def _record(row, race, gender, age, raw, box=(0, 0, 10, 10)):
    return {"row": row, "truth": {"race": race, "gender": gender, "age": age}, "raw": raw,
            "box": list(box) if box else None, "error": None, "outputs": {},
            "fairface_aligned": True}


class ImportTests(unittest.TestCase):
    def test_scoring_needs_no_deep_learning_framework(self):
        # A fresh interpreter, since other tests may already have imported torch in this one.
        probe = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location("
                 f"'m', {str(_SPEC.origin)!r}); spec.loader.exec_module("
                 "importlib.util.module_from_spec(spec)); "
                 "print(sorted({'torch', 'tensorflow'} & set(sys.modules)))")
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                check=True, cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")})
        self.assertEqual(result.stdout.strip(), "[]")


class AgeBucketTests(unittest.TestCase):
    def test_continuous_ages_use_the_displayed_integer(self):
        cases = {0: 0, 2.4: 0, 2.5: 0, 2.6: 1, 9.4: 1, 10: 2, 19.6: 3, 29: 3, 30: 4,
                 69.4: 7, 70: 8, 105: 8, -1: 0}
        for age, bucket in cases.items():
            self.assertEqual(eval_heldout.age_to_bucket(age), bucket, age)

    def test_caffe_buckets_map_by_midpoint(self):
        mapped = [eval_heldout.caffe_to_fairface_bucket(i) for i in range(len(AGE_LIST))]
        # (0-2) (4-6) (8-12) (15-20) (25-32) (38-43) (48-53) (60-100)
        self.assertEqual(mapped, [0, 1, 2, 2, 3, 5, 6, 8])


class RaceMappingTests(unittest.TestCase):
    def test_dataset_and_backend_labels_reach_the_six_canonical_classes(self):
        for labels in (eval_heldout.DATASET_RACES, RACE_LABELS_FAIRFACE, RACE_LABELS_DEEPFACE):
            canonical = {eval_heldout.canonical_race(label) for label in labels}
            self.assertEqual(canonical, set(RACE_CANONICAL_LABELS), labels)

    def test_east_and_southeast_asian_merge(self):
        self.assertEqual(eval_heldout.canonical_race("East Asian"), "asian")
        self.assertEqual(eval_heldout.canonical_race("Southeast Asian"), "asian")
        self.assertEqual(eval_heldout.canonical_race("latino hispanic"), "latino")
        self.assertEqual(eval_heldout.canonical_race("Latino_Hispanic"), "latino")

    def test_runner_up_label_scores_its_first_class_only(self):
        self.assertEqual(eval_heldout.shown_top1("White (52%)/Black (47%)"), "White")
        self.assertEqual(eval_heldout.shown_top1("Middle Eastern"), "Middle Eastern")

    def test_backend_top1_comes_from_the_backend_label_order(self):
        raw = {"race_probs/fairface": [0, 0, 0, 0.2, 0.7, 0.1, 0],
               "race_probs/deepface": [0, 0, 0, 0, 0, 1.0]}
        self.assertEqual(eval_heldout.race_labels(raw), {"fairface": "asian", "deepface": "latino"})


class SamplingTests(unittest.TestCase):
    strata = [i % 3 for i in range(30)] + [3] * 10

    def test_sample_is_proportional_deterministic_and_nested(self):
        sample = eval_heldout.stratified_sample(self.strata, 20, seed=7)
        self.assertEqual(sample, eval_heldout.stratified_sample(self.strata, 20, seed=7))
        self.assertEqual(len(sample), 20)
        counts = [sum(self.strata[i] == key for i in sample) for key in range(4)]
        self.assertEqual(counts, [5, 5, 5, 5])
        smaller = eval_heldout.stratified_sample(self.strata, 8, seed=7)
        self.assertTrue(set(smaller) <= set(sample))

    def test_split_is_disjoint_and_balanced_per_stratum(self):
        sample = eval_heldout.stratified_sample(self.strata, 24, seed=7)
        fit, test = eval_heldout.stratified_split(sample, self.strata, seed=7)
        self.assertEqual(sorted(fit + test), sample)
        self.assertFalse(set(fit) & set(test))
        for key in range(4):
            self.assertEqual(sum(self.strata[i] == key for i in fit),
                             sum(self.strata[i] == key for i in test))


class DetectionTests(unittest.TestCase):
    def test_largest_box_is_ranked_by_in_frame_area(self):
        boxes = [[10, 10, 60, 60], [150, 150, 900, 900], [-5, -5, 30, 30]]
        self.assertEqual(eval_heldout.largest_in_frame(boxes, 100, 100), [10, 10, 60, 60])

    def test_boxes_wholly_outside_the_frame_are_a_miss(self):
        self.assertIsNone(eval_heldout.largest_in_frame([[200, 200, 400, 400]], 100, 100))
        self.assertIsNone(eval_heldout.largest_in_frame([], 100, 100))


class BootstrapTests(unittest.TestCase):
    values = [1] * 70 + [0] * 30

    def test_same_seed_gives_the_same_interval(self):
        first = eval_heldout.bootstrap_ci(self.values, n_resamples=500, seed=3)
        self.assertEqual(first, eval_heldout.bootstrap_ci(self.values, n_resamples=500, seed=3))
        self.assertNotEqual(first, eval_heldout.bootstrap_ci(self.values, n_resamples=500, seed=4))
        self.assertLess(first[0], 0.7)
        self.assertGreater(first[1], 0.7)

    def test_degenerate_inputs(self):
        self.assertEqual(eval_heldout.bootstrap_ci([1, 1, 1], seed=0), (1.0, 1.0))
        self.assertTrue(all(math.isnan(v) for v in eval_heldout.bootstrap_ci([], seed=0)))


class FusionWeightTests(unittest.TestCase):
    def test_log_odds_weights(self):
        weights = eval_heldout.log_odds_weights({"a": 0.9, "b": 0.6, "c": 0.4}, n_classes=2)
        self.assertAlmostEqual(weights["a"], math.log(9), places=3)
        self.assertAlmostEqual(weights["b"], math.log(1.5), places=3)
        self.assertEqual(weights["c"], 0.0)

    def test_multiclass_weights_count_above_chance_not_above_half(self):
        weights = eval_heldout.log_odds_weights({"a": 0.3}, n_classes=6)
        self.assertAlmostEqual(weights["a"], math.log(5 * 0.3 / 0.7), places=3)

    def test_all_at_chance_falls_back_to_equal_weights(self):
        self.assertEqual(eval_heldout.log_odds_weights({"a": 0.5, "b": 0.4}, 2), {"a": 1.0, "b": 1.0})

    def test_substitute_weights_change_the_answer_without_touching_shipped_ones(self):
        shipped_gender, shipped_race = dict(GENDER_FUSION_WEIGHTS), dict(RACE_FUSION_WEIGHTS)
        raw = {"mivolo/face": [30.0, "Male"], "gender_probs/fairface": [0.2, 0.8],
               "gender_probs/caffe": [0.1, 0.9], "gender_probs/deepface": [0.9, 0.1]}
        self.assertEqual(eval_heldout.fused_gender(raw), "Male")
        without_mivolo = {"mivolo": 0.0, "fairface": 1.0, "caffe": 1.0, "deepface": 1.0}
        self.assertEqual(eval_heldout.fused_gender(raw, without_mivolo), "Female")
        race_raw = {"race_probs/fairface": [0.9, 0.1, 0, 0, 0, 0, 0],
                    "race_probs/deepface": [0, 0, 1.0, 0, 0, 0]}
        self.assertEqual(eval_heldout.fused_race(race_raw), "black")
        self.assertEqual(eval_heldout.fused_race(race_raw, {"fairface": 3.0, "deepface": 1.0}), "white")
        self.assertEqual(GENDER_FUSION_WEIGHTS, shipped_gender)
        self.assertEqual(RACE_FUSION_WEIGHTS, shipped_race)


class ScoreTests(unittest.TestCase):
    def test_score_counts_misses_and_fits_on_the_fit_half_only(self):
        white, black = 3, 2  # dataset race indices
        good = {"gender_probs/fairface": [0.9, 0.1], "gender_probs/caffe": [0.2, 0.8],
                "race_probs/fairface": [0.9, 0.1, 0, 0, 0, 0, 0],
                "race_probs/deepface": [0, 0, 0.6, 0.4, 0, 0],
                "age_probs/fairface": [0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                "mivolo/face": [34.0, "Male"]}
        records = [_record(row, white, 0, 3, good) for row in range(4)]
        records.append(_record(4, black, 1, 4, {}, box=None))
        report = eval_heldout.score(records, fit_rows=[0, 1], test_rows=[2, 3, 4], seed=0,
                                    n_resamples=200)
        self.assertAlmostEqual(report["detection_recall"]["test"]["value"], 2 / 3, places=4)
        self.assertEqual(report["gender"]["backends"]["fairface"]["value"], 1.0)
        self.assertEqual(report["gender"]["backends"]["caffe"]["value"], 0.0)
        self.assertEqual(report["race"]["backends"]["deepface"]["value"], 0.0)
        self.assertEqual(report["age"]["backends"]["mivolo"]["mean_bucket_offset"]["value"], 1.0)
        self.assertEqual(report["age"]["best_shipped"]["chosen_backend"]["mivolo"], 2)
        self.assertEqual(report["fusion"]["gender"]["fitted_weights"]["caffe"], 0.0)
        self.assertEqual(report["fusion"]["race"]["fit_half_accuracy"], {"fairface": 1.0, "deepface": 0.0})


if __name__ == "__main__":
    unittest.main()
