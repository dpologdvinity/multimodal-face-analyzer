"""Pin the learned stacker's JSON schema and its numpy inference against scikit-learn's predictions."""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# tools/ is a script directory, not a package; load the module by path.
_SPEC = importlib.util.spec_from_file_location("fit_stacker", ROOT / "tools" / "fit_stacker.py")
fit_stacker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fit_stacker)

REPORT = json.loads((ROOT / "docs" / "eval" / "stacker_fairface.json").read_text())
# A few cached test-half faces with scikit-learn's label and probabilities, written by
# `tools/fit_stacker.py --fixture` from the same fit as the report.
ROWS = json.loads((ROOT / "tests" / "fixtures" / "stacker_rows.json").read_text())
OPS = {"log", "logit", "value", "log1p", "bucket", "equals"}


class SchemaTests(unittest.TestCase):
    def test_every_model_is_plain_json_with_consistent_shapes(self):
        self.assertEqual(set(REPORT["attributes"]), {"gender", "race", "age"})
        for attribute, entry in REPORT["attributes"].items():
            model = entry["model"]
            n_features, n_classes = len(model["features"]), len(model["classes"])
            self.assertEqual(model["attribute"], attribute)
            self.assertEqual(len(model["mean"]), n_features)
            self.assertEqual(len(model["scale"]), n_features)
            rows = 1 if n_classes == 2 else n_classes
            self.assertEqual(np.shape(model["coef"]), (rows, n_features))
            self.assertEqual(len(model["intercept"]), rows)
            for feature in model["features"]:
                self.assertIn(feature["op"], OPS)
                self.assertIn("/", feature["source"])
            self.assertIsInstance(entry["result"]["adopted"], bool)

    def test_class_orders(self):
        attributes = REPORT["attributes"]
        self.assertEqual(attributes["gender"]["model"]["classes"], ["Female", "Male"])
        self.assertEqual(sorted(attributes["race"]["model"]["classes"]),
                         sorted(fit_stacker.heldout.CANONICAL_RACES))
        self.assertEqual(attributes["age"]["model"]["classes"], list(range(9)))


class InferenceTests(unittest.TestCase):
    def test_numpy_inference_matches_scikit_learn_on_cached_rows(self):
        for row in ROWS:
            for attribute, expected in row["expected"].items():
                with self.subTest(row=row["row"], attribute=attribute):
                    label, probs = fit_stacker.predict_stacked(REPORT["attributes"][attribute]["model"], row["raw"])
                    self.assertEqual(label, expected["label"])
                    np.testing.assert_allclose(probs, expected["probs"], rtol=0, atol=1e-12)

    def test_missing_backend_falls_back_to_the_shipped_best_model(self):
        raw = dict(ROWS[0]["raw"])
        del raw["gender_probs/caffe"]
        model = REPORT["attributes"]["gender"]["model"]
        self.assertIsNone(fit_stacker.predict_stacked(model, raw))
        self.assertEqual(fit_stacker.stacked_or_shipped(model, raw, "gender"), raw["mivolo/face"][1])
        raw["mivolo/face"] = None
        self.assertIsNone(fit_stacker.stacker_features(model["features"], raw))

    def test_saturated_probabilities_stay_finite(self):
        spec = [{"source": "p", "op": "log", "indices": [0]}, {"source": "p", "op": "logit", "indices": [1]}]
        self.assertTrue(np.all(np.isfinite(fit_stacker.stacker_features(spec, {"p": [0.0, 1.0]}))))


if __name__ == "__main__":
    unittest.main()
