"""Keep confirmed evaluation independent of guessed labels and missing outputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark


class ConfirmedBenchmarkTests(unittest.TestCase):
    def test_reads_exact_labels_without_widening_age(self):
        labels = benchmark.parse_confirmed("### portrait.jpg\n* Age = 77\n* Hair = White\n* Eyes = Dark brown / Black\n* Race = White\n")
        self.assertEqual(labels, {"portrait.jpg": {"age": 77, "hair color": ["white"], "eye color": ["brown", "black"]}})

    def test_rejects_duplicate_or_unsafe_image_names(self):
        for text in ("### ../portrait.jpg\n* Age = 77", "### p.jpg\n* Age = 3\n### p.jpg\n* Age = 4"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                benchmark.parse_confirmed(text)

    def test_missing_output_stays_in_denominator(self):
        records = [
            {"truth": {"emotion": ["happy"]}, "outputs": [{"Feature": "EMOTION", "Model": "fused", "Output": "happy"}]},
            {"truth": {"emotion": ["sad"]}, "outputs": []},
        ]
        score = benchmark.score_confirmed(records, {"emotion": ["fused", "unavailable"]})
        self.assertEqual(score["emotion"]["fused"]["correct"], 1)
        self.assertEqual(score["emotion"]["fused"]["total"], 2)
        self.assertEqual(score["emotion"]["unavailable"]["correct"], 0)
        self.assertEqual(score["emotion"]["unavailable"]["total"], 2)

    def test_age_bucket_not_exact_and_uncertain_not_answer(self):
        records = [{"truth": {"age": 77}, "outputs": [
            {"Feature": "AGE", "Model": "bucket", "Output": "70+"},
            {"Feature": "AGE", "Model": "best", "Output": "74 (mivolo)"},
            {"Feature": "AGE", "Model": "mivolo", "Output": "74"},
            {"Feature": "AGE", "Model": "uncertain", "Output": "uncertain (mean 77, SD 12)"},
        ]}]
        score = benchmark.score_confirmed(records, {"age": ["bucket", "best", "uncertain"]})["age"]
        self.assertEqual(score["bucket"]["correct"], 0)
        self.assertEqual(score["bucket"]["bucket_contains"], 1)
        self.assertEqual(score["best"]["correct"], 0)
        self.assertEqual(score["best"]["within_5_years"], 1)
        self.assertEqual(score["best"]["mae_years"], 3)
        self.assertEqual(score["uncertain"]["answered"], 0)

    def test_headline_cannot_launder_bucket_or_uncertainty(self):
        for source, prediction, headline in (("fairface", "3-9", "6 (fairface)"),
                                             ("dex", "uncertain (mean 6, SD 12)", "6 (dex)")):
            record = {"truth": {"age": 6}, "outputs": [
                {"Feature": "AGE", "Model": "best", "Output": headline},
                {"Feature": "AGE", "Model": source, "Output": prediction},
            ]}
            score = benchmark.score_confirmed([record], {"age": ["best"]})["age"]["best"]
            self.assertEqual(score["correct"], 0)
            self.assertEqual(score["numeric_answers"], 0)

    def test_no_empty_or_malformed_labels(self):
        for text in ("", "### p.jpg\n* Age = old", "### p.jpg\n* Eyes =", "### p.jpg\n* Age = 4\n* Age = 5"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                benchmark.parse_confirmed(text)


if __name__ == "__main__":
    unittest.main()
