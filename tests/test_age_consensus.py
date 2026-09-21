import unittest

from src import inference


class AgeConsensusTests(unittest.TestCase):
    def test_consensus_uses_only_close_continuous_predictions(self):
        self.assertEqual(
            inference.conservative_age_consensus(
                [("caffe", "25-32"), ("ssrnet", "33"), ("mivolo", "32")]
            ),
            "32 (2 models agree within 1y)",
        )

    def test_consensus_rejects_disagreement_and_single_model(self):
        self.assertIsNone(inference.conservative_age_consensus([("ssrnet", "30"), ("mivolo", "50")]))
        self.assertIsNone(inference.conservative_age_consensus([("ssrnet", "30")]))

    def test_consensus_ignores_uncertain_or_malformed_outputs(self):
        self.assertIsNone(
            inference.conservative_age_consensus(
                [("dex", "uncertain (mean 39, SD 20)"), ("ssrnet", "39")]
            )
        )
        self.assertIsNone(inference.conservative_age_consensus([("ssrnet", "nan"), ("mivolo", "39")]))

    def test_model_results_keep_each_model_and_add_consensus_separately(self):
        rows = inference.age_model_results([("ssrnet", "31"), ("mivolo", "32")])
        self.assertEqual(
            rows,
            [
                {"Feature": "AGE", "Model": "ssrnet", "Output": "31"},
                {"Feature": "AGE", "Model": "mivolo", "Output": "32"},
                {"Feature": "AGE", "Model": "consensus", "Output": "32 (2 models agree within 1y)"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
