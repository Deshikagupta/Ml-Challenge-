import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import (
    macro_fbeta,
    precision_recall_fbeta,
    predictions_at_threshold,
    sweep_threshold,
)


class PrecisionRecallFbetaTests(unittest.TestCase):
    def test_zero_true_matches_zero_predicted_is_perfect(self):
        p, r, f = precision_recall_fbeta(set(), set())
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_zero_true_matches_with_false_positive(self):
        # singleton entity, matcher wrongly predicted something -> precision-heavy
        # metric must punish this hard
        p, r, f = precision_recall_fbeta(set(), {"S2-1"})
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 1.0)  # vacuously recalled all 0 true matches
        self.assertEqual(f, 0.0)

    def test_zero_predicted_with_true_matches_is_full_miss(self):
        p, r, f = precision_recall_fbeta({"S2-1", "S3-2"}, set())
        self.assertEqual(p, 1.0)  # vacuously precise (predicted nothing wrong)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)  # recall=0 drags f to 0 regardless of precision

    def test_perfect_match(self):
        p, r, f = precision_recall_fbeta({"S2-1", "S3-2"}, {"S2-1", "S3-2"})
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_false_positive_among_correct(self):
        # 2 true, predicted 2 true + 1 false -> precision=2/3, recall=1
        p, r, f = precision_recall_fbeta({"S2-1", "S3-2"}, {"S2-1", "S3-2", "S2-9"})
        self.assertAlmostEqual(p, 2 / 3)
        self.assertEqual(r, 1.0)
        # F0.5 weights precision more than recall
        beta_sq = 0.25
        expected = (1 + beta_sq) * p * r / (beta_sq * p + r)
        self.assertAlmostEqual(f, expected)

    def test_false_negative_among_correct(self):
        # 2 true, predicted only 1 of them -> precision=1, recall=1/2
        p, r, f = precision_recall_fbeta({"S2-1", "S3-2"}, {"S2-1"})
        self.assertEqual(p, 1.0)
        self.assertAlmostEqual(r, 0.5)
        self.assertLess(f, 1.0)

    def test_multiple_true_matches_across_s2_and_s3(self):
        true_ids = {"S2-1", "S2-2", "S3-1"}
        predicted = {"S2-1", "S3-1"}
        p, r, f = precision_recall_fbeta(true_ids, predicted)
        self.assertEqual(p, 1.0)
        self.assertAlmostEqual(r, 2 / 3)

    def test_f0_5_penalizes_false_positives_harder_than_f1_would(self):
        # one true id, one false positive alongside it
        p, r, f05 = precision_recall_fbeta({"S2-1"}, {"S2-1", "S2-2"})
        # naive F1 for comparison
        f1 = 2 * p * r / (p + r)
        self.assertLess(f05, f1)


class MacroFBetaTests(unittest.TestCase):
    def test_includes_zero_true_match_singletons_in_macro_average(self):
        true_by_s1 = {"S1-1": {"S2-1"}}  # S1-2 has no ground-truth entry -> empty
        predicted_by_s1 = {"S1-1": {"S2-1"}, "S1-2": set()}
        report = macro_fbeta(true_by_s1, predicted_by_s1, all_s1_ids=["S1-1", "S1-2"])
        self.assertEqual(report.entity_count, 2)
        self.assertEqual(report.macro_fbeta, 1.0)  # both entities perfectly correct

    def test_missing_entity_from_predicted_dict_defaults_to_empty(self):
        true_by_s1 = {"S1-1": set()}
        predicted_by_s1 = {}  # S1-1 simply absent -- must default to no predictions
        report = macro_fbeta(true_by_s1, predicted_by_s1, all_s1_ids=["S1-1"])
        self.assertEqual(report.macro_fbeta, 1.0)

    def test_macro_average_is_per_entity_not_pooled(self):
        # one entity with a single wrong prediction (singleton, f=0), one perfect entity
        true_by_s1 = {"S1-1": set(), "S1-2": {"S2-5"}}
        predicted_by_s1 = {"S1-1": {"S2-1"}, "S1-2": {"S2-5"}}
        report = macro_fbeta(true_by_s1, predicted_by_s1, all_s1_ids=["S1-1", "S1-2"])
        self.assertAlmostEqual(report.macro_fbeta, 0.5)  # average of 0.0 and 1.0

    def test_per_entity_breakdown_optional(self):
        report = macro_fbeta(
            {"S1-1": {"S2-1"}}, {"S1-1": {"S2-1"}}, all_s1_ids=["S1-1"], include_per_entity=True
        )
        self.assertIn("S1-1", report.per_entity)
        self.assertEqual(report.per_entity["S1-1"], (1.0, 1.0, 1.0))

        report_off = macro_fbeta(
            {"S1-1": {"S2-1"}}, {"S1-1": {"S2-1"}}, all_s1_ids=["S1-1"], include_per_entity=False
        )
        self.assertEqual(report_off.per_entity, {})


class ThresholdSweepTests(unittest.TestCase):
    def test_sweep_picks_threshold_maximizing_macro_fbeta(self):
        # S1-1: true match is S2-1 (score .9); S2-2 is a false candidate (score .4)
        # S1-2: singleton (no true match); its only candidate S3-1 scores .3
        scored_pairs = [
            ("S1-1", "S2-1", 0.9),
            ("S1-1", "S2-2", 0.4),
            ("S1-2", "S3-1", 0.3),
        ]
        true_by_s1 = {"S1-1": {"S2-1"}, "S1-2": set()}
        best_threshold, best_report, all_reports = sweep_threshold(
            scored_pairs, true_by_s1, all_s1_ids=["S1-1", "S1-2"],
            thresholds=[0.2, 0.35, 0.5, 0.8, 0.95],
        )
        # threshold 0.5 lets S2-1 through, excludes S2-2 and S3-1 -> perfect on both
        self.assertEqual(best_threshold, 0.5)
        self.assertEqual(best_report.macro_fbeta, 1.0)
        self.assertEqual(len(all_reports), 5)

    def test_predictions_at_threshold(self):
        grouped = {"S1-1": [("S2-1", 0.9), ("S2-2", 0.2)]}
        preds = predictions_at_threshold(grouped, 0.5)
        self.assertEqual(preds, {"S1-1": {"S2-1"}})

    def test_sweep_never_needs_test_labels(self):
        # sweep_threshold's signature only accepts validation true_by_s1 -- this test
        # documents (rather than "enforces" at runtime) that callers must pass
        # validation-split ground truth, never test-split, per the no-leakage rule.
        import inspect
        params = list(inspect.signature(sweep_threshold).parameters)
        self.assertEqual(params, ["scored_pairs", "true_by_s1", "all_s1_ids", "thresholds", "beta"])


if __name__ == "__main__":
    unittest.main()
