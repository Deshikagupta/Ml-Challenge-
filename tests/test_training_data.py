import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import Normalizer
from blocking import BlockingConfig, BlockingIndex, generate_candidate_pairs
from features import RecordLookup, FEATURE_COLUMNS, ID_COLUMNS
from training_data import (
    LABEL_COLUMN,
    TRAINING_COLUMNS,
    build_training_table,
    filter_training_table_by_entities,
    read_training_table,
    split_s1_entities,
)


class LabelingTests(unittest.TestCase):
    def setUp(self):
        s1_raw = [
            {"entity_id": "S1-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Zeta Holdings Limited",
             "business_address": "5 Lake Street, Rivertown 90210", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Orphan Ventures Inc",
             "business_address": "1 Nowhere Lane, Nowhereville 00000", "country": "US"},
        ]
        s2_raw = [
            {"entity_id": "S2-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
        ]
        s3_raw = [
            {"entity_id": "S3-1", "business_name": "Zeta Holdings Limited",
             "business_address": "5 Lake Street, Rivertown 90210", "country": "US"},
        ]
        normalizer = Normalizer().fit(s1_raw + s2_raw + s3_raw)
        self.s1 = normalizer.transform(s1_raw)
        self.s2 = normalizer.transform(s2_raw)
        self.s3 = normalizer.transform(s3_raw)

        config = BlockingConfig(enabled_blocks=frozenset({"exact_name", "token", "address"}))
        self.s2_index = BlockingIndex("S2", config).build(self.s2)
        self.s3_index = BlockingIndex("S3", config).build(self.s3)

        self.tmp = tempfile.TemporaryDirectory()
        self.candidate_pairs_path = Path(self.tmp.name) / "candidate_pairs.tsv"
        generate_candidate_pairs(self.s1, self.s2_index, self.s3_index, config, self.candidate_pairs_path)

        self.s1_lookup = RecordLookup(self.s1)
        self.s2_lookup = RecordLookup(self.s2)
        self.s3_lookup = RecordLookup(self.s3)

        # ground truth: S1-1 matches S2-1 (S2-2 is a same-name decoy, must be a negative);
        # S1-2 matches S3-1; S1-3 has a "true match" that blocking will never find (missed)
        self.ground_truth = {
            "S1-1": {"S2-1"},
            "S1-2": {"S3-1"},
            "S1-3": {"S2-999"},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, **kwargs):
        out_path = Path(self.tmp.name) / "labeled.tsv"
        report = build_training_table(
            self.s1_lookup, self.s2_lookup, self.s3_lookup,
            self.candidate_pairs_path, self.ground_truth, out_path, **kwargs
        )
        return out_path, report

    def test_positive_and_negative_labels_assigned_correctly(self):
        out_path, report = self._build(max_negatives_per_s1=None)
        rows = list(read_training_table(out_path))
        by_candidate = {r["candidate_entity_id"]: r[LABEL_COLUMN] for r in rows}
        self.assertEqual(by_candidate["S2-1"], 1)   # true match, found by blocking
        self.assertEqual(by_candidate["S2-2"], 0)   # decoy, correctly negative
        self.assertEqual(by_candidate["S3-1"], 1)

    def test_missed_match_reported_not_silently_negative(self):
        out_path, report = self._build(max_negatives_per_s1=None)
        rows = list(read_training_table(out_path))
        candidate_ids = {r["candidate_entity_id"] for r in rows}
        self.assertNotIn("S2-999", candidate_ids)  # never existed as a candidate at all
        self.assertEqual(report.missed_matches_count, 1)
        self.assertEqual(report.missed_matches[0], ("S1-3", "S2-999"))

    def test_zero_candidate_entity_counted_but_writes_no_rows(self):
        out_path, report = self._build(max_negatives_per_s1=None)
        self.assertEqual(report.s1_entities, 3)
        self.assertEqual(report.s1_with_zero_candidates, 1)  # S1-3 has no real candidates

    def test_all_positives_kept_under_negative_subsampling(self):
        out_path, report = self._build(max_negatives_per_s1=0)
        rows = list(read_training_table(out_path))
        self.assertEqual(report.negatives, 0)
        self.assertEqual(report.positives, 2)  # both true positives always kept
        self.assertTrue(all(r[LABEL_COLUMN] == 1 for r in rows))

    def test_output_schema_matches_training_columns(self):
        out_path, _ = self._build()
        header = out_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        self.assertEqual(header, TRAINING_COLUMNS)
        self.assertEqual(header, list(ID_COLUMNS) + FEATURE_COLUMNS + [LABEL_COLUMN])

    def test_deterministic_given_seed(self):
        out_a, _ = self._build(max_negatives_per_s1=1, seed=7)
        out_b_path = Path(self.tmp.name) / "labeled_b.tsv"
        build_training_table(
            self.s1_lookup, self.s2_lookup, self.s3_lookup,
            self.candidate_pairs_path, self.ground_truth, out_b_path,
            max_negatives_per_s1=1, seed=7,
        )
        self.assertEqual(out_a.read_text(encoding="utf-8"), out_b_path.read_text(encoding="utf-8"))


class SplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_covers_all_entities(self):
        ids = [f"S1-{i}" for i in range(100)]
        train_ids, val_ids = split_s1_entities(ids, val_fraction=0.2, seed=1)
        self.assertEqual(train_ids | val_ids, set(ids))
        self.assertEqual(train_ids & val_ids, set())
        self.assertAlmostEqual(len(val_ids) / len(ids), 0.2, delta=0.02)

    def test_split_is_reproducible_given_same_seed(self):
        ids = [f"S1-{i}" for i in range(50)]
        a = split_s1_entities(ids, seed=99)
        b = split_s1_entities(ids, seed=99)
        self.assertEqual(a, b)

    def test_different_seeds_generally_differ(self):
        ids = [f"S1-{i}" for i in range(50)]
        a = split_s1_entities(ids, seed=1)
        b = split_s1_entities(ids, seed=2)
        self.assertNotEqual(a, b)

    def test_filter_by_entities_keeps_all_pairs_of_an_entity_together(self):
        rows = [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2"},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-1"},
        ]
        filtered = list(filter_training_table_by_entities(rows, {"S1-1"}))
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(r["source1_entity_id"] == "S1-1" for r in filtered))


if __name__ == "__main__":
    unittest.main()
