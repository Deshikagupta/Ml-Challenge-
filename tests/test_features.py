import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import Normalizer
from blocking import BlockingConfig, BlockingIndex, generate_candidate_pairs
from features import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    RecordLookup,
    compute_pair_features,
    iter_pair_feature_groups,
    iter_pair_features,
    read_candidate_pairs,
    write_features_tsv,
)


class FeatureComputationTests(unittest.TestCase):
    def setUp(self):
        raw_pool = [
            {"entity_id": "S1-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S2-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Totally Different Bakery",
             "business_address": "77 Elsewhere Ave, Metropolis 55555", "country": "France"},
            {"entity_id": "S3-1", "business_name": "Acme Retal Corp",
             "business_address": "", "country": ""},
        ]
        self.normalizer = Normalizer().fit(raw_pool)
        records = {r["entity_id"]: self.normalizer.transform([r])[0] for r in raw_pool}
        self.s1 = records["S1-1"]
        self.s2_match = records["S2-1"]
        self.s2_unrelated = records["S2-2"]
        self.s3_sparse = records["S3-1"]

    # exact match features
    def test_identical_records_score_high(self):
        feats = compute_pair_features(self.s1, self.s2_match, "S2")
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["name_levenshtein_ratio"], 1.0)
        self.assertEqual(feats["address_exact_match"], 1.0)
        self.assertEqual(feats["country_match"], 1.0)
        self.assertEqual(feats["country_both_present"], 1.0)
        self.assertEqual(feats["postal_code_match"], 1.0)
        self.assertEqual(feats["source_is_s2"], 1.0)

    def test_unrelated_records_score_low(self):
        feats = compute_pair_features(self.s1, self.s2_unrelated, "S2")
        self.assertEqual(feats["name_exact_match"], 0.0)
        self.assertLess(feats["name_token_jaccard"], 0.3)
        self.assertEqual(feats["country_match"], 0.0)  # US vs France, both present
        self.assertEqual(feats["country_both_present"], 1.0)

    # missing fields must not crash and must be flagged, not silently treated as mismatch
    def test_missing_fields_flagged_not_crashed(self):
        feats = compute_pair_features(self.s1, self.s3_sparse, "S3")
        self.assertEqual(feats["country_both_present"], 0.0)
        self.assertEqual(feats["country_match"], 0.0)
        self.assertEqual(feats["postal_code_both_present"], 0.0)
        self.assertEqual(feats["source_is_s2"], 0.0)  # S3 candidate
        # every declared column must be present and finite
        for col in FEATURE_COLUMNS:
            self.assertIn(col, feats)
            self.assertIsInstance(feats[col], float)

    def test_confidence_is_bottleneck_of_the_two_records(self):
        # S1's country confidence is "high" (schema-mapped column); S3-1's country is
        # missing entirely -> confidence_available must be 0, not silently high.
        feats = compute_pair_features(self.s1, self.s3_sparse, "S3")
        self.assertEqual(feats["country_confidence_available"], 0.0)
        self.assertEqual(feats["country_min_confidence"], -1.0)

    def test_deterministic(self):
        first = compute_pair_features(self.s1, self.s2_match, "S2")
        second = compute_pair_features(self.s1, self.s2_match, "S2")
        self.assertEqual(first, second)

    def test_feature_columns_cover_required_families(self):
        # sanity: every promised family appears at least once in the fixed schema
        joined = " ".join(FEATURE_COLUMNS)
        for token in (
            "name_exact_match", "name_levenshtein_ratio", "name_char_ngram_jaccard",
            "name_token_jaccard", "name_token_weighted_jaccard",
            "address_exact_match", "country_match", "postal_code_match",
            "house_no_candidates_match", "street_indicators_jaccard",
            "locality_candidates_jaccard", "numbers_jaccard", "shares_any_number",
            "number_count_diff", "country_min_confidence", "source_is_s2",
        ):
            self.assertIn(token, joined)


class EndToEndFeatureGenerationTests(unittest.TestCase):
    """Exercises features.py against real blocking output (not hand-built fixtures)."""

    def setUp(self):
        s1_raw = [
            {"entity_id": "S1-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Orphan Ventures Inc",
             "business_address": "1 Nowhere Lane, Nowhereville 00000", "country": "US"},
        ]
        s2_raw = [
            {"entity_id": "S2-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
        ]
        s3_raw = [
            {"entity_id": "S3-1", "business_name": "Someone Else Entirely",
             "business_address": "9 Random Way, Nowhereplace 99999", "country": "India"},
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

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_candidate_pairs_roundtrip(self):
        pairs = dict(read_candidate_pairs(self.candidate_pairs_path))
        self.assertIn("S1-1", pairs)
        self.assertIn("S1-2", pairs)
        self.assertIn("S2-1", pairs["S1-1"])

    def test_grouped_iterator_includes_zero_candidate_s1(self):
        groups = dict(iter_pair_feature_groups(
            self.s1_lookup, self.s2_lookup, self.s3_lookup, self.candidate_pairs_path
        ))
        self.assertIn("S1-2", groups)  # Orphan Ventures: no real candidates
        self.assertEqual(groups["S1-2"], [])
        self.assertGreater(len(groups["S1-1"]), 0)
        self.assertEqual(groups["S1-1"][0]["candidate_entity_id"], "S2-1")

    def test_missing_candidate_id_is_skipped_not_fatal(self):
        # doctor the candidate_pairs.tsv to reference a nonexistent candidate id
        bad_path = Path(self.tmp.name) / "bad_candidate_pairs.tsv"
        bad_path.write_text(
            "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-999,S2-1\n", encoding="utf-8"
        )
        groups = dict(iter_pair_feature_groups(
            self.s1_lookup, self.s2_lookup, self.s3_lookup, bad_path, on_missing="skip"
        ))
        ids = [row["candidate_entity_id"] for row in groups["S1-1"]]
        self.assertEqual(ids, ["S2-1"])  # S2-999 silently skipped, not crashed

    def test_missing_candidate_id_raises_when_requested(self):
        bad_path = Path(self.tmp.name) / "bad_candidate_pairs.tsv"
        bad_path.write_text(
            "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-999\n", encoding="utf-8"
        )
        with self.assertRaises(KeyError):
            list(iter_pair_feature_groups(
                self.s1_lookup, self.s2_lookup, self.s3_lookup, bad_path, on_missing="raise"
            ))

    def test_write_features_tsv_has_fixed_schema_and_streams_correctly(self):
        rows = iter_pair_features(self.s1_lookup, self.s2_lookup, self.s3_lookup, self.candidate_pairs_path)
        out_path = Path(self.tmp.name) / "pair_features.tsv"
        count = write_features_tsv(out_path, rows)
        self.assertEqual(count, 1)  # only S1-1/S2-1 survives blocking in this tiny fixture
        header = out_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        self.assertEqual(header, list(ID_COLUMNS) + FEATURE_COLUMNS)

    def test_record_lookup_basic(self):
        lookup = RecordLookup(self.s2, id_field="entity_id")
        self.assertEqual(len(lookup), 1)
        self.assertIsNotNone(lookup.get("S2-1"))
        self.assertIsNone(lookup.get("S2-999"))


if __name__ == "__main__":
    unittest.main()
