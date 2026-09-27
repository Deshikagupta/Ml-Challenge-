import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import Normalizer
from blocking import (
    BlockingConfig,
    BlockingIndex,
    CandidateGenerator,
    evaluate_blocking,
    generate_candidate_pairs,
    read_ground_truth,
    write_candidate_pairs,
)


def _normalize(records, schema=None):
    """Run real records through the real (frozen) Normalizer -- blocking tests
    always operate on genuine Normalizer output, never on hand-built fixtures
    that skip the Normalizer entirely."""
    normalizer = Normalizer().fit(records, schema=schema)
    return normalizer, normalizer.transform(records)


class BlockingIndexTests(unittest.TestCase):
    def setUp(self):
        self.s1_raw = [
            {"entity_id": "S1-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Zeta Holdings Limited",
             "business_address": "5 Lake Street, Rivertown 90210", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Orphan Ventures Inc",
             "business_address": "1 Nowhere Lane, Nowhereville 00000", "country": "US"},
        ]
        self.s2_raw = [
            {"entity_id": "S2-1", "business_name": "Acme Retail Corporation",
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Acme Retal Corp",  # typo, near-match for TF-IDF
             "business_address": "10 Main Road, Springfield 12345", "country": "US"},
            {"entity_id": "S2-3", "business_name": "Totally Unrelated Bakery",
             "business_address": "77 Elsewhere Ave, Metropolis 55555", "country": "France"},
        ]
        self.s3_raw = [
            {"entity_id": "S3-1", "business_name": "Zeta Holdings Limited",
             "business_address": "999 Different Blvd, Faraway 11111", "country": "US"},
            {"entity_id": "S3-2", "business_name": "Zeta Holdings Ltd",
             "business_address": "5 Lake Street, Rivertown 90210", "country": None},
        ]
        # Fit one normalizer across everything so vocab/abbreviations are shared,
        # exactly like the real pipeline would (fit on training, transform everywhere).
        fit_pool = self.s1_raw + self.s2_raw + self.s3_raw
        self.normalizer = Normalizer().fit(fit_pool)
        self.s1 = self.normalizer.transform(self.s1_raw)
        self.s2 = self.normalizer.transform(self.s2_raw)
        self.s3 = self.normalizer.transform(self.s3_raw)

        self.config = BlockingConfig(tfidf_min_similarity=0.2, tfidf_name_top_k=5)
        self.s2_index = BlockingIndex("S2", self.config).build(self.s2)
        self.s2_index.build_tfidf()
        self.s3_index = BlockingIndex("S3", self.config).build(self.s3)
        self.s3_index.build_tfidf()
        self.generator = CandidateGenerator(self.s2_index, self.s3_index, self.config)

        # A tiny 2-3 document TF-IDF corpus (as in these fixtures) doesn't have enough
        # documents for IDF to down-weight generic character n-grams, so even unrelated
        # short business names can show non-trivial char n-gram cosine similarity purely
        # by chance -- a small-corpus artifact that doesn't occur at real (large) scale,
        # where IDF properly discriminates. Tests that check "no relation at all" isolate
        # Blocks A/B/C/E with TF-IDF disabled; test_tfidf_recovers_near_text_match above
        # is what actually exercises Block D.
        self.config_no_tfidf = BlockingConfig(
            enabled_blocks=frozenset({"exact_name", "token", "address", "country_token"})
        )
        self.s2_index_no_tfidf = BlockingIndex("S2", self.config_no_tfidf).build(self.s2)
        self.s3_index_no_tfidf = BlockingIndex("S3", self.config_no_tfidf).build(self.s3)
        self.generator_no_tfidf = CandidateGenerator(
            self.s2_index_no_tfidf, self.s3_index_no_tfidf, self.config_no_tfidf
        )

    def _final_for(self, s1_record):
        blocks = self.generator.generate_batch([s1_record])[0]
        return CandidateGenerator.union_final(blocks)

    # 1. exact normalized-name match becomes a candidate
    def test_exact_name_match_is_candidate(self):
        acme = self.s1[0]
        candidates = self._final_for(acme)
        self.assertIn("S2-1", candidates)

    # 2. address-based match becomes a candidate
    def test_address_match_is_candidate(self):
        # S1-2 and S3-2 share exact street+postal but different-looking name; address
        # blocking (country+postal / country+address-token) should still surface it.
        zeta = self.s1[1]
        candidates = self._final_for(zeta)
        self.assertIn("S3-2", candidates)

    # 3. TF-IDF retrieval can recover a near-text match
    def test_tfidf_recovers_near_text_match(self):
        acme = self.s1[0]
        blocks = self.generator.generate_batch([acme])[0]
        # S2-2 ("Acme Retal Corp", a typo) should NOT match exactly but should be
        # recoverable via the TF-IDF name pass specifically.
        self.assertIn("S2-2", blocks["S2"]["tfidf_name"])

    # 4. country-aware blocking does not hard-code countries
    def test_country_blocking_is_data_driven(self):
        # No country value anywhere in source is a recognized ISO code; blocking
        # must still work purely from whatever strings appear in the data.
        raw = [{"entity_id": "S1-9", "business_name": "Wibble Traders",
                "business_address": "3 Odd Street, Weirdton 44444", "country": "Wobbleland"}]
        normalizer = Normalizer().fit(raw + [
            {"entity_id": "S2-9", "business_name": "Wibble Traders",
             "business_address": "3 Odd Street, Weirdton 44444", "country": "Wobbleland"},
        ])
        s1 = normalizer.transform(raw)
        s2 = normalizer.transform([{"entity_id": "S2-9", "business_name": "Wibble Traders",
                                     "business_address": "3 Odd Street, Weirdton 44444",
                                     "country": "Wobbleland"}])
        config = BlockingConfig()
        s2_index = BlockingIndex("S2", config).build(s2)
        s2_index.build_tfidf()
        s3_index = BlockingIndex("S3", config).build([])
        generator = CandidateGenerator(s2_index, s3_index, config)
        blocks = generator.generate_batch(s1)[0]
        final = CandidateGenerator.union_final(blocks)
        self.assertIn("S2-9", final)  # worked despite an entirely made-up country string

    # 5. S2 and S3 candidates remain valid (never cross-contaminated, never S1-S1)
    def test_s2_and_s3_never_mixed(self):
        acme = self.s1[0]
        blocks = self.generator.generate_batch([acme])[0]
        for source_label, source_blocks in blocks.items():
            for block_name, ids in source_blocks.items():
                for candidate_id in ids:
                    prefix = candidate_id.split("-")[0]  # "S2" or "S3"
                    self.assertEqual(prefix, source_label)
        # also: no S1 id ever appears as a candidate anywhere
        all_ids = set()
        for source_blocks in blocks.values():
            for ids in source_blocks.values():
                all_ids |= ids
        self.assertFalse(any(cid.startswith("S1-") for cid in all_ids))

    # 6. duplicate candidates are removed
    def test_duplicates_removed_in_union(self):
        acme = self.s1[0]
        blocks = self.generator.generate_batch([acme])[0]
        final = CandidateGenerator.union_final(blocks)
        self.assertEqual(len(final), len(set(final)))
        # S2-1 is found by exact_name AND token AND address AND tfidf_name -- still appears once
        self.assertEqual(sum(1 for x in final if x == "S2-1"), 1)

    # 7. missing fields do not crash blocking
    def test_missing_fields_do_not_crash(self):
        sparse_raw = [{"entity_id": "S1-8", "business_name": "", "business_address": "", "country": ""}]
        normalizer = Normalizer().fit(self.s1_raw + sparse_raw)
        sparse = normalizer.transform(sparse_raw)
        try:
            blocks = self.generator.generate_batch(sparse)[0]
            final = CandidateGenerator.union_final(blocks)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"blocking crashed on sparse/missing-field record: {exc}")
        self.assertIsInstance(final, set)

    # 8. an S1 with no candidates is still present in output
    def test_s1_with_no_candidates_present_in_output(self):
        orphan = self.s1[2]  # "Orphan Ventures Inc" shares nothing with S2/S3
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "candidate_pairs.tsv"
            generate_candidate_pairs(
                [orphan], self.s2_index_no_tfidf, self.s3_index_no_tfidf, self.config_no_tfidf, out_path
            )
            lines = out_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
        self.assertEqual(len(lines), 2)
        s1_id, candidates = lines[1].split("\t")
        self.assertEqual(s1_id, "S1-3")
        self.assertEqual(candidates, "")

    # 9. output format is exactly correct
    def test_output_format_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "candidate_pairs.tsv"
            generate_candidate_pairs(self.s1, self.s2_index, self.s3_index, self.config, out_path)
            text = out_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
        seen_ids = set()
        for line in lines[1:]:
            self.assertIn("\t", line)
            s1_id, candidates = line.split("\t")
            self.assertTrue(s1_id.startswith("S1-"))
            seen_ids.add(s1_id)
            if candidates:
                ids = candidates.split(",")
                self.assertEqual(len(ids), len(set(ids)))  # no duplicates within a row
                for cid in ids:
                    self.assertTrue(cid.startswith(("S2-", "S3-")))
        self.assertEqual(seen_ids, {"S1-1", "S1-2", "S1-3"})  # exactly one row per S1

    def test_write_candidate_pairs_helper_dedupes_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "candidate_pairs.tsv"
            write_candidate_pairs(out_path, [("S1-1", ["S2-2", "S2-1", "S2-1"])])
            text = out_path.read_text(encoding="utf-8")
        self.assertEqual(text, "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-2\n")

    # 10. training ground-truth evaluation correctly detects missed matches
    def test_evaluation_detects_missed_matches(self):
        ground_truth = {
            "S1-1": {"S2-1"},          # will be found
            "S1-2": {"S3-2"},          # will be found
            "S1-3": {"S2-3"},          # impossible: nothing links Orphan Ventures to S2-3
        }
        report = evaluate_blocking(
            self.s1, self.s2_index_no_tfidf, self.s3_index_no_tfidf, self.config_no_tfidf, ground_truth
        )
        self.assertEqual(report["ground_truth_matches"], 3)
        self.assertEqual(report["recovered_true_matches"], 2)
        self.assertAlmostEqual(report["candidate_recall"], 2 / 3)
        self.assertEqual(report["missed_matches_count"], 1)
        self.assertEqual(report["missed_matches_sample"][0], ("S1-3", "S2-3"))
        self.assertIn("exact_name", report["per_block_recall"])

    def test_block_can_be_disabled(self):
        config = BlockingConfig(enabled_blocks=frozenset({"exact_name"}))
        s2_index = BlockingIndex("S2", config).build(self.s2)
        s2_index.build_tfidf()  # no-op: tfidf_name/tfidf_address not enabled
        self.assertFalse(s2_index.tfidf_name.is_built)
        generator = CandidateGenerator(s2_index, self.s3_index, config)
        blocks = generator.generate_batch([self.s1[0]])[0]
        self.assertEqual(set(blocks["S2"].keys()), {"exact_name"})

    def test_posting_list_cap_bans_overly_common_token(self):
        config = BlockingConfig(token_max_posting_list=1, token_min_length=1)
        # two S2 records share the token "corp" (via "corporation" -> tokenized), forcing a cap breach
        raw = [
            {"entity_id": "S2-100", "business_name": "One Corporation",
             "business_address": "1 A St, Town 1", "country": "US"},
            {"entity_id": "S2-101", "business_name": "Two Corporation",
             "business_address": "2 B St, Town 2", "country": "US"},
        ]
        normalizer = Normalizer().fit(raw)
        s2 = normalizer.transform(raw)
        index = BlockingIndex("S2", config).build(s2)
        # "corporation" appears in both -> posting list length 2 > cap of 1 -> banned/dropped
        self.assertEqual(index.token_index.posting_length("corporation"), 0)

    def test_ground_truth_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.tsv"
            path.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-4\nS1-2\t\n",
                encoding="utf-8",
            )
            truth = read_ground_truth(path)
        self.assertEqual(truth["S1-1"], {"S2-1", "S3-4"})
        self.assertEqual(truth["S1-2"], set())


if __name__ == "__main__":
    unittest.main()
