import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import Normalizer
from blocking import BlockingConfig, BlockingIndex, generate_candidate_pairs
from features import RecordLookup
from matching_model import MatcherModel
from prediction import predict_matches
from train_matcher import train as train_matcher


def _write_tsv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")


class EndToEndPipelineTests(unittest.TestCase):
    """Builds a small but real dataset (repeated/varied so there's enough signal for
    the model to learn something and enough entities for a meaningful entity-level
    split), and runs the actual train_matcher.train() + prediction.predict_matches()
    functions -- not mocks -- then checks the output format and behavior."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        s1_raw, s2_raw, s3_raw, ground_truth_rows = [], [], [], []
        names = ["Acme Retail", "Zeta Holdings", "Blue Ocean Traders", "Pine Grove Bakery", "Silver Line Logistics"]
        addrs = ["10 Main Road, Springfield 12345", "5 Lake Street, Rivertown 90210",
                 "77 Elsewhere Ave, Metropolis 55555", "3 Harbor Way, Portville 11223"]
        for i in range(20):
            name = names[i % len(names)]
            addr = addrs[i % len(addrs)]
            s1_raw.append({"entity_id": f"S1-{i}", "business_name": name, "business_address": addr, "country": "US"})
            if i % 4 != 0:  # most S1 entities have a genuine S2 match; some are singletons
                s2_raw.append({"entity_id": f"S2-{i}", "business_name": name, "business_address": addr, "country": "US"})
                ground_truth_rows.append({"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"})
            else:
                ground_truth_rows.append({"source1_entity_id": f"S1-{i}", "matched_entity_ids": ""})
            # decoys sharing a name but a different address, to give the model negatives to learn from
            s3_raw.append({"entity_id": f"S3-{i}", "business_name": name,
                            "business_address": addrs[(i + 1) % len(addrs)], "country": "US"})

        self.normalizer = Normalizer().fit(s1_raw + s2_raw + s3_raw)
        self.s1 = self.normalizer.transform(s1_raw)
        self.s2 = self.normalizer.transform(s2_raw)
        self.s3 = self.normalizer.transform(s3_raw)

        self.normalized_dir = self.root / "normalized"
        self.normalized_dir.mkdir()
        from normalizer import write_tsv_records
        write_tsv_records(self.normalized_dir / "s1.tsv", self.s1)
        write_tsv_records(self.normalized_dir / "s2.tsv", self.s2)
        write_tsv_records(self.normalized_dir / "s3.tsv", self.s3)

        _write_tsv(self.root / "ground_truth.tsv", ground_truth_rows,
                   ["source1_entity_id", "matched_entity_ids"])

        config = BlockingConfig(enabled_blocks=frozenset({"exact_name", "token", "address"}))
        s2_index = BlockingIndex("S2", config).build(self.s2)
        s3_index = BlockingIndex("S3", config).build(self.s3)
        self.candidate_pairs_path = self.root / "candidate_pairs.tsv"
        generate_candidate_pairs(self.s1, s2_index, s3_index, config, self.candidate_pairs_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_then_predict_then_validate_format(self):
        output_dir = self.root / "train_output"
        summary = train_matcher(
            str(self.normalized_dir / "s1.tsv"),
            str(self.normalized_dir / "s2.tsv"),
            str(self.normalized_dir / "s3.tsv"),
            str(self.candidate_pairs_path),
            str(self.root / "ground_truth.tsv"),
            str(output_dir),
            val_fraction=0.3,
            seed=1,
            max_negatives_per_s1=10,
        )
        self.assertIn("metadata", summary)
        self.assertGreaterEqual(summary["metadata"]["validation_macro_fbeta"], 0.0)
        self.assertTrue((output_dir / "matcher_model.pkl").exists())
        self.assertTrue((output_dir / "matcher_metadata.json").exists())

        model = MatcherModel.load(output_dir / "matcher_model.pkl")
        s1_lookup = RecordLookup(self.s1)
        s2_lookup = RecordLookup(self.s2)
        s3_lookup = RecordLookup(self.s3)

        results_path = self.root / "matching_results.tsv"
        stats = predict_matches(
            s1_lookup, s2_lookup, s3_lookup, str(self.candidate_pairs_path),
            model, threshold=summary["metadata"]["threshold"], output_path=str(results_path),
        )
        self.assertEqual(stats["s1_entities"], 20)

        lines = results_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
        self.assertEqual(len(lines), 21)  # header + 20 S1 entities, all present
        seen_ids = set()
        for line in lines[1:]:
            s1_id, matched = line.split("\t")
            seen_ids.add(s1_id)
            if matched:
                for mid in matched.split(","):
                    self.assertTrue(mid.startswith(("S2-", "S3-")))
        self.assertEqual(seen_ids, {f"S1-{i}" for i in range(20)})

    def test_validator_accepts_generated_output(self):
        # run the actual pipeline, then invoke utils/validate_submission.py as a subprocess
        # to check real compatibility, not just format assumptions.
        output_dir = self.root / "train_output"
        summary = train_matcher(
            str(self.normalized_dir / "s1.tsv"), str(self.normalized_dir / "s2.tsv"),
            str(self.normalized_dir / "s3.tsv"), str(self.candidate_pairs_path),
            str(self.root / "ground_truth.tsv"), str(output_dir),
            val_fraction=0.3, seed=1, max_negatives_per_s1=10,
        )
        model = MatcherModel.load(output_dir / "matcher_model.pkl")
        results_path = self.root / "matching_results.tsv"
        predict_matches(
            RecordLookup(self.s1), RecordLookup(self.s2), RecordLookup(self.s3),
            str(self.candidate_pairs_path), model, summary["metadata"]["threshold"], str(results_path),
        )

        test_dir = self.root / "dataset_test"
        test_dir.mkdir()
        _write_tsv(test_dir / "test_source1.tsv", [{"entity_id": r["entity_id"]} for r in [{"entity_id": f"S1-{i}"} for i in range(20)]], ["entity_id"])

        validator = Path(__file__).resolve().parents[1] / "utils" / "validate_submission.py"
        if not validator.exists():
            self.skipTest("utils/validate_submission.py not present in this checkout")
        result = subprocess.run(
            [sys.executable, str(validator),
             "--matching", str(results_path),
             "--candidate", str(self.candidate_pairs_path),
             "--test-dir", str(test_dir)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
