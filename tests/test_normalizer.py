import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from normalizer import Normalizer, read_tsv_records


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for _ in range(3):
            self.records.append({
                "entity_id": "S1-1",
                "business_name": "Acme Retail Corporation",
                "business_address": "10 Road, Springfield 12345",
                "country": "US",
            })
            self.records.append({
                "entity_id": "S1-2",
                "business_name": "Acme Retail Corp",
                "business_address": "10 Road, Springfield 12345",
                "country": "US",
            })
            self.records.append({
                "entity_id": "S1-3",
                "business_name": "Acme Private Limited",
                "business_address": "10 Road, Springfield 12345",
                "country": "US",
            })

    def test_fit_derives_tables_and_transforms_non_destructively(self):
        normalizer = Normalizer().fit(self.records)
        result = normalizer.transform([{
            "entity_id": "S2-1",
            "business_name": "ACME Retail Corp.",
            "business_address": "10 Road, Springfield 12345",
            "country": "France",
        }])[0]

        self.assertEqual(normalizer.schema["name"], "business_name")
        self.assertEqual(result["business_name"], "ACME Retail Corp.")
        self.assertEqual(result["normalized_name"], "acme retail corporation")
        self.assertEqual(result["components"]["country"], "France")
        self.assertEqual(result["components"]["postal_code"], "12345")
        self.assertIn("road", normalizer.street_words)
        self.assertIn("limited", normalizer.legal_suffixes)
        self.assertEqual(result["numeric_features"]["number_count"], 2)

    def test_fit_transform_accepts_one_shot_iterables(self):
        result = Normalizer().fit_transform(iter(self.records))
        self.assertEqual(len(result), len(self.records))

    def test_iter_transform_streams_records(self):
        normalizer = Normalizer().fit(self.records)
        result = normalizer.iter_transform(iter(self.records))
        self.assertEqual(next(result)["entity_id"], "S1-1")

    def test_artifact_round_trip_and_incremental_fit(self):
        normalizer = Normalizer().fit(self.records)
        original_hash = normalizer.dataset_sha256
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "vocab_v1.json"
            normalizer.save(artifact)
            restored = Normalizer.load(artifact)
            self.assertEqual(restored.dataset_sha256, original_hash)
            self.assertEqual(restored.row_count, len(self.records))
            restored.partial_fit([{
                "entity_id": "S4-1",
                "business_name": "New Company LLC",
                "business_address": "44 Avenue, Lyon 69000",
                "country": "France",
            }])
            self.assertEqual(restored.row_count, len(self.records) + 1)
            self.assertNotEqual(restored.dataset_sha256, original_hash)
            self.assertEqual(restored.transform([{
                "entity_id": "S4-1",
                "business_name": "New Company LLC",
                "business_address": "44 Avenue, Lyon 69000",
                "country": "France",
            }])[0]["components"]["country"], "France")

    def test_validation_reports_schema_and_unknown_tokens(self):
        normalizer = Normalizer().fit(self.records)
        report = normalizer.validate(self.records)
        self.assertTrue(report["passed"])
        self.assertEqual(report["row_count"], len(self.records))

    def test_tsv_reader_handles_tab_separated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.tsv"
            path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme\t10 Main Road\tUS\n",
                encoding="utf-8",
            )
            rows = list(read_tsv_records(path))
        self.assertEqual(rows[0]["business_name"], "Acme")

    def test_tsv_reader_streams_zip_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "nested/records.tsv",
                    "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                    "S1-1\tAcme\t10 Main Road\tUS\n",
                )
            rows = list(read_tsv_records(path, "nested/records.tsv"))
        self.assertEqual(rows[0]["entity_id"], "S1-1")

    def test_ambiguous_columns_accept_explicit_mapping(self):
        rows = [
            {"first": "North Holdings", "second": "10 Main Road"},
            {"first": "West Partners", "second": "20 Lake Street"},
        ]
        normalizer = Normalizer().fit(rows, schema={"name": "first", "address": "second"})
        self.assertEqual(normalizer.schema["name"], "first")
        self.assertEqual(normalizer.transform(rows)[0]["raw_name"], "North Holdings")


if __name__ == "__main__":
    unittest.main()
