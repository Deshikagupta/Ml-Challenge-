import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matching_model import MatcherModel, TrainingMetadata, rows_to_labels, rows_to_matrix


def _synthetic_dataset(n=200, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, n_features).astype(np.float32)
    # a simple, learnable rule: positive iff mean of the first two features is high
    y = ((X[:, 0] + X[:, 1]) / 2 > 0.7).astype(np.int32)
    return X, y


class MatcherModelTests(unittest.TestCase):
    def test_train_and_predict_proba_shapes(self):
        X, y = _synthetic_dataset()
        model = MatcherModel.train(X, y, feature_columns=["a", "b", "c", "d"], random_state=1)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (X.shape[0],))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_model_actually_learns_the_synthetic_rule(self):
        X, y = _synthetic_dataset(n=500, seed=3)
        split = 400
        model = MatcherModel.train(X[:split], y[:split], feature_columns=["a", "b", "c", "d"])
        proba = model.predict_proba(X[split:])
        predicted = (proba >= 0.5).astype(int)
        accuracy = (predicted == y[split:]).mean()
        self.assertGreater(accuracy, 0.85)

    def test_reproducible_given_same_random_state(self):
        X, y = _synthetic_dataset()
        model_a = MatcherModel.train(X, y, feature_columns=["a", "b", "c", "d"], random_state=5)
        model_b = MatcherModel.train(X, y, feature_columns=["a", "b", "c", "d"], random_state=5)
        np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X))

    def test_save_and_load_round_trip_preserves_predictions(self):
        X, y = _synthetic_dataset()
        model = MatcherModel.train(X, y, feature_columns=["a", "b", "c", "d"], random_state=2)
        before = model.predict_proba(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            model.save(path)
            restored = MatcherModel.load(path)
        after = restored.predict_proba(X)
        np.testing.assert_allclose(before, after)
        self.assertEqual(restored.feature_columns, ["a", "b", "c", "d"])

    def test_logistic_regression_baseline_trains(self):
        X, y = _synthetic_dataset()
        model = MatcherModel.train(
            X, y, feature_columns=["a", "b", "c", "d"], model_kind="logistic_regression", random_state=1
        )
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (X.shape[0],))

    def test_unknown_model_kind_raises(self):
        X, y = _synthetic_dataset()
        with self.assertRaises(ValueError):
            MatcherModel.train(X, y, model_kind="not_a_real_model")

    def test_rows_to_matrix_uses_explicit_column_order(self):
        rows = [{"b": 2.0, "a": 1.0, "label": 1}, {"b": 4.0, "a": 3.0, "label": 0}]
        matrix = rows_to_matrix(rows, feature_columns=["a", "b"])
        np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        labels = rows_to_labels(rows)
        np.testing.assert_array_equal(labels, np.array([1, 0]))


class TrainingMetadataTests(unittest.TestCase):
    def test_round_trip(self):
        meta = TrainingMetadata(
            threshold=0.42, beta=0.5, validation_macro_fbeta=0.81,
            validation_macro_precision=0.9, validation_macro_recall=0.75,
            validation_entity_count=100, feature_columns=["a", "b"],
            model_kind="hist_gradient_boosting", random_state=42, seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            meta.save(path)
            restored = TrainingMetadata.load(path)
        self.assertEqual(restored, meta)


if __name__ == "__main__":
    unittest.main()
