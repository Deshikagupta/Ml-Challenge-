"""The pairwise matching model.

A HistGradientBoostingClassifier is the default: it is already available through
scikit-learn (no new dependency), handles tabular features with mixed scales and
missing/sentinel values (our -1 "confidence not available" sentinel, in particular)
without manual imputation, and is a practical, well-understood choice for
pairwise/entity-resolution feature sets at the row counts this pipeline produces
after blocking + negative subsampling. A LogisticRegression baseline is included
for a fast, interpretable sanity check (are the features even linearly informative
before trusting the boosted model's threshold sweep).

Class imbalance (true matches are rare relative to blocked negatives even after
subsampling) is handled via ``class_weight="balanced"``, not by further undersampling
inside the model -- that undersampling already happened, deliberately, in
training_data.py so both the model and this module have a single, inspectable place
each responsible for one concern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from features import FEATURE_COLUMNS

ARTIFACT_VERSION = 1


@dataclass
class MatcherModel:
    """Wraps a fitted sklearn classifier plus the exact feature-column order it was
    trained on, so scoring can never silently misalign columns between training and
    prediction (a common, hard-to-notice source of leakage/bugs in these pipelines)."""

    model: Any
    feature_columns: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    model_kind: str = "hist_gradient_boosting"
    random_state: int = 42

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        model_kind: str = "hist_gradient_boosting",
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> "MatcherModel":
        if model_kind == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            defaults = dict(
                max_depth=6,
                learning_rate=0.1,
                max_iter=200,
                class_weight="balanced",
                random_state=random_state,
            )
            defaults.update(model_kwargs)
            model = HistGradientBoostingClassifier(**defaults)
        elif model_kind == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            defaults = dict(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
            defaults.update(model_kwargs)
            model = LogisticRegression(**defaults)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind!r}")

        model.fit(X, y)
        return cls(model=model, feature_columns=list(feature_columns), model_kind=model_kind, random_state=random_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive (match) class, shape (n_rows,)."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        import pickle

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ARTIFACT_VERSION,
            "model_kind": self.model_kind,
            "feature_columns": self.feature_columns,
            "random_state": self.random_state,
            "model": self.model,
        }
        with target.open("wb") as stream:
            pickle.dump(payload, stream)

    @classmethod
    def load(cls, path: str | Path) -> "MatcherModel":
        import pickle

        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported matcher artifact version: {payload.get('version')!r}")
        return cls(
            model=payload["model"],
            feature_columns=payload["feature_columns"],
            model_kind=payload["model_kind"],
            random_state=payload["random_state"],
        )


def rows_to_matrix(rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    """Labeled/feature rows (as dicts, e.g. from training_data.read_training_table)
    -> a float32 matrix in a fixed, explicit column order."""
    return np.asarray(
        [[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float32
    )


def rows_to_labels(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int32)


@dataclass
class TrainingMetadata:
    """Everything downstream prediction needs besides the model weights themselves:
    the chosen decision threshold and the validation report it was chosen from."""

    threshold: float
    beta: float
    validation_macro_fbeta: float
    validation_macro_precision: float
    validation_macro_recall: float
    validation_entity_count: int
    feature_columns: List[str]
    model_kind: str
    random_state: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingMetadata":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
