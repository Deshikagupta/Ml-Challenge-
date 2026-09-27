"""Training CLI: normalized train sources + train candidate_pairs.tsv + train_ground_truth.tsv
-> a trained matcher artifact + selected threshold.

Pipeline (matches the roadmap's Step B/C/D/E):
    1. read normalized train S1/S2/S3
    2. read train candidate_pairs.tsv (built separately via `blocking.py build` on the
       TRAIN sources -- this CLI does not re-run blocking itself, so blocking's own
       indexes/config are reused rather than rebuilt)
    3. label every candidate pair against train_ground_truth.tsv (training_data.py),
       with negative subsampling for scale
    4. split by S1 ENTITY (never by pair) into train/val
    5. train the matcher on the train split
    6. score the val split, sweep thresholds against the exact macro-F0.5 metric
    7. save the model + threshold/metadata artifacts

No test data, and no test labels, are read anywhere in this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

from blocking import read_ground_truth, read_normalized_tsv
from evaluation import macro_fbeta, sweep_threshold
from features import RecordLookup
from matching_model import MatcherModel, TrainingMetadata, rows_to_labels, rows_to_matrix
from training_data import (
    build_training_table,
    filter_training_table_by_entities,
    read_training_table,
    split_s1_entities,
)


def _score_pairs(model: MatcherModel, rows: List[Dict]) -> List[tuple]:
    if not rows:
        return []
    X = rows_to_matrix(rows, feature_columns=model.feature_columns)
    scores = model.predict_proba(X)
    return [
        (row["source1_entity_id"], row["candidate_entity_id"], float(score))
        for row, score in zip(rows, scores)
    ]


def train(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    candidate_pairs_path: str,
    ground_truth_path: str,
    output_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_negatives_per_s1: int = 50,
    model_kind: str = "hist_gradient_boosting",
    beta: float = 0.5,
    id_field: str = "entity_id",
) -> Dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading S1 ({s1_path})...", file=sys.stderr)
    s1_lookup = RecordLookup(read_normalized_tsv(s1_path), id_field=id_field)
    print(f"Loading S2 ({s2_path})...", file=sys.stderr)
    s2_lookup = RecordLookup(read_normalized_tsv(s2_path), id_field=id_field)
    print(f"Loading S3 ({s3_path})...", file=sys.stderr)
    s3_lookup = RecordLookup(read_normalized_tsv(s3_path), id_field=id_field)
    print(f"Loading ground truth ({ground_truth_path})...", file=sys.stderr)
    ground_truth: Dict[str, Set[str]] = read_ground_truth(ground_truth_path)

    labeled_path = output / "labeled_pairs.tsv"
    print(f"Labeling candidate pairs -> {labeled_path} ...", file=sys.stderr)
    report = build_training_table(
        s1_lookup, s2_lookup, s3_lookup, candidate_pairs_path, ground_truth, labeled_path,
        max_negatives_per_s1=max_negatives_per_s1, seed=seed,
    )
    print(
        f"  {report.total_pairs_written} labeled rows "
        f"({report.positives} positive, {report.negatives} negative); "
        f"{report.missed_matches_count} true matches missed by blocking (see report.json)",
        file=sys.stderr,
    )

    all_s1_ids = s1_lookup.ids()
    train_ids, val_ids = split_s1_entities(all_s1_ids, val_fraction=val_fraction, seed=seed)
    print(f"Entity-level split: {len(train_ids)} train / {len(val_ids)} val", file=sys.stderr)

    all_rows = list(read_training_table(labeled_path))
    train_rows = list(filter_training_table_by_entities(all_rows, train_ids))
    val_rows = list(filter_training_table_by_entities(all_rows, val_ids))
    print(f"  {len(train_rows)} training rows, {len(val_rows)} validation rows", file=sys.stderr)

    from features import FEATURE_COLUMNS

    X_train = rows_to_matrix(train_rows, FEATURE_COLUMNS)
    y_train = rows_to_labels(train_rows)
    print(f"Training {model_kind}...", file=sys.stderr)
    model = MatcherModel.train(X_train, y_train, feature_columns=FEATURE_COLUMNS, model_kind=model_kind, random_state=seed)

    print("Scoring validation split and sweeping threshold...", file=sys.stderr)
    scored_val = _score_pairs(model, val_rows)
    true_by_s1_val = {s1_id: ids for s1_id, ids in ground_truth.items() if s1_id in val_ids}
    best_threshold, best_report, _ = sweep_threshold(scored_val, true_by_s1_val, all_s1_ids=val_ids, beta=beta)
    print(
        f"  best threshold={best_threshold:.3f} "
        f"macro F{beta}={best_report.macro_fbeta:.4f} "
        f"(precision={best_report.macro_precision:.4f}, recall={best_report.macro_recall:.4f})",
        file=sys.stderr,
    )

    model_path = output / "matcher_model.pkl"
    model.save(model_path)
    metadata = TrainingMetadata(
        threshold=best_threshold,
        beta=beta,
        validation_macro_fbeta=best_report.macro_fbeta,
        validation_macro_precision=best_report.macro_precision,
        validation_macro_recall=best_report.macro_recall,
        validation_entity_count=best_report.entity_count,
        feature_columns=list(FEATURE_COLUMNS),
        model_kind=model_kind,
        random_state=seed,
        seed=seed,
    )
    metadata_path = output / "matcher_metadata.json"
    metadata.save(metadata_path)
    print(f"Saved model -> {model_path}", file=sys.stderr)
    print(f"Saved metadata -> {metadata_path}", file=sys.stderr)

    return {
        "labeling_report": report.to_dict(),
        "train_entities": len(train_ids),
        "val_entities": len(val_ids),
        "metadata": metadata.to_dict(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, help="Normalized TRAIN S1 TSV")
    parser.add_argument("--s2", required=True, help="Normalized TRAIN S2 TSV")
    parser.add_argument("--s3", required=True, help="Normalized TRAIN S3 TSV")
    parser.add_argument("--candidate-pairs", required=True, help="candidate_pairs.tsv built from TRAIN sources")
    parser.add_argument("--ground-truth", required=True, help="train_ground_truth.tsv")
    parser.add_argument("--output-dir", required=True, help="Directory for labeled_pairs.tsv, model, metadata")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-negatives-per-s1", type=int, default=50)
    parser.add_argument("--model-kind", choices=["hist_gradient_boosting", "logistic_regression"], default="hist_gradient_boosting")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--id-field", default="entity_id")
    args = parser.parse_args()

    import json

    summary = train(
        args.s1, args.s2, args.s3, args.candidate_pairs, args.ground_truth, args.output_dir,
        val_fraction=args.val_fraction, seed=args.seed, max_negatives_per_s1=args.max_negatives_per_s1,
        model_kind=args.model_kind, beta=args.beta, id_field=args.id_field,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
