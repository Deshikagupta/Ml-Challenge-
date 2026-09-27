"""Prediction CLI: normalized test sources + test candidate_pairs.tsv + trained
matcher -> matching_results.tsv.

Streaming/scale: works through ``features.iter_pair_feature_groups`` one S1 entity
at a time -- only the current entity's (blocking-bounded) candidate set and its
small feature matrix are ever in memory, never the full test candidate set. Every
required S1 entity is written exactly one row, including entities with zero
candidates or zero predictions above threshold (an empty ``matched_entity_ids``),
matching what ``utils/validate_submission.py`` requires.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from blocking import read_normalized_tsv
from features import FEATURE_COLUMNS, RecordLookup, iter_pair_feature_groups
from matching_model import MatcherModel, TrainingMetadata, rows_to_matrix


def predict_matches(
    s1_lookup: RecordLookup,
    s2_lookup: RecordLookup,
    s3_lookup: RecordLookup,
    candidate_pairs_path: str,
    model: MatcherModel,
    threshold: float,
    output_path: str,
    on_missing: str = "skip",
) -> Dict[str, int]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    s1_entities = 0
    s1_with_predictions = 0
    total_predicted = 0

    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, rows in iter_pair_feature_groups(
            s1_lookup, s2_lookup, s3_lookup, candidate_pairs_path, on_missing=on_missing
        ):
            s1_entities += 1
            matched_ids: List[str] = []
            if rows:
                X = rows_to_matrix(rows, feature_columns=model.feature_columns)
                scores = model.predict_proba(X)
                matched_ids = sorted(
                    row["candidate_entity_id"]
                    for row, score in zip(rows, scores)
                    if score >= threshold
                )
            if matched_ids:
                s1_with_predictions += 1
                total_predicted += len(matched_ids)
            stream.write(f"{s1_id}\t{','.join(matched_ids)}\n")

    return {
        "s1_entities": s1_entities,
        "s1_with_predictions": s1_with_predictions,
        "total_predicted_matches": total_predicted,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, help="Normalized TEST S1 TSV")
    parser.add_argument("--s2", required=True, help="Normalized TEST S2 TSV")
    parser.add_argument("--s3", required=True, help="Normalized TEST S3 TSV")
    parser.add_argument("--candidate-pairs", required=True, help="candidate_pairs.tsv built from TEST sources")
    parser.add_argument("--model", required=True, help="Path to matcher_model.pkl")
    parser.add_argument("--metadata", default=None, help="Path to matcher_metadata.json (for the threshold)")
    parser.add_argument("--threshold", type=float, default=None, help="Override the threshold from metadata")
    parser.add_argument("--output", required=True, help="Path to write matching_results.tsv")
    parser.add_argument("--id-field", default="entity_id")
    parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    args = parser.parse_args()

    if args.threshold is None and args.metadata is None:
        parser.error("provide --metadata (to read the selected threshold) or --threshold explicitly")

    threshold = args.threshold
    if threshold is None:
        threshold = TrainingMetadata.load(args.metadata).threshold
    print(f"Using threshold={threshold}", file=sys.stderr)

    print(f"Loading model ({args.model})...", file=sys.stderr)
    model = MatcherModel.load(args.model)

    print(f"Loading S1 ({args.s1})...", file=sys.stderr)
    s1_lookup = RecordLookup(read_normalized_tsv(args.s1), id_field=args.id_field)
    print(f"Loading S2 ({args.s2})...", file=sys.stderr)
    s2_lookup = RecordLookup(read_normalized_tsv(args.s2), id_field=args.id_field)
    print(f"Loading S3 ({args.s3})...", file=sys.stderr)
    s3_lookup = RecordLookup(read_normalized_tsv(args.s3), id_field=args.id_field)

    print(f"Predicting -> {args.output} ...", file=sys.stderr)
    stats = predict_matches(
        s1_lookup, s2_lookup, s3_lookup, args.candidate_pairs, model, threshold, args.output,
        on_missing=args.on_missing,
    )
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _main()
