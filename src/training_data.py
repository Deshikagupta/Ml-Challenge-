"""Training-set construction: candidate_pairs.tsv + train_ground_truth.tsv -> labeled
pairs, plus the reproducible entity-level train/val split.

Label = 1 iff the pair is in ground truth AND blocking actually produced it as a
candidate. A true match ground truth misses because blocking didn't surface it is
NOT a negative example -- it is a blocking-recall failure, and is reported
separately (``missed_matches``) rather than silently folded into the negatives,
per the explicit no-silent-negatives requirement.

Streaming/scale: every S1 entity's candidate set (already bounded by blocking) is
the unit of work; nothing beyond one entity's rows is held in memory at a time
while writing the labeled table to disk. Because blocking's candidate volume can
still be large per entity (observed avg ~1355/entity on the 10k/20k/20k sample --
itself worth revisiting in blocking tuning separately), negatives are optionally
subsampled per S1 entity (``max_negatives_per_s1``) so the labeled training table
stays a bounded, trainable size regardless of blocking's raw candidate volume; all
positives are always kept.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from features import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    RecordLookup,
    iter_pair_feature_groups,
)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)

LABEL_COLUMN = "label"
TRAINING_COLUMNS: List[str] = list(ID_COLUMNS) + FEATURE_COLUMNS + [LABEL_COLUMN]


class TrainingDataReport:
    def __init__(self) -> None:
        self.s1_entities = 0
        self.total_pairs_written = 0
        self.positives = 0
        self.negatives = 0
        self.s1_with_zero_candidates = 0
        self.missed_matches: List[Tuple[str, str]] = []  # (s1_id, missed_match_id), capped sample
        self.missed_matches_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "s1_entities": self.s1_entities,
            "total_pairs_written": self.total_pairs_written,
            "positives": self.positives,
            "negatives": self.negatives,
            "s1_with_zero_candidates": self.s1_with_zero_candidates,
            "missed_matches_count": self.missed_matches_count,
            "missed_matches_sample": self.missed_matches[:50],
        }


def build_training_table(
    s1_lookup: RecordLookup,
    s2_lookup: RecordLookup,
    s3_lookup: RecordLookup,
    candidate_pairs_path: str | Path,
    ground_truth: Mapping[str, Set[str]],
    output_path: str | Path,
    max_negatives_per_s1: Optional[int] = 50,
    seed: int = 42,
    on_missing: str = "skip",
) -> TrainingDataReport:
    """Stream-label every (S1, candidate) pair blocking produced for the training
    sources, writing the labeled table to ``output_path`` one S1 entity at a time.

    ``max_negatives_per_s1``: cap on negative (label=0) rows kept per S1 entity,
    randomly subsampled with ``seed`` for reproducibility; ``None`` keeps every
    negative (only safe on small samples -- see module docstring). All positives
    are always kept regardless of this cap, so recall of true matches into the
    training table is unaffected by subsampling.
    """
    report = TrainingDataReport()
    rng = random.Random(seed)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(TRAINING_COLUMNS) + "\n")

        for s1_id, rows in iter_pair_feature_groups(
            s1_lookup, s2_lookup, s3_lookup, candidate_pairs_path, on_missing=on_missing
        ):
            report.s1_entities += 1
            true_matches = ground_truth.get(s1_id, set())
            candidate_ids = {row["candidate_entity_id"] for row in rows}

            missed = true_matches - candidate_ids
            if missed:
                report.missed_matches_count += len(missed)
                if len(report.missed_matches) < 50:
                    report.missed_matches.extend((s1_id, mid) for mid in missed)

            if not rows:
                report.s1_with_zero_candidates += 1
                continue

            positive_rows = [r for r in rows if r["candidate_entity_id"] in true_matches]
            negative_rows = [r for r in rows if r["candidate_entity_id"] not in true_matches]
            if max_negatives_per_s1 is not None and len(negative_rows) > max_negatives_per_s1:
                negative_rows = rng.sample(negative_rows, max_negatives_per_s1)

            for row in positive_rows:
                _write_row(stream, row, label=1)
                report.positives += 1
                report.total_pairs_written += 1
            for row in negative_rows:
                _write_row(stream, row, label=0)
                report.negatives += 1
                report.total_pairs_written += 1

    return report


def _write_row(stream, row: Mapping[str, Any], label: int) -> None:
    stream.write(
        "\t".join(_format_cell(row.get(col)) for col in list(ID_COLUMNS) + FEATURE_COLUMNS)
        + "\t" + str(label) + "\n"
    )


def read_training_table(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Stream the labeled table back, decoding feature columns to float and the
    label to int. Used by matching_model.py; kept separate from build so it can
    also be reused for chunked reading later without touching the writer."""
    with open(path, encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        for line in stream:
            values = line.rstrip("\n").split("\t")
            row: Dict[str, Any] = dict(zip(header, values))
            for col in FEATURE_COLUMNS:
                row[col] = float(row[col]) if row[col] != "" else 0.0
            row[LABEL_COLUMN] = int(row[LABEL_COLUMN])
            yield row


# ======================================================================================
# Entity-level train/val split -- whole S1 entities move together, never individual pairs
# ======================================================================================

def split_s1_entities(
    s1_ids: Sequence[str], val_fraction: float = 0.2, seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """Reproducibly partition S1 entity ids into (train_ids, val_ids). All candidate
    pairs for a given S1 entity always land in the same split -- splitting by pair
    instead would leak the same entity's other candidates across train/val."""
    ids = sorted(set(s1_ids))  # sort first so shuffling is deterministic regardless of input order
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(round(len(ids) * (1 - val_fraction)))
    return set(ids[:cut]), set(ids[cut:])


def filter_training_table_by_entities(
    rows: Iterable[Mapping[str, Any]], allowed_s1_ids: Set[str]
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        if row["source1_entity_id"] in allowed_s1_ids:
            yield row


def _format_report(report: TrainingDataReport) -> str:
    lines = [
        "Training Data Construction",
        "---------------------------",
        f"S1 entities: {report.s1_entities}",
        f"S1 entities with zero candidates: {report.s1_with_zero_candidates}",
        f"Labeled pairs written: {report.total_pairs_written} "
        f"(positives: {report.positives}, negatives: {report.negatives})",
        f"True matches missed by blocking (not in any candidate set): {report.missed_matches_count}",
    ]
    if report.missed_matches:
        lines.append("  sample: " + ", ".join(f"{s1}->{mid}" for s1, mid in report.missed_matches[:10]))
    return "\n".join(lines)


# ======================================================================================
# CLI
# ======================================================================================

def _main() -> None:
    import argparse
    import json

    from blocking import read_ground_truth
    from blocking import read_normalized_tsv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, help="Normalized train S1 TSV")
    parser.add_argument("--s2", required=True, help="Normalized train S2 TSV")
    parser.add_argument("--s3", required=True, help="Normalized train S3 TSV")
    parser.add_argument("--candidate-pairs", required=True, help="candidate_pairs.tsv built from the TRAIN sources")
    parser.add_argument("--ground-truth", required=True, help="train_ground_truth.tsv")
    parser.add_argument("--output", required=True, help="Path to write the labeled training table")
    parser.add_argument("--max-negatives-per-s1", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-field", default="entity_id")
    parser.add_argument("--report-output", default=None, help="Optional path to also write the report as JSON")
    args = parser.parse_args()

    print(f"Loading S1 ({args.s1})...", file=sys.stderr)
    s1_lookup = RecordLookup(read_normalized_tsv(args.s1), id_field=args.id_field)
    print(f"Loading S2 ({args.s2})...", file=sys.stderr)
    s2_lookup = RecordLookup(read_normalized_tsv(args.s2), id_field=args.id_field)
    print(f"Loading S3 ({args.s3})...", file=sys.stderr)
    s3_lookup = RecordLookup(read_normalized_tsv(args.s3), id_field=args.id_field)
    print(f"Loading ground truth ({args.ground_truth})...", file=sys.stderr)
    ground_truth = read_ground_truth(args.ground_truth)

    max_neg = args.max_negatives_per_s1 if args.max_negatives_per_s1 >= 0 else None
    report = build_training_table(
        s1_lookup, s2_lookup, s3_lookup, args.candidate_pairs, ground_truth, args.output,
        max_negatives_per_s1=max_neg, seed=args.seed,
    )
    print(_format_report(report))
    if args.report_output:
        Path(args.report_output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
