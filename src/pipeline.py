"""Single entry point for the whole logical pipeline, as thin subcommands over the
existing per-stage CLIs (blocking.py, training_data.py, train_matcher.py,
prediction.py, utils/validate_submission.py). This module adds no new pipeline
logic of its own -- it only sequences the other modules' public functions so the
whole thing can be run (or scripted) as one command, without forcing the full
dataset into RAM any more than the stage it's calling already does.

Subcommands:
    pipeline.py train    normalized train sources + train ground truth -> model + threshold
    pipeline.py predict  normalized test sources + trained model -> matching_results.tsv
    pipeline.py validate matching_results.tsv (+ candidate_pairs.tsv) -> pass/fail

Blocking itself (``blocking.py build``) is intentionally NOT wrapped here: it is
run once per source pair (train, test) and its own CLI already exposes every
tunable needed for the large-scale run; re-exposing it here would just be a
second, driftable copy of the same argument list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> None:
    from train_matcher import train

    summary = train(
        args.s1, args.s2, args.s3, args.candidate_pairs, args.ground_truth, args.output_dir,
        val_fraction=args.val_fraction, seed=args.seed, max_negatives_per_s1=args.max_negatives_per_s1,
        model_kind=args.model_kind, beta=args.beta, id_field=args.id_field,
    )
    print(json.dumps(summary, indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    from blocking import read_normalized_tsv
    from features import RecordLookup
    from matching_model import MatcherModel, TrainingMetadata
    from prediction import predict_matches

    threshold = args.threshold
    if threshold is None:
        threshold = TrainingMetadata.load(args.metadata).threshold

    model = MatcherModel.load(args.model)
    s1_lookup = RecordLookup(read_normalized_tsv(args.s1), id_field=args.id_field)
    s2_lookup = RecordLookup(read_normalized_tsv(args.s2), id_field=args.id_field)
    s3_lookup = RecordLookup(read_normalized_tsv(args.s3), id_field=args.id_field)

    stats = predict_matches(
        s1_lookup, s2_lookup, s3_lookup, args.candidate_pairs, model, threshold, args.output,
        on_missing=args.on_missing,
    )
    print(json.dumps(stats, indent=2))


def _cmd_validate(args: argparse.Namespace) -> None:
    validator = Path(__file__).resolve().parent / "utils" / "validate_submission.py"
    if not validator.exists():
        raise SystemExit(f"utils/validate_submission.py not found at {validator}")
    command = [
        sys.executable, str(validator),
        "--matching", args.matching,
        "--test-dir", args.test_dir,
    ]
    if args.candidate:
        command += ["--candidate", args.candidate]
    if args.check_ids:
        command.append("--check-ids")
    result = subprocess.run(command)
    sys.exit(result.returncode)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the matcher and select a threshold")
    train_parser.add_argument("--s1", required=True)
    train_parser.add_argument("--s2", required=True)
    train_parser.add_argument("--s3", required=True)
    train_parser.add_argument("--candidate-pairs", required=True)
    train_parser.add_argument("--ground-truth", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-negatives-per-s1", type=int, default=50)
    train_parser.add_argument("--model-kind", choices=["hist_gradient_boosting", "logistic_regression"], default="hist_gradient_boosting")
    train_parser.add_argument("--beta", type=float, default=0.5)
    train_parser.add_argument("--id-field", default="entity_id")
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Score test candidates and write matching_results.tsv")
    predict_parser.add_argument("--s1", required=True)
    predict_parser.add_argument("--s2", required=True)
    predict_parser.add_argument("--s3", required=True)
    predict_parser.add_argument("--candidate-pairs", required=True)
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--metadata", default=None)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--id-field", default="entity_id")
    predict_parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    predict_parser.set_defaults(func=_cmd_predict)

    validate_parser = subparsers.add_parser("validate", help="Run utils/validate_submission.py")
    validate_parser.add_argument("--matching", required=True)
    validate_parser.add_argument("--candidate", default=None)
    validate_parser.add_argument("--test-dir", required=True)
    validate_parser.add_argument("--check-ids", action="store_true")
    validate_parser.set_defaults(func=_cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
