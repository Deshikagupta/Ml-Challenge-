#!/usr/bin/env python3
import argparse
import os
import sys

DELIM = "\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def examples(items):
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def load_match_targets(test_dir, warnings):
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            warnings.append(f"{path} not found - skipping ID-existence check.")
            return None
        targets |= read_ids(path)
    return targets


def validate_id_list_file(path, expected_header, col_label, required, valid_ids, errors):
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping = {}
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    n_rows = empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(f"{name}: header looks COMMA-separated.")
            return None
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(f"{name}: unexpected header {cols}. Expected exactly {expected_header}.")
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue
            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}."),
        (intra_dupes, "{name}: repeated ID inside a {col} list for: {ex}."),
        (self_matches, "{name}: {col} contains Source-1 IDs (self-matches): {ex}."),
        (wrong_prefix, "{name}: {col} contains IDs without an S2-/S3- prefix: {ex}."),
        (unknown, "{name}: {col} references IDs not in the test Source-2/3 files: {ex}."),
        (required - seen, "{name}: required S1 entity(ies) missing: {ex}."),
        (seen - required, "{name}: row(s) using an S1 ID that is not in the test set: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))

    print(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping


def validate(matching_path, candidate_path, test_dir, check_ids=False):
    errors, warnings = [], []
    source1 = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1):
        errors.append(f"Test source1 file not found: {source1} (check --test-dir).")
        return errors, warnings
    required = read_ids(source1)
    print(f"  required S1 entities: {len(required)}")

    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
    else:
        valid_ids = None
        warnings.append("ID-existence check is OFF (the default).")

    matched = validate_id_list_file(matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors)

    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = validate_id_list_file(candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids, errors)
    elif candidate_path:
        warnings.append(f"{candidate_path} not found - skipping candidate_pairs.tsv checks.")

    if matched is not None and candidate is not None:
        offenders = {s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())}
        if offenders:
            warnings.append(f"{len(offenders)} S1 entity(ies) have matched IDs not present in candidate_pairs.tsv.")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv")
    parser.add_argument("--candidate", "-c", default=None)
    parser.add_argument("--test-dir", "-t", default="dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()
    candidate_path = args.candidate or "output/candidate_pairs.tsv"
    print("ML Challenge 2026 - submission validator")
    print(f"  test dir: {args.test_dir}")
    errors, warnings = validate(args.matching, candidate_path, args.test_dir, check_ids=args.check_ids)
    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL - {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1
    print("PASS - no blocking issues found. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
