"""Pairwise / candidate-pair feature generation.

Sits between blocking and the (not-yet-built) matching model:

    normalized records -> blocking -> candidate_pairs.tsv -> THIS MODULE ->
    pair_features.tsv -> (future) training-set construction / matching model

Every feature is computed from fields the Normalizer already produced
(``normalized_name``, ``normalized_address``, ``components``,
``numeric_features``, ``token_weights``, ``component_confidence``) plus which
source (S2 or S3) the candidate came from. Nothing here re-tokenizes,
re-normalizes, or re-derives anything -- and nothing here touches training
labels, so it is safe to run unchanged on both train and test candidate pairs
(label-joining is a separate, later step).

Public entry points:

    compute_pair_features(s1_record, candidate_record, source_label) -> dict
    iter_pair_features(s1_lookup, s2_lookup, s3_lookup, candidate_pairs_path) -> Iterator[dict]
    write_features_tsv(path, rows) -> None

and a small CLI (``features.py build ...``) that wires these together using
the existing normalized-TSV reader from blocking.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from normalizer import read_tsv_records  # reuse existing infra, do not reimplement
from blocking import read_normalized_tsv  # reuse existing JSON-column-aware normalized reader

# ======================================================================================
# Small, shared helpers -- pure functions, no I/O, so they're trivial to unit test
# ======================================================================================

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _norm_text(value: Any) -> str:
    """Casefold/strip only -- never re-tokenizes. Used for exact-match comparisons
    of values that are already normalized text."""
    if value is None:
        return ""
    return str(value).strip().casefold()


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = text.replace(" ", "")
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    union = a | b
    if not union:
        return 0.0  # no information either way -- not treated as a match signal
    return len(a & b) / len(union)


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = _levenshtein_distance(a, b)
    return 1 - distance / max(len(a), len(b))


def _token_weighted_jaccard(
    weights: Mapping[str, float], tokens_a: Set[str], tokens_b: Set[str]
) -> float:
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    numerator = sum(min(weights.get(t, 0.0), weights.get(t, 0.0)) for t in (tokens_a & tokens_b))
    # weights is a single record's token_weights dict; the pairwise weighted-Jaccard
    # below actually needs *both* records' weights, so this helper takes both maps.
    return numerator / max(sum(weights.get(t, 0.0) for t in union), 1e-9)


def _pairwise_weighted_jaccard(
    weights_a: Mapping[str, float], weights_b: Mapping[str, float],
    tokens_a: Set[str], tokens_b: Set[str],
) -> float:
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    shared = tokens_a & tokens_b
    numerator = sum(min(weights_a.get(t, 0.0), weights_b.get(t, 0.0)) for t in shared)
    denominator = sum(max(weights_a.get(t, 0.0), weights_b.get(t, 0.0)) for t in union)
    return numerator / denominator if denominator else 0.0


def _confidence_rank(level: Optional[str]) -> int:
    return _CONFIDENCE_RANK.get(level, -1)


def _component(record: Mapping[str, Any], key: str) -> Any:
    components = record.get("components")
    if not isinstance(components, dict):
        return None
    return components.get(key)


def _confidence(record: Mapping[str, Any], key: str) -> Optional[str]:
    conf = record.get("component_confidence")
    if not isinstance(conf, dict):
        return None
    return conf.get(key)


def _as_set(value: Any) -> Set[str]:
    """A component value may be a scalar or a list; always compare as a set
    of normalized strings so scalar and list-valued components share one helper."""
    if value is None:
        return set()
    if isinstance(value, list):
        return {_norm_text(v) for v in value if v not in (None, "")}
    text = _norm_text(value)
    return {text} if text else set()


def _scalar_match(record_a: Mapping[str, Any], record_b: Mapping[str, Any], key: str) -> Tuple[float, float]:
    """Exact-match feature for a single-valued component: (match, both_present)."""
    value_a, value_b = _component(record_a, key), _component(record_b, key)
    if isinstance(value_a, list):
        value_a = value_a[0] if value_a else None
    if isinstance(value_b, list):
        value_b = value_b[0] if value_b else None
    text_a, text_b = _norm_text(value_a), _norm_text(value_b)
    if not text_a or not text_b:
        return 0.0, 0.0
    return (1.0 if text_a == text_b else 0.0), 1.0


def _set_overlap(record_a: Mapping[str, Any], record_b: Mapping[str, Any], key: str) -> Tuple[float, float]:
    """Jaccard-overlap feature for a list-valued component: (jaccard, both_present)."""
    set_a, set_b = _as_set(_component(record_a, key)), _as_set(_component(record_b, key))
    if not set_a or not set_b:
        return 0.0, 0.0
    return _jaccard(set_a, set_b), 1.0


def _min_confidence(record_a: Mapping[str, Any], record_b: Mapping[str, Any], key: str) -> Tuple[float, float]:
    """(min_rank, available) -- the weaker of the two records' confidence for a
    component, so a match backed by two low-confidence extractions is trusted less
    than one backed by two high-confidence extractions."""
    rank_a, rank_b = _confidence_rank(_confidence(record_a, key)), _confidence_rank(_confidence(record_b, key))
    if rank_a < 0 or rank_b < 0:
        return -1.0, 0.0
    return float(min(rank_a, rank_b)), 1.0


# ======================================================================================
# The feature set
# ======================================================================================

ID_COLUMNS = ("source1_entity_id", "candidate_entity_id", "source")

_TEXT_FIELD_SUFFIXES = ("exact_match", "levenshtein_ratio", "char_ngram_jaccard", "token_jaccard", "token_weighted_jaccard")
_COMPONENT_SCALAR_KEYS = ("country", "city", "postal_code", "house_no_candidates")
_COMPONENT_SET_KEYS = ("street_indicators", "locality_candidates")

FEATURE_COLUMNS: List[str] = (
    [f"name_{suffix}" for suffix in _TEXT_FIELD_SUFFIXES]
    + [f"address_{suffix}" for suffix in _TEXT_FIELD_SUFFIXES]
    + [f"{key}_match" for key in _COMPONENT_SCALAR_KEYS]
    + [f"{key}_both_present" for key in _COMPONENT_SCALAR_KEYS]
    + [f"{key}_jaccard" for key in _COMPONENT_SET_KEYS]
    + [f"{key}_both_present" for key in _COMPONENT_SET_KEYS]
    + ["numbers_jaccard", "shares_any_number", "number_count_diff"]
    + [f"{key}_min_confidence" for key in _COMPONENT_SCALAR_KEYS + _COMPONENT_SET_KEYS]
    + [f"{key}_confidence_available" for key in _COMPONENT_SCALAR_KEYS + _COMPONENT_SET_KEYS]
    + ["source_is_s2"]
)


def _text_features(prefix: str, text_a: str, text_b: str, weights_a: Mapping[str, float], weights_b: Mapping[str, float]) -> Dict[str, float]:
    tokens_a, tokens_b = set(text_a.split()), set(text_b.split())
    return {
        f"{prefix}_exact_match": 1.0 if text_a and text_a == text_b else 0.0,
        f"{prefix}_levenshtein_ratio": _levenshtein_ratio(text_a, text_b),
        f"{prefix}_char_ngram_jaccard": _jaccard(_char_ngrams(text_a), _char_ngrams(text_b)),
        f"{prefix}_token_jaccard": _jaccard(tokens_a, tokens_b),
        f"{prefix}_token_weighted_jaccard": _pairwise_weighted_jaccard(weights_a, weights_b, tokens_a, tokens_b),
    }


def compute_pair_features(
    s1_record: Mapping[str, Any], candidate_record: Mapping[str, Any], source_label: str
) -> Dict[str, float]:
    """Compute the fixed feature vector for one (S1, candidate) pair.

    ``source_label`` is "S2" or "S3" -- the source the candidate record came from.
    Every value returned is a float (categoricals/booleans included) so the row is
    ready to hand to a tabular model without further encoding.
    """
    weights_a = s1_record.get("token_weights") or {}
    weights_b = candidate_record.get("token_weights") or {}

    features: Dict[str, float] = {}
    features.update(_text_features(
        "name", _norm_text(s1_record.get("normalized_name")), _norm_text(candidate_record.get("normalized_name")),
        weights_a, weights_b,
    ))
    features.update(_text_features(
        "address", _norm_text(s1_record.get("normalized_address")), _norm_text(candidate_record.get("normalized_address")),
        weights_a, weights_b,
    ))

    for key in _COMPONENT_SCALAR_KEYS:
        match, both_present = _scalar_match(s1_record, candidate_record, key)
        features[f"{key}_match"] = match
        features[f"{key}_both_present"] = both_present

    for key in _COMPONENT_SET_KEYS:
        jaccard, both_present = _set_overlap(s1_record, candidate_record, key)
        features[f"{key}_jaccard"] = jaccard
        features[f"{key}_both_present"] = both_present

    numbers_a = set((s1_record.get("numeric_features") or {}).get("numbers") or [])
    numbers_b = set((candidate_record.get("numeric_features") or {}).get("numbers") or [])
    count_a = (s1_record.get("numeric_features") or {}).get("number_count", 0) or 0
    count_b = (candidate_record.get("numeric_features") or {}).get("number_count", 0) or 0
    features["numbers_jaccard"] = _jaccard(numbers_a, numbers_b)
    features["shares_any_number"] = 1.0 if numbers_a & numbers_b else 0.0
    features["number_count_diff"] = float(abs(count_a - count_b))

    for key in _COMPONENT_SCALAR_KEYS + _COMPONENT_SET_KEYS:
        min_confidence, available = _min_confidence(s1_record, candidate_record, key)
        features[f"{key}_min_confidence"] = min_confidence
        features[f"{key}_confidence_available"] = available

    features["source_is_s2"] = 1.0 if source_label == "S2" else 0.0

    return features


# ======================================================================================
# ID-based record lookup -- deliberately a small, swappable interface.
#
# This in-memory implementation is correct and is what Step 1 needs to validate the
# pipeline end-to-end on the tested 10k/20k/20k sample. It is NOT what should run
# against the full 1.6-5 GB normalized files on a 16 GB machine -- that is exactly
# the large-scale performance pass planned as its own later step (a disk-backed or
# sort-merge join would replace RecordLookup's storage without changing any of the
# feature-computation code above it).
# ======================================================================================

class RecordLookup:
    """entity_id -> normalized record, built once per source."""

    def __init__(self, records: Iterable[Mapping[str, Any]], id_field: str = "entity_id"):
        self._by_id: Dict[str, Mapping[str, Any]] = {}
        for record in records:
            entity_id = record.get(id_field)
            if entity_id in (None, ""):
                continue
            self._by_id[str(entity_id)] = record

    def get(self, entity_id: str) -> Optional[Mapping[str, Any]]:
        return self._by_id.get(entity_id)

    def ids(self) -> List[str]:
        return list(self._by_id.keys())

    def __len__(self) -> int:
        return len(self._by_id)


# ======================================================================================
# candidate_pairs.tsv reading + streamed pair-feature generation
# ======================================================================================

def read_candidate_pairs(path: str | Path) -> Iterator[Tuple[str, List[str]]]:
    """Stream candidate_pairs.tsv: yields (source1_entity_id, [candidate_ids])."""
    for row in read_tsv_records(path):
        s1_id = row.get("source1_entity_id")
        if not s1_id:
            continue
        raw = row.get("candidate_entity_ids") or ""
        candidate_ids = [cid for cid in raw.split(",") if cid]
        yield s1_id, candidate_ids


def iter_pair_feature_groups(
    s1_lookup: RecordLookup,
    s2_lookup: RecordLookup,
    s3_lookup: RecordLookup,
    candidate_pairs_path: str | Path,
    on_missing: Optional[str] = "skip",
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Stream ``(source1_entity_id, [feature_row, ...])`` -- one entry per S1 row in
    candidate_pairs.tsv, **including S1 entities with an empty candidate list**
    (feature_row list will just be empty for those). This is the primitive both
    training-data labeling and prediction build on, specifically so that neither
    one can silently drop a zero-candidate S1 entity -- both matching_results.tsv
    and the labeled training table must account for every S1 row.

    Only the current S1 entity's own (typically small, blocking-bounded) candidate
    set is held in memory at a time -- never the whole candidate_pairs.tsv.

    Rows for candidate IDs that don't resolve to a record in the given lookups (or
    S1 IDs missing from ``s1_lookup``) are skipped with a warning to stderr rather
    than crashing the whole run; pass ``on_missing="raise"`` to fail loudly instead
    (useful in tests / small correctness runs).
    """
    for s1_id, candidate_ids in read_candidate_pairs(candidate_pairs_path):
        s1_record = s1_lookup.get(s1_id)
        if s1_record is None:
            message = f"features: S1 id {s1_id!r} not found in the S1 source; skipping its candidates"
            if on_missing == "raise":
                raise KeyError(message)
            print(message, file=sys.stderr)
            yield s1_id, []
            continue

        rows: List[Dict[str, Any]] = []
        for candidate_id in candidate_ids:
            if candidate_id.startswith("S2-"):
                source_label, lookup = "S2", s2_lookup
            elif candidate_id.startswith("S3-"):
                source_label, lookup = "S3", s3_lookup
            else:
                message = f"features: candidate id {candidate_id!r} has neither an S2- nor S3- prefix; skipping"
                if on_missing == "raise":
                    raise ValueError(message)
                print(message, file=sys.stderr)
                continue

            candidate_record = lookup.get(candidate_id)
            if candidate_record is None:
                message = f"features: candidate id {candidate_id!r} not found in source {source_label}; skipping"
                if on_missing == "raise":
                    raise KeyError(message)
                print(message, file=sys.stderr)
                continue

            row: Dict[str, Any] = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "source": source_label,
            }
            row.update(compute_pair_features(s1_record, candidate_record, source_label))
            rows.append(row)

        yield s1_id, rows


def iter_pair_features(
    s1_lookup: RecordLookup,
    s2_lookup: RecordLookup,
    s3_lookup: RecordLookup,
    candidate_pairs_path: str | Path,
    on_missing: Optional[str] = "skip",
) -> Iterator[Dict[str, Any]]:
    """Flat stream of feature rows (one per candidate pair) -- a thin wrapper over
    ``iter_pair_feature_groups`` for callers (e.g. bulk feature-table export) that
    don't need the per-S1 grouping. S1 entities with zero candidates simply
    contribute no rows here; use ``iter_pair_feature_groups`` directly wherever
    every S1 entity must be accounted for (labeling, prediction)."""
    for _s1_id, rows in iter_pair_feature_groups(
        s1_lookup, s2_lookup, s3_lookup, candidate_pairs_path, on_missing=on_missing
    ):
        yield from rows


def write_features_tsv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Stream feature rows straight to disk; never materializes the full table in
    memory. Returns the number of rows written. The header is the fixed
    ID_COLUMNS + FEATURE_COLUMNS list, not inferred from the first row, so every
    file this writes has an identical, predictable schema."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(ID_COLUMNS) + FEATURE_COLUMNS
    count = 0
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(_format_cell(row.get(col)) for col in columns) + "\n")
            count += 1
    return count


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


# ======================================================================================
# CLI
# ======================================================================================

def _cmd_build(args: argparse.Namespace) -> None:
    print(f"Loading S1 ({args.s1})...", file=sys.stderr)
    s1_lookup = RecordLookup(read_normalized_tsv(args.s1), id_field=args.id_field)
    print(f"Loading S2 ({args.s2})...", file=sys.stderr)
    s2_lookup = RecordLookup(read_normalized_tsv(args.s2), id_field=args.id_field)
    print(f"Loading S3 ({args.s3})...", file=sys.stderr)
    s3_lookup = RecordLookup(read_normalized_tsv(args.s3), id_field=args.id_field)
    print(f"S1={len(s1_lookup)} S2={len(s2_lookup)} S3={len(s3_lookup)} records loaded", file=sys.stderr)

    print(f"Generating pair features -> {args.output} ...", file=sys.stderr)
    rows = iter_pair_features(
        s1_lookup, s2_lookup, s3_lookup, args.candidate_pairs, on_missing=args.on_missing
    )
    count = write_features_tsv(args.output, rows)
    print(f"Wrote {count} feature rows to {args.output}", file=sys.stderr)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Generate pair_features.tsv from candidate_pairs.tsv")
    build_parser.add_argument("--s1", required=True, help="Normalized S1 TSV")
    build_parser.add_argument("--s2", required=True, help="Normalized S2 TSV")
    build_parser.add_argument("--s3", required=True, help="Normalized S3 TSV")
    build_parser.add_argument("--candidate-pairs", required=True, help="candidate_pairs.tsv from blocking.py")
    build_parser.add_argument("--output", required=True, help="Path to write pair_features.tsv")
    build_parser.add_argument("--id-field", default="entity_id")
    build_parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    build_parser.set_defaults(func=_cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
