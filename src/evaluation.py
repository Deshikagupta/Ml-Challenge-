"""Entity-level macro F-beta evaluation -- the exact challenge metric.

For each S1 entity, predicted match IDs and true match IDs are compared as SETS
(never top-1). Precision/recall/F-beta are computed per entity, then macro-averaged
across every required S1 entity -- including ones with zero true matches, whose
correct prediction is an empty set.

The challenge metric is F0.5 (beta=0.5): precision-heavy, so a false positive on a
singleton entity (zero true matches) drives that entity's score straight to 0. This
module is intentionally generic over beta but every call site in this project uses
beta=0.5 to match the challenge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def precision_recall_fbeta(
    true_ids: Set[str], predicted_ids: Set[str], beta: float = 0.5
) -> Tuple[float, float, float]:
    """Precision/recall/F-beta for one S1 entity, comparing match ID sets.

    Conventions (deliberately explicit, since 0/0 cases are exactly what the
    "singletons must be included" requirement hinges on):
      - true_ids empty & predicted_ids empty  -> precision=1, recall=1, f=1 (correct no-match)
      - true_ids empty & predicted_ids non-empty -> recall=1 (vacuously, nothing to
        recall), precision < 1 for every false positive -> f < 1
      - true_ids non-empty & predicted_ids empty -> recall=0, precision=1
        (vacuously, nothing wrong was predicted) -> f=0 because recall=0
      - true_ids non-empty & predicted_ids non-empty -> ordinary precision/recall
    """
    true_positives = len(true_ids & predicted_ids)
    precision = (true_positives / len(predicted_ids)) if predicted_ids else 1.0
    recall = (true_positives / len(true_ids)) if true_ids else 1.0

    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    fbeta = ((1 + beta_sq) * precision * recall / denominator) if denominator > 0 else 0.0
    return precision, recall, fbeta


@dataclass
class MacroFBetaReport:
    macro_precision: float
    macro_recall: float
    macro_fbeta: float
    beta: float
    entity_count: int
    per_entity: Dict[str, Tuple[float, float, float]]


def macro_fbeta(
    true_by_s1: Mapping[str, Set[str]],
    predicted_by_s1: Mapping[str, Set[str]],
    all_s1_ids: Iterable[str],
    beta: float = 0.5,
    include_per_entity: bool = False,
) -> MacroFBetaReport:
    """Macro-average precision/recall/F-beta across every id in ``all_s1_ids``.

    ``all_s1_ids`` (typically every required S1 entity, e.g. from test_source1.tsv
    or the validation split) is the authority on which entities are scored --
    entities missing from either ``true_by_s1`` or ``predicted_by_s1`` default to
    the empty set, so a genuinely zero-match/zero-prediction entity is still
    included exactly as the challenge requires.
    """
    precisions: List[float] = []
    recalls: List[float] = []
    fbetas: List[float] = []
    per_entity: Dict[str, Tuple[float, float, float]] = {}

    for s1_id in all_s1_ids:
        true_ids = true_by_s1.get(s1_id, set())
        predicted_ids = predicted_by_s1.get(s1_id, set())
        precision, recall, f = precision_recall_fbeta(true_ids, predicted_ids, beta=beta)
        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(f)
        if include_per_entity:
            per_entity[s1_id] = (precision, recall, f)

    n = len(fbetas)
    return MacroFBetaReport(
        macro_precision=sum(precisions) / n if n else 0.0,
        macro_recall=sum(recalls) / n if n else 0.0,
        macro_fbeta=sum(fbetas) / n if n else 0.0,
        beta=beta,
        entity_count=n,
        per_entity=per_entity,
    )


# ======================================================================================
# Threshold sweeping
# ======================================================================================

def group_scores_by_entity(
    scored_pairs: Iterable[Tuple[str, str, float]]
) -> Dict[str, List[Tuple[str, float]]]:
    """(s1_id, candidate_id, score) triples -> {s1_id: [(candidate_id, score), ...]}."""
    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, candidate_id, score in scored_pairs:
        grouped.setdefault(s1_id, []).append((candidate_id, score))
    return grouped


def predictions_at_threshold(
    grouped_scores: Mapping[str, Sequence[Tuple[str, float]]], threshold: float
) -> Dict[str, Set[str]]:
    return {
        s1_id: {cid for cid, score in pairs if score >= threshold}
        for s1_id, pairs in grouped_scores.items()
    }


def sweep_threshold(
    scored_pairs: Iterable[Tuple[str, str, float]],
    true_by_s1: Mapping[str, Set[str]],
    all_s1_ids: Iterable[str],
    thresholds: Optional[Sequence[float]] = None,
    beta: float = 0.5,
) -> Tuple[float, MacroFBetaReport, List[Tuple[float, MacroFBetaReport]]]:
    """Sweep candidate thresholds against validation scores; pick the one that
    maximizes macro F-beta (the actual challenge metric -- never pairwise
    accuracy/F1, never top-1). Returns (best_threshold, best_report, all_reports).

    Ties are broken by the lower threshold (more conservative w.r.t. recall,
    deterministic).
    """
    if thresholds is None:
        thresholds = [round(t, 3) for t in _default_threshold_grid()]

    all_s1_ids = list(all_s1_ids)
    grouped = group_scores_by_entity(scored_pairs)

    results: List[Tuple[float, MacroFBetaReport]] = []
    best_threshold, best_report = None, None
    for threshold in thresholds:
        predicted = predictions_at_threshold(grouped, threshold)
        report = macro_fbeta(true_by_s1, predicted, all_s1_ids, beta=beta)
        results.append((threshold, report))
        if best_report is None or report.macro_fbeta > best_report.macro_fbeta:
            best_threshold, best_report = threshold, report

    return best_threshold, best_report, results


def _default_threshold_grid() -> List[float]:
    return [i / 100 for i in range(1, 100)]
