"""Strict representation-ranking metrics with invalid endpoint policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PairRankingRecord:
    query_id: str
    gallery_id: str
    label: bool
    visual_score: float | None
    valid: bool = True
    query_valid: bool = True
    gallery_valid: bool = True

    def __post_init__(self) -> None:
        for field_name in ("label", "valid", "query_valid", "gallery_valid"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool.")
        if self.valid:
            if (
                isinstance(self.visual_score, bool)
                or not isinstance(self.visual_score, (int, float))
                or not math.isfinite(float(self.visual_score))
                or not 0.0 <= float(self.visual_score) <= 1.0
            ):
                raise ValueError("A valid ranking pair requires finite visual_score in [0, 1].")
        elif self.visual_score is not None:
            raise ValueError("An invalid ranking pair must have visual_score=None.")


def evaluate_pair_ranking(
    records: tuple[PairRankingRecord, ...],
    *,
    k_values: tuple[int, ...] = (1, 3),
    candidate_validity: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate AP/AUROC/retrieval with strict invalid rank buckets.

    Invalid negatives are ranked above every valid score; invalid positives are
    ranked below every valid score. The rank key exists only inside evaluator
    diagnostics and is never a replacement for saved ``visual_score``.
    """

    if not records:
        return {"status": "not_supported", "reason": "empty_pair_set"}
    positives = sum(1 for item in records if item.label)
    negatives = len(records) - positives
    groups = _rank_groups(records)
    ap = None if positives == 0 else _average_precision(groups)
    auroc = None if positives == 0 or negatives == 0 else _auroc(records)
    retrieval = _bidirectional_retrieval(records, k_values=k_values)
    valid_records = tuple(item for item in records if _pair_is_valid(item))
    valid_positive = sum(1 for item in valid_records if item.label)
    valid_negative = len(valid_records) - valid_positive
    validity_by_candidate = _candidate_validity(records, candidate_validity)
    valid_retrieval = _bidirectional_retrieval(valid_records, k_values=k_values)
    for direction in ("forward", "reverse"):
        primary_count = retrieval[direction]["query_count"]
        valid_count = valid_retrieval[direction]["query_count"]
        valid_retrieval[direction]["eligible_query_count_before_filter"] = primary_count
        valid_retrieval[direction]["query_coverage"] = _safe_ratio(
            valid_count,
            primary_count,
        )
    primary_query_count = (
        retrieval["forward"]["query_count"] + retrieval["reverse"]["query_count"]
    )
    valid_query_count = (
        valid_retrieval["forward"]["query_count"]
        + valid_retrieval["reverse"]["query_count"]
    )
    valid_retrieval["aggregate"]["eligible_query_count_before_filter"] = primary_query_count
    valid_retrieval["aggregate"]["query_coverage"] = _safe_ratio(
        valid_query_count,
        primary_query_count,
    )
    valid_retrieval["eligible_query_count_before_filter"] = primary_query_count
    valid_retrieval["query_coverage"] = valid_retrieval["aggregate"]["query_coverage"]
    return {
        "status": "supported",
        "pair_count": len(records),
        "positive_count": positives,
        "negative_count": negatives,
        "invalid_pair_count": len(records) - len(valid_records),
        "average_precision": ap,
        "auroc": auroc,
        "retrieval": retrieval,
        "valid_only": {
            "candidate_count": len(validity_by_candidate),
            "valid_candidate_count": sum(validity_by_candidate.values()),
            "candidate_coverage": _safe_ratio(
                sum(validity_by_candidate.values()),
                len(validity_by_candidate),
            ),
            "pair_count": len(valid_records),
            "pair_coverage": _safe_ratio(len(valid_records), len(records)),
            "average_precision": None if valid_positive == 0 else _average_precision(_rank_groups(valid_records)),
            "auroc": None if valid_positive == 0 or valid_negative == 0 else _auroc(valid_records),
            "retrieval": valid_retrieval,
        },
    }


def _candidate_validity(
    records: tuple[PairRankingRecord, ...],
    supplied: Mapping[str, bool] | None,
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    if supplied is not None:
        for candidate_id, validity in supplied.items():
            if not isinstance(candidate_id, str) or not candidate_id:
                raise TypeError("candidate_validity keys must be non-empty strings.")
            if not isinstance(validity, bool):
                raise TypeError("candidate_validity values must be bool.")
            result[candidate_id] = validity
    for record in records:
        for candidate_id, validity in (
            (record.query_id, record.query_valid),
            (record.gallery_id, record.gallery_valid),
        ):
            previous = result.setdefault(candidate_id, validity)
            if previous is not validity:
                raise ValueError("Candidate validity must be consistent across ranking records.")
    return result


def _bidirectional_retrieval(
    records: tuple[PairRankingRecord, ...],
    *,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    forward = _retrieval(records, k_values=k_values)
    reverse_records = tuple(
        PairRankingRecord(
            query_id=item.gallery_id,
            gallery_id=item.query_id,
            label=item.label,
            visual_score=item.visual_score,
            valid=item.valid,
            query_valid=item.gallery_valid,
            gallery_valid=item.query_valid,
        )
        for item in records
    )
    reverse = _retrieval(reverse_records, k_values=k_values)
    aggregate = _aggregate_retrieval(forward, reverse, k_values=k_values)
    return {
        "forward": forward,
        "reverse": reverse,
        "aggregate": aggregate,
        **aggregate,
    }


def _rank_bucket(record: PairRankingRecord) -> int:
    if _pair_is_valid(record):
        return 1
    return 0 if record.label else 2


def _rank_group_key(record: PairRankingRecord) -> tuple[int, float]:
    score = (
        record.visual_score
        if _pair_is_valid(record) and record.visual_score is not None
        else 0.0
    )
    return (-_rank_bucket(record), -score)


def _pair_is_valid(record: PairRankingRecord) -> bool:
    return record.valid and record.query_valid and record.gallery_valid


def _rank_groups(records: tuple[PairRankingRecord, ...] | list[PairRankingRecord]) -> tuple[tuple[PairRankingRecord, ...], ...]:
    by_key: dict[tuple[int, float], list[PairRankingRecord]] = {}
    for record in records:
        by_key.setdefault(_rank_group_key(record), []).append(record)
    return tuple(tuple(by_key[key]) for key in sorted(by_key))


def _average_precision(groups: tuple[tuple[PairRankingRecord, ...], ...]) -> float:
    hits = 0
    total = 0.0
    positives = sum(1 for group in groups for item in group if item.label)
    rank_before = 0
    for group in groups:
        group_size = len(group)
        group_positives = sum(1 for item in group if item.label)
        if group_positives:
            total += _expected_ap_contribution(
                hits_before=hits,
                rank_before=rank_before,
                group_size=group_size,
                group_positives=group_positives,
            )
            hits += group_positives
        rank_before += group_size
    return total / positives


def _expected_ap_contribution(
    *,
    hits_before: int,
    rank_before: int,
    group_size: int,
    group_positives: int,
) -> float:
    """Expected AP numerator contribution under uniform tie permutations."""

    if group_size == 1:
        return (hits_before + 1) / (rank_before + 1)
    contribution = 0.0
    positive_probability = group_positives / group_size
    for offset in range(1, group_size + 1):
        expected_positive_before = (offset - 1) * (group_positives - 1) / (group_size - 1)
        contribution += positive_probability * (
            hits_before + expected_positive_before + 1
        ) / (rank_before + offset)
    return contribution


def _auroc(records: tuple[PairRankingRecord, ...]) -> float:
    positives = [item for item in records if item.label]
    negatives = [item for item in records if not item.label]
    total = 0.0
    for positive in positives:
        for negative in negatives:
            pos_key = (_rank_bucket(positive), positive.visual_score if _pair_is_valid(positive) and positive.visual_score is not None else 0.0)
            neg_key = (_rank_bucket(negative), negative.visual_score if _pair_is_valid(negative) and negative.visual_score is not None else 0.0)
            if pos_key[0] > neg_key[0]:
                total += 1.0
            elif pos_key[0] < neg_key[0]:
                total += 0.0
            elif pos_key[1] > neg_key[1]:
                total += 1.0
            elif pos_key[1] == neg_key[1]:
                total += 0.5
    return total / (len(positives) * len(negatives))


def _retrieval(records: tuple[PairRankingRecord, ...], *, k_values: tuple[int, ...]) -> dict[str, Any]:
    by_query: dict[str, list[PairRankingRecord]] = {}
    for record in records:
        by_query.setdefault(record.query_id, []).append(record)
    query_rows = []
    for query_id in sorted(by_query):
        values = by_query[query_id]
        if not any(item.label for item in values):
            continue
        query_validity = {item.query_valid for item in values}
        if len(query_validity) != 1:
            raise ValueError("All ranking rows for one query must agree on query_valid.")
        if not next(iter(query_validity)):
            query_rows.append({
                "query_id": query_id,
                "query_valid": False,
                "ap": 0.0,
                "rr": 0.0,
                **{f"recall@{k}": 0.0 for k in k_values},
            })
            continue
        groups = _rank_groups(values)
        average_first_rank = _average_first_positive_rank(groups)
        row = {
            "query_id": query_id,
            "query_valid": True,
            "ap": _average_precision(groups),
            "rr": 0.0 if average_first_rank is None else 1.0 / average_first_rank,
        }
        for k in k_values:
            row[f"recall@{k}"] = 1.0 if average_first_rank is not None and average_first_rank <= min(k, len(values)) else 0.0
        query_rows.append(row)
    if not query_rows:
        return {
            "query_count": 0,
            "invalid_query_count": 0,
            "map": None,
            "mrr": None,
            **{f"recall@{k}": None for k in k_values},
            "per_query": [],
        }
    return {
        "query_count": len(query_rows),
        "invalid_query_count": sum(not row["query_valid"] for row in query_rows),
        "map": sum(row["ap"] for row in query_rows) / len(query_rows),
        "mrr": sum(row["rr"] for row in query_rows) / len(query_rows),
        **{
            f"recall@{k}": sum(row[f"recall@{k}"] for row in query_rows) / len(query_rows)
            for k in k_values
        },
        "per_query": query_rows,
    }


def _aggregate_retrieval(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    *,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aggregation": "unweighted_mean_of_forward_and_reverse",
        "query_count": forward["query_count"] + reverse["query_count"],
        "invalid_query_count": (
            forward["invalid_query_count"] + reverse["invalid_query_count"]
        ),
    }
    for key in ("map", "mrr", *(f"recall@{k}" for k in k_values)):
        values = [value for value in (forward[key], reverse[key]) if value is not None]
        result[key] = None if not values else sum(values) / len(values)
    return result


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _average_first_positive_rank(groups: tuple[tuple[PairRankingRecord, ...], ...]) -> float | None:
    rank_before = 0
    for group in groups:
        group_size = len(group)
        group_positives = sum(1 for item in group if item.label)
        if group_positives:
            expected_min_offset = (group_size + 1) / (group_positives + 1)
            return rank_before + expected_min_offset
        rank_before += group_size
    return None
