"""Prediction-only resolution of masks that aggregate several candidates.

The resolver uses only candidate geometry.  It does not inspect annotations,
evaluation metrics, model scores, or object identities.  Inputs are local masks
positioned in one frame coordinate system; the returned candidate IDs let a
caller apply the decision without this module mutating pipeline state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


DEFAULT_MASK_CONTAINMENT = 0.50
DEFAULT_MINIMUM_COVERED_MASKS = 3


class AggregateResolutionReason(str, Enum):
    """Stable reason codes for one candidate resolution decision."""

    REMOVED_AGGREGATE_MASK = "removed_aggregate_mask"
    KEPT_BELOW_MINIMUM = "kept_below_minimum_covered_masks"
    KEPT_MISSING_MASK = "kept_missing_mask"
    KEPT_EMPTY_MASK = "kept_empty_mask"


@dataclass(frozen=True, slots=True, eq=False)
class PositionedCandidateMask:
    """One candidate mask positioned in a shared frame coordinate system.

    ``mask`` is a local Boolean crop whose top-left pixel is at
    ``(origin_x, origin_y)``.  ``None`` represents a candidate for which no mask
    is available; absence is never evidence for removal.
    """

    candidate_id: str
    frame_id: str
    mask: np.ndarray | None
    origin_x: int = 0
    origin_y: int = 0

    def __post_init__(self) -> None:
        _require_token(self.candidate_id, "candidate_id")
        _require_token(self.frame_id, "frame_id")
        _require_nonnegative_integer(self.origin_x, "origin_x")
        _require_nonnegative_integer(self.origin_y, "origin_y")
        if self.mask is None:
            return
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("mask must be a NumPy ndarray or None.")
        if self.mask.ndim != 2:
            raise ValueError("mask must have shape (height, width).")
        if self.mask.shape[0] <= 0 or self.mask.shape[1] <= 0:
            raise ValueError("mask height and width must be positive.")
        if self.mask.dtype != np.dtype(np.bool_):
            raise TypeError("mask dtype must be bool.")
        frozen = np.ascontiguousarray(self.mask.astype(np.bool_, copy=True))
        frozen.flags["WRITEABLE"] = False
        object.__setattr__(self, "mask", frozen)


@dataclass(frozen=True, slots=True)
class CoveredMaskEvidence:
    """Directed evidence that one candidate covers another candidate mask."""

    candidate_id: str
    intersection_pixels: int
    candidate_mask_area_pixels: int
    containment: float


@dataclass(frozen=True, slots=True)
class CandidateAggregateDecision:
    """Resolution decision and its complete positive containment evidence."""

    candidate_id: str
    reason: AggregateResolutionReason
    mask_area_pixels: int | None
    covered_masks: tuple[CoveredMaskEvidence, ...]

    @property
    def removed(self) -> bool:
        return self.reason is AggregateResolutionReason.REMOVED_AGGREGATE_MASK

    @property
    def covered_mask_count(self) -> int:
        return len(self.covered_masks)


@dataclass(frozen=True, slots=True)
class AggregateResolutionResult:
    """Deterministic, non-mutating resolution result for one frame."""

    frame_id: str | None
    mask_containment_at_least: float
    minimum_covered_masks: int
    decisions: tuple[CandidateAggregateDecision, ...]

    @property
    def removed_candidate_ids(self) -> tuple[str, ...]:
        return tuple(decision.candidate_id for decision in self.decisions if decision.removed)

    @property
    def kept_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.candidate_id for decision in self.decisions if not decision.removed
        )

    @property
    def skipped_candidate_ids(self) -> tuple[str, ...]:
        skipped_reasons = {
            AggregateResolutionReason.KEPT_MISSING_MASK,
            AggregateResolutionReason.KEPT_EMPTY_MASK,
        }
        return tuple(
            decision.candidate_id
            for decision in self.decisions
            if decision.reason in skipped_reasons
        )

    def decision_for(self, candidate_id: str) -> CandidateAggregateDecision:
        for decision in self.decisions:
            if decision.candidate_id == candidate_id:
                return decision
        raise KeyError(candidate_id)


def resolve_aggregate_masks(
    candidates: Sequence[PositionedCandidateMask],
    *,
    mask_containment_at_least: float = DEFAULT_MASK_CONTAINMENT,
    minimum_covered_masks: int = DEFAULT_MINIMUM_COVERED_MASKS,
) -> AggregateResolutionResult:
    """Identify aggregate masks without mutating or consulting ground truth.

    For a directed relation ``A -> B``, containment is
    ``|A intersection B| / |B|``.  Candidate ``A`` is removed when it covers
    at least ``minimum_covered_masks`` other non-empty masks.  All decisions
    are calculated from the original input simultaneously, so a nested
    aggregate can be removed together with its parent while ordinary child
    masks remain available.
    """

    threshold = _containment_threshold(mask_containment_at_least)
    minimum = _positive_integer(minimum_covered_masks, "minimum_covered_masks")
    ordered = _validated_candidates(candidates)
    frame_id = None if not ordered else ordered[0].frame_id

    usable = tuple(
        candidate
        for candidate in ordered
        if candidate.mask is not None and bool(np.any(candidate.mask))
    )
    mask_areas = {
        candidate.candidate_id: int(np.count_nonzero(candidate.mask))
        for candidate in usable
    }

    decisions: list[CandidateAggregateDecision] = []
    for candidate in ordered:
        if candidate.mask is None:
            decisions.append(
                CandidateAggregateDecision(
                    candidate_id=candidate.candidate_id,
                    reason=AggregateResolutionReason.KEPT_MISSING_MASK,
                    mask_area_pixels=None,
                    covered_masks=(),
                )
            )
            continue

        area = int(np.count_nonzero(candidate.mask))
        if area == 0:
            decisions.append(
                CandidateAggregateDecision(
                    candidate_id=candidate.candidate_id,
                    reason=AggregateResolutionReason.KEPT_EMPTY_MASK,
                    mask_area_pixels=0,
                    covered_masks=(),
                )
            )
            continue

        covered: list[CoveredMaskEvidence] = []
        for other in usable:
            if other.candidate_id == candidate.candidate_id:
                continue
            intersection = _intersection_pixels(candidate, other)
            other_area = mask_areas[other.candidate_id]
            containment = intersection / other_area
            if containment >= threshold:
                covered.append(
                    CoveredMaskEvidence(
                        candidate_id=other.candidate_id,
                        intersection_pixels=intersection,
                        candidate_mask_area_pixels=other_area,
                        containment=containment,
                    )
                )

        covered.sort(key=lambda evidence: evidence.candidate_id.encode("utf-8"))
        reason = (
            AggregateResolutionReason.REMOVED_AGGREGATE_MASK
            if len(covered) >= minimum
            else AggregateResolutionReason.KEPT_BELOW_MINIMUM
        )
        decisions.append(
            CandidateAggregateDecision(
                candidate_id=candidate.candidate_id,
                reason=reason,
                mask_area_pixels=area,
                covered_masks=tuple(covered),
            )
        )

    return AggregateResolutionResult(
        frame_id=frame_id,
        mask_containment_at_least=threshold,
        minimum_covered_masks=minimum,
        decisions=tuple(decisions),
    )


def _validated_candidates(
    candidates: Sequence[PositionedCandidateMask],
) -> tuple[PositionedCandidateMask, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of PositionedCandidateMask values.")
    checked: list[PositionedCandidateMask] = []
    for candidate in candidates:
        if not isinstance(candidate, PositionedCandidateMask):
            raise TypeError("candidates must contain only PositionedCandidateMask values.")
        checked.append(candidate)
    ordered = tuple(sorted(checked, key=lambda item: item.candidate_id.encode("utf-8")))
    candidate_ids = [candidate.candidate_id for candidate in ordered]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique within one resolution call.")
    frame_ids = {candidate.frame_id for candidate in ordered}
    if len(frame_ids) > 1:
        raise ValueError("all candidates in one resolution call must share frame_id.")
    return ordered


def _intersection_pixels(
    left: PositionedCandidateMask,
    right: PositionedCandidateMask,
) -> int:
    if left.mask is None or right.mask is None:
        raise ValueError("intersection requires two available masks.")
    left_x2 = left.origin_x + left.mask.shape[1]
    left_y2 = left.origin_y + left.mask.shape[0]
    right_x2 = right.origin_x + right.mask.shape[1]
    right_y2 = right.origin_y + right.mask.shape[0]
    overlap_x1 = max(left.origin_x, right.origin_x)
    overlap_y1 = max(left.origin_y, right.origin_y)
    overlap_x2 = min(left_x2, right_x2)
    overlap_y2 = min(left_y2, right_y2)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0

    left_crop = left.mask[
        overlap_y1 - left.origin_y : overlap_y2 - left.origin_y,
        overlap_x1 - left.origin_x : overlap_x2 - left.origin_x,
    ]
    right_crop = right.mask[
        overlap_y1 - right.origin_y : overlap_y2 - right.origin_y,
        overlap_x1 - right.origin_x : overlap_x2 - right.origin_x,
    ]
    return int(np.count_nonzero(np.logical_and(left_crop, right_crop)))


def _require_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _containment_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("mask_containment_at_least must be a real number.")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("mask_containment_at_least must be finite and in [0, 1].")
    return converted


def _positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be positive.")
    return value


__all__ = [
    "DEFAULT_MASK_CONTAINMENT",
    "DEFAULT_MINIMUM_COVERED_MASKS",
    "AggregateResolutionReason",
    "AggregateResolutionResult",
    "CandidateAggregateDecision",
    "CoveredMaskEvidence",
    "PositionedCandidateMask",
    "resolve_aggregate_masks",
]
