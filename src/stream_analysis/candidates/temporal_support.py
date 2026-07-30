"""Prediction-only temporal support for low-score object proposals.

Base proposals are kept directly. Lower-score proposals require matching
geometry in an adjacent frame and must not be contained by a base proposal in
the same frame. The selection depends only on model predictions and never reads
evaluation annotations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from stream_analysis.contracts import BBox


TEMPORAL_PROFILE_ID = "temporal_low_score_v1"
TEMPORAL_BASE_SCORE = 0.15
TEMPORAL_LOW_SCORE = 0.10
TEMPORAL_SUPPORT_IOU = 0.50
TEMPORAL_CONTAINMENT_REJECT = 0.70
TEMPORAL_ADJACENT_RADIUS = 1


@dataclass(frozen=True, slots=True)
class Proposal:
    frame_id: str
    frame_index: int
    prediction_index: int
    score: float
    phrase: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class TemporalSelection:
    final_proposals: tuple[Proposal, ...]
    supplemental_proposals: tuple[Proposal, ...]
    audit_rows: tuple[dict[str, Any], ...]


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection = left.intersection(right)
    if intersection is None:
        return 0.0
    union = left.area + right.area - intersection.area
    return 0.0 if union <= 0.0 else intersection.area / union


def class_agnostic_nms(
    proposals: Iterable[Proposal], iou_threshold: float
) -> tuple[Proposal, ...]:
    """Apply deterministic class-agnostic non-maximum suppression."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    by_frame: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        by_frame[proposal.frame_id].append(proposal)
    kept: list[Proposal] = []
    for frame_id in sorted(by_frame):
        accepted: list[Proposal] = []
        ordered = sorted(
            by_frame[frame_id],
            key=lambda item: (-item.score, item.prediction_index),
        )
        for proposal in ordered:
            if all(
                bbox_iou(proposal.bbox, earlier.bbox) <= iou_threshold
                for earlier in accepted
            ):
                accepted.append(proposal)
        kept.extend(sorted(accepted, key=lambda item: item.prediction_index))
    return tuple(kept)


def geometry_rejects(
    proposal: Proposal,
    frame_size: Mapping[str, Any],
    *,
    min_area_ratio: float | None,
    min_span_ratio: float | None,
) -> bool:
    if min_area_ratio is None:
        return False
    width = _positive_number(frame_size, "width")
    height = _positive_number(frame_size, "height")
    area_ratio = proposal.bbox.area / (width * height)
    if area_ratio < min_area_ratio:
        return False
    if min_span_ratio is None:
        return True
    return (
        proposal.bbox.width / width >= min_span_ratio
        or proposal.bbox.height / height >= min_span_ratio
    )


def temporal_support_selection(
    raw_proposals: Sequence[Proposal],
    *,
    frame_ids: Sequence[str],
    frame_sizes: Mapping[str, Mapping[str, Any]],
    nms_iou: float,
    min_area_ratio: float | None,
    min_span_ratio: float | None,
) -> TemporalSelection:
    """Select temporally supported proposals without evaluation annotations."""

    ordered_frames = tuple(frame_ids)
    if not ordered_frames or len(set(ordered_frames)) != len(ordered_frames):
        raise ValueError("frame_ids must contain a non-empty unique sequence")
    if set(frame_sizes) != set(ordered_frames):
        raise ValueError("frame_sizes must exactly match frame_ids")
    low_scored = tuple(
        proposal for proposal in raw_proposals if proposal.score >= TEMPORAL_LOW_SCORE
    )
    low_nms = class_agnostic_nms(low_scored, nms_iou)
    low_pool = tuple(
        proposal
        for proposal in low_nms
        if not geometry_rejects(
            proposal,
            frame_sizes[proposal.frame_id],
            min_area_ratio=min_area_ratio,
            min_span_ratio=min_span_ratio,
        )
    )
    base = tuple(
        proposal for proposal in low_pool if proposal.score >= TEMPORAL_BASE_SCORE
    )
    low_by_frame = _proposals_by_frame(low_pool)
    base_by_frame = _proposals_by_frame(base)
    frame_position = {frame_id: index for index, frame_id in enumerate(ordered_frames)}
    supplemental: list[Proposal] = []
    audit: list[dict[str, Any]] = []
    for proposal in low_pool:
        if proposal.score >= TEMPORAL_BASE_SCORE:
            continue
        position = frame_position.get(proposal.frame_id)
        if position is None:
            raise ValueError(
                f"proposal frame is absent from frame_ids: {proposal.frame_id}"
            )
        neighbor_ids = tuple(
            ordered_frames[index]
            for index in (
                position - TEMPORAL_ADJACENT_RADIUS,
                position + TEMPORAL_ADJACENT_RADIUS,
            )
            if 0 <= index < len(ordered_frames)
        )
        support_rows = [
            (bbox_iou(proposal.bbox, other.bbox), other)
            for neighbor_id in neighbor_ids
            for other in low_by_frame.get(neighbor_id, ())
        ]
        best_support_iou, best_support = max(
            support_rows,
            key=lambda item: (
                item[0],
                item[1].score,
                -item[1].prediction_index,
            ),
            default=(0.0, None),
        )
        maximum_containment = 0.0
        for primary in base_by_frame.get(proposal.frame_id, ()):
            intersection = proposal.bbox.intersection(primary.bbox)
            if intersection is not None:
                maximum_containment = max(
                    maximum_containment,
                    intersection.area / proposal.bbox.area,
                )
        supported = best_support_iou >= TEMPORAL_SUPPORT_IOU
        containment_rejected = maximum_containment >= TEMPORAL_CONTAINMENT_REJECT
        accepted = supported and not containment_rejected
        if accepted:
            supplemental.append(proposal)
        audit.append(
            {
                "frame_id": proposal.frame_id,
                "frame_index": proposal.frame_index,
                "prediction_index": proposal.prediction_index,
                "score": proposal.score,
                "bbox": _bbox_payload(proposal.bbox),
                "neighbor_frame_ids": list(neighbor_ids),
                "best_support_iou": best_support_iou,
                "best_support": (
                    None
                    if best_support is None
                    else {
                        "frame_id": best_support.frame_id,
                        "prediction_index": best_support.prediction_index,
                        "score": best_support.score,
                    }
                ),
                "maximum_base_containment": maximum_containment,
                "temporal_supported": supported,
                "containment_rejected": containment_rejected,
                "accepted": accepted,
                "decision": (
                    "accepted_supplemental"
                    if accepted
                    else (
                        "rejected_containment"
                        if containment_rejected
                        else "rejected_no_temporal_support"
                    )
                ),
            }
        )
    final = tuple(
        sorted(
            (*base, *supplemental),
            key=lambda item: (item.frame_id, item.prediction_index),
        )
    )
    return TemporalSelection(
        final_proposals=final,
        supplemental_proposals=tuple(
            sorted(
                supplemental,
                key=lambda item: (item.frame_id, item.prediction_index),
            )
        ),
        audit_rows=tuple(
            sorted(audit, key=lambda item: (item["frame_id"], item["prediction_index"]))
        ),
    )


def _positive_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(item)


def _bbox_payload(bbox: BBox) -> dict[str, float]:
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }


def _proposals_by_frame(
    proposals: Sequence[Proposal],
) -> dict[str, tuple[Proposal, ...]]:
    result: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        result[proposal.frame_id].append(proposal)
    return {key: tuple(value) for key, value in result.items()}


__all__ = [
    "Proposal",
    "TEMPORAL_ADJACENT_RADIUS",
    "TEMPORAL_BASE_SCORE",
    "TEMPORAL_CONTAINMENT_REJECT",
    "TEMPORAL_LOW_SCORE",
    "TEMPORAL_PROFILE_ID",
    "TEMPORAL_SUPPORT_IOU",
    "TemporalSelection",
    "bbox_iou",
    "class_agnostic_nms",
    "geometry_rejects",
    "temporal_support_selection",
]
