"""Canonical all-pairs scoring and neighboring-frame matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..contracts import (
    AssignmentProvenance,
    CandidateEndpoint,
    CandidateRecord,
    ErrorRecord,
    FrameMatchingResult,
    FramePair,
    MatchDecisionStatus,
    MatchRecord,
    PairEligibility,
    PairwiseScoreRecord,
    RecordEnvelope,
    RepresentationRecord,
    Severity,
    SpatialEvidence,
    StageContext,
    UnmatchedRecord,
    UnmatchedSide,
    ValidityStatus,
    WarningRecord,
)
from ..representations.scoring import CanonicalScorer
from .assignment import AugmentedAssignmentResult, solve_augmented_assignment
from .config import MatchingConfig, PairScoringConfig

PAIR_SCORE_SCHEMA_VERSION = "pair-score-1.0"
MATCH_SCHEMA_VERSION = "match-1.0"
UNMATCHED_SCHEMA_VERSION = "unmatched-1.0"
FRAME_MATCHING_SCHEMA_VERSION = "frame-matching-1.0"
MATCHING_ERROR_SCHEMA_VERSION = "matching-error-1.0"


@dataclass(frozen=True, slots=True)
class NeighboringMatchingBatch:
    result: FrameMatchingResult
    warnings: tuple[WarningRecord, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()


class _EndpointPrerequisiteError(Exception):
    def __init__(self, error_ids: tuple[str, ...]) -> None:
        super().__init__("Invalid representation endpoint prerequisite.")
        self.error_ids = error_ids


def match_neighboring_frames(
    *,
    stream_id: str,
    frame_pair: FramePair,
    from_candidates: Iterable[CandidateRecord],
    to_candidates: Iterable[CandidateRecord],
    representations: Iterable[RepresentationRecord],
    representation_variant_id: str,
    scorer: CanonicalScorer,
    scoring_config: PairScoringConfig,
    matching_config: MatchingConfig,
) -> NeighboringMatchingBatch:
    """Score every real pair and solve one deterministic augmented assignment."""

    if not isinstance(frame_pair, FramePair):
        raise TypeError("frame_pair must be FramePair.")
    if not isinstance(stream_id, str) or not stream_id or any(char.isspace() for char in stream_id):
        raise ValueError("stream_id must be a non-empty identifier.")
    if scorer.scorer_id != scoring_config.scorer_id or scorer.scorer_version != scoring_config.scorer_version:
        raise ValueError("scorer identity must match scoring_config.")
    left = tuple(sorted(from_candidates, key=lambda item: item.candidate_id))
    right = tuple(sorted(to_candidates, key=lambda item: item.candidate_id))
    _validate_candidates(left, frame_pair, UnmatchedSide.FROM)
    _validate_candidates(right, frame_pair, UnmatchedSide.TO)
    all_candidates = left + right
    if any(candidate.envelope.stream_id != stream_id for candidate in all_candidates):
        raise ValueError("All candidates must belong to stream_id.")
    records = tuple(representations)
    if len({record.candidate_id for record in records}) != len(records):
        raise ValueError("representations must be unique by candidate_id.")
    if any(record.envelope.stream_id != stream_id for record in records):
        raise ValueError("All representations must belong to stream_id.")
    representation_by_candidate = {record.candidate_id: record for record in records}
    comparison_id = f"comparison:{frame_pair.from_frame_id}:{frame_pair.to_frame_id}"

    errors: list[ErrorRecord] = []
    endpoint_error_ids: dict[str, tuple[str, ...]] = {}
    for candidate in all_candidates:
        record = representation_by_candidate.get(candidate.candidate_id)
        reason = _representation_prerequisite_reason(
            candidate, record, representation_variant_id, scorer
        )
        if reason is None:
            continue
        error_id = f"error:representation_prerequisite:{comparison_id}:{candidate.candidate_id}"
        warning_ids = _ordered_unique(
            candidate.envelope.warning_ids
            + (() if record is None else record.envelope.warning_ids)
        )
        error = ErrorRecord(
            record_id=error_id,
            schema_version=MATCHING_ERROR_SCHEMA_VERSION,
            stream_id=stream_id,
            code="REPRESENTATION_PREREQUISITE_INVALID",
            stage="pair_scoring",
            message=reason,
            producer=scoring_config.producer,
            severity=Severity.ERROR,
            context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
            provenance_refs=(
                candidate.candidate_id,
                *((record.envelope.record_id,) if record is not None else ()),
            ),
            upstream_warning_ids=warning_ids,
            upstream_error_ids=() if record is None else record.envelope.error_ids,
        )
        errors.append(error)
        endpoint_error_ids[candidate.candidate_id] = (error_id,)

    left_endpoints = tuple(_endpoint(candidate) for candidate in left)
    right_endpoints = tuple(_endpoint(candidate) for candidate in right)
    pair_scores: list[PairwiseScoreRecord] = []
    costs = np.full((len(left), len(right)), np.inf, dtype=np.float64)
    spatial = np.zeros((len(left), len(right)), dtype=np.float64)

    for i, left_candidate in enumerate(left):
        for j, right_candidate in enumerate(right):
            pair_id = f"pair:{comparison_id}:{left_candidate.candidate_id}:{right_candidate.candidate_id}"
            distance = _spatial_distance(left_candidate, right_candidate)
            spatial[i, j] = distance
            left_record = representation_by_candidate.get(left_candidate.candidate_id)
            right_record = representation_by_candidate.get(right_candidate.candidate_id)
            warning_ids = _ordered_unique(
                left_candidate.envelope.warning_ids
                + right_candidate.envelope.warning_ids
                + (() if left_record is None else left_record.envelope.warning_ids)
                + (() if right_record is None else right_record.envelope.warning_ids)
            )
            provenance_refs = _ordered_unique(
                (
                    left_candidate.candidate_id,
                    right_candidate.candidate_id,
                    *((left_record.envelope.record_id,) if left_record is not None else ()),
                    *((right_record.envelope.record_id,) if right_record is not None else ()),
                )
            )
            try:
                prerequisite_error_ids = _ordered_unique(
                    endpoint_error_ids.get(left_candidate.candidate_id, ())
                    + endpoint_error_ids.get(right_candidate.candidate_id, ())
                )
                if prerequisite_error_ids:
                    raise _EndpointPrerequisiteError(prerequisite_error_ids)
                assert left_record is not None and right_record is not None
                canonical = scorer.score(left_record, right_record)
            except _EndpointPrerequisiteError as exc:
                envelope = RecordEnvelope(
                    record_id=pair_id,
                    schema_version=PAIR_SCORE_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=scoring_config.producer,
                    context=StageContext(pair_id=pair_id),
                    validity_status=ValidityStatus.INVALID,
                    warning_ids=warning_ids,
                    error_ids=exc.error_ids,
                    provenance_refs=provenance_refs,
                )
                pair_scores.append(
                    _invalid_pair_score(
                        envelope=envelope,
                        pair_id=pair_id,
                        comparison_id=comparison_id,
                        frame_pair=frame_pair,
                        from_endpoint=left_endpoints[i],
                        to_endpoint=right_endpoints[j],
                        left_record=left_record,
                        right_record=right_record,
                        representation_variant_id=representation_variant_id,
                        scorer=scorer,
                        distance=distance,
                    )
                )
                continue
            except (TypeError, ValueError) as exc:
                error_id = f"error:{pair_id}"
                error = ErrorRecord(
                    record_id=error_id,
                    schema_version=MATCHING_ERROR_SCHEMA_VERSION,
                    stream_id=stream_id,
                    code="PAIR_SCORING_FAILED",
                    stage="pair_scoring",
                    message=str(exc),
                    producer=scoring_config.producer,
                    severity=Severity.ERROR,
                    context=StageContext(pair_id=pair_id),
                    provenance_refs=provenance_refs,
                    upstream_warning_ids=warning_ids,
                    upstream_error_ids=_ordered_unique(
                        (() if left_record is None else left_record.envelope.error_ids)
                        + (() if right_record is None else right_record.envelope.error_ids)
                    ),
                )
                errors.append(error)
                envelope = RecordEnvelope(
                    record_id=pair_id,
                    schema_version=PAIR_SCORE_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=scoring_config.producer,
                    context=StageContext(pair_id=pair_id),
                    validity_status=ValidityStatus.INVALID,
                    warning_ids=warning_ids,
                    error_ids=(error_id,),
                    provenance_refs=provenance_refs,
                )
                pair_scores.append(
                    _invalid_pair_score(
                        envelope=envelope,
                        pair_id=pair_id,
                        comparison_id=comparison_id,
                        frame_pair=frame_pair,
                        from_endpoint=left_endpoints[i],
                        to_endpoint=right_endpoints[j],
                        left_record=left_record,
                        right_record=right_record,
                        representation_variant_id=representation_variant_id,
                        scorer=scorer,
                        distance=distance,
                    )
                )
                continue

            visual_pass = canonical.visual_score >= scoring_config.visual_gate
            spatial_pass = scoring_config.spatial_gate is None or distance <= scoring_config.spatial_gate
            eligible = visual_pass and spatial_pass
            reason = None if eligible else ("BELOW_VISUAL_GATE" if not visual_pass else "SPATIAL_GATE")
            real_cost = 1.0 - canonical.visual_score if eligible else None
            if eligible:
                costs[i, j] = real_cost
            envelope = RecordEnvelope(
                record_id=pair_id,
                schema_version=PAIR_SCORE_SCHEMA_VERSION,
                stream_id=stream_id,
                producer=scoring_config.producer,
                context=StageContext(pair_id=pair_id),
                warning_ids=warning_ids,
                provenance_refs=provenance_refs,
            )
            pair_scores.append(
                PairwiseScoreRecord(
                    envelope=envelope,
                    pair_id=pair_id,
                    comparison_id=comparison_id,
                    frame_pair=frame_pair,
                    from_endpoint=left_endpoints[i],
                    to_endpoint=right_endpoints[j],
                    representation_record_ids=(left_record.envelope.record_id, right_record.envelope.record_id),
                    representation_variant_id=representation_variant_id,
                    scorer_id=scorer.scorer_id,
                    scorer_version=scorer.scorer_version,
                    raw_metric_name=canonical.raw_metric_name,
                    raw_metric_value=canonical.raw_metric_value,
                    metric_orientation=canonical.metric_orientation,
                    visual_score=canonical.visual_score,
                    real_cost=real_cost,
                    spatial_evidence=SpatialEvidence(normalized_distance=distance, details={"source": "bbox_centers"}),
                    eligibility=PairEligibility.ELIGIBLE if eligible else PairEligibility.INELIGIBLE,
                    eligibility_reason=reason,
                    gate_results={
                        "spatial_gate_enabled": scoring_config.spatial_gate is not None,
                        "spatial_passed": spatial_pass,
                        "visual_passed": visual_pass,
                    },
                    candidate_quality_refs=(left_candidate.candidate_id, right_candidate.candidate_id),
                )
            )

    assignment_result = solve_augmented_assignment(costs, spatial, matching_config.unmatched_pair_cost)
    assignment = AssignmentProvenance(
        assignment_id=f"assignment:{comparison_id}",
        policy_id=matching_config.assignment_policy_id,
        policy_version=matching_config.assignment_policy_version,
        config_digest=matching_config.config_digest,
        details={
            "backend": "scipy.optimize.linear_sum_assignment",
            "matrix_order": assignment_result.matrix_order,
            "primary_total_cost": assignment_result.primary_total_cost,
            "primary_tie_tolerance": assignment_result.primary_tie_tolerance,
            "spatial_tie_tolerance": assignment_result.spatial_tie_tolerance,
            "spatial_policy": "lexicographic_then_candidate_ids",
        },
    )
    score_by_endpoints = {
        (score.from_endpoint.candidate_id, score.to_endpoint.candidate_id): score for score in pair_scores
    }
    quality_flags_by_candidate = {
        candidate.candidate_id: set(candidate.quality_flags).union(
            () if representation_by_candidate.get(candidate.candidate_id) is None
            else representation_by_candidate[candidate.candidate_id].input_quality_metadata.quality_flags
        )
        for candidate in all_candidates
    }
    matches = _build_matches(
        assignment_result,
        left,
        right,
        score_by_endpoints,
        quality_flags_by_candidate,
        assignment,
        matching_config,
        stream_id,
    )
    unmatched = _build_unmatched(
        assignment_result,
        left,
        right,
        pair_scores,
        assignment,
        matching_config,
        stream_id,
        comparison_id,
        frame_pair,
        endpoint_error_ids,
    )
    error_ids = tuple(error.record_id for error in errors)
    result_warning_ids = tuple(sorted({
        *(
            warning_id
            for candidate in all_candidates
            for warning_id in candidate.envelope.warning_ids
        ),
        *(
            warning_id
            for record in records
            if record.candidate_id in {candidate.candidate_id for candidate in all_candidates}
            for warning_id in record.envelope.warning_ids
        ),
        *(warning_id for score in pair_scores for warning_id in score.envelope.warning_ids),
    }))
    result_envelope = RecordEnvelope(
        record_id=f"matching_result:{comparison_id}",
        schema_version=FRAME_MATCHING_SCHEMA_VERSION,
        stream_id=stream_id,
        producer=matching_config.producer,
        context=StageContext(pair_id=comparison_id),
        validity_status=ValidityStatus.INVALID if errors else ValidityStatus.VALID,
        warning_ids=result_warning_ids,
        error_ids=error_ids,
        provenance_refs=tuple(sorted(candidate.candidate_id for candidate in all_candidates)),
    )
    result = FrameMatchingResult(
        envelope=result_envelope,
        comparison_id=comparison_id,
        frame_pair=frame_pair,
        representation_variant_id=representation_variant_id,
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.scorer_version,
        assignment=assignment,
        from_candidates=left_endpoints,
        to_candidates=right_endpoints,
        pairwise_scores=tuple(pair_scores),
        selected_matches=matches,
        unmatched=unmatched,
        matrix_summary={
            "eligible_pair_count": int(np.count_nonzero(np.isfinite(costs))),
            "forbidden_cost": assignment_result.forbidden_cost,
            "matrix_order": assignment_result.matrix_order,
            "primary_total_cost": assignment_result.primary_total_cost,
            "primary_tie_tolerance": assignment_result.primary_tie_tolerance,
            "spatial_tie_tolerance": assignment_result.spatial_tie_tolerance,
            "real_pair_count": len(left) * len(right),
            "unmatched_left_cost": assignment_result.unmatched_left_cost,
            "unmatched_right_cost": assignment_result.unmatched_right_cost,
        },
    )
    return NeighboringMatchingBatch(result=result, errors=tuple(errors))


def _build_matches(
    solved: AugmentedAssignmentResult,
    left: tuple[CandidateRecord, ...],
    right: tuple[CandidateRecord, ...],
    score_by_endpoints: dict[tuple[str, str], PairwiseScoreRecord],
    quality_flags_by_candidate: dict[str, set[str]],
    assignment: AssignmentProvenance,
    config: MatchingConfig,
    stream_id: str,
) -> tuple[MatchRecord, ...]:
    records: list[MatchRecord] = []
    for decision in solved.real_matches:
        left_candidate = left[decision.left_index]
        right_candidate = right[decision.right_index]
        score = score_by_endpoints[(left_candidate.candidate_id, right_candidate.candidate_id)]
        assert score.visual_score is not None and score.real_cost is not None
        flags = (
            quality_flags_by_candidate[left_candidate.candidate_id]
            | quality_flags_by_candidate[right_candidate.candidate_id]
        )
        margins_pass = (
            _margin_pass(decision.delta_row, config.local_margin_gate)
            and _margin_pass(decision.delta_col, config.local_margin_gate)
            and _margin_pass(decision.delta_global, config.global_margin_gate)
        )
        status = (
            MatchDecisionStatus.ACCEPTED
            if margins_pass and not flags.intersection(config.severe_quality_flags)
            else MatchDecisionStatus.UNCERTAIN
        )
        alternatives = tuple(
            candidate.pair_id
            for candidate in sorted(
                (
                    item
                    for item in score_by_endpoints.values()
                    if item.eligibility is PairEligibility.ELIGIBLE
                    and item.pair_id != score.pair_id
                    and (
                        item.from_endpoint.candidate_id == left_candidate.candidate_id
                        or item.to_endpoint.candidate_id == right_candidate.candidate_id
                    )
                ),
                key=lambda item: (-float(item.visual_score or 0.0), item.pair_id),
            )[:4]
        )
        confidence_components = {
            "visual_score": score.visual_score,
            **({"delta_row": decision.delta_row} if decision.delta_row is not None else {}),
            **({"delta_col": decision.delta_col} if decision.delta_col is not None else {}),
            **({"delta_global": decision.delta_global} if decision.delta_global is not None else {}),
        }
        match_id = f"match:{score.comparison_id}:{left_candidate.candidate_id}:{right_candidate.candidate_id}"
        records.append(
            MatchRecord(
                envelope=RecordEnvelope(
                    record_id=match_id,
                    schema_version=MATCH_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=config.producer,
                    context=StageContext(pair_id=score.pair_id),
                    warning_ids=score.envelope.warning_ids,
                    provenance_refs=(score.pair_id,),
                ),
                match_id=match_id,
                comparison_id=score.comparison_id,
                pair_id=score.pair_id,
                frame_pair=score.frame_pair,
                from_endpoint=score.from_endpoint,
                to_endpoint=score.to_endpoint,
                pairwise_score_id=score.pair_id,
                assignment=assignment,
                status=status,
                visual_score=score.visual_score,
                real_cost=score.real_cost,
                spatial_evidence=score.spatial_evidence,
                confidence_components=confidence_components,
                delta_row=decision.delta_row,
                delta_col=decision.delta_col,
                delta_global=decision.delta_global,
                alternative_pair_ids=alternatives,
            )
        )
    return tuple(records)


def _build_unmatched(
    solved: AugmentedAssignmentResult,
    left: tuple[CandidateRecord, ...],
    right: tuple[CandidateRecord, ...],
    scores: list[PairwiseScoreRecord],
    assignment: AssignmentProvenance,
    config: MatchingConfig,
    stream_id: str,
    comparison_id: str,
    frame_pair: FramePair,
    endpoint_error_ids: dict[str, tuple[str, ...]],
) -> tuple[UnmatchedRecord, ...]:
    records: list[UnmatchedRecord] = []
    for decision, side, candidates in (
        *((item, UnmatchedSide.FROM, left) for item in solved.unmatched_left),
        *((item, UnmatchedSide.TO, right) for item in solved.unmatched_right),
    ):
        candidate = candidates[decision.index]
        endpoint = _endpoint(candidate)
        related = [
            score for score in scores
            if (side is UnmatchedSide.FROM and score.from_endpoint.candidate_id == candidate.candidate_id)
            or (side is UnmatchedSide.TO and score.to_endpoint.candidate_id == candidate.candidate_id)
        ]
        best = max(
            (score for score in related if score.visual_score is not None),
            key=lambda item: (float(item.visual_score or 0.0), item.pair_id),
            default=None,
        )
        reason = (
            "INVALID_REPRESENTATION"
            if candidate.candidate_id in endpoint_error_ids
            else _unmatched_reason(related)
        )
        related_error_ids = _ordered_unique(
            endpoint_error_ids.get(candidate.candidate_id, ())
            + tuple(
                error_id
                for score in related
                for error_id in score.envelope.error_ids
            )
        )
        unmatched_id = f"unmatched:{comparison_id}:{side.value}:{candidate.candidate_id}"
        validity = (
            ValidityStatus.INVALID
            if candidate.candidate_id in endpoint_error_ids
            or related_error_ids and related and all(
                score.envelope.validity_status is ValidityStatus.INVALID for score in related
            )
            else ValidityStatus.VALID
        )
        records.append(
            UnmatchedRecord(
                envelope=RecordEnvelope(
                    record_id=unmatched_id,
                    schema_version=UNMATCHED_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=config.producer,
                    context=StageContext(
                        frame_id=candidate.frame_id,
                        candidate_id=candidate.candidate_id,
                        pair_id=comparison_id,
                    ),
                    validity_status=validity,
                    warning_ids=_ordered_unique(
                        candidate.envelope.warning_ids
                        + tuple(
                            warning_id
                            for score in related
                            for warning_id in score.envelope.warning_ids
                        )
                    ),
                    error_ids=related_error_ids if validity is ValidityStatus.INVALID else (),
                    provenance_refs=(candidate.candidate_id,),
                ),
                unmatched_id=unmatched_id,
                comparison_id=comparison_id,
                frame_pair=frame_pair,
                endpoint=endpoint,
                side=side,
                reason_code=reason,
                assignment=assignment,
                selected_dummy_id=f"dummy:{side.value}:{candidate.candidate_id}",
                unmatched_cost=(
                    solved.unmatched_left_cost if side is UnmatchedSide.FROM else solved.unmatched_right_cost
                ),
                global_unmatched_margin=decision.delta_global,
                best_pair_id=None if best is None else best.pair_id,
                best_visual_score=None if best is None else best.visual_score,
                candidate_quality_refs=(candidate.candidate_id,),
            )
        )
    return tuple(records)


def _unmatched_reason(scores: list[PairwiseScoreRecord]) -> str:
    if not scores or all(score.envelope.validity_status is ValidityStatus.INVALID for score in scores):
        return "INVALID_REPRESENTATION" if scores else "NO_ADMISSIBLE_PAIR"
    reasons = {score.eligibility_reason for score in scores if score.eligibility_reason}
    if "BELOW_VISUAL_GATE" in reasons:
        return "BELOW_VISUAL_GATE"
    if "SPATIAL_GATE" in reasons:
        return "SPATIAL_GATE"
    if any(score.eligibility is PairEligibility.ELIGIBLE for score in scores):
        return "LOST_GLOBAL_ASSIGNMENT"
    return "NO_ADMISSIBLE_PAIR"


def _representation_prerequisite_reason(
    candidate: CandidateRecord,
    record: RepresentationRecord | None,
    representation_variant_id: str,
    scorer: CanonicalScorer,
) -> str | None:
    if record is None:
        return "Missing representation for candidate endpoint."
    if record.envelope.stream_id != candidate.envelope.stream_id:
        return "Representation stream does not match candidate stream."
    if record.frame_id != candidate.frame_id:
        return "Representation frame does not match candidate frame."
    if record.envelope.validity_status is not ValidityStatus.VALID or record.payload is None:
        return "Candidate endpoint has an invalid representation prerequisite."
    if record.input_variant != representation_variant_id:
        return "Representation variant does not match requested variant."
    if record.family is not scorer.family:
        return "Representation family is incompatible with scorer."
    return None


def _invalid_pair_score(
    *,
    envelope: RecordEnvelope,
    pair_id: str,
    comparison_id: str,
    frame_pair: FramePair,
    from_endpoint: CandidateEndpoint,
    to_endpoint: CandidateEndpoint,
    left_record: RepresentationRecord | None,
    right_record: RepresentationRecord | None,
    representation_variant_id: str,
    scorer: CanonicalScorer,
    distance: float,
) -> PairwiseScoreRecord:
    return PairwiseScoreRecord(
        envelope=envelope,
        pair_id=pair_id,
        comparison_id=comparison_id,
        frame_pair=frame_pair,
        from_endpoint=from_endpoint,
        to_endpoint=to_endpoint,
        representation_record_ids=(
            left_record.envelope.record_id if left_record else f"missing:{from_endpoint.candidate_id}",
            right_record.envelope.record_id if right_record else f"missing:{to_endpoint.candidate_id}",
        ),
        representation_variant_id=representation_variant_id,
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.scorer_version,
        raw_metric_name=None,
        raw_metric_value=None,
        metric_orientation=_fallback_orientation(scorer),
        visual_score=None,
        real_cost=None,
        spatial_evidence=SpatialEvidence(
            normalized_distance=distance,
            details={"source": "bbox_centers"},
        ),
        eligibility=PairEligibility.INELIGIBLE,
        eligibility_reason="INVALID_REPRESENTATION",
        gate_results={"validity": "failed"},
        candidate_quality_refs=(from_endpoint.candidate_id, to_endpoint.candidate_id),
    )


def _validate_candidates(candidates: tuple[CandidateRecord, ...], pair: FramePair, side: UnmatchedSide) -> None:
    expected = (
        (pair.from_frame_id, pair.from_frame_index, pair.from_frame_size)
        if side is UnmatchedSide.FROM else
        (pair.to_frame_id, pair.to_frame_index, pair.to_frame_size)
    )
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, CandidateRecord):
            raise TypeError("candidate collections must contain CandidateRecord values.")
        if (candidate.frame_id, candidate.frame_index, candidate.frame_size) != expected:
            raise ValueError(f"Candidate does not belong to the {side.value} frame.")
        if candidate.candidate_id in ids:
            raise ValueError("Candidate IDs must be unique per side.")
        ids.add(candidate.candidate_id)


def _endpoint(candidate: CandidateRecord) -> CandidateEndpoint:
    return CandidateEndpoint(
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        frame_index=candidate.frame_index,
    )


def _spatial_distance(left: CandidateRecord, right: CandidateRecord) -> float:
    left_x = left.center.x / left.frame_size.width
    left_y = left.center.y / left.frame_size.height
    right_x = right.center.x / right.frame_size.width
    right_y = right.center.y / right.frame_size.height
    return math.hypot(left_x - right_x, left_y - right_y) / math.sqrt(2.0)


def _margin_pass(value: float | None, gate: float) -> bool:
    return value is None or value >= gate


def _fallback_orientation(scorer: CanonicalScorer):
    return scorer.metric_orientation


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "FRAME_MATCHING_SCHEMA_VERSION",
    "MATCHING_ERROR_SCHEMA_VERSION",
    "MATCH_SCHEMA_VERSION",
    "PAIR_SCORE_SCHEMA_VERSION",
    "UNMATCHED_SCHEMA_VERSION",
    "NeighboringMatchingBatch",
    "match_neighboring_frames",
]
