"""Deterministic change-event derivation from primary visual-type membership."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..contracts import (
    CalibrationStatus,
    CandidateRecord,
    ChangeEvent,
    ConfidenceMetadata,
    EmissionStatus,
    ErrorRecord,
    EventEvidence,
    EventKind,
    EventStatus,
    FrameComparison,
    FrameMatchingResult,
    GroupingResult,
    MatchDecisionStatus,
    MembershipStatus,
    RecordEnvelope,
    Severity,
    StageContext,
    TypePresenceSummary,
    ValidityStatus,
    WarningRecord,
)
from .config import EventConfig

CHANGE_EVENT_SCHEMA_VERSION = "change-event-1.0"
FRAME_COMPARISON_SCHEMA_VERSION = "frame-comparison-1.0"
EVENT_ERROR_SCHEMA_VERSION = "event-error-1.0"


@dataclass(frozen=True, slots=True)
class EventGenerationBatch:
    comparisons: tuple[FrameComparison, ...]
    events: tuple[ChangeEvent, ...]
    warnings: tuple[WarningRecord, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()


def build_change_events(
    *,
    stream_id: str,
    candidates: Iterable[CandidateRecord],
    grouping: GroupingResult,
    matching_results: Iterable[FrameMatchingResult],
    config: EventConfig,
) -> EventGenerationBatch:
    """Emit the five contract event kinds, or withhold an invalid comparison."""

    candidate_values = tuple(candidates)
    candidate_by_id = {item.candidate_id: item for item in candidate_values}
    if len(candidate_by_id) != len(candidate_values):
        raise ValueError("candidates must be unique by candidate_id.")
    if grouping.envelope.stream_id != stream_id:
        raise ValueError("grouping must belong to stream_id.")
    assignment_by_id = {item.candidate_id: item for item in grouping.assignments}
    type_by_id = {item.type_id: item for item in grouping.recurring_types}
    type_ids = tuple(item.type_id for item in grouping.recurring_types)
    results = tuple(
        sorted(
            matching_results,
            key=lambda item: (
                item.frame_pair.from_frame_index,
                item.frame_pair.to_frame_index,
                item.comparison_id,
            ),
        )
    )
    comparisons: list[FrameComparison] = []
    events: list[ChangeEvent] = []
    errors: list[ErrorRecord] = []

    for result in results:
        if result.envelope.stream_id != stream_id:
            raise ValueError("matching_results must belong to stream_id.")
        if result.representation_variant_id != grouping.representation_variant_id:
            raise ValueError("matching and grouping representation variants must agree.")
        if result.envelope.validity_status is ValidityStatus.INVALID:
            error = _invalid_matching_error(stream_id, result, config)
            errors.append(error)
            comparisons.append(_withheld_comparison(result, grouping, config, error))
            continue

        from_presence = _presence(type_ids, result.frame_pair.from_frame_id, assignment_by_id)
        to_presence = _presence(type_ids, result.frame_pair.to_frame_id, assignment_by_id)
        comparison_events: list[ChangeEvent] = []
        for type_id in type_ids:
            primary_from = _primary_members(type_id, result.frame_pair.from_frame_id, assignment_by_id)
            primary_to = _primary_members(type_id, result.frame_pair.to_frame_id, assignment_by_id)
            kinds: list[tuple[EventKind, float | None]] = []
            if primary_from and primary_to:
                kinds.append((EventKind.PERSISTED, None))
                if len(primary_from) != len(primary_to):
                    kinds.append((EventKind.COUNT_CHANGED, None))
                elif _position_is_decidable(primary_from, primary_to, candidate_by_id):
                    shift = _position_shift(primary_from, primary_to, candidate_by_id)
                    if shift > config.position_threshold_norm:
                        kinds.append((EventKind.POSITION_CHANGED, shift))
            elif not primary_from and primary_to:
                kinds.append((EventKind.APPEARED, None))
            elif primary_from and not primary_to:
                kinds.append((EventKind.DISAPPEARED, None))

            for kind, shift in kinds:
                comparison_events.append(
                    _event(
                        result=result,
                        type_id=type_id,
                        kind=kind,
                        position_shift=shift,
                        primary_from=primary_from,
                        primary_to=primary_to,
                        from_presence=from_presence[type_id],
                        to_presence=to_presence[type_id],
                        assignment_by_id=assignment_by_id,
                        type_record=type_by_id[type_id],
                        candidate_by_id=candidate_by_id,
                        config=config,
                    )
                )
        comparison_events.sort(key=lambda item: (item.predicted_type_id, item.kind.value, item.event_id))
        events.extend(comparison_events)
        comparisons.append(
            _produced_comparison(result, grouping, from_presence, to_presence, comparison_events, config)
        )

    return EventGenerationBatch(
        comparisons=tuple(comparisons),
        events=tuple(events),
        errors=tuple(errors),
    )


def _presence(type_ids, frame_id, assignment_by_id) -> dict[str, TypePresenceSummary]:
    result: dict[str, TypePresenceSummary] = {}
    frame_assignments = tuple(item for item in assignment_by_id.values() if item.frame_id == frame_id)
    for type_id in type_ids:
        possible = tuple(
            sorted(
                item.candidate_id
                for item in frame_assignments
                if item.primary_type_id == type_id or type_id in item.alternative_type_ids
            )
        )
        firm_count = sum(
            item.primary_type_id == type_id and item.membership_status is MembershipStatus.FIRM
            for item in frame_assignments
        )
        result[type_id] = TypePresenceSummary(
            type_id=type_id,
            candidate_ids=possible,
            firm_count=firm_count,
            possible_count=len(possible),
        )
    return result


def _primary_members(type_id, frame_id, assignment_by_id) -> tuple[str, ...]:
    return tuple(
        sorted(
            item.candidate_id
            for item in assignment_by_id.values()
            if item.frame_id == frame_id and item.primary_type_id == type_id
        )
    )


def _position_is_decidable(from_ids, to_ids, candidate_by_id) -> bool:
    return bool(from_ids) and len(from_ids) == len(to_ids) and all(
        candidate_id in candidate_by_id for candidate_id in (*from_ids, *to_ids)
    )


def _position_shift(from_ids, to_ids, candidate_by_id) -> float:
    left = _normalized_centroid(from_ids, candidate_by_id)
    right = _normalized_centroid(to_ids, candidate_by_id)
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _normalized_centroid(candidate_ids, candidate_by_id) -> tuple[float, float]:
    points = [
        (
            candidate_by_id[item].center.x / candidate_by_id[item].frame_size.width,
            candidate_by_id[item].center.y / candidate_by_id[item].frame_size.height,
        )
        for item in candidate_ids
    ]
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _event(
    *, result, type_id, kind, position_shift, primary_from, primary_to,
    from_presence, to_presence, assignment_by_id, type_record, candidate_by_id, config,
) -> ChangeEvent:
    accepted_ids: list[str] = []
    uncertain_ids: list[str] = []
    from_set, to_set = set(primary_from), set(primary_to)
    for match in result.selected_matches:
        if match.from_endpoint.candidate_id in from_set and match.to_endpoint.candidate_id in to_set:
            target = accepted_ids if match.status is MatchDecisionStatus.ACCEPTED else uncertain_ids
            target.append(match.match_id)

    primary_ids = (*primary_from, *primary_to)
    assignments = [assignment_by_id[item] for item in primary_ids]
    all_firm = all(item.membership_status is MembershipStatus.FIRM for item in assignments)
    no_alternatives = (
        set(from_presence.candidate_ids) == from_set
        and set(to_presence.candidate_ids) == to_set
    )
    severe_flags = sorted(
        {
            flag
            for candidate_id in primary_ids
            for flag in candidate_by_id[candidate_id].quality_flags
            if flag in config.severe_quality_flags
        }
    )
    status = (
        EventStatus.CERTAIN
        if all_firm and no_alternatives and not uncertain_ids and not severe_flags
        else EventStatus.UNCERTAIN
    )
    event_id = f"event:{result.comparison_id}:{type_id}:{kind.value}"
    warning_ids = tuple(
        sorted(
            {
                *result.envelope.warning_ids,
                *type_record.envelope.warning_ids,
                *(
                    warning_id
                    for candidate_id in primary_ids
                    for warning_id in (
                        candidate_by_id[candidate_id].envelope.warning_ids
                        + assignment_by_id[candidate_id].envelope.warning_ids
                    )
                ),
            }
        )
    )
    position_details = {}
    if position_shift is not None:
        position_details = {
            "from_centroid_norm": _normalized_centroid(primary_from, candidate_by_id),
            "to_centroid_norm": _normalized_centroid(primary_to, candidate_by_id),
            "position_threshold_norm": config.position_threshold_norm,
        }
    evidence = EventEvidence(
        from_count=len(primary_from),
        to_count=len(primary_to),
        from_firm_count=sum(
            assignment_by_id[item].membership_status is MembershipStatus.FIRM for item in primary_from
        ),
        to_firm_count=sum(
            assignment_by_id[item].membership_status is MembershipStatus.FIRM for item in primary_to
        ),
        from_member_ids=primary_from,
        to_member_ids=primary_to,
        accepted_match_ids=tuple(accepted_ids),
        uncertain_match_ids=tuple(uncertain_ids),
        position_shift_norm=position_shift,
        details={
            "from_possible_count": from_presence.possible_count,
            "to_possible_count": to_presence.possible_count,
            "severe_quality_flags": severe_flags,
            "membership_policy": "primary_with_uncertainty_v1",
            "alternative_membership_affects_evidence": not no_alternatives,
            **position_details,
        },
    )
    confidence = ConfidenceMetadata(
        value=None,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        components={
            "from_firm_fraction": (
                evidence.from_firm_count / evidence.from_count if evidence.from_count else 1.0
            ),
            "to_firm_fraction": (
                evidence.to_firm_count / evidence.to_count if evidence.to_count else 1.0
            ),
        },
    )
    return ChangeEvent(
        envelope=RecordEnvelope(
            record_id=event_id,
            schema_version=CHANGE_EVENT_SCHEMA_VERSION,
            stream_id=result.envelope.stream_id,
            producer=config.producer,
            context=StageContext(pair_id=result.comparison_id, type_id=type_id, event_id=event_id),
            warning_ids=warning_ids,
            provenance_refs=tuple(
                dict.fromkeys((result.envelope.record_id, *primary_ids, *accepted_ids, *uncertain_ids))
            ),
        ),
        event_id=event_id,
        comparison_id=result.comparison_id,
        frame_pair=result.frame_pair,
        kind=kind,
        predicted_type_id=type_id,
        status=status,
        evidence=evidence,
        confidence=confidence,
        policy_id=config.event_policy_id,
        policy_version=config.event_policy_version,
    )


def _produced_comparison(result, grouping, from_presence, to_presence, events, config):
    event_ids = tuple(item.event_id for item in events)
    return FrameComparison(
        envelope=RecordEnvelope(
            record_id=result.comparison_id,
            schema_version=FRAME_COMPARISON_SCHEMA_VERSION,
            stream_id=result.envelope.stream_id,
            producer=config.producer,
            context=StageContext(pair_id=result.comparison_id),
            warning_ids=tuple(sorted(set(result.envelope.warning_ids) | set(grouping.envelope.warning_ids))),
            provenance_refs=(result.envelope.record_id, grouping.envelope.record_id, *event_ids),
        ),
        comparison_id=result.comparison_id,
        frame_pair=result.frame_pair,
        emission_status=EmissionStatus.PRODUCED,
        representation_variant_id=result.representation_variant_id,
        scorer_id=result.scorer_id,
        scorer_version=result.scorer_version,
        event_policy_id=config.event_policy_id,
        event_policy_version=config.event_policy_version,
        accepted_match_ids=tuple(
            item.match_id for item in result.selected_matches
            if item.status is MatchDecisionStatus.ACCEPTED
        ),
        uncertain_match_ids=tuple(
            item.match_id for item in result.selected_matches
            if item.status is MatchDecisionStatus.UNCERTAIN
        ),
        unmatched_ids=tuple(item.unmatched_id for item in result.unmatched),
        from_type_presence=from_presence,
        to_type_presence=to_presence,
        change_event_ids=event_ids,
    )


def _invalid_matching_error(stream_id, result, config):
    error_id = f"event_error:{result.comparison_id}:invalid_matching"
    return ErrorRecord(
        record_id=error_id,
        schema_version=EVENT_ERROR_SCHEMA_VERSION,
        stream_id=stream_id,
        code="UPSTREAM_MATCHING_INVALID",
        stage="events",
        message="Change events were withheld because neighboring matching is invalid.",
        producer=config.producer,
        severity=Severity.ERROR,
        context=StageContext(pair_id=result.comparison_id),
        provenance_refs=(result.envelope.record_id,),
        upstream_warning_ids=result.envelope.warning_ids,
        upstream_error_ids=result.envelope.error_ids,
    )


def _withheld_comparison(result, grouping, config, error):
    return FrameComparison(
        envelope=RecordEnvelope(
            record_id=result.comparison_id,
            schema_version=FRAME_COMPARISON_SCHEMA_VERSION,
            stream_id=result.envelope.stream_id,
            producer=config.producer,
            context=StageContext(pair_id=result.comparison_id),
            validity_status=ValidityStatus.INVALID,
            warning_ids=tuple(sorted(set(result.envelope.warning_ids) | set(grouping.envelope.warning_ids))),
            error_ids=(error.record_id,),
            provenance_refs=(result.envelope.record_id, grouping.envelope.record_id),
        ),
        comparison_id=result.comparison_id,
        frame_pair=result.frame_pair,
        emission_status=EmissionStatus.WITHHELD,
        representation_variant_id=result.representation_variant_id,
        scorer_id=result.scorer_id,
        scorer_version=result.scorer_version,
        event_policy_id=config.event_policy_id,
        event_policy_version=config.event_policy_version,
        accepted_match_ids=(),
        uncertain_match_ids=(),
        unmatched_ids=(),
        from_type_presence={},
        to_type_presence={},
        withheld_diagnostic_ids=(error.record_id,),
    )


__all__ = [
    "CHANGE_EVENT_SCHEMA_VERSION",
    "EVENT_ERROR_SCHEMA_VERSION",
    "FRAME_COMPARISON_SCHEMA_VERSION",
    "EventGenerationBatch",
    "build_change_events",
]
