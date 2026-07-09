"""Deterministic stream-level recurring visual type grouping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..contracts import (
    CalibrationStatus,
    CandidateRecord,
    ConfidenceMetadata,
    FrameMatchingResult,
    GroupingResult,
    MatchDecisionStatus,
    MembershipStatus,
    RecordEnvelope,
    RecurringVisualType,
    RepresentationRecord,
    StageContext,
    TypeAssignmentRecord,
    ValidityStatus,
    VisualTypeStatus,
)
from ..representations.scoring import CanonicalScorer
from .config import GroupingConfig

GROUPING_RESULT_SCHEMA_VERSION = "grouping-result-1.0"
TYPE_ASSIGNMENT_SCHEMA_VERSION = "type-assignment-1.0"
RECURRING_TYPE_SCHEMA_VERSION = "recurring-visual-type-1.0"


@dataclass(frozen=True, slots=True)
class GroupEvidence:
    left_key: str
    right_key: str
    medoid_score: float
    quantile_score: float
    support_ratio: float
    group_score: float
    structurally_eligible: bool
    coframe_overlap_count: int = 0
    coframe_geometry_compatible: bool = True
    coframe_visual_compatible: bool = True
    coframe_min_visual_score: float | None = None
    margin: float | None = None


def group_recurring_visual_types(
    *,
    stream_id: str,
    candidates: Iterable[CandidateRecord],
    representations: Iterable[RepresentationRecord],
    matching_results: Iterable[FrameMatchingResult],
    representation_variant_id: str,
    scorer: CanonicalScorer,
    config: GroupingConfig,
) -> GroupingResult:
    """Build firm temporal seeds, then reconcile components by medoid evidence."""

    if config.representation_variant_id != representation_variant_id:
        raise ValueError("grouping config representation variant does not match request.")
    if config.scorer_id != scorer.scorer_id or config.scorer_version != scorer.scorer_version:
        raise ValueError("grouping config scorer identity does not match scorer adapter.")

    candidate_values = tuple(candidates)
    candidate_by_id = {item.candidate_id: item for item in candidate_values}
    if len(candidate_by_id) != len(candidate_values):
        raise ValueError("candidates must be unique by candidate_id.")
    representation_values = tuple(representations)
    representation_by_id = {item.candidate_id: item for item in representation_values}
    if len(representation_by_id) != len(representation_values):
        raise ValueError("representations must be unique by candidate_id.")
    valid_ids = tuple(
        sorted(
            candidate_id
            for candidate_id, record in representation_by_id.items()
            if candidate_id in candidate_by_id
            and record.envelope.validity_status is ValidityStatus.VALID
            and record.payload is not None
            and record.input_variant == representation_variant_id
        )
    )
    if any(representation_by_id[candidate_id].family is not scorer.family for candidate_id in valid_ids):
        raise ValueError("Grouping representation family is incompatible with scorer.")
    if any(candidate_by_id[candidate_id].envelope.stream_id != stream_id for candidate_id in valid_ids):
        raise ValueError("All grouping candidates must belong to stream_id.")

    score_cache: dict[tuple[str, str], float] = {}

    def score(left_id: str, right_id: str) -> float:
        if left_id == right_id:
            return 1.0
        key = tuple(sorted((left_id, right_id)))
        if key not in score_cache:
            score_cache[key] = scorer.score(
                representation_by_id[key[0]], representation_by_id[key[1]]
            ).visual_score
        return score_cache[key]

    parent = {candidate_id: candidate_id for candidate_id in valid_ids}

    def find(candidate_id: str) -> str:
        while parent[candidate_id] != candidate_id:
            parent[candidate_id] = parent[parent[candidate_id]]
            candidate_id = parent[candidate_id]
        return candidate_id

    def union(left_id: str, right_id: str) -> None:
        left_root, right_root = find(left_id), find(right_id)
        if left_root == right_root:
            return
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep

    seed_match_ids: dict[str, set[str]] = {candidate_id: set() for candidate_id in valid_ids}
    weak_seed_match_ids: dict[str, set[str]] = {candidate_id: set() for candidate_id in valid_ids}
    result_values = tuple(matching_results)
    for result in result_values:
        if result.envelope.stream_id != stream_id:
            raise ValueError("matching_results must belong to stream_id.")
        if (
            result.scorer_id != scorer.scorer_id
            or result.scorer_version != scorer.scorer_version
        ):
            raise ValueError("matching_results and grouping scorer identities must agree.")
        if result.representation_variant_id != representation_variant_id:
            raise ValueError("matching_results and grouping representation variants must agree.")
        for match in result.selected_matches:
            is_accepted_seed = match.status is MatchDecisionStatus.ACCEPTED
            is_weak_seed = (
                match.status is MatchDecisionStatus.UNCERTAIN
                and _uncertain_temporal_seed_allowed(match, candidate_by_id, config)
            )
            if not is_accepted_seed and not is_weak_seed:
                continue
            left_id = match.from_endpoint.candidate_id
            right_id = match.to_endpoint.candidate_id
            if left_id in parent and right_id in parent:
                union(left_id, right_id)
                seed_match_ids[left_id].add(match.match_id)
                seed_match_ids[right_id].add(match.match_id)
                if is_weak_seed:
                    weak_seed_match_ids[left_id].add(match.match_id)
                    weak_seed_match_ids[right_id].add(match.match_id)

    components = _components_from_parent(valid_ids, find)
    component_seed_ids = {
        key: set().union(*(seed_match_ids[item] for item in members))
        for key, members in components.items()
    }
    component_weak_seed_ids = {
        key: set().union(*(weak_seed_match_ids[item] for item in members))
        for key, members in components.items()
    }
    component_merge_evidence: dict[str, list[GroupEvidence]] = {
        key: [] for key in components
    }

    while True:
        evidence = _all_evidence(components, score, config, candidate_by_id)
        eligible = _with_margins(evidence, config)
        mergeable = [
            item
            for item in eligible
            if item.structurally_eligible
            and item.margin is not None
            and _merge_margin_passes(item, config)
        ]
        if not mergeable:
            break
        chosen = sorted(
            mergeable,
            key=lambda item: (-item.group_score, -float(item.margin or 0.0), item.left_key, item.right_key),
        )[0]
        merged_members = components.pop(chosen.left_key) | components.pop(chosen.right_key)
        merge_history = (
            component_merge_evidence.pop(chosen.left_key)
            + component_merge_evidence.pop(chosen.right_key)
            + [chosen]
        )
        new_key = min(merged_members)
        components[new_key] = merged_members
        component_merge_evidence[new_key] = merge_history
        seeds = component_seed_ids.pop(chosen.left_key) | component_seed_ids.pop(chosen.right_key)
        component_seed_ids[new_key] = seeds
        weak_seeds = component_weak_seed_ids.pop(chosen.left_key) | component_weak_seed_ids.pop(chosen.right_key)
        component_weak_seed_ids[new_key] = weak_seeds

    final_evidence = _with_margins(_all_evidence(components, score, config, candidate_by_id), config)
    alternatives_by_component: dict[str, set[str]] = {key: set() for key in components}
    evidence_by_component: dict[str, list[GroupEvidence]] = {key: [] for key in components}
    for item in final_evidence:
        if item.structurally_eligible:
            alternatives_by_component[item.left_key].add(item.right_key)
            alternatives_by_component[item.right_key].add(item.left_key)
            evidence_by_component[item.left_key].append(item)
            evidence_by_component[item.right_key].append(item)

    ordered_keys = sorted(
        components,
        key=lambda key: (
            min(candidate_by_id[item].frame_index for item in components[key]),
            min(components[key]),
        ),
    )
    type_id_by_key = {key: f"type_{index:03d}" for index, key in enumerate(ordered_keys, start=1)}
    assignments: list[TypeAssignmentRecord] = []
    recurring_types: list[RecurringVisualType] = []
    all_warning_ids: set[str] = set()
    all_warning_ids.update(
        warning_id for result in result_values for warning_id in result.envelope.warning_ids
    )

    for key in ordered_keys:
        members = tuple(sorted(components[key]))
        medoid = _medoid(members, score)
        diagnostic_alternatives = tuple(
            sorted(type_id_by_key[item] for item in alternatives_by_component[key])
        )
        alternatives = diagnostic_alternatives if config.expose_unmerged_alternatives else ()
        seeds = tuple(sorted(component_seed_ids.get(key, set())))
        weak_seeds = tuple(sorted(component_weak_seed_ids.get(key, set())))
        has_recurrence = len(members) > 1
        severe_quality = {
            flag
            for candidate_id in members
            for flag in (
                candidate_by_id[candidate_id].quality_flags
                + representation_by_id[candidate_id].input_quality_metadata.quality_flags
            )
            if flag in config.severe_quality_flags
        }
        if has_recurrence and (alternatives or severe_quality):
            membership_status = MembershipStatus.UNCERTAIN
            type_status = VisualTypeStatus.UNCERTAIN
        elif has_recurrence:
            membership_status = MembershipStatus.FIRM
            type_status = VisualTypeStatus.ESTABLISHED
        else:
            membership_status = MembershipStatus.PROVISIONAL
            type_status = VisualTypeStatus.PROVISIONAL
        type_id = type_id_by_key[key]
        warnings = _ordered_unique(
            warning_id
            for candidate_id in members
            for warning_id in (
                candidate_by_id[candidate_id].envelope.warning_ids
                + representation_by_id[candidate_id].envelope.warning_ids
            )
        )
        all_warning_ids.update(warnings)
        internal_scores = [
            score(left_id, right_id)
            for index, left_id in enumerate(members)
            for right_id in members[index + 1 :]
        ]
        internal_summary = {
            "member_count": len(members),
            "minimum_similarity": min(internal_scores) if internal_scores else 1.0,
            "mean_similarity": sum(internal_scores) / len(internal_scores) if internal_scores else 1.0,
                "medoid_mean_similarity": sum(score(medoid, item) for item in members) / len(members),
                "severe_quality_flag_count": len(severe_quality),
        }
        frame_ids = sorted(
            {candidate_by_id[item].frame_id for item in members},
            key=lambda frame_id: (
                min(
                    candidate_by_id[item].frame_index
                    for item in members
                    if candidate_by_id[item].frame_id == frame_id
                ),
                frame_id,
            ),
        )
        instances = {
            frame_id: tuple(sorted(item for item in members if candidate_by_id[item].frame_id == frame_id))
            for frame_id in frame_ids
        }
        confidence = ConfidenceMetadata(
            value=None,
            calibration_status=CalibrationStatus.NOT_CALIBRATED,
            components={
                "internal_minimum_similarity": internal_summary["minimum_similarity"],
                "internal_mean_similarity": internal_summary["mean_similarity"],
            },
        )
        recurring_types.append(
            RecurringVisualType(
                envelope=RecordEnvelope(
                    record_id=type_id,
                    schema_version=RECURRING_TYPE_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=config.producer,
                    context=StageContext(type_id=type_id),
                    warning_ids=warnings,
                    provenance_refs=_ordered_unique((*members, *seeds)),
                ),
                type_id=type_id,
                representation_variant_id=representation_variant_id,
                grouping_policy_id=config.grouping_policy_id,
                grouping_policy_version=config.grouping_policy_version,
                member_candidate_ids=members,
                instances_by_frame=instances,
                count_by_frame={frame_id: len(frame_members) for frame_id, frame_members in instances.items()},
                representative_policy="medoid",
                representative_candidate_id=medoid,
                seed_match_ids=seeds,
                presence_summary={
                    "first_frame_index": min(candidate_by_id[item].frame_index for item in members),
                    "last_frame_index": max(candidate_by_id[item].frame_index for item in members),
                    "present_frame_ids": tuple(frame_ids),
                },
                internal_similarity_summary=internal_summary,
                confidence=confidence,
                status=type_status,
                alternative_type_ids=alternatives,
            )
        )
        evidence_values = evidence_by_component[key]
        merge_values = component_merge_evidence[key]
        best_group = max((item.group_score for item in evidence_values), default=0.0)
        best_margin = max((float(item.margin or 0.0) for item in evidence_values), default=0.0)
        merge_scores = (
            {
                "merge_min_medoid_score": min(item.medoid_score for item in merge_values),
                "merge_min_quantile_score": min(item.quantile_score for item in merge_values),
                "merge_min_support_ratio": min(item.support_ratio for item in merge_values),
                "merge_min_group_score": min(item.group_score for item in merge_values),
                "merge_min_group_margin": min(float(item.margin or 0.0) for item in merge_values),
                "merge_max_coframe_overlap_count": max(item.coframe_overlap_count for item in merge_values),
                "merge_min_coframe_visual_score": min(
                    (
                        item.coframe_min_visual_score
                        for item in merge_values
                        if item.coframe_min_visual_score is not None
                    ),
                    default=1.0,
                ),
            }
            if merge_values else {}
        )
        for candidate_id in members:
            assignment_id = f"type_assignment:{candidate_id}"
            assignments.append(
                TypeAssignmentRecord(
                    envelope=RecordEnvelope(
                        record_id=assignment_id,
                        schema_version=TYPE_ASSIGNMENT_SCHEMA_VERSION,
                        stream_id=stream_id,
                        producer=config.producer,
                        context=StageContext(
                            frame_id=candidate_by_id[candidate_id].frame_id,
                            candidate_id=candidate_id,
                            type_id=type_id,
                        ),
                        warning_ids=warnings,
                        provenance_refs=_ordered_unique(
                            (candidate_id, representation_by_id[candidate_id].envelope.record_id, *seeds)
                        ),
                    ),
                    assignment_id=assignment_id,
                    candidate_id=candidate_id,
                    frame_id=candidate_by_id[candidate_id].frame_id,
                    primary_type_id=type_id,
                    membership_status=membership_status,
                    evidence_kind=(
                        "accepted_neighbor_component" if seeds and len(weak_seeds) < len(seeds) else
                        "weak_neighbor_component" if weak_seeds else
                        "robust_group_merge" if has_recurrence else
                        "standalone_component"
                    ),
                    scores={
                        "diagnostic_alternative_type_count": len(diagnostic_alternatives),
                        "weak_temporal_seed_count": len(weak_seeds),
                        "best_alternative_group_margin": best_margin,
                        "best_alternative_group_score": best_group,
                        "similarity_to_medoid": score(candidate_id, medoid),
                        **merge_scores,
                    },
                    confidence=confidence,
                    alternative_type_ids=alternatives,
                )
            )

    grouping_id = f"grouping_result:{stream_id}:{representation_variant_id}"
    return GroupingResult(
        envelope=RecordEnvelope(
            record_id=grouping_id,
            schema_version=GROUPING_RESULT_SCHEMA_VERSION,
            stream_id=stream_id,
            producer=config.producer,
            context=StageContext(),
            warning_ids=tuple(sorted(all_warning_ids)),
            provenance_refs=_ordered_unique(
                tuple(valid_ids) + tuple(result.envelope.record_id for result in result_values)
            ),
        ),
        representation_variant_id=representation_variant_id,
        grouping_policy_id=config.grouping_policy_id,
        grouping_policy_version=config.grouping_policy_version,
        candidate_ids=valid_ids,
        assignments=tuple(assignments),
        recurring_types=tuple(recurring_types),
    )


def _components_from_parent(candidate_ids: tuple[str, ...], find) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for candidate_id in candidate_ids:
        values.setdefault(find(candidate_id), set()).add(candidate_id)
    return {min(members): members for members in values.values()}


def _medoid(members: tuple[str, ...], score) -> str:
    return min(
        members,
        key=lambda candidate_id: (
            sum(1.0 - score(candidate_id, other_id) for other_id in members),
            candidate_id,
        ),
    )


def _all_evidence(
    components: dict[str, set[str]],
    score,
    config: GroupingConfig,
    candidate_by_id: dict[str, CandidateRecord],
) -> list[GroupEvidence]:
    keys = sorted(components)
    medoids = {key: _medoid(tuple(sorted(components[key])), score) for key in keys}
    result: list[GroupEvidence] = []
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1 :]:
            cross_scores = sorted(
                score(left_id, right_id)
                for left_id in components[left_key]
                for right_id in components[right_key]
            )
            quantile_index = max(0, math.ceil(config.support_quantile * len(cross_scores)) - 1)
            quantile_score = cross_scores[quantile_index]
            support_ratio = sum(value >= config.support_pair_gate for value in cross_scores) / len(cross_scores)
            medoid_score = score(medoids[left_key], medoids[right_key])
            group_score = (
                config.medoid_weight * medoid_score
                + config.quantile_weight * quantile_score
                + config.support_ratio_weight * support_ratio
            )
            (
                coframe_overlap_count,
                coframe_geometry_compatible,
                coframe_visual_compatible,
                coframe_min_visual_score,
            ) = _coframe_compatibility_evidence(
                components[left_key],
                components[right_key],
                candidate_by_id,
                config,
                score,
            )
            structurally_eligible = (
                medoid_score >= config.medoid_gate
                and quantile_score >= config.quantile_gate
                and support_ratio >= config.support_ratio_gate
                and group_score >= config.visual_gate
                and coframe_geometry_compatible
                and coframe_visual_compatible
            )
            result.append(
                GroupEvidence(
                    left_key=left_key,
                    right_key=right_key,
                    medoid_score=medoid_score,
                    quantile_score=quantile_score,
                    support_ratio=support_ratio,
                    group_score=group_score,
                    structurally_eligible=structurally_eligible,
                    coframe_overlap_count=coframe_overlap_count,
                    coframe_geometry_compatible=coframe_geometry_compatible,
                    coframe_visual_compatible=coframe_visual_compatible,
                    coframe_min_visual_score=coframe_min_visual_score,
                )
            )
    return result


def _with_margins(evidence: list[GroupEvidence], config: GroupingConfig) -> list[GroupEvidence]:
    eligible = [item for item in evidence if item.structurally_eligible]
    result: list[GroupEvidence] = []
    for item in evidence:
        if not item.structurally_eligible:
            result.append(item)
            continue
        competitors = [
            candidate.group_score
            for candidate in eligible
            if candidate is not item
            and (
                item.left_key in (candidate.left_key, candidate.right_key)
                or item.right_key in (candidate.left_key, candidate.right_key)
            )
        ]
        alternative = max([config.visual_gate, *competitors])
        result.append(
            GroupEvidence(
                left_key=item.left_key,
                right_key=item.right_key,
                medoid_score=item.medoid_score,
                quantile_score=item.quantile_score,
                support_ratio=item.support_ratio,
                group_score=item.group_score,
                structurally_eligible=True,
                coframe_overlap_count=item.coframe_overlap_count,
                coframe_geometry_compatible=item.coframe_geometry_compatible,
                coframe_visual_compatible=item.coframe_visual_compatible,
                coframe_min_visual_score=item.coframe_min_visual_score,
                margin=item.group_score - alternative,
            )
        )
    return result


def _merge_margin_passes(item: GroupEvidence, config: GroupingConfig) -> bool:
    if (
        config.coframe_margin_bypass
        and item.coframe_overlap_count > 0
        and item.coframe_visual_compatible
    ):
        return True
    assert item.margin is not None
    return item.margin >= config.second_best_margin


def _uncertain_temporal_seed_allowed(match, candidate_by_id: dict[str, CandidateRecord], config: GroupingConfig) -> bool:
    if not config.use_uncertain_temporal_seeds:
        return False
    left_id = match.from_endpoint.candidate_id
    right_id = match.to_endpoint.candidate_id
    if left_id not in candidate_by_id or right_id not in candidate_by_id:
        return False
    return (
        match.visual_score >= config.uncertain_seed_visual_gate
        and match.spatial_evidence.normalized_distance <= config.uncertain_seed_spatial_gate
        and _geometry_compatible(candidate_by_id[left_id], candidate_by_id[right_id], config)
    )


def _coframe_compatibility_evidence(
    left_members: set[str],
    right_members: set[str],
    candidate_by_id: dict[str, CandidateRecord],
    config: GroupingConfig,
    score,
) -> tuple[int, bool, bool, float | None]:
    overlap_count = 0
    geometry_compatible = True
    coframe_scores: list[float] = []
    for left_id in left_members:
        left = candidate_by_id[left_id]
        for right_id in right_members:
            right = candidate_by_id[right_id]
            if left.frame_id != right.frame_id:
                continue
            overlap_count += 1
            if not _geometry_compatible(left, right, config):
                geometry_compatible = False
            if config.coframe_visual_gate is not None:
                coframe_scores.append(score(left_id, right_id))
    minimum_score = min(coframe_scores) if coframe_scores else None
    visual_compatible = (
        config.coframe_visual_gate is None
        or minimum_score is None
        or minimum_score >= config.coframe_visual_gate
    )
    return overlap_count, geometry_compatible, visual_compatible, minimum_score


def _geometry_compatible(left: CandidateRecord, right: CandidateRecord, config: GroupingConfig) -> bool:
    return (
        _abs_log_ratio(_aspect_ratio(left), _aspect_ratio(right)) <= config.coframe_aspect_log_gate
        and _abs_log_ratio(_area(left), _area(right)) <= config.coframe_area_log_gate
        and abs(_foreground_fill_ratio(left) - _foreground_fill_ratio(right)) <= config.coframe_fill_ratio_gate
        and abs(_hole_count(left) - _hole_count(right)) <= config.coframe_hole_count_gate
    )


def _aspect_ratio(candidate: CandidateRecord) -> float:
    width = max(float(candidate.bbox.width), 1e-12)
    height = max(float(candidate.bbox.height), 1e-12)
    return width / height


def _area(candidate: CandidateRecord) -> float:
    return max(float(candidate.bbox.width) * float(candidate.bbox.height), 1e-12)


def _foreground_fill_ratio(candidate: CandidateRecord) -> float:
    value = candidate.geometry.values.get("foreground_fill_ratio") if candidate.geometry else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _hole_count(candidate: CandidateRecord) -> int:
    value = candidate.geometry.values.get("hole_count") if candidate.geometry else None
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _abs_log_ratio(left: float, right: float) -> float:
    return abs(math.log(max(left, 1e-12) / max(right, 1e-12)))


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "GROUPING_RESULT_SCHEMA_VERSION",
    "RECURRING_TYPE_SCHEMA_VERSION",
    "TYPE_ASSIGNMENT_SCHEMA_VERSION",
    "GroupEvidence",
    "group_recurring_visual_types",
]
