"""Versioned configurations for matching, grouping, and change events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import StageConfig, semantic_config_digest
from ..contracts import ProducerProvenance

PAIR_SCORING_CONFIG_SCHEMA_ID = "stream_analysis.pair_scoring_config.v1"
MATCHING_CONFIG_SCHEMA_ID = "stream_analysis.matching_config.v1"
GROUPING_CONFIG_SCHEMA_ID = "stream_analysis.grouping_config.v1"
EVENT_CONFIG_SCHEMA_ID = "stream_analysis.event_config.v1"

GLOBAL_ASSIGNMENT_POLICY_ID = "augmented_global_linear_assignment_v1"
GLOBAL_ASSIGNMENT_POLICY_VERSION = "1.1.0"
GROUPING_POLICY_ID = "temporal_medoid_agglomerative_v1"
GROUPING_POLICY_VERSION = "1.3.0"
EVENT_POLICY_ID = "primary_membership_change_events_v1"
EVENT_POLICY_VERSION = "1.0.0"


def _number(value: float, name: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, infinity)"
        raise ValueError(f"{name} must be finite and in {interval}.")
    return result


def _version(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be a non-empty version token.")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be a non-empty identifier.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PairScoringConfig:
    scorer_id: str
    scorer_version: str
    visual_gate: float
    spatial_gate: float | None
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scorer_id", _identifier(self.scorer_id, "scorer_id"))
        object.__setattr__(self, "scorer_version", _version(self.scorer_version, "scorer_version"))
        object.__setattr__(self, "config_version", _version(self.config_version, "config_version"))
        object.__setattr__(self, "visual_gate", _number(self.visual_gate, "visual_gate", maximum=1.0))
        if self.spatial_gate is not None:
            object.__setattr__(
                self,
                "spatial_gate",
                _number(self.spatial_gate, "spatial_gate", maximum=1.0),
            )

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="scorer",
            schema_id=PAIR_SCORING_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters={
                "scorer_id": self.scorer_id,
                "scorer_version": self.scorer_version,
                "visual_gate": self.visual_gate,
                "spatial_gate": self.spatial_gate,
            },
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="pair_scoring",
            producer_version=self.scorer_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchingConfig:
    unmatched_pair_cost: float
    local_margin_gate: float
    global_margin_gate: float
    severe_quality_flags: tuple[str, ...]
    config_version: str = "1.0.0"
    assignment_policy_id: str = GLOBAL_ASSIGNMENT_POLICY_ID
    assignment_policy_version: str = GLOBAL_ASSIGNMENT_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_version", _version(self.config_version, "config_version"))
        object.__setattr__(self, "assignment_policy_id", _identifier(self.assignment_policy_id, "assignment_policy_id"))
        object.__setattr__(self, "assignment_policy_version", _version(self.assignment_policy_version, "assignment_policy_version"))
        object.__setattr__(self, "unmatched_pair_cost", _number(self.unmatched_pair_cost, "unmatched_pair_cost", minimum=1e-15))
        object.__setattr__(self, "local_margin_gate", _number(self.local_margin_gate, "local_margin_gate"))
        object.__setattr__(self, "global_margin_gate", _number(self.global_margin_gate, "global_margin_gate"))
        flags = tuple(sorted(set(self.severe_quality_flags)))
        if len(flags) != len(self.severe_quality_flags) or any(not flag for flag in flags):
            raise ValueError("severe_quality_flags must contain unique non-empty strings.")
        object.__setattr__(self, "severe_quality_flags", flags)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="matching",
            schema_id=MATCHING_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters={
                "assignment_policy_id": self.assignment_policy_id,
                "assignment_policy_version": self.assignment_policy_version,
                "unmatched_pair_cost": self.unmatched_pair_cost,
                "unmatched_left_cost": self.unmatched_pair_cost / 2.0,
                "unmatched_right_cost": self.unmatched_pair_cost / 2.0,
                "local_margin_gate": self.local_margin_gate,
                "global_margin_gate": self.global_margin_gate,
                "severe_quality_flags": self.severe_quality_flags,
            },
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="matching",
            producer_version=self.assignment_policy_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupingConfig:
    medoid_gate: float
    support_quantile: float
    quantile_gate: float
    support_pair_gate: float
    support_ratio_gate: float
    visual_gate: float
    medoid_weight: float
    quantile_weight: float
    support_ratio_weight: float
    second_best_margin: float
    representation_variant_id: str
    scorer_id: str
    scorer_version: str
    expose_unmerged_alternatives: bool = False
    use_uncertain_temporal_seeds: bool = True
    uncertain_seed_visual_gate: float = 0.95
    uncertain_seed_spatial_gate: float = 0.08
    coframe_aspect_log_gate: float = 0.35
    coframe_area_log_gate: float = 0.45
    coframe_fill_ratio_gate: float = 0.30
    coframe_hole_count_gate: int = 2
    coframe_visual_gate: float | None = None
    coframe_margin_bypass: bool = False
    severe_quality_flags: tuple[str, ...] = ()
    config_version: str = "1.0.0"
    grouping_policy_id: str = GROUPING_POLICY_ID
    grouping_policy_version: str = GROUPING_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_version", _version(self.config_version, "config_version"))
        object.__setattr__(
            self,
            "representation_variant_id",
            _identifier(self.representation_variant_id, "representation_variant_id"),
        )
        object.__setattr__(self, "scorer_id", _identifier(self.scorer_id, "scorer_id"))
        object.__setattr__(self, "scorer_version", _version(self.scorer_version, "scorer_version"))
        if not isinstance(self.expose_unmerged_alternatives, bool):
            raise TypeError("expose_unmerged_alternatives must be bool.")
        if not isinstance(self.use_uncertain_temporal_seeds, bool):
            raise TypeError("use_uncertain_temporal_seeds must be bool.")
        object.__setattr__(self, "grouping_policy_id", _identifier(self.grouping_policy_id, "grouping_policy_id"))
        object.__setattr__(self, "grouping_policy_version", _version(self.grouping_policy_version, "grouping_policy_version"))
        for name in (
            "medoid_gate", "support_quantile", "quantile_gate", "support_pair_gate",
            "support_ratio_gate", "visual_gate", "medoid_weight", "quantile_weight",
            "support_ratio_weight", "uncertain_seed_visual_gate", "uncertain_seed_spatial_gate",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, maximum=1.0))
        for name in ("coframe_aspect_log_gate", "coframe_area_log_gate", "coframe_fill_ratio_gate"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.coframe_visual_gate is not None:
            object.__setattr__(
                self,
                "coframe_visual_gate",
                _number(self.coframe_visual_gate, "coframe_visual_gate", maximum=1.0),
            )
        if not isinstance(self.coframe_margin_bypass, bool):
            raise TypeError("coframe_margin_bypass must be bool.")
        if self.coframe_margin_bypass and self.coframe_visual_gate is None:
            raise ValueError("coframe_margin_bypass requires coframe_visual_gate.")
        if isinstance(self.coframe_hole_count_gate, bool) or not isinstance(self.coframe_hole_count_gate, int):
            raise TypeError("coframe_hole_count_gate must be int.")
        if self.coframe_hole_count_gate < 0:
            raise ValueError("coframe_hole_count_gate must be non-negative.")
        if self.support_quantile <= 0.0:
            raise ValueError("support_quantile must be in (0, 1].")
        object.__setattr__(self, "second_best_margin", _number(self.second_best_margin, "second_best_margin"))
        flags = tuple(sorted(set(self.severe_quality_flags)))
        if len(flags) != len(self.severe_quality_flags) or any(not flag for flag in flags):
            raise ValueError("severe_quality_flags must contain unique non-empty strings.")
        object.__setattr__(self, "severe_quality_flags", flags)
        weight_sum = self.medoid_weight + self.quantile_weight + self.support_ratio_weight
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Grouping score weights must sum to 1.")

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="grouping",
            schema_id=GROUPING_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters={
                "grouping_policy_id": self.grouping_policy_id,
                "grouping_policy_version": self.grouping_policy_version,
                "representation_variant_id": self.representation_variant_id,
                "scorer_id": self.scorer_id,
                "scorer_version": self.scorer_version,
                "expose_unmerged_alternatives": self.expose_unmerged_alternatives,
                "use_uncertain_temporal_seeds": self.use_uncertain_temporal_seeds,
                "uncertain_seed_visual_gate": self.uncertain_seed_visual_gate,
                "uncertain_seed_spatial_gate": self.uncertain_seed_spatial_gate,
                "coframe_aspect_log_gate": self.coframe_aspect_log_gate,
                "coframe_area_log_gate": self.coframe_area_log_gate,
                "coframe_fill_ratio_gate": self.coframe_fill_ratio_gate,
                "coframe_hole_count_gate": self.coframe_hole_count_gate,
                "coframe_visual_gate": self.coframe_visual_gate,
                "coframe_margin_bypass": self.coframe_margin_bypass,
                "medoid_gate": self.medoid_gate,
                "support_quantile": self.support_quantile,
                "quantile_gate": self.quantile_gate,
                "support_pair_gate": self.support_pair_gate,
                "support_ratio_gate": self.support_ratio_gate,
                "visual_gate": self.visual_gate,
                "medoid_weight": self.medoid_weight,
                "quantile_weight": self.quantile_weight,
                "support_ratio_weight": self.support_ratio_weight,
                "second_best_margin": self.second_best_margin,
                "severe_quality_flags": self.severe_quality_flags,
            },
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="grouping",
            producer_version=self.grouping_policy_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EventConfig:
    position_threshold_norm: float
    severe_quality_flags: tuple[str, ...]
    config_version: str = "1.0.0"
    event_policy_id: str = EVENT_POLICY_ID
    event_policy_version: str = EVENT_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_version", _version(self.config_version, "config_version"))
        object.__setattr__(self, "event_policy_id", _identifier(self.event_policy_id, "event_policy_id"))
        object.__setattr__(self, "event_policy_version", _version(self.event_policy_version, "event_policy_version"))
        object.__setattr__(
            self,
            "position_threshold_norm",
            _number(self.position_threshold_norm, "position_threshold_norm", maximum=math.sqrt(2.0)),
        )
        flags = tuple(sorted(set(self.severe_quality_flags)))
        if len(flags) != len(self.severe_quality_flags) or any(not flag for flag in flags):
            raise ValueError("severe_quality_flags must contain unique non-empty strings.")
        object.__setattr__(self, "severe_quality_flags", flags)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="events",
            schema_id=EVENT_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters={
                "event_policy_id": self.event_policy_id,
                "event_policy_version": self.event_policy_version,
                "membership_policy": "primary_with_uncertainty_v1",
                "position_center_policy": "bbox_center_set_centroid_v1",
                "position_threshold_norm": self.position_threshold_norm,
                "severe_quality_flags": self.severe_quality_flags,
            },
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="events",
            producer_version=self.event_policy_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


__all__ = [
    "EVENT_CONFIG_SCHEMA_ID", "EVENT_POLICY_ID", "EVENT_POLICY_VERSION",
    "GLOBAL_ASSIGNMENT_POLICY_ID", "GLOBAL_ASSIGNMENT_POLICY_VERSION",
    "GROUPING_CONFIG_SCHEMA_ID", "GROUPING_POLICY_ID", "GROUPING_POLICY_VERSION",
    "MATCHING_CONFIG_SCHEMA_ID", "PAIR_SCORING_CONFIG_SCHEMA_ID",
    "EventConfig", "GroupingConfig", "MatchingConfig", "PairScoringConfig",
]
