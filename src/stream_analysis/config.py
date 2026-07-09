"""Versioned in-memory configuration contracts and deterministic digests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from .contracts.common import _require_identifier, _require_version

ANALYSIS_CONFIG_SCHEMA_ID = "stream_analysis.analysis_config.v1"
EVALUATION_CONFIG_SCHEMA_ID = "stream_analysis.evaluation_config.v1"
CANONICAL_SEMANTIC_ENCODING_ID = "stream_analysis.canonical_semantic_json.v1"


ConfigScalar: TypeAlias = None | str | bool | int | float | Enum
ConfigValue: TypeAlias = ConfigScalar | tuple["ConfigValue", ...] | Mapping[str, "ConfigValue"]


def _empty_mapping() -> Mapping[str, ConfigValue]:
    return MappingProxyType({})


def _sorted_string_keys(value: Mapping[Any, Any], path: str) -> tuple[str, ...]:
    keys = tuple(value.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{path} keys must be non-empty strings.")
    return tuple(sorted(keys, key=lambda item: item.encode("utf-8")))


def _freeze_config_value(value: Any, path: str) -> ConfigValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if isinstance(value, Enum):
        _freeze_config_value(value.value, f"{path}.value")
        return value
    if isinstance(value, Mapping):
        frozen = {
            key: _freeze_config_value(value[key], f"{path}.{key}")
            for key in _sorted_string_keys(value, path)
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_config_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} must not contain unordered set values.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{path} must not contain byte strings.")
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


def _freeze_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, ConfigValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    frozen = _freeze_config_value(value, path)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return frozen


_ANALYZE_FORBIDDEN_KEY_TOKENS = {
    "annotation",
    "annotationdigest",
    "annotationpath",
    "datarole",
    "datasetrole",
    "evaluationmetric",
    "evaluationmetrics",
    "evaluatormappingpolicy",
    "finaldatarole",
    "gt",
    "gtlabel",
    "gtlabels",
    "gtmapping",
    "mappingpolicy",
    "metrics",
}


def _key_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reject_analyze_only_leakage(value: ConfigValue, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            token = _key_token(key)
            if (
                token in _ANALYZE_FORBIDDEN_KEY_TOKENS
                or token.startswith("annotation")
                or token.startswith("evaluation")
                or token.startswith("evaluator")
                or token.startswith("groundtruth")
                or token.startswith("gt")
                or token.endswith("datarole")
                or token.endswith("datasetrole")
                or token.startswith("metrics")
                or token.endswith("metrics")
                or token.startswith("mappingpolicy")
                or token.endswith("mappingpolicy")
            ):
                raise ValueError(f"{path} must not contain evaluator-only key {key!r}.")
            _reject_analyze_only_leakage(nested, f"{path}.{key}")
    elif isinstance(value, tuple):
        for index, nested in enumerate(value):
            _reject_analyze_only_leakage(nested, f"{path}[{index}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class StageConfig:
    stage_id: str
    schema_id: str
    config_version: str
    semantic_parameters: Mapping[str, ConfigValue] = field(
        default_factory=_empty_mapping,
        hash=False,
    )
    runtime_parameters: Mapping[str, ConfigValue] = field(
        default_factory=_empty_mapping,
        hash=False,
    )
    diagnostic_parameters: Mapping[str, ConfigValue] = field(
        default_factory=_empty_mapping,
        hash=False,
    )

    def __post_init__(self) -> None:
        _require_identifier(self.stage_id, "stage_id")
        _require_identifier(self.schema_id, "schema_id")
        _require_version(self.config_version, "config_version")
        object.__setattr__(
            self,
            "semantic_parameters",
            _freeze_mapping(self.semantic_parameters, "semantic_parameters"),
        )
        object.__setattr__(
            self,
            "runtime_parameters",
            _freeze_mapping(self.runtime_parameters, "runtime_parameters"),
        )
        object.__setattr__(
            self,
            "diagnostic_parameters",
            _freeze_mapping(self.diagnostic_parameters, "diagnostic_parameters"),
        )


_ANALYSIS_STAGE_FIELDS = (
    ("stream_input", "stream_input"),
    ("candidate_extraction", "candidate_extraction"),
    ("representation", "representation"),
    ("scorer", "scorer"),
    ("matching", "matching"),
    ("grouping", "grouping"),
    ("events", "events"),
    ("reporting", "reporting"),
    ("runtime_policy", "runtime_policy"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisRunConfig:
    config_version: str
    stream_input: StageConfig
    candidate_extraction: StageConfig
    representation: StageConfig
    scorer: StageConfig
    matching: StageConfig
    grouping: StageConfig
    events: StageConfig
    reporting: StageConfig
    runtime_policy: StageConfig
    schema_id: str = ANALYSIS_CONFIG_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != ANALYSIS_CONFIG_SCHEMA_ID:
            raise ValueError(f"schema_id must be {ANALYSIS_CONFIG_SCHEMA_ID!r}.")
        _require_version(self.config_version, "config_version")
        for field_name, expected_stage_id in _ANALYSIS_STAGE_FIELDS:
            stage_config = getattr(self, field_name)
            if not isinstance(stage_config, StageConfig):
                raise TypeError(f"{field_name} must be StageConfig.")
            if stage_config.stage_id != expected_stage_id:
                raise ValueError(
                    f"{field_name}.stage_id must be {expected_stage_id!r}."
                )
            for parameter_axis in (
                "semantic_parameters",
                "runtime_parameters",
                "diagnostic_parameters",
            ):
                _reject_analyze_only_leakage(
                    getattr(stage_config, parameter_axis),
                    f"{field_name}.{parameter_axis}",
                )

    @property
    def stage_configs(self) -> tuple[StageConfig, ...]:
        return tuple(getattr(self, field_name) for field_name, _ in _ANALYSIS_STAGE_FIELDS)


class EvaluationDataRole(str, Enum):
    SMOKE_UNIT = "smoke_unit"
    PROBE_DEVELOPMENT = "probe_development"
    FINAL_HELD_OUT = "final_held_out"


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationPolicy:
    policy_id: str
    policy_version: str
    parameters: Mapping[str, ConfigValue] = field(default_factory=_empty_mapping, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        _require_version(self.policy_version, "policy_version")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationConfig:
    config_version: str
    metric_policy: EvaluationPolicy
    invalid_status_policy: EvaluationPolicy
    mapping_policy: EvaluationPolicy
    data_role: EvaluationDataRole
    selection_freeze_metadata: Mapping[str, ConfigValue]
    artifact_policy: EvaluationPolicy
    schema_id: str = EVALUATION_CONFIG_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != EVALUATION_CONFIG_SCHEMA_ID:
            raise ValueError(f"schema_id must be {EVALUATION_CONFIG_SCHEMA_ID!r}.")
        _require_version(self.config_version, "config_version")
        for field_name in (
            "metric_policy",
            "invalid_status_policy",
            "mapping_policy",
            "artifact_policy",
        ):
            if not isinstance(getattr(self, field_name), EvaluationPolicy):
                raise TypeError(f"{field_name} must be EvaluationPolicy.")
        if not isinstance(self.data_role, EvaluationDataRole):
            raise TypeError("data_role must be EvaluationDataRole.")
        object.__setattr__(
            self,
            "selection_freeze_metadata",
            _freeze_mapping(
                self.selection_freeze_metadata,
                "selection_freeze_metadata",
            ),
        )


def _canonical_node(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, Enum):
        return _canonical_node(value.value)
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical values must not contain NaN or infinity.")
        normalized = 0.0 if value == 0.0 else value
        return ["float", normalized.hex()]
    if isinstance(value, Mapping):
        return [
            "mapping",
            [
                [key, _canonical_node(value[key])]
                for key in _sorted_string_keys(value, "canonical mapping")
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical_node(item) for item in value]]
    raise TypeError(f"Canonical encoding does not support {type(value).__name__}.")


def _stage_semantic_payload(config: StageConfig) -> Mapping[str, Any]:
    return {
        "config_version": config.config_version,
        "schema_id": config.schema_id,
        "semantic_parameters": config.semantic_parameters,
        "stage_id": config.stage_id,
    }


def _analysis_semantic_payload(config: AnalysisRunConfig) -> Mapping[str, Any]:
    return {
        "config_version": config.config_version,
        "schema_id": config.schema_id,
        "stages": tuple(_stage_semantic_payload(stage) for stage in config.stage_configs),
    }


def _evaluation_payload(config: EvaluationConfig) -> Mapping[str, Any]:
    def policy_payload(policy: EvaluationPolicy) -> Mapping[str, Any]:
        return {
            "parameters": policy.parameters,
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
        }

    return {
        "artifact_policy": policy_payload(config.artifact_policy),
        "config_version": config.config_version,
        "data_role": config.data_role,
        "invalid_status_policy": policy_payload(config.invalid_status_policy),
        "mapping_policy": policy_payload(config.mapping_policy),
        "metric_policy": policy_payload(config.metric_policy),
        "schema_id": config.schema_id,
        "selection_freeze_metadata": config.selection_freeze_metadata,
    }


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    document = {
        "encoding": CANONICAL_SEMANTIC_ENCODING_ID,
        "payload": _canonical_node(payload),
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_semantic_bytes(config: StageConfig | AnalysisRunConfig) -> bytes:
    """Return canonical UTF-8 bytes with no trailing newline."""

    if isinstance(config, StageConfig):
        return _canonical_bytes(_stage_semantic_payload(config))
    if isinstance(config, AnalysisRunConfig):
        return _canonical_bytes(_analysis_semantic_payload(config))
    raise TypeError("config must be StageConfig or AnalysisRunConfig.")


def semantic_config_digest(config: StageConfig | AnalysisRunConfig) -> str:
    return f"sha256:{hashlib.sha256(canonical_semantic_bytes(config)).hexdigest()}"


def canonical_evaluation_bytes(config: EvaluationConfig) -> bytes:
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be EvaluationConfig.")
    return _canonical_bytes(_evaluation_payload(config))


def evaluation_config_digest(config: EvaluationConfig) -> str:
    return f"sha256:{hashlib.sha256(canonical_evaluation_bytes(config)).hexdigest()}"


__all__ = [
    "ANALYSIS_CONFIG_SCHEMA_ID",
    "CANONICAL_SEMANTIC_ENCODING_ID",
    "EVALUATION_CONFIG_SCHEMA_ID",
    "AnalysisRunConfig",
    "EvaluationConfig",
    "EvaluationDataRole",
    "EvaluationPolicy",
    "StageConfig",
    "canonical_evaluation_bytes",
    "canonical_semantic_bytes",
    "evaluation_config_digest",
    "semantic_config_digest",
]
