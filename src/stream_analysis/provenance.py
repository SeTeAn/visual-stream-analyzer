"""Immutable run-provenance contracts without automatic environment discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from .config import (
    ConfigValue,
    _freeze_mapping,
    _reject_analyze_only_leakage,
    _sorted_string_keys,
)
from .contracts.common import (
    _freeze_id_refs,
    _require_identifier,
    _require_version,
)

RUN_PROVENANCE_SCHEMA_ID = "stream_analysis.run_provenance.v1"

_ALGORITHM_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _empty_metadata() -> Mapping[str, ConfigValue]:
    return MappingProxyType({})


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value or value != value.strip() or any(char.isspace() for char in value):
        raise ValueError(f"{field_name} must be a non-empty token without whitespace.")
    return value


def _typed_tuple(values: Iterable[Any], expected_type: type, field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {expected_type.__name__} values.")
    frozen = tuple(values)
    if not all(isinstance(value, expected_type) for value in frozen):
        raise TypeError(f"{field_name} must contain only {expected_type.__name__} values.")
    return frozen


def _freeze_version_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    frozen: dict[str, str] = {}
    for key in _sorted_string_keys(value, field_name):
        _require_identifier(key, f"{field_name} key")
        frozen[key] = _require_version(value[key], f"{field_name}[{key}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True, kw_only=True)
class Fingerprint:
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or not _ALGORITHM_PATTERN.fullmatch(
            self.algorithm
        ):
            raise ValueError("algorithm must be a lowercase algorithm identifier.")
        value = _require_text(self.value, "value")
        if self.algorithm == "sha256":
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError("A sha256 fingerprint must contain exactly 64 hexadecimal digits.")
            value = value.lower()
        object.__setattr__(self, "value", value)


def _freeze_fingerprint_mapping(
    value: Mapping[str, Fingerprint],
    field_name: str,
) -> Mapping[str, Fingerprint]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    frozen: dict[str, Fingerprint] = {}
    for key in _sorted_string_keys(value, field_name):
        _require_identifier(key, f"{field_name} key")
        fingerprint = value[key]
        if not isinstance(fingerprint, Fingerprint):
            raise TypeError(f"{field_name} values must be Fingerprint.")
        frozen[key] = fingerprint
    return MappingProxyType(frozen)


class RunRole(str, Enum):
    PRIMARY_ANALYZE = "primary_analyze"
    DEVELOPMENT_DIAGNOSTIC = "development_diagnostic"
    FINAL_DIAGNOSTIC = "final_diagnostic"
    ABLATION = "ablation"
    PAIRED_COMPARISON = "paired_comparison"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunTimestamps:
    started_at: str
    finished_at: str

    def __post_init__(self) -> None:
        parsed: dict[str, datetime] = {}
        for field_name in ("started_at", "finished_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be an ISO-8601 string.")
            try:
                instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp.") from error
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError(f"{field_name} must include an explicit UTC offset.")
            parsed[field_name] = instant
        if parsed["finished_at"] < parsed["started_at"]:
            raise ValueError("finished_at must not precede started_at.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceProvenance:
    revision: str
    tree_fingerprint: Fingerprint

    def __post_init__(self) -> None:
        _require_text(self.revision, "revision")
        if not isinstance(self.tree_fingerprint, Fingerprint):
            raise TypeError("tree_fingerprint must be Fingerprint.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DataProvenance:
    manifest_digest: Fingerprint
    frame_content_digests: Mapping[str, Fingerprint]
    candidate_snapshot_digest: Fingerprint | None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_digest, Fingerprint):
            raise TypeError("manifest_digest must be Fingerprint.")
        object.__setattr__(
            self,
            "frame_content_digests",
            _freeze_fingerprint_mapping(
                self.frame_content_digests,
                "frame_content_digests",
            ),
        )
        if self.candidate_snapshot_digest is not None and not isinstance(
            self.candidate_snapshot_digest,
            Fingerprint,
        ):
            raise TypeError("candidate_snapshot_digest must be Fingerprint or None.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StageConfigProvenance:
    stage_id: str
    schema_id: str
    config_version: str
    semantic_digest: Fingerprint
    runtime_parameters: Mapping[str, ConfigValue] = field(
        default_factory=_empty_metadata,
        hash=False,
    )
    diagnostic_parameters: Mapping[str, ConfigValue] = field(
        default_factory=_empty_metadata,
        hash=False,
    )

    def __post_init__(self) -> None:
        _require_identifier(self.stage_id, "stage_id")
        _require_identifier(self.schema_id, "schema_id")
        _require_version(self.config_version, "config_version")
        if not isinstance(self.semantic_digest, Fingerprint):
            raise TypeError("semantic_digest must be Fingerprint.")
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProvenance:
    model_id: str
    model_version: str
    source_fingerprint: Fingerprint
    checkpoint_fingerprint: Fingerprint
    metadata: Mapping[str, ConfigValue] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, "model_id")
        _require_version(self.model_version, "model_version")
        if not isinstance(self.source_fingerprint, Fingerprint):
            raise TypeError("source_fingerprint must be Fingerprint.")
        if not isinstance(self.checkpoint_fingerprint, Fingerprint):
            raise TypeError("checkpoint_fingerprint must be Fingerprint.")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentProvenance:
    dependency_versions: Mapping[str, str]
    environment_metadata: Mapping[str, ConfigValue]
    requested_device: str
    resolved_device: str
    dtype: str
    determinism_enabled: bool
    determinism_metadata: Mapping[str, ConfigValue] = field(
        default_factory=_empty_metadata,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dependency_versions",
            _freeze_version_mapping(self.dependency_versions, "dependency_versions"),
        )
        object.__setattr__(
            self,
            "environment_metadata",
            _freeze_mapping(self.environment_metadata, "environment_metadata"),
        )
        for field_name in ("requested_device", "resolved_device", "dtype"):
            _require_text(getattr(self, field_name), field_name)
        if not isinstance(self.determinism_enabled, bool):
            raise TypeError("determinism_enabled must be bool.")
        object.__setattr__(
            self,
            "determinism_metadata",
            _freeze_mapping(self.determinism_metadata, "determinism_metadata"),
        )


def _validate_common_run(
    *,
    run_id: str,
    role: RunRole,
    timestamps: RunTimestamps,
    source: SourceProvenance,
    data: DataProvenance,
    stage_configs: Iterable[StageConfigProvenance],
    models: Iterable[ModelProvenance],
    environment: EnvironmentProvenance,
    output_schema_versions: Mapping[str, str],
) -> tuple[
    tuple[StageConfigProvenance, ...],
    tuple[ModelProvenance, ...],
    Mapping[str, str],
]:
    _require_identifier(run_id, "run_id")
    if not isinstance(role, RunRole):
        raise TypeError("role must be RunRole.")
    if not isinstance(timestamps, RunTimestamps):
        raise TypeError("timestamps must be RunTimestamps.")
    if not isinstance(source, SourceProvenance):
        raise TypeError("source must be SourceProvenance.")
    if not isinstance(data, DataProvenance):
        raise TypeError("data must be DataProvenance.")
    frozen_stages = _typed_tuple(stage_configs, StageConfigProvenance, "stage_configs")
    stage_ids = [stage.stage_id for stage in frozen_stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("stage_configs must have unique stage_id values.")
    frozen_models = _typed_tuple(models, ModelProvenance, "models")
    model_ids = [model.model_id for model in frozen_models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("models must have unique model_id values.")
    if not isinstance(environment, EnvironmentProvenance):
        raise TypeError("environment must be EnvironmentProvenance.")
    schemas = _freeze_version_mapping(output_schema_versions, "output_schema_versions")
    return (
        tuple(sorted(frozen_stages, key=lambda item: item.stage_id)),
        tuple(sorted(frozen_models, key=lambda item: item.model_id)),
        schemas,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzeRunProvenance:
    run_id: str
    role: RunRole
    timestamps: RunTimestamps
    source: SourceProvenance
    data: DataProvenance
    stage_configs: tuple[StageConfigProvenance, ...]
    models: tuple[ModelProvenance, ...]
    environment: EnvironmentProvenance
    output_schema_versions: Mapping[str, str]
    parent_run_ids: tuple[str, ...] = ()
    schema_id: str = RUN_PROVENANCE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != RUN_PROVENANCE_SCHEMA_ID:
            raise ValueError(f"schema_id must be {RUN_PROVENANCE_SCHEMA_ID!r}.")
        stages, models, schemas = _validate_common_run(
            run_id=self.run_id,
            role=self.role,
            timestamps=self.timestamps,
            source=self.source,
            data=self.data,
            stage_configs=self.stage_configs,
            models=self.models,
            environment=self.environment,
            output_schema_versions=self.output_schema_versions,
        )
        if self.role is RunRole.EVALUATION:
            raise ValueError("AnalyzeRunProvenance cannot use the evaluation role.")
        for stage in stages:
            _reject_analyze_only_leakage(
                stage.runtime_parameters,
                f"stage_configs[{stage.stage_id}].runtime_parameters",
            )
            _reject_analyze_only_leakage(
                stage.diagnostic_parameters,
                f"stage_configs[{stage.stage_id}].diagnostic_parameters",
            )
        for model in models:
            _reject_analyze_only_leakage(
                model.metadata,
                f"models[{model.model_id}].metadata",
            )
        _reject_analyze_only_leakage(
            self.environment.environment_metadata,
            "environment.environment_metadata",
        )
        _reject_analyze_only_leakage(
            self.environment.determinism_metadata,
            "environment.determinism_metadata",
        )
        _reject_analyze_only_leakage(
            self.environment.dependency_versions,
            "environment.dependency_versions",
        )
        _reject_analyze_only_leakage(schemas, "output_schema_versions")
        parents = _freeze_id_refs(self.parent_run_ids, "parent_run_ids")
        if self.run_id in parents:
            raise ValueError("parent_run_ids must not contain run_id itself.")
        object.__setattr__(self, "stage_configs", stages)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "output_schema_versions", schemas)
        object.__setattr__(self, "parent_run_ids", tuple(sorted(parents)))


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRunProvenance:
    run_id: str
    role: RunRole
    parent_prediction_run_id: str
    timestamps: RunTimestamps
    source: SourceProvenance
    data: DataProvenance
    annotation_digest: Fingerprint
    evaluation_config_digest: Fingerprint
    stage_configs: tuple[StageConfigProvenance, ...]
    models: tuple[ModelProvenance, ...]
    environment: EnvironmentProvenance
    evaluator_version: str
    output_schema_versions: Mapping[str, str]
    schema_id: str = RUN_PROVENANCE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != RUN_PROVENANCE_SCHEMA_ID:
            raise ValueError(f"schema_id must be {RUN_PROVENANCE_SCHEMA_ID!r}.")
        stages, models, schemas = _validate_common_run(
            run_id=self.run_id,
            role=self.role,
            timestamps=self.timestamps,
            source=self.source,
            data=self.data,
            stage_configs=self.stage_configs,
            models=self.models,
            environment=self.environment,
            output_schema_versions=self.output_schema_versions,
        )
        if self.role is not RunRole.EVALUATION:
            raise ValueError("EvaluationRunProvenance requires the evaluation role.")
        _require_identifier(self.parent_prediction_run_id, "parent_prediction_run_id")
        if self.parent_prediction_run_id == self.run_id:
            raise ValueError("parent_prediction_run_id must differ from run_id.")
        if not isinstance(self.annotation_digest, Fingerprint):
            raise TypeError("annotation_digest must be Fingerprint.")
        if not isinstance(self.evaluation_config_digest, Fingerprint):
            raise TypeError("evaluation_config_digest must be Fingerprint.")
        _require_version(self.evaluator_version, "evaluator_version")
        object.__setattr__(self, "stage_configs", stages)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "output_schema_versions", schemas)


__all__ = [
    "RUN_PROVENANCE_SCHEMA_ID",
    "AnalyzeRunProvenance",
    "DataProvenance",
    "EnvironmentProvenance",
    "EvaluationRunProvenance",
    "Fingerprint",
    "ModelProvenance",
    "RunRole",
    "RunTimestamps",
    "SourceProvenance",
    "StageConfigProvenance",
]
