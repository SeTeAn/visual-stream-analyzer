"""Deterministic in-memory JSON boundary for supported immutable contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, TypeAlias

from .contracts import (
    ArtifactReference,
    ChangeEvent,
    ConfidenceMetadata,
    EventEvidence,
    FrameComparison,
    FramePair,
    ImageSize,
    ProducerProvenance,
    RecordEnvelope,
    RepresentationSummary,
    RuntimeMetadata,
    StageContext,
    StreamAnalysisResult,
    TypePresenceSummary,
    VersionedMetadata,
)

PRIMARY_STREAM_RESULT_SCHEMA_ID = "stream_analysis.primary_stream_result.v1"

JsonScalar: TypeAlias = None | str | bool | int | float
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_SUPPORTED_MODULE_PREFIXES = (
    "stream_analysis.config",
    "stream_analysis.contracts.",
    "stream_analysis.provenance",
)

_PRIMARY_FIELD_WHITELIST: Mapping[type, frozenset[str]] = {
    StreamAnalysisResult: frozenset(
        {
            "envelope",
            "run_id",
            "pipeline_version",
            "run_status",
            "frame_ids",
            "data_provenance",
            "model_provenance",
            "runtime_summary",
            "candidate_extraction_result_id",
            "candidate_record_ids",
            "candidate_manifest_ref",
            "representation_summaries",
            "frame_matching_result_ids",
            "grouping_result_id",
            "recurring_type_ids",
            "type_assignment_ids",
            "frame_comparisons",
            "change_events",
            "artifacts",
            "status_summary",
        }
    ),
    RecordEnvelope: frozenset(
        {
            "record_id",
            "schema_version",
            "stream_id",
            "producer",
            "context",
            "validity_status",
            "warning_ids",
            "error_ids",
            "provenance_refs",
        }
    ),
    ProducerProvenance: frozenset(
        {"producer_stage", "producer_version", "config_version", "config_digest"}
    ),
    StageContext: frozenset(
        {"frame_id", "candidate_id", "pair_id", "type_id", "event_id"}
    ),
    VersionedMetadata: frozenset({"identifier", "version", "details"}),
    RuntimeMetadata: frozenset({"runtime_id", "details"}),
    RepresentationSummary: frozenset(
        {
            "representation_record_id",
            "candidate_id",
            "frame_id",
            "family",
            "representation_type",
            "representation_version",
            "input_variant",
            "semantic_config_digest",
            "validity_status",
            "embedding_dimension",
            "feature_group_names",
            "warning_ids",
        }
    ),
    FrameComparison: frozenset(
        {
            "envelope",
            "comparison_id",
            "frame_pair",
            "emission_status",
            "representation_variant_id",
            "scorer_id",
            "scorer_version",
            "event_policy_id",
            "event_policy_version",
            "accepted_match_ids",
            "uncertain_match_ids",
            "unmatched_ids",
            "from_type_presence",
            "to_type_presence",
            "change_event_ids",
            "withheld_diagnostic_ids",
        }
    ),
    FramePair: frozenset(
        {
            "from_frame_id",
            "from_frame_index",
            "from_frame_size",
            "to_frame_id",
            "to_frame_index",
            "to_frame_size",
        }
    ),
    ImageSize: frozenset({"width", "height"}),
    TypePresenceSummary: frozenset(
        {"type_id", "candidate_ids", "firm_count", "possible_count"}
    ),
    ChangeEvent: frozenset(
        {
            "envelope",
            "event_id",
            "comparison_id",
            "frame_pair",
            "kind",
            "predicted_type_id",
            "status",
            "evidence",
            "confidence",
            "policy_id",
            "policy_version",
        }
    ),
    EventEvidence: frozenset(
        {
            "from_count",
            "to_count",
            "from_firm_count",
            "to_firm_count",
            "from_member_ids",
            "to_member_ids",
            "accepted_match_ids",
            "uncertain_match_ids",
            "position_shift_norm",
            "details",
        }
    ),
    ConfidenceMetadata: frozenset({"value", "calibration_status", "components"}),
    ArtifactReference: frozenset(
        {"artifact_id", "artifact_kind", "reference", "producer_record_id", "metadata"}
    ),
}

_PRIMARY_METADATA_FIELDS = {
    (StreamAnalysisResult, "status_summary"),
    (VersionedMetadata, "details"),
    (RuntimeMetadata, "details"),
    (EventEvidence, "details"),
    (ArtifactReference, "metadata"),
}

_COMPACT_SCALAR_KEY_TOKENS = {
    "artifactformat",
    "calibrationstatus",
    "candidatecount",
    "checkpoint",
    "checkpointdigest",
    "checkpointfingerprint",
    "comparisoncount",
    "deterministic",
    "device",
    "dtype",
    "durationms",
    "elapsedms",
    "errorcount",
    "eventcount",
    "framecount",
    "height",
    "manifest",
    "manifestdigest",
    "matchcount",
    "message",
    "mimetype",
    "modelname",
    "modelversion",
    "normalization",
    "normalized",
    "positionthreshold",
    "precisevalue",
    "predicate",
    "provider",
    "providerversion",
    "reason",
    "reasoncode",
    "requesteddevice",
    "resolveddevice",
    "sourcerevision",
    "sourcefingerprint",
    "status",
    "streamid",
    "streamrole",
    "threshold",
    "typecount",
    "unmatchedcount",
    "warningcount",
    "width",
}

_COMPACT_POINT_KEY_TOKENS = {"normalizedfromcenter", "normalizedtocenter"}
_COMPACT_INTEGER_SEQUENCE_KEY_TOKENS = {"dimensions", "shape"}
_COMPACT_STRING_SEQUENCE_KEY_TOKENS = {
    "artifactrefs",
    "featuregroupnames",
    "omittedfeaturegroups",
    "reasoncodes",
}

_CONFIDENCE_COMPONENT_KEY_TOKENS = {
    "calibratedconfidence",
    "columnmargin",
    "decisionstrength",
    "deltagroup",
    "globalmargin",
    "globalsupport",
    "fromfirmfraction",
    "groupscore",
    "localsupport",
    "medoidsimilarity",
    "membershipsupport",
    "quantilesupport",
    "rowmargin",
    "support",
    "tofirmfraction",
}


def _key_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _sorted_mapping_keys(value: Mapping[Any, Any], path: str) -> tuple[str, ...]:
    keys = tuple(value.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{path} keys must be non-empty strings.")
    return tuple(sorted(keys, key=lambda item: item.encode("utf-8")))


def _is_supported_dataclass(value: Any) -> bool:
    module = type(value).__module__
    return is_dataclass(value) and any(
        module == prefix or module.startswith(prefix)
        for prefix in _SUPPORTED_MODULE_PREFIXES
    )


def _to_json_value(value: Any, path: str) -> JsonValue:
    if isinstance(value, Enum):
        return _to_json_value(value.value, f"{path}.value")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if _is_supported_dataclass(value):
        return {
            item.name: _to_json_value(getattr(value, item.name), f"{path}.{item.name}")
            for item in sorted(fields(value), key=lambda item: item.name.encode("utf-8"))
        }
    if isinstance(value, Mapping):
        return {
            key: _to_json_value(value[key], f"{path}.{key}")
            for key in _sorted_mapping_keys(value, path)
        }
    if isinstance(value, (list, tuple)):
        return [
            _to_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} must not contain unordered set values.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{path} must not contain byte strings.")
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


def to_json_compatible(value: Any) -> JsonValue:
    """Convert a supported immutable value to deterministic JSON primitives."""

    return _to_json_value(value, "value")


def _compact_scalar(value: Any, path: str) -> JsonScalar:
    converted = _to_json_value(value, path)
    if isinstance(converted, (dict, list)):
        raise TypeError(f"{path} must be a compact scalar value.")
    return converted


def _compact_metadata(value: Any, path: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    compact: dict[str, JsonValue] = {}
    for key in _sorted_mapping_keys(value, path):
        token = _key_token(key)
        nested = value[key]
        if token in _COMPACT_SCALAR_KEY_TOKENS:
            compact[key] = _compact_scalar(nested, f"{path}.{key}")
            continue
        if token in _COMPACT_POINT_KEY_TOKENS:
            if not isinstance(nested, (list, tuple)) or len(nested) != 2:
                raise TypeError(f"{path}.{key} must be a two-value normalized point.")
            point: list[JsonValue] = []
            for index, coordinate in enumerate(nested):
                if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                    raise TypeError(f"{path}.{key}[{index}] must be numeric.")
                converted = float(coordinate)
                if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
                    raise ValueError(f"{path}.{key}[{index}] must be within [0, 1].")
                point.append(converted)
            compact[key] = point
            continue
        if token in _COMPACT_INTEGER_SEQUENCE_KEY_TOKENS:
            if not isinstance(nested, (list, tuple)) or not 1 <= len(nested) <= 4:
                raise TypeError(f"{path}.{key} must contain one to four integers.")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in nested):
                raise TypeError(f"{path}.{key} must contain only integers.")
            compact[key] = list(nested)
            continue
        if token in _COMPACT_STRING_SEQUENCE_KEY_TOKENS:
            if not isinstance(nested, (list, tuple)) or len(nested) > 64:
                raise TypeError(f"{path}.{key} must be a bounded string sequence.")
            if any(not isinstance(item, str) for item in nested):
                raise TypeError(f"{path}.{key} must contain only strings.")
            compact[key] = list(nested)
            continue
        raise ValueError(
            f"Primary payload must not contain unapproved metadata field {path}.{key}."
        )
    return compact


def _confidence_components(value: Any, path: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if len(value) > len(_CONFIDENCE_COMPONENT_KEY_TOKENS):
        raise ValueError(f"{path} contains too many confidence components.")
    compact: dict[str, JsonValue] = {}
    for key in _sorted_mapping_keys(value, path):
        if _key_token(key) not in _CONFIDENCE_COMPONENT_KEY_TOKENS:
            raise ValueError(f"{path}.{key} is not an allowed confidence component.")
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)):
            raise TypeError(f"{path}.{key} must be numeric.")
        compact[key] = _compact_scalar(value[key], f"{path}.{key}")
    return compact


def _primary_to_json(value: Any, path: str) -> JsonValue:
    if isinstance(value, Enum):
        return _primary_to_json(value.value, f"{path}.value")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if is_dataclass(value):
        value_type = type(value)
        allowed_fields = _PRIMARY_FIELD_WHITELIST.get(value_type)
        if allowed_fields is None:
            raise TypeError(f"{path} contains unsupported primary type {value_type.__name__}.")
        actual_fields = {item.name for item in fields(value)}
        if actual_fields != allowed_fields:
            raise ValueError(
                f"{value_type.__name__} fields do not match primary schema "
                f"{PRIMARY_STREAM_RESULT_SCHEMA_ID}."
            )
        converted: dict[str, JsonValue] = {}
        for field_name in sorted(allowed_fields, key=lambda item: item.encode("utf-8")):
            nested = getattr(value, field_name)
            nested_path = f"{path}.{field_name}"
            if (value_type, field_name) in _PRIMARY_METADATA_FIELDS:
                converted[field_name] = _compact_metadata(nested, nested_path)
            elif value_type is ConfidenceMetadata and field_name == "components":
                converted[field_name] = _confidence_components(nested, nested_path)
            else:
                converted[field_name] = _primary_to_json(nested, nested_path)
        return converted
    if isinstance(value, Mapping):
        return {
            key: _primary_to_json(value[key], f"{path}.{key}")
            for key in _sorted_mapping_keys(value, path)
        }
    if isinstance(value, (list, tuple)):
        return [
            _primary_to_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} must not contain unordered set values.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{path} must not contain byte strings.")
    raise TypeError(f"{path} contains unsupported primary value {type(value).__name__}.")


def primary_result_payload(result: StreamAnalysisResult) -> dict[str, JsonValue]:
    if not isinstance(result, StreamAnalysisResult):
        raise TypeError("result must be StreamAnalysisResult.")
    compact_result = _primary_to_json(result, "result")
    if not isinstance(compact_result, dict):
        raise TypeError("StreamAnalysisResult serialization must produce a mapping.")
    payload: dict[str, JsonValue] = {
        "result": compact_result,
        "schema_id": PRIMARY_STREAM_RESULT_SCHEMA_ID,
    }
    return payload


def serialize_primary_result(result: StreamAnalysisResult) -> str:
    """Serialize a compact primary result without writing to the filesystem."""

    return json.dumps(
        primary_result_payload(result),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "PRIMARY_STREAM_RESULT_SCHEMA_ID",
    "primary_result_payload",
    "serialize_primary_result",
    "to_json_compatible",
]
