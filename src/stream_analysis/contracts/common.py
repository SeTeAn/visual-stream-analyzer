"""Shared record and diagnostic contracts.

The contracts deliberately model only record validity. Decision, emission and
run statuses belong to later pipeline stages and are not represented here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ValidityStatus(str, Enum):
    """Whether a record is eligible for normal downstream decisions."""

    VALID = "valid"
    INVALID = "invalid"


class Severity(str, Enum):
    """Severity of a structured warning or error record."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must start with an ASCII letter or digit and contain "
            "only ASCII letters, digits, '.', '_', ':' or '-'."
        )
    return value


def _require_version(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not _VERSION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty version token without whitespace.")
    return value


def _require_digest(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value or value != value.strip() or any(char.isspace() for char in value):
        raise ValueError(f"{field_name} must be a non-empty token without whitespace.")
    return value


def _freeze_id_refs(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of identifiers, not a string.")
    refs = tuple(_require_identifier(value, f"{field_name} item") for value in values)
    if len(refs) != len(set(refs)):
        raise ValueError(f"{field_name} must not contain duplicate identifiers.")
    return refs


def _freeze_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{path} keys must be non-empty strings.")
        frozen: dict[str, Any] = {}
        for key in sorted(keys, key=lambda item: item.encode("utf-8")):
            frozen[key] = _freeze_json_value(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}.")


def _empty_metadata() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class StageContext:
    """Optional identifiers locating a record within a pipeline stage."""

    frame_id: str | None = None
    candidate_id: str | None = None
    pair_id: str | None = None
    type_id: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("frame_id", "candidate_id", "pair_id", "type_id", "event_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_identifier(value, field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProducerProvenance:
    """Producer and semantic-configuration provenance for a record."""

    producer_stage: str
    producer_version: str
    config_version: str
    config_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.producer_stage, "producer_stage")
        _require_version(self.producer_version, "producer_version")
        _require_version(self.config_version, "config_version")
        _require_digest(self.config_digest, "config_digest")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordEnvelope:
    """Common envelope carried by future stage-specific records."""

    record_id: str
    schema_version: str
    stream_id: str
    producer: ProducerProvenance
    context: StageContext = field(default_factory=StageContext)
    validity_status: ValidityStatus = ValidityStatus.VALID
    warning_ids: tuple[str, ...] = ()
    error_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.record_id, "record_id")
        _require_version(self.schema_version, "schema_version")
        _require_identifier(self.stream_id, "stream_id")
        if not isinstance(self.producer, ProducerProvenance):
            raise TypeError("producer must be ProducerProvenance.")
        if not isinstance(self.context, StageContext):
            raise TypeError("context must be StageContext.")
        if not isinstance(self.validity_status, ValidityStatus):
            raise TypeError("validity_status must be ValidityStatus.")

        warning_ids = _freeze_id_refs(self.warning_ids, "warning_ids")
        error_ids = _freeze_id_refs(self.error_ids, "error_ids")
        provenance_refs = _freeze_id_refs(self.provenance_refs, "provenance_refs")
        object.__setattr__(self, "warning_ids", warning_ids)
        object.__setattr__(self, "error_ids", error_ids)
        object.__setattr__(self, "provenance_refs", provenance_refs)

        if self.record_id in provenance_refs:
            raise ValueError("provenance_refs must not contain record_id itself.")
        if self.validity_status is ValidityStatus.VALID and error_ids:
            raise ValueError("A valid record must not reference error records.")
        if self.validity_status is ValidityStatus.INVALID and not error_ids:
            raise ValueError("An invalid record must reference at least one error record.")


def _validate_diagnostic(
    *,
    record_id: str,
    schema_version: str,
    stream_id: str,
    code: str,
    stage: str,
    message: str,
    producer: ProducerProvenance,
    context: StageContext,
    metadata: Mapping[str, Any],
    provenance_refs: Iterable[str],
    upstream_warning_ids: Iterable[str],
    upstream_error_ids: Iterable[str],
) -> tuple[Mapping[str, Any], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    _require_identifier(record_id, "record_id")
    _require_version(schema_version, "schema_version")
    _require_identifier(stream_id, "stream_id")
    if not isinstance(code, str) or not _CODE_PATTERN.fullmatch(code):
        raise ValueError("code must be an uppercase machine-readable identifier.")
    _require_identifier(stage, "stage")
    if not isinstance(message, str):
        raise TypeError("message must be a string.")
    if not message or message != message.strip():
        raise ValueError("message must be non-empty and must not have surrounding whitespace.")
    if not isinstance(producer, ProducerProvenance):
        raise TypeError("producer must be ProducerProvenance.")
    if producer.producer_stage != stage:
        raise ValueError("stage must match producer.producer_stage.")
    if not isinstance(context, StageContext):
        raise TypeError("context must be StageContext.")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")

    frozen_metadata = _freeze_json_value(metadata, "metadata")
    frozen_provenance = _freeze_id_refs(provenance_refs, "provenance_refs")
    frozen_warnings = _freeze_id_refs(upstream_warning_ids, "upstream_warning_ids")
    frozen_errors = _freeze_id_refs(upstream_error_ids, "upstream_error_ids")
    for field_name, references in (
        ("provenance_refs", frozen_provenance),
        ("upstream_warning_ids", frozen_warnings),
        ("upstream_error_ids", frozen_errors),
    ):
        if record_id in references:
            raise ValueError(f"{field_name} must not contain record_id itself.")
    return frozen_metadata, frozen_provenance, frozen_warnings, frozen_errors


@dataclass(frozen=True, slots=True, kw_only=True)
class WarningRecord:
    """Structured non-fatal diagnostic with explicit upstream lineage."""

    record_id: str
    schema_version: str
    stream_id: str
    code: str
    stage: str
    message: str
    producer: ProducerProvenance
    severity: Severity = Severity.WARNING
    context: StageContext = field(default_factory=StageContext)
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)
    provenance_refs: tuple[str, ...] = ()
    upstream_warning_ids: tuple[str, ...] = ()
    upstream_error_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            raise TypeError("severity must be Severity.")
        if self.severity not in (Severity.INFO, Severity.WARNING):
            raise ValueError("WarningRecord severity must be INFO or WARNING.")
        metadata, provenance, warnings, errors = _validate_diagnostic(
            record_id=self.record_id,
            schema_version=self.schema_version,
            stream_id=self.stream_id,
            code=self.code,
            stage=self.stage,
            message=self.message,
            producer=self.producer,
            context=self.context,
            metadata=self.metadata,
            provenance_refs=self.provenance_refs,
            upstream_warning_ids=self.upstream_warning_ids,
            upstream_error_ids=self.upstream_error_ids,
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance_refs", provenance)
        object.__setattr__(self, "upstream_warning_ids", warnings)
        object.__setattr__(self, "upstream_error_ids", errors)

    @property
    def id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorRecord:
    """Structured error diagnostic with explicit upstream lineage."""

    record_id: str
    schema_version: str
    stream_id: str
    code: str
    stage: str
    message: str
    producer: ProducerProvenance
    severity: Severity = Severity.ERROR
    context: StageContext = field(default_factory=StageContext)
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)
    provenance_refs: tuple[str, ...] = ()
    upstream_warning_ids: tuple[str, ...] = ()
    upstream_error_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            raise TypeError("severity must be Severity.")
        if self.severity not in (Severity.ERROR, Severity.CRITICAL):
            raise ValueError("ErrorRecord severity must be ERROR or CRITICAL.")
        metadata, provenance, warnings, errors = _validate_diagnostic(
            record_id=self.record_id,
            schema_version=self.schema_version,
            stream_id=self.stream_id,
            code=self.code,
            stage=self.stage,
            message=self.message,
            producer=self.producer,
            context=self.context,
            metadata=self.metadata,
            provenance_refs=self.provenance_refs,
            upstream_warning_ids=self.upstream_warning_ids,
            upstream_error_ids=self.upstream_error_ids,
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance_refs", provenance)
        object.__setattr__(self, "upstream_warning_ids", warnings)
        object.__setattr__(self, "upstream_error_ids", errors)

    @property
    def id(self) -> str:
        return self.record_id
