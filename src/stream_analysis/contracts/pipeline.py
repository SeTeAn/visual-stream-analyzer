"""Immutable cross-stage records for the stream analysis pipeline.

This module defines data exchanged between stages.  It intentionally contains
no extraction, scoring, assignment, grouping, event-generation, hashing or
serialization algorithms.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from .common import (
    ErrorRecord,
    RecordEnvelope,
    StageContext,
    ValidityStatus,
    WarningRecord,
    _freeze_id_refs,
    _freeze_json_value,
    _require_digest,
    _require_identifier,
    _require_version,
)
from .geometry import BBox, ImageSize, Point


class RepresentationFamily(str, Enum):
    DINO_V2 = "dino_v2"
    HANDCRAFTED = "handcrafted"


class MetricOrientation(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class PairEligibility(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class MatchDecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"


class UnmatchedSide(str, Enum):
    FROM = "from"
    TO = "to"


class MembershipStatus(str, Enum):
    FIRM = "firm"
    UNCERTAIN = "uncertain"
    PROVISIONAL = "provisional"


class VisualTypeStatus(str, Enum):
    ESTABLISHED = "established"
    UNCERTAIN = "uncertain"
    PROVISIONAL = "provisional"


class EventKind(str, Enum):
    PERSISTED = "persisted"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    COUNT_CHANGED = "count_changed"
    POSITION_CHANGED = "position_changed"


class EventStatus(str, Enum):
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


class EmissionStatus(str, Enum):
    PRODUCED = "produced"
    WITHHELD = "withheld"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    PARTIAL = "partial"
    FAILED = "failed"


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    NOT_CALIBRATED = "not_calibrated"
    NOT_APPLICABLE = "not_applicable"


def _finite_float(
    value: int | float,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return converted


def _optional_finite_float(
    value: int | float | None,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name, minimum=minimum, maximum=maximum)


def _freeze_metadata(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return _freeze_json_value(value, field_name)


def _sorted_mapping_keys(value: Mapping[Any, Any], field_name: str) -> tuple[str, ...]:
    keys = tuple(value.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{field_name} keys must be non-empty strings.")
    return tuple(sorted(keys, key=lambda item: item.encode("utf-8")))


def _freeze_numeric_mapping(
    value: Mapping[str, int | float],
    field_name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    frozen = {
        key: _finite_float(value[key], f"{field_name}.{key}")
        for key in _sorted_mapping_keys(value, field_name)
    }
    return MappingProxyType(frozen)


def _empty_metadata() -> Mapping[str, Any]:
    return MappingProxyType({})


def _typed_tuple(values: Iterable[Any], expected_type: type, field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {expected_type.__name__} values.")
    result = tuple(values)
    if not all(isinstance(value, expected_type) for value in result):
        raise TypeError(f"{field_name} must contain only {expected_type.__name__} values.")
    return result


def _require_envelope(
    envelope: RecordEnvelope,
    *,
    context: StageContext | None = None,
    record_id: str | None = None,
) -> None:
    if not isinstance(envelope, RecordEnvelope):
        raise TypeError("envelope must be RecordEnvelope.")
    if context is not None and envelope.context != context:
        raise ValueError(f"envelope.context must equal {context!r}.")
    if record_id is not None and envelope.record_id != record_id:
        raise ValueError("envelope.record_id must match the stage record ID.")


def _same_stream(parent: RecordEnvelope, child: RecordEnvelope, field_name: str) -> None:
    if child.stream_id != parent.stream_id:
        raise ValueError(f"{field_name} must belong to stream {parent.stream_id!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskReference:
    """Compact mask provenance; never contains the mask bitmap."""

    mask_ref: str | None
    mask_digest: str | None
    producer_version: str
    coordinate_bbox: BBox
    validity_status: ValidityStatus
    warning_ids: tuple[str, ...] = ()
    error_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_version(self.producer_version, "producer_version")
        if not isinstance(self.coordinate_bbox, BBox):
            raise TypeError("coordinate_bbox must be BBox.")
        if not isinstance(self.validity_status, ValidityStatus):
            raise TypeError("validity_status must be ValidityStatus.")
        warning_ids = _freeze_id_refs(self.warning_ids, "warning_ids")
        error_ids = _freeze_id_refs(self.error_ids, "error_ids")
        object.__setattr__(self, "warning_ids", warning_ids)
        object.__setattr__(self, "error_ids", error_ids)

        if self.validity_status is ValidityStatus.VALID:
            if self.mask_ref is None or self.mask_digest is None:
                raise ValueError("A valid mask requires mask_ref and mask_digest.")
            if error_ids:
                raise ValueError("A valid mask must not reference errors.")
        elif not error_ids:
            raise ValueError("An invalid mask must reference at least one error.")

        if self.mask_ref is not None:
            _require_digest(self.mask_ref, "mask_ref")
        if self.mask_digest is not None:
            _require_digest(self.mask_digest, "mask_digest")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryFeatureMetadata:
    feature_schema_id: str
    producer_version: str
    config_digest: str
    values: Mapping[str, int | float | bool]

    def __post_init__(self) -> None:
        _require_identifier(self.feature_schema_id, "feature_schema_id")
        _require_version(self.producer_version, "producer_version")
        _require_digest(self.config_digest, "config_digest")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping.")
        frozen: dict[str, int | float | bool] = {}
        for name in _sorted_mapping_keys(self.values, "values"):
            _require_identifier(name, "geometry feature name")
            value = self.values[name]
            if isinstance(value, bool):
                frozen[name] = value
            elif isinstance(value, int):
                frozen[name] = value
            else:
                frozen[name] = _finite_float(value, f"geometry feature {name}")
        object.__setattr__(self, "values", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameCandidateDiagnostics:
    envelope: RecordEnvelope
    frame_id: str
    frame_index: int
    image_size: ImageSize
    summary: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.frame_id, "frame_id")
        _require_envelope(self.envelope, context=StageContext(frame_id=self.frame_id))
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be ImageSize.")
        object.__setattr__(self, "summary", _freeze_metadata(self.summary, "summary"))
        object.__setattr__(self, "artifact_refs", _freeze_id_refs(self.artifact_refs, "artifact_refs"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateRecord:
    envelope: RecordEnvelope
    candidate_id: str
    frame_id: str
    frame_index: int
    frame_size: ImageSize
    bbox: BBox
    center: Point
    geometry: GeometryFeatureMetadata
    candidate_source: str
    mask: MaskReference | None = None
    candidate_confidence: float | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.frame_id, "frame_id")
        _require_envelope(
            self.envelope,
            context=StageContext(frame_id=self.frame_id, candidate_id=self.candidate_id),
            record_id=self.candidate_id,
        )
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if not isinstance(self.frame_size, ImageSize):
            raise TypeError("frame_size must be ImageSize.")
        if not isinstance(self.bbox, BBox) or not self.bbox.is_within(self.frame_size):
            raise ValueError("bbox must be a valid BBox within frame_size.")
        if not isinstance(self.center, Point) or self.center != self.bbox.center:
            raise ValueError("center must equal the canonical bbox center.")
        if not isinstance(self.geometry, GeometryFeatureMetadata):
            raise TypeError("geometry must be GeometryFeatureMetadata.")
        _require_identifier(self.candidate_source, "candidate_source")
        if self.mask is not None:
            if not isinstance(self.mask, MaskReference):
                raise TypeError("mask must be MaskReference or None.")
            if self.mask.coordinate_bbox != self.bbox:
                raise ValueError("mask coordinate_bbox must equal the candidate bbox.")
        object.__setattr__(
            self,
            "candidate_confidence",
            _optional_finite_float(
                self.candidate_confidence,
                "candidate_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(self, "quality_flags", _freeze_id_refs(self.quality_flags, "quality_flags"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateExtractionResult:
    envelope: RecordEnvelope
    extractor_source: str
    candidates: tuple[CandidateRecord, ...]
    frame_diagnostics: tuple[FrameCandidateDiagnostics, ...]
    warnings: tuple[WarningRecord, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_envelope(self.envelope, context=StageContext())
        _require_identifier(self.extractor_source, "extractor_source")
        candidates = _typed_tuple(self.candidates, CandidateRecord, "candidates")
        diagnostics = _typed_tuple(
            self.frame_diagnostics,
            FrameCandidateDiagnostics,
            "frame_diagnostics",
        )
        warnings = _typed_tuple(self.warnings, WarningRecord, "warnings")
        errors = _typed_tuple(self.errors, ErrorRecord, "errors")
        if not diagnostics:
            raise ValueError("frame_diagnostics must not be empty.")

        candidate_ids = [candidate.candidate_id for candidate in candidates]
        diagnostic_frame_ids = [diagnostic.frame_id for diagnostic in diagnostics]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within an extraction result.")
        if len(diagnostic_frame_ids) != len(set(diagnostic_frame_ids)):
            raise ValueError("frame diagnostics must be unique by frame_id.")
        diagnostics_by_frame = {diagnostic.frame_id: diagnostic for diagnostic in diagnostics}
        for candidate in candidates:
            diagnostic = diagnostics_by_frame.get(candidate.frame_id)
            if diagnostic is None:
                raise ValueError("Every candidate frame must have frame diagnostics.")
            if candidate.frame_index != diagnostic.frame_index:
                raise ValueError("Candidate frame_index must match its frame diagnostics.")
            if candidate.frame_size != diagnostic.image_size:
                raise ValueError("Candidate frame_size must match its frame diagnostics.")

        for field_name, records in (
            ("candidates", candidates),
            ("frame_diagnostics", diagnostics),
            ("warnings", warnings),
            ("errors", errors),
        ):
            for record in records:
                record_envelope = getattr(record, "envelope", None)
                if record_envelope is not None:
                    _same_stream(self.envelope, record_envelope, field_name)
                elif record.stream_id != self.envelope.stream_id:
                    raise ValueError(f"{field_name} must belong to the result stream.")

        warning_ids = tuple(warning.record_id for warning in warnings)
        error_ids = tuple(error.record_id for error in errors)
        if set(warning_ids) != set(self.envelope.warning_ids):
            raise ValueError("warnings must exactly resolve envelope.warning_ids.")
        if set(error_ids) != set(self.envelope.error_ids):
            raise ValueError("errors must exactly resolve envelope.error_ids.")

        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda item: (item.frame_index, item.candidate_id))),
        )
        object.__setattr__(
            self,
            "frame_diagnostics",
            tuple(sorted(diagnostics, key=lambda item: (item.frame_index, item.frame_id))),
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True, kw_only=True)
class VersionedMetadata:
    identifier: str
    version: str
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.identifier, "identifier")
        _require_version(self.version, "version")
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeMetadata:
    runtime_id: str
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.runtime_id, "runtime_id")
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class InputQualityMetadata:
    candidate_confidence: float | None = None
    quality_flags: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_confidence",
            _optional_finite_float(
                self.candidate_confidence,
                "candidate_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(self, "quality_flags", _freeze_id_refs(self.quality_flags, "quality_flags"))
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class DinoEmbeddingPayload:
    embedding: tuple[float, ...]
    embedding_dimension: int
    l2_normalized: bool

    def __post_init__(self) -> None:
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int):
            raise TypeError("embedding_dimension must be an integer.")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")
        if not isinstance(self.l2_normalized, bool):
            raise TypeError("l2_normalized must be bool.")
        if isinstance(self.embedding, (str, bytes)):
            raise TypeError("embedding must be an iterable of finite floats.")
        embedding = tuple(
            _finite_float(value, f"embedding[{index}]")
            for index, value in enumerate(self.embedding)
        )
        if len(embedding) != self.embedding_dimension:
            raise ValueError("embedding_dimension must equal the embedding length.")
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True, slots=True, kw_only=True)
class HandcraftedFeatureGroup:
    group_name: str
    values: Mapping[str, float]
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.group_name, "group_name")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool.")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping.")
        frozen: dict[str, float] = {}
        for name in _sorted_mapping_keys(self.values, "values"):
            _require_identifier(name, "feature name")
            frozen[name] = _finite_float(self.values[name], f"feature {name}")
        if self.valid and not frozen:
            raise ValueError("A valid feature group must contain at least one value.")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("A valid feature group must not have invalid_reason.")
        if not self.valid:
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason.strip():
                raise ValueError("An invalid feature group requires invalid_reason.")
        object.__setattr__(self, "values", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True, kw_only=True)
class HandcraftedPayload:
    feature_schema_id: str
    feature_groups: tuple[HandcraftedFeatureGroup, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.feature_schema_id, "feature_schema_id")
        groups = _typed_tuple(self.feature_groups, HandcraftedFeatureGroup, "feature_groups")
        names = [group.group_name for group in groups]
        if not groups:
            raise ValueError("feature_groups must not be empty.")
        if len(names) != len(set(names)):
            raise ValueError("feature group names must be unique.")
        object.__setattr__(self, "feature_groups", tuple(sorted(groups, key=lambda item: item.group_name)))


RepresentationPayload: TypeAlias = DinoEmbeddingPayload | HandcraftedPayload


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationRecord:
    envelope: RecordEnvelope
    candidate_id: str
    frame_id: str
    family: RepresentationFamily
    representation_type: str
    representation_version: str
    input_variant: str
    semantic_config_digest: str
    payload: RepresentationPayload | None
    preprocessing_metadata: VersionedMetadata
    provider_metadata: VersionedMetadata
    model_metadata: VersionedMetadata | None
    runtime_metadata: RuntimeMetadata
    input_quality_metadata: InputQualityMetadata

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.frame_id, "frame_id")
        _require_envelope(
            self.envelope,
            context=StageContext(frame_id=self.frame_id, candidate_id=self.candidate_id),
        )
        if not isinstance(self.family, RepresentationFamily):
            raise TypeError("family must be RepresentationFamily.")
        _require_identifier(self.representation_type, "representation_type")
        _require_version(self.representation_version, "representation_version")
        _require_identifier(self.input_variant, "input_variant")
        _require_digest(self.semantic_config_digest, "semantic_config_digest")
        if self.semantic_config_digest != self.envelope.producer.config_digest:
            raise ValueError("semantic_config_digest must match envelope producer config_digest.")
        for field_name, value, expected_type in (
            ("preprocessing_metadata", self.preprocessing_metadata, VersionedMetadata),
            ("provider_metadata", self.provider_metadata, VersionedMetadata),
            ("runtime_metadata", self.runtime_metadata, RuntimeMetadata),
            ("input_quality_metadata", self.input_quality_metadata, InputQualityMetadata),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}.")
        if self.model_metadata is not None and not isinstance(self.model_metadata, VersionedMetadata):
            raise TypeError("model_metadata must be VersionedMetadata or None.")
        if self.envelope.validity_status is ValidityStatus.VALID and self.payload is None:
            raise ValueError("A valid representation requires a payload.")
        if self.payload is not None:
            if self.family is RepresentationFamily.DINO_V2 and not isinstance(
                self.payload, DinoEmbeddingPayload
            ):
                raise TypeError("DINO_V2 representation requires DinoEmbeddingPayload.")
            if self.family is RepresentationFamily.HANDCRAFTED and not isinstance(
                self.payload, HandcraftedPayload
            ):
                raise TypeError("HANDCRAFTED representation requires HandcraftedPayload.")
        if self.family is RepresentationFamily.DINO_V2 and self.model_metadata is None:
            raise ValueError("DINO_V2 representation requires model_metadata.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationSummary:
    """Compact top-level metadata without an embedding or feature payload."""

    representation_record_id: str
    candidate_id: str
    frame_id: str
    family: RepresentationFamily
    representation_type: str
    representation_version: str
    input_variant: str
    semantic_config_digest: str
    validity_status: ValidityStatus
    embedding_dimension: int | None = None
    feature_group_names: tuple[str, ...] = ()
    warning_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "representation_record_id",
            "candidate_id",
            "frame_id",
            "representation_type",
            "input_variant",
        ):
            _require_identifier(getattr(self, field_name), field_name)
        _require_version(self.representation_version, "representation_version")
        _require_digest(self.semantic_config_digest, "semantic_config_digest")
        if not isinstance(self.family, RepresentationFamily):
            raise TypeError("family must be RepresentationFamily.")
        if not isinstance(self.validity_status, ValidityStatus):
            raise TypeError("validity_status must be ValidityStatus.")
        if self.embedding_dimension is not None:
            if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int):
                raise TypeError("embedding_dimension must be an integer or None.")
            if self.embedding_dimension <= 0:
                raise ValueError("embedding_dimension must be positive.")
        feature_names = _freeze_id_refs(self.feature_group_names, "feature_group_names")
        if self.family is RepresentationFamily.DINO_V2 and self.embedding_dimension is None:
            raise ValueError("DINO_V2 summary requires embedding_dimension.")
        if self.family is RepresentationFamily.HANDCRAFTED and self.embedding_dimension is not None:
            raise ValueError("HANDCRAFTED summary must not define embedding_dimension.")
        object.__setattr__(self, "feature_group_names", tuple(sorted(feature_names)))
        object.__setattr__(self, "warning_ids", _freeze_id_refs(self.warning_ids, "warning_ids"))


@dataclass(frozen=True, slots=True, kw_only=True)
class FramePair:
    """Two neighboring frames in chronological order."""

    from_frame_id: str
    from_frame_index: int
    from_frame_size: ImageSize
    to_frame_id: str
    to_frame_index: int
    to_frame_size: ImageSize

    def __post_init__(self) -> None:
        _require_identifier(self.from_frame_id, "from_frame_id")
        _require_identifier(self.to_frame_id, "to_frame_id")
        if self.from_frame_id == self.to_frame_id:
            raise ValueError("A frame pair must contain two different frame IDs.")
        for field_name in ("from_frame_index", "to_frame_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.to_frame_index != self.from_frame_index + 1:
            raise ValueError("FramePair must describe neighboring frames in forward order.")
        if not isinstance(self.from_frame_size, ImageSize) or not isinstance(
            self.to_frame_size, ImageSize
        ):
            raise TypeError("FramePair sizes must be ImageSize values.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateEndpoint:
    candidate_id: str
    frame_id: str
    frame_index: int

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.frame_id, "frame_id")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")


def _validate_endpoint_side(
    endpoint: CandidateEndpoint,
    frame_pair: FramePair,
    side: UnmatchedSide,
) -> None:
    expected = (
        (frame_pair.from_frame_id, frame_pair.from_frame_index)
        if side is UnmatchedSide.FROM
        else (frame_pair.to_frame_id, frame_pair.to_frame_index)
    )
    if (endpoint.frame_id, endpoint.frame_index) != expected:
        raise ValueError(f"Endpoint does not belong to the {side.value} frame.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialEvidence:
    normalized_distance: float | None
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_distance",
            _optional_finite_float(
                self.normalized_distance,
                "normalized_distance",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class AssignmentProvenance:
    assignment_id: str
    policy_id: str
    policy_version: str
    config_digest: str
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.assignment_id, "assignment_id")
        _require_identifier(self.policy_id, "policy_id")
        _require_version(self.policy_version, "policy_version")
        _require_digest(self.config_digest, "config_digest")
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class PairwiseScoreRecord:
    envelope: RecordEnvelope
    pair_id: str
    comparison_id: str
    frame_pair: FramePair
    from_endpoint: CandidateEndpoint
    to_endpoint: CandidateEndpoint
    representation_record_ids: tuple[str, str]
    representation_variant_id: str
    scorer_id: str
    scorer_version: str
    raw_metric_name: str | None
    raw_metric_value: float | None
    metric_orientation: MetricOrientation
    visual_score: float | None
    real_cost: float | None
    spatial_evidence: SpatialEvidence
    eligibility: PairEligibility
    eligibility_reason: str | None = None
    gate_results: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)
    candidate_quality_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.pair_id, "pair_id")
        _require_identifier(self.comparison_id, "comparison_id")
        _require_envelope(
            self.envelope,
            context=StageContext(pair_id=self.pair_id),
            record_id=self.pair_id,
        )
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        if not isinstance(self.from_endpoint, CandidateEndpoint) or not isinstance(
            self.to_endpoint, CandidateEndpoint
        ):
            raise TypeError("Pair endpoints must be CandidateEndpoint values.")
        _validate_endpoint_side(self.from_endpoint, self.frame_pair, UnmatchedSide.FROM)
        _validate_endpoint_side(self.to_endpoint, self.frame_pair, UnmatchedSide.TO)
        if self.from_endpoint.candidate_id == self.to_endpoint.candidate_id:
            raise ValueError("A candidate pair must contain two different endpoints.")
        representation_ids = _freeze_id_refs(
            self.representation_record_ids,
            "representation_record_ids",
        )
        if len(representation_ids) != 2:
            raise ValueError("representation_record_ids must contain one ID per endpoint.")
        _require_identifier(self.representation_variant_id, "representation_variant_id")
        _require_identifier(self.scorer_id, "scorer_id")
        _require_version(self.scorer_version, "scorer_version")
        if not isinstance(self.metric_orientation, MetricOrientation):
            raise TypeError("metric_orientation must be MetricOrientation.")
        if not isinstance(self.eligibility, PairEligibility):
            raise TypeError("eligibility must be PairEligibility.")
        if not isinstance(self.spatial_evidence, SpatialEvidence):
            raise TypeError("spatial_evidence must be SpatialEvidence.")

        raw_value = _optional_finite_float(self.raw_metric_value, "raw_metric_value")
        visual_score = _optional_finite_float(
            self.visual_score,
            "visual_score",
            minimum=0.0,
            maximum=1.0,
        )
        real_cost = _optional_finite_float(
            self.real_cost,
            "real_cost",
            minimum=0.0,
            maximum=1.0,
        )
        if self.raw_metric_name is not None:
            _require_identifier(self.raw_metric_name, "raw_metric_name")
        if (self.raw_metric_name is None) != (raw_value is None):
            raise ValueError("raw_metric_name and raw_metric_value must be present together.")
        if self.envelope.validity_status is ValidityStatus.VALID:
            if raw_value is None or visual_score is None:
                raise ValueError("A valid pair score requires raw metric and visual_score.")
        if self.eligibility is PairEligibility.ELIGIBLE:
            if self.envelope.validity_status is not ValidityStatus.VALID:
                raise ValueError("An eligible pair score must be valid.")
            if real_cost is None or visual_score is None:
                raise ValueError("An eligible pair requires visual_score and real_cost.")
            if not math.isclose(real_cost, 1.0 - visual_score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("real_cost must equal 1 - visual_score.")
            if self.eligibility_reason is not None:
                raise ValueError("An eligible pair must not have eligibility_reason.")
        else:
            if not isinstance(self.eligibility_reason, str) or not self.eligibility_reason.strip():
                raise ValueError("An ineligible pair requires eligibility_reason.")
        object.__setattr__(self, "representation_record_ids", representation_ids)
        object.__setattr__(self, "raw_metric_value", raw_value)
        object.__setattr__(self, "visual_score", visual_score)
        object.__setattr__(self, "real_cost", real_cost)
        object.__setattr__(self, "gate_results", _freeze_metadata(self.gate_results, "gate_results"))
        object.__setattr__(
            self,
            "candidate_quality_refs",
            _freeze_id_refs(self.candidate_quality_refs, "candidate_quality_refs"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchRecord:
    envelope: RecordEnvelope
    match_id: str
    comparison_id: str
    pair_id: str
    frame_pair: FramePair
    from_endpoint: CandidateEndpoint
    to_endpoint: CandidateEndpoint
    pairwise_score_id: str
    assignment: AssignmentProvenance
    status: MatchDecisionStatus
    visual_score: float
    real_cost: float
    spatial_evidence: SpatialEvidence
    confidence_components: Mapping[str, int | float] = field(
        default_factory=_empty_metadata,
        hash=False,
    )
    delta_row: float | None = None
    delta_col: float | None = None
    delta_global: float | None = None
    alternative_pair_ids: tuple[str, ...] = ()
    physical_identity_claim: bool = False

    def __post_init__(self) -> None:
        for field_name in ("match_id", "comparison_id", "pair_id", "pairwise_score_id"):
            _require_identifier(getattr(self, field_name), field_name)
        _require_envelope(
            self.envelope,
            context=StageContext(pair_id=self.pair_id),
            record_id=self.match_id,
        )
        if self.pairwise_score_id != self.pair_id:
            raise ValueError("pairwise_score_id must equal pair_id.")
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        if not isinstance(self.from_endpoint, CandidateEndpoint) or not isinstance(
            self.to_endpoint, CandidateEndpoint
        ):
            raise TypeError("Match endpoints must be CandidateEndpoint values.")
        _validate_endpoint_side(self.from_endpoint, self.frame_pair, UnmatchedSide.FROM)
        _validate_endpoint_side(self.to_endpoint, self.frame_pair, UnmatchedSide.TO)
        if self.from_endpoint.candidate_id == self.to_endpoint.candidate_id:
            raise ValueError("A match must contain two different endpoints.")
        if not isinstance(self.assignment, AssignmentProvenance):
            raise TypeError("assignment must be AssignmentProvenance.")
        if not isinstance(self.status, MatchDecisionStatus):
            raise TypeError("status must be MatchDecisionStatus.")
        if not isinstance(self.spatial_evidence, SpatialEvidence):
            raise TypeError("spatial_evidence must be SpatialEvidence.")
        if self.envelope.validity_status is not ValidityStatus.VALID:
            raise ValueError("A selected match must be valid.")
        if not isinstance(self.physical_identity_claim, bool):
            raise TypeError("physical_identity_claim must be bool.")
        if self.physical_identity_claim:
            raise ValueError("Matching contracts must not claim physical identity.")
        object.__setattr__(
            self,
            "visual_score",
            _finite_float(self.visual_score, "visual_score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "real_cost",
            _finite_float(self.real_cost, "real_cost", minimum=0.0, maximum=1.0),
        )
        if not math.isclose(self.real_cost, 1.0 - self.visual_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("real_cost must equal 1 - visual_score.")
        object.__setattr__(
            self,
            "confidence_components",
            _freeze_numeric_mapping(self.confidence_components, "confidence_components"),
        )
        for field_name in ("delta_row", "delta_col", "delta_global"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite_float(getattr(self, field_name), field_name),
            )
        alternatives = _freeze_id_refs(self.alternative_pair_ids, "alternative_pair_ids")
        if self.pair_id in alternatives:
            raise ValueError("alternative_pair_ids must not contain the selected pair_id.")
        object.__setattr__(self, "alternative_pair_ids", alternatives)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnmatchedRecord:
    envelope: RecordEnvelope
    unmatched_id: str
    comparison_id: str
    frame_pair: FramePair
    endpoint: CandidateEndpoint
    side: UnmatchedSide
    reason_code: str
    assignment: AssignmentProvenance
    selected_dummy_id: str
    unmatched_cost: float | None = None
    global_unmatched_margin: float | None = None
    best_pair_id: str | None = None
    best_visual_score: float | None = None
    candidate_quality_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.unmatched_id, "unmatched_id")
        _require_identifier(self.comparison_id, "comparison_id")
        if not isinstance(self.endpoint, CandidateEndpoint):
            raise TypeError("endpoint must be CandidateEndpoint.")
        _require_envelope(
            self.envelope,
            context=StageContext(
                frame_id=self.endpoint.frame_id,
                candidate_id=self.endpoint.candidate_id,
                pair_id=self.comparison_id,
            ),
            record_id=self.unmatched_id,
        )
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        if not isinstance(self.side, UnmatchedSide):
            raise TypeError("side must be UnmatchedSide.")
        _validate_endpoint_side(self.endpoint, self.frame_pair, self.side)
        _require_identifier(self.reason_code, "reason_code")
        if not isinstance(self.assignment, AssignmentProvenance):
            raise TypeError("assignment must be AssignmentProvenance.")
        _require_identifier(self.selected_dummy_id, "selected_dummy_id")
        if self.best_pair_id is not None:
            _require_identifier(self.best_pair_id, "best_pair_id")
        object.__setattr__(
            self,
            "unmatched_cost",
            _optional_finite_float(self.unmatched_cost, "unmatched_cost"),
        )
        object.__setattr__(
            self,
            "global_unmatched_margin",
            _optional_finite_float(self.global_unmatched_margin, "global_unmatched_margin"),
        )
        object.__setattr__(
            self,
            "best_visual_score",
            _optional_finite_float(
                self.best_visual_score,
                "best_visual_score",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "candidate_quality_refs",
            _freeze_id_refs(self.candidate_quality_refs, "candidate_quality_refs"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameMatchingResult:
    envelope: RecordEnvelope
    comparison_id: str
    frame_pair: FramePair
    representation_variant_id: str
    scorer_id: str
    scorer_version: str
    assignment: AssignmentProvenance
    from_candidates: tuple[CandidateEndpoint, ...]
    to_candidates: tuple[CandidateEndpoint, ...]
    pairwise_scores: tuple[PairwiseScoreRecord, ...]
    selected_matches: tuple[MatchRecord, ...]
    unmatched: tuple[UnmatchedRecord, ...]
    matrix_summary: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.comparison_id, "comparison_id")
        _require_envelope(self.envelope, context=StageContext(pair_id=self.comparison_id))
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        _require_identifier(self.representation_variant_id, "representation_variant_id")
        _require_identifier(self.scorer_id, "scorer_id")
        _require_version(self.scorer_version, "scorer_version")
        if not isinstance(self.assignment, AssignmentProvenance):
            raise TypeError("assignment must be AssignmentProvenance.")
        from_candidates = _typed_tuple(
            self.from_candidates,
            CandidateEndpoint,
            "from_candidates",
        )
        to_candidates = _typed_tuple(self.to_candidates, CandidateEndpoint, "to_candidates")
        scores = _typed_tuple(self.pairwise_scores, PairwiseScoreRecord, "pairwise_scores")
        matches = _typed_tuple(self.selected_matches, MatchRecord, "selected_matches")
        unmatched = _typed_tuple(self.unmatched, UnmatchedRecord, "unmatched")

        for endpoint in from_candidates:
            _validate_endpoint_side(endpoint, self.frame_pair, UnmatchedSide.FROM)
        for endpoint in to_candidates:
            _validate_endpoint_side(endpoint, self.frame_pair, UnmatchedSide.TO)
        all_endpoints = from_candidates + to_candidates
        all_candidate_ids = [endpoint.candidate_id for endpoint in all_endpoints]
        if len(all_candidate_ids) != len(set(all_candidate_ids)):
            raise ValueError("Candidate endpoints must be unique within a frame comparison.")
        endpoint_by_id = {endpoint.candidate_id: endpoint for endpoint in all_endpoints}

        score_ids: set[str] = set()
        scored_endpoint_pairs: set[tuple[str, str]] = set()
        score_by_id: dict[str, PairwiseScoreRecord] = {}
        for score in scores:
            _same_stream(self.envelope, score.envelope, "pairwise_scores")
            if score.comparison_id != self.comparison_id or score.frame_pair != self.frame_pair:
                raise ValueError("Pairwise score belongs to another frame comparison.")
            if score.representation_variant_id != self.representation_variant_id:
                raise ValueError(
                    "Every pairwise score must use the result representation_variant_id."
                )
            if score.scorer_id != self.scorer_id:
                raise ValueError("Every pairwise score must use the result scorer_id.")
            if score.scorer_version != self.scorer_version:
                raise ValueError("Every pairwise score must use the result scorer_version.")
            endpoint_pair = (
                score.from_endpoint.candidate_id,
                score.to_endpoint.candidate_id,
            )
            if score.pair_id in score_ids or endpoint_pair in scored_endpoint_pairs:
                raise ValueError("Pairwise scores must have unique IDs and endpoint pairs.")
            if endpoint_by_id.get(endpoint_pair[0]) != score.from_endpoint or endpoint_by_id.get(
                endpoint_pair[1]
            ) != score.to_endpoint:
                raise ValueError("Pairwise score endpoints are not declared by the result.")
            score_ids.add(score.pair_id)
            scored_endpoint_pairs.add(endpoint_pair)
            score_by_id[score.pair_id] = score

        selected_endpoint_ids: set[str] = set()
        match_ids: set[str] = set()
        for match in matches:
            _same_stream(self.envelope, match.envelope, "selected_matches")
            if match.comparison_id != self.comparison_id or match.frame_pair != self.frame_pair:
                raise ValueError("Selected match belongs to another frame comparison.")
            if match.assignment != self.assignment:
                raise ValueError("Selected match must share result assignment provenance.")
            score = score_by_id.get(match.pairwise_score_id)
            if score is None:
                raise ValueError("Each selected match must reference a pairwise score in the result.")
            if score.eligibility is not PairEligibility.ELIGIBLE:
                raise ValueError("A selected match must reference an eligible pairwise score.")
            if (match.from_endpoint, match.to_endpoint) != (
                score.from_endpoint,
                score.to_endpoint,
            ):
                raise ValueError("Selected match endpoints must equal pairwise score endpoints.")
            if match.visual_score != score.visual_score or match.real_cost != score.real_cost:
                raise ValueError("Selected match must preserve the pairwise visual score and cost.")
            endpoint_ids = {
                match.from_endpoint.candidate_id,
                match.to_endpoint.candidate_id,
            }
            if selected_endpoint_ids & endpoint_ids:
                raise ValueError("An endpoint may occur in only one selected match.")
            if match.match_id in match_ids:
                raise ValueError("Selected match IDs must be unique.")
            selected_endpoint_ids.update(endpoint_ids)
            match_ids.add(match.match_id)

        unmatched_endpoint_ids: set[str] = set()
        unmatched_ids: set[str] = set()
        for decision in unmatched:
            _same_stream(self.envelope, decision.envelope, "unmatched")
            if decision.comparison_id != self.comparison_id or decision.frame_pair != self.frame_pair:
                raise ValueError("Unmatched decision belongs to another frame comparison.")
            if decision.assignment != self.assignment:
                raise ValueError("Unmatched decision must share result assignment provenance.")
            candidate_id = decision.endpoint.candidate_id
            if endpoint_by_id.get(candidate_id) != decision.endpoint:
                raise ValueError("Unmatched endpoint is not declared by the result.")
            if candidate_id in selected_endpoint_ids:
                raise ValueError("A selected endpoint, including an uncertain match, cannot be unmatched.")
            if candidate_id in unmatched_endpoint_ids:
                raise ValueError("An endpoint may occur in only one unmatched decision.")
            if decision.unmatched_id in unmatched_ids:
                raise ValueError("Unmatched decision IDs must be unique.")
            unmatched_endpoint_ids.add(candidate_id)
            unmatched_ids.add(decision.unmatched_id)

        if self.envelope.validity_status is ValidityStatus.VALID:
            decided = selected_endpoint_ids | unmatched_endpoint_ids
            if decided != set(all_candidate_ids):
                raise ValueError("Every endpoint in a valid comparison requires one final decision.")

        object.__setattr__(
            self,
            "from_candidates",
            tuple(sorted(from_candidates, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(
            self,
            "to_candidates",
            tuple(sorted(to_candidates, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(self, "pairwise_scores", tuple(sorted(scores, key=lambda item: item.pair_id)))
        object.__setattr__(self, "selected_matches", tuple(sorted(matches, key=lambda item: item.match_id)))
        object.__setattr__(
            self,
            "unmatched",
            tuple(sorted(unmatched, key=lambda item: item.unmatched_id)),
        )
        object.__setattr__(
            self,
            "matrix_summary",
            _freeze_metadata(self.matrix_summary, "matrix_summary"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfidenceMetadata:
    value: float | None
    calibration_status: CalibrationStatus
    components: Mapping[str, int | float] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_status, CalibrationStatus):
            raise TypeError("calibration_status must be CalibrationStatus.")
        value = _optional_finite_float(
            self.value,
            "value",
            minimum=0.0,
            maximum=1.0,
        )
        if self.calibration_status is CalibrationStatus.CALIBRATED and value is None:
            raise ValueError("Calibrated confidence requires a value.")
        if self.calibration_status is CalibrationStatus.NOT_APPLICABLE and value is not None:
            raise ValueError("NOT_APPLICABLE confidence must have a null value.")
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "components",
            _freeze_numeric_mapping(self.components, "components"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TypeAssignmentRecord:
    envelope: RecordEnvelope
    assignment_id: str
    candidate_id: str
    frame_id: str
    primary_type_id: str
    membership_status: MembershipStatus
    evidence_kind: str
    scores: Mapping[str, int | float]
    confidence: ConfidenceMetadata
    alternative_type_ids: tuple[str, ...] = ()
    physical_instance_claim: bool = False

    def __post_init__(self) -> None:
        for field_name in ("assignment_id", "candidate_id", "frame_id", "primary_type_id"):
            _require_identifier(getattr(self, field_name), field_name)
        _require_envelope(
            self.envelope,
            context=StageContext(
                frame_id=self.frame_id,
                candidate_id=self.candidate_id,
                type_id=self.primary_type_id,
            ),
            record_id=self.assignment_id,
        )
        if self.envelope.validity_status is not ValidityStatus.VALID:
            raise ValueError("A primary type assignment must be valid.")
        if not isinstance(self.membership_status, MembershipStatus):
            raise TypeError("membership_status must be MembershipStatus.")
        _require_identifier(self.evidence_kind, "evidence_kind")
        if not isinstance(self.confidence, ConfidenceMetadata):
            raise TypeError("confidence must be ConfidenceMetadata.")
        alternatives = _freeze_id_refs(self.alternative_type_ids, "alternative_type_ids")
        if self.primary_type_id in alternatives:
            raise ValueError("alternative_type_ids must not contain primary_type_id.")
        if not isinstance(self.physical_instance_claim, bool):
            raise TypeError("physical_instance_claim must be bool.")
        if self.physical_instance_claim:
            raise ValueError("Grouping contracts must not claim physical-instance identity.")
        object.__setattr__(self, "scores", _freeze_numeric_mapping(self.scores, "scores"))
        object.__setattr__(self, "alternative_type_ids", alternatives)


def _freeze_instances_by_frame(
    value: Mapping[str, Iterable[str]],
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("instances_by_frame must be a mapping.")
    frozen: dict[str, tuple[str, ...]] = {}
    seen_candidates: set[str] = set()
    for frame_id in _sorted_mapping_keys(value, "instances_by_frame"):
        _require_identifier(frame_id, "instances_by_frame key")
        members = _freeze_id_refs(value[frame_id], f"instances_by_frame[{frame_id}]")
        overlap = seen_candidates.intersection(members)
        if overlap:
            raise ValueError("A candidate cannot occur under multiple frames in one type.")
        seen_candidates.update(members)
        frozen[frame_id] = tuple(sorted(members))
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True, kw_only=True)
class RecurringVisualType:
    envelope: RecordEnvelope
    type_id: str
    representation_variant_id: str
    grouping_policy_id: str
    grouping_policy_version: str
    member_candidate_ids: tuple[str, ...]
    instances_by_frame: Mapping[str, tuple[str, ...]]
    count_by_frame: Mapping[str, int]
    representative_policy: str
    representative_candidate_id: str
    seed_match_ids: tuple[str, ...]
    presence_summary: Mapping[str, Any]
    internal_similarity_summary: Mapping[str, int | float]
    confidence: ConfidenceMetadata
    status: VisualTypeStatus
    alternative_type_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.type_id, "type_id")
        _require_envelope(
            self.envelope,
            context=StageContext(type_id=self.type_id),
            record_id=self.type_id,
        )
        if self.envelope.validity_status is not ValidityStatus.VALID:
            raise ValueError("A recurring visual type must be a valid predicted record.")
        _require_identifier(self.representation_variant_id, "representation_variant_id")
        _require_identifier(self.grouping_policy_id, "grouping_policy_id")
        _require_version(self.grouping_policy_version, "grouping_policy_version")
        members = _freeze_id_refs(self.member_candidate_ids, "member_candidate_ids")
        if not members:
            raise ValueError("A recurring visual type must contain at least one member.")
        instances = _freeze_instances_by_frame(self.instances_by_frame)
        flattened = tuple(
            candidate_id
            for frame_members in instances.values()
            for candidate_id in frame_members
        )
        if set(flattened) != set(members) or len(flattened) != len(members):
            raise ValueError("instances_by_frame must contain every member exactly once.")
        if not isinstance(self.count_by_frame, Mapping):
            raise TypeError("count_by_frame must be a mapping.")
        counts: dict[str, int] = {}
        for frame_id in _sorted_mapping_keys(self.count_by_frame, "count_by_frame"):
            _require_identifier(frame_id, "count_by_frame key")
            count = self.count_by_frame[frame_id]
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("count_by_frame values must be integers.")
            if count < 0:
                raise ValueError("count_by_frame values must be non-negative.")
            counts[frame_id] = count
        if set(counts) != set(instances):
            raise ValueError("count_by_frame and instances_by_frame must have the same frames.")
        if any(counts[frame_id] != len(instances[frame_id]) for frame_id in counts):
            raise ValueError("count_by_frame must equal the number of instances per frame.")
        _require_identifier(self.representative_policy, "representative_policy")
        _require_identifier(self.representative_candidate_id, "representative_candidate_id")
        if self.representative_candidate_id not in members:
            raise ValueError("representative_candidate_id must be a type member.")
        if not isinstance(self.confidence, ConfidenceMetadata):
            raise TypeError("confidence must be ConfidenceMetadata.")
        if not isinstance(self.status, VisualTypeStatus):
            raise TypeError("status must be VisualTypeStatus.")
        alternatives = _freeze_id_refs(self.alternative_type_ids, "alternative_type_ids")
        if self.type_id in alternatives:
            raise ValueError("alternative_type_ids must not contain type_id.")
        object.__setattr__(self, "member_candidate_ids", tuple(sorted(members)))
        object.__setattr__(self, "instances_by_frame", instances)
        object.__setattr__(self, "count_by_frame", MappingProxyType(counts))
        object.__setattr__(self, "seed_match_ids", _freeze_id_refs(self.seed_match_ids, "seed_match_ids"))
        object.__setattr__(
            self,
            "presence_summary",
            _freeze_metadata(self.presence_summary, "presence_summary"),
        )
        object.__setattr__(
            self,
            "internal_similarity_summary",
            _freeze_numeric_mapping(
                self.internal_similarity_summary,
                "internal_similarity_summary",
            ),
        )
        object.__setattr__(self, "alternative_type_ids", alternatives)


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupingResult:
    envelope: RecordEnvelope
    representation_variant_id: str
    grouping_policy_id: str
    grouping_policy_version: str
    candidate_ids: tuple[str, ...]
    assignments: tuple[TypeAssignmentRecord, ...]
    recurring_types: tuple[RecurringVisualType, ...]

    def __post_init__(self) -> None:
        _require_envelope(self.envelope, context=StageContext())
        _require_identifier(self.representation_variant_id, "representation_variant_id")
        _require_identifier(self.grouping_policy_id, "grouping_policy_id")
        _require_version(self.grouping_policy_version, "grouping_policy_version")
        candidate_ids = _freeze_id_refs(self.candidate_ids, "candidate_ids")
        assignments = _typed_tuple(self.assignments, TypeAssignmentRecord, "assignments")
        recurring_types = _typed_tuple(
            self.recurring_types,
            RecurringVisualType,
            "recurring_types",
        )
        assignment_ids = [assignment.assignment_id for assignment in assignments]
        assigned_candidate_ids = [assignment.candidate_id for assignment in assignments]
        type_ids = [visual_type.type_id for visual_type in recurring_types]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("Type assignment IDs must be unique.")
        if len(assigned_candidate_ids) != len(set(assigned_candidate_ids)):
            raise ValueError("Each candidate must have exactly one primary type assignment.")
        if len(type_ids) != len(set(type_ids)):
            raise ValueError("Recurring type IDs must be unique.")
        if set(assigned_candidate_ids) != set(candidate_ids):
            raise ValueError("Every included candidate requires exactly one assignment.")
        type_by_id = {visual_type.type_id: visual_type for visual_type in recurring_types}
        known_type_ids = set(type_by_id)
        assigned_by_type: dict[str, set[str]] = {type_id: set() for type_id in type_ids}
        frame_by_candidate: dict[str, str] = {}
        for assignment in assignments:
            _same_stream(self.envelope, assignment.envelope, "assignments")
            if assignment.primary_type_id not in type_by_id:
                raise ValueError("Every primary_type_id must resolve to a recurring type.")
            if not set(assignment.alternative_type_ids).issubset(known_type_ids):
                raise ValueError(
                    "Every assignment alternative_type_id must resolve to a recurring type."
                )
            assigned_by_type[assignment.primary_type_id].add(assignment.candidate_id)
            frame_by_candidate[assignment.candidate_id] = assignment.frame_id
        for visual_type in recurring_types:
            _same_stream(self.envelope, visual_type.envelope, "recurring_types")
            if not set(visual_type.alternative_type_ids).issubset(known_type_ids):
                raise ValueError(
                    "Every recurring type alternative_type_id must resolve within the result."
                )
            if visual_type.representation_variant_id != self.representation_variant_id:
                raise ValueError("Recurring type uses another representation variant.")
            if (
                visual_type.grouping_policy_id != self.grouping_policy_id
                or visual_type.grouping_policy_version != self.grouping_policy_version
            ):
                raise ValueError("Recurring type uses another grouping policy.")
            if set(visual_type.member_candidate_ids) != assigned_by_type[visual_type.type_id]:
                raise ValueError("Type members must equal candidates primarily assigned to the type.")
            for frame_id, members in visual_type.instances_by_frame.items():
                if any(frame_by_candidate[candidate_id] != frame_id for candidate_id in members):
                    raise ValueError("Type instance frame must match its assignment frame.")
        object.__setattr__(self, "candidate_ids", tuple(sorted(candidate_ids)))
        object.__setattr__(
            self,
            "assignments",
            tuple(sorted(assignments, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(
            self,
            "recurring_types",
            tuple(sorted(recurring_types, key=lambda item: item.type_id)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TypePresenceSummary:
    type_id: str
    candidate_ids: tuple[str, ...]
    firm_count: int
    possible_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.type_id, "type_id")
        candidate_ids = _freeze_id_refs(self.candidate_ids, "candidate_ids")
        for field_name in ("firm_count", "possible_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.firm_count > self.possible_count:
            raise ValueError("firm_count must not exceed possible_count.")
        if self.possible_count != len(candidate_ids):
            raise ValueError("possible_count must equal the number of candidate_ids.")
        object.__setattr__(self, "candidate_ids", tuple(sorted(candidate_ids)))


def _freeze_presence_by_type(
    value: Mapping[str, TypePresenceSummary],
    field_name: str,
) -> Mapping[str, TypePresenceSummary]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    frozen: dict[str, TypePresenceSummary] = {}
    for type_id in _sorted_mapping_keys(value, field_name):
        _require_identifier(type_id, f"{field_name} key")
        summary = value[type_id]
        if not isinstance(summary, TypePresenceSummary):
            raise TypeError(f"{field_name} values must be TypePresenceSummary.")
        if summary.type_id != type_id:
            raise ValueError(f"{field_name} keys must equal summary type_id values.")
        frozen[type_id] = summary
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True, kw_only=True)
class EventEvidence:
    from_count: int
    to_count: int
    from_firm_count: int
    to_firm_count: int
    from_member_ids: tuple[str, ...] = ()
    to_member_ids: tuple[str, ...] = ()
    accepted_match_ids: tuple[str, ...] = ()
    uncertain_match_ids: tuple[str, ...] = ()
    position_shift_norm: float | None = None
    details: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        for field_name in ("from_count", "to_count", "from_firm_count", "to_firm_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.from_firm_count > self.from_count or self.to_firm_count > self.to_count:
            raise ValueError("Firm event counts must not exceed total counts.")
        from_members = _freeze_id_refs(self.from_member_ids, "from_member_ids")
        to_members = _freeze_id_refs(self.to_member_ids, "to_member_ids")
        if len(from_members) != self.from_count or len(to_members) != self.to_count:
            raise ValueError("Event member IDs must agree with from/to counts.")
        accepted = _freeze_id_refs(self.accepted_match_ids, "accepted_match_ids")
        uncertain = _freeze_id_refs(self.uncertain_match_ids, "uncertain_match_ids")
        if set(accepted) & set(uncertain):
            raise ValueError("A match cannot be both accepted and uncertain evidence.")
        object.__setattr__(self, "from_member_ids", tuple(sorted(from_members)))
        object.__setattr__(self, "to_member_ids", tuple(sorted(to_members)))
        object.__setattr__(self, "accepted_match_ids", tuple(sorted(accepted)))
        object.__setattr__(self, "uncertain_match_ids", tuple(sorted(uncertain)))
        object.__setattr__(
            self,
            "position_shift_norm",
            _optional_finite_float(
                self.position_shift_norm,
                "position_shift_norm",
                minimum=0.0,
                maximum=math.sqrt(2.0),
            ),
        )
        object.__setattr__(self, "details", _freeze_metadata(self.details, "details"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ChangeEvent:
    envelope: RecordEnvelope
    event_id: str
    comparison_id: str
    frame_pair: FramePair
    kind: EventKind
    predicted_type_id: str
    status: EventStatus
    evidence: EventEvidence
    confidence: ConfidenceMetadata
    policy_id: str
    policy_version: str

    def __post_init__(self) -> None:
        for field_name in ("event_id", "comparison_id", "predicted_type_id", "policy_id"):
            _require_identifier(getattr(self, field_name), field_name)
        _require_version(self.policy_version, "policy_version")
        _require_envelope(
            self.envelope,
            context=StageContext(
                pair_id=self.comparison_id,
                type_id=self.predicted_type_id,
                event_id=self.event_id,
            ),
            record_id=self.event_id,
        )
        if self.envelope.validity_status is not ValidityStatus.VALID:
            raise ValueError("An emitted change event must be valid.")
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        if not isinstance(self.kind, EventKind):
            raise TypeError("kind must be EventKind.")
        if not isinstance(self.status, EventStatus):
            raise TypeError("status must be EventStatus.")
        if not isinstance(self.evidence, EventEvidence):
            raise TypeError("evidence must be EventEvidence.")
        if not isinstance(self.confidence, ConfidenceMetadata):
            raise TypeError("confidence must be ConfidenceMetadata.")


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameComparison:
    envelope: RecordEnvelope
    comparison_id: str
    frame_pair: FramePair
    emission_status: EmissionStatus
    representation_variant_id: str
    scorer_id: str
    scorer_version: str
    event_policy_id: str
    event_policy_version: str
    accepted_match_ids: tuple[str, ...]
    uncertain_match_ids: tuple[str, ...]
    unmatched_ids: tuple[str, ...]
    from_type_presence: Mapping[str, TypePresenceSummary]
    to_type_presence: Mapping[str, TypePresenceSummary]
    change_event_ids: tuple[str, ...] = ()
    withheld_diagnostic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.comparison_id, "comparison_id")
        _require_envelope(
            self.envelope,
            context=StageContext(pair_id=self.comparison_id),
            record_id=self.comparison_id,
        )
        if not isinstance(self.frame_pair, FramePair):
            raise TypeError("frame_pair must be FramePair.")
        if not isinstance(self.emission_status, EmissionStatus):
            raise TypeError("emission_status must be EmissionStatus.")
        for field_name in ("representation_variant_id", "scorer_id", "event_policy_id"):
            _require_identifier(getattr(self, field_name), field_name)
        _require_version(self.scorer_version, "scorer_version")
        _require_version(self.event_policy_version, "event_policy_version")
        accepted = _freeze_id_refs(self.accepted_match_ids, "accepted_match_ids")
        uncertain = _freeze_id_refs(self.uncertain_match_ids, "uncertain_match_ids")
        unmatched = _freeze_id_refs(self.unmatched_ids, "unmatched_ids")
        if set(accepted) & set(uncertain):
            raise ValueError("A selected match cannot be both accepted and uncertain.")
        events = _freeze_id_refs(self.change_event_ids, "change_event_ids")
        withheld = _freeze_id_refs(
            self.withheld_diagnostic_ids,
            "withheld_diagnostic_ids",
        )
        if self.emission_status is EmissionStatus.PRODUCED:
            if self.envelope.validity_status is not ValidityStatus.VALID:
                raise ValueError("An invalid comparison must withhold event emission.")
            if withheld:
                raise ValueError("A produced comparison must not have withheld diagnostics.")
        else:
            if not withheld:
                raise ValueError("A withheld comparison requires a structured diagnostic reference.")
            if events:
                raise ValueError("A withheld comparison must not reference emitted events.")
            if not set(withheld).issubset(
                set(self.envelope.warning_ids) | set(self.envelope.error_ids)
            ):
                raise ValueError("Withheld diagnostics must resolve through envelope lineage.")
        object.__setattr__(self, "accepted_match_ids", tuple(sorted(accepted)))
        object.__setattr__(self, "uncertain_match_ids", tuple(sorted(uncertain)))
        object.__setattr__(self, "unmatched_ids", tuple(sorted(unmatched)))
        object.__setattr__(
            self,
            "from_type_presence",
            _freeze_presence_by_type(self.from_type_presence, "from_type_presence"),
        )
        object.__setattr__(
            self,
            "to_type_presence",
            _freeze_presence_by_type(self.to_type_presence, "to_type_presence"),
        )
        object.__setattr__(self, "change_event_ids", tuple(sorted(events)))
        object.__setattr__(self, "withheld_diagnostic_ids", tuple(sorted(withheld)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactReference:
    artifact_id: str
    artifact_kind: str
    reference: str
    producer_record_id: str
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "artifact_id")
        _require_identifier(self.artifact_kind, "artifact_kind")
        _require_identifier(self.producer_record_id, "producer_record_id")
        if not isinstance(self.reference, str):
            raise TypeError("reference must be a string.")
        if not self.reference or self.reference != self.reference.strip():
            raise ValueError("reference must be a non-empty string without surrounding whitespace.")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamAnalysisResult:
    """Compact primary result; full representation payloads remain stage-local."""

    envelope: RecordEnvelope
    run_id: str
    pipeline_version: str
    run_status: RunStatus
    frame_ids: tuple[str, ...]
    data_provenance: VersionedMetadata
    model_provenance: tuple[VersionedMetadata, ...]
    runtime_summary: RuntimeMetadata
    candidate_extraction_result_id: str | None = None
    candidate_record_ids: tuple[str, ...] = ()
    candidate_manifest_ref: str | None = None
    representation_summaries: tuple[RepresentationSummary, ...] = ()
    frame_matching_result_ids: tuple[str, ...] = ()
    grouping_result_id: str | None = None
    recurring_type_ids: tuple[str, ...] = ()
    type_assignment_ids: tuple[str, ...] = ()
    frame_comparisons: tuple[FrameComparison, ...] = ()
    change_events: tuple[ChangeEvent, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    status_summary: Mapping[str, Any] = field(default_factory=_empty_metadata, hash=False)

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id")
        _require_envelope(self.envelope, context=StageContext(), record_id=self.run_id)
        _require_version(self.pipeline_version, "pipeline_version")
        if not isinstance(self.run_status, RunStatus):
            raise TypeError("run_status must be RunStatus.")
        if not isinstance(self.data_provenance, VersionedMetadata):
            raise TypeError("data_provenance must be VersionedMetadata.")
        if not isinstance(self.runtime_summary, RuntimeMetadata):
            raise TypeError("runtime_summary must be RuntimeMetadata.")
        model_provenance = _typed_tuple(
            self.model_provenance,
            VersionedMetadata,
            "model_provenance",
        )
        model_ids = [metadata.identifier for metadata in model_provenance]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model_provenance identifiers must be unique.")
        frame_ids = _freeze_id_refs(self.frame_ids, "frame_ids")
        if not frame_ids and self.run_status is not RunStatus.FAILED:
            raise ValueError("A non-failed result must contain at least one frame ID.")
        if self.candidate_extraction_result_id is not None:
            _require_identifier(
                self.candidate_extraction_result_id,
                "candidate_extraction_result_id",
            )
        if self.candidate_manifest_ref is not None:
            if not isinstance(self.candidate_manifest_ref, str):
                raise TypeError("candidate_manifest_ref must be a string or None.")
            if not self.candidate_manifest_ref or self.candidate_manifest_ref != self.candidate_manifest_ref.strip():
                raise ValueError("candidate_manifest_ref must be non-empty without surrounding whitespace.")
        if self.candidate_extraction_result_id is not None and self.candidate_manifest_ref is not None:
            raise ValueError("Use either an extraction result or a candidate manifest reference, not both.")
        candidate_ids = _freeze_id_refs(self.candidate_record_ids, "candidate_record_ids")
        summaries = _typed_tuple(
            self.representation_summaries,
            RepresentationSummary,
            "representation_summaries",
        )
        summary_ids = [summary.representation_record_id for summary in summaries]
        if len(summary_ids) != len(set(summary_ids)):
            raise ValueError("Representation summary IDs must be unique.")
        if any(summary.candidate_id not in set(candidate_ids) for summary in summaries):
            raise ValueError("Representation summaries must reference listed candidates.")
        matching_ids = _freeze_id_refs(
            self.frame_matching_result_ids,
            "frame_matching_result_ids",
        )
        if self.grouping_result_id is not None:
            _require_identifier(self.grouping_result_id, "grouping_result_id")
        type_ids = _freeze_id_refs(self.recurring_type_ids, "recurring_type_ids")
        assignment_ids = _freeze_id_refs(self.type_assignment_ids, "type_assignment_ids")
        comparisons = _typed_tuple(self.frame_comparisons, FrameComparison, "frame_comparisons")
        events = _typed_tuple(self.change_events, ChangeEvent, "change_events")
        artifacts = _typed_tuple(self.artifacts, ArtifactReference, "artifacts")
        comparison_ids = [comparison.comparison_id for comparison in comparisons]
        event_ids = [event.event_id for event in events]
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("Frame comparison IDs must be unique.")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Change event IDs must be unique.")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Artifact IDs must be unique.")
        event_by_id = {event.event_id: event for event in events}
        referenced_event_ids: set[str] = set()
        known_frames = set(frame_ids)
        known_types = set(type_ids)
        for comparison in comparisons:
            _same_stream(self.envelope, comparison.envelope, "frame_comparisons")
            if comparison.frame_pair.from_frame_id not in known_frames or comparison.frame_pair.to_frame_id not in known_frames:
                raise ValueError("Frame comparison references a frame outside the result.")
            presence_type_ids = set(comparison.from_type_presence) | set(
                comparison.to_type_presence
            )
            if not presence_type_ids.issubset(known_types):
                raise ValueError(
                    "Frame comparison presence summaries must reference predicted recurring types."
                )
            for event_id in comparison.change_event_ids:
                event = event_by_id.get(event_id)
                if event is None:
                    raise ValueError("Every comparison event ID must resolve in change_events.")
                if event.comparison_id != comparison.comparison_id or event.frame_pair != comparison.frame_pair:
                    raise ValueError("Change event must agree with its frame comparison.")
                if not set(event.evidence.accepted_match_ids).issubset(
                    comparison.accepted_match_ids
                ):
                    raise ValueError(
                        "Event accepted-match evidence must resolve in its frame comparison."
                    )
                if not set(event.evidence.uncertain_match_ids).issubset(
                    comparison.uncertain_match_ids
                ):
                    raise ValueError(
                        "Event uncertain-match evidence must resolve in its frame comparison."
                    )
                if event_id in referenced_event_ids:
                    raise ValueError("A change event may belong to only one frame comparison.")
                referenced_event_ids.add(event_id)
        for event in events:
            _same_stream(self.envelope, event.envelope, "change_events")
            if event.predicted_type_id not in known_types:
                raise ValueError("Change event must reference a predicted recurring type.")
        if referenced_event_ids != set(event_ids):
            raise ValueError("Every change event must be referenced by one frame comparison.")
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "model_provenance", tuple(sorted(model_provenance, key=lambda item: item.identifier)))
        object.__setattr__(self, "candidate_record_ids", tuple(sorted(candidate_ids)))
        object.__setattr__(
            self,
            "representation_summaries",
            tuple(sorted(summaries, key=lambda item: item.representation_record_id)),
        )
        object.__setattr__(self, "frame_matching_result_ids", tuple(sorted(matching_ids)))
        object.__setattr__(self, "recurring_type_ids", tuple(sorted(type_ids)))
        object.__setattr__(self, "type_assignment_ids", tuple(sorted(assignment_ids)))
        object.__setattr__(
            self,
            "frame_comparisons",
            tuple(
                sorted(
                    comparisons,
                    key=lambda item: (
                        item.frame_pair.from_frame_index,
                        item.frame_pair.to_frame_index,
                        item.comparison_id,
                    ),
                )
            ),
        )
        object.__setattr__(self, "change_events", tuple(sorted(events, key=lambda item: item.event_id)))
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.artifact_id)))
        object.__setattr__(
            self,
            "status_summary",
            _freeze_metadata(self.status_summary, "status_summary"),
        )
