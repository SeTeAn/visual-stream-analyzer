"""F08 DINOv2 configuration, batch assembly and representation records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from ..candidates import CandidateExtractionSnapshot
from ..config import StageConfig, semantic_config_digest
from ..contracts import (
    CandidateRecord,
    DinoEmbeddingPayload,
    ErrorRecord,
    InputQualityMetadata,
    ProducerProvenance,
    RecordEnvelope,
    RepresentationFamily,
    RepresentationRecord,
    RuntimeMetadata,
    StageContext,
    ValidityStatus,
    VersionedMetadata,
    WarningRecord,
)
from ..input import DecodedFrame, DecodedStream
from .dinov2_preprocessing import (
    DINO_BBOX_VARIANT,
    DINO_IMAGENET_MEAN,
    DINO_IMAGENET_STD,
    DINO_INPUT_SIZE,
    DINO_MASK_NEUTRAL_VARIANT,
    DINO_NEUTRAL_RGB,
    DINO_RGB_INTERPOLATION,
    DinoV2PreprocessingError,
    DinoV2PreprocessingResult,
    DinoV2Variant,
    preprocess_dinov2_candidate,
)
from .dinov2_provider import (
    DINO_EMBEDDING_DIMENSION,
    DINO_MODEL_NAME,
    DINO_MODEL_VERSION,
    DINO_PROVIDER_ID,
    DINO_PROVIDER_VERSION,
    DevicePolicy,
    DinoV2BatchProviderProtocol,
    DinoV2ModelSpec,
    DinoV2ProviderError,
    DinoV2ProviderOutput,
)


DINO_REPRESENTATION_TYPE: Final = "dinov2_vits14_cls"
DINO_REPRESENTATION_VERSION: Final = "dinov2-vits14-cls-1.0"
DINO_CONFIG_SCHEMA_ID: Final = "stream_analysis.dinov2_representation_config.v1"
DINO_RECORD_SCHEMA_VERSION: Final = "dinov2-representation-record-1.0"
DINO_WARNING_SCHEMA_VERSION: Final = "dinov2-representation-warning-1.0"
DINO_ERROR_SCHEMA_VERSION: Final = "dinov2-representation-error-1.0"
DINO_PREPROCESSING_VERSION: Final = "1.0"


@dataclass(frozen=True, slots=True)
class DinoV2RepresentationConfig:
    """One homogeneous semantic variant plus runtime-only device/batch policy."""

    expected_checkpoint_sha256: str
    expected_source_tree_fingerprint: str
    expected_checkpoint_size_bytes: int | None = None
    variant: DinoV2Variant = DINO_BBOX_VARIANT
    config_version: str = "1.0"
    input_size: int = DINO_INPUT_SIZE
    context_padding_ratio: float = 0.0
    neutral_rgb: tuple[int, int, int] = DINO_NEUTRAL_RGB
    imagenet_mean: tuple[float, float, float] = DINO_IMAGENET_MEAN
    imagenet_std: tuple[float, float, float] = DINO_IMAGENET_STD
    device_policy: DevicePolicy = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        spec = self.model_spec
        if self.variant not in {DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT}:
            raise ValueError("variant must be a supported DINOv2 v1 variant.")
        if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError("input_size must be a positive integer.")
        if self.input_size != DINO_INPUT_SIZE:
            raise ValueError(f"input_size must remain {DINO_INPUT_SIZE} for F08 v1.")
        if isinstance(self.context_padding_ratio, bool) or not isinstance(
            self.context_padding_ratio, (int, float)
        ):
            raise TypeError("context_padding_ratio must be a real number.")
        ratio = float(self.context_padding_ratio)
        if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError("context_padding_ratio must be finite and in [0, 1].")
        object.__setattr__(self, "context_padding_ratio", ratio)
        if self.device_policy not in {"cpu", "cuda", "auto"}:
            raise ValueError("device_policy must be cpu, cuda or auto.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if (
            not isinstance(self.neutral_rgb, tuple)
            or len(self.neutral_rgb) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.neutral_rgb)
        ):
            raise ValueError("neutral_rgb must contain three integer channels.")
        if any(value < 0 or value > 255 for value in self.neutral_rgb):
            raise ValueError("neutral_rgb channels must be in [0, 255].")
        for name, values, positive in (
            ("imagenet_mean", self.imagenet_mean, False),
            ("imagenet_std", self.imagenet_std, True),
        ):
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError(f"{name} must contain three values.")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                for value in values
            ):
                raise ValueError(f"{name} contains invalid values.")
        StageConfig(
            stage_id="representation",
            schema_id=DINO_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
        )
        del spec

    @property
    def model_spec(self) -> DinoV2ModelSpec:
        return DinoV2ModelSpec(
            expected_checkpoint_sha256=self.expected_checkpoint_sha256,
            expected_source_tree_fingerprint=self.expected_source_tree_fingerprint,
            expected_checkpoint_size_bytes=self.expected_checkpoint_size_bytes,
        )

    @property
    def semantic_parameters(self) -> Mapping[str, object]:
        return {
            "variant": self.variant,
            "representation_type": DINO_REPRESENTATION_TYPE,
            "representation_version": DINO_REPRESENTATION_VERSION,
            "model": {
                "model_name": DINO_MODEL_NAME,
                "model_version": DINO_MODEL_VERSION,
                "output_token": "cls",
                "embedding_dimension": DINO_EMBEDDING_DIMENSION,
                "expected_checkpoint_sha256": self.model_spec.expected_checkpoint_sha256,
                "expected_source_tree_fingerprint": self.model_spec.expected_source_tree_fingerprint,
                "expected_checkpoint_size_bytes": self.expected_checkpoint_size_bytes,
            },
            "preprocessing": {
                "input_size": self.input_size,
                "context_padding_ratio": self.context_padding_ratio,
                "aspect_policy": "preserve_with_square_letterbox",
                "rgb_interpolation": DINO_RGB_INTERPOLATION,
                "bbox_fill_policy": "bbox_border_median",
                "mask_policy": (
                    "mask_required_neutral_background"
                    if self.variant == DINO_MASK_NEUTRAL_VARIANT
                    else "bbox_only"
                ),
                "neutral_rgb": self.neutral_rgb,
                "normalization": "imagenet_mean_std",
                "imagenet_mean": self.imagenet_mean,
                "imagenet_std": self.imagenet_std,
                "input_dtype": "uint8",
                "working_dtype": "float32",
            },
        }

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="representation",
            schema_id=DINO_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters=self.semantic_parameters,
            runtime_parameters={
                "device_policy": self.device_policy,
                "batch_size": self.batch_size,
            },
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="representation",
            producer_version=DINO_REPRESENTATION_VERSION,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True)
class DinoV2RepresentationBatch:
    """Homogeneous DINOv2 records and newly produced diagnostics."""

    variant: DinoV2Variant
    semantic_config_digest: str
    requested_device: DevicePolicy
    resolved_device: str | None
    records: tuple[RepresentationRecord, ...]
    warnings: tuple[WarningRecord, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in {DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT}:
            raise ValueError("Invalid DINOv2 batch variant.")
        records = tuple(self.records)
        warnings = tuple(self.warnings)
        errors = tuple(self.errors)
        if any(not isinstance(item, RepresentationRecord) for item in records):
            raise TypeError("records must contain RepresentationRecord values.")
        if any(not isinstance(item, WarningRecord) for item in warnings):
            raise TypeError("warnings must contain WarningRecord values.")
        if any(not isinstance(item, ErrorRecord) for item in errors):
            raise TypeError("errors must contain ErrorRecord values.")
        if len({item.candidate_id for item in records}) != len(records):
            raise ValueError("records must be unique by candidate_id.")
        for record in records:
            if record.input_variant != self.variant:
                raise ValueError("Every record must use the batch variant.")
            if record.semantic_config_digest != self.semantic_config_digest:
                raise ValueError("Every record must use the batch semantic config digest.")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    candidate: CandidateRecord
    preprocessing: DinoV2PreprocessingResult


@dataclass(frozen=True, slots=True)
class _EmbeddingResult:
    embedding: np.ndarray
    provider_output: DinoV2ProviderOutput
    batch_index: int
    batch_position: int


def build_dinov2_representations(
    decoded_stream: DecodedStream,
    candidate_snapshot: CandidateExtractionSnapshot,
    provider: DinoV2BatchProviderProtocol,
    config: DinoV2RepresentationConfig | None = None,
) -> DinoV2RepresentationBatch:
    """Build one ordered DINOv2 record per canonical F05 candidate."""

    if not isinstance(decoded_stream, DecodedStream):
        raise TypeError("decoded_stream must be DecodedStream.")
    if not isinstance(candidate_snapshot, CandidateExtractionSnapshot):
        raise TypeError("candidate_snapshot must be CandidateExtractionSnapshot.")
    if not hasattr(provider, "embed_batch") or not hasattr(provider, "model_spec"):
        raise TypeError("provider must implement DinoV2BatchProviderProtocol.")
    effective_config = config or DinoV2RepresentationConfig(
        expected_checkpoint_sha256=provider.model_spec.expected_checkpoint_sha256,
        expected_source_tree_fingerprint=provider.model_spec.expected_source_tree_fingerprint,
        expected_checkpoint_size_bytes=provider.model_spec.expected_checkpoint_size_bytes,
        device_policy=provider.requested_device,
        batch_size=provider.batch_size,
    )
    if not isinstance(effective_config, DinoV2RepresentationConfig):
        raise TypeError("config must be DinoV2RepresentationConfig or None.")
    _validate_provider_config(provider, effective_config)
    _validate_stream_snapshot(decoded_stream, candidate_snapshot)

    frames = {frame.frame_id: frame for frame in decoded_stream.frames}
    prepared: list[_PreparedCandidate] = []
    error_by_candidate: dict[str, ErrorRecord] = {}
    preprocessing_by_candidate: dict[str, DinoV2PreprocessingResult] = {}
    warnings: list[WarningRecord] = []
    errors: list[ErrorRecord] = []

    for candidate in candidate_snapshot.result.candidates:
        frame = frames[candidate.frame_id]
        try:
            mask_record = None
            if effective_config.variant == DINO_MASK_NEUTRAL_VARIANT:
                try:
                    mask_record = candidate_snapshot.mask_for_candidate(candidate.candidate_id)
                except KeyError as error:
                    raise DinoV2PreprocessingError(
                        "MASK_REQUIRED",
                        "Mask-neutral DINOv2 preprocessing requires an in-memory mask record.",
                    ) from error
            result = preprocess_dinov2_candidate(
                frame,
                candidate,
                variant=effective_config.variant,
                input_size=effective_config.input_size,
                context_padding_ratio=effective_config.context_padding_ratio,
                neutral_rgb=effective_config.neutral_rgb,
                imagenet_mean=effective_config.imagenet_mean,
                imagenet_std=effective_config.imagenet_std,
                mask_record=mask_record,
            )
            preprocessing_by_candidate[candidate.candidate_id] = result
            prepared.append(_PreparedCandidate(candidate=candidate, preprocessing=result))
        except DinoV2PreprocessingError as failure:
            error = _error(
                candidate,
                candidate_snapshot,
                effective_config,
                failure.code,
                failure.message,
                failure.metadata,
            )
            errors.append(error)
            error_by_candidate[candidate.candidate_id] = error

    embedding_by_candidate: dict[str, _EmbeddingResult] = {}
    for batch_index, start in enumerate(range(0, len(prepared), effective_config.batch_size)):
        chunk = prepared[start:start + effective_config.batch_size]
        normalized = np.stack(
            [item.preprocessing.normalized_chw for item in chunk],
            axis=0,
        ).astype(np.float32, copy=False)
        try:
            output = provider.embed_batch(normalized)
            _validate_provider_output(output, len(chunk))
        except DinoV2ProviderError as failure:
            for item in prepared[start:]:
                error = _error(
                    item.candidate,
                    candidate_snapshot,
                    effective_config,
                    failure.code,
                    failure.message,
                    failure.metadata,
                )
                errors.append(error)
                error_by_candidate[item.candidate.candidate_id] = error
            break
        except (TypeError, ValueError) as failure:
            for item in prepared[start:]:
                error = _error(
                    item.candidate,
                    candidate_snapshot,
                    effective_config,
                    "INVALID_PROVIDER_OUTPUT",
                    "DINOv2 provider returned an invalid batch output.",
                    {"cause": type(failure).__name__},
                )
                errors.append(error)
                error_by_candidate[item.candidate.candidate_id] = error
            break

        for position, item in enumerate(chunk):
            embedding_by_candidate[item.candidate.candidate_id] = _EmbeddingResult(
                embedding=output.embeddings[position],
                provider_output=output,
                batch_index=batch_index,
                batch_position=position,
            )
            if output.warning_code == "DEVICE_FALLBACK_CPU":
                warnings.append(
                    _warning(
                        item.candidate,
                        candidate_snapshot,
                        effective_config,
                        code="DEVICE_FALLBACK_CPU",
                        message="Auto device policy selected CPU because CUDA is unavailable.",
                        metadata={
                            "requested_device": effective_config.device_policy,
                            "resolved_device": "cpu",
                        },
                    )
                )

    warnings_by_candidate = {
        warning.context.candidate_id: warning
        for warning in warnings
        if warning.context.candidate_id is not None
    }
    records: list[RepresentationRecord] = []
    for candidate in candidate_snapshot.result.candidates:
        embedding_result = embedding_by_candidate.get(candidate.candidate_id)
        if embedding_result is not None:
            records.append(
                _valid_record(
                    candidate,
                    candidate_snapshot,
                    effective_config,
                    provider,
                    preprocessing_by_candidate[candidate.candidate_id],
                    embedding_result,
                    warnings_by_candidate.get(candidate.candidate_id),
                )
            )
        else:
            error = error_by_candidate.get(candidate.candidate_id)
            if error is None:
                error = _error(
                    candidate,
                    candidate_snapshot,
                    effective_config,
                    "INFERENCE_NOT_PRODUCED",
                    "No DINOv2 embedding or structured provider failure was produced.",
                    {},
                )
                errors.append(error)
            records.append(
                _invalid_record(
                    candidate,
                    candidate_snapshot,
                    effective_config,
                    provider,
                    preprocessing_by_candidate.get(candidate.candidate_id),
                    error,
                )
            )

    return DinoV2RepresentationBatch(
        variant=effective_config.variant,
        semantic_config_digest=effective_config.config_digest,
        requested_device=effective_config.device_policy,
        resolved_device=provider.resolved_device,
        records=tuple(records),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _valid_record(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: DinoV2RepresentationConfig,
    provider: DinoV2BatchProviderProtocol,
    preprocessing: DinoV2PreprocessingResult,
    result: _EmbeddingResult,
    new_warning: WarningRecord | None,
) -> RepresentationRecord:
    vector = tuple(float(value) for value in result.embedding)
    runtime_details = {
        **dict(result.provider_output.runtime_details),
        "batch_index": result.batch_index,
        "batch_position": result.batch_position,
    }
    warning_ids = _ordered_unique(
        (
            *candidate.envelope.warning_ids,
            *(candidate.mask.warning_ids if candidate.mask else ()),
            *(() if new_warning is None else (new_warning.record_id,)),
        )
    )
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=_record_id(candidate, config),
            schema_version=DINO_RECORD_SCHEMA_VERSION,
            stream_id=candidate.envelope.stream_id,
            producer=config.producer,
            context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
            warning_ids=warning_ids,
            provenance_refs=_provenance_refs(candidate, snapshot, config),
        ),
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        family=RepresentationFamily.DINO_V2,
        representation_type=DINO_REPRESENTATION_TYPE,
        representation_version=DINO_REPRESENTATION_VERSION,
        input_variant=config.variant,
        semantic_config_digest=config.config_digest,
        payload=DinoEmbeddingPayload(
            embedding=vector,
            embedding_dimension=DINO_EMBEDDING_DIMENSION,
            l2_normalized=True,
        ),
        preprocessing_metadata=VersionedMetadata(
            identifier=f"{config.variant}.preprocessing",
            version=DINO_PREPROCESSING_VERSION,
            details={**dict(preprocessing.details), "config_digest": config.config_digest},
        ),
        provider_metadata=VersionedMetadata(
            identifier=DINO_PROVIDER_ID,
            version=DINO_PROVIDER_VERSION,
            details=provider.provider_metadata(),
        ),
        model_metadata=VersionedMetadata(
            identifier=DINO_MODEL_NAME,
            version=DINO_MODEL_VERSION,
            details=provider.model_metadata(),
        ),
        runtime_metadata=RuntimeMetadata(
            runtime_id=f"dinov2_{provider.resolved_device or 'unresolved'}",
            details=runtime_details,
        ),
        input_quality_metadata=_input_quality(candidate),
    )


def _invalid_record(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: DinoV2RepresentationConfig,
    provider: DinoV2BatchProviderProtocol,
    preprocessing: DinoV2PreprocessingResult | None,
    error: ErrorRecord,
) -> RepresentationRecord:
    details: dict[str, object] = {
        "variant": config.variant,
        "candidate_bbox": _bbox_details(candidate),
        "context_padding_ratio": config.context_padding_ratio,
        "input_size": config.input_size,
        "mask_policy": (
            "mask_required_neutral_background"
            if config.variant == DINO_MASK_NEUTRAL_VARIANT
            else "bbox_only"
        ),
        "failure_code": error.code,
        "config_digest": config.config_digest,
    }
    if preprocessing is not None:
        details.update(dict(preprocessing.details))
    warning_ids = _ordered_unique(
        (*candidate.envelope.warning_ids, *(candidate.mask.warning_ids if candidate.mask else ()))
    )
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=_record_id(candidate, config),
            schema_version=DINO_RECORD_SCHEMA_VERSION,
            stream_id=candidate.envelope.stream_id,
            producer=config.producer,
            context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
            validity_status=ValidityStatus.INVALID,
            warning_ids=warning_ids,
            error_ids=(error.record_id,),
            provenance_refs=_provenance_refs(candidate, snapshot, config),
        ),
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        family=RepresentationFamily.DINO_V2,
        representation_type=DINO_REPRESENTATION_TYPE,
        representation_version=DINO_REPRESENTATION_VERSION,
        input_variant=config.variant,
        semantic_config_digest=config.config_digest,
        payload=None,
        preprocessing_metadata=VersionedMetadata(
            identifier=f"{config.variant}.preprocessing",
            version=DINO_PREPROCESSING_VERSION,
            details=details,
        ),
        provider_metadata=VersionedMetadata(
            identifier=DINO_PROVIDER_ID,
            version=DINO_PROVIDER_VERSION,
            details=provider.provider_metadata(),
        ),
        model_metadata=VersionedMetadata(
            identifier=DINO_MODEL_NAME,
            version=DINO_MODEL_VERSION,
            details=provider.model_metadata(),
        ),
        runtime_metadata=RuntimeMetadata(
            runtime_id=f"dinov2_{provider.resolved_device or 'unresolved'}",
            details={
                "requested_device": config.device_policy,
                "resolved_device": provider.resolved_device,
                "batch_size": config.batch_size,
                "dtype": "float32",
            },
        ),
        input_quality_metadata=_input_quality(candidate),
    )


def _validate_stream_snapshot(decoded: DecodedStream, snapshot: CandidateExtractionSnapshot) -> None:
    if decoded.stream.stream_id != snapshot.result.envelope.stream_id:
        raise ValueError("Candidate snapshot and decoded stream must have the same stream_id.")
    frames = {frame.frame_id: frame for frame in decoded.frames}
    for candidate in snapshot.result.candidates:
        frame = frames.get(candidate.frame_id)
        if frame is None:
            raise ValueError("Every candidate frame must exist in the decoded stream.")
        if candidate.frame_size != frame.image_size:
            raise ValueError("Candidate frame_size must match the decoded frame.")


def _validate_provider_config(
    provider: DinoV2BatchProviderProtocol,
    config: DinoV2RepresentationConfig,
) -> None:
    if provider.model_spec != config.model_spec:
        raise ValueError("Provider model spec must match the representation semantic config.")
    if provider.requested_device != config.device_policy:
        raise ValueError("Provider requested device must match config runtime policy.")
    if provider.batch_size != config.batch_size:
        raise ValueError("Provider batch size must match config runtime policy.")


def _validate_provider_output(output: DinoV2ProviderOutput, expected_count: int) -> None:
    if not isinstance(output, DinoV2ProviderOutput):
        raise TypeError("provider output must be DinoV2ProviderOutput.")
    embeddings = output.embeddings
    if embeddings.dtype != np.dtype(np.float32):
        raise ValueError("provider embeddings must use float32.")
    if embeddings.shape != (expected_count, DINO_EMBEDDING_DIMENSION):
        raise ValueError("provider embeddings have an unexpected shape.")
    if not np.isfinite(embeddings).all():
        raise ValueError("provider embeddings contain non-finite values.")
    if not np.allclose(np.linalg.norm(embeddings.astype(np.float64), axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("provider embeddings are not L2-normalized.")


def _input_quality(candidate: CandidateRecord) -> InputQualityMetadata:
    details: dict[str, object] = {
        "candidate_source": candidate.candidate_source,
        "candidate_geometry_schema_id": candidate.geometry.feature_schema_id,
        "candidate_geometry_producer_version": candidate.geometry.producer_version,
        "candidate_geometry_config_digest": candidate.geometry.config_digest,
        "candidate_geometry": dict(candidate.geometry.values),
    }
    if candidate.mask is not None:
        details.update(
            {
                "mask_validity_status": candidate.mask.validity_status.value,
                "mask_ref": candidate.mask.mask_ref,
                "mask_digest": candidate.mask.mask_digest,
                "mask_producer_version": candidate.mask.producer_version,
            }
        )
    return InputQualityMetadata(
        candidate_confidence=candidate.candidate_confidence,
        quality_flags=candidate.quality_flags,
        details=details,
    )


def _warning(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: DinoV2RepresentationConfig,
    *,
    code: str,
    message: str,
    metadata: Mapping[str, object],
) -> WarningRecord:
    return WarningRecord(
        record_id=f"warn_repr_{candidate.candidate_id}_{code.casefold()}",
        schema_version=DINO_WARNING_SCHEMA_VERSION,
        stream_id=candidate.envelope.stream_id,
        code=code,
        stage="representation",
        message=message,
        producer=config.producer,
        context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
        metadata=metadata,
        provenance_refs=_provenance_refs(candidate, snapshot, config),
        upstream_warning_ids=_upstream_warning_ids(candidate),
        upstream_error_ids=_upstream_error_ids(candidate),
    )


def _error(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: DinoV2RepresentationConfig,
    code: str,
    message: str,
    metadata: Mapping[str, object],
) -> ErrorRecord:
    return ErrorRecord(
        record_id=f"err_repr_{candidate.candidate_id}_{code.casefold()}",
        schema_version=DINO_ERROR_SCHEMA_VERSION,
        stream_id=candidate.envelope.stream_id,
        code=code,
        stage="representation",
        message=message,
        producer=config.producer,
        context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
        metadata=metadata,
        provenance_refs=_provenance_refs(candidate, snapshot, config),
        upstream_warning_ids=_upstream_warning_ids(candidate),
        upstream_error_ids=_upstream_error_ids(candidate),
    )


def _record_id(candidate: CandidateRecord, config: DinoV2RepresentationConfig) -> str:
    digest = config.config_digest.split(":", 1)[-1]
    return f"representation:{candidate.candidate_id}:{config.variant}:{digest}"


def _provenance_refs(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: DinoV2RepresentationConfig,
) -> tuple[str, ...]:
    refs = [snapshot.result.envelope.record_id, candidate.candidate_id]
    if (
        config.variant == DINO_MASK_NEUTRAL_VARIANT
        and candidate.mask is not None
        and candidate.mask.mask_ref is not None
    ):
        refs.append(candidate.mask.mask_ref)
    return _ordered_unique(tuple(refs))


def _upstream_warning_ids(candidate: CandidateRecord) -> tuple[str, ...]:
    return _ordered_unique(
        (*candidate.envelope.warning_ids, *(candidate.mask.warning_ids if candidate.mask else ()))
    )


def _upstream_error_ids(candidate: CandidateRecord) -> tuple[str, ...]:
    return _ordered_unique(
        (*candidate.envelope.error_ids, *(candidate.mask.error_ids if candidate.mask else ()))
    )


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _bbox_details(candidate: CandidateRecord) -> Mapping[str, float]:
    return {
        "x": candidate.bbox.x,
        "y": candidate.bbox.y,
        "width": candidate.bbox.width,
        "height": candidate.bbox.height,
    }


__all__ = [
    "DINO_CONFIG_SCHEMA_ID",
    "DINO_ERROR_SCHEMA_VERSION",
    "DINO_PREPROCESSING_VERSION",
    "DINO_RECORD_SCHEMA_VERSION",
    "DINO_REPRESENTATION_TYPE",
    "DINO_REPRESENTATION_VERSION",
    "DINO_WARNING_SCHEMA_VERSION",
    "DinoV2RepresentationBatch",
    "DinoV2RepresentationConfig",
    "build_dinov2_representations",
]
