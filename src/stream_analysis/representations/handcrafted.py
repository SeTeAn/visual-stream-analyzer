"""Versioned handcrafted representations for extracted candidates.

The module is deliberately representation-only: it does not read annotations,
create candidates, assign visual types, use position in visual features, or
apply matching thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import cv2
import numpy as np

from ..candidates import CandidateExtractionSnapshot, CandidateMaskRecord
from ..config import StageConfig, semantic_config_digest
from ..contracts import (
    CandidateRecord,
    ErrorRecord,
    HandcraftedFeatureGroup,
    HandcraftedPayload,
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


HANDCRAFTED_MASK_VARIANT: Final = "handcrafted_mask_v1"
HANDCRAFTED_BBOX_VARIANT: Final = "handcrafted_bbox_v1"
HANDCRAFTED_REPRESENTATION_TYPE: Final = "handcrafted_features"
HANDCRAFTED_CONFIG_SCHEMA_ID: Final = (
    "stream_analysis.handcrafted_representation_config.v1"
)
HANDCRAFTED_RECORD_SCHEMA_VERSION: Final = "handcrafted-representation-record-1.0"
HANDCRAFTED_WARNING_SCHEMA_VERSION: Final = "handcrafted-representation-warning-1.0"
HANDCRAFTED_ERROR_SCHEMA_VERSION: Final = "handcrafted-representation-error-1.0"
HANDCRAFTED_PROVIDER_VERSION: Final = "1.0"
HANDCRAFTED_PREPROCESSING_VERSION: Final = "1.0"
HANDCRAFTED_MASK_REPRESENTATION_VERSION: Final = "handcrafted-mask-1.0"
HANDCRAFTED_BBOX_REPRESENTATION_VERSION: Final = "handcrafted-bbox-1.0"
HANDCRAFTED_MASK_FEATURE_SCHEMA_ID: Final = "handcrafted_mask_v1"
HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID: Final = "handcrafted_bbox_v1"
CIELAB_CONVERTER_ID: Final = "srgb_d65_cielab_reference_v1"

HandcraftedVariant = Literal["handcrafted_mask_v1", "handcrafted_bbox_v1"]

MASK_SHAPE_FEATURE_NAMES: Final = (
    "hc_elongation",
    "hc_circularity",
    "hc_solidity",
    "hc_hole_count_norm",
    "hc_hole_area_ratio",
    "hc_radial_mass_1",
    "hc_radial_mass_2",
    "hc_radial_mass_3",
    "hc_radial_mass_4",
)
MASK_STRUCTURE_FEATURE_NAMES: Final = (
    "hc_edge_density",
    "hc_radial_edge_1",
    "hc_radial_edge_2",
    "hc_radial_edge_3",
    "hc_radial_edge_4",
)
MASK_COLOR_FEATURE_NAMES: Final = ("hc_lab_l", "hc_lab_a", "hc_lab_b")
MASK_SIZE_FEATURE_NAMES: Final = ("hc_area_ratio_to_frame",)
MASK_FEATURE_NAMES: Final = (
    *MASK_SHAPE_FEATURE_NAMES,
    *MASK_STRUCTURE_FEATURE_NAMES,
    *MASK_COLOR_FEATURE_NAMES,
    *MASK_SIZE_FEATURE_NAMES,
)

BBOX_SHAPE_FEATURE_NAMES: Final = ("hcb_box_elongation",)
BBOX_STRUCTURE_FEATURE_NAMES: Final = (
    "hcb_edge_density",
    "hcb_radial_edge_1",
    "hcb_radial_edge_2",
    "hcb_radial_edge_3",
    "hcb_radial_edge_4",
)
BBOX_COLOR_FEATURE_NAMES: Final = ("hcb_lab_l", "hcb_lab_a", "hcb_lab_b")
BBOX_SIZE_FEATURE_NAMES: Final = ("hcb_bbox_area_ratio_to_frame",)
BBOX_FEATURE_NAMES: Final = (
    *BBOX_SHAPE_FEATURE_NAMES,
    *BBOX_STRUCTURE_FEATURE_NAMES,
    *BBOX_COLOR_FEATURE_NAMES,
    *BBOX_SIZE_FEATURE_NAMES,
)

FEATURE_GROUP_NAMES: Final = ("shape", "structure", "color", "size")

_LAB_MATRIX: Final = np.array(
    (
        (0.4124564, 0.3575761, 0.1804375),
        (0.2126729, 0.7151522, 0.0721750),
        (0.0193339, 0.1191920, 0.9503041),
    ),
    dtype=np.float64,
)
_LAB_WHITE: Final = np.array((0.95047, 1.0, 1.08883), dtype=np.float64)


def _validate_weights(
    values: tuple[float, ...],
    expected_length: int,
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != expected_length:
        raise ValueError(f"{field_name} must contain {expected_length} weights.")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} values must be real numbers.")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{field_name} values must be finite and non-negative.")
        converted.append(number)
    if not math.isclose(sum(converted), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field_name} values must sum to 1.")
    return tuple(converted)


@dataclass(frozen=True, slots=True)
class HandcraftedRepresentationConfig:
    """Versioned semantic configuration for exactly one homogeneous variant."""

    variant: HandcraftedVariant = HANDCRAFTED_MASK_VARIANT
    config_version: str = "1.1"
    normalized_size: int = 128
    gaussian_kernel_size: int = 3
    canny_low_threshold: float = 50.0
    canny_high_threshold: float = 150.0
    mask_dilation_pixels: int = 1
    radial_bin_count: int = 4
    significant_hole_area_ratio: float = 0.0
    max_holes: int = 4
    min_mask_pixels: int = 2
    epsilon: float = 1e-12
    size_ratio_cap: float = 3.0
    shape_component_weights: tuple[float, float] = (0.7, 0.3)
    structure_component_weights: tuple[float, float] = (0.4, 0.6)
    mask_group_weights: tuple[float, float, float, float] = (0.40, 0.40, 0.15, 0.05)
    bbox_group_weights: tuple[float, float, float, float] = (0.25, 0.55, 0.15, 0.05)

    def __post_init__(self) -> None:
        if self.variant not in {HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT}:
            raise ValueError("variant must be handcrafted_mask_v1 or handcrafted_bbox_v1.")
        StageConfig(
            stage_id="representation",
            schema_id=HANDCRAFTED_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
        )
        for field_name in (
            "normalized_size",
            "gaussian_kernel_size",
            "radial_bin_count",
            "max_holes",
            "min_mask_pixels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be odd.")
        if self.radial_bin_count != 4:
            raise ValueError("radial_bin_count must be 4 for the v1 feature schemas.")
        if (
            isinstance(self.mask_dilation_pixels, bool)
            or not isinstance(self.mask_dilation_pixels, int)
            or self.mask_dilation_pixels < 0
        ):
            raise ValueError("mask_dilation_pixels must be a non-negative integer.")
        for field_name in (
            "canny_low_threshold",
            "canny_high_threshold",
            "significant_hole_area_ratio",
            "epsilon",
            "size_ratio_cap",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite.")
        if not 0.0 <= self.canny_low_threshold < self.canny_high_threshold <= 255.0:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255.")
        if not 0.0 <= self.significant_hole_area_ratio < 1.0:
            raise ValueError("significant_hole_area_ratio must be in [0, 1).")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.size_ratio_cap <= 1.0:
            raise ValueError("size_ratio_cap must be greater than 1.")
        object.__setattr__(
            self,
            "shape_component_weights",
            _validate_weights(self.shape_component_weights, 2, "shape_component_weights"),
        )
        object.__setattr__(
            self,
            "structure_component_weights",
            _validate_weights(
                self.structure_component_weights,
                2,
                "structure_component_weights",
            ),
        )
        object.__setattr__(
            self,
            "mask_group_weights",
            _validate_weights(self.mask_group_weights, 4, "mask_group_weights"),
        )
        object.__setattr__(
            self,
            "bbox_group_weights",
            _validate_weights(self.bbox_group_weights, 4, "bbox_group_weights"),
        )

    @property
    def feature_schema_id(self) -> str:
        if self.variant == HANDCRAFTED_MASK_VARIANT:
            return HANDCRAFTED_MASK_FEATURE_SCHEMA_ID
        return HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID

    @property
    def representation_version(self) -> str:
        if self.variant == HANDCRAFTED_MASK_VARIANT:
            return HANDCRAFTED_MASK_REPRESENTATION_VERSION
        return HANDCRAFTED_BBOX_REPRESENTATION_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.variant == HANDCRAFTED_MASK_VARIANT:
            return MASK_FEATURE_NAMES
        return BBOX_FEATURE_NAMES

    @property
    def group_weights(self) -> Mapping[str, float]:
        values = (
            self.mask_group_weights
            if self.variant == HANDCRAFTED_MASK_VARIANT
            else self.bbox_group_weights
        )
        return MappingProxyType(dict(zip(FEATURE_GROUP_NAMES, values, strict=True)))

    @property
    def semantic_parameters(self) -> Mapping[str, object]:
        edge: dict[str, object] = {
            "normalized_size": self.normalized_size,
            "gaussian_kernel_size": self.gaussian_kernel_size,
            "canny_low_threshold": self.canny_low_threshold,
            "canny_high_threshold": self.canny_high_threshold,
            "rgb_interpolation": "cv2.INTER_LINEAR",
        }
        parameters: dict[str, object] = {
            "variant": self.variant,
            "feature_schema_id": self.feature_schema_id,
            "representation_version": self.representation_version,
            "mask_policy": (
                "mask_required"
                if self.variant == HANDCRAFTED_MASK_VARIANT
                else "bbox_only"
            ),
            "color": {
                "converter": CIELAB_CONVERTER_ID,
                "input_dtype": "uint8",
                "input_range": (0, 255),
                "working_dtype": "float64",
            },
            "edge": edge,
            "radial_bin_count": self.radial_bin_count,
            "epsilon": self.epsilon,
            "size_ratio_cap": self.size_ratio_cap,
            "structure_component_weights": self.structure_component_weights,
            "group_weights": self.group_weights,
            "feature_names": self.feature_names,
        }
        if self.variant == HANDCRAFTED_MASK_VARIANT:
            edge.update(
                {
                    "mask_dilation_pixels": self.mask_dilation_pixels,
                    "mask_interpolation": "cv2.INTER_NEAREST",
                }
            )
            parameters.update(
                {
                    "significant_hole_area_ratio": self.significant_hole_area_ratio,
                    "max_holes": self.max_holes,
                    "min_mask_pixels": self.min_mask_pixels,
                    "shape_component_weights": self.shape_component_weights,
                }
            )
        return parameters

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="representation",
            schema_id=HANDCRAFTED_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters=self.semantic_parameters,
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="representation",
            producer_version=self.representation_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True)
class HandcraftedRepresentationBatch:
    """Homogeneous in-memory records plus newly produced diagnostics."""

    variant: HandcraftedVariant
    feature_schema_id: str
    semantic_config_digest: str
    records: tuple[RepresentationRecord, ...]
    warnings: tuple[WarningRecord, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in {HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT}:
            raise ValueError("Invalid handcrafted variant.")
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
            if record.payload is not None and record.payload.feature_schema_id != self.feature_schema_id:
                raise ValueError("Every payload must use the batch feature schema.")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class _ExpectedRepresentationFailure(Exception):
    code: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict)


def srgb_uint8_to_normalized_cielab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 pixels to normalized D65 CIELAB in float64.

    The final axis must have length three.  The returned array is independent
    from the input and follows the representation's reference color conversion.
    """

    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a NumPy ndarray.")
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have a final channel dimension of length 3.")
    if rgb.dtype != np.dtype(np.uint8):
        raise TypeError("rgb dtype must be uint8.")
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ _LAB_MATRIX.T
    scaled = xyz / _LAB_WHITE
    delta = 6.0 / 29.0
    transformed = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(transformed, dtype=np.float64)
    lab[..., 0] = 116.0 * transformed[..., 1] - 16.0
    lab[..., 1] = 500.0 * (transformed[..., 0] - transformed[..., 1])
    lab[..., 2] = 200.0 * (transformed[..., 1] - transformed[..., 2])
    normalized = np.empty_like(lab, dtype=np.float64)
    normalized[..., 0] = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    normalized[..., 1] = np.clip((lab[..., 1] + 128.0) / 255.0, 0.0, 1.0)
    normalized[..., 2] = np.clip((lab[..., 2] + 128.0) / 255.0, 0.0, 1.0)
    return normalized


def build_handcrafted_representations(
    decoded_stream: DecodedStream,
    candidate_snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig | None = None,
) -> HandcraftedRepresentationBatch:
    """Build one homogeneous representation record per canonical candidate."""

    if not isinstance(decoded_stream, DecodedStream):
        raise TypeError("decoded_stream must be DecodedStream.")
    if not isinstance(candidate_snapshot, CandidateExtractionSnapshot):
        raise TypeError("candidate_snapshot must be CandidateExtractionSnapshot.")
    effective_config = config or HandcraftedRepresentationConfig()
    if not isinstance(effective_config, HandcraftedRepresentationConfig):
        raise TypeError("config must be HandcraftedRepresentationConfig or None.")
    _validate_stream_snapshot(decoded_stream, candidate_snapshot)

    frames = {frame.frame_id: frame for frame in decoded_stream.frames}
    records: list[RepresentationRecord] = []
    warnings: list[WarningRecord] = []
    errors: list[ErrorRecord] = []
    for candidate in candidate_snapshot.result.candidates:
        frame = frames[candidate.frame_id]
        try:
            if effective_config.variant == HANDCRAFTED_MASK_VARIANT:
                try:
                    mask_record = candidate_snapshot.mask_for_candidate(candidate.candidate_id)
                except KeyError as error:
                    raise _ExpectedRepresentationFailure(
                        "MASK_REQUIRED",
                        "Mask-aware handcrafted representation requires an in-memory candidate mask.",
                    ) from error
                payload, preprocessing_details, structure_valid = _mask_payload(
                    frame,
                    candidate,
                    mask_record,
                    effective_config,
                )
            else:
                payload, preprocessing_details, structure_valid = _bbox_payload(
                    frame,
                    candidate,
                    effective_config,
                )
            new_warning: WarningRecord | None = None
            if not structure_valid and effective_config.variant == HANDCRAFTED_MASK_VARIANT:
                new_warning = _warning(
                    candidate=candidate,
                    snapshot=candidate_snapshot,
                    config=effective_config,
                    code="EMPTY_EDGE_STRUCTURE",
                    message="No edge pixels were available for the mask-aware structure group.",
                )
                warnings.append(new_warning)
            records.append(
                _valid_record(
                    candidate=candidate,
                    snapshot=candidate_snapshot,
                    config=effective_config,
                    payload=payload,
                    preprocessing_details=preprocessing_details,
                    new_warning=new_warning,
                )
            )
        except _ExpectedRepresentationFailure as failure:
            error = _error(
                candidate=candidate,
                snapshot=candidate_snapshot,
                config=effective_config,
                failure=failure,
            )
            errors.append(error)
            records.append(
                _invalid_record(
                    candidate=candidate,
                    snapshot=candidate_snapshot,
                    config=effective_config,
                    error=error,
                )
            )

    return HandcraftedRepresentationBatch(
        variant=effective_config.variant,
        feature_schema_id=effective_config.feature_schema_id,
        semantic_config_digest=effective_config.config_digest,
        records=tuple(records),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _validate_stream_snapshot(
    decoded_stream: DecodedStream,
    snapshot: CandidateExtractionSnapshot,
) -> None:
    stream_id = decoded_stream.stream.stream_id
    if snapshot.result.envelope.stream_id != stream_id:
        raise ValueError("Candidate snapshot and decoded stream must have the same stream_id.")
    frames = {frame.frame_id: frame for frame in decoded_stream.frames}
    for candidate in snapshot.result.candidates:
        frame = frames.get(candidate.frame_id)
        if frame is None:
            raise ValueError("Every candidate frame must exist in the decoded stream.")
        if candidate.frame_size != frame.image_size:
            raise ValueError("Candidate frame_size must match the decoded frame.")
        _bbox_slices(candidate)


def _frame_array(frame: DecodedFrame) -> np.ndarray:
    array = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    )
    return array


def _bbox_slices(candidate: CandidateRecord) -> tuple[slice, slice]:
    values = (candidate.bbox.x, candidate.bbox.y, candidate.bbox.width, candidate.bbox.height)
    integers = tuple(int(value) for value in values)
    if any(float(integer) != float(value) for integer, value in zip(integers, values, strict=True)):
        raise ValueError("Handcrafted v1 requires integer-valued canonical bbox coordinates.")
    x, y, width, height = integers
    if width <= 0 or height <= 0:
        raise ValueError("Canonical bbox must have positive dimensions.")
    return slice(y, y + height), slice(x, x + width)


def _candidate_crop(frame: DecodedFrame, candidate: CandidateRecord) -> np.ndarray:
    ys, xs = _bbox_slices(candidate)
    crop = _frame_array(frame)[ys, xs]
    if crop.shape != (int(candidate.bbox.height), int(candidate.bbox.width), 3):
        raise ValueError("Candidate bbox crop must match the canonical bbox dimensions.")
    return np.ascontiguousarray(crop)


def _mask_payload(
    frame: DecodedFrame,
    candidate: CandidateRecord,
    mask_record: CandidateMaskRecord,
    config: HandcraftedRepresentationConfig,
) -> tuple[HandcraftedPayload, Mapping[str, object], bool]:
    if candidate.mask is None or candidate.mask.validity_status is not ValidityStatus.VALID:
        raise _ExpectedRepresentationFailure(
            "MASK_REQUIRED",
            "Mask-aware handcrafted representation requires a valid candidate mask.",
        )
    if mask_record.candidate_id != candidate.candidate_id:
        raise _ExpectedRepresentationFailure("MASK_CANDIDATE_MISMATCH", "Mask candidate_id mismatch.")
    if mask_record.frame_id != candidate.frame_id:
        raise _ExpectedRepresentationFailure("MASK_FRAME_MISMATCH", "Mask frame_id mismatch.")
    if mask_record.coordinate_bbox != candidate.bbox:
        raise _ExpectedRepresentationFailure("MASK_BBOX_MISMATCH", "Mask bbox mismatch.")
    mask = np.asarray(mask_record.mask, dtype=np.bool_)
    foreground_area = int(np.count_nonzero(mask))
    if foreground_area < config.min_mask_pixels:
        raise _ExpectedRepresentationFailure(
            "MASK_TOO_SMALL",
            "Mask-aware representation requires more foreground pixels.",
            {"foreground_pixel_count": foreground_area},
        )
    component_count = _foreground_component_count(mask)
    if component_count != 1:
        raise _ExpectedRepresentationFailure(
            "MASK_COMPONENT_COUNT_INVALID",
            "Candidate mask must contain exactly one foreground component.",
            {"component_count": component_count},
        )
    crop = _candidate_crop(frame, candidate)
    shape = _mask_shape_features(mask, config)
    color = _median_lab_features(
        crop[mask],
        names=MASK_COLOR_FEATURE_NAMES,
    )
    edge_values, edge_valid, fill_source = _mask_edge_features(crop, mask, config)
    size = {
        "hc_area_ratio_to_frame": foreground_area / float(candidate.frame_size.area),
    }
    groups = (
        HandcraftedFeatureGroup(group_name="shape", values=shape),
        HandcraftedFeatureGroup(
            group_name="structure",
            values=edge_values,
            valid=edge_valid,
            invalid_reason=None if edge_valid else "empty_edge_set",
        ),
        HandcraftedFeatureGroup(group_name="color", values=color),
        HandcraftedFeatureGroup(group_name="size", values=size),
    )
    details: Mapping[str, object] = {
        **_common_preprocessing_details(candidate, config),
        "feature_order": MASK_FEATURE_NAMES,
        "letterbox_fill_source": fill_source,
        "mask_ref": mask_record.mask_ref,
        "mask_digest": mask_record.mask_digest,
        "mask_producer_version": candidate.mask.producer_version,
        "mask_policy": "mask_required",
        "edge_profile_valid": edge_valid,
        "mask_interpolation": "cv2.INTER_NEAREST",
        "mask_dilation_pixels": config.mask_dilation_pixels,
        "significant_hole_area_ratio": config.significant_hole_area_ratio,
        "max_holes": config.max_holes,
        "min_mask_pixels": config.min_mask_pixels,
        "shape_component_weights": config.shape_component_weights,
    }
    return (
        HandcraftedPayload(
            feature_schema_id=HANDCRAFTED_MASK_FEATURE_SCHEMA_ID,
            feature_groups=groups,
        ),
        details,
        edge_valid,
    )


def _bbox_payload(
    frame: DecodedFrame,
    candidate: CandidateRecord,
    config: HandcraftedRepresentationConfig,
) -> tuple[HandcraftedPayload, Mapping[str, object], bool]:
    crop = _candidate_crop(frame, candidate)
    width = float(candidate.bbox.width)
    height = float(candidate.bbox.height)
    shape = {"hcb_box_elongation": min(width, height) / max(width, height)}
    color = _median_lab_features(crop.reshape(-1, 3), names=BBOX_COLOR_FEATURE_NAMES)
    edge_values, edge_profile_valid, fill_source = _bbox_edge_features(crop, config)
    size = {
        "hcb_bbox_area_ratio_to_frame": candidate.bbox.area / candidate.frame_size.area,
    }
    groups = (
        HandcraftedFeatureGroup(group_name="shape", values=shape),
        HandcraftedFeatureGroup(group_name="structure", values=edge_values),
        HandcraftedFeatureGroup(group_name="color", values=color),
        HandcraftedFeatureGroup(group_name="size", values=size),
    )
    details: Mapping[str, object] = {
        **_common_preprocessing_details(candidate, config),
        "feature_order": BBOX_FEATURE_NAMES,
        "letterbox_fill_source": fill_source,
        "mask_policy": "bbox_only",
        "edge_profile_valid": edge_profile_valid,
    }
    return (
        HandcraftedPayload(
            feature_schema_id=HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID,
            feature_groups=groups,
        ),
        details,
        True,
    )


def _common_preprocessing_details(
    candidate: CandidateRecord,
    config: HandcraftedRepresentationConfig,
) -> Mapping[str, object]:
    return {
        "variant": config.variant,
        "feature_schema_id": config.feature_schema_id,
        "crop_bbox": {
            "x": candidate.bbox.x,
            "y": candidate.bbox.y,
            "width": candidate.bbox.width,
            "height": candidate.bbox.height,
        },
        "cielab_converter": CIELAB_CONVERTER_ID,
        "cielab_input_dtype": "uint8",
        "cielab_input_range": (0, 255),
        "cielab_working_dtype": "float64",
        "normalized_size": config.normalized_size,
        "rgb_interpolation": "cv2.INTER_LINEAR",
        "edge_operator": "cv2.Canny",
        "gaussian_kernel_size": config.gaussian_kernel_size,
        "canny_low_threshold": config.canny_low_threshold,
        "canny_high_threshold": config.canny_high_threshold,
        "radial_bin_count": config.radial_bin_count,
        "config_digest": config.config_digest,
    }


def _median_lab_features(
    pixels: np.ndarray,
    *,
    names: tuple[str, str, str],
) -> Mapping[str, float]:
    if pixels.size == 0:
        raise _ExpectedRepresentationFailure("EMPTY_COLOR_INPUT", "No pixels available for color features.")
    lab = srgb_uint8_to_normalized_cielab(np.asarray(pixels, dtype=np.uint8))
    medians = np.median(lab.reshape(-1, 3), axis=0)
    return {name: float(value) for name, value in zip(names, medians, strict=True)}


def _foreground_component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(count - 1)


def _mask_shape_features(
    mask: np.ndarray,
    config: HandcraftedRepresentationConfig,
) -> Mapping[str, float]:
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    points = np.column_stack((xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5))
    centered = points - np.mean(points, axis=0)
    covariance = (centered.T @ centered) / float(area)
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda2, lambda1 = float(eigenvalues[0]), float(eigenvalues[1])
    elongation = math.sqrt(max(0.0, lambda2) / (max(0.0, lambda1) + config.epsilon))

    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if len(contours) != 1:
        raise _ExpectedRepresentationFailure(
            "MASK_EXTERNAL_CONTOUR_INVALID",
            "Candidate mask must have exactly one external contour.",
            {"external_contour_count": len(contours)},
        )
    perimeter = float(cv2.arcLength(contours[0], True))
    if perimeter <= 0.0:
        raise _ExpectedRepresentationFailure(
            "MASK_PERIMETER_INVALID",
            "Candidate mask perimeter must be positive.",
        )
    circularity = min(1.0, max(0.0, 4.0 * math.pi * area / (perimeter**2)))

    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32).reshape(-1, 1, 2))
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = int(np.count_nonzero(hull_mask))
    solidity = min(1.0, area / float(max(hull_area, 1)))

    all_hole_areas = _hole_areas(mask)
    outer_area = area + sum(all_hole_areas)
    significant_holes = tuple(
        hole_area
        for hole_area in all_hole_areas
        if hole_area / float(max(outer_area, 1)) >= config.significant_hole_area_ratio
    )
    hole_count_norm = min(len(significant_holes), config.max_holes) / float(config.max_holes)
    hole_area_ratio = sum(significant_holes) / float(max(outer_area, 1))
    radial = _radial_profile(mask, config.radial_bin_count, config.epsilon)
    return {
        "hc_elongation": _bounded(elongation),
        "hc_circularity": _bounded(circularity),
        "hc_solidity": _bounded(solidity),
        "hc_hole_count_norm": _bounded(hole_count_norm),
        "hc_hole_area_ratio": _bounded(hole_area_ratio),
        **{f"hc_radial_mass_{index + 1}": float(value) for index, value in enumerate(radial)},
    }


def _hole_areas(mask: np.ndarray) -> tuple[int, ...]:
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=8,
    )
    border_labels = set(int(value) for value in labels[0, :])
    border_labels.update(int(value) for value in labels[-1, :])
    border_labels.update(int(value) for value in labels[:, 0])
    border_labels.update(int(value) for value in labels[:, -1])
    return tuple(
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if label not in border_labels
    )


def _radial_profile(mask: np.ndarray, bins: int, epsilon: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    center_x = float(np.mean(xs.astype(np.float64) + 0.5))
    center_y = float(np.mean(ys.astype(np.float64) + 0.5))
    distances = np.hypot(xs.astype(np.float64) + 0.5 - center_x, ys.astype(np.float64) + 0.5 - center_y)
    maximum = float(np.max(distances))
    normalized = distances / (maximum + epsilon)
    indices = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.float64)
    profile = counts / float(len(indices))
    return profile


def _mask_edge_features(
    crop: np.ndarray,
    mask: np.ndarray,
    config: HandcraftedRepresentationConfig,
) -> tuple[Mapping[str, float], bool, str]:
    fill, fill_source = _letterbox_fill(crop, mask)
    square, mask_square, _content = _letterbox(crop, mask, fill, config.normalized_size)
    assert mask_square is not None
    edge_map = _edge_map(square, config)
    if config.mask_dilation_pixels > 0:
        size = 2 * config.mask_dilation_pixels + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        roi = cv2.dilate(mask_square.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        roi = mask_square
    selected = edge_map & roi
    density = np.count_nonzero(selected) / float(max(np.count_nonzero(roi), 1))
    profile = _edge_radial_profile(selected, roi, mask_square, config)
    valid = profile is not None
    radial = np.zeros(config.radial_bin_count, dtype=np.float64) if profile is None else profile
    return (
        {
            "hc_edge_density": _bounded(density),
            **{f"hc_radial_edge_{index + 1}": float(value) for index, value in enumerate(radial)},
        },
        valid,
        fill_source,
    )


def _bbox_edge_features(
    crop: np.ndarray,
    config: HandcraftedRepresentationConfig,
) -> tuple[Mapping[str, float], bool, str]:
    fill, fill_source = _letterbox_fill(crop, None)
    square, _mask_square, content = _letterbox(crop, None, fill, config.normalized_size)
    edge_map = _edge_map(square, config)
    selected = edge_map & content
    density = np.count_nonzero(selected) / float(max(np.count_nonzero(content), 1))
    profile = _edge_radial_profile(selected, content, content, config)
    valid = profile is not None
    radial = np.zeros(config.radial_bin_count, dtype=np.float64) if profile is None else profile
    return (
        {
            "hcb_edge_density": _bounded(density),
            **{f"hcb_radial_edge_{index + 1}": float(value) for index, value in enumerate(radial)},
        },
        valid,
        fill_source,
    )


def _letterbox_fill(crop: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, str]:
    if mask is not None and np.any(~mask):
        pixels = crop[~mask]
        source = "mask_background_median"
    else:
        pixels = np.concatenate(
            (crop[0, :, :], crop[-1, :, :], crop[:, 0, :], crop[:, -1, :]),
            axis=0,
        )
        source = "bbox_border_median"
    fill = np.rint(np.median(pixels.astype(np.float64), axis=0)).astype(np.uint8)
    return fill, source


def _letterbox(
    crop: np.ndarray,
    mask: np.ndarray | None,
    fill: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    height, width = crop.shape[:2]
    scale = min(size / float(width), size / float(height))
    new_width = min(size, max(1, int(round(width * scale))))
    new_height = min(size, max(1, int(round(height * scale))))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    square = np.empty((size, size, 3), dtype=np.uint8)
    square[:, :] = fill
    x0 = (size - new_width) // 2
    y0 = (size - new_height) // 2
    square[y0:y0 + new_height, x0:x0 + new_width] = resized
    content = np.zeros((size, size), dtype=bool)
    content[y0:y0 + new_height, x0:x0 + new_width] = True
    mask_square: np.ndarray | None = None
    if mask is not None:
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_square = np.zeros((size, size), dtype=bool)
        mask_square[y0:y0 + new_height, x0:x0 + new_width] = resized_mask.astype(bool)
    return square, mask_square, content


def _edge_map(square: np.ndarray, config: HandcraftedRepresentationConfig) -> np.ndarray:
    lab = srgb_uint8_to_normalized_cielab(square)
    luminance = np.rint(lab[..., 0] * 255.0).astype(np.uint8)
    if config.gaussian_kernel_size > 1:
        luminance = cv2.GaussianBlur(
            luminance,
            (config.gaussian_kernel_size, config.gaussian_kernel_size),
            0,
        )
    return cv2.Canny(
        luminance,
        config.canny_low_threshold,
        config.canny_high_threshold,
    ).astype(bool)


def _edge_radial_profile(
    selected_edges: np.ndarray,
    roi: np.ndarray,
    center_mask: np.ndarray,
    config: HandcraftedRepresentationConfig,
) -> np.ndarray | None:
    edge_y, edge_x = np.nonzero(selected_edges)
    if len(edge_x) == 0:
        return None
    center_y_values, center_x_values = np.nonzero(center_mask)
    if len(center_x_values) == 0:
        return None
    center_x = float(np.mean(center_x_values.astype(np.float64) + 0.5))
    center_y = float(np.mean(center_y_values.astype(np.float64) + 0.5))
    roi_y, roi_x = np.nonzero(roi)
    max_radius = float(
        np.max(
            np.hypot(
                roi_x.astype(np.float64) + 0.5 - center_x,
                roi_y.astype(np.float64) + 0.5 - center_y,
            )
        )
    )
    distances = np.hypot(
        edge_x.astype(np.float64) + 0.5 - center_x,
        edge_y.astype(np.float64) + 0.5 - center_y,
    )
    normalized = distances / (max_radius + config.epsilon)
    indices = np.minimum(
        (normalized * config.radial_bin_count).astype(np.int64),
        config.radial_bin_count - 1,
    )
    counts = np.bincount(indices, minlength=config.radial_bin_count).astype(np.float64)
    return counts / float(len(indices))


def _valid_record(
    *,
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig,
    payload: HandcraftedPayload,
    preprocessing_details: Mapping[str, object],
    new_warning: WarningRecord | None,
) -> RepresentationRecord:
    warning_ids = _ordered_unique(
        (*candidate.envelope.warning_ids, *(candidate.mask.warning_ids if candidate.mask else ()), *(() if new_warning is None else (new_warning.record_id,)))
    )
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=_record_id(candidate, config),
            schema_version=HANDCRAFTED_RECORD_SCHEMA_VERSION,
            stream_id=candidate.envelope.stream_id,
            producer=config.producer,
            context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
            warning_ids=warning_ids,
            provenance_refs=_provenance_refs(candidate, snapshot, config),
        ),
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        family=RepresentationFamily.HANDCRAFTED,
        representation_type=HANDCRAFTED_REPRESENTATION_TYPE,
        representation_version=config.representation_version,
        input_variant=config.variant,
        semantic_config_digest=config.config_digest,
        payload=payload,
        preprocessing_metadata=VersionedMetadata(
            identifier=f"{config.variant}.preprocessing",
            version=HANDCRAFTED_PREPROCESSING_VERSION,
            details=preprocessing_details,
        ),
        provider_metadata=VersionedMetadata(
            identifier="numpy_opencv_handcrafted_provider",
            version=HANDCRAFTED_PROVIDER_VERSION,
            details={"execution": "deterministic_cpu"},
        ),
        model_metadata=None,
        runtime_metadata=RuntimeMetadata(
            runtime_id="handcrafted_cpu",
            details={"numpy_version": np.__version__, "opencv_version": cv2.__version__},
        ),
        input_quality_metadata=_input_quality(candidate),
    )


def _invalid_record(
    *,
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig,
    error: ErrorRecord,
) -> RepresentationRecord:
    warning_ids = _ordered_unique(
        (*candidate.envelope.warning_ids, *(candidate.mask.warning_ids if candidate.mask else ()))
    )
    details: dict[str, object] = {
        **_common_preprocessing_details(candidate, config),
        "feature_order": config.feature_names,
        "mask_policy": "mask_required" if config.variant == HANDCRAFTED_MASK_VARIANT else "bbox_only",
        "failure_code": error.code,
    }
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=_record_id(candidate, config),
            schema_version=HANDCRAFTED_RECORD_SCHEMA_VERSION,
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
        family=RepresentationFamily.HANDCRAFTED,
        representation_type=HANDCRAFTED_REPRESENTATION_TYPE,
        representation_version=config.representation_version,
        input_variant=config.variant,
        semantic_config_digest=config.config_digest,
        payload=None,
        preprocessing_metadata=VersionedMetadata(
            identifier=f"{config.variant}.preprocessing",
            version=HANDCRAFTED_PREPROCESSING_VERSION,
            details=details,
        ),
        provider_metadata=VersionedMetadata(
            identifier="numpy_opencv_handcrafted_provider",
            version=HANDCRAFTED_PROVIDER_VERSION,
        ),
        model_metadata=None,
        runtime_metadata=RuntimeMetadata(
            runtime_id="handcrafted_cpu",
            details={"numpy_version": np.__version__, "opencv_version": cv2.__version__},
        ),
        input_quality_metadata=_input_quality(candidate),
    )


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
    *,
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig,
    code: str,
    message: str,
) -> WarningRecord:
    return WarningRecord(
        record_id=f"warn_repr_{candidate.candidate_id}_{code.casefold()}",
        schema_version=HANDCRAFTED_WARNING_SCHEMA_VERSION,
        stream_id=candidate.envelope.stream_id,
        code=code,
        stage="representation",
        message=message,
        producer=config.producer,
        context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
        provenance_refs=_provenance_refs(candidate, snapshot, config),
        upstream_warning_ids=_upstream_warning_ids(candidate),
        upstream_error_ids=_upstream_error_ids(candidate),
    )


def _error(
    *,
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig,
    failure: _ExpectedRepresentationFailure,
) -> ErrorRecord:
    return ErrorRecord(
        record_id=f"err_repr_{candidate.candidate_id}_{failure.code.casefold()}",
        schema_version=HANDCRAFTED_ERROR_SCHEMA_VERSION,
        stream_id=candidate.envelope.stream_id,
        code=failure.code,
        stage="representation",
        message=failure.message,
        producer=config.producer,
        context=StageContext(frame_id=candidate.frame_id, candidate_id=candidate.candidate_id),
        metadata=failure.metadata,
        provenance_refs=_provenance_refs(candidate, snapshot, config),
        upstream_warning_ids=_upstream_warning_ids(candidate),
        upstream_error_ids=_upstream_error_ids(candidate),
    )


def _upstream_warning_ids(candidate: CandidateRecord) -> tuple[str, ...]:
    mask_warning_ids = candidate.mask.warning_ids if candidate.mask is not None else ()
    return _ordered_unique((*candidate.envelope.warning_ids, *mask_warning_ids))


def _upstream_error_ids(candidate: CandidateRecord) -> tuple[str, ...]:
    mask_error_ids = candidate.mask.error_ids if candidate.mask is not None else ()
    return _ordered_unique((*candidate.envelope.error_ids, *mask_error_ids))


def _record_id(candidate: CandidateRecord, config: HandcraftedRepresentationConfig) -> str:
    digest = config.config_digest.split(":", 1)[-1]
    return f"representation:{candidate.candidate_id}:{config.variant}:{digest}"


def _provenance_refs(
    candidate: CandidateRecord,
    snapshot: CandidateExtractionSnapshot,
    config: HandcraftedRepresentationConfig,
) -> tuple[str, ...]:
    refs = [snapshot.result.envelope.record_id, candidate.candidate_id]
    if (
        config.variant == HANDCRAFTED_MASK_VARIANT
        and candidate.mask is not None
        and candidate.mask.mask_ref is not None
    ):
        refs.append(candidate.mask.mask_ref)
    return _ordered_unique(tuple(refs))


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _bounded(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _ExpectedRepresentationFailure("NONFINITE_FEATURE", "A handcrafted feature is non-finite.")
    return min(1.0, max(0.0, number))


__all__ = [
    "BBOX_COLOR_FEATURE_NAMES",
    "BBOX_FEATURE_NAMES",
    "BBOX_SHAPE_FEATURE_NAMES",
    "BBOX_SIZE_FEATURE_NAMES",
    "BBOX_STRUCTURE_FEATURE_NAMES",
    "CIELAB_CONVERTER_ID",
    "FEATURE_GROUP_NAMES",
    "HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID",
    "HANDCRAFTED_BBOX_REPRESENTATION_VERSION",
    "HANDCRAFTED_BBOX_VARIANT",
    "HANDCRAFTED_CONFIG_SCHEMA_ID",
    "HANDCRAFTED_MASK_FEATURE_SCHEMA_ID",
    "HANDCRAFTED_MASK_REPRESENTATION_VERSION",
    "HANDCRAFTED_MASK_VARIANT",
    "HANDCRAFTED_REPRESENTATION_TYPE",
    "MASK_COLOR_FEATURE_NAMES",
    "MASK_FEATURE_NAMES",
    "MASK_SHAPE_FEATURE_NAMES",
    "MASK_SIZE_FEATURE_NAMES",
    "MASK_STRUCTURE_FEATURE_NAMES",
    "HandcraftedRepresentationBatch",
    "HandcraftedRepresentationConfig",
    "build_handcrafted_representations",
    "srgb_uint8_to_normalized_cielab",
]
