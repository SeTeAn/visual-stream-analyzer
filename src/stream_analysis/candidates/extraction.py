"""Production candidate extraction assembly for controlled smooth-background streams."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from ..config import StageConfig, semantic_config_digest
from ..contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    FrameCandidateDiagnostics,
    GeometryFeatureMetadata,
    ImageSize,
    MaskReference,
    ProducerProvenance,
    RecordEnvelope,
    Severity,
    StageContext,
    ValidityStatus,
    WarningRecord,
)
from ..contracts.common import _require_identifier, _require_version
from ..input import DecodedFrame, DecodedStream
from .background import (
    BackgroundModel,
    BackgroundModelConfig,
    HysteresisMaskConfig,
    MorphologyCleanupConfig,
    cleanup_morphology,
    compute_residual,
    estimate_background_model,
    hysteresis_threshold_mask,
)
from .diagnostics import (
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    candidate_mask_digest,
)


CANDIDATE_EXTRACTION_CONFIG_SCHEMA_ID = "stream_analysis.candidate_extraction_config.v3"
CANDIDATE_EXTRACTION_RESULT_SCHEMA_VERSION = "candidate-extraction-result-1.0"
CANDIDATE_RECORD_SCHEMA_VERSION = "candidate-record-1.0"
FRAME_CANDIDATE_DIAGNOSTICS_SCHEMA_VERSION = "frame-candidate-diagnostics-1.0"
CANDIDATE_WARNING_SCHEMA_VERSION = "candidate-warning-1.0"
CANDIDATE_GEOMETRY_SCHEMA_ID = "candidate_geometry_v1"
DEFAULT_CANDIDATE_SOURCE = "controlled_background_components_v1"

ComponentConnectivity = Literal[4, 8]
ComponentPolicy = Literal["allow", "warn", "reject"]
BackgroundStrategy = Literal["stream_model", "frame_border_median", "first_frame"]


@dataclass(frozen=True, slots=True)
class CandidateExtractionConfig:
    """Versioned explicit configuration for candidate extraction."""

    config_version: str = "3.2"
    background_strategy: BackgroundStrategy = "frame_border_median"
    background: BackgroundModelConfig = field(default_factory=BackgroundModelConfig)
    background_border_ratio: float = 0.08
    hysteresis: HysteresisMaskConfig = field(
        default_factory=lambda: HysteresisMaskConfig(
            weak_threshold=10.0,
            strong_threshold=16.0,
            connectivity=8,
        )
    )
    morphology: MorphologyCleanupConfig = field(
        default_factory=lambda: MorphologyCleanupConfig(
            open_kernel_size=3,
            close_kernel_size=7,
            kernel_shape="ellipse",
            iterations=1,
            fill_holes=False,
        )
    )
    component_connectivity: ComponentConnectivity = 8
    min_component_area_ratio: float = 0.0
    max_component_area_ratio: float | None = None
    bbox_padding: int = 1
    min_bbox_side_ratio: float = 0.045
    giant_component_area_ratio: float = 0.90
    giant_component_policy: ComponentPolicy = "reject"
    border_touching_policy: ComponentPolicy = "warn"
    candidate_source: str = DEFAULT_CANDIDATE_SOURCE
    candidate_source_version: str = "1.0"
    geometry_feature_schema_id: str = CANDIDATE_GEOMETRY_SCHEMA_ID
    mask_ref_prefix: str = "mask"

    def __post_init__(self) -> None:
        _require_version(self.config_version, "config_version")
        if self.background_strategy not in {
            "stream_model",
            "frame_border_median",
            "first_frame",
        }:
            raise ValueError(
                "background_strategy must be 'stream_model', "
                "'frame_border_median' or 'first_frame'."
            )
        if not isinstance(self.background, BackgroundModelConfig):
            raise TypeError("background must be BackgroundModelConfig.")
        border_ratio = _finite_float(
            self.background_border_ratio,
            "background_border_ratio",
        )
        if border_ratio <= 0.0 or border_ratio > 0.5:
            raise ValueError("background_border_ratio must be in (0, 0.5].")
        object.__setattr__(self, "background_border_ratio", border_ratio)
        if not isinstance(self.hysteresis, HysteresisMaskConfig):
            raise TypeError("hysteresis must be HysteresisMaskConfig.")
        if not isinstance(self.morphology, MorphologyCleanupConfig):
            raise TypeError("morphology must be MorphologyCleanupConfig.")
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        min_area_ratio = _bounded_ratio(
            self.min_component_area_ratio,
            "min_component_area_ratio",
            allow_zero=True,
        )
        object.__setattr__(self, "min_component_area_ratio", min_area_ratio)
        if self.max_component_area_ratio is not None:
            max_area_ratio = _bounded_ratio(
                self.max_component_area_ratio,
                "max_component_area_ratio",
                allow_zero=False,
            )
            if max_area_ratio < min_area_ratio:
                raise ValueError(
                    "max_component_area_ratio must be at least "
                    "min_component_area_ratio."
                )
            object.__setattr__(self, "max_component_area_ratio", max_area_ratio)
        _non_negative_int(self.bbox_padding, "bbox_padding")
        min_bbox_side_ratio = _bounded_ratio(
            self.min_bbox_side_ratio,
            "min_bbox_side_ratio",
            allow_zero=True,
        )
        object.__setattr__(self, "min_bbox_side_ratio", min_bbox_side_ratio)
        ratio = _finite_float(
            self.giant_component_area_ratio,
            "giant_component_area_ratio",
        )
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError("giant_component_area_ratio must be in (0, 1].")
        object.__setattr__(self, "giant_component_area_ratio", ratio)
        if self.giant_component_policy not in {"allow", "warn", "reject"}:
            raise ValueError("giant_component_policy must be allow, warn or reject.")
        if self.border_touching_policy not in {"allow", "warn", "reject"}:
            raise ValueError("border_touching_policy must be allow, warn or reject.")
        _require_identifier(self.candidate_source, "candidate_source")
        _require_version(self.candidate_source_version, "candidate_source_version")
        _require_identifier(self.geometry_feature_schema_id, "geometry_feature_schema_id")
        if not isinstance(self.mask_ref_prefix, str) or not self.mask_ref_prefix.strip():
            raise ValueError("mask_ref_prefix must be a non-empty token.")
        if any(char.isspace() for char in self.mask_ref_prefix):
            raise ValueError("mask_ref_prefix must not contain whitespace.")

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="candidate_extraction",
            schema_id=CANDIDATE_EXTRACTION_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters=self.semantic_parameters,
        )

    @property
    def semantic_parameters(self) -> Mapping[str, object]:
        return {
            "background": {
                "aggregation": self.background.aggregation,
                "smoothing_kernel_size": self.background.smoothing_kernel_size,
                "smoothing_sigma": self.background.smoothing_sigma,
            },
            "background_border_ratio": self.background_border_ratio,
            "background_strategy": self.background_strategy,
            "bbox_padding": self.bbox_padding,
            "border_touching_policy": self.border_touching_policy,
            "candidate_source": self.candidate_source,
            "candidate_source_version": self.candidate_source_version,
            "component_connectivity": self.component_connectivity,
            "geometry_feature_schema_id": self.geometry_feature_schema_id,
            "giant_component_area_ratio": self.giant_component_area_ratio,
            "giant_component_policy": self.giant_component_policy,
            "hysteresis": {
                "connectivity": self.hysteresis.connectivity,
                "strong_threshold": self.hysteresis.strong_threshold,
                "weak_threshold": self.hysteresis.weak_threshold,
            },
            "mask_ref_prefix": self.mask_ref_prefix,
            "max_component_area_ratio": self.max_component_area_ratio,
            "min_bbox_side_ratio": self.min_bbox_side_ratio,
            "min_component_area_ratio": self.min_component_area_ratio,
            "morphology": {
                "close_kernel_size": self.morphology.close_kernel_size,
                "fill_holes": self.morphology.fill_holes,
                "iterations": self.morphology.iterations,
                "kernel_shape": self.morphology.kernel_shape,
                "open_kernel_size": self.morphology.open_kernel_size,
            },
        }

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="candidate_extraction",
            producer_version=self.candidate_source_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )


@dataclass(frozen=True, slots=True)
class _ComponentProposal:
    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    border_touching: bool
    giant_component: bool
    bbox: BBox
    local_mask: np.ndarray
    mask_centroid_x: float
    mask_centroid_y: float
    contour_count: int
    hole_count: int
    reject_reasons: tuple[str, ...]
    warning_codes: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.reject_reasons

    @property
    def sort_key(self) -> tuple[int, int, int, int, int, int]:
        return (self.y, self.x, self.y + self.height, self.x + self.width, self.area, self.label)


def extract_candidates(
    decoded_stream: DecodedStream,
    config: CandidateExtractionConfig | None = None,
) -> CandidateExtractionSnapshot:
    """Extract canonical candidates and local masks from a decoded stream."""

    if not isinstance(decoded_stream, DecodedStream):
        raise TypeError("decoded_stream must be DecodedStream.")
    effective_config = config or CandidateExtractionConfig()
    if not isinstance(effective_config, CandidateExtractionConfig):
        raise TypeError("config must be CandidateExtractionConfig or None.")

    frames = decoded_stream.frames
    frame_arrays = tuple(_frame_to_rgb_array(frame) for frame in frames)
    stream_background = None
    if effective_config.background_strategy == "stream_model":
        stream_background = estimate_background_model(
            frame_arrays, effective_config.background
        )
    elif effective_config.background_strategy == "first_frame":
        stream_background = estimate_background_model(
            frame_arrays[:1], effective_config.background
        )
    producer = effective_config.producer
    stream_id = decoded_stream.stream.stream_id

    warnings: list[WarningRecord] = []
    diagnostics: list[FrameCandidateDiagnostics] = []
    candidates: list[CandidateRecord] = []
    masks: dict[str, CandidateMaskRecord] = {}

    for frame, frame_rgb in zip(frames, frame_arrays, strict=True):
        frame_warning_ids_before = len(warnings)
        frame_candidates, frame_masks, frame_summary = _extract_frame_candidates(
            frame=frame,
            frame_rgb=frame_rgb,
            background=_background_for_frame(
                frame_rgb=frame_rgb,
                stream_background=stream_background,
                config=effective_config,
            ),
            config=effective_config,
            producer=producer,
            stream_id=stream_id,
            warnings=warnings,
        )
        candidates.extend(frame_candidates)
        masks.update(frame_masks)

        if not frame_candidates:
            warning = _warning(
                record_id=f"warn_{frame.frame_id}_no_candidates",
                code="NO_CANDIDATES_IN_FRAME",
                message="No accepted candidates were produced for this frame.",
                stream_id=stream_id,
                producer=producer,
                frame_id=frame.frame_id,
                metadata={"frame_index": frame.record.index},
            )
            warnings.append(warning)

        frame_warning_ids = tuple(
            warning.record_id for warning in warnings[frame_warning_ids_before:]
        )
        frame_summary = {
            **frame_summary,
            "accepted_candidate_count": len(frame_candidates),
            "warning_count": len(frame_warning_ids),
        }
        diagnostics.append(
            FrameCandidateDiagnostics(
                envelope=RecordEnvelope(
                    record_id=f"candidate_diag_{frame.frame_id}",
                    schema_version=FRAME_CANDIDATE_DIAGNOSTICS_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame.frame_id),
                    warning_ids=frame_warning_ids,
                ),
                frame_id=frame.frame_id,
                frame_index=frame.record.index,
                image_size=frame.image_size,
                summary=frame_summary,
            )
        )

    result = CandidateExtractionResult(
        envelope=RecordEnvelope(
            record_id=f"candidate_result_{stream_id}",
            schema_version=CANDIDATE_EXTRACTION_RESULT_SCHEMA_VERSION,
            stream_id=stream_id,
            producer=producer,
            context=StageContext(),
            warning_ids=tuple(warning.record_id for warning in warnings),
        ),
        extractor_source=effective_config.candidate_source,
        candidates=tuple(candidates),
        frame_diagnostics=tuple(diagnostics),
        warnings=tuple(warnings),
    )
    return CandidateExtractionSnapshot(result=result, masks=masks)


def _extract_frame_candidates(
    *,
    frame: DecodedFrame,
    frame_rgb: np.ndarray,
    background: object,
    config: CandidateExtractionConfig,
    producer: ProducerProvenance,
    stream_id: str,
    warnings: list[WarningRecord],
) -> tuple[list[CandidateRecord], dict[str, CandidateMaskRecord], dict[str, object]]:
    residual = compute_residual(frame_rgb, background)  # type: ignore[arg-type]
    raw_mask = hysteresis_threshold_mask(residual, config.hysteresis)
    cleaned_mask = cleanup_morphology(raw_mask, config.morphology)

    proposals = _component_proposals(
        cleaned_mask=cleaned_mask,
        config=config,
        frame_size=frame.image_size,
    )
    raw_component_count = _component_count(raw_mask, config.component_connectivity)
    accepted = sorted((proposal for proposal in proposals if proposal.accepted), key=lambda item: item.sort_key)

    frame_warnings_by_label: dict[int, tuple[str, ...]] = {}
    for proposal in proposals:
        proposal_warning_ids: list[str] = []
        if proposal.giant_component and config.giant_component_policy in {"warn", "reject"}:
            rejected_as_giant = "giant" in proposal.reject_reasons
            warning = _warning(
                record_id=f"warn_{frame.frame_id}_giant_{proposal.label:03d}",
                code="GIANT_COMPONENT_REJECTED" if rejected_as_giant else "GIANT_COMPONENT",
                message=(
                    "A giant foreground component was rejected by the giant-component policy."
                    if rejected_as_giant
                    else "A foreground component met the giant-component warning threshold."
                ),
                stream_id=stream_id,
                producer=producer,
                frame_id=frame.frame_id,
                metadata=_component_warning_metadata(proposal, frame.record.index),
            )
            warnings.append(warning)
            proposal_warning_ids.append(warning.record_id)
        if proposal.border_touching and config.border_touching_policy in {"warn", "reject"}:
            rejected_as_border = "border" in proposal.reject_reasons
            warning = _warning(
                record_id=f"warn_{frame.frame_id}_border_{proposal.label:03d}",
                code=(
                    "BORDER_TOUCHING_COMPONENT_REJECTED"
                    if rejected_as_border
                    else "BORDER_TOUCHING_COMPONENT"
                ),
                message=(
                    "A border-touching component was rejected by the border policy."
                    if rejected_as_border
                    else "A component met the border-touching warning condition."
                ),
                stream_id=stream_id,
                producer=producer,
                frame_id=frame.frame_id,
                metadata=_component_warning_metadata(proposal, frame.record.index),
            )
            warnings.append(warning)
            proposal_warning_ids.append(warning.record_id)
        if proposal_warning_ids:
            frame_warnings_by_label[proposal.label] = tuple(proposal_warning_ids)

    records: list[CandidateRecord] = []
    masks: dict[str, CandidateMaskRecord] = {}
    for ordinal, proposal in enumerate(accepted, start=1):
        candidate_id = f"cand_{frame.frame_id}_{ordinal:03d}"
        warning_ids = frame_warnings_by_label.get(proposal.label, ())
        quality_flags = proposal.warning_codes
        geometry = _geometry_metadata(
            proposal=proposal,
            frame_size=frame.image_size,
            config=config,
        )
        mask_ref = f"{config.mask_ref_prefix}:{candidate_id}"
        mask_digest = candidate_mask_digest(proposal.local_mask)
        mask_reference = MaskReference(
            mask_ref=mask_ref,
            mask_digest=mask_digest,
            producer_version=config.candidate_source_version,
            coordinate_bbox=proposal.bbox,
            validity_status=ValidityStatus.VALID,
            warning_ids=warning_ids,
        )
        candidate = CandidateRecord(
            envelope=RecordEnvelope(
                record_id=candidate_id,
                schema_version=CANDIDATE_RECORD_SCHEMA_VERSION,
                stream_id=stream_id,
                producer=producer,
                context=StageContext(frame_id=frame.frame_id, candidate_id=candidate_id),
                warning_ids=warning_ids,
            ),
            candidate_id=candidate_id,
            frame_id=frame.frame_id,
            frame_index=frame.record.index,
            frame_size=frame.image_size,
            bbox=proposal.bbox,
            center=proposal.bbox.center,
            geometry=geometry,
            candidate_source=config.candidate_source,
            mask=mask_reference,
            candidate_confidence=None,
            quality_flags=quality_flags,
        )
        records.append(candidate)
        masks[mask_ref] = CandidateMaskRecord(
            mask_ref=mask_ref,
            mask_digest=mask_digest,
            candidate_id=candidate_id,
            frame_id=frame.frame_id,
            coordinate_bbox=proposal.bbox,
            mask=proposal.local_mask,
        )

    rejection_counts = _rejection_counts(proposals)
    summary: dict[str, object] = {
        "background_frame_count": int(background.frame_count),  # type: ignore[attr-defined]
        "background_strategy": config.background_strategy,
        "background_border_ratio": config.background_border_ratio,
        "raw_foreground_pixel_count": int(np.count_nonzero(raw_mask)),
        "cleaned_foreground_pixel_count": int(np.count_nonzero(cleaned_mask)),
        "raw_component_count": raw_component_count,
        "cleaned_component_count": len(proposals),
        "rejected_component_count": len([proposal for proposal in proposals if not proposal.accepted]),
        "rejected_small_count": rejection_counts["small"],
        "rejected_large_count": rejection_counts["large"],
        "rejected_border_count": rejection_counts["border"],
        "rejected_giant_count": rejection_counts["giant"],
        "border_touching_component_count": len([proposal for proposal in proposals if proposal.border_touching]),
        "giant_component_count": len([proposal for proposal in proposals if proposal.giant_component]),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
        "residual_mean": float(np.mean(residual)),
    }
    return records, masks, summary


def _background_for_frame(
    *,
    frame_rgb: np.ndarray,
    stream_background: BackgroundModel | None,
    config: CandidateExtractionConfig,
) -> BackgroundModel:
    if config.background_strategy in {"stream_model", "first_frame"}:
        if stream_background is None:
            raise ValueError(
                "stream_background is required for stream_model and first_frame strategies."
            )
        return stream_background
    return BackgroundModel(
        rgb=_frame_border_median_background(
            frame_rgb,
            border_ratio=config.background_border_ratio,
        ),
        frame_count=1,
        config=config.background,
    )


def _frame_border_median_background(
    frame_rgb: np.ndarray,
    *,
    border_ratio: float,
) -> np.ndarray:
    height, width, _channels = frame_rgb.shape
    border_width = max(1, int(round(min(height, width) * border_ratio)))
    border_mask = np.zeros((height, width), dtype=np.bool_)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = frame_rgb[border_mask].astype(np.float32, copy=False)
    background_color = np.median(border_pixels, axis=0).astype(np.float32)
    background = np.empty(frame_rgb.shape, dtype=np.float32)
    background[:, :] = background_color
    return background


def _component_proposals(
    *,
    cleaned_mask: np.ndarray,
    config: CandidateExtractionConfig,
    frame_size: ImageSize,
) -> tuple[_ComponentProposal, ...]:
    binary = np.ascontiguousarray(cleaned_mask.astype(np.uint8))
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=config.component_connectivity,
    )
    proposals: list[_ComponentProposal] = []
    frame_area = frame_size.area
    for label in range(1, stats.shape[0]):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        border_touching = x == 0 or y == 0 or x + width >= frame_size.width or y + height >= frame_size.height
        area_ratio = area / frame_area
        giant_component = area_ratio >= config.giant_component_area_ratio
        bbox = _padded_bbox(
            x=x,
            y=y,
            width=width,
            height=height,
            frame_size=frame_size,
            padding=config.bbox_padding,
        )
        local_mask = _local_component_mask(labels, label, bbox)
        centroid_x, centroid_y = _mask_centroid(labels, label)
        contour_count, hole_count = _contour_metadata(local_mask)
        reject_reasons = _reject_reasons(
            area_ratio=area_ratio,
            width_ratio=width / frame_size.width,
            height_ratio=height / frame_size.height,
            border_touching=border_touching,
            giant_component=giant_component,
            config=config,
        )
        warning_codes = _quality_flags(
            border_touching=border_touching,
            giant_component=giant_component,
            config=config,
        )
        proposals.append(
            _ComponentProposal(
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                border_touching=border_touching,
                giant_component=giant_component,
                bbox=bbox,
                local_mask=local_mask,
                mask_centroid_x=centroid_x,
                mask_centroid_y=centroid_y,
                contour_count=contour_count,
                hole_count=hole_count,
                reject_reasons=reject_reasons,
                warning_codes=warning_codes,
            )
        )
    return tuple(proposals)


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=connectivity,
    )
    return int(count - 1)


def _reject_reasons(
    *,
    area_ratio: float,
    width_ratio: float,
    height_ratio: float,
    border_touching: bool,
    giant_component: bool,
    config: CandidateExtractionConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if area_ratio < config.min_component_area_ratio:
        reasons.append("small")
    if (
        config.max_component_area_ratio is not None
        and area_ratio > config.max_component_area_ratio
    ):
        reasons.append("large")
    if (
        width_ratio < config.min_bbox_side_ratio
        or height_ratio < config.min_bbox_side_ratio
    ):
        reasons.append("small_bbox")
    if giant_component and config.giant_component_policy == "reject":
        reasons.append("giant")
    if border_touching and config.border_touching_policy == "reject":
        reasons.append("border")
    return tuple(reasons)


def _quality_flags(
    *,
    border_touching: bool,
    giant_component: bool,
    config: CandidateExtractionConfig,
) -> tuple[str, ...]:
    flags: list[str] = []
    if border_touching and config.border_touching_policy == "warn":
        flags.append("BORDER_TOUCHING")
    if giant_component and config.giant_component_policy == "warn":
        flags.append("GIANT_COMPONENT")
    return tuple(flags)


def _rejection_counts(proposals: tuple[_ComponentProposal, ...]) -> dict[str, int]:
    return {
        "small": sum("small" in proposal.reject_reasons or "small_bbox" in proposal.reject_reasons for proposal in proposals),
        "large": sum("large" in proposal.reject_reasons for proposal in proposals),
        "border": sum("border" in proposal.reject_reasons for proposal in proposals),
        "giant": sum("giant" in proposal.reject_reasons for proposal in proposals),
    }


def _padded_bbox(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    frame_size: ImageSize,
    padding: int,
) -> BBox:
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(frame_size.width, x + width + padding)
    bottom = min(frame_size.height, y + height + padding)
    return BBox(float(left), float(top), float(right - left), float(bottom - top))


def _local_component_mask(labels: np.ndarray, label: int, bbox: BBox) -> np.ndarray:
    left = int(bbox.x)
    top = int(bbox.y)
    right = int(bbox.right)
    bottom = int(bbox.bottom)
    local = labels[top:bottom, left:right] == label
    result = np.ascontiguousarray(local.astype(np.bool_, copy=True))
    result.flags["WRITEABLE"] = False
    return result


def _mask_centroid(labels: np.ndarray, label: int) -> tuple[float, float]:
    ys, xs = np.nonzero(labels == label)
    if len(xs) == 0:
        raise ValueError("component label has no pixels.")
    return float(np.mean(xs + 0.5)), float(np.mean(ys + 0.5))


def _contour_metadata(local_mask: np.ndarray) -> tuple[int, int]:
    contours, hierarchy = cv2.findContours(
        local_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return 0, 0
    hole_count = sum(1 for item in hierarchy[0] if int(item[3]) != -1)
    return len(contours), int(hole_count)


def _geometry_metadata(
    *,
    proposal: _ComponentProposal,
    frame_size: ImageSize,
    config: CandidateExtractionConfig,
) -> GeometryFeatureMetadata:
    bbox_area = proposal.bbox.area
    frame_area = frame_size.area
    return GeometryFeatureMetadata(
        feature_schema_id=config.geometry_feature_schema_id,
        producer_version=config.candidate_source_version,
        config_digest=config.config_digest,
        values={
            "area": proposal.area,
            "area_ratio_to_frame": proposal.area / frame_area,
            "aspect_ratio": proposal.bbox.width / proposal.bbox.height,
            "bbox_area": bbox_area,
            "bbox_height": proposal.bbox.height,
            "bbox_width": proposal.bbox.width,
            "border_touching": proposal.border_touching,
            "contour_count": proposal.contour_count,
            "foreground_fill_ratio": proposal.area / bbox_area,
            "giant_component": proposal.giant_component,
            "hole_count": proposal.hole_count,
            "mask_centroid_x": proposal.mask_centroid_x,
            "mask_centroid_y": proposal.mask_centroid_y,
            "raw_bbox_height": proposal.height,
            "raw_bbox_width": proposal.width,
        },
    )


def _component_warning_metadata(
    proposal: _ComponentProposal,
    frame_index: int,
) -> Mapping[str, object]:
    return {
        "area": proposal.area,
        "bbox_height": proposal.bbox.height,
        "bbox_width": proposal.bbox.width,
        "border_touching": proposal.border_touching,
        "frame_index": frame_index,
        "giant_component": proposal.giant_component,
        "component_number": proposal.label,
    }


def _warning(
    *,
    record_id: str,
    code: str,
    message: str,
    stream_id: str,
    producer: ProducerProvenance,
    frame_id: str,
    metadata: Mapping[str, object],
) -> WarningRecord:
    return WarningRecord(
        record_id=record_id,
        schema_version=CANDIDATE_WARNING_SCHEMA_VERSION,
        stream_id=stream_id,
        code=code,
        stage="candidate_extraction",
        message=message,
        producer=producer,
        severity=Severity.WARNING,
        context=StageContext(frame_id=frame_id),
        metadata=metadata,
    )


def _frame_to_rgb_array(frame: DecodedFrame) -> np.ndarray:
    width = frame.image_size.width
    height = frame.image_size.height
    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape((height, width, 3))


def _positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


def _non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


def _bounded_ratio(value: float, field_name: str, *, allow_zero: bool) -> float:
    converted = _finite_float(value, field_name)
    lower_bound_valid = converted >= 0.0 if allow_zero else converted > 0.0
    if not lower_bound_valid or converted > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{field_name} must be in {interval}.")
    return converted


__all__ = [
    "CANDIDATE_EXTRACTION_CONFIG_SCHEMA_ID",
    "CANDIDATE_EXTRACTION_RESULT_SCHEMA_VERSION",
    "CANDIDATE_GEOMETRY_SCHEMA_ID",
    "CANDIDATE_RECORD_SCHEMA_VERSION",
    "CANDIDATE_WARNING_SCHEMA_VERSION",
    "DEFAULT_CANDIDATE_SOURCE",
    "FRAME_CANDIDATE_DIAGNOSTICS_SCHEMA_VERSION",
    "BackgroundStrategy",
    "CandidateExtractionConfig",
    "ComponentConnectivity",
    "ComponentPolicy",
    "extract_candidates",
]
