"""Shared Grounding DINO -> SAM2 -> DINOv2 stream-analysis engine."""

from __future__ import annotations

import gc
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..candidates import (
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    DEFAULT_MASK_CONTAINMENT,
    DEFAULT_MINIMUM_COVERED_MASKS,
    PositionedCandidateMask,
    TEMPORAL_ADJACENT_RADIUS,
    TEMPORAL_BASE_SCORE,
    TEMPORAL_CONTAINMENT_REJECT,
    TEMPORAL_LOW_SCORE,
    TEMPORAL_PROFILE_ID,
    TEMPORAL_SUPPORT_IOU,
    TemporalProposal,
    candidate_mask_digest,
    resolve_aggregate_masks,
    temporal_support_selection,
)
from ..contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    FrameCandidateDiagnostics,
    FramePair,
    GeometryFeatureMetadata,
    MaskReference,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
    ValidityStatus,
)
from ..input import DecodedStream
from ..matching import (
    EventConfig,
    GroupingConfig,
    MatchingConfig,
    PairScoringConfig,
    build_change_events,
    group_recurring_visual_types,
    match_neighboring_frames,
)
from ..representations import (
    DINO_MASK_NEUTRAL_VARIANT,
    DinoV2CosineScorer,
    DinoV2RepresentationConfig,
    LocalDinoV2Provider,
    build_dinov2_representations,
)
from .config import RuntimeAssets, RuntimeProfile
from .grounding_dino import (
    RAW_POSTPROCESS_FLOOR,
    FilterProfile,
    GeometryProfile,
    PromptProfile,
    RawPrediction,
    frame_array,
    infer_stream,
    load_model_processor,
    select_predictions_for_profile,
    surface_like,
)
from .sam2 import GroundedMaskResult, MaskCleanupConfig, load_local_sam2_bbox_refiner
from .workspace import RuntimeWorkspace, StoredEmbeddings, StoredMask


class RuntimePipelineError(RuntimeError):
    """Raised when a fixed inference stage cannot produce a complete result."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectorStreamOutcome:
    decoded: DecodedStream
    raw_predictions: tuple[RawPrediction, ...]
    postprocess_records: tuple[Mapping[str, Any], ...]
    accepted_candidates: tuple[Mapping[str, Any], ...]
    temporal_support: Mapping[str, Any]
    runtime: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeMaskRecord:
    candidate_id: str
    frame_id: str
    frame_index: int
    score: float
    phrase: str
    source_bbox: BBox
    result: GroundedMaskResult
    artifacts: StoredMask


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskStreamOutcome:
    decoded: DecodedStream
    records: tuple[RuntimeMaskRecord, ...]
    resolved_candidate_ids: tuple[str, ...]
    aggregate_resolution: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisStreamOutcome:
    decoded: DecodedStream
    snapshot: CandidateExtractionSnapshot
    embeddings: np.ndarray
    embedding_artifact: StoredEmbeddings
    representation_batch: Any
    matching_results: tuple[Any, ...]
    grouping: Any
    event_batch: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeStreamOutcome:
    detector: DetectorStreamOutcome
    masks: MaskStreamOutcome
    analysis: AnalysisStreamOutcome


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeOutcome:
    profile_id: str
    profile_sha256: str
    streams: tuple[RuntimeStreamOutcome, ...]
    detector_elapsed_seconds: float
    mask_elapsed_seconds: float
    analysis_elapsed_seconds: float


def _mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise RuntimePipelineError(f"{context}.{key} must be an object")
    return result


def _bbox_payload(value: BBox) -> dict[str, float]:
    return {
        "x": value.x,
        "y": value.y,
        "width": value.width,
        "height": value.height,
    }


def bbox_from_payload(value: Mapping[str, Any]) -> BBox:
    try:
        return BBox(
            float(value["x"]),
            float(value["y"]),
            float(value["width"]),
            float(value["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimePipelineError(f"invalid detector bbox: {error}") from error


def detector_profiles(profile: RuntimeProfile) -> tuple[PromptProfile, GeometryProfile]:
    payload = _mapping(profile.pipeline, "candidate_extraction", "pipeline")
    if (
        float(payload["raw_score_floor"]) != RAW_POSTPROCESS_FLOOR
        or float(payload["score_threshold"]) != TEMPORAL_BASE_SCORE
        or RAW_POSTPROCESS_FLOOR > TEMPORAL_LOW_SCORE
    ):
        raise RuntimePipelineError("runtime detector gates differ from the fixed implementation")
    geometry_payload = _mapping(payload, "geometry_filter", "candidate extraction")
    return (
        PromptProfile("p_object", str(payload["prompt"]), True, "fixed_runtime_profile"),
        GeometryProfile(
            str(geometry_payload["policy_id"]),
            min_area_ratio=float(geometry_payload["reject_when_area_ratio_at_least"]),
            min_span_ratio=float(geometry_payload["and_either_span_ratio_at_least"]),
        ),
    )


def profile_detector_predictions(
    raw: Sequence[RawPrediction],
    *,
    decoded: DecodedStream,
    profile: RuntimeProfile,
    prompt: PromptProfile,
    geometry: GeometryProfile,
) -> tuple[dict[str, Any], ...]:
    payload = _mapping(profile.pipeline, "candidate_extraction", "pipeline")
    surface_geometries = (
        GeometryProfile("g_frame_080", min_area_ratio=0.80),
        geometry,
    )
    profiled = select_predictions_for_profile(
        raw,
        filter_profile=FilterProfile(
            score_threshold=float(payload["score_threshold"]),
            class_agnostic_nms_iou=float(payload["class_agnostic_nms_iou"]),
        ),
        geometry=geometry,
        image_sizes={frame.frame_id: frame.image_size for frame in decoded.frames},
        surface_geometries=surface_geometries,
    )
    result: list[dict[str, Any]] = []
    for item in profiled:
        row = item.prediction
        result.append(
            {
                "source_prediction_key": (
                    f"{prompt.prompt_id}:{row.frame_id}:{row.prediction_index:03d}"
                ),
                "candidate_id": (
                    f"grounding-dino-candidate:{prompt.prompt_id}:{geometry.geometry_id}:"
                    f"{row.frame_id}:{row.prediction_index:03d}"
                ),
                "frame_id": row.frame_id,
                "frame_index": row.frame_index,
                "prediction_index": row.prediction_index,
                "score": row.score,
                "phrase": row.phrase,
                "bbox": _bbox_payload(row.bbox),
                "bbox_area_ratio": item.bbox_area_ratio,
                "bbox_width_ratio": item.bbox_width_ratio,
                "bbox_height_ratio": item.bbox_height_ratio,
                "surface_like": item.surface_like,
                "geometry_rejected": item.geometry_rejected,
            }
        )
    return tuple(result)


def temporal_detector_records(
    *,
    raw: Sequence[RawPrediction],
    decoded: DecodedStream,
    prompt: PromptProfile,
    geometry: GeometryProfile,
    baseline_accepted: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    proposals = tuple(
        TemporalProposal(
            frame_id=row.frame_id,
            frame_index=row.frame_index,
            prediction_index=row.prediction_index,
            score=row.score,
            phrase=row.phrase,
            bbox=row.bbox,
        )
        for row in raw
    )
    frame_sizes = {
        frame.frame_id: {
            "width": frame.image_size.width,
            "height": frame.image_size.height,
        }
        for frame in decoded.frames
    }
    selection = temporal_support_selection(
        proposals,
        frame_ids=tuple(frame.frame_id for frame in decoded.frames),
        frame_sizes=frame_sizes,
        nms_iou=0.30,
        min_area_ratio=geometry.min_area_ratio,
        min_span_ratio=geometry.min_span_ratio,
    )
    baseline_by_key = {
        (str(row["frame_id"]), int(row["prediction_index"])): dict(row)
        for row in baseline_accepted
    }
    expected_base_keys = {
        (proposal.frame_id, proposal.prediction_index)
        for proposal in selection.final_proposals
        if proposal.score >= TEMPORAL_BASE_SCORE
    }
    if set(baseline_by_key) != expected_base_keys:
        raise RuntimePipelineError("temporal support did not preserve the base candidates")
    supplemental_keys = {
        (proposal.frame_id, proposal.prediction_index)
        for proposal in selection.supplemental_proposals
    }
    image_sizes = {frame.frame_id: frame.image_size for frame in decoded.frames}
    surface_geometries = (
        GeometryProfile("g_frame_080", min_area_ratio=0.80),
        geometry,
    )
    accepted: list[dict[str, Any]] = []
    for proposal in selection.final_proposals:
        key = (proposal.frame_id, proposal.prediction_index)
        baseline = baseline_by_key.get(key)
        if baseline is not None:
            baseline["selection_role"] = "base_score"
            accepted.append(baseline)
            continue
        if key not in supplemental_keys:
            raise RuntimePipelineError("temporal support emitted an unclassified candidate")
        image_size = image_sizes[proposal.frame_id]
        bbox = proposal.bbox
        accepted.append(
            {
                "source_prediction_key": (
                    f"{prompt.prompt_id}:{proposal.frame_id}:{proposal.prediction_index:03d}"
                ),
                "candidate_id": (
                    f"grounding-dino-temporal-support:{TEMPORAL_PROFILE_ID}:"
                    f"{proposal.frame_id}:{proposal.prediction_index:03d}"
                ),
                "selection_role": "supplemental_temporal_support",
                "frame_id": proposal.frame_id,
                "frame_index": proposal.frame_index,
                "prediction_index": proposal.prediction_index,
                "score": proposal.score,
                "phrase": proposal.phrase,
                "bbox": _bbox_payload(bbox),
                "bbox_area_ratio": bbox.area / (image_size.width * image_size.height),
                "bbox_width_ratio": bbox.width / image_size.width,
                "bbox_height_ratio": bbox.height / image_size.height,
                "surface_like": surface_like(bbox, image_size, surface_geometries),
                "geometry_rejected": False,
            }
        )
    accepted.sort(key=lambda row: (int(row["frame_index"]), str(row["candidate_id"])))
    audit = {
        "enabled": True,
        "profile_id": TEMPORAL_PROFILE_ID,
        "base_score_inclusive": TEMPORAL_BASE_SCORE,
        "supplemental_score_min_inclusive": TEMPORAL_LOW_SCORE,
        "supplemental_score_max_exclusive": TEMPORAL_BASE_SCORE,
        "temporal_support_bbox_iou_at_least": TEMPORAL_SUPPORT_IOU,
        "adjacent_frame_radius": TEMPORAL_ADJACENT_RADIUS,
        "base_containment_reject_at_least": TEMPORAL_CONTAINMENT_REJECT,
        "base_candidate_count": len(baseline_accepted),
        "supplemental_candidate_count": len(selection.supplemental_proposals),
        "final_candidate_count": len(accepted),
        "decisions": list(selection.audit_rows),
    }
    return tuple(accepted), audit


def run_detector_streams(
    decoded_streams: Sequence[DecodedStream],
    *,
    profile: RuntimeProfile,
    assets: RuntimeAssets,
) -> tuple[tuple[DetectorStreamOutcome, ...], float]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimePipelineError("the fixed runtime requires CUDA")
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    prompt, geometry = detector_profiles(profile)
    started = time.perf_counter()
    processor = model = None
    outputs: list[DetectorStreamOutcome] = []
    try:
        processor, model = load_model_processor(
            assets.grounding_dino_directory,
            torch_module=torch,
            device="cuda",
        )
        for decoded in decoded_streams:
            raw, runtime = infer_stream(
                decoded,
                processor=processor,
                model=model,
                torch_module=torch,
                device="cuda",
                prompt=prompt.text,
            )
            records = profile_detector_predictions(
                raw,
                decoded=decoded,
                profile=profile,
                prompt=prompt,
                geometry=geometry,
            )
            baseline = tuple(row for row in records if row["geometry_rejected"] is False)
            accepted, temporal = temporal_detector_records(
                raw=raw,
                decoded=decoded,
                prompt=prompt,
                geometry=geometry,
                baseline_accepted=baseline,
            )
            outputs.append(
                DetectorStreamOutcome(
                    decoded=decoded,
                    raw_predictions=raw,
                    postprocess_records=records,
                    accepted_candidates=accepted,
                    temporal_support=temporal,
                    runtime=runtime,
                )
            )
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        finally:
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
    return tuple(outputs), time.perf_counter() - started


def _cleanup_config(profile: RuntimeProfile) -> MaskCleanupConfig:
    cleanup = _mapping(
        _mapping(profile.pipeline, "mask_refinement", "pipeline"),
        "cleanup",
        "mask refinement",
    )
    if cleanup.get("fill_holes") is not False or cleanup.get(
        "remove_only_when_below_both_thresholds"
    ) is not True:
        raise RuntimePipelineError("mask cleanup policy differs from the fixed implementation")
    return MaskCleanupConfig(
        min_component_pixels=int(cleanup["min_component_pixels"]),
        min_component_area_ratio=float(cleanup["min_component_area_ratio"]),
        connectivity=int(cleanup["connectivity"]),  # type: ignore[arg-type]
    )


def resolve_aggregate_mask_records(
    records: Sequence[RuntimeMaskRecord],
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    by_frame: dict[str, list[RuntimeMaskRecord]] = defaultdict(list)
    for record in records:
        by_frame[record.frame_id].append(record)
    kept_ids: list[str] = []
    frame_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        rows = sorted(by_frame[frame_id], key=lambda item: item.candidate_id)
        resolution = resolve_aggregate_masks(
            tuple(
                PositionedCandidateMask(
                    candidate_id=row.candidate_id,
                    frame_id=frame_id,
                    mask=np.ascontiguousarray(row.result.cleaned_mask, dtype=np.bool_),
                )
                for row in rows
            )
        )
        kept_ids.extend(resolution.kept_candidate_ids)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "input_candidate_count": len(rows),
                "kept_candidate_ids": list(resolution.kept_candidate_ids),
                "removed_candidate_ids": list(resolution.removed_candidate_ids),
                "decisions": [
                    {
                        "candidate_id": decision.candidate_id,
                        "reason": decision.reason.value,
                        "mask_area_pixels": decision.mask_area_pixels,
                        "covered_masks": [
                            {
                                "candidate_id": evidence.candidate_id,
                                "intersection_pixels": evidence.intersection_pixels,
                                "candidate_mask_area_pixels": evidence.candidate_mask_area_pixels,
                                "containment": evidence.containment,
                            }
                            for evidence in decision.covered_masks
                        ],
                    }
                    for decision in resolution.decisions
                ],
            }
        )
    all_ids = {row.candidate_id for row in records}
    if len(kept_ids) != len(set(kept_ids)) or not set(kept_ids) <= all_ids:
        raise RuntimePipelineError("aggregate resolution produced an invalid candidate set")
    payload = {
        "enabled": True,
        "policy_id": "aggregate_mask_containment_v1",
        "mask_containment_at_least": DEFAULT_MASK_CONTAINMENT,
        "minimum_covered_masks": DEFAULT_MINIMUM_COVERED_MASKS,
        "input_candidate_count": len(records),
        "resolved_candidate_count": len(kept_ids),
        "removed_candidate_count": len(all_ids) - len(kept_ids),
        "frames": frame_rows,
    }
    return tuple(kept_ids), payload


def run_mask_refinement_streams(
    detector_outcomes: Sequence[DetectorStreamOutcome],
    *,
    profile: RuntimeProfile,
    assets: RuntimeAssets,
    workspace: RuntimeWorkspace,
) -> tuple[tuple[MaskStreamOutcome, ...], float]:
    cleanup = _cleanup_config(profile)
    started = time.perf_counter()
    refiner = None
    outputs: list[MaskStreamOutcome] = []
    try:
        refiner = load_local_sam2_bbox_refiner(assets.sam2_directory, device="cuda")
        for detector in detector_outcomes:
            decoded = detector.decoded
            by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for candidate in detector.accepted_candidates:
                by_frame[str(candidate["frame_id"])].append(candidate)
            records: list[RuntimeMaskRecord] = []
            for frame in decoded.frames:
                candidates = sorted(
                    by_frame.get(frame.frame_id, ()),
                    key=lambda row: str(row["candidate_id"]),
                )
                if not candidates:
                    continue
                boxes = tuple(
                    bbox_from_payload(_mapping(row, "bbox", "candidate"))
                    for row in candidates
                )
                results = refiner.refine(frame_array(frame), boxes, cleanup=cleanup)
                if len(results) != len(candidates):
                    raise RuntimePipelineError("SAM2 result count differs from candidate count")
                for source_index, (candidate, result) in enumerate(
                    zip(candidates, results, strict=True)
                ):
                    if (
                        result.status != "valid"
                        or result.raw_mask is None
                        or result.cleaned_mask is None
                    ):
                        raise RuntimePipelineError(
                            f"SAM2 did not produce a valid mask for {candidate['candidate_id']}: "
                            f"{result.fallback_reason}"
                        )
                    artifacts = workspace.store_masks(
                        stream_id=decoded.stream.stream_id,
                        frame_id=frame.frame_id,
                        candidate_id=str(candidate["candidate_id"]),
                        source_index=source_index,
                        raw_mask=result.raw_mask,
                        cleaned_mask=result.cleaned_mask,
                    )
                    records.append(
                        RuntimeMaskRecord(
                            candidate_id=str(candidate["candidate_id"]),
                            frame_id=frame.frame_id,
                            frame_index=int(candidate["frame_index"]),
                            score=float(candidate["score"]),
                            phrase=str(candidate["phrase"]),
                            source_bbox=boxes[source_index],
                            result=result,
                            artifacts=artifacts,
                        )
                    )
            records.sort(key=lambda row: (row.frame_index, row.candidate_id))
            if len(records) != len(detector.accepted_candidates):
                raise RuntimePipelineError("SAM2 coverage differs from detector coverage")
            resolved, resolution = resolve_aggregate_mask_records(records)
            outputs.append(
                MaskStreamOutcome(
                    decoded=decoded,
                    records=tuple(records),
                    resolved_candidate_ids=resolved,
                    aggregate_resolution=resolution,
                )
            )
    finally:
        if refiner is not None:
            del refiner
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return tuple(outputs), time.perf_counter() - started


def build_candidate_snapshot(
    detector: DetectorStreamOutcome,
    masks: MaskStreamOutcome,
    *,
    config_sha256: str,
    record_namespace: str = "visual-stream-runtime",
    candidate_source: str = "grounding_dino_sam2_visual_stream_v1",
) -> CandidateExtractionSnapshot:
    decoded = detector.decoded
    accepted_by_id = {
        str(row["candidate_id"]): row for row in detector.accepted_candidates
    }
    mask_by_id = {row.candidate_id: row for row in masks.records}
    if set(accepted_by_id) != set(mask_by_id):
        raise RuntimePipelineError("detector and SAM2 candidate membership differs")
    producer = ProducerProvenance(
        producer_stage=f"{record_namespace}-candidate-extraction",
        producer_version="1.0.0",
        config_version="1.0.0",
        config_digest=f"sha256:{config_sha256}",
    )
    frame_lookup = {frame.frame_id: frame for frame in decoded.frames}
    candidate_records: list[CandidateRecord] = []
    mask_store: dict[str, CandidateMaskRecord] = {}
    per_frame_counts: dict[str, int] = defaultdict(int)
    for candidate_id in masks.resolved_candidate_ids:
        row = accepted_by_id[candidate_id]
        observation = mask_by_id[candidate_id]
        frame_id = str(row["frame_id"])
        frame = frame_lookup.get(frame_id)
        if frame is None:
            raise RuntimePipelineError(f"candidate references unknown frame: {candidate_id}")
        bbox = bbox_from_payload(_mapping(row, "bbox", "candidate"))
        if bbox != observation.source_bbox:
            raise RuntimePipelineError(f"candidate and SAM2 bbox differ: {candidate_id}")
        full_mask = np.asarray(observation.result.cleaned_mask, dtype=np.bool_)
        x, y, width, height = (int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height))
        local_mask = np.ascontiguousarray(full_mask[y : y + height, x : x + width], dtype=np.bool_)
        if local_mask.shape != (height, width) or not local_mask.any():
            raise RuntimePipelineError(f"invalid local mask crop: {candidate_id}")
        mask_ref = (
            f"{record_namespace}-mask:"
            f"{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:24]}"
        )
        mask_digest = candidate_mask_digest(local_mask)
        mask_store[mask_ref] = CandidateMaskRecord(
            mask_ref=mask_ref,
            mask_digest=mask_digest,
            candidate_id=candidate_id,
            frame_id=frame_id,
            coordinate_bbox=bbox,
            mask=local_mask,
        )
        geometry = GeometryFeatureMetadata(
            feature_schema_id=f"{record_namespace}-geometry-v1",
            producer_version="1.0.0",
            config_digest=f"sha256:{config_sha256}",
            values={
                "area": bbox.area,
                "area_ratio": bbox.area / (frame.image_size.width * frame.image_size.height),
                "width_ratio": bbox.width / frame.image_size.width,
                "height_ratio": bbox.height / frame.image_size.height,
            },
        )
        candidate_records.append(
            CandidateRecord(
                envelope=RecordEnvelope(
                    record_id=candidate_id,
                    schema_version=f"{record_namespace}-candidate-record-1.0",
                    stream_id=decoded.stream.stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame_id, candidate_id=candidate_id),
                ),
                candidate_id=candidate_id,
                frame_id=frame_id,
                frame_index=frame.record.index,
                frame_size=frame.image_size,
                bbox=bbox,
                center=bbox.center,
                geometry=geometry,
                candidate_source=candidate_source,
                mask=MaskReference(
                    mask_ref=mask_ref,
                    mask_digest=mask_digest,
                    producer_version="1.0.0",
                    coordinate_bbox=bbox,
                    validity_status=ValidityStatus.VALID,
                ),
                candidate_confidence=float(row["score"]),
            )
        )
        per_frame_counts[frame_id] += 1
    candidate_records.sort(key=lambda item: (item.frame_index, item.candidate_id))
    diagnostics = tuple(
        FrameCandidateDiagnostics(
            envelope=RecordEnvelope(
                record_id=(
                    f"{record_namespace}-candidate-diagnostics:"
                    f"{decoded.stream.stream_id}:{frame.frame_id}"
                ),
                schema_version=f"{record_namespace}-frame-candidate-diagnostics-1.0",
                stream_id=decoded.stream.stream_id,
                producer=producer,
                context=StageContext(frame_id=frame.frame_id),
            ),
            frame_id=frame.frame_id,
            frame_index=frame.record.index,
            image_size=frame.image_size,
            summary={"candidate_count": per_frame_counts.get(frame.frame_id, 0)},
        )
        for frame in decoded.frames
    )
    extraction = CandidateExtractionResult(
        envelope=RecordEnvelope(
            record_id=f"{record_namespace}-candidate-extraction:{decoded.stream.stream_id}",
            schema_version=f"{record_namespace}-candidate-extraction-result-1.0",
            stream_id=decoded.stream.stream_id,
            producer=producer,
            context=StageContext(),
        ),
        extractor_source=candidate_source,
        candidates=tuple(candidate_records),
        frame_diagnostics=diagnostics,
    )
    return CandidateExtractionSnapshot(result=extraction, masks=mask_store)


def matching_configs(
    pipeline: Mapping[str, Any],
) -> tuple[PairScoringConfig, MatchingConfig, GroupingConfig, EventConfig]:
    score = _mapping(pipeline, "scoring", "pipeline")
    match = _mapping(pipeline, "matching", "pipeline")
    group = _mapping(pipeline, "grouping", "pipeline")
    event = _mapping(pipeline, "events", "pipeline")
    scoring = PairScoringConfig(
        scorer_id=str(score["scorer_id"]),
        scorer_version=str(score["scorer_version"]),
        visual_gate=float(score["visual_gate"]),
        spatial_gate=score["spatial_gate"],
    )
    matching = MatchingConfig(
        unmatched_pair_cost=float(match["unmatched_pair_cost"]),
        local_margin_gate=float(match["local_margin_gate"]),
        global_margin_gate=float(match["global_margin_gate"]),
        severe_quality_flags=tuple(match["severe_quality_flags"]),
        assignment_policy_id=str(match["assignment_policy_id"]),
        assignment_policy_version=str(match["assignment_policy_version"]),
    )
    grouping = GroupingConfig(
        medoid_gate=float(group["medoid_gate"]),
        support_quantile=float(group["support_quantile"]),
        quantile_gate=float(group["quantile_gate"]),
        support_pair_gate=float(group["support_pair_gate"]),
        support_ratio_gate=float(group["support_ratio_gate"]),
        visual_gate=float(group["visual_gate"]),
        medoid_weight=float(group["medoid_weight"]),
        quantile_weight=float(group["quantile_weight"]),
        support_ratio_weight=float(group["support_ratio_weight"]),
        second_best_margin=float(group["second_best_margin"]),
        representation_variant_id=DINO_MASK_NEUTRAL_VARIANT,
        scorer_id=str(score["scorer_id"]),
        scorer_version=str(score["scorer_version"]),
        expose_unmerged_alternatives=bool(group["expose_unmerged_alternatives"]),
        use_uncertain_temporal_seeds=bool(group["use_uncertain_temporal_seeds"]),
        uncertain_seed_visual_gate=float(group["uncertain_seed_visual_gate"]),
        uncertain_seed_spatial_gate=float(group["uncertain_seed_spatial_gate"]),
        coframe_aspect_log_gate=float(group["coframe_aspect_log_gate"]),
        coframe_area_log_gate=float(group["coframe_area_log_gate"]),
        coframe_fill_ratio_gate=float(group["coframe_fill_ratio_gate"]),
        coframe_hole_count_gate=int(group["coframe_hole_count_gate"]),
        coframe_visual_gate=float(group["coframe_visual_gate"]),
        coframe_margin_bypass=bool(group["coframe_margin_bypass"]),
        severe_quality_flags=tuple(group["severe_quality_flags"]),
        grouping_policy_id=str(group["grouping_policy_id"]),
        grouping_policy_version=str(group["grouping_policy_version"]),
    )
    events = EventConfig(
        position_threshold_norm=float(event["position_threshold_norm"]),
        severe_quality_flags=tuple(event["severe_quality_flags"]),
        event_policy_id=str(event["event_policy_id"]),
        event_policy_version=str(event["event_policy_version"]),
    )
    return scoring, matching, grouping, events


def run_analysis_streams(
    mask_outcomes: Sequence[MaskStreamOutcome],
    detector_outcomes: Sequence[DetectorStreamOutcome],
    *,
    profile: RuntimeProfile,
    assets: RuntimeAssets,
    workspace: RuntimeWorkspace,
    record_namespace: str = "visual-stream-runtime",
    candidate_source: str = "grounding_dino_sam2_visual_stream_v1",
) -> tuple[tuple[AnalysisStreamOutcome, ...], float]:
    detector_by_stream = {
        item.decoded.stream.stream_id: item for item in detector_outcomes
    }
    models = profile.models
    model_spec = _mapping(models, "representation", "models")
    checkpoint_spec = _mapping(model_spec, "checkpoint", "representation")
    representation_payload = _mapping(profile.pipeline, "representation", "pipeline")
    provider = LocalDinoV2Provider(
        source_dir=assets.dinov2_source_directory,
        checkpoint_path=assets.dinov2_checkpoint,
        expected_checkpoint_sha256=str(checkpoint_spec["sha256"]),
        expected_checkpoint_size_bytes=int(checkpoint_spec["size_bytes"]),
        expected_source_tree_fingerprint=str(model_spec["source_tree_fingerprint_sha256"]),
        model_name=str(model_spec["model_name"]),
        embedding_dimension=int(model_spec["embedding_dimension"]),
        device_policy="cuda",
        batch_size=int(representation_payload["batch_size"]),
    )
    representation_config = DinoV2RepresentationConfig(
        expected_checkpoint_sha256=str(checkpoint_spec["sha256"]),
        expected_checkpoint_size_bytes=int(checkpoint_spec["size_bytes"]),
        expected_source_tree_fingerprint=str(model_spec["source_tree_fingerprint_sha256"]),
        model_name=str(model_spec["model_name"]),
        embedding_dimension=int(model_spec["embedding_dimension"]),
        variant=DINO_MASK_NEUTRAL_VARIANT,
        input_size=int(representation_payload["input_size"]),
        context_padding_ratio=float(representation_payload["context_padding_ratio"]),
        device_policy="cuda",
        batch_size=int(representation_payload["batch_size"]),
    )
    scoring, matching, grouping_config, events_config = matching_configs(profile.pipeline)
    scorer = DinoV2CosineScorer()
    started = time.perf_counter()
    outputs: list[AnalysisStreamOutcome] = []
    try:
        for masks in mask_outcomes:
            decoded = masks.decoded
            detector = detector_by_stream.get(decoded.stream.stream_id)
            if detector is None:
                raise RuntimePipelineError("mask stage references an unknown detector stream")
            snapshot = build_candidate_snapshot(
                detector,
                masks,
                config_sha256=profile.sha256,
                record_namespace=record_namespace,
                candidate_source=candidate_source,
            )
            batch = build_dinov2_representations(
                decoded, snapshot, provider, representation_config
            )
            if batch.errors or len(batch.records) != len(snapshot.result.candidates):
                raise RuntimePipelineError(
                    f"DINOv2 coverage failed for {decoded.stream.stream_id}: "
                    f"records={len(batch.records)}, candidates={len(snapshot.result.candidates)}, "
                    f"errors={len(batch.errors)}"
                )
            representation_by_id = {record.candidate_id: record for record in batch.records}
            candidate_ids = [candidate.candidate_id for candidate in snapshot.result.candidates]
            embeddings = (
                np.stack(
                    [
                        np.asarray(
                            representation_by_id[candidate_id].payload.embedding,
                            dtype=np.float32,
                        )
                        for candidate_id in candidate_ids
                    ],
                    axis=0,
                )
                if candidate_ids
                else np.empty((0, int(model_spec["embedding_dimension"])), dtype=np.float32)
            )
            stored_embeddings = workspace.store_embeddings(
                stream_id=decoded.stream.stream_id,
                embeddings=embeddings,
            )
            matching_results: list[Any] = []
            candidates = snapshot.result.candidates
            for left, right in zip(decoded.frames, decoded.frames[1:]):
                frame_pair = FramePair(
                    from_frame_id=left.frame_id,
                    from_frame_index=left.record.index,
                    from_frame_size=left.image_size,
                    to_frame_id=right.frame_id,
                    to_frame_index=right.record.index,
                    to_frame_size=right.image_size,
                )
                matched = match_neighboring_frames(
                    stream_id=decoded.stream.stream_id,
                    frame_pair=frame_pair,
                    from_candidates=tuple(
                        candidate for candidate in candidates if candidate.frame_id == left.frame_id
                    ),
                    to_candidates=tuple(
                        candidate
                        for candidate in candidates
                        if candidate.frame_id == right.frame_id
                    ),
                    representations=batch.records,
                    representation_variant_id=representation_config.variant,
                    scorer=scorer,
                    scoring_config=scoring,
                    matching_config=matching,
                )
                if matched.errors:
                    raise RuntimePipelineError(
                        f"neighbor matching failed for {decoded.stream.stream_id}"
                    )
                matching_results.append(matched.result)
            matching_tuple = tuple(matching_results)
            grouping = group_recurring_visual_types(
                stream_id=decoded.stream.stream_id,
                candidates=candidates,
                representations=batch.records,
                matching_results=matching_tuple,
                representation_variant_id=representation_config.variant,
                scorer=scorer,
                config=grouping_config,
            )
            event_batch = build_change_events(
                stream_id=decoded.stream.stream_id,
                candidates=candidates,
                grouping=grouping,
                matching_results=matching_tuple,
                config=events_config,
            )
            if event_batch.errors:
                raise RuntimePipelineError(
                    f"event generation failed for {decoded.stream.stream_id}"
                )
            outputs.append(
                AnalysisStreamOutcome(
                    decoded=decoded,
                    snapshot=snapshot,
                    embeddings=embeddings,
                    embedding_artifact=stored_embeddings,
                    representation_batch=batch,
                    matching_results=matching_tuple,
                    grouping=grouping,
                    event_batch=event_batch,
                )
            )
    finally:
        del provider
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return tuple(outputs), time.perf_counter() - started


def execute_pipeline(
    decoded_streams: Sequence[DecodedStream],
    *,
    profile: RuntimeProfile,
    assets: RuntimeAssets,
    workspace: RuntimeWorkspace,
    record_namespace: str = "visual-stream-runtime",
    candidate_source: str = "grounding_dino_sam2_visual_stream_v1",
) -> RuntimeOutcome:
    streams = tuple(decoded_streams)
    if not streams:
        raise RuntimePipelineError("the runtime requires at least one decoded stream")
    stream_ids = [stream.stream.stream_id for stream in streams]
    if len(stream_ids) != len(set(stream_ids)):
        raise RuntimePipelineError("runtime stream IDs must be unique")
    try:
        detector, detector_seconds = run_detector_streams(
            streams, profile=profile, assets=assets
        )
    except RuntimePipelineError:
        raise
    except Exception as error:
        raise RuntimePipelineError(
            f"Grounding DINO stage failed ({type(error).__name__})."
        ) from error
    try:
        masks, mask_seconds = run_mask_refinement_streams(
            detector,
            profile=profile,
            assets=assets,
            workspace=workspace,
        )
    except RuntimePipelineError:
        raise
    except Exception as error:
        raise RuntimePipelineError(f"SAM2 stage failed ({type(error).__name__}).") from error
    try:
        analyses, analysis_seconds = run_analysis_streams(
            masks,
            detector,
            profile=profile,
            assets=assets,
            workspace=workspace,
            record_namespace=record_namespace,
            candidate_source=candidate_source,
        )
    except RuntimePipelineError:
        raise
    except Exception as error:
        raise RuntimePipelineError(f"DINOv2 stage failed ({type(error).__name__}).") from error
    masks_by_stream = {item.decoded.stream.stream_id: item for item in masks}
    analysis_by_stream = {item.decoded.stream.stream_id: item for item in analyses}
    outcomes = tuple(
        RuntimeStreamOutcome(
            detector=item,
            masks=masks_by_stream[item.decoded.stream.stream_id],
            analysis=analysis_by_stream[item.decoded.stream.stream_id],
        )
        for item in detector
    )
    return RuntimeOutcome(
        profile_id=profile.profile_id,
        profile_sha256=profile.sha256,
        streams=outcomes,
        detector_elapsed_seconds=detector_seconds,
        mask_elapsed_seconds=mask_seconds,
        analysis_elapsed_seconds=analysis_seconds,
    )


__all__ = [
    "AnalysisStreamOutcome",
    "DetectorStreamOutcome",
    "MaskStreamOutcome",
    "RuntimeMaskRecord",
    "RuntimeOutcome",
    "RuntimePipelineError",
    "RuntimeStreamOutcome",
    "bbox_from_payload",
    "build_candidate_snapshot",
    "detector_profiles",
    "execute_pipeline",
    "matching_configs",
    "profile_detector_predictions",
    "resolve_aggregate_mask_records",
    "run_analysis_streams",
    "run_detector_streams",
    "run_mask_refinement_streams",
    "temporal_detector_records",
]
