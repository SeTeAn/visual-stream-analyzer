"""Evaluate DINOv2 with OCID oracle object localization.

This is an evaluation-only diagnostic. OCID instance masks and identities are
used to construct perfect object crops and labels; they are never passed to the
normal ``analyze`` orchestration path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image

from stream_analysis import (
    BBox,
    CandidateRecord,
    DinoEmbeddingPayload,
    FramePair,
    GeometryFeatureMetadata,
    ImageSize,
    InputQualityMetadata,
    ManifestLoadRequest,
    MaskReference,
    MatchDecisionStatus,
    ProducerProvenance,
    RecordEnvelope,
    RepresentationFamily,
    RepresentationRecord,
    RuntimeMetadata,
    StageContext,
    ValidityStatus,
    VersionedMetadata,
    load_decoded_stream,
)
from stream_analysis.candidates import CandidateMaskRecord, candidate_mask_digest
from stream_analysis.evaluation import PairRankingRecord, evaluate_pair_ranking
from stream_analysis.matching import (
    MatchingConfig,
    PairScoringConfig,
    match_neighboring_frames,
)
from stream_analysis.representations import (
    DINO_BBOX_VARIANT,
    DINO_MASK_NEUTRAL_VARIANT,
    DinoV2CosineScorer,
    LocalDinoV2Provider,
    preprocess_dinov2_candidate,
)


ORACLE_SCHEMA_VERSION = "ocid-oracle-dinov2-evaluation-0.1"
SUPPORTED_VARIANTS = (DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT)
SUPPORTED_ORACLE_ANNOTATION_SCOPES = {
    "candidate_extraction_bbox_only",
    "ocid_candidate_extraction_bbox_from_instance_masks",
}
_IDENTITY_PATTERN = re.compile(r"^ocid_physical_instance_(\d+)$")


@dataclass(frozen=True, slots=True)
class OracleInstance:
    stream_id: str
    frame_id: str
    frame_position: int
    instance_id: str
    identity_id: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class PreparedOracleInstance:
    oracle: OracleInstance
    frame: Any
    candidate: CandidateRecord
    mask_record: CandidateMaskRecord


@dataclass(frozen=True, slots=True)
class MatcherPolicy:
    policy_id: str
    visual_gate: float
    unmatched_pair_cost: float
    local_margin_gate: float
    global_margin_gate: float


MATCHER_POLICIES = (
    MatcherPolicy("current_v1", 0.8, 0.7, 0.01, 0.01),
    MatcherPolicy("zero_margin_diagnostic_v1", 0.8, 0.7, 0.0, 0.0),
)


def _object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _relative_path(value: object, name: str) -> PurePosixPath:
    text = _text(value, name)
    posix = PurePosixPath(text.replace("\\", "/"))
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ValueError(f"{name} must remain inside its declared root.")
    return posix


def _producer(stage: str, digest: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0",
        config_version="1.0",
        config_digest=digest,
    )


def _load_stream(stream_directory: Path):
    return load_decoded_stream(
        ManifestLoadRequest(
            stream_root=stream_directory.resolve(strict=True),
            producer=_producer("stream_input", "sha256:ocid-oracle-stream-input"),
        )
    )


def _annotation_rows(
    annotation: dict[str, Any],
    *,
    frame_positions: dict[str, int],
) -> tuple[OracleInstance, ...]:
    if annotation.get("annotation_scope") not in SUPPORTED_ORACLE_ANNOTATION_SCOPES:
        raise ValueError("Oracle evaluation requires an OCID candidate-only bbox annotation.")
    stream_id = _text(annotation.get("stream_id"), "annotation.stream_id")
    rows = annotation.get("expected_element_instances")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Annotation must contain expected_element_instances.")
    instances: list[OracleInstance] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Annotation instance rows must be objects.")
        instance_id = _text(row.get("instance_id"), "instance_id")
        if instance_id in seen:
            raise ValueError(f"Duplicate annotation instance_id: {instance_id}.")
        seen.add(instance_id)
        frame_id = _text(row.get("frame_id"), "frame_id")
        if frame_id not in frame_positions:
            raise ValueError(f"Annotation references unknown frame_id: {frame_id}.")
        identity_id = _text(row.get("visual_type_id"), "visual_type_id")
        if _IDENTITY_PATTERN.fullmatch(identity_id) is None:
            raise ValueError("OCID oracle identity must use the physical-instance proxy format.")
        bbox_value = row.get("bbox")
        if not isinstance(bbox_value, dict):
            raise ValueError("Every annotation instance must contain bbox.")
        bbox = BBox(
            _non_negative_int(bbox_value.get("x"), "bbox.x"),
            _non_negative_int(bbox_value.get("y"), "bbox.y"),
            _positive_int(bbox_value.get("width"), "bbox.width"),
            _positive_int(bbox_value.get("height"), "bbox.height"),
        )
        instances.append(
            OracleInstance(
                stream_id=stream_id,
                frame_id=frame_id,
                frame_position=frame_positions[frame_id],
                instance_id=instance_id,
                identity_id=identity_id,
                bbox=bbox,
            )
        )
    return tuple(sorted(instances, key=lambda item: (item.frame_position, item.instance_id)))


def _label_number(identity_id: str) -> int:
    match = _IDENTITY_PATTERN.fullmatch(identity_id)
    if match is None:
        raise ValueError(f"Unsupported OCID identity: {identity_id}.")
    return int(match.group(1))


def _load_label_mask(path: Path, image_size: ImageSize) -> np.ndarray:
    try:
        with Image.open(path) as image:
            labels = np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f"Cannot decode OCID label mask {path}: {error}") from error
    if labels.shape != (image_size.height, image_size.width):
        raise ValueError(f"OCID label mask geometry does not match RGB frame: {path}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"OCID label mask must use an integer dtype: {path}.")
    return labels


def prepare_oracle_instances(
    *,
    ocid_root: Path,
    stream_directory: Path,
    annotation_path: Path,
) -> tuple[Any, tuple[PreparedOracleInstance, ...]]:
    decoded = _load_stream(stream_directory)
    annotation = _object(annotation_path.resolve(strict=True), "oracle annotation")
    if annotation.get("stream_id") != decoded.stream.stream_id:
        raise ValueError("Annotation and decoded stream_id must match.")
    frame_positions = {frame.frame_id: position for position, frame in enumerate(decoded.frames)}
    oracle_rows = _annotation_rows(annotation, frame_positions=frame_positions)
    metadata = annotation.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("ground_truth_role") != "evaluation_only":
        raise ValueError("OCID annotations must declare ground_truth_role=evaluation_only.")
    source_sequence = _relative_path(metadata.get("source_sequence"), "source_sequence")
    root = ocid_root.resolve(strict=True)
    label_root = root.joinpath(*source_sequence.parts, "label").resolve(strict=True)
    try:
        label_root.relative_to(root)
    except ValueError as error:
        raise ValueError("Resolved OCID label directory escapes ocid_root.") from error

    frames = {frame.frame_id: frame for frame in decoded.frames}
    manifest_frames = {frame.frame_id: frame for frame in decoded.manifest.frames}
    labels_by_frame: dict[str, np.ndarray] = {}
    producer = _producer("oracle_evaluation", "sha256:ocid-oracle-candidates")
    prepared: list[PreparedOracleInstance] = []
    for oracle in oracle_rows:
        frame = frames[oracle.frame_id]
        if not oracle.bbox.is_within(frame.image_size):
            raise ValueError(f"Oracle bbox is outside frame geometry: {oracle.instance_id}.")
        if oracle.frame_id not in labels_by_frame:
            source_filename = _text(
                manifest_frames[oracle.frame_id].metadata.get("ocid_source_filename"),
                "ocid_source_filename",
            )
            if PurePosixPath(source_filename).name != source_filename:
                raise ValueError("ocid_source_filename must be a filename.")
            labels_by_frame[oracle.frame_id] = _load_label_mask(
                label_root / source_filename,
                frame.image_size,
            )
        labels = labels_by_frame[oracle.frame_id]
        x = int(oracle.bbox.x)
        y = int(oracle.bbox.y)
        width = int(oracle.bbox.width)
        height = int(oracle.bbox.height)
        mask = np.ascontiguousarray(
            labels[y:y + height, x:x + width] == _label_number(oracle.identity_id)
        )
        if mask.shape != (height, width) or not np.any(mask):
            raise ValueError(f"Oracle mask is empty or misaligned: {oracle.instance_id}.")
        mask_digest = candidate_mask_digest(mask)
        mask_ref = f"mask:{oracle.instance_id}"
        candidate = CandidateRecord(
            envelope=RecordEnvelope(
                record_id=oracle.instance_id,
                schema_version="candidate-record-1.0",
                stream_id=oracle.stream_id,
                producer=producer,
                context=StageContext(
                    frame_id=oracle.frame_id,
                    candidate_id=oracle.instance_id,
                ),
            ),
            candidate_id=oracle.instance_id,
            frame_id=oracle.frame_id,
            frame_index=frame.record.index,
            frame_size=frame.image_size,
            bbox=oracle.bbox,
            center=oracle.bbox.center,
            geometry=GeometryFeatureMetadata(
                feature_schema_id="oracle_geometry_v1",
                producer_version="1.0",
                config_digest=producer.config_digest,
                values={
                    "area": int(np.count_nonzero(mask)),
                    "bbox_area": oracle.bbox.area,
                    "foreground_fill_ratio": float(np.count_nonzero(mask) / oracle.bbox.area),
                    "area_ratio_to_frame": float(np.count_nonzero(mask) / frame.image_size.area),
                },
            ),
            candidate_source="ocid_oracle_evaluation",
            mask=MaskReference(
                mask_ref=mask_ref,
                mask_digest=mask_digest,
                producer_version="1.0",
                coordinate_bbox=oracle.bbox,
                validity_status=ValidityStatus.VALID,
            ),
            candidate_confidence=1.0,
        )
        mask_record = CandidateMaskRecord(
            mask_ref=mask_ref,
            mask_digest=mask_digest,
            candidate_id=oracle.instance_id,
            frame_id=oracle.frame_id,
            coordinate_bbox=oracle.bbox,
            mask=mask,
        )
        prepared.append(
            PreparedOracleInstance(
                oracle=oracle,
                frame=frame,
                candidate=candidate,
                mask_record=mask_record,
            )
        )
    return decoded, tuple(prepared)


def embed_oracle_instances(
    prepared: tuple[PreparedOracleInstance, ...],
    *,
    variant: str,
    provider: Any,
) -> dict[str, np.ndarray]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported oracle preprocessing variant: {variant}.")
    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(prepared), provider.batch_size):
        chunk = prepared[start:start + provider.batch_size]
        batch = np.stack(
            [
                preprocess_dinov2_candidate(
                    item.frame,
                    item.candidate,
                    variant=variant,
                    mask_record=item.mask_record if variant == DINO_MASK_NEUTRAL_VARIANT else None,
                ).normalized_chw
                for item in chunk
            ]
        )
        output = provider.embed_batch(batch)
        if output.embeddings.shape[0] != len(chunk):
            raise ValueError("Embedding provider changed oracle batch cardinality.")
        for item, embedding in zip(chunk, output.embeddings, strict=True):
            embeddings[item.oracle.instance_id] = embedding.copy()
    return embeddings


def _unit_score(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left.astype(np.float64), right.astype(np.float64)))
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


def _pair_records(
    instances: tuple[OracleInstance, ...],
    embeddings: dict[str, np.ndarray],
    *,
    namespace: str,
) -> tuple[PairRankingRecord, ...]:
    by_frame: dict[int, list[OracleInstance]] = {}
    for item in instances:
        if item.instance_id not in embeddings:
            raise ValueError(f"Missing oracle embedding: {item.instance_id}.")
        by_frame.setdefault(item.frame_position, []).append(item)
    records: list[PairRankingRecord] = []
    for position in range(min(by_frame, default=0), max(by_frame, default=-1)):
        left = by_frame.get(position, [])
        right = by_frame.get(position + 1, [])
        for query in left:
            for gallery in right:
                records.append(
                    PairRankingRecord(
                        query_id=f"{namespace}:{query.instance_id}",
                        gallery_id=f"{namespace}:{gallery.instance_id}",
                        label=query.identity_id == gallery.identity_id,
                        visual_score=_unit_score(
                            embeddings[query.instance_id],
                            embeddings[gallery.instance_id],
                        ),
                    )
                )
    if not records:
        raise ValueError("Oracle evaluation produced no adjacent-frame pairs.")
    return tuple(records)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _threshold_metrics(records: tuple[PairRankingRecord, ...], threshold: float) -> dict[str, float | int | None]:
    tp = sum(bool(item.label and item.visual_score is not None and item.visual_score >= threshold) for item in records)
    fp = sum(bool(not item.label and item.visual_score is not None and item.visual_score >= threshold) for item in records)
    fn = sum(bool(item.label and item.visual_score is not None and item.visual_score < threshold) for item in records)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"threshold": threshold, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}


def _margins(records: tuple[PairRankingRecord, ...]) -> dict[str, Any]:
    def one_direction(rows: tuple[PairRankingRecord, ...]) -> list[float]:
        grouped: dict[str, list[PairRankingRecord]] = {}
        for row in rows:
            grouped.setdefault(row.query_id, []).append(row)
        margins: list[float] = []
        for values in grouped.values():
            positives = [float(item.visual_score) for item in values if item.label and item.visual_score is not None]
            negatives = [float(item.visual_score) for item in values if not item.label and item.visual_score is not None]
            if positives and negatives:
                margins.append(max(positives) - max(negatives))
        return margins

    reverse = tuple(
        PairRankingRecord(
            query_id=item.gallery_id,
            gallery_id=item.query_id,
            label=item.label,
            visual_score=item.visual_score,
        )
        for item in records
    )
    forward = one_direction(records)
    backward = one_direction(reverse)
    combined = forward + backward
    return {
        "definition": "best_positive_score_minus_hardest_negative_score",
        "forward": _distribution(forward),
        "reverse": _distribution(backward),
        "aggregate": _distribution(combined),
        "positive_margin_rate": None if not combined else sum(value > 0 for value in combined) / len(combined),
    }


def evaluate_oracle_embeddings(
    instances: tuple[OracleInstance, ...],
    embeddings: dict[str, np.ndarray],
    *,
    namespace: str,
) -> tuple[dict[str, Any], tuple[PairRankingRecord, ...]]:
    records = _pair_records(instances, embeddings, namespace=namespace)
    positive_scores = [float(item.visual_score) for item in records if item.label and item.visual_score is not None]
    negative_scores = [float(item.visual_score) for item in records if not item.label and item.visual_score is not None]
    return (
        {
            "ranking": evaluate_pair_ranking(records, k_values=(1, 3, 5)),
            "score_distributions": {
                "positive": _distribution(positive_scores),
                "negative": _distribution(negative_scores),
            },
            "hard_negative_margin": _margins(records),
            "threshold_0_8": _threshold_metrics(records, 0.8),
        },
        records,
    )


def anonymize_oracle_candidates(
    prepared: tuple[PreparedOracleInstance, ...],
) -> tuple[dict[str, CandidateRecord], dict[str, str]]:
    """Replace label-derived IDs with frame-local geometry-order IDs.

    The returned candidate records contain no physical-instance proxy. The
    second mapping remains evaluation-only and is never passed to matching.
    """

    by_frame: dict[tuple[int, str], list[PreparedOracleInstance]] = {}
    for item in prepared:
        by_frame.setdefault(
            (item.oracle.frame_position, item.oracle.frame_id), []
        ).append(item)
    candidates_by_original_id: dict[str, CandidateRecord] = {}
    identity_by_anonymous_id: dict[str, str] = {}
    producer = _producer("oracle_evaluation", "sha256:ocid-oracle-anonymous-candidates")
    for (_position, frame_id), rows in sorted(by_frame.items()):
        geometry_keys = [
            (row.oracle.bbox.x, row.oracle.bbox.y, row.oracle.bbox.width, row.oracle.bbox.height)
            for row in rows
        ]
        if len(set(geometry_keys)) != len(geometry_keys):
            raise ValueError("Oracle frame contains duplicate bboxes; anonymous ordering is ambiguous.")
        ordered = sorted(
            rows,
            key=lambda row: (
                row.oracle.bbox.y,
                row.oracle.bbox.x,
                row.oracle.bbox.height,
                row.oracle.bbox.width,
            ),
        )
        for index, item in enumerate(ordered):
            candidate_id = f"candidate:{frame_id}:{index:03d}"
            candidate = replace(
                item.candidate,
                envelope=RecordEnvelope(
                    record_id=candidate_id,
                    schema_version="oracle-anonymous-candidate-1.0",
                    stream_id=item.oracle.stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame_id, candidate_id=candidate_id),
                ),
                candidate_id=candidate_id,
                candidate_source="ocid_oracle_anonymous_evaluation",
                mask=None,
            )
            candidates_by_original_id[item.oracle.instance_id] = candidate
            identity_by_anonymous_id[candidate_id] = item.oracle.identity_id
    return candidates_by_original_id, identity_by_anonymous_id


def _matching_representation(
    candidate: CandidateRecord,
    embedding: np.ndarray,
    *,
    variant: str,
    model_name: str,
) -> RepresentationRecord:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Matcher embedding must be one-dimensional and finite.")
    semantic_digest = f"sha256:ocid-oracle-matcher-{model_name}-{variant}"
    producer = _producer("representation", semantic_digest)
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=f"representation:{candidate.candidate_id}",
            schema_version="oracle-dinov2-representation-1.0",
            stream_id=candidate.envelope.stream_id,
            producer=producer,
            context=StageContext(
                frame_id=candidate.frame_id,
                candidate_id=candidate.candidate_id,
            ),
            provenance_refs=(candidate.candidate_id,),
        ),
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        family=RepresentationFamily.DINO_V2,
        representation_type=f"{model_name}_cls",
        representation_version=f"{model_name.replace('_', '-')}-cls-1.0",
        input_variant=variant,
        semantic_config_digest=semantic_digest,
        payload=DinoEmbeddingPayload(
            embedding=tuple(float(value) for value in vector),
            embedding_dimension=len(vector),
            l2_normalized=True,
        ),
        preprocessing_metadata=VersionedMetadata(
            identifier=f"{variant}.preprocessing",
            version="1.0",
            details={"scope": "oracle_localization_evaluation_only"},
        ),
        provider_metadata=VersionedMetadata(
            identifier="local_dinov2_torch_provider",
            version="1.2",
        ),
        model_metadata=VersionedMetadata(
            identifier=model_name,
            version=f"{model_name.replace('_', '-')}-pretrained-1.0",
        ),
        runtime_metadata=RuntimeMetadata(runtime_id="oracle_matcher_evaluation"),
        input_quality_metadata=InputQualityMetadata(
            candidate_confidence=candidate.candidate_confidence,
            quality_flags=candidate.quality_flags,
        ),
    )


def _set_metrics(truth: set[tuple[str, ...]], predicted: set[tuple[str, ...]]) -> dict[str, Any]:
    true_positive = len(truth & predicted)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    precision = None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = None if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "truth_count": len(truth),
        "prediction_count": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _aggregate_set_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {
        "truth_count": sum(int(row["truth_count"]) for row in rows),
        "prediction_count": sum(int(row["prediction_count"]) for row in rows),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        ),
    }


def evaluate_oracle_matcher(
    decoded: Any,
    prepared: tuple[PreparedOracleInstance, ...],
    embeddings: dict[str, np.ndarray],
    *,
    variant: str,
    model_name: str,
    policy: MatcherPolicy,
) -> dict[str, Any]:
    """Run the production neighboring matcher, then evaluate its decisions."""

    candidate_by_original, identity_by_candidate = anonymize_oracle_candidates(prepared)
    representations = tuple(
        _matching_representation(
            candidate_by_original[item.oracle.instance_id],
            embeddings[item.oracle.instance_id],
            variant=variant,
            model_name=model_name,
        )
        for item in prepared
    )
    candidates_by_frame: dict[str, tuple[CandidateRecord, ...]] = {}
    for frame in decoded.frames:
        candidates_by_frame[frame.frame_id] = tuple(
            sorted(
                (
                    candidate_by_original[item.oracle.instance_id]
                    for item in prepared
                    if item.oracle.frame_id == frame.frame_id
                ),
                key=lambda candidate: candidate.candidate_id,
            )
        )
    scorer = DinoV2CosineScorer()
    scoring_config = PairScoringConfig(
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.scorer_version,
        visual_gate=policy.visual_gate,
        spatial_gate=None,
    )
    matching_config = MatchingConfig(
        unmatched_pair_cost=policy.unmatched_pair_cost,
        local_margin_gate=policy.local_margin_gate,
        global_margin_gate=policy.global_margin_gate,
        severe_quality_flags=(),
    )
    frame_rows: list[dict[str, Any]] = []
    for left_frame, right_frame in zip(decoded.frames, decoded.frames[1:]):
        left = candidates_by_frame[left_frame.frame_id]
        right = candidates_by_frame[right_frame.frame_id]
        batch = match_neighboring_frames(
            stream_id=decoded.stream.stream_id,
            frame_pair=FramePair(
                from_frame_id=left_frame.frame_id,
                from_frame_index=left_frame.record.index,
                from_frame_size=left_frame.image_size,
                to_frame_id=right_frame.frame_id,
                to_frame_index=right_frame.record.index,
                to_frame_size=right_frame.image_size,
            ),
            from_candidates=left,
            to_candidates=right,
            representations=representations,
            representation_variant_id=variant,
            scorer=scorer,
            scoring_config=scoring_config,
            matching_config=matching_config,
        )
        if batch.errors:
            raise ValueError("Production matcher produced errors in the oracle evaluation.")
        result = batch.result
        left_identity = {item.candidate_id: identity_by_candidate[item.candidate_id] for item in left}
        right_identity = {item.candidate_id: identity_by_candidate[item.candidate_id] for item in right}
        truth_matches = {
            (left_id, right_id)
            for left_id, left_value in left_identity.items()
            for right_id, right_value in right_identity.items()
            if left_value == right_value
        }
        selected_matches = {
            (item.from_endpoint.candidate_id, item.to_endpoint.candidate_id)
            for item in result.selected_matches
        }
        accepted_matches = {
            (item.from_endpoint.candidate_id, item.to_endpoint.candidate_id)
            for item in result.selected_matches
            if item.status is MatchDecisionStatus.ACCEPTED
        }
        uncertain_matches = selected_matches - accepted_matches
        right_values = set(right_identity.values())
        left_values = set(left_identity.values())
        truth_unmatched = {
            *(("from", candidate_id) for candidate_id, identity in left_identity.items() if identity not in right_values),
            *(("to", candidate_id) for candidate_id, identity in right_identity.items() if identity not in left_values),
        }
        predicted_unmatched = {
            (item.side.value, item.endpoint.candidate_id) for item in result.unmatched
        }
        frame_rows.append(
            {
                "from_frame_id": left_frame.frame_id,
                "to_frame_id": right_frame.frame_id,
                "selected": _set_metrics(truth_matches, selected_matches),
                "accepted": _set_metrics(truth_matches, accepted_matches),
                "uncertain": _set_metrics(truth_matches, uncertain_matches),
                "unmatched": _set_metrics(truth_unmatched, predicted_unmatched),
                "eligible_pair_count": result.matrix_summary["eligible_pair_count"],
                "real_pair_count": result.matrix_summary["real_pair_count"],
            }
        )
    return {
        "policy": {
            "policy_id": policy.policy_id,
            "visual_gate": policy.visual_gate,
            "spatial_gate": None,
            "unmatched_pair_cost": policy.unmatched_pair_cost,
            "local_margin_gate": policy.local_margin_gate,
            "global_margin_gate": policy.global_margin_gate,
        },
        "frame_pair_count": len(frame_rows),
        "selected": _aggregate_set_metric_rows([row["selected"] for row in frame_rows]),
        "accepted": _aggregate_set_metric_rows([row["accepted"] for row in frame_rows]),
        "uncertain": _aggregate_set_metric_rows([row["uncertain"] for row in frame_rows]),
        "unmatched": _aggregate_set_metric_rows([row["unmatched"] for row in frame_rows]),
        "eligible_pair_count": sum(int(row["eligible_pair_count"]) for row in frame_rows),
        "real_pair_count": sum(int(row["real_pair_count"]) for row in frame_rows),
        "per_frame_pair": frame_rows,
    }


def aggregate_oracle_matcher_reports(reports: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not reports:
        raise ValueError("reports must not be empty.")
    policy = reports[0]["policy"]
    if any(report["policy"] != policy for report in reports):
        raise ValueError("Matcher reports must use the same policy.")
    return {
        "policy": policy,
        "stream_count": len(reports),
        "frame_pair_count": sum(int(report["frame_pair_count"]) for report in reports),
        **{
            metric: _aggregate_set_metric_rows([report[metric] for report in reports])
            for metric in ("selected", "accepted", "uncertain", "unmatched")
        },
        "eligible_pair_count": sum(int(report["eligible_pair_count"]) for report in reports),
        "real_pair_count": sum(int(report["real_pair_count"]) for report in reports),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_oracle_evaluation(
    *,
    ocid_root: Path,
    stream_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    variants: tuple[str, ...],
    provider: Any,
    output_path: Path,
) -> dict[str, Any]:
    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise ValueError("Supply one annotation for every non-empty stream list.")
    if not variants or any(variant not in SUPPORTED_VARIANTS for variant in variants):
        raise ValueError("variants must contain supported DINOv2 preprocessing variants.")
    started = time.perf_counter()
    prepared_streams: list[tuple[Any, tuple[PreparedOracleInstance, ...]]] = []
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        decoded, prepared = prepare_oracle_instances(
            ocid_root=ocid_root,
            stream_directory=stream_directory,
            annotation_path=annotation_path,
        )
        prepared_streams.append((decoded, prepared))

    variant_results: dict[str, Any] = {}
    for variant in variants:
        stream_results: dict[str, Any] = {}
        aggregate_records: list[PairRankingRecord] = []
        matching_reports: dict[str, list[dict[str, Any]]] = {
            policy.policy_id: [] for policy in MATCHER_POLICIES
        }
        for decoded, prepared in prepared_streams:
            stream_id = decoded.stream.stream_id
            embeddings = embed_oracle_instances(prepared, variant=variant, provider=provider)
            evaluation, records = evaluate_oracle_embeddings(
                tuple(item.oracle for item in prepared),
                embeddings,
                namespace=stream_id,
            )
            stream_results[stream_id] = {
                "instance_count": len(prepared),
                **evaluation,
                "matching": {},
            }
            for policy in MATCHER_POLICIES:
                matcher_report = evaluate_oracle_matcher(
                    decoded,
                    prepared,
                    embeddings,
                    variant=variant,
                    model_name=provider.model_spec.model_name,
                    policy=policy,
                )
                stream_results[stream_id]["matching"][policy.policy_id] = matcher_report
                matching_reports[policy.policy_id].append(matcher_report)
            aggregate_records.extend(records)
        aggregate_tuple = tuple(aggregate_records)
        positive_scores = [float(item.visual_score) for item in aggregate_tuple if item.label and item.visual_score is not None]
        negative_scores = [float(item.visual_score) for item in aggregate_tuple if not item.label and item.visual_score is not None]
        variant_results[variant] = {
            "streams": stream_results,
            "aggregate": {
                "ranking": evaluate_pair_ranking(aggregate_tuple, k_values=(1, 3, 5)),
                "score_distributions": {
                    "positive": _distribution(positive_scores),
                    "negative": _distribution(negative_scores),
                },
                "hard_negative_margin": _margins(aggregate_tuple),
                "threshold_0_8": _threshold_metrics(aggregate_tuple, 0.8),
                "matching": {
                    policy.policy_id: aggregate_oracle_matcher_reports(
                        tuple(matching_reports[policy.policy_id])
                    )
                    for policy in MATCHER_POLICIES
                },
            },
        }

    payload: dict[str, Any] = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "status": "completed",
        "scope": "evaluation_only_oracle_localization",
        "ground_truth_boundary": "OCID masks and identities were used only by this diagnostic, never by analyze.",
        "identity_semantics": "physical_instance_proxy_not_visual_type_ground_truth",
        "model": dict(provider.model_metadata()),
        "provider": dict(provider.provider_metadata()),
        "variants": variant_results,
        "stream_count": len(prepared_streams),
        "instance_count": sum(len(prepared) for _decoded, prepared in prepared_streams),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--stream-directory", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-tree-fingerprint", required=True)
    parser.add_argument("--expected-checkpoint-size-bytes", type=int, required=True)
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--embedding-dimension", type=int, default=768)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provider = LocalDinoV2Provider(
        source_dir=args.source_dir,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_source_tree_fingerprint=args.expected_source_tree_fingerprint,
        expected_checkpoint_size_bytes=args.expected_checkpoint_size_bytes,
        model_name=args.model_name,
        embedding_dimension=args.embedding_dimension,
        device_policy=args.device,
        batch_size=args.batch_size,
    )
    payload = run_oracle_evaluation(
        ocid_root=args.ocid_root,
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        variants=tuple(args.variant or SUPPORTED_VARIANTS),
        provider=provider,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.resolve(strict=False)),
                "model": payload["model"]["model_name"],
                "stream_count": payload["stream_count"],
                "instance_count": payload["instance_count"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
