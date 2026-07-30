"""Evaluate configured DINOv2 representations on the OCID development split.

This runner deliberately persists RGB/predicted-mask embeddings before opening
reviewed OCID annotations.  It never accepts a held-out stream and never uses
a reviewed mask as a DINOv2 input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from stream_analysis import (
    BBox, CandidateRecord, GeometryFeatureMetadata, ImageSize, MaskReference,
    ProducerProvenance, RecordEnvelope, StageContext, ValidityStatus,
    load_decoded_stream,
)
from stream_analysis.candidates import CandidateMaskRecord, candidate_mask_digest
from stream_analysis.evaluation import (
    PairRankingRecord, PredictedCandidate, binary_mask_sha256,
    evaluate_pair_ranking, load_annotation,
)
from stream_analysis.evaluation.runner import _evaluate_candidates
from stream_analysis.input import ManifestLoadRequest
from stream_analysis.representations import (
    DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT, LocalDinoV2Provider,
    preprocess_dinov2_candidate,
)

from tools.evaluate_ocid_grounded_sam2_masks import (
    INFERENCE_SCHEMA_VERSION as SAM2_INFERENCE_SCHEMA,
    _bbox as _selected_bbox, _sha256, load_selected_candidates,
)
from tools.evaluate_ocid_oracle_dinov2 import (
    OracleInstance, PreparedOracleInstance, MatcherPolicy,
    aggregate_oracle_matcher_reports, evaluate_oracle_embeddings,
    evaluate_oracle_matcher,
)
from tools.ocid_evaluation_common import (
    DEFAULT_BENCHMARK_SPEC, DEFAULT_REVIEWED_ROOT, DevelopmentStream,
    OcidEvaluationError, load_development_inventory, write_canonical_json_atomic,
)


SCHEMA_VERSION = "ocid-masked-dinov2-evaluation-v1"
INFERENCE_SCHEMA_VERSION = "ocid-dinov2-inference-v1"
EMBEDDING_SCHEMA_VERSION = "ocid-dinov2-embeddings-v1"
SUPPORTED_VARIANTS = (DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT)
FROZEN_MODEL_NAME = "dinov2_vitb14"
FROZEN_EMBEDDING_DIMENSION = 768
FROZEN_BATCH_SIZE = 4
FROZEN_INPUT_SIZE = 224
FROZEN_CONTEXT_PADDING = 0.0
FROZEN_DEVELOPMENT_STREAM_COUNT = 10
FROZEN_CANDIDATE_COUNT = 1060
FROZEN_VALID_MASK_COUNT = 1060
VISUAL_SIMILARITY_REFERENCE = 0.80
MATCHER_POLICY = MatcherPolicy("current_v1", 0.80, 0.70, 0.01, 0.01)
MARKER_STREAM = "ocid_arid20_floor_bottom_seq12"
MARKER_FRAME = "frame_0019"
MARKER_LABELS = (3, 9)
OCCLUSION_SOURCE_LABEL = 10
OCCLUSION_FLAG_FRAME = "frame_0020"
OCCLUSION_VISIBILITY_THRESHOLD = 0.10
OCCLUSION_SOURCE_SEQUENCE = "ARID20/floor/bottom/seq12"


def _expected_sha256(path: Path, expected: str, label: str) -> str:
    normalized = expected.casefold() if isinstance(expected, str) else ""
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise OcidEvaluationError(f"{label} expected SHA-256 must contain 64 hexadecimal digits")
    actual = _sha256(path.resolve(strict=True))
    if actual != normalized:
        raise OcidEvaluationError(f"{label} SHA-256 mismatch: expected {normalized}, got {actual}")
    return actual


@dataclass(frozen=True, slots=True)
class CandidateInput:
    stream_id: str
    candidate: CandidateRecord
    mask_record: CandidateMaskRecord


def _producer(stage: str, digest: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage, producer_version="1.0.0", config_version="1.0.0",
        config_digest=digest,
    )


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        result = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcidEvaluationError(f"cannot load {label}: {error}") from error
    if not isinstance(result, dict):
        raise OcidEvaluationError(f"{label} must be a JSON object")
    return result


def _under(root: Path, relative: str, label: str) -> Path:
    value = PurePosixPath(relative.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise OcidEvaluationError(f"{label} must remain under the SAM2 artifact root")
    resolved = (root / Path(*value.parts)).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise OcidEvaluationError(f"{label} escapes the SAM2 artifact root") from error
    return resolved


def _same_bbox(left: BBox, right: BBox) -> bool:
    return (left.x, left.y, left.width, left.height) == (right.x, right.y, right.width, right.height)


def _load_binary_mask(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            values = np.asarray(source).copy()
    except OSError as error:
        raise OcidEvaluationError(f"cannot decode predicted SAM2 mask {path}: {error}") from error
    if values.ndim != 2:
        raise OcidEvaluationError("predicted SAM2 mask must be a two-dimensional image")
    if not np.isin(values, (0, 255)).all():
        raise OcidEvaluationError("predicted SAM2 mask must be binary 0/255")
    return np.asarray(values > 0, dtype=np.bool_)


def _candidate_from_selected(
    row: Any, *, stream_id: str, size: ImageSize, frame_record_index: int,
    mask: CandidateMaskRecord,
) -> CandidateRecord:
    bbox = _selected_bbox(row.bbox if isinstance(row, Mapping) else {
        "x": row.bbox.x, "y": row.bbox.y, "width": row.bbox.width, "height": row.bbox.height
    })
    bbox = row.bbox if hasattr(row, "bbox") else bbox
    candidate_id = row.candidate_id
    return CandidateRecord(
        envelope=RecordEnvelope(
            record_id=candidate_id, schema_version="ocid-dinov2-candidate-1.0",
            stream_id=stream_id, producer=_producer("ocid_representation_evaluation_input", "sha256:ocid-dinov2-input-v1"),
            context=StageContext(frame_id=row.frame_id, candidate_id=candidate_id),
        ),
        candidate_id=candidate_id, frame_id=row.frame_id, frame_index=frame_record_index,
        frame_size=size, bbox=bbox, center=bbox.center,
        geometry=GeometryFeatureMetadata(
            feature_schema_id="grounding_dino_selected_bbox_v1", producer_version="1.0.0",
            config_digest="sha256:grounding-dino-selected-candidates", values={"area": bbox.area},
        ),
        candidate_source="grounding_dino_ocid_evaluation_1_selected",
        mask=MaskReference(mask_ref=mask.mask_ref, mask_digest=mask.mask_digest,
                           producer_version="1.0.0", coordinate_bbox=bbox,
                           validity_status=ValidityStatus.VALID),
        candidate_confidence=float(row.score),
    )


def load_predicted_candidate_inputs(
    *, inventory: Sequence[DevelopmentStream], selected_candidates_path: Path,
    sam2_inference_manifest_path: Path, expected_selected_candidates_sha256: str,
    expected_sam2_inference_sha256: str,
) -> tuple[dict[str, tuple[CandidateInput, ...]], dict[str, Any], dict[str, Any]]:
    """Validate every configured input before decoding RGB or loading DINOv2."""
    if len(inventory) != FROZEN_DEVELOPMENT_STREAM_COUNT:
        raise OcidEvaluationError(
            f"OCID representation evaluation requires exactly {FROZEN_DEVELOPMENT_STREAM_COUNT} development streams"
        )
    selected_digest = _expected_sha256(
        selected_candidates_path, expected_selected_candidates_sha256, "selected candidates"
    )
    sam2_digest = _expected_sha256(
        sam2_inference_manifest_path, expected_sam2_inference_sha256, "SAM2 inference manifest"
    )
    selected, selected_payload = load_selected_candidates(selected_candidates_path, inventory)
    manifest = _object(sam2_inference_manifest_path, "SAM2 inference manifest")
    if (
        manifest.get("schema_version") != SAM2_INFERENCE_SCHEMA
        or manifest.get("status") != "rgb_inference_completed_before_ground_truth"
        or manifest.get("scope") != "development_only"
        or manifest.get("heldout_access") != "none"
    ):
        raise OcidEvaluationError("SAM2 manifest must be a persisted configured inference")
    provenance = manifest.get("selected_candidates")
    if not isinstance(provenance, Mapping) or provenance.get("sha256") != selected_digest:
        raise OcidEvaluationError("SAM2 manifest does not match the frozen selected candidate set")
    by_stream = manifest.get("records_by_stream")
    if not isinstance(by_stream, Mapping) or set(by_stream) != {item.stream_id for item in inventory}:
        raise OcidEvaluationError("SAM2 manifest must contain the exact development stream inventory")
    artifact_root = sam2_inference_manifest_path.resolve(strict=True).parent
    result: dict[str, tuple[CandidateInput, ...]] = {}
    for stream in inventory:
        decoded = _load_stream(stream.stream_directory)
        frames = {frame.frame_id: frame for frame in decoded.frames}
        positions = {frame.frame_id: position for position, frame in enumerate(decoded.frames)}
        expected = {item.candidate_id: item for item in selected[stream.stream_id]}
        rows = by_stream[stream.stream_id]
        if not isinstance(rows, list) or {row.get("candidate_id") for row in rows if isinstance(row, Mapping)} != set(expected):
            raise OcidEvaluationError(f"SAM2 candidate set mismatch for {stream.stream_id}")
        parsed: list[CandidateInput] = []
        for raw in rows:
            if not isinstance(raw, Mapping) or raw.get("status") != "valid":
                raise OcidEvaluationError("all selected candidates require valid predicted SAM2 masks")
            candidate_id = str(raw.get("candidate_id"))
            selected_row = expected[candidate_id]
            if raw.get("stream_id") != stream.stream_id or raw.get("frame_id") != selected_row.frame_id:
                raise OcidEvaluationError("SAM2 candidate stream/frame identity mismatch")
            source_bbox = raw.get("source_bbox")
            if not isinstance(source_bbox, Mapping) or not _same_bbox(_selected_bbox(source_bbox), selected_row.bbox):
                raise OcidEvaluationError("SAM2 candidate bbox does not match selected Grounding DINO bbox")
            frame = frames.get(selected_row.frame_id)
            if frame is None or positions[selected_row.frame_id] != selected_row.frame_index:
                raise OcidEvaluationError("selected candidate references an invalid development frame")
            mask_info = raw.get("cleaned_mask")
            if not isinstance(mask_info, Mapping) or not isinstance(mask_info.get("path"), str):
                raise OcidEvaluationError("SAM2 candidate is missing a cleaned predicted mask")
            full = _load_binary_mask(_under(artifact_root, mask_info["path"], "cleaned_mask.path"))
            if full.shape != (frame.image_size.height, frame.image_size.width):
                raise OcidEvaluationError("predicted SAM2 mask geometry does not match RGB frame")
            recorded_full_digest = mask_info.get("binary_mask_sha256")
            if not isinstance(recorded_full_digest, str):
                raise OcidEvaluationError("cleaned_mask.binary_mask_sha256 is required")
            if binary_mask_sha256(full) != recorded_full_digest:
                raise OcidEvaluationError("cleaned predicted SAM2 mask binary hash mismatch")
            # The input artifact hash is image-level and is preserved as provenance. The
            # canonical local crop has its own contract digest for preprocessing.
            x, y, w, h = map(int, (selected_row.bbox.x, selected_row.bbox.y, selected_row.bbox.width, selected_row.bbox.height))
            local = np.ascontiguousarray(full[y:y + h, x:x + w], dtype=np.bool_)
            if local.shape != (h, w) or not local.any():
                raise OcidEvaluationError("predicted SAM2 mask crop must be non-empty and match the exact bbox")
            local_digest = candidate_mask_digest(local)
            record = CandidateMaskRecord(
                mask_ref=f"sha256:{recorded_full_digest}", mask_digest=local_digest,
                candidate_id=candidate_id, frame_id=selected_row.frame_id,
                coordinate_bbox=selected_row.bbox, mask=local,
            )
            production = _candidate_from_selected(
                selected_row, stream_id=stream.stream_id, size=frame.image_size,
                frame_record_index=frame.record.index, mask=record,
            )
            parsed.append(CandidateInput(stream.stream_id, production, record))
        result[stream.stream_id] = tuple(sorted(parsed, key=lambda item: item.candidate.candidate_id))
    candidate_count = sum(len(rows) for rows in result.values())
    valid_mask_count = sum(1 for rows in result.values() for _item in rows)
    if candidate_count != FROZEN_CANDIDATE_COUNT or valid_mask_count != FROZEN_VALID_MASK_COUNT:
        raise OcidEvaluationError(
            "OCID representation evaluation requires exactly 1060 selected candidates and 1060 valid predicted masks"
        )
    if sam2_digest != expected_sam2_inference_sha256.casefold():  # defensive audit invariant
        raise OcidEvaluationError("SAM2 inference manifest digest changed during validation")
    return result, selected_payload, manifest


def _load_stream(stream_directory: Path):
    return load_decoded_stream(ManifestLoadRequest(
        stream_root=stream_directory.resolve(strict=True),
        producer=_producer("stream_input", "sha256:ocid-dinov2-rgb-input-v1"),
    ))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary_name, **arrays)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_provider_frozen(provider: Any) -> None:
    if getattr(provider, "batch_size", None) != FROZEN_BATCH_SIZE:
        raise OcidEvaluationError(f"OCID representation evaluation provider batch_size must be {FROZEN_BATCH_SIZE}")
    spec = getattr(provider, "model_spec", None)
    if spec is None:
        raise OcidEvaluationError("OCID representation evaluation provider must expose its frozen model_spec")
    if (
        getattr(spec, "model_name", None) != FROZEN_MODEL_NAME
        or getattr(spec, "embedding_dimension", None) != FROZEN_EMBEDDING_DIMENSION
        or getattr(spec, "expected_checkpoint_size_bytes", None) is None
    ):
        raise OcidEvaluationError("OCID representation evaluation provider does not match frozen ViT-B/14 model configuration")


def _validate_embedding_matrix(matrix: np.ndarray, count: int, label: str) -> np.ndarray:
    value = np.asarray(matrix)
    if value.dtype != np.dtype(np.float32):
        raise OcidEvaluationError(f"{label} embeddings must use float32")
    if value.shape != (count, FROZEN_EMBEDDING_DIMENSION):
        raise OcidEvaluationError(
            f"{label} embeddings must have shape ({count}, {FROZEN_EMBEDDING_DIMENSION})"
        )
    if not np.isfinite(value).all():
        raise OcidEvaluationError(f"{label} embeddings contain non-finite values")
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise OcidEvaluationError(f"{label} embeddings must be approximately unit-normalized")
    return np.ascontiguousarray(value)


def _candidate_index_row(position: int, item: CandidateInput) -> dict[str, Any]:
    return {
        "candidate_index": position,
        "stream_id": item.stream_id,
        "candidate_id": item.candidate.candidate_id,
        "frame_id": item.candidate.frame_id,
        "frame_index": item.candidate.frame_index,
        "bbox": {
            "x": item.candidate.bbox.x, "y": item.candidate.bbox.y,
            "width": item.candidate.bbox.width, "height": item.candidate.bbox.height,
        },
        "mask_digest": item.mask_record.mask_digest,
        "mask_ref": item.mask_record.mask_ref,
    }


def run_predicted_mask_inference(
    *, inventory: Sequence[DevelopmentStream], inputs: Mapping[str, Sequence[CandidateInput]],
    provider: Any, artifact_root: Path, selected_candidates_path: Path,
    sam2_inference_manifest_path: Path,
) -> dict[str, Any]:
    """Persist both variants before any reviewed annotation is opened."""
    started = time.perf_counter()
    _validate_provider_frozen(provider)
    ordered = [item for stream in inventory for item in inputs[stream.stream_id]]
    if len(inventory) != FROZEN_DEVELOPMENT_STREAM_COUNT or len(ordered) != FROZEN_CANDIDATE_COUNT:
        raise OcidEvaluationError("OCID representation evaluation inference requires the exact frozen development inventory")
    metadata = [_candidate_index_row(index, item) for index, item in enumerate(ordered)]
    decoded_by_stream = {stream.stream_id: _load_stream(stream.stream_directory) for stream in inventory}
    frame_lookup = {
        (stream_id, frame.frame_id): frame
        for stream_id, decoded in decoded_by_stream.items()
        for frame in decoded.frames
    }
    output_arrays: dict[str, np.ndarray] = {}
    variant_runtime: dict[str, Any] = {}
    for variant in SUPPORTED_VARIANTS:
        variant_started = time.perf_counter()
        embedding_chunks: list[np.ndarray] = []
        batch_count = 0
        for start in range(0, len(ordered), FROZEN_BATCH_SIZE):
            chunk = ordered[start:start + FROZEN_BATCH_SIZE]
            normalized = np.stack([
                preprocess_dinov2_candidate(
                    frame_lookup[(item.stream_id, item.candidate.frame_id)], item.candidate,
                    variant=variant, input_size=FROZEN_INPUT_SIZE,
                    context_padding_ratio=FROZEN_CONTEXT_PADDING,
                    mask_record=item.mask_record if variant == DINO_MASK_NEUTRAL_VARIANT else None,
                ).normalized_chw
                for item in chunk
            ]).astype(np.float32, copy=False)
            if normalized.shape[0] > FROZEN_BATCH_SIZE:
                raise OcidEvaluationError("OCID representation evaluation preprocessing exceeded the frozen provider batch")
            response = provider.embed_batch(normalized)
            batch_embeddings = _validate_embedding_matrix(
                np.asarray(response.embeddings), len(chunk), f"{variant} batch {batch_count}"
            )
            embedding_chunks.append(batch_embeddings.copy())
            batch_count += 1
            del normalized
        matrix = np.concatenate(embedding_chunks, axis=0)
        output_arrays[variant] = _validate_embedding_matrix(matrix, len(ordered), variant)
        variant_runtime[variant] = {
            "batch_count": batch_count,
            "max_batch_size": FROZEN_BATCH_SIZE,
            "candidate_count": len(ordered),
            "elapsed_seconds": time.perf_counter() - variant_started,
        }
    root = artifact_root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "embeddings.npz"
    _atomic_npz(npz_path, **output_arrays)
    manifest = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "status": "rgb_predicted_mask_inference_completed_before_ground_truth",
        "scope": "development_only_ocid_representation_evaluation", "heldout_access": "none",
        "ground_truth_boundary": "embeddings_npz_and_manifest_persisted_before_reviewed_annotation_open",
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "variants": list(SUPPORTED_VARIANTS), "input_size": FROZEN_INPUT_SIZE,
        "context_padding_ratio": FROZEN_CONTEXT_PADDING,
        "frozen_model": {"model_name": FROZEN_MODEL_NAME,
                         "embedding_dimension": FROZEN_EMBEDDING_DIMENSION,
                         "batch_size": FROZEN_BATCH_SIZE, "output_token": "cls"},
        "selected_candidates": {"path": selected_candidates_path.resolve().as_posix(),
                                "sha256": _sha256(selected_candidates_path.resolve(strict=True))},
        "sam2_inference_manifest": {"path": sam2_inference_manifest_path.resolve().as_posix(),
                                    "sha256": _sha256(sam2_inference_manifest_path.resolve(strict=True))},
        "embedding_artifact": {"path": npz_path.as_posix(), "sha256": _sha256(npz_path)},
        "candidate_index": metadata, "candidate_count": len(metadata),
        "provider": dict(provider.provider_metadata()) if hasattr(provider, "provider_metadata") else {},
        "model": dict(provider.model_metadata()) if hasattr(provider, "model_metadata") else {},
        "runtime": {"total_seconds": time.perf_counter() - started,
                    "by_variant": variant_runtime},
    }
    write_canonical_json_atomic(root / "inference_manifest.json", manifest)
    return manifest


def _assignment_identity(
    annotation: Any,
    candidates: Sequence[CandidateInput],
    *,
    iou_threshold: float = 0.50,
) -> tuple[dict[str, str], dict[str, float], dict[str, Any]]:
    predictions = tuple(PredictedCandidate(item.candidate.candidate_id, item.candidate.frame_id,
                                            item.candidate.bbox, "valid", (), (), item.candidate.frame_index)
                        for item in candidates)
    result = _evaluate_candidates(annotation, predictions, iou_threshold=iou_threshold)
    assignments = result.get("assignments", [])
    identities = {
        (row["candidate_id"] if isinstance(row, Mapping) else row.candidate_id):
        (row["visual_type_id"] if isinstance(row, Mapping) else row.visual_type_id)
        for row in assignments
    }
    ious = {
        (row["candidate_id"] if isinstance(row, Mapping) else row.candidate_id):
        float(row["iou"] if isinstance(row, Mapping) else row.iou)
        for row in assignments
    }
    for item in candidates:
        identities.setdefault(item.candidate.candidate_id, f"fp:{item.stream_id}:{item.candidate.candidate_id}")
    return identities, ious, result


def _mask_iou_slices(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    # Ground-truth mask IoU is evaluation-only; the representation evaluation preserves the
    # slice boundary in its report and uses None where no matched mask exists.
    counts = {"<0.70": 0, "0.70-0.80": 0, "0.80-0.90": 0, ">=0.90": 0, "unavailable": 0}
    for row in rows:
        value = row.get("mask_iou")
        if value is None: counts["unavailable"] += 1
        elif value < .70: counts["<0.70"] += 1
        elif value < .80: counts["0.70-0.80"] += 1
        elif value < .90: counts["0.80-0.90"] += 1
        else: counts[">=0.90"] += 1
    return counts


MASK_IOU_SLICE_IDS = ("<0.70", "0.70-0.80", "0.80-0.90", ">=0.90")


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p10": None, "median": None,
                "mean": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(np.min(array)),
        "p10": float(np.quantile(array, .10)), "median": float(np.median(array)),
        "mean": float(np.mean(array)), "p90": float(np.quantile(array, .90)),
        "max": float(np.max(array)),
    }


def _pair_margins(records: Sequence[PairRankingRecord]) -> dict[str, Any]:
    def direction(values: Sequence[PairRankingRecord]) -> list[float]:
        grouped: dict[str, list[PairRankingRecord]] = {}
        for row in values:
            grouped.setdefault(row.query_id, []).append(row)
        result: list[float] = []
        for rows in grouped.values():
            positives = [float(row.visual_score) for row in rows if row.label and row.visual_score is not None]
            negatives = [float(row.visual_score) for row in rows if not row.label and row.visual_score is not None]
            if positives and negatives:
                result.append(max(positives) - max(negatives))
        return result
    forward = direction(records)
    reverse = direction(tuple(PairRankingRecord(
        query_id=row.gallery_id, gallery_id=row.query_id, label=row.label,
        visual_score=row.visual_score, valid=row.valid,
        query_valid=row.gallery_valid, gallery_valid=row.query_valid,
    ) for row in records))
    combined = forward + reverse
    return {
        "definition": "best_positive_score_minus_hardest_negative_score",
        "forward": _distribution(forward), "reverse": _distribution(reverse),
        "aggregate": _distribution(combined),
        "positive_margin_rate": None if not combined else sum(value > 0 for value in combined) / len(combined),
    }


def _pair_report(records: Sequence[PairRankingRecord]) -> dict[str, Any]:
    rows = tuple(records)
    positive = [float(row.visual_score) for row in rows if row.label and row.visual_score is not None]
    negative = [float(row.visual_score) for row in rows if not row.label and row.visual_score is not None]
    return {
        "ranking": evaluate_pair_ranking(rows, k_values=(1, 3, 5)),
        "score_distributions": {"positive": _distribution(positive), "negative": _distribution(negative)},
        "hard_negative_margin": _pair_margins(rows),
        "threshold_0_80": {
            "positive_count": len(positive), "passed_count": sum(value >= VISUAL_SIMILARITY_REFERENCE for value in positive),
            "pass_rate": None if not positive else sum(value >= VISUAL_SIMILARITY_REFERENCE for value in positive) / len(positive),
        },
    }


def _mask_slice_id(value: float) -> str:
    if value < .70: return "<0.70"
    if value < .80: return "0.70-0.80"
    if value < .90: return "0.80-0.90"
    return ">=0.90"


def _positive_pair_mask_slice_scores(
    items: Sequence[CandidateInput], identities: Mapping[str, str],
    embeddings: Mapping[str, np.ndarray],
    mask_iou: Mapping[tuple[str, str], float],
) -> dict[str, list[float]]:
    result = {key: [] for key in MASK_IOU_SLICE_IDS}
    by_position: dict[int, list[CandidateInput]] = {}
    for item in items:
        by_position.setdefault(item.candidate.frame_index, []).append(item)
    for position in sorted(by_position):
        left = by_position[position]
        right = by_position.get(position + 1, [])
        for first in left:
            first_id = first.candidate.candidate_id
            first_key = (first.stream_id, first_id)
            if identities[first_id].startswith("fp:") or first_key not in mask_iou:
                continue
            for second in right:
                second_id = second.candidate.candidate_id
                second_key = (second.stream_id, second_id)
                if identities[first_id] != identities[second_id] or second_key not in mask_iou:
                    continue
                conservative_iou = min(mask_iou[first_key], mask_iou[second_key])
                result[_mask_slice_id(conservative_iou)].append(
                    _score(embeddings[first_id], embeddings[second_id])
                )
    return result


def _summarize_mask_slice_scores(values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {
        key: {
            "positive_pair_count": len(values.get(key, ())),
            "score_distribution": _distribution(values.get(key, ())),
            "reference_threshold": VISUAL_SIMILARITY_REFERENCE,
            "reference_pass_count": sum(score >= VISUAL_SIMILARITY_REFERENCE for score in values.get(key, ())),
            "reference_pass_rate": None if not values.get(key) else (
                sum(score >= VISUAL_SIMILARITY_REFERENCE for score in values[key]) / len(values[key])
            ),
        }
        for key in MASK_IOU_SLICE_IDS
    }


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return None if not defined else sum(defined) / len(defined)


def _recommendation(variants: Mapping[str, Any]) -> dict[str, Any]:
    bbox = variants[DINO_BBOX_VARIANT]["aggregate"]
    masked = variants[DINO_MASK_NEUTRAL_VARIANT]["aggregate"]
    b_macro = bbox["scene_group_macro"]["accepted"]["f1"]
    m_macro = masked["scene_group_macro"]["accepted"]["f1"]
    b_worst = bbox["worst_scene_group"]["accepted_f1"]
    m_worst = masked["worst_scene_group"]["accepted_f1"]
    b_ap = bbox["pairwise"]["ranking"]["average_precision"]
    m_ap = masked["pairwise"]["ranking"]["average_precision"]
    b_margin = bbox["pairwise"]["hard_negative_margin"]["aggregate"]["mean"]
    m_margin = masked["pairwise"]["hard_negative_margin"]["aggregate"]["mean"]
    same_coverage = (
        masked["coverage"] == bbox["coverage"]
        and masked["coverage"]["embedding_failures"] == 0
        and bbox["coverage"]["embedding_failures"] == 0
        and masked["coverage"]["hidden_fallback_count"] == 0
        and bbox["coverage"]["hidden_fallback_count"] == 0
    )
    macro_safe = m_macro is not None and b_macro is not None and m_macro >= b_macro - .01
    worst_safe = m_worst is not None and b_worst is not None and m_worst >= b_worst - .02
    primary_better = m_macro is not None and b_macro is not None and m_macro > b_macro
    practical_tie = m_macro is not None and b_macro is not None and abs(m_macro - b_macro) < .005
    secondary_better = all(value is not None for value in (b_ap, m_ap, b_margin, m_margin)) and m_ap > b_ap and m_margin > b_margin
    eligible = same_coverage and macro_safe and worst_safe and (primary_better or practical_tie and secondary_better)
    reasons = {
        "coverage_equal_and_no_hidden_fallback": same_coverage,
        "macro_accepted_f1_within_0.01": macro_safe,
        "worst_scene_accepted_f1_within_0.02": worst_safe,
        "primary_macro_accepted_f1_higher": primary_better,
        "practical_tie_under_0.005": practical_tie,
        "tie_secondary_pair_ap_and_margin_both_higher": secondary_better,
    }
    return {
        "rule_id": "ocid_representation_configured_selection_rule_v1",
        "selected_variant": DINO_MASK_NEUTRAL_VARIANT if eligible else DINO_BBOX_VARIANT,
        "mask_variant_eligible": eligible,
        "bbox_fallback_applied": not eligible,
        "bbox_fallback_on_practical_tie": practical_tie and not eligible,
        "checks": reasons,
        "values": {"bbox_macro_accepted_f1": b_macro, "mask_macro_accepted_f1": m_macro,
                   "bbox_worst_accepted_f1": b_worst, "mask_worst_accepted_f1": m_worst,
                   "bbox_pair_ap": b_ap, "mask_pair_ap": m_ap,
                   "bbox_mean_hard_negative_margin": b_margin,
                   "mask_mean_hard_negative_margin": m_margin},
    }


def _mask_iou_by_candidate(
    inference: Mapping[str, Any],
) -> dict[tuple[str, str], float]:
    """Read mask-evaluation data only after the persisted inference boundary.

    The report is provenance for an already-reviewed predicted-mask quality
    slice; it is never a model input and missing reports remain explicit.
    """
    sam2 = inference.get("sam2_inference_manifest")
    if not isinstance(sam2, Mapping) or not isinstance(sam2.get("path"), str):
        return {}
    manifest_path = Path(sam2["path"]).resolve(strict=True)
    report_path = manifest_path.parent.parent / "grounded_sam2_report.json"
    if not report_path.is_file():
        return {}
    report = _object(report_path, "mask evaluation report")
    report_inference = report.get("inference_manifest")
    if not isinstance(report_inference, Mapping) or report_inference.get("sha256") != _sha256(manifest_path):
        raise OcidEvaluationError("mask report does not match the persisted SAM2 inference manifest")
    values: dict[tuple[str, str], float] = {}
    for row in report.get("matched_observations", []):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("stream_id"), str)
            or not isinstance(row.get("candidate_id"), str)
        ):
            continue
        cleaned = row.get("cleaned")
        if isinstance(cleaned, Mapping) and isinstance(cleaned.get("iou"), (int, float)):
            values[(row["stream_id"], row["candidate_id"])] = float(cleaned["iou"])
    return values


def _validate_candidate_index(index: object, expected: Sequence[CandidateInput]) -> None:
    if not isinstance(index, list) or len(index) != len(expected):
        raise OcidEvaluationError("embedding candidate-index cardinality mismatch")
    for position, (actual, item) in enumerate(zip(index, expected, strict=True)):
        wanted = _candidate_index_row(position, item)
        if not isinstance(actual, Mapping):
            raise OcidEvaluationError(f"candidate_index[{position}] must be an object")
        for key in ("candidate_index", "stream_id", "candidate_id", "frame_id", "frame_index",
                    "mask_digest", "mask_ref"):
            if actual.get(key) != wanted[key]:
                raise OcidEvaluationError(f"candidate_index[{position}].{key} mismatch")
        bbox = actual.get("bbox")
        if not isinstance(bbox, Mapping) or not _same_bbox(_selected_bbox(bbox), item.candidate.bbox):
            raise OcidEvaluationError(f"candidate_index[{position}].bbox mismatch")


def _load_embedding_npz(path: Path, expected_count: int) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        if set(archive.files) != set(SUPPORTED_VARIANTS):
            raise OcidEvaluationError("embedding NPZ must contain exactly the two frozen variants")
        return {
            variant: _validate_embedding_matrix(
                np.asarray(archive[variant]), expected_count, variant
            ).copy()
            for variant in SUPPORTED_VARIANTS
        }


def evaluate_persisted_inference(
    *, inventory: Sequence[DevelopmentStream], inputs: Mapping[str, Sequence[CandidateInput]],
    inference_manifest_path: Path,
) -> dict[str, Any]:
    """Open reviewed annotations only after validating a persisted inference boundary."""
    manifest = _object(inference_manifest_path, "OCID representation evaluation inference manifest")
    if (
        manifest.get("schema_version") != INFERENCE_SCHEMA_VERSION
        or manifest.get("embedding_schema_version") != EMBEDDING_SCHEMA_VERSION
        or manifest.get("status") != "rgb_predicted_mask_inference_completed_before_ground_truth"
        or manifest.get("scope") != "development_only_ocid_representation_evaluation"
        or manifest.get("heldout_access") != "none"
    ):
        raise OcidEvaluationError("reviewed evaluation requires a persisted OCID representation evaluation inference boundary")
    if (
        manifest.get("candidate_count") != FROZEN_CANDIDATE_COUNT
        or manifest.get("variants") != list(SUPPORTED_VARIANTS)
        or manifest.get("frozen_model") != {
            "model_name": FROZEN_MODEL_NAME,
            "embedding_dimension": FROZEN_EMBEDDING_DIMENSION,
            "batch_size": FROZEN_BATCH_SIZE,
            "output_token": "cls",
        }
    ):
        raise OcidEvaluationError("persisted OCID representation inference violates its configured settings")
    for key, label in (("selected_candidates", "selected candidates"),
                       ("sam2_inference_manifest", "SAM2 inference manifest")):
        provenance = manifest.get(key)
        if not isinstance(provenance, Mapping) or not isinstance(provenance.get("path"), str):
            raise OcidEvaluationError(f"inference manifest is missing {label} provenance")
        source_path = Path(provenance["path"]).resolve(strict=True)
        if provenance.get("sha256") != _sha256(source_path):
            raise OcidEvaluationError(f"persisted {label} provenance SHA-256 mismatch")
    artifact = manifest.get("embedding_artifact")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        raise OcidEvaluationError("inference manifest is missing its embedding artifact")
    npz_path = Path(artifact["path"]).resolve(strict=True)
    if artifact.get("sha256") != _sha256(npz_path):
        raise OcidEvaluationError("embedding artifact SHA-256 mismatch")
    index = manifest.get("candidate_index")
    expected = [item for stream in inventory for item in inputs[stream.stream_id]]
    if len(inventory) != FROZEN_DEVELOPMENT_STREAM_COUNT or len(expected) != FROZEN_CANDIDATE_COUNT:
        raise OcidEvaluationError("evaluation requires the exact frozen OCID representation evaluation inventory")
    _validate_candidate_index(index, expected)
    embeddings = _load_embedding_npz(npz_path, len(expected))
    reports: dict[str, Any] = {variant: {"per_stream": {}} for variant in SUPPORTED_VARIANTS}
    pair_records: dict[str, dict[str, tuple[PairRankingRecord, ...]]] = {
        variant: {} for variant in SUPPORTED_VARIANTS
    }
    matcher_reports: dict[str, dict[str, dict[str, Any]]] = {
        variant: {} for variant in SUPPORTED_VARIANTS
    }
    slice_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        variant: {} for variant in SUPPORTED_VARIANTS
    }
    mask_iou = _mask_iou_by_candidate(manifest)
    cursor = 0
    marker_rows: dict[str, Any] = {variant: {"hypothesis_only": True, "status": "not_available"} for variant in SUPPORTED_VARIANTS}
    occlusion_rows: dict[str, Any] = {
        variant: {"status": "not_available"} for variant in SUPPORTED_VARIANTS
    }
    occlusion_visibility: dict[str, Any] | None = None
    marker_contact_sheet: dict[str, str] | None = None
    for stream in inventory:
        items = list(inputs[stream.stream_id]); positions = list(range(cursor, cursor + len(items))); cursor += len(items)
        annotation = load_annotation(stream.annotation_path, manifest_path=stream.stream_directory / "manifest.json")
        identity, bbox_iou, _candidate_evaluation = _assignment_identity(annotation, items)
        detector_tp = len(bbox_iou)
        detector_ceiling = {
            "bbox_iou_threshold": .50, "candidate_count": len(items),
            "gt_count": len(annotation.instances), "tp": detector_tp,
            "fp": len(items) - detector_tp, "fn": len(annotation.instances) - detector_tp,
        }
        oracle = tuple(OracleInstance(stream.stream_id, item.candidate.frame_id, item.candidate.frame_index,
                                      item.candidate.candidate_id, identity[item.candidate.candidate_id], item.candidate.bbox)
                       for item in items)
        decoded = _load_stream(stream.stream_directory)
        if stream.stream_id == MARKER_STREAM:
            occlusion_visibility = _source_label_visibility_proxy(stream, annotation)
            marker_contact_sheet = _write_marker_contact_sheet(
                decoded=decoded, items=items, identities=identity, annotation=annotation,
                artifact_root=inference_manifest_path.resolve().parent,
            )
        for variant in SUPPORTED_VARIANTS:
            by_id = {item.candidate.candidate_id: embeddings[variant][index].copy() for item, index in zip(items, positions, strict=True)}
            ranking, pairs = evaluate_oracle_embeddings(oracle, by_id, namespace=stream.stream_id)
            prepared = tuple(PreparedOracleInstance(row, None, item.candidate, item.mask_record)
                             for row, item in zip(oracle, items, strict=True))
            matcher = evaluate_oracle_matcher(decoded, prepared, by_id, variant=variant,
                                              model_name=FROZEN_MODEL_NAME, policy=MATCHER_POLICY)
            stream_slices = _positive_pair_mask_slice_scores(items, identity, by_id, mask_iou)
            pair_records[variant][stream.stream_id] = pairs
            matcher_reports[variant][stream.stream_id] = matcher
            slice_scores[variant][stream.stream_id] = stream_slices
            reports[variant]["per_stream"][stream.stream_id] = {
                "scene_group_id": stream.scene_group_id, "ranking": ranking, "matcher": matcher,
                "detector_ceiling": detector_ceiling,
                "mask_iou_positive_pair_slices": _summarize_mask_slice_scores(stream_slices),
            }
            if stream.stream_id == MARKER_STREAM:
                marker_rows[variant] = _marker_diagnostic(
                    annotation, items, by_id, identity, bbox_iou, mask_iou
                )
                if occlusion_visibility is None:
                    raise OcidEvaluationError(
                        "partial-occlusion visibility proxy was not initialized"
                    )
                occlusion_rows[variant] = _partial_occlusion_diagnostic(
                    annotation,
                    items,
                    by_id,
                    identity,
                    bbox_iou,
                    mask_iou,
                    occlusion_visibility,
                )
    scene_streams: dict[str, list[str]] = {}
    for stream in inventory:
        scene_streams.setdefault(stream.scene_group_id, []).append(stream.stream_id)
    for variant in SUPPORTED_VARIANTS:
        all_pairs = tuple(
            row for stream in inventory for row in pair_records[variant][stream.stream_id]
        )
        pooled_matcher = aggregate_oracle_matcher_reports(tuple(
            matcher_reports[variant][stream.stream_id] for stream in inventory
        ))
        pooled_slices = {key: [] for key in MASK_IOU_SLICE_IDS}
        for stream in inventory:
            for key in MASK_IOU_SLICE_IDS:
                pooled_slices[key].extend(slice_scores[variant][stream.stream_id][key])
        scene_groups: dict[str, Any] = {}
        for scene_id, stream_ids in sorted(scene_streams.items()):
            group_pairs = tuple(row for stream_id in stream_ids for row in pair_records[variant][stream_id])
            group_slices = {key: [] for key in MASK_IOU_SLICE_IDS}
            for stream_id in stream_ids:
                for key in MASK_IOU_SLICE_IDS:
                    group_slices[key].extend(slice_scores[variant][stream_id][key])
            group_detector = {
                field: sum(int(reports[variant]["per_stream"][stream_id]["detector_ceiling"][field]) for stream_id in stream_ids)
                for field in ("candidate_count", "gt_count", "tp", "fp", "fn")
            }
            scene_groups[scene_id] = {
                "stream_ids": sorted(stream_ids), "pairwise": _pair_report(group_pairs),
                "matcher": aggregate_oracle_matcher_reports(tuple(
                    matcher_reports[variant][stream_id] for stream_id in stream_ids
                )),
                "detector_ceiling": {"bbox_iou_threshold": .50, **group_detector},
                "mask_iou_positive_pair_slices": _summarize_mask_slice_scores(group_slices),
            }
        macro = {
            metric: _mean_defined([
                group["matcher"]["accepted"][metric] for group in scene_groups.values()
            ])
            for metric in ("precision", "recall", "f1")
        }
        worst_id = min(
            scene_groups,
            key=lambda scene_id: (
                -1.0 if scene_groups[scene_id]["matcher"]["accepted"]["f1"] is None
                else scene_groups[scene_id]["matcher"]["accepted"]["f1"],
                scene_id,
            ),
        )
        pooled_detector = {
            field: sum(int(reports[variant]["per_stream"][stream.stream_id]["detector_ceiling"][field]) for stream in inventory)
            for field in ("candidate_count", "gt_count", "tp", "fp", "fn")
        }
        reports[variant]["aggregate"] = {
            "coverage": {"candidate_count": len(expected), "valid_embedding_count": len(expected),
                         "embedding_failures": 0, "hidden_fallback_count": 0},
            "pairwise": _pair_report(all_pairs), "matcher": pooled_matcher,
            "detector_ceiling": {"bbox_iou_threshold": .50, **pooled_detector},
            "mask_iou_positive_pair_slices": _summarize_mask_slice_scores(pooled_slices),
            "scene_groups": scene_groups,
            "scene_group_macro": {"accepted": macro},
            "worst_scene_group": {
                "scene_group_id": worst_id,
                "accepted_f1": scene_groups[worst_id]["matcher"]["accepted"]["f1"],
            },
        }
    if occlusion_visibility is None:
        raise OcidEvaluationError("frozen partial-occlusion development stream was not evaluated")
    recommendation = _recommendation(reports)
    return {
        "schema_version": SCHEMA_VERSION, "status": "completed_development_only", "heldout_access": "none",
        "ground_truth_boundary": "validated_persisted_inference_manifest_before_reviewed_annotation_open",
        "variants": reports, "recommendation": recommendation,
        "marker_003_009": {"variants": marker_rows, "contact_sheet": marker_contact_sheet},
        "partial_occlusion": {
            "visibility_proxy": occlusion_visibility,
            "variants": occlusion_rows,
        },
        "limitations": [
            "Marker 003/009 is a hypothesis-only diagnostic, not a global visual-type threshold.",
            "Partial occlusion uses source-label visible area as an evaluation-only proxy, not physical visibility.",
            "False-positive candidates receive unique evaluation-only identities.",
            "Mask-IoU slices are evaluation-only and are never model inputs.",
            "The contact sheet uses development RGB and predicted SAM2 masks, never reviewed masks.",
            "No held-out data is read, embedded, visualized, or evaluated.",
        ],
    }


def _score(left: np.ndarray, right: np.ndarray) -> float:
    return (float(np.clip(np.dot(left.astype(np.float64), right.astype(np.float64)), -1.0, 1.0)) + 1.0) / 2.0


def _marker_diagnostic(
    annotation: Any, items: Sequence[CandidateInput], embeddings: Mapping[str, np.ndarray],
    identities: Mapping[str, str], bbox_iou: Mapping[str, float],
    mask_iou: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    reviewed_rows = [row for row in annotation.raw["expected_element_instances"] if isinstance(row, Mapping)]
    reference_labels = {
        int(row["source_label"]) for row in reviewed_rows
        if row.get("frame_id") == MARKER_FRAME and isinstance(row.get("source_label"), int)
    }
    if not set(MARKER_LABELS).issubset(reference_labels):
        raise OcidEvaluationError("reference frame_0019 must contain reviewed source labels 003 and 009")
    label_identities: dict[int, str] = {}
    for label in MARKER_LABELS:
        values = {str(row["visual_type_id"]) for row in reviewed_rows if row.get("source_label") == label}
        if len(values) != 1:
            raise OcidEvaluationError(f"reviewed marker label {label:03d} must map to one physical identity")
        label_identities[label] = next(iter(values))
    observations = {
        label: sorted(
            (item for item in items if identities[item.candidate.candidate_id] == label_identities[label]),
            key=lambda item: (item.candidate.frame_index, item.candidate.candidate_id),
        )
        for label in MARKER_LABELS
    }
    same_instance: dict[str, Any] = {}
    for label, values in observations.items():
        scores = [
            _score(embeddings[left.candidate.candidate_id], embeddings[right.candidate.candidate_id])
            for left, right in zip(values, values[1:])
            if right.candidate.frame_index == left.candidate.frame_index + 1
        ]
        same_instance[f"{label:03d}"] = _distribution(scores)
    cross_all = [
        _score(embeddings[first.candidate.candidate_id], embeddings[second.candidate.candidate_id])
        for first in observations[3] for second in observations[9]
    ]
    by_frame = {
        label: {item.candidate.frame_id: item for item in values}
        for label, values in observations.items()
    }
    co_visible_frames = sorted(set(by_frame[3]).intersection(by_frame[9]))
    cross_same_frame = [
        _score(embeddings[by_frame[3][frame].candidate.candidate_id],
               embeddings[by_frame[9][frame].candidate.candidate_id])
        for frame in co_visible_frames
    ]
    all_by_frame: dict[str, list[CandidateInput]] = {}
    for item in items:
        all_by_frame.setdefault(item.candidate.frame_id, []).append(item)
    rank_rows: list[dict[str, Any]] = []
    marker_to_other: list[float] = []
    for frame_id in co_visible_frames:
        for query_label, counterpart_label in ((3, 9), (9, 3)):
            query = by_frame[query_label][frame_id]
            counterpart = by_frame[counterpart_label][frame_id]
            query_id = query.candidate.candidate_id
            counterpart_id = counterpart.candidate.candidate_id
            scored = sorted(
                ((other.candidate.candidate_id, _score(embeddings[query_id], embeddings[other.candidate.candidate_id]))
                 for other in all_by_frame[frame_id] if other.candidate.candidate_id != query_id),
                key=lambda row: (-row[1], row[0]),
            )
            rank = next(index + 1 for index, row in enumerate(scored) if row[0] == counterpart_id)
            cross_score = next(score for candidate_id, score in scored if candidate_id == counterpart_id)
            others = [score for candidate_id, score in scored if candidate_id != counterpart_id]
            marker_to_other.extend(others)
            hardest = None if not others else max(others)
            rank_rows.append({
                "frame_id": frame_id, "query_label": f"{query_label:03d}",
                "counterpart_label": f"{counterpart_label:03d}", "counterpart_rank": rank,
                "gallery_count": len(scored), "cross_marker_score": cross_score,
                "hardest_other_score": hardest,
                "cross_minus_hardest_other_margin": None if hardest is None else cross_score - hardest,
                "reference_0_80_passed": cross_score >= VISUAL_SIMILARITY_REFERENCE,
            })
    continuity: dict[str, list[dict[str, Any]]] = {}
    frame_order = {frame_id: index for index, frame_id in enumerate(getattr(annotation, "frame_ids", ()), start=1)}
    for label in MARKER_LABELS:
        detected = by_frame[label]
        previous: CandidateInput | None = None
        rows: list[dict[str, Any]] = []
        gt_rows = sorted(
            (row for row in reviewed_rows if row.get("source_label") == label),
            key=lambda row: (frame_order.get(str(row.get("frame_id")), 10**9), str(row.get("frame_id"))),
        )
        for gt in gt_rows:
            frame_id = str(gt["frame_id"])
            item = detected.get(frame_id)
            similarity = None
            if item is not None and previous is not None:
                similarity = _score(
                    embeddings[previous.candidate.candidate_id], embeddings[item.candidate.candidate_id]
                )
            rows.append({
                "frame_id": frame_id, "gt_present": True,
                "detector_status": "tp" if item is not None else "miss",
                "candidate_id": None if item is None else item.candidate.candidate_id,
                "bbox_iou": None if item is None else bbox_iou.get(item.candidate.candidate_id),
                "mask_iou": None if item is None else mask_iou.get(
                    (item.stream_id, item.candidate.candidate_id)
                ),
                "similarity_to_previous_detected_observation": similarity,
            })
            if item is not None:
                previous = item
        continuity[f"{label:03d}"] = rows
    separation = {
        "exists": bool(cross_all and marker_to_other and min(cross_all) > max(marker_to_other)),
        "cross_marker_min": None if not cross_all else min(cross_all),
        "marker_to_other_max": None if not marker_to_other else max(marker_to_other),
        "gap": None if not cross_all or not marker_to_other else min(cross_all) - max(marker_to_other),
        "interpretation": "diagnostic interval on this frozen marker example only",
    }
    return {
        "hypothesis_only": True, "status": "available", "stream_id": MARKER_STREAM,
        "reference_frame": MARKER_FRAME, "reference_frame_reviewed_labels_verified": True,
        "reviewed_source_labels": ["003", "009"],
        "same_instance_adjacent_similarity": same_instance,
        "cross_marker_all_pairs": _distribution(cross_all),
        "cross_marker_same_frame": _distribution(cross_same_frame),
        "co_visible_frame_ranks_and_margins": rank_rows,
        "per_frame_continuity": continuity,
        "separation_interval": separation,
        "claim_boundary": (
            "Hypothesis-only diagnostic for one predeclared pair; it is not OCID visual-type ground truth, "
            "does not learn a threshold, and does not establish global color/viewpoint invariance."
        ),
    }


def _source_label_visibility_proxy(
    stream: DevelopmentStream, annotation: Any,
) -> dict[str, Any]:
    """Reproduce the benchmark's frozen low-visibility flag on development.

    Source label masks are evaluation-only and are opened only after the
    persisted DINOv2 inference boundary has been validated.
    """
    if stream.stream_id != MARKER_STREAM:
        raise OcidEvaluationError(
            "partial-occlusion proxy is restricted to the frozen development stream"
        )
    metadata = annotation.raw.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("role") != "development":
        raise OcidEvaluationError("partial-occlusion annotation must have the development role")
    source_sequence = metadata.get("source_sequence")
    if not isinstance(source_sequence, str):
        raise OcidEvaluationError("partial-occlusion annotation is missing source_sequence")
    if source_sequence != OCCLUSION_SOURCE_SEQUENCE:
        raise OcidEvaluationError(
            "partial-occlusion source_sequence does not match the frozen development target"
        )
    sequence_path = PurePosixPath(source_sequence.replace("\\", "/"))
    if sequence_path.is_absolute() or ".." in sequence_path.parts:
        raise OcidEvaluationError("partial-occlusion source_sequence must be a safe relative path")

    annotation_path = Path(annotation.path).resolve(strict=True)
    reviewed_root = annotation_path.parents[2]
    raw_root = (reviewed_root.parents[1] / "raw" / "OCID-dataset").resolve(strict=True)
    label_root = (raw_root / Path(*sequence_path.parts) / "label").resolve(strict=True)
    try:
        label_root.relative_to(raw_root)
    except ValueError as error:
        raise OcidEvaluationError("partial-occlusion label root escapes the OCID raw root") from error

    manifest = _object(
        stream.stream_directory / "manifest.json", "partial-occlusion stream manifest"
    )
    frame_entries = manifest.get("frames")
    if not isinstance(frame_entries, list):
        raise OcidEvaluationError("partial-occlusion stream manifest frames must be a list")
    entries: dict[str, Mapping[str, Any]] = {}
    for row in frame_entries:
        if not isinstance(row, Mapping) or not isinstance(row.get("frame_id"), str):
            raise OcidEvaluationError("partial-occlusion stream manifest has a malformed frame")
        entries[row["frame_id"]] = row
    if set(entries) != set(annotation.frame_ids):
        raise OcidEvaluationError("partial-occlusion manifest and annotation frame sets differ")

    areas: dict[str, int] = {}
    masks: dict[str, np.ndarray] = {}
    source_labels: dict[str, np.ndarray] = {}
    for frame_id in annotation.frame_ids:
        frame_metadata = entries[frame_id].get("metadata")
        if not isinstance(frame_metadata, Mapping):
            raise OcidEvaluationError("partial-occlusion frame metadata is missing")
        filename = frame_metadata.get("ocid_source_filename")
        if not isinstance(filename, str):
            raise OcidEvaluationError("partial-occlusion frame source filename is missing")
        source_path = _under(label_root, filename, "partial-occlusion source label")
        try:
            with Image.open(source_path) as image:
                labels = np.asarray(image).copy()
        except OSError as error:
            raise OcidEvaluationError(
                f"cannot decode partial-occlusion source label: {error}"
            ) from error
        if labels.ndim != 2:
            raise OcidEvaluationError("partial-occlusion source label must be two-dimensional")
        mask = np.asarray(labels == OCCLUSION_SOURCE_LABEL, dtype=np.bool_)
        area = int(np.count_nonzero(mask))
        if area:
            areas[frame_id] = area
            masks[frame_id] = mask
            source_labels[frame_id] = labels

    reviewed_frames = {
        str(row["frame_id"])
        for row in annotation.raw.get("expected_element_instances", [])
        if isinstance(row, Mapping) and row.get("source_label") == OCCLUSION_SOURCE_LABEL
    }
    if set(areas) != reviewed_frames:
        raise OcidEvaluationError(
            "partial-occlusion source-label and reviewed observation frames differ"
        )
    if OCCLUSION_FLAG_FRAME not in areas:
        raise OcidEvaluationError("partial-occlusion flagged frame is missing the target observation")
    maximum = max(areas.values())
    trajectory = [
        {
            "frame_id": frame_id,
            "visible_pixels": areas[frame_id],
            "relative_to_instance_max": areas[frame_id] / maximum,
        }
        for frame_id in annotation.frame_ids
        if frame_id in areas
    ]
    flagged_ratio = areas[OCCLUSION_FLAG_FRAME] / maximum
    if flagged_ratio >= OCCLUSION_VISIBILITY_THRESHOLD:
        raise OcidEvaluationError(
            "partial-occlusion flagged frame does not satisfy the frozen visibility threshold"
        )

    flagged_position = annotation.frame_ids.index(OCCLUSION_FLAG_FRAME)
    if flagged_position == 0:
        raise OcidEvaluationError("partial-occlusion flagged frame has no predecessor")
    previous_frame = annotation.frame_ids[flagged_position - 1]
    if previous_frame not in masks:
        raise OcidEvaluationError(
            "partial-occlusion target is not visible in the predecessor frame"
        )
    previous = masks[previous_frame]
    current = masks[OCCLUSION_FLAG_FRAME]
    if previous.shape != current.shape:
        raise OcidEvaluationError("partial-occlusion source label shapes differ")
    lost = np.logical_and(previous, np.logical_not(current))
    lost_pixels = int(np.count_nonzero(lost))
    support_label = metadata.get("audited_support_label")
    if isinstance(support_label, bool) or not isinstance(support_label, int):
        raise OcidEvaluationError(
            "partial-occlusion annotation has no audited support label"
        )
    reassigned = np.logical_and(
        lost, source_labels[OCCLUSION_FLAG_FRAME] > support_label
    )
    reassigned_pixels = int(np.count_nonzero(reassigned))

    timeline_path = (
        reviewed_root / "review" / "timelines" / f"{stream.stream_id}.csv"
    ).resolve(strict=True)
    with timeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        timeline_rows = list(csv.DictReader(handle))
    flagged_rows = [
        row for row in timeline_rows if row.get("frame_id") == OCCLUSION_FLAG_FRAME
    ]
    target_id = f"{stream.stream_id}__label_{OCCLUSION_SOURCE_LABEL:03d}"
    if len(flagged_rows) != 1:
        raise OcidEvaluationError(
            "partial-occlusion timeline must contain exactly one flagged frame"
        )
    timeline = flagged_rows[0]
    flags = set(str(timeline.get("automatic_flags", "")).split(";"))
    severe_ids = set(
        str(timeline.get("severe_interframe_visibility_drop_ids", "")).split(";")
    )
    if "relative_visible_area_below_0_10" not in flags or target_id not in severe_ids:
        raise OcidEvaluationError(
            "partial-occlusion target is not backed by the frozen benchmark flags"
        )

    return {
        "proxy_id": "source_label_visible_area_v1",
        "ground_truth_role": "evaluation_only_after_persisted_inference",
        "stream_id": stream.stream_id,
        "source_sequence": source_sequence,
        "source_label": f"{OCCLUSION_SOURCE_LABEL:03d}",
        "physical_instance_proxy_id": target_id,
        "automatic_flag": "relative_visible_area_below_0_10",
        "threshold": OCCLUSION_VISIBILITY_THRESHOLD,
        "flagged_frame": OCCLUSION_FLAG_FRAME,
        "maximum_visible_pixels": maximum,
        "flagged_visible_pixels": areas[OCCLUSION_FLAG_FRAME],
        "flagged_relative_to_instance_max": flagged_ratio,
        "trajectory": trajectory,
        "final_transition": {
            "from_frame_id": previous_frame,
            "to_frame_id": OCCLUSION_FLAG_FRAME,
            "previous_visible_pixels": areas[previous_frame],
            "current_visible_pixels": areas[OCCLUSION_FLAG_FRAME],
            "current_over_previous": areas[OCCLUSION_FLAG_FRAME] / areas[previous_frame],
            "lost_visible_pixels": lost_pixels,
            "lost_pixels_reassigned_to_other_object_labels": reassigned_pixels,
            "reassigned_fraction_of_lost": (
                None if not lost_pixels else reassigned_pixels / lost_pixels
            ),
        },
        "claim_boundary": (
            "Visible source-mask area is an occlusion proxy, not proof of physical removal "
            "or full 3D visibility."
        ),
    }


def _partial_occlusion_diagnostic(
    annotation: Any,
    items: Sequence[CandidateInput],
    embeddings: Mapping[str, np.ndarray],
    identities: Mapping[str, str],
    bbox_iou: Mapping[str, float],
    mask_iou: Mapping[tuple[str, str], float],
    visibility: Mapping[str, Any],
) -> dict[str, Any]:
    reviewed_rows = [
        row
        for row in annotation.raw.get("expected_element_instances", [])
        if isinstance(row, Mapping) and row.get("source_label") == OCCLUSION_SOURCE_LABEL
    ]
    identity_values = {str(row["visual_type_id"]) for row in reviewed_rows}
    if len(identity_values) != 1:
        raise OcidEvaluationError(
            "partial-occlusion target must map to one physical identity"
        )
    target_identity = next(iter(identity_values))
    detected = {
        item.candidate.frame_id: item
        for item in items
        if identities[item.candidate.candidate_id] == target_identity
    }
    frame_order = {
        frame_id: index for index, frame_id in enumerate(annotation.frame_ids)
    }
    expected_frames = sorted(
        (str(row["frame_id"]) for row in reviewed_rows), key=frame_order.__getitem__
    )
    visibility_rows = {
        str(row["frame_id"]): row
        for row in visibility.get("trajectory", [])
        if isinstance(row, Mapping) and isinstance(row.get("frame_id"), str)
    }
    if set(visibility_rows) != set(expected_frames):
        raise OcidEvaluationError(
            "partial-occlusion visibility and reviewed trajectories differ"
        )
    rows: list[dict[str, Any]] = []
    adjacent_scores: list[float] = []
    previous_frame: str | None = None
    for frame_id in expected_frames:
        item = detected.get(frame_id)
        similarity = None
        if item is not None and previous_frame is not None:
            previous_item = detected.get(previous_frame)
            if previous_item is not None:
                similarity = _score(
                    embeddings[previous_item.candidate.candidate_id],
                    embeddings[item.candidate.candidate_id],
                )
                adjacent_scores.append(similarity)
        rows.append(
            {
                **dict(visibility_rows[frame_id]),
                "detector_status": "tp" if item is not None else "miss",
                "candidate_id": None if item is None else item.candidate.candidate_id,
                "bbox_iou": (
                    None if item is None else bbox_iou.get(item.candidate.candidate_id)
                ),
                "mask_iou": (
                    None
                    if item is None
                    else mask_iou.get((item.stream_id, item.candidate.candidate_id))
                ),
                "similarity_to_previous_frame": similarity,
            }
        )
        previous_frame = frame_id
    misses = [row["frame_id"] for row in rows if row["detector_status"] == "miss"]
    detected_frames = [
        row["frame_id"] for row in rows if row["detector_status"] == "tp"
    ]
    flagged = next(row for row in rows if row["frame_id"] == OCCLUSION_FLAG_FRAME)
    return {
        "status": "available",
        "source_label": f"{OCCLUSION_SOURCE_LABEL:03d}",
        "expected_observation_count": len(rows),
        "detected_observation_count": len(detected_frames),
        "missed_observation_count": len(misses),
        "detected_frames": detected_frames,
        "missed_frames": misses,
        "first_missed_frame": None if not misses else misses[0],
        "last_detected_frame": None if not detected_frames else detected_frames[-1],
        "flagged_frame_detector_status": flagged["detector_status"],
        "same_instance_adjacent_similarity": _distribution(adjacent_scores),
        "adjacent_reference_pass_count": sum(
            score >= VISUAL_SIMILARITY_REFERENCE for score in adjacent_scores
        ),
        "trajectory": rows,
        "claim_boundary": (
            "DINOv2 continuity is measurable only while the extractor supplies a candidate; "
            "misses are extractor limits, not embedding failures."
        ),
    }


def _write_marker_contact_sheet(
    *, decoded: Any, items: Sequence[CandidateInput], identities: Mapping[str, str],
    annotation: Any, artifact_root: Path,
) -> dict[str, str]:
    labels = {
        int(row["source_label"]): str(row["visual_type_id"])
        for row in annotation.raw["expected_element_instances"]
        if isinstance(row, Mapping) and row.get("frame_id") == MARKER_FRAME
        and isinstance(row.get("source_label"), int) and row.get("source_label") in MARKER_LABELS
    }
    frame = next(value for value in decoded.frames if value.frame_id == MARKER_FRAME)
    canvas = Image.frombytes("RGB", (frame.image_size.width, frame.image_size.height), frame.rgb_bytes).convert("RGBA")
    colors = {3: (255, 210, 0, 100), 9: (0, 220, 255, 100)}
    draw = ImageDraw.Draw(canvas)
    for label in MARKER_LABELS:
        matches = [item for item in items if item.candidate.frame_id == MARKER_FRAME
                   and identities[item.candidate.candidate_id] == labels[label]]
        for item in matches:
            x, y, w, h = map(int, (item.candidate.bbox.x, item.candidate.bbox.y,
                                   item.candidate.bbox.width, item.candidate.bbox.height))
            local_alpha = Image.fromarray(item.mask_record.mask.astype(np.uint8) * colors[label][3], mode="L")
            overlay = Image.new("RGBA", canvas.size, colors[label][:3] + (0,))
            overlay.paste(colors[label][:3] + (colors[label][3],), (x, y, x + w, y + h), local_alpha)
            canvas = Image.alpha_composite(canvas, overlay)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((x, y, x + w - 1, y + h - 1), outline=colors[label][:3] + (255,), width=2)
            draw.text((x + 3, max(0, y - 13)), f"label {label:03d}", fill=colors[label][:3] + (255,))
    destination = artifact_root.resolve(strict=False) / "marker_003_009_contact_sheet.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination, format="PNG", optimize=True)
    return {"relative_path": destination.relative_to(artifact_root.resolve()).as_posix(),
            "sha256": _sha256(destination)}


def run_ocid_representation_evaluation(*, selected_candidates_path: Path, sam2_inference_manifest_path: Path,
                benchmark_spec_path: Path, reviewed_root: Path, dino_source: Path, dino_checkpoint: Path,
                expected_selected_candidates_sha256: str, expected_sam2_inference_sha256: str,
                expected_checkpoint_sha256: str, expected_checkpoint_size_bytes: int,
                expected_source_fingerprint: str, device: str,
                artifact_root: Path, output_path: Path) -> dict[str, Any]:
    if (
        isinstance(expected_checkpoint_size_bytes, bool)
        or not isinstance(expected_checkpoint_size_bytes, int)
        or expected_checkpoint_size_bytes <= 0
    ):
        raise OcidEvaluationError("expected checkpoint size must be a positive integer")
    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    inputs, _selected, _sam2 = load_predicted_candidate_inputs(
        inventory=inventory, selected_candidates_path=selected_candidates_path,
        sam2_inference_manifest_path=sam2_inference_manifest_path,
        expected_selected_candidates_sha256=expected_selected_candidates_sha256,
        expected_sam2_inference_sha256=expected_sam2_inference_sha256,
    )
    provider = LocalDinoV2Provider(source_dir=dino_source, checkpoint_path=dino_checkpoint,
                                   expected_checkpoint_sha256=expected_checkpoint_sha256,
                                   expected_checkpoint_size_bytes=expected_checkpoint_size_bytes,
                                   expected_source_tree_fingerprint=expected_source_fingerprint,
                                   model_name=FROZEN_MODEL_NAME,
                                   embedding_dimension=FROZEN_EMBEDDING_DIMENSION,
                                   device_policy=device, batch_size=FROZEN_BATCH_SIZE)
    inference = run_predicted_mask_inference(inventory=inventory, inputs=inputs, provider=provider, artifact_root=artifact_root,
                                             selected_candidates_path=selected_candidates_path,
                                             sam2_inference_manifest_path=sam2_inference_manifest_path)
    report = evaluate_persisted_inference(inventory=inventory, inputs=inputs,
                                          inference_manifest_path=artifact_root / "inference_manifest.json")
    report["inference_manifest"] = {"path": (artifact_root / "inference_manifest.json").resolve().as_posix(),
                                    "sha256": _sha256((artifact_root / "inference_manifest.json").resolve())}
    write_canonical_json_atomic(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--sam2-inference-manifest", type=Path, required=True)
    parser.add_argument("--expected-selected-candidates-sha256", required=True)
    parser.add_argument("--expected-sam2-inference-sha256", required=True)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--dino-source", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-size-bytes", type=int, required=True)
    parser.add_argument("--expected-source-fingerprint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_ocid_representation_evaluation(selected_candidates_path=args.selected_candidates, sam2_inference_manifest_path=args.sam2_inference_manifest,
                    expected_selected_candidates_sha256=args.expected_selected_candidates_sha256,
                    expected_sam2_inference_sha256=args.expected_sam2_inference_sha256,
                    benchmark_spec_path=args.benchmark_spec, reviewed_root=args.reviewed_root, dino_source=args.dino_source,
                    dino_checkpoint=args.dino_checkpoint, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                    expected_checkpoint_size_bytes=args.expected_checkpoint_size_bytes,
                    expected_source_fingerprint=args.expected_source_fingerprint, device=args.device,
                    artifact_root=args.artifact_root, output_path=args.output)
    except (OcidEvaluationError, OSError, RuntimeError, ValueError) as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
