"""Evaluate bbox-prompted SAM2 masks for the selected OCID development candidates.

All RGB inference and prediction-only artifacts are completed before this tool
opens reviewed annotations or OCID label images.  The selected Grounding DINO
bbox remains the candidate and explicit fallback; SAM2 is evaluated only as a
mask enrichment layer.  The frozen held-out streams are rejected by the shared
OCID evaluation inventory contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation import PredictedCandidate
from stream_analysis.evaluation.annotation import load_annotation
from stream_analysis.evaluation.component_review import (
    analyze_component_mask,
    binary_mask_sha256,
    component_union_mask,
)
from stream_analysis.evaluation.runner import _assign_frame_candidates

try:  # Support module and direct-script execution.
    from tools import evaluate_ocid_grounding_dino_extractor as grounding
    from tools.evaluate_ocid_grounding_dino_hardening import (
        SELECTED_CANDIDATE_SCHEMA,
    )
    from tools.ocid_evaluation_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        DevelopmentStream,
        OcidEvaluationError,
        load_development_inventory,
        write_canonical_json_atomic,
    )
    from tools.ocid_grounded_sam2_refinement import (
        GroundedMaskResult,
        MaskCleanupConfig,
        load_local_sam2_bbox_refiner,
        mask_bbox_from_full_frame,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback.
    import evaluate_ocid_grounding_dino_extractor as grounding  # type: ignore[no-redef]
    from evaluate_ocid_grounding_dino_hardening import (  # type: ignore[no-redef]
        SELECTED_CANDIDATE_SCHEMA,
    )
    from ocid_evaluation_common import (  # type: ignore[no-redef]
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        DevelopmentStream,
        OcidEvaluationError,
        load_development_inventory,
        write_canonical_json_atomic,
    )
    from ocid_grounded_sam2_refinement import (  # type: ignore[no-redef]
        GroundedMaskResult,
        MaskCleanupConfig,
        load_local_sam2_bbox_refiner,
        mask_bbox_from_full_frame,
    )


SCHEMA_VERSION = "ocid-grounded-sam2-mask-v1"
INFERENCE_SCHEMA_VERSION = "ocid-grounded-sam2-rgb-inference.v1"
FROZEN_BBOX_IOU = 0.50
FROZEN_CLEANUP = MaskCleanupConfig(
    min_component_pixels=16,
    min_component_area_ratio=0.005,
    connectivity=8,
)
SAM2_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "sam2.1_hiera_t.yaml",
)

# A fixed high-contrast palette keeps a candidate's color tied to its original
# frame-local index.  Twenty-four entries avoid the ambiguous four-color reuse
# that made dense prediction overlays difficult to audit.
OVERLAY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 220, 255),
    (255, 120, 30),
    (120, 255, 80),
    (220, 80, 255),
    (255, 210, 0),
    (60, 140, 255),
    (255, 70, 120),
    (100, 240, 200),
    (185, 105, 255),
    (255, 175, 105),
    (55, 205, 120),
    (255, 105, 210),
    (175, 225, 50),
    (75, 180, 235),
    (245, 75, 75),
    (125, 235, 245),
    (215, 155, 255),
    (245, 200, 120),
    (80, 170, 80),
    (250, 135, 170),
    (145, 145, 255),
    (210, 180, 55),
    (40, 190, 180),
    (240, 105, 105),
)


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    frame_id: str
    frame_index: int
    score: float
    phrase: str
    bbox: BBox


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_asset_provenance(model_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SAM2_CONFIG_FILES:
        path = (model_root / name).resolve(strict=True)
        if not path.is_file():
            raise OcidEvaluationError(f"SAM2 model asset must be a file: {name}")
        result[name] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcidEvaluationError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise OcidEvaluationError(f"{label} must be a JSON object")
    return payload


def _bbox(payload: Mapping[str, Any]) -> BBox:
    try:
        value = BBox(
            x=float(payload["x"]),
            y=float(payload["y"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OcidEvaluationError(f"malformed candidate bbox: {error}") from error
    if value.width <= 0 or value.height <= 0:
        raise OcidEvaluationError("candidate bbox must have positive area")
    return value


def load_selected_candidates(
    path: Path,
    inventory: Sequence[DevelopmentStream],
) -> tuple[dict[str, tuple[CandidateRecord, ...]], dict[str, Any]]:
    """Validate the selected candidate manifest and its configured inventory."""

    payload = _json_object(path, "selected candidate manifest")
    if payload.get("schema_version") != SELECTED_CANDIDATE_SCHEMA:
        raise OcidEvaluationError("unsupported selected candidate schema")
    if (
        payload.get("scope") != "development_only_selected_candidates"
        or payload.get("heldout_access") != "none"
    ):
        raise OcidEvaluationError("selected candidates do not match the configured inventory")
    streams = payload.get("streams")
    expected = {item.stream_id for item in inventory}
    if not isinstance(streams, Mapping) or set(streams) != expected:
        raise OcidEvaluationError("selected candidates must contain exact development streams")

    output: dict[str, tuple[CandidateRecord, ...]] = {}
    for item in inventory:
        stream = streams[item.stream_id]
        if not isinstance(stream, Mapping):
            raise OcidEvaluationError(f"malformed selected stream {item.stream_id}")
        frame_sizes = stream.get("frame_sizes")
        rows = stream.get("candidates")
        if not isinstance(frame_sizes, Mapping) or len(frame_sizes) != item.frame_count:
            raise OcidEvaluationError(f"frame-size contract mismatch for {item.stream_id}")
        if not isinstance(rows, list):
            raise OcidEvaluationError(f"candidate rows missing for {item.stream_id}")
        parsed: list[CandidateRecord] = []
        stream_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise OcidEvaluationError("candidate row must be an object")
            candidate_id = str(row.get("candidate_id", ""))
            frame_id = str(row.get("frame_id", ""))
            bbox_payload = row.get("bbox")
            if not candidate_id or candidate_id in stream_ids:
                raise OcidEvaluationError("candidate IDs must be non-empty and unique per stream")
            if frame_id not in frame_sizes or not isinstance(bbox_payload, Mapping):
                raise OcidEvaluationError("candidate references an unknown frame or bbox")
            if row.get("geometry_rejected") is not False:
                raise OcidEvaluationError("selected manifest contains a rejected candidate")
            stream_ids.add(candidate_id)
            parsed.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    frame_id=frame_id,
                    frame_index=int(row["frame_index"]),
                    score=float(row["score"]),
                    phrase=str(row.get("phrase", "")),
                    bbox=_bbox(bbox_payload),
                )
            )
        output[item.stream_id] = tuple(parsed)
    return output, payload


def _safe_artifact_name(candidate_id: str, source_index: int) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"candidate_{source_index:03d}_{digest}"


def _save_mask(path: Path, mask: np.ndarray) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent
    )
    os.close(handle)
    try:
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
            temporary_name, format="PNG", optimize=True
        )
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()


def _bbox_payload(value: BBox | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value.x, "y": value.y, "width": value.width, "height": value.height}


def _result_record(
    candidate: CandidateRecord,
    result: GroundedMaskResult,
    *,
    stream_id: str,
    raw_path: Path | None,
    cleaned_path: Path | None,
    artifact_root: Path,
) -> dict[str, Any]:
    quality = result.quality
    return {
        "stream_id": stream_id,
        "candidate_id": candidate.candidate_id,
        "frame_id": candidate.frame_id,
        "frame_index": candidate.frame_index,
        "score": candidate.score,
        "phrase": candidate.phrase,
        "source_bbox": _bbox_payload(candidate.bbox),
        "status": result.status,
        "fallback_reason": result.fallback_reason,
        "mask_bbox": _bbox_payload(result.mask_bbox),
        "raw_mask": None
        if raw_path is None
        else {
            "path": _relative(raw_path, artifact_root),
            "binary_mask_sha256": binary_mask_sha256(result.raw_mask),
        },
        "cleaned_mask": None
        if cleaned_path is None
        else {
            "path": _relative(cleaned_path, artifact_root),
            "binary_mask_sha256": binary_mask_sha256(result.cleaned_mask),
        },
        "quality": None
        if quality is None
        else {
            "selected_mask_index": quality.selected_mask_index,
            "predicted_iou": quality.predicted_iou,
            "object_score_logit": quality.object_score_logit,
            "raw_foreground_pixels": quality.raw_foreground_pixels,
            "cleaned_foreground_pixels": quality.cleaned_foreground_pixels,
            "removed_component_count": quality.removed_component_count,
            "removed_foreground_pixels": quality.removed_foreground_pixels,
        },
        "details": dict(result.details),
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    interior = value.copy()
    interior[1:, :] &= value[:-1, :]
    interior[:-1, :] &= value[1:, :]
    interior[:, 1:] &= value[:, :-1]
    interior[:, :-1] &= value[:, 1:]
    return value & ~interior


def _overlay_indices(indices: Sequence[int], *, candidate_count: int, label: str) -> frozenset[int]:
    normalized: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{label} must contain integer candidate indices")
        if not 0 <= index < candidate_count:
            raise ValueError(f"{label} contains out-of-range candidate index {index}")
        normalized.add(index)
    return frozenset(normalized)


def _prediction_label(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("prediction index must be a non-negative integer")
    return f"P{index:02d}"


def render_prediction_overlay(
    rgb: np.ndarray,
    candidates: Sequence[CandidateRecord],
    results: Sequence[GroundedMaskResult],
    *,
    hidden_indices: Sequence[int] = (),
) -> Image.Image:
    """Render one auditable prediction overlay without changing mask data.

    Colors are assigned from the original frame-local candidate index, even
    when other candidates are hidden.  Larger masks are composited first so
    that smaller masks remain visible instead of being painted over by a union
    mask.  ``hidden_indices`` is visualization-only and does not alter the
    candidates, results, persisted masks, or evaluation.
    """

    pairs = tuple(zip(candidates, results, strict=True))
    hidden = _overlay_indices(
        hidden_indices,
        candidate_count=len(pairs),
        label="hidden_indices",
    )
    canvas = np.asarray(rgb, dtype=np.float32).copy()
    mask_rows = [
        (index, result, int(np.count_nonzero(result.cleaned_mask)))
        for index, (_, result) in enumerate(pairs)
        if index not in hidden and result.cleaned_mask is not None
    ]
    for index, result, _ in sorted(mask_rows, key=lambda row: (-row[2], row[0])):
        color = np.asarray(
            OVERLAY_PALETTE[index % len(OVERLAY_PALETTE)],
            dtype=np.float32,
        )
        mask = np.asarray(result.cleaned_mask, dtype=np.bool_)
        canvas[mask] = canvas[mask] * 0.62 + color * 0.38
        canvas[_mask_boundary(mask)] = color
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, (candidate, result) in enumerate(pairs):
        if index in hidden:
            continue
        color = (
            OVERLAY_PALETTE[index % len(OVERLAY_PALETTE)]
            if result.status == "valid"
            else (255, 40, 40)
        )
        box = candidate.bbox
        draw.rectangle((box.left, box.top, box.right - 1, box.bottom - 1), outline=color, width=2)
        draw.text(
            (box.left + 2, max(0, box.top - 12)),
            _prediction_label(index),
            fill=color,
        )
    return image


def render_individual_prediction_overlays(
    rgb: np.ndarray,
    candidates: Sequence[CandidateRecord],
    results: Sequence[GroundedMaskResult],
    *,
    candidate_indices: Sequence[int] | None = None,
) -> dict[int, Image.Image]:
    """Render one isolated overlay per requested frame-local candidate index."""

    pairs = tuple(zip(candidates, results, strict=True))
    selected = (
        frozenset(range(len(pairs)))
        if candidate_indices is None
        else _overlay_indices(
            candidate_indices,
            candidate_count=len(pairs),
            label="candidate_indices",
        )
    )
    all_indices = frozenset(range(len(pairs)))
    return {
        index: render_prediction_overlay(
            rgb,
            candidates,
            results,
            hidden_indices=tuple(sorted(all_indices - {index})),
        )
        for index in sorted(selected)
    }


def _prediction_overlay(
    rgb: np.ndarray,
    candidates: Sequence[CandidateRecord],
    results: Sequence[GroundedMaskResult],
) -> Image.Image:
    """Compatibility wrapper for callers of ``_prediction_overlay``."""

    return render_prediction_overlay(rgb, candidates, results)


def render_presentation_overlays(
    *,
    inventory: Sequence[DevelopmentStream],
    inference_manifest: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, list[str]]:
    """Render mask/contour-only overlays from persisted RGB-only predictions."""

    palette = (
        (0, 220, 255),
        (255, 120, 30),
        (120, 255, 80),
        (220, 80, 255),
        (255, 210, 0),
        (60, 140, 255),
        (255, 70, 120),
        (100, 240, 200),
    )
    output: dict[str, list[str]] = {}
    for item in inventory:
        manifest = _json_object(item.stream_directory / "manifest.json", "analysis manifest")
        frame_path = {
            str(row["frame_id"]): item.stream_directory / str(row["image_path"])
            for row in manifest["frames"]
        }
        by_frame: dict[str, list[Mapping[str, Any]]] = {}
        for record in inference_manifest["records_by_stream"][item.stream_id]:
            by_frame.setdefault(str(record["frame_id"]), []).append(record)
        stream_paths: list[str] = []
        for frame_id, records in sorted(by_frame.items()):
            with Image.open(frame_path[frame_id]) as source:
                canvas = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
            fallback_boxes: list[Mapping[str, Any]] = []
            for index, record in enumerate(records):
                mask_payload = record.get("cleaned_mask")
                if not isinstance(mask_payload, Mapping):
                    fallback_boxes.append(record["source_bbox"])
                    continue
                mask = _load_mask(artifact_root / str(mask_payload["path"]))
                color = np.asarray(palette[index % len(palette)], dtype=np.float32)
                canvas[mask] = canvas[mask] * 0.58 + color * 0.42
                boundary = _mask_boundary(mask)
                canvas[boundary] = color
            image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
            if fallback_boxes:
                draw = ImageDraw.Draw(image)
                for box in fallback_boxes:
                    draw.rectangle(
                        (
                            float(box["x"]),
                            float(box["y"]),
                            float(box["x"]) + float(box["width"]) - 1,
                            float(box["y"]) + float(box["height"]) - 1,
                        ),
                        outline=(255, 40, 40),
                        width=2,
                    )
            path = artifact_root / "presentation_overlays" / item.stream_id / f"{frame_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            stream_paths.append(_relative(path, artifact_root))
        output[item.stream_id] = stream_paths
    write_canonical_json_atomic(
        artifact_root / "presentation_overlay_manifest.json",
        {
            "schema_version": "ocid-grounded-sam2-presentation-overlays.v1",
            "scope": "development_only_prediction_visualization",
            "heldout_access": "none",
            "rendering": "cleaned_mask_alpha_plus_contour_no_bbox_for_valid_masks",
            "fallback_rendering": "red_bbox_only",
            "overlays": output,
        },
    )
    return output


def run_rgb_inference(
    *,
    inventory: Sequence[DevelopmentStream],
    selected: Mapping[str, Sequence[CandidateRecord]],
    selected_manifest_path: Path,
    model_directory: Path,
    expected_model_sha256: str,
    device: str,
    artifact_root: Path,
) -> dict[str, Any]:
    """Complete RGB-only SAM2 inference and persist a pre-GT manifest."""

    model_root = model_directory.resolve(strict=True)
    checkpoint = model_root / "model.safetensors"
    if not checkpoint.is_file() or _sha256(checkpoint) != expected_model_sha256.casefold():
        raise OcidEvaluationError("SAM2 model.safetensors SHA-256 mismatch")
    refiner = load_local_sam2_bbox_refiner(model_root, device=device)
    torch_module = refiner.torch_module
    import transformers

    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records_by_stream: dict[str, list[dict[str, Any]]] = {}
    runtime_rows: list[dict[str, Any]] = []
    overlay_paths: dict[str, list[str]] = {}

    for item in inventory:
        decoded = grounding._load_stream(item.stream_directory)
        by_frame: dict[str, list[CandidateRecord]] = {}
        for candidate in selected[item.stream_id]:
            by_frame.setdefault(candidate.frame_id, []).append(candidate)
        records: list[dict[str, Any]] = []
        overlay_paths[item.stream_id] = []
        for frame in decoded.frames:
            frame_candidates = tuple(by_frame.get(frame.frame_id, ()))
            if not frame_candidates:
                continue
            rgb = grounding._frame_array(frame)
            if device == "cuda":
                torch_module.cuda.synchronize()
            frame_started = time.perf_counter()
            results = refiner.refine(
                rgb,
                tuple(candidate.bbox for candidate in frame_candidates),
                cleanup=FROZEN_CLEANUP,
            )
            if device == "cuda":
                torch_module.cuda.synchronize()
            elapsed = time.perf_counter() - frame_started
            runtime_rows.append(
                {
                    "stream_id": item.stream_id,
                    "frame_id": frame.frame_id,
                    "candidate_count": len(frame_candidates),
                    "elapsed_seconds": elapsed,
                }
            )
            frame_root = artifact_root / "masks" / item.stream_id / frame.frame_id
            for source_index, (candidate, result) in enumerate(
                zip(frame_candidates, results, strict=True)
            ):
                raw_path = cleaned_path = None
                if result.status == "valid":
                    base = _safe_artifact_name(candidate.candidate_id, source_index)
                    raw_path = frame_root / f"{base}_raw.png"
                    cleaned_path = frame_root / f"{base}_cleaned.png"
                    _save_mask(raw_path, result.raw_mask)
                    _save_mask(cleaned_path, result.cleaned_mask)
                records.append(
                    _result_record(
                        candidate,
                        result,
                        stream_id=item.stream_id,
                        raw_path=raw_path,
                        cleaned_path=cleaned_path,
                        artifact_root=artifact_root,
                    )
                )
            overlay_path = artifact_root / "prediction_overlays" / item.stream_id / f"{frame.frame_id}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            _prediction_overlay(rgb, frame_candidates, results).save(overlay_path, format="PNG")
            overlay_paths[item.stream_id].append(_relative(overlay_path, artifact_root))
        records_by_stream[item.stream_id] = records

    latencies = [row["elapsed_seconds"] for row in runtime_rows]
    manifest = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "status": "rgb_inference_completed_before_ground_truth",
        "scope": "development_only",
        "heldout_access": "none",
        "selected_candidates": {
            "path": selected_manifest_path.resolve(strict=True).as_posix(),
            "sha256": _sha256(selected_manifest_path.resolve(strict=True)),
        },
        "model": {
            "family": "sam2.1_hiera_tiny_bbox_prompted",
            "directory": model_root.as_posix(),
            "model_safetensors_sha256": _sha256(checkpoint),
            "model_safetensors_size_bytes": checkpoint.stat().st_size,
            "configuration_assets": _model_asset_provenance(model_root),
            "processor_class": (
                f"{type(refiner.processor).__module__}.{type(refiner.processor).__qualname__}"
            ),
            "model_class": f"{type(refiner.model).__module__}.{type(refiner.model).__qualname__}",
            "transformers_version": transformers.__version__,
            "torch_version": torch_module.__version__,
            "device": device,
            "dtype": "float32",
            "local_files_only": True,
            "compatibility_note": (
                "Local config declares Sam2VideoModel; the static bbox-prompt path "
                "loads the shared image model as Sam2Model and Transformers emits "
                "a reproducible compatibility warning."
            ),
        },
        "cleanup": {
            "min_component_pixels": FROZEN_CLEANUP.min_component_pixels,
            "min_component_area_ratio": FROZEN_CLEANUP.min_component_area_ratio,
            "connectivity": FROZEN_CLEANUP.connectivity,
            "component_removal_logic": "below_both_absolute_and_relative_thresholds",
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "processed_frame_count": len(runtime_rows),
            "candidate_count": sum(len(rows) for rows in records_by_stream.values()),
            "mean_seconds_per_processed_frame": None
            if not runtime_rows
            else sum(latencies) / len(latencies),
            "latency_seconds": _distribution(latencies),
            "peak_gpu_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated())
            if device == "cuda"
            else None,
            "peak_gpu_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved())
            if device == "cuda"
            else None,
            "frames": runtime_rows,
        },
        "records_by_stream": records_by_stream,
        "prediction_overlays": overlay_paths,
    }
    write_canonical_json_atomic(artifact_root / "inference_manifest.json", manifest)
    return manifest


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path.resolve(strict=True)) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _pixel_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(predicted, dtype=np.bool_)
    right = np.asarray(expected, dtype=np.bool_)
    if left.shape != right.shape:
        raise OcidEvaluationError("predicted and reviewed masks have different shapes")
    intersection = int(np.count_nonzero(left & right))
    predicted_pixels = int(np.count_nonzero(left))
    expected_pixels = int(np.count_nonzero(right))
    union = predicted_pixels + expected_pixels - intersection
    precision = 0.0 if predicted_pixels == 0 else intersection / predicted_pixels
    recall = 0.0 if expected_pixels == 0 else intersection / expected_pixels
    return {
        "intersection_pixels": intersection,
        "predicted_pixels": predicted_pixels,
        "expected_pixels": expected_pixels,
        "iou": 0.0 if union == 0 else intersection / union,
        "dice": 0.0
        if predicted_pixels + expected_pixels == 0
        else 2 * intersection / (predicted_pixels + expected_pixels),
        "precision": precision,
        "recall": recall,
    }


def _bbox_mask(bbox: BBox, size: ImageSize) -> np.ndarray:
    clipped = bbox.clip_to(size)
    output = np.zeros((size.height, size.width), dtype=np.bool_)
    if clipped is None:
        return output
    left = max(0, min(size.width, int(math.floor(clipped.left))))
    top = max(0, min(size.height, int(math.floor(clipped.top))))
    right = max(0, min(size.width, int(math.ceil(clipped.right))))
    bottom = max(0, min(size.height, int(math.ceil(clipped.bottom))))
    output[top:bottom, left:right] = True
    return output


def _same_bbox(left: BBox, right: BBox) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        for a, b in zip(
            (left.x, left.y, left.width, left.height),
            (right.x, right.y, right.width, right.height),
            strict=True,
        )
    )


def _has_forbidden_ocid_path_token(parts: Sequence[str]) -> bool:
    for part in parts:
        token = part.casefold()
        if "heldout" in token or "held-out" in token or "held_out" in token:
            return True
    return False


def _ocid_relative_parts(value: str, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise OcidEvaluationError(f"{label} must be a non-empty relative path")
    posix_value = PurePosixPath(value.replace("\\", "/"))
    windows_value = PureWindowsPath(value)
    if (
        posix_value.is_absolute()
        or windows_value.is_absolute()
        or bool(windows_value.drive)
        or bool(windows_value.root)
    ):
        raise OcidEvaluationError(f"{label} must not be absolute")
    parts = tuple(posix_value.parts)
    if not parts or any(part.rstrip(" .") in {"", ".", ".."} for part in parts):
        raise OcidEvaluationError(f"{label} must not contain traversal components")
    if _has_forbidden_ocid_path_token(parts):
        raise OcidEvaluationError(f"{label} contains a forbidden held-out path token")
    return parts


def _resolve_strict_path(path: Path) -> Path:
    """Small seam for deterministic symlink-containment tests."""

    return path.resolve(strict=True)


def _resolve_ocid_label_path(
    *,
    ocid_root: Path,
    source_sequence: str,
    source_filename: str,
) -> Path:
    sequence_parts = _ocid_relative_parts(
        source_sequence, label="OCID source sequence"
    )
    filename_parts = _ocid_relative_parts(
        source_filename, label="OCID source filename"
    )
    if len(filename_parts) != 1:
        raise OcidEvaluationError("OCID source filename must be a basename")
    try:
        resolved_root = _resolve_strict_path(Path(ocid_root))
    except (OSError, ValueError) as error:
        raise OcidEvaluationError(f"cannot resolve OCID root: {error}") from error
    if not resolved_root.is_dir():
        raise OcidEvaluationError("OCID root must be a directory")
    if _has_forbidden_ocid_path_token(resolved_root.parts):
        raise OcidEvaluationError("resolved OCID root contains a forbidden path token")

    supplied_path = resolved_root.joinpath(
        *sequence_parts,
        "label",
        filename_parts[0],
    )
    try:
        resolved_path = _resolve_strict_path(supplied_path)
    except (OSError, ValueError) as error:
        raise OcidEvaluationError(f"cannot resolve OCID label path: {error}") from error
    try:
        relative_resolved = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise OcidEvaluationError("resolved OCID label path escapes OCID root") from error
    if _has_forbidden_ocid_path_token(resolved_path.parts) or (
        _has_forbidden_ocid_path_token(relative_resolved.parts)
    ):
        raise OcidEvaluationError(
            "resolved OCID label path contains a forbidden path token"
        )
    if not resolved_path.is_file():
        raise OcidEvaluationError("resolved OCID label path must be a file")
    return resolved_path


def _reviewed_mask(
    *,
    stream_id: str,
    frame_id: str,
    instance_payload: Mapping[str, Any],
    source_sequence: str,
    source_filename: str,
    ocid_root: Path,
    decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_label = int(instance_payload["source_label"])
    label_path = _resolve_ocid_label_path(
        ocid_root=ocid_root,
        source_sequence=source_sequence,
        source_filename=source_filename,
    )
    with Image.open(label_path) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise OcidEvaluationError(f"invalid OCID label image: {label_path}")
    raw = np.asarray(labels == source_label, dtype=np.bool_)
    analysis = analyze_component_mask(raw)
    case_id = f"{stream_id}__{frame_id}__label_{source_label:03d}"
    decision = decisions.get(case_id)
    if decision is None:
        if analysis.review_required:
            raise OcidEvaluationError(f"review-required mask lacks decision: {case_id}")
        retained = analysis.automatic_retained_component_ids
    else:
        retained_value = decision.get("final_retained_component_ids")
        if not isinstance(retained_value, list) or not all(
            isinstance(item, str) for item in retained_value
        ):
            raise OcidEvaluationError(f"malformed retained components for {case_id}")
        recorded_digest = decision.get("analysis", {}).get("source_mask_sha256")
        if recorded_digest != analysis.source_mask_sha256:
            raise OcidEvaluationError(f"source mask digest mismatch for {case_id}")
        retained = tuple(retained_value)
    final = component_union_mask(analysis, retained)
    return final, {
        "case_id": case_id,
        "source_label": source_label,
        "source_label_path": label_path.as_posix(),
        "source_mask_sha256": analysis.source_mask_sha256,
        "reviewed_mask_sha256": binary_mask_sha256(final),
        "retained_component_ids": list(retained),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], profile: str) -> dict[str, Any]:
    summary = {
        name: _distribution([float(row[profile][name]) for row in rows])
        for name in ("iou", "dice", "precision", "recall")
    }
    summary["iou_pass_rate"] = {
        f"{threshold:.2f}": 0.0
        if not rows
        else sum(float(row[profile]["iou"]) >= threshold for row in rows) / len(rows)
        for threshold in (0.50, 0.60, 0.70, 0.80, 0.90)
    }
    return summary


def _correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int | None]:
    if len(left) != len(right):
        raise ValueError("correlation vectors must have equal length")
    if len(left) < 2:
        return {"count": len(left), "pearson": None, "spearman": None}
    from scipy.stats import spearmanr

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    pearson = None if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    spearman_value = spearmanr(x, y).statistic
    return {
        "count": len(left),
        "pearson": pearson,
        "spearman": None if not math.isfinite(float(spearman_value)) else float(spearman_value),
    }


def _mask_profile_selection(rows: Sequence[Mapping[str, Any]], removed_count: int) -> dict[str, Any]:
    raw_iou = [float(row["raw"]["iou"]) for row in rows]
    cleaned_iou = [float(row["cleaned"]["iou"]) for row in rows]
    raw_precision = [float(row["raw"]["precision"]) for row in rows]
    cleaned_precision = [float(row["cleaned"]["precision"]) for row in rows]
    mean_raw_iou = float(np.mean(raw_iou)) if raw_iou else 0.0
    mean_cleaned_iou = float(np.mean(cleaned_iou)) if cleaned_iou else 0.0
    mean_raw_precision = float(np.mean(raw_precision)) if raw_precision else 0.0
    mean_cleaned_precision = float(np.mean(cleaned_precision)) if cleaned_precision else 0.0
    worst_drop = max(
        (raw - cleaned for raw, cleaned in zip(raw_iou, cleaned_iou, strict=True)),
        default=0.0,
    )
    reasons: list[str] = []
    if removed_count <= 0:
        reasons.append("no_secondary_component_removed")
    if mean_raw_iou - mean_cleaned_iou > 0.001 + 1e-12:
        reasons.append("mean_iou_safety_failed")
    if worst_drop > 0.01 + 1e-12:
        reasons.append("individual_iou_safety_failed")
    if not (
        mean_cleaned_iou > mean_raw_iou + 1e-12
        or mean_cleaned_precision > mean_raw_precision + 1e-12
    ):
        reasons.append("no_mean_iou_or_precision_gain")
    return {
        "selected_profile": "m_conservative_components" if not reasons else "m_raw",
        "status": "selected_cleaned" if not reasons else "retained_raw",
        "ineligibility_reasons": reasons,
        "mean_raw_iou": mean_raw_iou,
        "mean_cleaned_iou": mean_cleaned_iou,
        "mean_raw_precision": mean_raw_precision,
        "mean_cleaned_precision": mean_cleaned_precision,
        "worst_individual_iou_drop": worst_drop,
        "removed_component_count": removed_count,
    }


def _aggregate_mask_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    inference_records: Sequence[Mapping[str, Any]],
    stream_to_scene: Mapping[str, str],
) -> dict[str, Any]:
    fallback_count = sum(record["status"] != "valid" for record in inference_records)
    removed_components = sum(
        int(record["quality"]["removed_component_count"])
        for record in inference_records
        if isinstance(record.get("quality"), Mapping)
    )
    removed_pixels = sum(
        int(record["quality"]["removed_foreground_pixels"])
        for record in inference_records
        if isinstance(record.get("quality"), Mapping)
    )
    by_stream: dict[str, Any] = {}
    for stream_id in sorted({str(row["stream_id"]) for row in rows}):
        selected = [row for row in rows if row["stream_id"] == stream_id]
        by_stream[stream_id] = {
            "matched_mask_count": len(selected),
            "raw": _metric_summary(selected, "raw"),
            "cleaned": _metric_summary(selected, "cleaned"),
            "bbox_rectangle": _metric_summary(selected, "bbox_rectangle"),
        }
    by_scene: dict[str, Any] = {}
    for scene_id in sorted(set(stream_to_scene.values())):
        selected = [row for row in rows if stream_to_scene[str(row["stream_id"])] == scene_id]
        by_scene[scene_id] = {
            "matched_mask_count": len(selected),
            "raw": _metric_summary(selected, "raw"),
            "cleaned": _metric_summary(selected, "cleaned"),
            "bbox_rectangle": _metric_summary(selected, "bbox_rectangle"),
        }
    quality_rows = [
        float(record["quality"]["predicted_iou"])
        for record in inference_records
        if isinstance(record.get("quality"), Mapping)
    ]
    raw_wins = sum(float(row["raw"]["iou"]) > float(row["bbox_rectangle"]["iou"]) + 1e-12 for row in rows)
    cleaned_wins = sum(float(row["cleaned"]["iou"]) > float(row["bbox_rectangle"]["iou"]) + 1e-12 for row in rows)
    matched_quality = [float(row["sam2_predicted_iou"]) for row in rows]
    return {
        "bbox_assignment_iou": FROZEN_BBOX_IOU,
        "candidate_count": len(inference_records),
        "valid_mask_count": len(inference_records) - fallback_count,
        "fallback_count": fallback_count,
        "fallback_rate": 0.0 if not inference_records else fallback_count / len(inference_records),
        "bbox_matched_valid_mask_count": len(rows),
        "sam2_predicted_iou": _distribution(quality_rows),
        "sam2_quality_calibration": {
            "versus_raw_mask_iou": _correlation(
                matched_quality, [float(row["raw"]["iou"]) for row in rows]
            ),
            "versus_cleaned_mask_iou": _correlation(
                matched_quality, [float(row["cleaned"]["iou"]) for row in rows]
            ),
        },
        "removed_component_count": removed_components,
        "removed_foreground_pixels": removed_pixels,
        "raw": _metric_summary(rows, "raw"),
        "cleaned": _metric_summary(rows, "cleaned"),
        "bbox_rectangle": _metric_summary(rows, "bbox_rectangle"),
        "raw_iou_better_than_bbox_count": raw_wins,
        "cleaned_iou_better_than_bbox_count": cleaned_wins,
        "raw_iou_gain_vs_bbox": _distribution(
            [float(row["raw"]["iou"]) - float(row["bbox_rectangle"]["iou"]) for row in rows]
        ),
        "cleaned_iou_gain_vs_bbox": _distribution(
            [float(row["cleaned"]["iou"]) - float(row["bbox_rectangle"]["iou"]) for row in rows]
        ),
        "per_stream": by_stream,
        "per_scene_group": by_scene,
        "profile_selection": _mask_profile_selection(rows, removed_components),
    }


def evaluate_masks(
    *,
    inventory: Sequence[DevelopmentStream],
    selected: Mapping[str, Sequence[CandidateRecord]],
    inference_manifest: Mapping[str, Any],
    reviewed_root: Path,
    ocid_root: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Open reviewed masks only after the persisted RGB-inference boundary."""

    if inference_manifest.get("status") != "rgb_inference_completed_before_ground_truth":
        raise OcidEvaluationError("RGB inference boundary is not complete")
    benchmark_manifest_path = reviewed_root / "benchmark_manifest.json"
    benchmark_manifest = _json_object(benchmark_manifest_path, "reviewed benchmark manifest")
    if benchmark_manifest.get("heldout_lock", {}).get("predictions_unlocked") is not False:
        raise OcidEvaluationError("held-out benchmark lock must remain closed")
    stream_metadata = {
        str(row["stream_id"]): row for row in benchmark_manifest.get("streams", [])
    }
    decisions_path = reviewed_root / str(
        benchmark_manifest["component_review"]["decisions_artifact"]
    )
    if _sha256(decisions_path) != benchmark_manifest["component_review"]["decisions_sha256"]:
        raise OcidEvaluationError("component-review decision digest mismatch")
    decisions_payload = _json_object(decisions_path, "component review decisions")
    decisions = {
        str(row["case_id"]): row for row in decisions_payload.get("decisions", [])
    }
    inference_rows = [
        row
        for stream_rows in inference_manifest["records_by_stream"].values()
        for row in stream_rows
    ]
    inference_by_id = {
        (str(row["stream_id"]), str(row["candidate_id"])): row
        for row in inference_rows
    }
    if len(inference_by_id) != len(inference_rows):
        raise OcidEvaluationError("inference manifest candidate IDs are not unique")

    evaluation_rows: list[dict[str, Any]] = []
    reviewed_masks: dict[str, np.ndarray] = {}
    bbox_counts = {"tp": 0, "fp": 0, "fn": 0}
    stream_to_scene = {item.stream_id: item.scene_group_id for item in inventory}
    for item in inventory:
        annotation = load_annotation(
            item.annotation_path,
            manifest_path=item.stream_directory / "manifest.json",
        )
        raw_instances = {
            str(row["instance_id"]): row
            for row in annotation.raw["expected_element_instances"]
        }
        manifest = _json_object(item.stream_directory / "manifest.json", "analysis manifest")
        source_filename = {
            str(row["frame_id"]): str(row["metadata"]["ocid_source_filename"])
            for row in manifest["frames"]
        }
        predicted = tuple(
            PredictedCandidate(
                candidate_id=row.candidate_id,
                frame_id=row.frame_id,
                frame_index=row.frame_index,
                bbox=row.bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
            for row in selected[item.stream_id]
        )
        predicted_by_frame: dict[str, list[PredictedCandidate]] = {}
        for row in predicted:
            predicted_by_frame.setdefault(row.frame_id, []).append(row)
        source_sequence = str(stream_metadata[item.stream_id]["source_sequence"])
        for frame_id in annotation.frame_ids:
            frame_predictions = tuple(predicted_by_frame.get(frame_id, ()))
            frame_expected = annotation.instances_by_frame.get(frame_id, ())
            matched_pred, matched_gt, assignments = _assign_frame_candidates(
                frame_predictions,
                frame_expected,
                FROZEN_BBOX_IOU,
            )
            bbox_counts["tp"] += len(assignments)
            bbox_counts["fp"] += len(frame_predictions) - len(matched_pred)
            bbox_counts["fn"] += len(frame_expected) - len(matched_gt)
            expected_by_id = {row.instance_id: row for row in frame_expected}
            for assignment in assignments:
                inference = inference_by_id[(item.stream_id, assignment.candidate_id)]
                if inference["status"] != "valid":
                    continue
                expected_instance = expected_by_id[assignment.instance_id]
                instance_payload = raw_instances[assignment.instance_id]
                expected_mask, provenance = _reviewed_mask(
                    stream_id=item.stream_id,
                    frame_id=frame_id,
                    instance_payload=instance_payload,
                    source_sequence=source_sequence,
                    source_filename=source_filename[frame_id],
                    ocid_root=ocid_root,
                    decisions=decisions,
                )
                expected_bbox = mask_bbox_from_full_frame(expected_mask)
                if expected_bbox is None or not _same_bbox(expected_bbox, expected_instance.bbox):
                    raise OcidEvaluationError(
                        f"reviewed mask bbox differs from the stored annotation: {assignment.instance_id}"
                    )
                raw_path = artifact_root / str(inference["raw_mask"]["path"])
                cleaned_path = artifact_root / str(inference["cleaned_mask"]["path"])
                raw_mask = _load_mask(raw_path)
                cleaned_mask = _load_mask(cleaned_path)
                if binary_mask_sha256(raw_mask) != inference["raw_mask"]["binary_mask_sha256"]:
                    raise OcidEvaluationError("raw predicted-mask digest mismatch")
                if binary_mask_sha256(cleaned_mask) != inference["cleaned_mask"]["binary_mask_sha256"]:
                    raise OcidEvaluationError("cleaned predicted-mask digest mismatch")
                candidate = next(
                    row for row in selected[item.stream_id] if row.candidate_id == assignment.candidate_id
                )
                size = ImageSize(width=expected_mask.shape[1], height=expected_mask.shape[0])
                reviewed_masks[f"{item.stream_id}::{assignment.candidate_id}"] = expected_mask
                evaluation_rows.append(
                    {
                        "stream_id": item.stream_id,
                        "scene_group_id": item.scene_group_id,
                        "frame_id": frame_id,
                        "candidate_id": assignment.candidate_id,
                        "instance_id": assignment.instance_id,
                        "bbox_iou": assignment.iou,
                        "sam2_predicted_iou": inference["quality"]["predicted_iou"],
                        "source_bbox": _bbox_payload(candidate.bbox),
                        "raw": _pixel_metrics(raw_mask, expected_mask),
                        "cleaned": _pixel_metrics(cleaned_mask, expected_mask),
                        "bbox_rectangle": _pixel_metrics(
                            _bbox_mask(candidate.bbox, size), expected_mask
                        ),
                        "reviewed_mask": provenance,
                    }
                )

    bbox_f1_denominator = 2 * bbox_counts["tp"] + bbox_counts["fp"] + bbox_counts["fn"]
    summary = _aggregate_mask_evaluation(
        evaluation_rows,
        inference_records=inference_rows,
        stream_to_scene=stream_to_scene,
    )
    summary["bbox_detection_unchanged"] = {
        **bbox_counts,
        "precision": bbox_counts["tp"] / (bbox_counts["tp"] + bbox_counts["fp"]),
        "recall": bbox_counts["tp"] / (bbox_counts["tp"] + bbox_counts["fn"]),
        "f1": 0.0 if bbox_f1_denominator == 0 else 2 * bbox_counts["tp"] / bbox_f1_denominator,
    }
    summary["reviewed_ground_truth"] = {
        "benchmark_manifest": {
            "path": benchmark_manifest_path.resolve(strict=True).as_posix(),
            "sha256": _sha256(benchmark_manifest_path.resolve(strict=True)),
        },
        "component_decisions": {
            "path": decisions_path.resolve(strict=True).as_posix(),
            "sha256": _sha256(decisions_path.resolve(strict=True)),
        },
        "ocid_root": ocid_root.resolve(strict=True).as_posix(),
    }
    return summary, evaluation_rows, reviewed_masks


def _evaluation_panel(
    *,
    rgb_path: Path,
    predicted_mask_path: Path,
    expected_mask: np.ndarray,
    row: Mapping[str, Any],
    width: int = 320,
) -> Image.Image:
    with Image.open(rgb_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    predicted = _load_mask(predicted_mask_path)
    canvas = rgb.copy()
    cyan = np.asarray((0, 220, 255), dtype=np.float32)
    canvas[predicted] = canvas[predicted] * 0.60 + cyan * 0.40
    canvas[_mask_boundary(predicted)] = cyan
    canvas[_mask_boundary(expected_mask)] = np.asarray((255, 0, 220), dtype=np.float32)
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    bbox = row["source_bbox"]
    draw.rectangle(
        (
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["x"]) + float(bbox["width"]) - 1,
            float(bbox["y"]) + float(bbox["height"]) - 1,
        ),
        outline=(255, 210, 0),
        width=2,
    )
    scale = width / image.width
    image = image.resize((width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
    header = Image.new("RGB", (image.width, 42), (18, 18, 18))
    label = (
        f"{row['stream_id']} | {row['frame_id']}\n"
        f"mask IoU={row['cleaned']['iou']:.3f}  bbox IoU-mask={row['bbox_rectangle']['iou']:.3f}"
    )
    ImageDraw.Draw(header).text((4, 4), label, fill=(240, 240, 240))
    panel = Image.new("RGB", (image.width, header.height + image.height), (18, 18, 18))
    panel.paste(header, (0, 0))
    panel.paste(image, (0, header.height))
    return panel


def _contact_sheet(panels: Sequence[Image.Image], path: Path, *, columns: int = 3) -> None:
    if not panels:
        return
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (8, 8, 8))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * cell_width, (index // columns) * cell_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def render_evaluation_summaries(
    *,
    rows: Sequence[Mapping[str, Any]],
    reviewed_masks: Mapping[str, np.ndarray],
    inference_manifest: Mapping[str, Any],
    inventory: Sequence[DevelopmentStream],
    artifact_root: Path,
) -> dict[str, str]:
    inference_by_id = {
        (str(record["stream_id"]), str(record["candidate_id"])): record
        for values in inference_manifest["records_by_stream"].values()
        for record in values
    }
    frame_path: dict[tuple[str, str], Path] = {}
    for item in inventory:
        manifest = _json_object(item.stream_directory / "manifest.json", "analysis manifest")
        for frame in manifest["frames"]:
            frame_path[(item.stream_id, str(frame["frame_id"]))] = (
                item.stream_directory / str(frame["image_path"])
            )

    def panel(row: Mapping[str, Any]) -> Image.Image:
        record = inference_by_id[(str(row["stream_id"]), str(row["candidate_id"]))]
        return _evaluation_panel(
            rgb_path=frame_path[(str(row["stream_id"]), str(row["frame_id"]))],
            predicted_mask_path=artifact_root / str(record["cleaned_mask"]["path"]),
            expected_mask=reviewed_masks[
                f"{row['stream_id']}::{row['candidate_id']}"
            ],
            row=row,
        )

    worst = sorted(rows, key=lambda row: (float(row["cleaned"]["iou"]), str(row["candidate_id"])))[:12]
    gains = sorted(
        rows,
        key=lambda row: (
            -(float(row["cleaned"]["iou"]) - float(row["bbox_rectangle"]["iou"])),
            str(row["candidate_id"]),
        ),
    )[:12]
    per_stream: list[Mapping[str, Any]] = []
    for stream_id in sorted({str(row["stream_id"]) for row in rows}):
        stream_rows = sorted(
            (row for row in rows if row["stream_id"] == stream_id),
            key=lambda row: (float(row["cleaned"]["iou"]), str(row["candidate_id"])),
        )
        if stream_rows:
            per_stream.append(stream_rows[len(stream_rows) // 2])
    paths = {
        "worst_12": "evaluation_summaries/mask_quality_worst_12.png",
        "best_gain_12": "evaluation_summaries/mask_quality_best_gain_12.png",
        "one_per_stream": "evaluation_summaries/mask_quality_one_per_stream.png",
    }
    _contact_sheet([panel(row) for row in worst], artifact_root / paths["worst_12"])
    _contact_sheet([panel(row) for row in gains], artifact_root / paths["best_gain_12"])
    _contact_sheet([panel(row) for row in per_stream], artifact_root / paths["one_per_stream"])
    return paths


def run_benchmark(
    *,
    selected_candidates_path: Path,
    model_directory: Path,
    expected_model_sha256: str,
    benchmark_spec_path: Path,
    reviewed_root: Path,
    ocid_root: Path,
    device: str,
    artifact_root: Path,
    output_path: Path,
    reuse_inference_manifest_path: Path | None = None,
) -> dict[str, Any]:
    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    selected, selected_payload = load_selected_candidates(selected_candidates_path, inventory)
    destination = artifact_root.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    if reuse_inference_manifest_path is None:
        inference = run_rgb_inference(
            inventory=inventory,
            selected=selected,
            selected_manifest_path=selected_candidates_path,
            model_directory=model_directory,
            expected_model_sha256=expected_model_sha256,
            device=device,
            artifact_root=destination,
        )
    else:
        inference = _json_object(reuse_inference_manifest_path, "RGB inference manifest")
        if inference.get("schema_version") != INFERENCE_SCHEMA_VERSION:
            raise OcidEvaluationError("unsupported reusable inference manifest")
        selected_provenance = inference.get("selected_candidates")
        if (
            not isinstance(selected_provenance, Mapping)
            or selected_provenance.get("sha256") != _sha256(selected_candidates_path.resolve(strict=True))
            or inference.get("heldout_access") != "none"
        ):
            raise OcidEvaluationError("reusable inference manifest does not match selected candidates")
        model_provenance = inference.get("model")
        if (
            not isinstance(model_provenance, Mapping)
            or model_provenance.get("model_safetensors_sha256") != expected_model_sha256.casefold()
        ):
            raise OcidEvaluationError("reusable inference manifest model digest mismatch")
    presentation_overlays = render_presentation_overlays(
        inventory=inventory,
        inference_manifest=inference,
        artifact_root=destination,
    )
    # Ground truth is first opened below this persisted inference boundary.
    evaluation, rows, reviewed_masks = evaluate_masks(
        inventory=inventory,
        selected=selected,
        inference_manifest=inference,
        reviewed_root=reviewed_root.resolve(strict=True),
        ocid_root=ocid_root.resolve(strict=True),
        artifact_root=destination,
    )
    summaries = render_evaluation_summaries(
        rows=rows,
        reviewed_masks=reviewed_masks,
        inference_manifest=inference,
        inventory=inventory,
        artifact_root=destination,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "development_only_mask_enrichment_ocid_evaluation_1",
        "heldout_access": "none",
        "ground_truth_boundary": "persisted_rgb_inference_manifest_before_reviewed_mask_evaluation",
        "selected_profile_id": selected_payload["selected_profile_id"],
        "inference_manifest": {
            "path": (destination / "inference_manifest.json").as_posix(),
            "sha256": _sha256(destination / "inference_manifest.json"),
        },
        "evaluation": evaluation,
        "matched_observations": rows,
        "evaluation_summaries": summaries,
        "presentation_overlays": presentation_overlays,
        "limitations": [
            "Mask quality is measured only on bbox-matched development true positives.",
            "False-positive candidates have no target mask and remain bbox detection errors.",
            "SAM2 predicted_iou is diagnostic and is not used as a rejection threshold.",
            "DINOv2 predicted-mask embeddings are deferred to OCID representation evaluation.",
            "Prepared OCID masks retain the selected instance-component boundary and do not redraw object contours pixel by pixel.",
        ],
    }
    write_canonical_json_atomic(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-inference-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            selected_candidates_path=args.selected_candidates,
            model_directory=args.model_directory,
            expected_model_sha256=args.expected_model_sha256,
            benchmark_spec_path=args.benchmark_spec,
            reviewed_root=args.reviewed_root,
            ocid_root=args.ocid_root,
            device=args.device,
            artifact_root=args.artifact_root,
            output_path=args.output,
            reuse_inference_manifest_path=args.reuse_inference_manifest,
        )
    except (OcidEvaluationError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
