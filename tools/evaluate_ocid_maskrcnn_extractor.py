"""Benchmark official Torchvision Mask R-CNN as an OCID candidate extractor.

This is an isolated evaluation tool.  OCID annotations are read only after
inference and are never provided to the model or to the normal analyze path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stream_analysis import ManifestLoadRequest, ProducerProvenance, load_decoded_stream
from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation import (
    PredictedCandidate,
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
    load_annotation,
)
try:  # Support both ``python -m tools...`` and direct script execution.
    from tools.ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke test.
    from ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )


SCHEMA_VERSION = "ocid-learned-extractor-gate-0.3"
GEOMETRY_VARIANTS = ("model_bbox", "mask_tight_bbox")
DEFAULT_SCORE_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)
DEFAULT_MATCH_IOU_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True, slots=True)
class RawPrediction:
    frame_id: str
    frame_index: int
    prediction_index: int
    score: float
    label: int
    model_bbox: BBox
    mask_tight_bbox: BBox | None


def profile_id(
    *,
    geometry_variant: str,
    score_threshold: float,
    nms_iou_threshold: float | None,
) -> str:
    """Return a stable identifier for one pre-registered candidate profile."""
    if geometry_variant not in GEOMETRY_VARIANTS:
        raise ValueError("Unsupported geometry_variant.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1].")
    if nms_iou_threshold is not None and not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in [0, 1] or None.")
    nms_token = "none" if nms_iou_threshold is None else f"{nms_iou_threshold:.2f}"
    return f"geometry_{geometry_variant}__score_{score_threshold:.2f}__nms_{nms_token}"


def profile_definition(
    *,
    geometry_variant: str,
    score_threshold: float,
    nms_iou_threshold: float | None,
) -> dict[str, Any]:
    """Structured, self-describing counterpart of :func:`profile_id`."""
    identifier = profile_id(
        geometry_variant=geometry_variant,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    return {
        "profile_id": identifier,
        "geometry_variant": geometry_variant,
        "score_threshold": score_threshold,
        "class_agnostic_nms_iou": nms_iou_threshold,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="1.0",
        config_digest="sha256:ocid-maskrcnn-extractor-gate",
    )


def _load_stream(stream_directory: Path):
    return load_decoded_stream(
        ManifestLoadRequest(
            stream_root=stream_directory.resolve(strict=True),
            producer=_producer(),
        )
    )


def _integral_box(values: np.ndarray, image_size: ImageSize) -> BBox | None:
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    left = max(0, min(image_size.width, int(np.floor(float(values[0])))))
    top = max(0, min(image_size.height, int(np.floor(float(values[1])))))
    right = max(0, min(image_size.width, int(np.ceil(float(values[2])))))
    bottom = max(0, min(image_size.height, int(np.ceil(float(values[3])))))
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right - left, bottom - top)


def _mask_box(mask: np.ndarray, image_size: ImageSize) -> BBox | None:
    foreground = np.asarray(mask) >= 0.5
    if foreground.shape != (image_size.height, image_size.width) or not np.any(foreground):
        return None
    rows, columns = np.nonzero(foreground)
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return BBox(left, top, right - left, bottom - top)


def predictions_at_threshold(
    raw: tuple[RawPrediction, ...],
    *,
    threshold: float,
    geometry_variant: str,
    class_agnostic_nms_iou: float | None = None,
) -> tuple[PredictedCandidate, ...]:
    if geometry_variant not in GEOMETRY_VARIANTS:
        raise ValueError("Unsupported geometry_variant.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if class_agnostic_nms_iou is not None and not 0.0 <= class_agnostic_nms_iou <= 1.0:
        raise ValueError("class_agnostic_nms_iou must be in [0, 1] or None.")
    selected: list[tuple[RawPrediction, BBox]] = []
    for item in raw:
        if item.score < threshold:
            continue
        bbox = item.model_bbox if geometry_variant == "model_bbox" else item.mask_tight_bbox
        if bbox is not None:
            selected.append((item, bbox))
    if class_agnostic_nms_iou is not None:
        selected = _class_agnostic_nms(selected, class_agnostic_nms_iou)
    candidates: list[PredictedCandidate] = []
    for item, bbox in selected:
        candidates.append(
            PredictedCandidate(
                candidate_id=f"maskrcnn:{item.frame_id}:{item.prediction_index:03d}",
                frame_id=item.frame_id,
                frame_index=item.frame_index,
                bbox=bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
        )
    return tuple(candidates)


def evaluate_profile_grid(
    annotation: Any,
    raw: tuple[RawPrediction, ...],
    *,
    thresholds: tuple[float, ...],
    match_iou_thresholds: tuple[float, ...],
    nms_iou_thresholds: tuple[float | None, ...] = (None,),
) -> dict[str, dict[str, Any]]:
    """Evaluate the IoU grid for fixed candidate profiles from one raw inference."""
    results: dict[str, dict[str, Any]] = {}
    for geometry_variant in GEOMETRY_VARIANTS:
        for threshold in thresholds:
            for nms_iou_threshold in nms_iou_thresholds:
                predictions = predictions_at_threshold(
                    raw,
                    threshold=threshold,
                    geometry_variant=geometry_variant,
                    class_agnostic_nms_iou=nms_iou_threshold,
                )
                definition = profile_definition(
                    geometry_variant=geometry_variant,
                    score_threshold=threshold,
                    nms_iou_threshold=nms_iou_threshold,
                )
                results[definition["profile_id"]] = {
                    **definition,
                    "metrics_by_iou": {
                        f"{match_iou_threshold:.2f}": evaluate_candidate_predictions(
                            annotation,
                            predictions,
                            iou_threshold=match_iou_threshold,
                        )
                        for match_iou_threshold in match_iou_thresholds
                    },
                }
    return results


def _class_agnostic_nms(
    selected: list[tuple[RawPrediction, BBox]],
    iou_threshold: float,
) -> list[tuple[RawPrediction, BBox]]:
    by_frame: dict[str, list[tuple[RawPrediction, BBox]]] = {}
    for item in selected:
        by_frame.setdefault(item[0].frame_id, []).append(item)
    kept: list[tuple[RawPrediction, BBox]] = []
    for frame_id in sorted(by_frame):
        frame_kept: list[tuple[RawPrediction, BBox]] = []
        ordered = sorted(
            by_frame[frame_id],
            key=lambda pair: (-pair[0].score, pair[0].prediction_index),
        )
        for candidate in ordered:
            if all(_bbox_iou(candidate[1], existing[1]) <= iou_threshold for existing in frame_kept):
                frame_kept.append(candidate)
        kept.extend(sorted(frame_kept, key=lambda pair: pair[0].prediction_index))
    return kept


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection_left = max(left.x, right.x)
    intersection_top = max(left.y, right.y)
    intersection_right = min(left.x + left.width, right.x + right.width)
    intersection_bottom = min(left.y + left.height, right.y + right.height)
    width = max(0.0, intersection_right - intersection_left)
    height = max(0.0, intersection_bottom - intersection_top)
    intersection = width * height
    union = left.area + right.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _frame_tensor(frame: Any, torch_module: Any):
    array = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    ).copy()
    return torch_module.from_numpy(array).permute(2, 0, 1).to(dtype=torch_module.float32) / 255.0


def _load_model(checkpoint: Path, torch_module: Any, device: str):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        box_score_thresh=0.0,
        box_detections_per_img=100,
    )
    state = torch_module.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _process_rss_bytes() -> int | None:
    """Return current process RSS when psutil is installed, otherwise None."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _latency_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _infer_stream(decoded: Any, model: Any, torch_module: Any, device: str) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    raw: list[RawPrediction] = []
    frame_rows: list[dict[str, Any]] = []
    peak_rss = _process_rss_bytes()
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    started_stream = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        tensor = _frame_tensor(frame, torch_module).to(device)
        started = time.perf_counter()
        with torch_module.inference_mode():
            output = model([tensor])[0]
        if device == "cuda":
            torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - started
        boxes = output["boxes"].detach().to("cpu").numpy()
        scores = output["scores"].detach().to("cpu").numpy()
        labels = output["labels"].detach().to("cpu").numpy()
        masks = output["masks"].detach().to("cpu").numpy()[:, 0]
        accepted = 0
        for index, (box_values, score, label, mask) in enumerate(
            zip(boxes, scores, labels, masks, strict=True)
        ):
            model_bbox = _integral_box(box_values, frame.image_size)
            if model_bbox is None:
                continue
            raw.append(
                RawPrediction(
                    frame_id=frame.frame_id,
                    frame_index=frame_index,
                    prediction_index=index,
                    score=float(score),
                    label=int(label),
                    model_bbox=model_bbox,
                    mask_tight_bbox=_mask_box(mask, frame.image_size),
                )
            )
            accepted += 1
        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "frame_index": frame_index,
                "raw_prediction_count": accepted,
                "inference_seconds": elapsed,
            }
        )
        current_rss = _process_rss_bytes()
        if current_rss is not None:
            peak_rss = max(peak_rss or current_rss, current_rss)
        del tensor, output
    elapsed_stream = time.perf_counter() - started_stream
    frame_latencies = [row["inference_seconds"] for row in frame_rows]
    return tuple(raw), {
        "frame_count": len(decoded.frames),
        "elapsed_seconds": elapsed_stream,
        "mean_seconds_per_frame": elapsed_stream / len(decoded.frames),
        "inference_latency_seconds": {
            "scope": "model_forward_only_excludes_preprocessing_and_output_transfer",
            "p50": _latency_percentile(frame_latencies, 50),
            "p95": _latency_percentile(frame_latencies, 95),
        },
        "process_peak_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "frames": frame_rows,
    }


def _bbox_dict(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_benchmark(
    *,
    stream_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    thresholds: tuple[float, ...],
    device: str,
    output_path: Path,
    match_iou_thresholds: tuple[float, ...] = DEFAULT_MATCH_IOU_THRESHOLDS,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> dict[str, Any]:
    # This is deliberately the first substantive operation: it rejects
    # held-out, subset and lookalike inputs before checkpoint hashing, model
    # imports or any caller-supplied stream/annotation access.
    inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    canonical_spec = Path(benchmark_spec_path).resolve(strict=True)
    canonical_root = Path(reviewed_root).resolve(strict=True)
    checkpoint = checkpoint_path.resolve(strict=True)
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_checkpoint_sha256.casefold():
        raise ValueError("Mask R-CNN checkpoint SHA-256 mismatch.")

    import torch
    import torchvision

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = _load_model(checkpoint, torch, device)
    stream_payloads: dict[str, Any] = {}
    metrics_by_profile: dict[str, dict[str, list[dict[str, Any]]]] = {}
    profile_definitions: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()
    for development_stream in inventory:
        stream_directory = development_stream.stream_directory
        annotation_path = development_stream.annotation_path
        decoded = _load_stream(stream_directory)
        raw, runtime = _infer_stream(decoded, model, torch, device)
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        if annotation.stream_id != decoded.stream.stream_id:
            raise ValueError("Annotation and stream_id mismatch.")
        evaluations = evaluate_profile_grid(
            annotation,
            raw,
            thresholds=thresholds,
            match_iou_thresholds=match_iou_thresholds,
        )
        for identifier, result in evaluations.items():
            profile_definitions.setdefault(identifier, {
                key: value for key, value in result.items() if key != "metrics_by_iou"
            })
            for iou_key, metrics in result["metrics_by_iou"].items():
                metrics_by_profile.setdefault(identifier, {}).setdefault(iou_key, []).append(metrics)
        stream_payloads[decoded.stream.stream_id] = {
            "runtime": runtime,
            "evaluations": evaluations,
            "raw_predictions": [
                {
                    "frame_id": item.frame_id,
                    "frame_index": item.frame_index,
                    "prediction_index": item.prediction_index,
                    "score": item.score,
                    "label": item.label,
                    "model_bbox": _bbox_dict(item.model_bbox),
                    "mask_tight_bbox": _bbox_dict(item.mask_tight_bbox),
                }
                for item in raw
            ],
        }

    pooled_profiles = {
        identifier: {
            **profile_definitions[identifier],
            "metrics_by_iou": {
                iou_key: aggregate_candidate_metrics(tuple(values))
                for iou_key, values in sorted(metrics_by_iou.items())
            },
        }
        for identifier, metrics_by_iou in sorted(metrics_by_profile.items())
    }
    # Expose the default-IoU aggregates through the compact compatibility view;
    # the complete IoU grid remains available in ``pooled_profiles``.
    legacy_aggregate = {
        geometry_variant: {
            f"{threshold:.4f}": pooled_profiles[
                profile_id(
                    geometry_variant=geometry_variant,
                    score_threshold=threshold,
                    nms_iou_threshold=None,
                )
            ]["metrics_by_iou"]["0.50"]
            for threshold in thresholds
        }
        for geometry_variant in GEOMETRY_VARIANTS
        if 0.50 in match_iou_thresholds
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "evaluation_only_candidate_extraction_gate",
        "ground_truth_boundary": "Annotations were used only after model inference.",
        "development_contract": {
            "benchmark_spec_path": canonical_spec.as_posix(),
            "benchmark_spec_sha256": _sha256(canonical_spec),
            "reviewed_root": canonical_root.as_posix(),
            "stream_ids": [stream.stream_id for stream in inventory],
        },
        "model": {
            "family": "torchvision_maskrcnn",
            "model_name": "maskrcnn_resnet50_fpn_v2",
            "weights": "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
            "semantic_classes": "COCO_closed_vocabulary_control",
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": actual_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": device,
            "dtype": "float32",
        },
        "thresholds": thresholds,
        "match_iou_thresholds": match_iou_thresholds,
        "class_agnostic_nms_iou": None,
        "geometry_variants": GEOMETRY_VARIANTS,
        "profile_definitions": [profile_definitions[key] for key in sorted(profile_definitions)],
        "streams": stream_payloads,
        "pooled_profiles": pooled_profiles,
        "aggregate": legacy_aggregate,
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    _atomic_write(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-directory", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--match-iou", type=float, action="append")
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = tuple(args.threshold or DEFAULT_SCORE_THRESHOLDS)
    payload = run_benchmark(
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        thresholds=thresholds,
        match_iou_thresholds=tuple(args.match_iou or DEFAULT_MATCH_IOU_THRESHOLDS),
        device=args.device,
        output_path=args.output,
        benchmark_spec_path=args.benchmark_spec,
        reviewed_root=args.reviewed_root,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.resolve(strict=False)),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
