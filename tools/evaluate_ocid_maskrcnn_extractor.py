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


SCHEMA_VERSION = "ocid-learned-extractor-gate-0.1"
GEOMETRY_VARIANTS = ("model_bbox", "mask_tight_bbox")


@dataclass(frozen=True, slots=True)
class RawPrediction:
    frame_id: str
    prediction_index: int
    score: float
    label: int
    model_bbox: BBox
    mask_tight_bbox: BBox | None


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
                frame_index=0,
                bbox=bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
        )
    return tuple(candidates)


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


def _infer_stream(decoded: Any, model: Any, torch_module: Any, device: str) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    raw: list[RawPrediction] = []
    frame_rows: list[dict[str, Any]] = []
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    started_stream = time.perf_counter()
    for frame in decoded.frames:
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
                "raw_prediction_count": accepted,
                "inference_seconds": elapsed,
            }
        )
        del tensor, output
    elapsed_stream = time.perf_counter() - started_stream
    return tuple(raw), {
        "frame_count": len(decoded.frames),
        "elapsed_seconds": elapsed_stream,
        "mean_seconds_per_frame": elapsed_stream / len(decoded.frames),
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
) -> dict[str, Any]:
    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise ValueError("Supply one annotation for every stream.")
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
    metrics_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    total_started = time.perf_counter()
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        decoded = _load_stream(stream_directory)
        raw, runtime = _infer_stream(decoded, model, torch, device)
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        if annotation.stream_id != decoded.stream.stream_id:
            raise ValueError("Annotation and stream_id mismatch.")
        evaluations: dict[str, Any] = {}
        for geometry_variant in GEOMETRY_VARIANTS:
            geometry_results: dict[str, Any] = {}
            for threshold in thresholds:
                key = f"{threshold:.4f}"
                metrics = evaluate_candidate_predictions(
                    annotation,
                    predictions_at_threshold(
                        raw,
                        threshold=threshold,
                        geometry_variant=geometry_variant,
                    ),
                )
                geometry_results[key] = metrics
                metrics_by_key.setdefault((geometry_variant, key), []).append(metrics)
            evaluations[geometry_variant] = geometry_results
        stream_payloads[decoded.stream.stream_id] = {
            "runtime": runtime,
            "evaluations": evaluations,
            "raw_predictions": [
                {
                    "frame_id": item.frame_id,
                    "prediction_index": item.prediction_index,
                    "score": item.score,
                    "label": item.label,
                    "model_bbox": _bbox_dict(item.model_bbox),
                    "mask_tight_bbox": _bbox_dict(item.mask_tight_bbox),
                }
                for item in raw
            ],
        }

    aggregate: dict[str, Any] = {}
    for geometry_variant in GEOMETRY_VARIANTS:
        aggregate[geometry_variant] = {
            f"{threshold:.4f}": aggregate_candidate_metrics(
                tuple(metrics_by_key[(geometry_variant, f"{threshold:.4f}")])
            )
            for threshold in thresholds
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "evaluation_only_candidate_extraction_gate",
        "ground_truth_boundary": "Annotations were used only after model inference.",
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
        "geometry_variants": GEOMETRY_VARIANTS,
        "streams": stream_payloads,
        "aggregate": aggregate,
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = tuple(args.threshold or (0.05, 0.10, 0.25, 0.50, 0.75))
    payload = run_benchmark(
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        thresholds=thresholds,
        device=args.device,
        output_path=args.output,
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
