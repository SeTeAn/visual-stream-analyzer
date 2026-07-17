"""Benchmark official MobileSAM as an OCID candidate extractor.

This is an isolated evaluation tool. OCID annotations are loaded only after
inference and are never provided to the model or to the normal analyze path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class FilterProfile:
    profile_id: str
    predicted_iou_threshold: float
    stability_threshold: float
    min_area_ratio: float
    max_area_ratio: float = 0.50

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty.")
        for field_name in ("predicted_iou_threshold", "stability_threshold"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1].")
        if not 0.0 <= self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("area ratios must satisfy 0 <= min < max <= 1.")


@dataclass(frozen=True, slots=True)
class RawPrediction:
    frame_id: str
    frame_index: int
    prediction_index: int
    bbox: BBox
    area: int
    image_area: int
    predicted_iou: float
    stability_score: float


DEFAULT_PROFILES = tuple(
    FilterProfile(
        profile_id=f"{quality_id}_area_{area_id}",
        predicted_iou_threshold=predicted_iou,
        stability_threshold=stability,
        min_area_ratio=min_area,
    )
    for quality_id, predicted_iou, stability in (
        ("permissive", 0.75, 0.85),
        ("balanced", 0.85, 0.90),
        ("official", 0.88, 0.95),
    )
    for area_id, min_area in (
        ("0001", 0.001),
        ("0003", 0.003),
        ("0010", 0.010),
    )
)


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
        config_digest="sha256:ocid-mobilesam-extractor-gate",
    )


def _load_stream(stream_directory: Path):
    return load_decoded_stream(
        ManifestLoadRequest(
            stream_root=stream_directory.resolve(strict=True),
            producer=_producer(),
        )
    )


def _mask_box(mask: np.ndarray, image_size: ImageSize) -> BBox | None:
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != (image_size.height, image_size.width) or not np.any(foreground):
        return None
    rows, columns = np.nonzero(foreground)
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return BBox(left, top, right - left, bottom - top)


def predictions_for_profile(
    raw: tuple[RawPrediction, ...],
    *,
    profile: FilterProfile,
) -> tuple[PredictedCandidate, ...]:
    candidates: list[PredictedCandidate] = []
    for item in raw:
        area_ratio = item.area / item.image_area
        if (
            item.predicted_iou < profile.predicted_iou_threshold
            or item.stability_score < profile.stability_threshold
            or area_ratio < profile.min_area_ratio
            or area_ratio > profile.max_area_ratio
        ):
            continue
        candidates.append(
            PredictedCandidate(
                candidate_id=f"mobilesam:{item.frame_id}:{item.prediction_index:03d}",
                frame_id=item.frame_id,
                frame_index=item.frame_index,
                bbox=item.bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
        )
    return tuple(candidates)


def _frame_array(frame: Any) -> np.ndarray:
    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    ).copy()


def _load_generator(
    checkpoint: Path,
    *,
    device: str,
    points_per_side: int,
    points_per_batch: int,
):
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.0,
        stability_score_thresh=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )


def _infer_stream(
    decoded: Any,
    generator: Any,
    torch_module: Any,
    device: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    raw: list[RawPrediction] = []
    frame_rows: list[dict[str, Any]] = []
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    started_stream = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        image = _frame_array(frame)
        started = time.perf_counter()
        masks = generator.generate(image)
        if device == "cuda":
            torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - started
        accepted = 0
        image_area = frame.image_size.width * frame.image_size.height
        for prediction_index, mask_record in enumerate(masks):
            bbox = _mask_box(mask_record["segmentation"], frame.image_size)
            if bbox is None:
                continue
            raw.append(
                RawPrediction(
                    frame_id=frame.frame_id,
                    frame_index=frame_index,
                    prediction_index=prediction_index,
                    bbox=bbox,
                    area=int(mask_record["area"]),
                    image_area=image_area,
                    predicted_iou=float(mask_record["predicted_iou"]),
                    stability_score=float(mask_record["stability_score"]),
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
        del image, masks
    elapsed_stream = time.perf_counter() - started_stream
    return tuple(raw), {
        "frame_count": len(decoded.frames),
        "elapsed_seconds": elapsed_stream,
        "mean_seconds_per_frame": elapsed_stream / len(decoded.frames),
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "frames": frame_rows,
    }


def _bbox_dict(bbox: BBox) -> dict[str, int]:
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
    profiles: tuple[FilterProfile, ...],
    points_per_side: int,
    points_per_batch: int,
    device: str,
    output_path: Path,
) -> dict[str, Any]:
    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise ValueError("Supply one annotation for every stream.")
    if not profiles or len({item.profile_id for item in profiles}) != len(profiles):
        raise ValueError("profiles must have unique profile IDs and must not be empty.")
    if points_per_side <= 0 or points_per_batch <= 0:
        raise ValueError("point sampling values must be positive.")
    checkpoint = checkpoint_path.resolve(strict=True)
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_checkpoint_sha256.casefold():
        raise ValueError("MobileSAM checkpoint SHA-256 mismatch.")

    import torch
    import torchvision

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    generator = _load_generator(
        checkpoint,
        device=device,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
    )
    stream_payloads: dict[str, Any] = {}
    metrics_by_profile: dict[str, list[dict[str, Any]]] = {
        profile.profile_id: [] for profile in profiles
    }
    total_started = time.perf_counter()
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        decoded = _load_stream(stream_directory)
        raw, runtime = _infer_stream(decoded, generator, torch, device)
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        if annotation.stream_id != decoded.stream.stream_id:
            raise ValueError("Annotation and stream_id mismatch.")
        evaluations: dict[str, Any] = {}
        for profile in profiles:
            metrics = evaluate_candidate_predictions(
                annotation,
                predictions_for_profile(raw, profile=profile),
            )
            evaluations[profile.profile_id] = metrics
            metrics_by_profile[profile.profile_id].append(metrics)
        stream_payloads[decoded.stream.stream_id] = {
            "runtime": runtime,
            "evaluations": evaluations,
            "raw_predictions": [
                {
                    "frame_id": item.frame_id,
                    "frame_index": item.frame_index,
                    "prediction_index": item.prediction_index,
                    "bbox": _bbox_dict(item.bbox),
                    "area": item.area,
                    "image_area": item.image_area,
                    "predicted_iou": item.predicted_iou,
                    "stability_score": item.stability_score,
                }
                for item in raw
            ],
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "evaluation_only_candidate_extraction_gate",
        "ground_truth_boundary": "Annotations were loaded only after model inference.",
        "model": {
            "family": "mobile_sam",
            "model_name": "vit_t",
            "weights": "mobile_sam.pt",
            "semantic_classes": "class_agnostic",
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": actual_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": device,
            "dtype": "float32",
        },
        "generator": {
            "points_per_side": points_per_side,
            "points_per_batch": points_per_batch,
            "prediction_iou_threshold": 0.0,
            "stability_score_threshold": 0.0,
            "box_nms_threshold": 0.7,
            "crop_n_layers": 0,
        },
        "profiles": [asdict(profile) for profile in profiles],
        "streams": stream_payloads,
        "aggregate": {
            profile.profile_id: aggregate_candidate_metrics(
                tuple(metrics_by_profile[profile.profile_id])
            )
            for profile in profiles
        },
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
    parser.add_argument("--points-per-side", type=int, default=8)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_benchmark(
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        profiles=DEFAULT_PROFILES,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
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
