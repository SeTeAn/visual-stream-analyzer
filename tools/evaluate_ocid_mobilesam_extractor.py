"""Benchmark official MobileSAM as an OCID candidate extractor.

This is an isolated evaluation tool. OCID annotations are loaded only after
inference and are never provided to the model or to the normal analyze path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
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

try:  # Supports both ``python tools/...py`` and ``import tools....``.
    from tools.ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use.
    from ocid_gate_c1_common import (  # type: ignore[no-redef]
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )


SCHEMA_VERSION = "ocid-learned-extractor-gate-0.2"
DEFAULT_IOU_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


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


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_seconds": None if not values else sum(values) / len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
    }


def _process_rss_sampler():
    """Return an RSS sampler when psutil is locally installed, otherwise None."""
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process(os.getpid())
    return lambda: int(process.memory_info().rss)


def _module_provenance(module_name: str) -> dict[str, str | None]:
    """Record the local Python source used by a lazily imported provider."""
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    source_path = Path(module_file).resolve(strict=True) if module_file else None
    return {
        "module": module_name,
        "module_version": getattr(module, "__version__", None),
        "module_file": None if source_path is None else source_path.as_posix(),
        "module_file_sha256": (
            None if source_path is None or not source_path.is_file() else _sha256(source_path)
        ),
    }


def _infer_stream(
    decoded: Any,
    generator: Any,
    torch_module: Any,
    device: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    raw: list[RawPrediction] = []
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    rss_sampler = _process_rss_sampler()
    rss_samples: list[int] = []
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
        latencies.append(elapsed)
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
                "process_rss_bytes": None if rss_sampler is None else rss_sampler(),
            }
        )
        if rss_sampler is not None:
            rss_samples.append(frame_rows[-1]["process_rss_bytes"])
        del image, masks
    elapsed_stream = time.perf_counter() - started_stream
    return tuple(raw), {
        "frame_count": len(decoded.frames),
        "elapsed_seconds": elapsed_stream,
        "total_seconds": elapsed_stream,
        "mean_seconds_per_frame": elapsed_stream / len(decoded.frames),
        "frame_latency_seconds": _latency_summary(latencies),
        "timing_scope": (
            "frame inference_seconds covers generator.generate plus CUDA synchronization; "
            "total_seconds also includes RGB decoding, mask-to-box conversion, and reporting."
        ),
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "process_rss": {
            "available": rss_sampler is not None,
            "peak_bytes": None if not rss_samples else max(rss_samples),
            "final_bytes": None if not rss_samples else rss_samples[-1],
        },
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
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    points_per_side: int,
    points_per_batch: int,
    device: str,
    output_path: Path,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> dict[str, Any]:
    # This is intentionally the first operation: no checkpoint, model or
    # annotation may be accessed before the exact development-only contract
    # has accepted all supplied input pairs.
    development_inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise ValueError("Supply one annotation for every stream.")
    if not profiles or len({item.profile_id for item in profiles}) != len(profiles):
        raise ValueError("profiles must have unique profile IDs and must not be empty.")
    if (
        not iou_thresholds
        or len(set(iou_thresholds)) != len(iou_thresholds)
        or any(not 0.0 <= threshold <= 1.0 for threshold in iou_thresholds)
    ):
        raise ValueError("iou_thresholds must contain unique values in [0, 1].")
    if points_per_side <= 0 or points_per_batch <= 0:
        raise ValueError("point sampling values must be positive.")
    checkpoint = checkpoint_path.resolve(strict=True)
    benchmark_spec = Path(benchmark_spec_path).resolve(strict=True)
    canonical_reviewed_root = Path(reviewed_root).resolve(strict=True)
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
    ordered_inputs = tuple(
        sorted(
            zip(stream_directories, annotation_paths, strict=True),
            key=lambda pair: pair[0].resolve(strict=False).as_posix(),
        )
    )
    metrics_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {
        (profile.profile_id, f"{threshold:.2f}"): []
        for profile in profiles
        for threshold in iou_thresholds
    }
    all_frame_latencies: list[float] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    peak_rss: list[int] = []
    rss_reporting_available = False
    total_started = time.perf_counter()
    for stream_directory, annotation_path in ordered_inputs:
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
            by_iou: dict[str, Any] = {}
            filtered_predictions = predictions_for_profile(raw, profile=profile)
            for threshold in iou_thresholds:
                key = f"{threshold:.2f}"
                metrics = evaluate_candidate_predictions(
                    annotation,
                    filtered_predictions,
                    iou_threshold=threshold,
                )
                by_iou[key] = metrics
                metrics_by_key[(profile.profile_id, key)].append(metrics)
            evaluations[profile.profile_id] = by_iou
        all_frame_latencies.extend(
            row["inference_seconds"] for row in runtime["frames"]
        )
        for metric_key, values in (
            ("peak_gpu_memory_allocated_bytes", peak_allocated),
            ("peak_gpu_memory_reserved_bytes", peak_reserved),
        ):
            value = runtime[metric_key]
            if value is not None:
                values.append(value)
        rss_peak = runtime["process_rss"]["peak_bytes"]
        rss_reporting_available = (
            rss_reporting_available or runtime["process_rss"]["available"]
        )
        if rss_peak is not None:
            peak_rss.append(rss_peak)
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
        "development_contract": {
            "benchmark_spec_path": benchmark_spec.as_posix(),
            "benchmark_spec_sha256": _sha256(benchmark_spec),
            "reviewed_root": canonical_reviewed_root.as_posix(),
            "stream_ids": [stream.stream_id for stream in development_inventory],
        },
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
            "local_source": _module_provenance("mobile_sam"),
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
        "iou_thresholds": list(iou_thresholds),
        "streams": stream_payloads,
        "aggregate": {
            profile.profile_id: {
                f"{threshold:.2f}": aggregate_candidate_metrics(
                    tuple(metrics_by_key[(profile.profile_id, f"{threshold:.2f}")])
                )
                for threshold in iou_thresholds
            }
            for profile in profiles
        },
        "resource_summary": {
            "frame_latency_seconds": _latency_summary(all_frame_latencies),
            "peak_gpu_memory_allocated_bytes": None if not peak_allocated else max(peak_allocated),
            "peak_gpu_memory_reserved_bytes": None if not peak_reserved else max(peak_reserved),
            "process_rss": {
                "available": rss_reporting_available,
                "peak_bytes": None if not peak_rss else max(peak_rss),
            },
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
    parser.add_argument("--iou-threshold", type=float, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
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
        iou_thresholds=tuple(args.iou_threshold or DEFAULT_IOU_THRESHOLDS),
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
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
