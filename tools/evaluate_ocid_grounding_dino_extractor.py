"""Grounding DINO inference and evaluation utilities for OCID streams.

Reusable inference functions read RGB streams only. The command-line evaluator
opens matching annotations only after every frame in a stream has been
processed and its predictions have been persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from stream_analysis import ManifestLoadRequest, ProducerProvenance, load_decoded_stream
from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.runtime import grounding_dino as runtime_grounding
from stream_analysis.evaluation import (
    PredictedCandidate,
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
    load_annotation,
)

try:  # Support both ``python -m tools...`` and direct ``python tools/...`` use.
    from tools.ocid_evaluation_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI invocation.
    from ocid_evaluation_common import (  # type: ignore[no-redef]
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )


SCHEMA_VERSION = "ocid-grounding-dino-v1"
MODEL_FAMILY = "grounding_dino_tiny"
HF_REVISION = runtime_grounding.HF_REVISION
REGISTERED_PROMPT = "object."
RAW_POSTPROCESS_FLOOR = runtime_grounding.RAW_POSTPROCESS_FLOOR
SCORE_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
NMS_LEVELS: tuple[float | None, ...] = (None, 0.30, 0.50, 0.70)
IOU_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90)
RawPrediction = runtime_grounding.RawPrediction
FilterProfile = runtime_grounding.FilterProfile


DEFAULT_PROFILES = tuple(
    FilterProfile(score_threshold=threshold, class_agnostic_nms_iou=nms)
    for threshold in SCORE_THRESHOLDS
    for nms in NMS_LEVELS
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
        config_digest="sha256:ocid-grounding-dino-config-v1",
    )


def _load_stream(stream_directory: Path):
    return load_decoded_stream(
        ManifestLoadRequest(
            stream_root=stream_directory.resolve(strict=True),
            producer=_producer(),
        )
    )


def _validate_prompt(prompt: str) -> str:
    if prompt != REGISTERED_PROMPT:
        raise ValueError(
            f"OCID evaluation uses only the configured generic prompt {REGISTERED_PROMPT!r}."
        )
    return prompt


def _benchmark_provenance(
    *,
    benchmark_spec_path: Path,
    reviewed_root: Path,
    stream_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Record configured inputs after the development-pair check passed."""

    resolved_spec = benchmark_spec_path.resolve(strict=True)
    resolved_root = reviewed_root.resolve(strict=True)
    return {
        "contract": "exact_canonical_ocid_evaluation_development_input_pairs",
        "benchmark_spec_path": resolved_spec.as_posix(),
        "benchmark_spec_sha256": _sha256(resolved_spec),
        "reviewed_root": resolved_root.as_posix(),
        "development_stream_count": len(stream_ids),
        "development_stream_ids": list(stream_ids),
    }


_integral_box = runtime_grounding.integral_box
_bbox_iou = runtime_grounding.bbox_iou
_class_agnostic_nms = runtime_grounding.class_agnostic_nms
_frame_array = runtime_grounding.frame_array
_move_to_device = runtime_grounding.move_to_device
_post_process = runtime_grounding.post_process
_normalise_processor_predictions = runtime_grounding.normalise_processor_predictions
_rss_bytes = runtime_grounding.rss_bytes
_percentile = runtime_grounding.percentile
_load_model_processor = runtime_grounding.load_model_processor
_model_dtype = runtime_grounding.model_dtype


def predictions_for_profile(
    raw: tuple[RawPrediction, ...],
    *,
    profile: FilterProfile,
) -> tuple[PredictedCandidate, ...]:
    selected = runtime_grounding.filter_predictions(raw, profile=profile)
    return tuple(
        PredictedCandidate(
            candidate_id=f"grounding-dino:{item.frame_id}:{item.prediction_index:03d}",
            frame_id=item.frame_id,
            frame_index=item.frame_index,
            bbox=item.bbox,
            validity_status="valid",
            warning_ids=(),
            error_ids=(),
        )
        for item in selected
    )


def _infer_stream(
    decoded: Any,
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    prompt: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    _validate_prompt(prompt)
    return runtime_grounding.infer_stream(
        decoded,
        processor=processor,
        model=model,
        torch_module=torch_module,
        device=device,
        prompt=prompt,
    )


def _profile_metrics(annotation: Any, raw: tuple[RawPrediction, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in DEFAULT_PROFILES:
        candidates = predictions_for_profile(raw, profile=profile)
        result[profile.profile_id] = {
            "candidate_count": len(candidates),
            "iou_metrics": {
                f"{threshold:.2f}": evaluate_candidate_predictions(
                    annotation, candidates, iou_threshold=threshold
                )
                for threshold in IOU_LEVELS
            },
        }
    return result


def _aggregate_profiles(stream_metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = tuple(stream_metrics)
    return {
        profile.profile_id: {
            "pooled_iou_metrics": {
                f"{threshold:.2f}": aggregate_candidate_metrics(
                    tuple(record[profile.profile_id]["iou_metrics"][f"{threshold:.2f}"] for record in records)
                )
                for threshold in IOU_LEVELS
            }
        }
        for profile in DEFAULT_PROFILES
    }


def _aggregate_runtime(streams: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(streams)
    frame_rows = [row for stream in values for row in stream["frames"]]
    latencies = [float(row["total_inference_seconds"]) for row in frame_rows]
    allocated = [
        int(stream["peak_gpu_memory_allocated_bytes"])
        for stream in values
        if stream["peak_gpu_memory_allocated_bytes"] is not None
    ]
    reserved = [
        int(stream["peak_gpu_memory_reserved_bytes"])
        for stream in values
        if stream["peak_gpu_memory_reserved_bytes"] is not None
    ]
    rss = [int(stream["process_rss_bytes"]) for stream in values if stream["process_rss_bytes"] is not None]
    elapsed = sum(float(stream["elapsed_seconds"]) for stream in values)
    return {
        "frame_count": len(frame_rows),
        "sum_stream_elapsed_seconds": elapsed,
        "mean_seconds_per_frame": None if not frame_rows else elapsed / len(frame_rows),
        "latency_seconds": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "max_process_rss_bytes": max(rss, default=None),
        "max_peak_gpu_memory_allocated_bytes": max(allocated, default=None),
        "max_peak_gpu_memory_reserved_bytes": max(reserved, default=None),
    }


def _bbox_dict(bbox: BBox) -> dict[str, float]:
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
    model_directory: Path,
    expected_model_sha256: str,
    device: str,
    output_path: Path,
    prompt: str = REGISTERED_PROMPT,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> dict[str, Any]:
    """Run configured profiles over development streams without model re-runs."""

    _validate_prompt(prompt)
    inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    # The validator derives these paths from the configured contract. Use that
    # canonical order rather than caller order for deterministic reports.
    stream_directories = tuple(item.stream_directory for item in inventory)
    annotation_paths = tuple(item.annotation_path for item in inventory)
    benchmark_provenance = _benchmark_provenance(
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
        stream_ids=tuple(item.stream_id for item in inventory),
    )
    model_root = model_directory.resolve(strict=True)
    checkpoint = model_root / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError("Expected model.safetensors directly inside --model-directory.")
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_model_sha256.casefold():
        raise ValueError("Grounding DINO model.safetensors SHA-256 mismatch.")
    import torch
    import transformers

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    processor, model = _load_model_processor(model_root, torch_module=torch, device=device)
    started = time.perf_counter()
    streams: dict[str, Any] = {}
    all_metrics: list[dict[str, Any]] = []
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        decoded = _load_stream(stream_directory)
        raw, runtime = _infer_stream(
            decoded,
            processor=processor,
            model=model,
            torch_module=torch,
            device=device,
            prompt=prompt,
        )
        # This is intentionally after all RGB inference for this stream.
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        if annotation.stream_id != decoded.stream.stream_id:
            raise ValueError("Annotation and stream_id mismatch.")
        profile_metrics = _profile_metrics(annotation, raw)
        all_metrics.append(profile_metrics)
        streams[decoded.stream.stream_id] = {
            "runtime": runtime,
            "profiles": profile_metrics,
            "raw_predictions": [
                {
                    "frame_id": item.frame_id,
                    "frame_index": item.frame_index,
                    "prediction_index": item.prediction_index,
                    "score": item.score,
                    "phrase": item.phrase,
                    "bbox": _bbox_dict(item.bbox),
                }
                for item in raw
            ],
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "development_only_candidate_extraction_ocid_evaluation",
        "ground_truth_boundary": "Annotations were loaded only after RGB model inference per stream.",
        "benchmark": benchmark_provenance,
        "prompt": prompt,
        "postprocess": {"raw_score_floor": RAW_POSTPROCESS_FLOOR},
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "score_threshold": profile.score_threshold,
                "class_agnostic_nms_iou": profile.class_agnostic_nms_iou,
            }
            for profile in DEFAULT_PROFILES
        ],
        "iou_levels": IOU_LEVELS,
        "model": {
            "family": MODEL_FAMILY,
            "hf_revision": HF_REVISION,
            "model_directory": model_root.as_posix(),
            "model_safetensors_sha256": actual_hash,
            "model_safetensors_size_bytes": checkpoint.stat().st_size,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": device,
            "dtype": _model_dtype(model),
            "local_files_only": True,
        },
        "streams": streams,
        "aggregate_runtime": _aggregate_runtime(
            tuple(item["runtime"] for item in streams.values())
        ),
        "pooled_metrics": _aggregate_profiles(all_metrics),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-directory", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--prompt", default=REGISTERED_PROMPT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_benchmark(
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        model_directory=args.model_directory,
        expected_model_sha256=args.expected_model_sha256,
        device=args.device,
        output_path=args.output,
        prompt=args.prompt,
        benchmark_spec_path=args.benchmark_spec,
        reviewed_root=args.reviewed_root,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
