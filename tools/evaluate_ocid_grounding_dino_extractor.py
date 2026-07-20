"""Benchmark a local Grounding DINO Tiny checkpoint on OCID candidates.

This development-only helper is deliberately outside the normal analysis
pipeline.  It runs inference from RGB streams only; the matching annotation is
opened only after every frame in that stream has been processed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stream_analysis import ManifestLoadRequest, ProducerProvenance, load_decoded_stream
from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation import (
    PredictedCandidate,
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
    load_annotation,
)

try:  # Support both ``python -m tools...`` and direct ``python tools/...`` use.
    from tools.ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI invocation.
    from ocid_gate_c1_common import (  # type: ignore[no-redef]
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )


SCHEMA_VERSION = "ocid-grounding-dino-tiny-gate-c1.v1"
MODEL_FAMILY = "grounding_dino_tiny"
HF_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
REGISTERED_PROMPT = "object."
RAW_POSTPROCESS_FLOOR = 0.01
SCORE_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
NMS_LEVELS: tuple[float | None, ...] = (None, 0.30, 0.50, 0.70)
IOU_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90)
@dataclass(frozen=True, slots=True)
class RawPrediction:
    """One processor post-processed prediction retained for all profiles."""

    frame_id: str
    frame_index: int
    prediction_index: int
    score: float
    phrase: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class FilterProfile:
    score_threshold: float
    class_agnostic_nms_iou: float | None

    @property
    def profile_id(self) -> str:
        nms = "none" if self.class_agnostic_nms_iou is None else f"{self.class_agnostic_nms_iou:.2f}"
        return f"score_{self.score_threshold:.2f}_nms_{nms}"


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
        config_digest="sha256:ocid-grounding-dino-extractor-gate-c1",
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
            f"Gate C1 permits only the pre-registered generic prompt {REGISTERED_PROMPT!r}."
        )
    return prompt


def _benchmark_provenance(
    *,
    benchmark_spec_path: Path,
    reviewed_root: Path,
    stream_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Record the frozen inputs after the exact development-pair guard passed."""

    resolved_spec = benchmark_spec_path.resolve(strict=True)
    resolved_root = reviewed_root.resolve(strict=True)
    return {
        "contract": "exact_canonical_gate_c1_development_input_pairs",
        "benchmark_spec_path": resolved_spec.as_posix(),
        "benchmark_spec_sha256": _sha256(resolved_spec),
        "reviewed_root": resolved_root.as_posix(),
        "development_stream_count": len(stream_ids),
        "development_stream_ids": list(stream_ids),
    }


def _integral_box(values: Any, image_size: ImageSize) -> BBox | None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        return None
    left = max(0, min(image_size.width, int(math.floor(float(array[0])))))
    top = max(0, min(image_size.height, int(math.floor(float(array[1])))))
    right = max(0, min(image_size.width, int(math.ceil(float(array[2])))))
    bottom = max(0, min(image_size.height, int(math.ceil(float(array[3])))))
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right - left, bottom - top)


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection = left.intersection(right)
    if intersection is None:
        return 0.0
    union = left.area + right.area - intersection.area
    return 0.0 if union <= 0.0 else intersection.area / union


def _class_agnostic_nms(
    selected: Iterable[RawPrediction],
    iou_threshold: float,
) -> tuple[RawPrediction, ...]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("class_agnostic_nms_iou must be in [0, 1].")
    by_frame: dict[str, list[RawPrediction]] = {}
    for item in selected:
        by_frame.setdefault(item.frame_id, []).append(item)
    kept: list[RawPrediction] = []
    for frame_id in sorted(by_frame):
        accepted: list[RawPrediction] = []
        ordered = sorted(by_frame[frame_id], key=lambda item: (-item.score, item.prediction_index))
        for item in ordered:
            if all(_bbox_iou(item.bbox, earlier.bbox) <= iou_threshold for earlier in accepted):
                accepted.append(item)
        kept.extend(sorted(accepted, key=lambda item: item.prediction_index))
    return tuple(kept)


def predictions_for_profile(
    raw: tuple[RawPrediction, ...],
    *,
    profile: FilterProfile,
) -> tuple[PredictedCandidate, ...]:
    if not 0.0 <= profile.score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1].")
    selected = tuple(item for item in raw if item.score >= profile.score_threshold)
    if profile.class_agnostic_nms_iou is not None:
        selected = _class_agnostic_nms(selected, profile.class_agnostic_nms_iou)
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


def _frame_array(frame: Any) -> np.ndarray:
    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height, frame.image_size.width, 3
    ).copy()


def _move_to_device(values: Any, device: str) -> Any:
    if hasattr(values, "to"):
        return values.to(device)
    if isinstance(values, dict):
        return {key: _move_to_device(value, device) for key, value in values.items()}
    return values


def _post_process(
    processor: Any,
    outputs: Any,
    inputs: Any,
    target_sizes: Any,
) -> Any:
    """Support the two documented Grounding DINO processor keyword variants."""

    method = processor.post_process_grounded_object_detection
    parameters = inspect.signature(method).parameters
    common: dict[str, Any] = {"target_sizes": target_sizes}
    if "input_ids" in parameters and hasattr(inputs, "input_ids"):
        common["input_ids"] = inputs.input_ids
    elif "input_ids" in parameters and isinstance(inputs, dict) and "input_ids" in inputs:
        common["input_ids"] = inputs["input_ids"]
    if "threshold" in parameters:
        common["threshold"] = RAW_POSTPROCESS_FLOOR
    else:
        common["box_threshold"] = RAW_POSTPROCESS_FLOOR
        if "text_threshold" in parameters:
            common["text_threshold"] = RAW_POSTPROCESS_FLOOR
    return method(outputs, **common)


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().to("cpu").tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _normalise_processor_predictions(
    result: Any,
    *,
    frame_id: str,
    frame_index: int,
    image_size: ImageSize,
) -> tuple[RawPrediction, ...]:
    if not isinstance(result, dict):
        raise TypeError("Grounding DINO processor result must be a dictionary.")
    boxes = _as_list(result.get("boxes", ()))
    scores = _as_list(result.get("scores", ()))
    # Transformers 5.14 names Grounding DINO's decoded string phrases
    # ``text_labels``.  ``labels`` remains a compatibility fallback for older
    # processor outputs; it is audit metadata and never affects filtering.
    labels = _as_list(result.get("text_labels", result.get("labels", ())))
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Grounding DINO processor boxes, scores and labels have different lengths.")
    sortable: list[tuple[float, str, BBox, int]] = []
    for source_index, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
        bbox = _integral_box(box, image_size)
        score_value = float(score)
        if bbox is None or not math.isfinite(score_value):
            continue
        sortable.append((score_value, str(label), bbox, source_index))
    sortable.sort(key=lambda item: (-item[0], item[2].x, item[2].y, item[2].width, item[2].height, item[1], item[3]))
    return tuple(
        RawPrediction(
            frame_id=frame_id,
            frame_index=frame_index,
            prediction_index=index,
            score=score,
            phrase=label,
            bbox=bbox,
        )
        for index, (score, label, bbox, _source_index) in enumerate(sortable)
    )


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _load_model_processor(model_directory: Path, *, torch_module: Any, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_directory), local_files_only=True, revision=HF_REVISION
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_directory), local_files_only=True, revision=HF_REVISION
    )
    model.to(device)
    model.eval()
    return processor, model


def _model_dtype(model: Any) -> str:
    try:
        value = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration):
        return "unknown"
    return value.removeprefix("torch.")


def _infer_stream(
    decoded: Any,
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    prompt: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    from PIL import Image

    _validate_prompt(prompt)
    raw: list[RawPrediction] = []
    frames: list[dict[str, Any]] = []
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    stream_started = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        image = Image.fromarray(_frame_array(frame), mode="RGB")
        if device == "cuda":
            torch_module.cuda.synchronize()
        started = time.perf_counter()
        inputs = _move_to_device(processor(images=image, text=prompt, return_tensors="pt"), device)
        model_started = time.perf_counter()
        with torch_module.inference_mode():
            outputs = model(**inputs)
        if device == "cuda":
            torch_module.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        target_sizes = torch_module.tensor([[frame.image_size.height, frame.image_size.width]])
        processed = _post_process(processor, outputs, inputs, target_sizes)
        if not isinstance(processed, (list, tuple)) or len(processed) != 1:
            raise ValueError("Grounding DINO processor must return one result for one frame.")
        rows = _normalise_processor_predictions(
            processed[0],
            frame_id=frame.frame_id,
            frame_index=frame_index,
            image_size=frame.image_size,
        )
        total_seconds = time.perf_counter() - started
        raw.extend(rows)
        frames.append(
            {
                "frame_id": frame.frame_id,
                "raw_prediction_count": len(rows),
                "model_forward_seconds": model_seconds,
                "total_inference_seconds": total_seconds,
            }
        )
        del image, inputs, outputs, processed
    total_seconds = time.perf_counter() - stream_started
    latencies = [float(row["total_inference_seconds"]) for row in frames]
    return tuple(raw), {
        "frame_count": len(frames),
        "elapsed_seconds": total_seconds,
        "mean_seconds_per_frame": None if not frames else total_seconds / len(frames),
        "latency_seconds": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "process_rss_bytes": _rss_bytes(),
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "frames": frames,
    }


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
    """Run fixed C1 profiles over development streams without model re-runs."""

    _validate_prompt(prompt)
    inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    # The validator derives these paths from the frozen contract.  Use that
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
        "scope": "development_only_candidate_extraction_gate_c1",
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
