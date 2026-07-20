"""Benchmark local SAM 2.1 Hiera Tiny automatic masks as OCID candidates.

This is an isolated, development-only evaluation tool.  It accepts a fully
local Hugging Face model directory, forces Transformers offline mode, and
loads OCID annotations only after all RGB inference for a stream is complete.
It intentionally does not register SAM 2.1 as a production ``analyze``
provider.
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

try:  # Supports both ``python -m tools...`` and direct ``python tools/...``.
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


SCHEMA_VERSION = "ocid-sam2-extractor-gate-0.1"
DEFAULT_IOU_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True, slots=True)
class QualitySetting:
    """One SAM2 inference policy, shared by its post-hoc area variants."""

    quality_id: str
    predicted_iou_threshold: float
    stability_threshold: float

    def __post_init__(self) -> None:
        if not self.quality_id.strip():
            raise ValueError("quality_id must not be empty.")
        for field_name in ("predicted_iou_threshold", "stability_threshold"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be a real value in [0, 1].")


@dataclass(frozen=True, slots=True)
class FilterProfile:
    """Post-hoc mask-area policy attached to one quality setting."""

    profile_id: str
    quality_id: str
    min_area_ratio: float
    max_area_ratio: float = 0.50

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty.")
        if not self.quality_id.strip():
            raise ValueError("quality_id must not be empty.")
        if (
            isinstance(self.min_area_ratio, bool)
            or isinstance(self.max_area_ratio, bool)
            or not isinstance(self.min_area_ratio, (int, float))
            or not isinstance(self.max_area_ratio, (int, float))
            or not 0.0 <= self.min_area_ratio < self.max_area_ratio <= 1.0
        ):
            raise ValueError("area ratios must satisfy 0 <= min < max <= 1.")


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Parameters shared by every automatic-mask profile."""

    points_per_crop: int = 16
    points_per_batch: int = 32
    crops_n_layers: int = 0
    crops_nms_threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.points_per_crop <= 0 or self.points_per_batch <= 0:
            raise ValueError("points_per_crop and points_per_batch must be positive.")
        if self.crops_n_layers < 0:
            raise ValueError("crops_n_layers must be non-negative.")
        if not 0.0 <= self.crops_nms_threshold <= 1.0:
            raise ValueError("crops_nms_threshold must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class RawPrediction:
    quality_id: str
    frame_id: str
    frame_index: int
    prediction_index: int
    bbox: BBox
    model_bbox: BBox | None
    mask_area: int
    image_area: int
    score: float


DEFAULT_QUALITY_SETTINGS = (
    QualitySetting("permissive", 0.75, 0.85),
    QualitySetting("balanced", 0.85, 0.90),
    QualitySetting("official", 0.88, 0.95),
)
DEFAULT_MIN_AREA_RATIOS = (("0001", 0.001), ("0003", 0.003), ("0010", 0.010))
DEFAULT_PROFILES = tuple(
    FilterProfile(
        profile_id=f"{quality.quality_id}_area_{area_id}",
        quality_id=quality.quality_id,
        min_area_ratio=min_area_ratio,
    )
    for quality in DEFAULT_QUALITY_SETTINGS
    for area_id, min_area_ratio in DEFAULT_MIN_AREA_RATIOS
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
        config_digest="sha256:ocid-sam2-extractor-gate",
    )


def _load_stream(stream_directory: Path):
    return load_decoded_stream(
        ManifestLoadRequest(
            stream_root=stream_directory.resolve(strict=True),
            producer=_producer(),
        )
    )


def _frame_array(frame: Any) -> np.ndarray:
    """Return a writable RGB HWC array; no annotation data enters this path."""

    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    ).copy()


def _integral_box(values: Any, image_size: ImageSize) -> BBox | None:
    """Convert a SAM XYXY float box into a clipped integral project box."""

    array = np.asarray(_to_numpy(values), dtype=np.float64).reshape(-1)
    if array.shape != (4,) or not np.isfinite(array).all():
        return None
    left = max(0, min(image_size.width, int(np.floor(array[0]))))
    top = max(0, min(image_size.height, int(np.floor(array[1]))))
    right = max(0, min(image_size.width, int(np.ceil(array[2]))))
    bottom = max(0, min(image_size.height, int(np.ceil(array[3]))))
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right - left, bottom - top)


def _mask_box(mask: Any, image_size: ImageSize) -> tuple[BBox, int] | None:
    foreground = np.asarray(_to_numpy(mask), dtype=bool)
    if foreground.shape != (image_size.height, image_size.width) or not np.any(foreground):
        return None
    rows, columns = np.nonzero(foreground)
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return BBox(left, top, right - left, bottom - top), int(foreground.sum())


def _to_numpy(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    to_cpu = getattr(value, "cpu", None)
    if callable(to_cpu):
        value = to_cpu()
    convert = getattr(value, "numpy", None)
    return convert() if callable(convert) else value


def _output_values(output: Any, key: str) -> list[Any]:
    if not isinstance(output, dict) or key not in output:
        raise ValueError(f"SAM2 mask-generation output is missing {key!r}.")
    value = _to_numpy(output[key])
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"SAM2 mask-generation output {key!r} is not a sequence.")


def raw_predictions_from_output(
    output: Any,
    *,
    quality_id: str,
    frame_id: str,
    frame_index: int,
    image_size: ImageSize,
) -> tuple[RawPrediction, ...]:
    """Normalize one Transformers output into deterministic, mask-backed rows."""

    masks = _output_values(output, "masks")
    scores = _output_values(output, "scores")
    boxes = _output_values(output, "bounding_boxes")
    if not (len(masks) == len(scores) == len(boxes)):
        raise ValueError("SAM2 masks, scores and bounding_boxes must have equal lengths.")
    image_area = image_size.width * image_size.height
    result: list[RawPrediction] = []
    for prediction_index, (mask, score, box) in enumerate(zip(masks, scores, boxes, strict=True)):
        mask_details = _mask_box(mask, image_size)
        if mask_details is None:
            continue
        mask_bbox, mask_area = mask_details
        model_bbox = _integral_box(box, image_size)
        # A malformed model box must not create an out-of-frame candidate.  The
        # actual binary mask remains an auditable, deterministic fallback.
        candidate_bbox = model_bbox or mask_bbox
        score_array = np.asarray(_to_numpy(score), dtype=np.float64).reshape(-1)
        if score_array.size != 1 or not np.isfinite(score_array[0]):
            continue
        result.append(
            RawPrediction(
                quality_id=quality_id,
                frame_id=frame_id,
                frame_index=frame_index,
                prediction_index=prediction_index,
                bbox=candidate_bbox,
                model_bbox=model_bbox,
                mask_area=mask_area,
                image_area=image_area,
                score=float(score_array[0]),
            )
        )
    return tuple(result)


def predictions_for_profile(
    raw: tuple[RawPrediction, ...],
    *,
    profile: FilterProfile,
) -> tuple[PredictedCandidate, ...]:
    """Apply only the profile's area policy to pipeline quality-filtered masks."""

    candidates: list[PredictedCandidate] = []
    for item in raw:
        if item.quality_id != profile.quality_id:
            continue
        area_ratio = item.mask_area / item.image_area
        if area_ratio < profile.min_area_ratio or area_ratio > profile.max_area_ratio:
            continue
        candidates.append(
            PredictedCandidate(
                candidate_id=f"sam2:{profile.profile_id}:{item.frame_id}:{item.prediction_index:03d}",
                frame_id=item.frame_id,
                frame_index=item.frame_index,
                bbox=item.bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
        )
    return tuple(candidates)


def _load_mask_pipeline(
    model_directory: Path,
    *,
    revision: str,
    device: str,
) -> tuple[Any, Any, Any]:
    """Load a float32 SAM2 pipeline strictly from an explicit local directory."""

    # The environment flags close any accidental hub fallback in processor or
    # model auto-loading.  Missing local files therefore fail deterministically.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import pipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    generator = pipeline(
        "mask-generation",
        model=str(model_directory),
        device=device,
        dtype=torch.float32,
        revision=revision,
        trust_remote_code=False,
        model_kwargs={
            "use_safetensors": True,
        },
    )
    return generator, torch, transformers


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _infer_stream(
    decoded: Any,
    generator: Any,
    torch_module: Any,
    *,
    quality_settings: tuple[QualitySetting, ...],
    generator_config: GeneratorConfig,
    device: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    from PIL import Image

    raw: list[RawPrediction] = []
    frame_rows: list[dict[str, Any]] = []
    timings: list[float] = []
    rss_samples = [value for value in (_process_rss_bytes(),) if value is not None]
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()
    started_stream = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        image = Image.fromarray(_frame_array(frame), mode="RGB")
        for quality in quality_settings:
            started = time.perf_counter()
            output = generator(
                image,
                points_per_crop=generator_config.points_per_crop,
                points_per_batch=generator_config.points_per_batch,
                crops_n_layers=generator_config.crops_n_layers,
                crops_nms_thresh=generator_config.crops_nms_threshold,
                pred_iou_thresh=quality.predicted_iou_threshold,
                stability_score_thresh=quality.stability_threshold,
                output_bboxes_mask=True,
            )
            if device == "cuda":
                torch_module.cuda.synchronize()
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            rows = raw_predictions_from_output(
                output,
                quality_id=quality.quality_id,
                frame_id=frame.frame_id,
                frame_index=frame_index,
                image_size=frame.image_size,
            )
            raw.extend(rows)
            rss = _process_rss_bytes()
            if rss is not None:
                rss_samples.append(rss)
            frame_rows.append(
                {
                    "frame_id": frame.frame_id,
                    "frame_index": frame_index,
                    "quality_id": quality.quality_id,
                    "raw_prediction_count": len(rows),
                    "end_to_end_seconds": elapsed,
                    "model_forward_seconds": None,
                    "model_forward_note": "not separable from Transformers automatic-mask pipeline",
                }
            )
        del image
    elapsed_stream = time.perf_counter() - started_stream
    return tuple(raw), {
        "frame_count": len(decoded.frames),
        "quality_frame_calls": len(frame_rows),
        "elapsed_seconds": elapsed_stream,
        "end_to_end_seconds": {
            "median": _percentile(timings, 50),
            "p95": _percentile(timings, 95),
            "mean": None if not timings else sum(timings) / len(timings),
        },
        "model_forward_seconds": None,
        "peak_process_rss_bytes": max(rss_samples) if rss_samples else None,
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
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


def _validate_iou_thresholds(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ValueError("iou_thresholds must not be empty.")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError("IoU thresholds must be real values in [0, 1].")
        normalized.append(float(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("IoU thresholds must be unique.")
    return tuple(normalized)


def run_benchmark(
    *,
    stream_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    model_directory: Path,
    expected_model_sha256: str,
    revision: str,
    quality_settings: tuple[QualitySetting, ...],
    profiles: tuple[FilterProfile, ...],
    generator_config: GeneratorConfig,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    device: str,
    output_path: Path,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> dict[str, Any]:
    """Run local RGB inference, then evaluate each profile at each IoU level."""

    # This must remain the first operational action: it rejects held-out,
    # subset, duplicate, cross-paired and path-lookalike inputs before model
    # files, hashes or annotations are touched.
    development_inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    if not revision.strip():
        raise ValueError("revision must not be empty.")
    if (
        not quality_settings
        or not all(isinstance(item, QualitySetting) for item in quality_settings)
        or len({item.quality_id for item in quality_settings}) != len(quality_settings)
    ):
        raise ValueError("quality_settings must have unique IDs and must not be empty.")
    if (
        not profiles
        or not all(isinstance(item, FilterProfile) for item in profiles)
        or len({item.profile_id for item in profiles}) != len(profiles)
    ):
        raise ValueError("profiles must have unique profile IDs and must not be empty.")
    known_quality_ids = {item.quality_id for item in quality_settings}
    if any(profile.quality_id not in known_quality_ids for profile in profiles):
        raise ValueError("Every profile must refer to a supplied quality setting.")
    if not isinstance(generator_config, GeneratorConfig):
        raise TypeError("generator_config must be GeneratorConfig.")
    thresholds = _validate_iou_thresholds(iou_thresholds)
    directory = model_directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("model_directory must be a directory.")
    weights = directory / "model.safetensors"
    actual_hash = _sha256(weights.resolve(strict=True))
    if actual_hash != expected_model_sha256.casefold():
        raise ValueError("SAM2 model.safetensors SHA-256 mismatch.")

    total_started = time.perf_counter()
    model_load_started = time.perf_counter()
    generator, torch_module, transformers_module = _load_mask_pipeline(
        directory,
        revision=revision,
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    stream_payloads: dict[str, Any] = {}
    metrics_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {
        (profile.profile_id, f"{threshold:.2f}"): []
        for profile in profiles
        for threshold in thresholds
    }
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        decoded = _load_stream(stream_directory)
        raw, runtime = _infer_stream(
            decoded,
            generator,
            torch_module,
            quality_settings=quality_settings,
            generator_config=generator_config,
            device=device,
        )
        # This is deliberately after _infer_stream: ground truth remains outside
        # candidate generation, including all SAM2 profile selection inputs.
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        if annotation.stream_id != decoded.stream.stream_id:
            raise ValueError("Annotation and stream_id mismatch.")
        evaluations: dict[str, Any] = {}
        for profile in profiles:
            candidates = predictions_for_profile(raw, profile=profile)
            by_iou: dict[str, Any] = {}
            for threshold in thresholds:
                key = f"{threshold:.2f}"
                metrics = evaluate_candidate_predictions(
                    annotation,
                    candidates,
                    iou_threshold=threshold,
                )
                by_iou[key] = metrics
                metrics_by_key[(profile.profile_id, key)].append(metrics)
            evaluations[profile.profile_id] = by_iou
        stream_payloads[decoded.stream.stream_id] = {
            "runtime": runtime,
            "evaluations": evaluations,
            "raw_predictions": [
                {
                    "quality_id": item.quality_id,
                    "frame_id": item.frame_id,
                    "frame_index": item.frame_index,
                    "prediction_index": item.prediction_index,
                    "bbox": _bbox_dict(item.bbox),
                    "model_bbox": _bbox_dict(item.model_bbox),
                    "mask_area": item.mask_area,
                    "image_area": item.image_area,
                    "score": item.score,
                }
                for item in raw
            ],
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "evaluation_only_candidate_extraction_gate",
        "ground_truth_boundary": "Annotations were loaded only after model inference for each stream.",
        "gate_c1_development_contract": {
            "benchmark_spec_path": benchmark_spec_path.resolve(strict=True).as_posix(),
            "benchmark_spec_sha256": _sha256(benchmark_spec_path.resolve(strict=True)),
            "reviewed_root": reviewed_root.resolve(strict=True).as_posix(),
            "stream_count": len(development_inventory),
            "stream_ids": [item.stream_id for item in development_inventory],
        },
        "model": {
            "family": "sam2_1_hiera_tiny",
            "model_directory": directory.as_posix(),
            "weights": "model.safetensors",
            "model_sha256": actual_hash,
            "model_size_bytes": weights.stat().st_size,
            "revision": revision,
            "transformers_version": transformers_module.__version__,
            "torch_version": torch_module.__version__,
            "device": device,
            "dtype": "float32",
            "local_files_only": True,
            "network_fallback": "disabled",
        },
        "generator": asdict(generator_config),
        "model_load_seconds": model_load_seconds,
        "quality_settings": [asdict(setting) for setting in quality_settings],
        "profiles": [asdict(profile) for profile in profiles],
        "iou_thresholds": list(thresholds),
        "streams": stream_payloads,
        "aggregate": {
            profile.profile_id: {
                f"{threshold:.2f}": aggregate_candidate_metrics(
                    tuple(metrics_by_key[(profile.profile_id, f"{threshold:.2f}")])
                )
                for threshold in thresholds
            }
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
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    defaults = GeneratorConfig()
    parser.add_argument("--points-per-crop", type=int, default=defaults.points_per_crop)
    parser.add_argument("--points-per-batch", type=int, default=defaults.points_per_batch)
    parser.add_argument("--iou-threshold", type=float, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_benchmark(
            stream_directories=tuple(args.stream_directory),
            annotation_paths=tuple(args.annotation),
            model_directory=args.model_directory,
            expected_model_sha256=args.expected_model_sha256,
            revision=args.revision,
            quality_settings=DEFAULT_QUALITY_SETTINGS,
            profiles=DEFAULT_PROFILES,
            generator_config=GeneratorConfig(
                points_per_crop=args.points_per_crop,
                points_per_batch=args.points_per_batch,
            ),
            iou_thresholds=tuple(args.iou_threshold or DEFAULT_IOU_THRESHOLDS),
            benchmark_spec_path=args.benchmark_spec,
            reviewed_root=args.reviewed_root,
            device=args.device,
            output_path=args.output,
        )
    except ValueError as error:
        _parser().error(str(error))
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
