"""Grounding DINO loading, inference, and prediction filtering."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stream_analysis.contracts import BBox, ImageSize


HF_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
RAW_POSTPROCESS_FLOOR = 0.01


@dataclass(frozen=True, slots=True)
class RawPrediction:
    """One processor post-processed detector prediction."""

    frame_id: str
    frame_index: int
    prediction_index: int
    score: float
    phrase: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class FilterProfile:
    """Score and class-agnostic NMS policy."""

    score_threshold: float
    class_agnostic_nms_iou: float | None

    @property
    def profile_id(self) -> str:
        nms = (
            "none"
            if self.class_agnostic_nms_iou is None
            else f"{self.class_agnostic_nms_iou:.2f}"
        )
        return f"score_{self.score_threshold:.2f}_nms_{nms}"


@dataclass(frozen=True, slots=True)
class PromptProfile:
    """Named text prompt and its inference source."""

    prompt_id: str
    text: str
    selectable: bool
    source: str

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.prompt_id.replace("_", "").isalnum():
            raise ValueError("prompt_id must contain letters, digits or underscores.")
        if not self.text or self.text != self.text.casefold() or not self.text.endswith("."):
            raise ValueError("prompt text must be lowercase and end with a period.")
        if not self.source:
            raise ValueError("prompt source must be non-empty.")


@dataclass(frozen=True, slots=True)
class GeometryProfile:
    """Large-box rejection policy expressed relative to the image size."""

    geometry_id: str
    min_area_ratio: float | None = None
    min_span_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.geometry_id or not self.geometry_id.replace("_", "").isalnum():
            raise ValueError("geometry_id must contain letters, digits or underscores.")
        for value in (self.min_area_ratio, self.min_span_ratio):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError("geometry ratios must be in (0, 1].")
        if self.min_span_ratio is not None and self.min_area_ratio is None:
            raise ValueError("a span threshold requires an area threshold.")


@dataclass(frozen=True, slots=True)
class ProfiledPrediction:
    """Prediction plus deterministic geometry-profile diagnostics."""

    prediction: RawPrediction
    bbox_area_ratio: float
    bbox_width_ratio: float
    bbox_height_ratio: float
    surface_like: bool
    geometry_rejected: bool


def integral_box(values: Any, image_size: ImageSize) -> BBox | None:
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


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection = left.intersection(right)
    if intersection is None:
        return 0.0
    union = left.area + right.area - intersection.area
    return 0.0 if union <= 0.0 else intersection.area / union


def class_agnostic_nms(
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
        ordered = sorted(
            by_frame[frame_id],
            key=lambda item: (-item.score, item.prediction_index),
        )
        for item in ordered:
            if all(
                bbox_iou(item.bbox, earlier.bbox) <= iou_threshold
                for earlier in accepted
            ):
                accepted.append(item)
        kept.extend(sorted(accepted, key=lambda item: item.prediction_index))
    return tuple(kept)


def filter_predictions(
    raw: Iterable[RawPrediction],
    *,
    profile: FilterProfile,
) -> tuple[RawPrediction, ...]:
    """Apply score threshold followed by optional class-agnostic NMS."""

    if not 0.0 <= profile.score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1].")
    selected = tuple(item for item in raw if item.score >= profile.score_threshold)
    if profile.class_agnostic_nms_iou is not None:
        selected = class_agnostic_nms(selected, profile.class_agnostic_nms_iou)
    return selected


def geometry_rejects(
    bbox: BBox,
    image_size: ImageSize,
    geometry: GeometryProfile,
) -> bool:
    frame_area = image_size.width * image_size.height
    if frame_area <= 0 or geometry.min_area_ratio is None:
        return False
    area_ratio = bbox.area / frame_area
    if area_ratio < geometry.min_area_ratio:
        return False
    if geometry.min_span_ratio is None:
        return True
    return (
        bbox.width / image_size.width >= geometry.min_span_ratio
        or bbox.height / image_size.height >= geometry.min_span_ratio
    )


def surface_like(
    bbox: BBox,
    image_size: ImageSize,
    geometries: Sequence[GeometryProfile],
) -> bool:
    return any(geometry_rejects(bbox, image_size, geometry) for geometry in geometries)


def select_predictions_for_profile(
    raw: Iterable[RawPrediction],
    *,
    filter_profile: FilterProfile,
    geometry: GeometryProfile,
    image_sizes: Mapping[str, ImageSize],
    surface_geometries: Sequence[GeometryProfile] = (),
) -> tuple[ProfiledPrediction, ...]:
    """Apply score/NMS and geometry policies while preserving stable order."""

    selected = filter_predictions(raw, profile=filter_profile)
    profiled: list[ProfiledPrediction] = []
    for row in selected:
        image_size = image_sizes.get(row.frame_id)
        if image_size is None:
            raise KeyError(f"missing frame size for {row.frame_id}")
        frame_area = image_size.width * image_size.height
        profiled.append(
            ProfiledPrediction(
                prediction=row,
                bbox_area_ratio=row.bbox.area / frame_area,
                bbox_width_ratio=row.bbox.width / image_size.width,
                bbox_height_ratio=row.bbox.height / image_size.height,
                surface_like=surface_like(row.bbox, image_size, surface_geometries),
                geometry_rejected=geometry_rejects(row.bbox, image_size, geometry),
            )
        )
    return tuple(profiled)


def frame_array(frame: Any) -> np.ndarray:
    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height, frame.image_size.width, 3
    ).copy()


def move_to_device(values: Any, device: str) -> Any:
    if hasattr(values, "to"):
        return values.to(device)
    if isinstance(values, dict):
        return {key: move_to_device(value, device) for key, value in values.items()}
    return values


def post_process(
    processor: Any,
    outputs: Any,
    inputs: Any,
    target_sizes: Any,
) -> Any:
    """Support both documented Grounding DINO processor keyword variants."""

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


def normalise_processor_predictions(
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
    labels = _as_list(result.get("text_labels", result.get("labels", ())))
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            "Grounding DINO processor boxes, scores and labels have different lengths."
        )
    sortable: list[tuple[float, str, BBox, int]] = []
    for source_index, (box, score, label) in enumerate(
        zip(boxes, scores, labels, strict=True)
    ):
        bbox = integral_box(box, image_size)
        score_value = float(score)
        if bbox is None or not math.isfinite(score_value):
            continue
        sortable.append((score_value, str(label), bbox, source_index))
    sortable.sort(
        key=lambda item: (
            -item[0],
            item[2].x,
            item[2].y,
            item[2].width,
            item[2].height,
            item[1],
            item[3],
        )
    )
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


def rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def load_model_processor(
    model_directory: Path,
    *,
    torch_module: Any,
    device: str,
) -> tuple[Any, Any]:
    """Load a local Grounding DINO processor/model pair without hub fallback."""

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


def model_dtype(model: Any) -> str:
    try:
        value = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration):
        return "unknown"
    return value.removeprefix("torch.")


def infer_stream(
    decoded: Any,
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    prompt: str,
) -> tuple[tuple[RawPrediction, ...], dict[str, Any]]:
    """Run one text prompt across a decoded RGB stream."""

    from PIL import Image

    raw: list[RawPrediction] = []
    frames: list[dict[str, Any]] = []
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    stream_started = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        image = Image.fromarray(frame_array(frame), mode="RGB")
        if device == "cuda":
            torch_module.cuda.synchronize()
        started = time.perf_counter()
        inputs = move_to_device(
            processor(images=image, text=prompt, return_tensors="pt"), device
        )
        model_started = time.perf_counter()
        with torch_module.inference_mode():
            outputs = model(**inputs)
        if device == "cuda":
            torch_module.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        target_sizes = torch_module.tensor(
            [[frame.image_size.height, frame.image_size.width]]
        )
        processed = post_process(processor, outputs, inputs, target_sizes)
        if not isinstance(processed, (list, tuple)) or len(processed) != 1:
            raise ValueError("Grounding DINO processor must return one result for one frame.")
        rows = normalise_processor_predictions(
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
    elapsed = time.perf_counter() - stream_started
    latencies = [float(item["total_inference_seconds"]) for item in frames]
    return tuple(raw), {
        "frame_count": len(frames),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_frame": None if not frames else elapsed / len(frames),
        "latency_seconds": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "process_rss_bytes": rss_bytes(),
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "frames": frames,
    }


__all__ = [
    "FilterProfile",
    "GeometryProfile",
    "HF_REVISION",
    "ProfiledPrediction",
    "PromptProfile",
    "RAW_POSTPROCESS_FLOOR",
    "RawPrediction",
    "bbox_iou",
    "class_agnostic_nms",
    "filter_predictions",
    "frame_array",
    "geometry_rejects",
    "infer_stream",
    "integral_box",
    "load_model_processor",
    "model_dtype",
    "move_to_device",
    "normalise_processor_predictions",
    "percentile",
    "post_process",
    "rss_bytes",
    "select_predictions_for_profile",
    "surface_like",
]
