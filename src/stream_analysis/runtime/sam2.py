"""Local bbox-prompted SAM2 refinement for detector candidates.

The module knows only RGB pixels and candidate boxes. A frame is sent through
SAM2 once with all valid boxes in one prompt batch, then every source box
receives either a usable mask result or an explicit bbox fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from stream_analysis.contracts import BBox, ImageSize


RefinementStatus = Literal["valid", "bbox_fallback"]


@dataclass(frozen=True, slots=True)
class MaskCleanupConfig:
    """Conservative connected-component removal policy for binary masks.

    Components are removed only when their area is below *both* explicitly
    configured thresholds.  The largest component is always retained, so a
    too-strict policy cannot turn a non-empty SAM2 output into an empty mask.
    """

    min_component_pixels: int = 16
    min_component_area_ratio: float = 0.005
    connectivity: Literal[4, 8] = 8

    def __post_init__(self) -> None:
        if isinstance(self.min_component_pixels, bool) or self.min_component_pixels < 0:
            raise ValueError("min_component_pixels must be a non-negative integer.")
        if not isinstance(self.min_component_pixels, int):
            raise TypeError("min_component_pixels must be an integer.")
        if (
            isinstance(self.min_component_area_ratio, bool)
            or not isinstance(self.min_component_area_ratio, (int, float))
            or not math.isfinite(float(self.min_component_area_ratio))
            or not 0.0 <= float(self.min_component_area_ratio) <= 1.0
        ):
            raise ValueError("min_component_area_ratio must be finite and in [0, 1].")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")


@dataclass(frozen=True, slots=True)
class MaskQuality:
    """Auditable quality and cleanup data for one selected SAM2 mask."""

    selected_mask_index: int
    predicted_iou: float
    object_score_logit: float | None
    raw_foreground_pixels: int
    cleaned_foreground_pixels: int
    removed_component_count: int
    removed_foreground_pixels: int

    def __post_init__(self) -> None:
        if self.selected_mask_index < 0:
            raise ValueError("selected_mask_index must be non-negative.")
        if not math.isfinite(self.predicted_iou):
            raise ValueError("predicted_iou must be finite.")
        if self.object_score_logit is not None and not math.isfinite(self.object_score_logit):
            raise ValueError("object_score_logit must be finite when supplied.")
        for field_name in (
            "raw_foreground_pixels",
            "cleaned_foreground_pixels",
            "removed_component_count",
            "removed_foreground_pixels",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class GroundedMaskResult:
    """Exactly one result for one source Grounding DINO bbox.

    ``raw_mask`` and ``cleaned_mask`` are full-frame boolean HxW arrays for
    successful refinement.  They are both ``None`` on a bbox fallback; the
    original ``source_bbox`` remains the usable localization in that case.
    """

    source_index: int
    source_bbox: BBox
    status: RefinementStatus
    raw_mask: np.ndarray | None
    cleaned_mask: np.ndarray | None
    mask_bbox: BBox | None
    quality: MaskQuality | None
    fallback_reason: str | None
    details: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative.")
        if not isinstance(self.source_bbox, BBox):
            raise TypeError("source_bbox must be BBox.")
        if self.status == "valid":
            if (
                self.raw_mask is None
                or self.cleaned_mask is None
                or self.mask_bbox is None
                or self.quality is None
                or self.fallback_reason is not None
            ):
                raise ValueError("Valid refinement requires masks, mask_bbox, and quality.")
        elif self.status == "bbox_fallback":
            if (
                self.raw_mask is not None
                or self.cleaned_mask is not None
                or self.mask_bbox is not None
                or self.quality is not None
                or not self.fallback_reason
            ):
                raise ValueError("BBox fallback requires a reason and no mask payload.")
        else:  # pragma: no cover - Literal is a typing aid, not runtime validation.
            raise ValueError("Unsupported refinement status.")

        for field_name in ("raw_mask", "cleaned_mask"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = np.ascontiguousarray(value, dtype=np.bool_)
            if normalized.ndim != 2:
                raise ValueError(f"{field_name} must be a two-dimensional mask.")
            normalized.flags["WRITEABLE"] = False
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


class SAM2ProcessorLike(Protocol):
    """Small protocol enabling synthetic tests without loading Transformers."""

    def __call__(self, **kwargs: Any) -> Any: ...

    def post_process_masks(self, masks: Any, original_sizes: Any, **kwargs: Any) -> Any: ...


class SAM2ModelLike(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class SAM2BBoxRefiner:
    """Reusable local SAM2 model/processor pair, configured for float32."""

    processor: SAM2ProcessorLike
    model: SAM2ModelLike
    device: Literal["cpu", "cuda"]
    torch_module: Any

    def refine(
        self,
        rgb: np.ndarray,
        boxes: Sequence[BBox],
        *,
        cleanup: MaskCleanupConfig = MaskCleanupConfig(),
    ) -> tuple[GroundedMaskResult, ...]:
        return refine_grounding_boxes(rgb, boxes, refiner=self, cleanup=cleanup)


def load_local_sam2_bbox_refiner(
    model_directory: Path,
    *,
    device: Literal["cpu", "cuda"] = "cuda",
) -> SAM2BBoxRefiner:
    """Load SAM2.1 from one existing local directory without hub fallback."""

    directory = model_directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("model_directory must be a directory.")
    import torch
    from transformers import Sam2Model, Sam2Processor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    processor = Sam2Processor.from_pretrained(str(directory), local_files_only=True)
    model = Sam2Model.from_pretrained(
        str(directory),
        local_files_only=True,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    return SAM2BBoxRefiner(processor=processor, model=model, device=device, torch_module=torch)


def refine_grounding_boxes(
    rgb: np.ndarray,
    boxes: Sequence[BBox],
    *,
    refiner: SAM2BBoxRefiner,
    cleanup: MaskCleanupConfig = MaskCleanupConfig(),
) -> tuple[GroundedMaskResult, ...]:
    """Refine all source boxes for one RGB image in a single SAM2 prompt call.

    Invalid or out-of-frame boxes never reach SAM2 and receive an explicit
    fallback.  A model/processor failure similarly returns one fallback per
    affected box instead of silently shortening the candidate sequence.
    """

    image = _validate_rgb(rgb)
    if not isinstance(refiner, SAM2BBoxRefiner):
        raise TypeError("refiner must be SAM2BBoxRefiner.")
    if not isinstance(cleanup, MaskCleanupConfig):
        raise TypeError("cleanup must be MaskCleanupConfig.")
    source_boxes = tuple(boxes)
    if not all(isinstance(box, BBox) for box in source_boxes):
        raise TypeError("boxes must contain only BBox instances.")
    if not source_boxes:
        return ()

    image_size = ImageSize(width=image.shape[1], height=image.shape[0])
    results: list[GroundedMaskResult | None] = [None] * len(source_boxes)
    valid: list[tuple[int, BBox]] = []
    for index, source_bbox in enumerate(source_boxes):
        clipped = source_bbox.clip_to(image_size)
        if clipped is None:
            results[index] = _fallback(index, source_bbox, "INVALID_BBOX", image_size)
        else:
            valid.append((index, clipped))
    if not valid:
        return tuple(_require_result(item) for item in results)

    try:
        masks, iou_scores, object_scores = _infer_prompted_masks(image, valid, refiner)
    except Exception as error:  # Model implementation failures must preserve candidates.
        reason = f"INFERENCE_FAILED:{type(error).__name__}"
        for index, source_bbox in valid:
            results[index] = _fallback(index, source_bbox, reason, image_size)
        return tuple(_require_result(item) for item in results)

    for batch_index, (source_index, clipped_bbox) in enumerate(valid):
        try:
            raw_options = _mask_options_for_box(masks, batch_index, image_size)
            quality_options = _quality_options_for_box(iou_scores, batch_index)
            if len(raw_options) != len(quality_options):
                raise ValueError("mask/quality option count mismatch")
            selected = _select_quality_index(quality_options)
            raw_mask = raw_options[selected]
            if not np.any(raw_mask):
                raise ValueError("selected mask is empty")
            cleaned_mask, cleanup_details = clean_mask_components(raw_mask, cleanup=cleanup)
            mask_bbox = mask_bbox_from_full_frame(cleaned_mask)
            if mask_bbox is None:
                raise ValueError("cleaned mask is empty")
            object_score = _object_score_for_box(object_scores, batch_index)
            raw_pixels = int(np.count_nonzero(raw_mask))
            cleaned_pixels = int(np.count_nonzero(cleaned_mask))
            results[source_index] = GroundedMaskResult(
                source_index=source_index,
                source_bbox=source_boxes[source_index],
                status="valid",
                raw_mask=raw_mask,
                cleaned_mask=cleaned_mask,
                mask_bbox=mask_bbox,
                quality=MaskQuality(
                    selected_mask_index=selected,
                    predicted_iou=float(quality_options[selected]),
                    object_score_logit=object_score,
                    raw_foreground_pixels=raw_pixels,
                    cleaned_foreground_pixels=cleaned_pixels,
                    removed_component_count=cleanup_details["removed_component_count"],
                    removed_foreground_pixels=cleanup_details["removed_foreground_pixels"],
                ),
                fallback_reason=None,
                details={
                    "prompt_bbox": _bbox_dict(clipped_bbox),
                    "image_size": {"width": image_size.width, "height": image_size.height},
                    "cleanup": cleanup_details,
                    "dtype": "bool",
                },
            )
        except Exception as error:
            results[source_index] = _fallback(
                source_index,
                source_boxes[source_index],
                f"QUALITY_FAILED:{type(error).__name__}",
                image_size,
            )
    return tuple(_require_result(item) for item in results)


def clean_mask_components(
    raw_mask: np.ndarray,
    *,
    cleanup: MaskCleanupConfig = MaskCleanupConfig(),
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Remove only very small disconnected islands and retain the largest one."""

    mask = np.ascontiguousarray(raw_mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("raw_mask must be two-dimensional.")
    if not isinstance(cleanup, MaskCleanupConfig):
        raise TypeError("cleanup must be MaskCleanupConfig.")
    components = _connected_components(mask, connectivity=cleanup.connectivity)
    raw_pixels = int(np.count_nonzero(mask))
    if not components:
        return mask.copy(), {
            "component_count": 0,
            "removed_component_count": 0,
            "removed_foreground_pixels": 0,
            "absolute_threshold_pixels": cleanup.min_component_pixels,
            "relative_threshold_pixels": 0,
        }
    absolute_threshold = cleanup.min_component_pixels
    relative_threshold = int(math.ceil(raw_pixels * cleanup.min_component_area_ratio))
    largest_index = max(range(len(components)), key=lambda index: len(components[index]))
    cleaned = mask.copy()
    removed_count = 0
    removed_pixels = 0
    for index, component in enumerate(components):
        # Both tests must hold. This deliberately preserves a disconnected
        # visible fragment unless it is tiny in absolute and relative terms.
        if (
            index != largest_index
            and len(component) < absolute_threshold
            and len(component) < relative_threshold
        ):
            removed_count += 1
            removed_pixels += len(component)
            for row, column in component:
                cleaned[row, column] = False
    return cleaned, {
        "component_count": len(components),
        "removed_component_count": removed_count,
        "removed_foreground_pixels": removed_pixels,
        "absolute_threshold_pixels": absolute_threshold,
        "relative_threshold_pixels": relative_threshold,
    }


def mask_bbox_from_full_frame(mask: np.ndarray) -> BBox | None:
    """Return the tight half-open bbox of a non-empty full-frame mask."""

    foreground = np.asarray(mask, dtype=np.bool_)
    if foreground.ndim != 2:
        raise ValueError("mask must be two-dimensional.")
    rows, columns = np.nonzero(foreground)
    if len(rows) == 0:
        return None
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return BBox(float(left), float(top), float(right - left), float(bottom - top))


def _infer_prompted_masks(
    image: np.ndarray,
    valid: Sequence[tuple[int, BBox]],
    refiner: SAM2BBoxRefiner,
) -> tuple[Any, Any, Any]:
    from PIL import Image

    xyxy = [[box.left, box.top, box.right, box.bottom] for _, box in valid]
    inputs = refiner.processor(
        images=Image.fromarray(image, mode="RGB"),
        input_boxes=[xyxy],
        return_tensors="pt",
    )
    moved = _move_to_device(inputs, refiner.device)
    with refiner.torch_module.inference_mode():
        output = refiner.model(**moved, multimask_output=True)
    raw_masks = _output_field(output, "pred_masks")
    iou_scores = _output_field(output, "iou_scores")
    original_sizes = _mapping_field(moved, "original_sizes")
    processed = refiner.processor.post_process_masks(
        raw_masks,
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
    )
    if not isinstance(processed, (list, tuple)) or len(processed) != 1:
        raise ValueError("SAM2 post-processing must return one image batch.")
    return processed[0], iou_scores, _optional_output_field(output, "object_score_logits")


def _mask_options_for_box(masks: Any, batch_index: int, image_size: ImageSize) -> list[np.ndarray]:
    array = np.asarray(_to_numpy(masks), dtype=np.bool_)
    if array.ndim != 4 or batch_index >= array.shape[0]:
        raise ValueError("processed SAM2 masks must have shape (boxes, masks, height, width).")
    options = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in array[batch_index]]
    if not options or any(mask.shape != (image_size.height, image_size.width) for mask in options):
        raise ValueError("processed SAM2 mask geometry does not match RGB image.")
    return options


def _quality_options_for_box(scores: Any, batch_index: int) -> np.ndarray:
    array = np.asarray(_to_numpy(scores), dtype=np.float64)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or batch_index >= array.shape[0]:
        raise ValueError("SAM2 iou_scores must have shape (batch, boxes, masks).")
    values = array[batch_index].reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("SAM2 quality has no finite mask score.")
    return values


def _object_score_for_box(scores: Any, batch_index: int) -> float | None:
    if scores is None:
        return None
    array = np.asarray(_to_numpy(scores), dtype=np.float64)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or batch_index >= array.shape[0] or array.shape[1] != 1:
        return None
    value = float(array[batch_index, 0])
    return value if math.isfinite(value) else None


def _select_quality_index(values: np.ndarray) -> int:
    finite = np.where(np.isfinite(values), values, -np.inf)
    index = int(np.argmax(finite))
    if not math.isfinite(float(finite[index])):
        raise ValueError("SAM2 quality has no finite mask score.")
    return index


def _connected_components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=np.bool_)
    if connectivity == 8:
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    components: list[list[tuple[int, int]]] = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        row, column = int(row), int(column)
        if seen[row, column]:
            continue
        seen[row, column] = True
        stack = [(row, column)]
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for offset_row, offset_column in offsets:
                next_row = current_row + offset_row
                next_column = current_column + offset_column
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not seen[next_row, next_column]
                ):
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))
        components.append(component)
    return components


def _validate_rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("rgb must have non-empty shape (height, width, 3).")
    if image.dtype != np.uint8:
        raise TypeError("rgb must use uint8 RGB pixels.")
    return np.ascontiguousarray(image)


def _move_to_device(inputs: Any, device: str) -> Any:
    to_method = getattr(inputs, "to", None)
    return to_method(device) if callable(to_method) else inputs


def _mapping_field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping) and key in value:
        return value[key]
    try:
        return value[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"SAM2 processor output is missing {key!r}.") from error


def _output_field(output: Any, key: str) -> Any:
    value = _optional_output_field(output, key)
    if value is None:
        raise ValueError(f"SAM2 output is missing {key!r}.")
    return value


def _optional_output_field(output: Any, key: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(key)
    return getattr(output, key, None)


def _to_numpy(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    to_cpu = getattr(value, "cpu", None)
    if callable(to_cpu):
        value = to_cpu()
    convert = getattr(value, "numpy", None)
    return convert() if callable(convert) else value


def _fallback(
    source_index: int,
    source_bbox: BBox,
    reason: str,
    image_size: ImageSize,
) -> GroundedMaskResult:
    return GroundedMaskResult(
        source_index=source_index,
        source_bbox=source_bbox,
        status="bbox_fallback",
        raw_mask=None,
        cleaned_mask=None,
        mask_bbox=None,
        quality=None,
        fallback_reason=reason,
        details={"image_size": {"width": image_size.width, "height": image_size.height}},
    )


def _require_result(value: GroundedMaskResult | None) -> GroundedMaskResult:
    if value is None:  # pragma: no cover - defensive invariant guard.
        raise RuntimeError("Internal refinement result was not populated.")
    return value


def _bbox_dict(bbox: BBox) -> dict[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


__all__ = [
    "GroundedMaskResult",
    "MaskCleanupConfig",
    "MaskQuality",
    "RefinementStatus",
    "SAM2BBoxRefiner",
    "clean_mask_components",
    "load_local_sam2_bbox_refiner",
    "mask_bbox_from_full_frame",
    "refine_grounding_boxes",
]
