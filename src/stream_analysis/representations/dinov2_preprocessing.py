"""Deterministic DINOv2 crop preprocessing for canonical F05 candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from PIL import Image

from ..candidates import CandidateMaskRecord
from ..contracts import BBox, CandidateRecord, ValidityStatus
from ..input import DecodedFrame


DINO_BBOX_VARIANT: Final = "bbox_rgb_letterbox_v1"
DINO_MASK_NEUTRAL_VARIANT: Final = "mask_neutral_letterbox_v1"
DINO_INPUT_SIZE: Final = 224
DINO_IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
DINO_IMAGENET_STD: Final = (0.229, 0.224, 0.225)
DINO_NEUTRAL_RGB: Final = (124, 116, 104)
DINO_RGB_INTERPOLATION: Final = "PIL.Image.Resampling.BICUBIC"

DinoV2Variant = Literal["bbox_rgb_letterbox_v1", "mask_neutral_letterbox_v1"]


class DinoV2PreprocessingError(ValueError):
    """Expected per-candidate preprocessing failure with a stable code."""

    def __init__(
        self,
        code: str,
        message: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = MappingProxyType(dict(metadata or {}))


@dataclass(frozen=True, slots=True, eq=False)
class DinoV2PreprocessingResult:
    """One immutable normalized CHW tensor and its diagnostic lineage."""

    normalized_chw: np.ndarray
    letterboxed_rgb: np.ndarray
    model_bbox: BBox
    details: Mapping[str, object] = field(hash=False)

    def __post_init__(self) -> None:
        normalized = np.ascontiguousarray(self.normalized_chw, dtype=np.float32)
        rgb = np.ascontiguousarray(self.letterboxed_rgb, dtype=np.uint8)
        if normalized.ndim != 3 or normalized.shape[0] != 3:
            raise ValueError("normalized_chw must have shape (3, height, width).")
        if rgb.shape != (normalized.shape[1], normalized.shape[2], 3):
            raise ValueError("letterboxed_rgb must align with normalized_chw.")
        if not np.isfinite(normalized).all():
            raise ValueError("normalized_chw must contain only finite values.")
        if not isinstance(self.model_bbox, BBox):
            raise TypeError("model_bbox must be BBox.")
        normalized.flags["WRITEABLE"] = False
        rgb.flags["WRITEABLE"] = False
        object.__setattr__(self, "normalized_chw", normalized)
        object.__setattr__(self, "letterboxed_rgb", rgb)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


def model_bbox_with_context(
    candidate: CandidateRecord,
    context_padding_ratio: float,
) -> BBox:
    """Return a clipped integer model bbox without mutating the candidate bbox."""

    if not isinstance(candidate, CandidateRecord):
        raise TypeError("candidate must be CandidateRecord.")
    if isinstance(context_padding_ratio, bool) or not isinstance(
        context_padding_ratio, (int, float)
    ):
        raise TypeError("context_padding_ratio must be a real number.")
    ratio = float(context_padding_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("context_padding_ratio must be finite and non-negative.")

    bbox = candidate.bbox
    x0 = max(0, math.floor(bbox.x - bbox.width * ratio))
    y0 = max(0, math.floor(bbox.y - bbox.height * ratio))
    x1 = min(
        candidate.frame_size.width,
        math.ceil(bbox.x + bbox.width + bbox.width * ratio),
    )
    y1 = min(
        candidate.frame_size.height,
        math.ceil(bbox.y + bbox.height + bbox.height * ratio),
    )
    if x1 <= x0 or y1 <= y0:
        raise DinoV2PreprocessingError(
            "EMPTY_CROP",
            "Candidate bbox has no model crop area after context padding and clipping.",
        )
    return BBox(x=float(x0), y=float(y0), width=float(x1 - x0), height=float(y1 - y0))


def preprocess_dinov2_candidate(
    frame: DecodedFrame,
    candidate: CandidateRecord,
    *,
    variant: DinoV2Variant,
    input_size: int = DINO_INPUT_SIZE,
    context_padding_ratio: float = 0.0,
    neutral_rgb: tuple[int, int, int] = DINO_NEUTRAL_RGB,
    imagenet_mean: tuple[float, float, float] = DINO_IMAGENET_MEAN,
    imagenet_std: tuple[float, float, float] = DINO_IMAGENET_STD,
    mask_record: CandidateMaskRecord | None = None,
) -> DinoV2PreprocessingResult:
    """Build one aspect-preserving DINOv2 input for a canonical candidate."""

    if not isinstance(frame, DecodedFrame):
        raise TypeError("frame must be DecodedFrame.")
    if not isinstance(candidate, CandidateRecord):
        raise TypeError("candidate must be CandidateRecord.")
    if frame.frame_id != candidate.frame_id or frame.image_size != candidate.frame_size:
        raise ValueError("Candidate and decoded frame identity/geometry must agree.")
    if variant not in {DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT}:
        raise ValueError("Unsupported DINOv2 preprocessing variant.")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("input_size must be a positive integer.")
    fill_rgb = _validate_rgb(neutral_rgb, "neutral_rgb")
    mean = _validate_triplet(imagenet_mean, "imagenet_mean", positive=False)
    std = _validate_triplet(imagenet_std, "imagenet_std", positive=True)

    model_bbox = model_bbox_with_context(candidate, context_padding_ratio)
    frame_rgb = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    )
    x0, y0, width, height = _integral_bbox(model_bbox, "model bbox")
    crop = np.ascontiguousarray(frame_rgb[y0:y0 + height, x0:x0 + width])
    if crop.shape != (height, width, 3):
        raise DinoV2PreprocessingError(
            "EMPTY_CROP",
            "Model crop does not match the clipped model bbox.",
        )

    mask_details: dict[str, object] = {}
    if variant == DINO_MASK_NEUTRAL_VARIANT:
        expanded_mask = _expanded_required_mask(
            candidate,
            mask_record,
            model_bbox,
        )
        if not np.any(expanded_mask):
            raise DinoV2PreprocessingError(
                "MASK_EMPTY",
                "Mask-neutral DINOv2 preprocessing requires non-empty foreground.",
            )
        working = crop.copy()
        working[~expanded_mask] = np.asarray(fill_rgb, dtype=np.uint8)
        letterbox_fill = fill_rgb
        fill_source = "imagenet_mean_neutral"
        assert mask_record is not None
        mask_details = {
            "mask_ref": mask_record.mask_ref,
            "mask_digest": mask_record.mask_digest,
            "mask_policy": "mask_required",
            "foreground_pixel_count": int(np.count_nonzero(expanded_mask)),
        }
    else:
        working = crop
        letterbox_fill = _border_median_rgb(crop)
        fill_source = "bbox_border_median"
        mask_details = {"mask_policy": "bbox_only"}

    letterboxed, resized_size, offsets = _letterbox_bicubic(
        working,
        input_size,
        letterbox_fill,
    )
    normalized = letterboxed.astype(np.float32) / np.float32(255.0)
    normalized = (normalized - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std,
        dtype=np.float32,
    )
    normalized_chw = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
    details: dict[str, object] = {
        "variant": variant,
        "candidate_bbox": _bbox_dict(candidate.bbox),
        "model_bbox": _bbox_dict(model_bbox),
        "context_padding_ratio": float(context_padding_ratio),
        "source_crop_size": {"width": width, "height": height},
        "input_size": input_size,
        "resized_content_size": {
            "width": resized_size[0],
            "height": resized_size[1],
        },
        "letterbox_offset": {"x": offsets[0], "y": offsets[1]},
        "letterbox_fill_source": fill_source,
        "letterbox_fill_rgb": letterbox_fill,
        "rgb_interpolation": DINO_RGB_INTERPOLATION,
        "color_mode": "RGB",
        "input_dtype": "uint8",
        "output_dtype": "float32",
        "normalization": "imagenet_mean_std",
        "imagenet_mean": mean,
        "imagenet_std": std,
        **mask_details,
    }
    return DinoV2PreprocessingResult(
        normalized_chw=normalized_chw,
        letterboxed_rgb=letterboxed,
        model_bbox=model_bbox,
        details=details,
    )


def _expanded_required_mask(
    candidate: CandidateRecord,
    mask_record: CandidateMaskRecord | None,
    model_bbox: BBox,
) -> np.ndarray:
    if candidate.mask is None or candidate.mask.validity_status is not ValidityStatus.VALID:
        raise DinoV2PreprocessingError(
            "MASK_REQUIRED",
            "Mask-neutral DINOv2 preprocessing requires a valid candidate mask.",
        )
    if mask_record is None:
        raise DinoV2PreprocessingError(
            "MASK_REQUIRED",
            "Mask-neutral DINOv2 preprocessing requires an in-memory mask record.",
        )
    if mask_record.candidate_id != candidate.candidate_id:
        raise DinoV2PreprocessingError("MASK_CANDIDATE_MISMATCH", "Mask candidate_id mismatch.")
    if mask_record.frame_id != candidate.frame_id:
        raise DinoV2PreprocessingError("MASK_FRAME_MISMATCH", "Mask frame_id mismatch.")
    if mask_record.coordinate_bbox != candidate.bbox:
        raise DinoV2PreprocessingError("MASK_BBOX_MISMATCH", "Mask bbox mismatch.")
    if mask_record.mask_digest != candidate.mask.mask_digest:
        raise DinoV2PreprocessingError("MASK_DIGEST_MISMATCH", "Mask digest mismatch.")

    model_x, model_y, model_width, model_height = _integral_bbox(model_bbox, "model bbox")
    cand_x, cand_y, cand_width, cand_height = _integral_bbox(candidate.bbox, "candidate bbox")
    mask = np.asarray(mask_record.mask, dtype=np.bool_)
    if mask.shape != (cand_height, cand_width):
        raise DinoV2PreprocessingError(
            "INVALID_MASK_SHAPE",
            "Candidate mask shape does not match the canonical bbox.",
            {"mask_shape": tuple(mask.shape), "bbox_size": (cand_height, cand_width)},
        )
    offset_x = cand_x - model_x
    offset_y = cand_y - model_y
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + cand_width > model_width
        or offset_y + cand_height > model_height
    ):
        raise DinoV2PreprocessingError(
            "MASK_BBOX_MISMATCH",
            "Canonical candidate mask does not fit inside the model bbox.",
        )
    expanded = np.zeros((model_height, model_width), dtype=np.bool_)
    expanded[offset_y:offset_y + cand_height, offset_x:offset_x + cand_width] = mask
    return expanded


def _letterbox_bicubic(
    crop: np.ndarray,
    input_size: int,
    fill_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise DinoV2PreprocessingError("EMPTY_CROP", "Letterbox received an empty crop.")
    scale = min(input_size / float(width), input_size / float(height))
    new_width = min(input_size, max(1, int(round(width * scale))))
    new_height = min(input_size, max(1, int(round(height * scale))))
    image = Image.fromarray(crop, mode="RGB")
    resized = np.asarray(
        image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    square = np.empty((input_size, input_size, 3), dtype=np.uint8)
    square[:, :] = np.asarray(fill_rgb, dtype=np.uint8)
    offset_x = (input_size - new_width) // 2
    offset_y = (input_size - new_height) // 2
    square[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    return square, (new_width, new_height), (offset_x, offset_y)


def _border_median_rgb(crop: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate(
        (crop[0, :, :], crop[-1, :, :], crop[:, 0, :], crop[:, -1, :]),
        axis=0,
    )
    values = np.rint(np.median(border.astype(np.float64), axis=0)).astype(np.uint8)
    return tuple(int(value) for value in values)


def _integral_bbox(bbox: BBox, field_name: str) -> tuple[int, int, int, int]:
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    integers = tuple(int(value) for value in values)
    if any(float(integer) != float(value) for integer, value in zip(integers, values, strict=True)):
        raise DinoV2PreprocessingError(
            "INVALID_BBOX",
            f"{field_name} must have integer-valued coordinates and extents.",
        )
    if integers[2] <= 0 or integers[3] <= 0:
        raise DinoV2PreprocessingError("INVALID_BBOX", f"{field_name} must be positive.")
    return integers


def _validate_rgb(value: tuple[int, int, int], field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three integer channels.")
    if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in value):
        raise TypeError(f"{field_name} channels must be integers.")
    if any(channel < 0 or channel > 255 for channel in value):
        raise ValueError(f"{field_name} channels must be in [0, 255].")
    return value


def _validate_triplet(
    value: tuple[float, float, float],
    field_name: str,
    *,
    positive: bool,
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three values.")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field_name} values must be real numbers.")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0.0):
            raise ValueError(f"{field_name} values must be finite{', positive' if positive else ''}.")
        converted.append(number)
    return tuple(converted)  # type: ignore[return-value]


def _bbox_dict(bbox: BBox) -> Mapping[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


__all__ = [
    "DINO_BBOX_VARIANT",
    "DINO_IMAGENET_MEAN",
    "DINO_IMAGENET_STD",
    "DINO_INPUT_SIZE",
    "DINO_MASK_NEUTRAL_VARIANT",
    "DINO_NEUTRAL_RGB",
    "DINO_RGB_INTERPOLATION",
    "DinoV2PreprocessingError",
    "DinoV2PreprocessingResult",
    "DinoV2Variant",
    "model_bbox_with_context",
    "preprocess_dinov2_candidate",
]
