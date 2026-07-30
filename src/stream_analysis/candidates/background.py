"""Background and mask primitives for controlled smooth RGB scenes.

This module stops at in-memory background, residual, and mask operations. It
does not create candidate records, stable IDs, bounding boxes, diagnostics, or
evaluation outputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


BackgroundAggregation = Literal["median", "mean"]
MorphologyKernelShape = Literal["rect", "ellipse", "cross"]


@dataclass(frozen=True, slots=True)
class BackgroundModelConfig:
    """Explicit parameters for estimating a per-stream RGB background model."""

    aggregation: BackgroundAggregation = "median"
    smoothing_kernel_size: int = 0
    smoothing_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be 'median' or 'mean'.")
        _validate_optional_odd_kernel_size(
            self.smoothing_kernel_size,
            "smoothing_kernel_size",
        )
        sigma = _finite_float(self.smoothing_sigma, "smoothing_sigma", minimum=0.0)
        object.__setattr__(self, "smoothing_sigma", sigma)
        if self.smoothing_kernel_size == 0 and sigma != 0.0:
            raise ValueError("smoothing_sigma must be 0.0 when smoothing is disabled.")
        if self.smoothing_kernel_size > 0 and sigma <= 0.0:
            raise ValueError("smoothing_sigma must be positive when smoothing is enabled.")


@dataclass(frozen=True, slots=True, eq=False)
class BackgroundModel:
    """Immutable RGB background estimate and its Lab representation."""

    rgb: np.ndarray
    frame_count: int
    config: BackgroundModelConfig = field(compare=False)
    lab: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise TypeError("frame_count must be an integer.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        if not isinstance(self.config, BackgroundModelConfig):
            raise TypeError("config must be BackgroundModelConfig.")
        rgb = _validate_rgb_frame(self.rgb, "rgb")
        lab = _rgb_float32_to_lab(rgb)
        object.__setattr__(self, "rgb", _readonly_float32(rgb))
        object.__setattr__(self, "lab", _readonly_float32(lab))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self.rgb.shape  # type: ignore[return-value]

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True, slots=True)
class ResidualMaskConfig:
    """Explicit single-threshold residual mask parameters."""

    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "threshold",
            _finite_float(self.threshold, "threshold", minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class HysteresisMaskConfig:
    """Weak/strong residual thresholds for connected hysteresis masking."""

    weak_threshold: float
    strong_threshold: float
    connectivity: Literal[4, 8] = 8

    def __post_init__(self) -> None:
        weak = _finite_float(self.weak_threshold, "weak_threshold", minimum=0.0)
        strong = _finite_float(self.strong_threshold, "strong_threshold", minimum=0.0)
        if weak > strong:
            raise ValueError("weak_threshold must be less than or equal to strong_threshold.")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8.")
        object.__setattr__(self, "weak_threshold", weak)
        object.__setattr__(self, "strong_threshold", strong)


@dataclass(frozen=True, slots=True)
class MorphologyCleanupConfig:
    """Explicit morphology cleanup parameters for boolean foreground masks."""

    open_kernel_size: int = 0
    close_kernel_size: int = 0
    kernel_shape: MorphologyKernelShape = "ellipse"
    iterations: int = 1
    fill_holes: bool = False

    def __post_init__(self) -> None:
        _validate_optional_odd_kernel_size(self.open_kernel_size, "open_kernel_size")
        _validate_optional_odd_kernel_size(self.close_kernel_size, "close_kernel_size")
        if self.kernel_shape not in {"rect", "ellipse", "cross"}:
            raise ValueError("kernel_shape must be 'rect', 'ellipse' or 'cross'.")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if not isinstance(self.fill_holes, bool):
            raise TypeError("fill_holes must be bool.")


def estimate_background_model(
    frames: Iterable[np.ndarray],
    config: BackgroundModelConfig | None = None,
) -> BackgroundModel:
    """Estimate a deterministic RGB background model from same-size frames."""

    effective_config = config or BackgroundModelConfig()
    if not isinstance(effective_config, BackgroundModelConfig):
        raise TypeError("config must be BackgroundModelConfig or None.")
    if isinstance(frames, (np.ndarray, str, bytes)):
        raise TypeError("frames must be an iterable of RGB arrays, not a single array.")
    try:
        supplied_frames = tuple(frames)
    except TypeError as error:
        raise TypeError("frames must be an iterable of RGB arrays.") from error
    if not supplied_frames:
        raise ValueError("frames must contain at least one RGB array.")

    validated = tuple(
        _validate_rgb_frame(frame, f"frames[{index}]")
        for index, frame in enumerate(supplied_frames)
    )
    reference_shape = validated[0].shape
    for index, frame in enumerate(validated[1:], start=1):
        if frame.shape != reference_shape:
            raise ValueError(
                f"frames[{index}] shape must match the first frame shape {reference_shape}."
            )

    stack = np.stack(validated, axis=0)
    if effective_config.aggregation == "median":
        background = np.median(stack, axis=0).astype(np.float32)
    else:
        background = np.mean(stack, axis=0, dtype=np.float32).astype(np.float32)

    if effective_config.smoothing_kernel_size > 0:
        kernel = (effective_config.smoothing_kernel_size,) * 2
        background = cv2.GaussianBlur(
            background,
            kernel,
            sigmaX=effective_config.smoothing_sigma,
            sigmaY=effective_config.smoothing_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ).astype(np.float32)

    return BackgroundModel(
        rgb=background,
        frame_count=len(validated),
        config=effective_config,
    )


def compute_residual(frame: np.ndarray, background: BackgroundModel) -> np.ndarray:
    """Return a 2D ``float32`` Lab-space residual image for one RGB frame."""

    if not isinstance(background, BackgroundModel):
        raise TypeError("background must be BackgroundModel.")
    rgb = _validate_rgb_frame(frame, "frame")
    if rgb.shape != background.rgb.shape:
        raise ValueError("frame shape must match the background model shape.")
    frame_lab = _rgb_float32_to_lab(rgb)
    diff = frame_lab - background.lab
    residual = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32)).astype(np.float32)
    return _readonly_float32(residual)


def threshold_residual_mask(
    residual: np.ndarray,
    config: ResidualMaskConfig,
) -> np.ndarray:
    """Threshold a residual image into an unambiguous boolean mask."""

    if not isinstance(config, ResidualMaskConfig):
        raise TypeError("config must be ResidualMaskConfig.")
    residual_image = _validate_residual_image(residual, "residual")
    return _readonly_bool(residual_image >= config.threshold)


def hysteresis_threshold_mask(
    residual: np.ndarray,
    config: HysteresisMaskConfig,
) -> np.ndarray:
    """Keep weak residual pixels only when connected to a strong residual region."""

    if not isinstance(config, HysteresisMaskConfig):
        raise TypeError("config must be HysteresisMaskConfig.")
    residual_image = _validate_residual_image(residual, "residual")
    weak = residual_image >= config.weak_threshold
    strong = residual_image >= config.strong_threshold
    current = np.logical_and(strong, weak).astype(np.uint8)
    if not current.any():
        return _readonly_bool(np.zeros(residual_image.shape, dtype=np.bool_))

    weak_uint8 = weak.astype(np.uint8)
    kernel = _connectivity_kernel(config.connectivity)
    while True:
        expanded = cv2.dilate(current, kernel, iterations=1)
        next_mask = np.logical_and(expanded > 0, weak_uint8 > 0).astype(np.uint8)
        if np.array_equal(next_mask, current):
            return _readonly_bool(next_mask > 0)
        current = next_mask


def cleanup_morphology(
    mask: np.ndarray,
    config: MorphologyCleanupConfig,
) -> np.ndarray:
    """Apply deterministic morphology cleanup to a boolean foreground mask."""

    if not isinstance(config, MorphologyCleanupConfig):
        raise TypeError("config must be MorphologyCleanupConfig.")
    current = _validate_bool_mask(mask, "mask").astype(np.uint8)

    if config.open_kernel_size > 0:
        current = cv2.morphologyEx(
            current,
            cv2.MORPH_OPEN,
            _morphology_kernel(config.open_kernel_size, config.kernel_shape),
            iterations=config.iterations,
        )
    if config.close_kernel_size > 0:
        current = cv2.morphologyEx(
            current,
            cv2.MORPH_CLOSE,
            _morphology_kernel(config.close_kernel_size, config.kernel_shape),
            iterations=config.iterations,
        )
    if config.fill_holes:
        current = _fill_holes(current)

    return _readonly_bool(current > 0)


def _validate_rgb_frame(value: np.ndarray, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy ndarray.")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{field_name} must have shape (height, width, 3).")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{field_name} height and width must be positive.")
    if value.dtype == np.dtype(np.uint8):
        converted = value.astype(np.float32, copy=True)
    elif value.dtype in {np.dtype(np.float32), np.dtype(np.float64)}:
        if not np.isfinite(value).all():
            raise ValueError(f"{field_name} must contain only finite values.")
        if np.any(value < 0.0) or np.any(value > 255.0):
            raise ValueError(f"{field_name} float values must be within [0, 255].")
        converted = value.astype(np.float32, copy=True)
    else:
        raise TypeError(f"{field_name} dtype must be uint8, float32 or float64.")
    return np.ascontiguousarray(converted, dtype=np.float32)


def _validate_residual_image(value: np.ndarray, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy ndarray.")
    if value.ndim != 2:
        raise ValueError(f"{field_name} must have shape (height, width).")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{field_name} height and width must be positive.")
    if value.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError(f"{field_name} dtype must be float32 or float64.")
    if not np.isfinite(value).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    if np.any(value < 0.0):
        raise ValueError(f"{field_name} values must be non-negative.")
    return np.ascontiguousarray(value.astype(np.float32, copy=True))


def _validate_bool_mask(value: np.ndarray, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy ndarray.")
    if value.ndim != 2:
        raise ValueError(f"{field_name} must have shape (height, width).")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{field_name} height and width must be positive.")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field_name} dtype must be bool.")
    return np.ascontiguousarray(value.astype(np.bool_, copy=True))


def _finite_float(
    value: int | float,
    field_name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return converted


def _validate_optional_odd_kernel_size(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    if value > 0 and value % 2 == 0:
        raise ValueError(f"{field_name} must be odd when enabled.")


def _rgb_float32_to_lab(rgb: np.ndarray) -> np.ndarray:
    normalized = np.ascontiguousarray(rgb.astype(np.float32, copy=True) / 255.0)
    lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    return np.ascontiguousarray(lab.astype(np.float32, copy=False))


def _connectivity_kernel(connectivity: Literal[4, 8]) -> np.ndarray:
    if connectivity == 4:
        return np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
    return np.ones((3, 3), dtype=np.uint8)


def _morphology_kernel(size: int, shape: MorphologyKernelShape) -> np.ndarray:
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shapes[shape], (size, size))


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(binary_mask.astype(np.uint8, copy=True))
    background = (result == 0).astype(np.uint8)
    height, width = background.shape
    seeds = [(x, 0) for x in range(width)]
    seeds.extend((x, height - 1) for x in range(width))
    seeds.extend((0, y) for y in range(height))
    seeds.extend((width - 1, y) for y in range(height))

    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for x, y in seeds:
        if background[y, x] == 1:
            flood_mask.fill(0)
            cv2.floodFill(background, flood_mask, (x, y), 2)
    result[background == 1] = 1
    return result


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value.astype(np.float32, copy=True))
    result.setflags(write=False)
    return result


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value.astype(np.bool_, copy=True))
    result.setflags(write=False)
    return result


__all__ = [
    "BackgroundAggregation",
    "BackgroundModel",
    "BackgroundModelConfig",
    "HysteresisMaskConfig",
    "MorphologyCleanupConfig",
    "MorphologyKernelShape",
    "ResidualMaskConfig",
    "cleanup_morphology",
    "compute_residual",
    "estimate_background_model",
    "hysteresis_threshold_mask",
    "threshold_residual_mask",
]
