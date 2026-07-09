"""Finite half-open geometry contracts used throughout the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_float(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_float(self.x, "x"))
        object.__setattr__(self, "y", _finite_float(self.y, "y"))


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("width must be an integer.")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("height must be an integer.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive.")

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned half-open box: ``[left, right) x [top, bottom)``."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        x = _finite_float(self.x, "x")
        y = _finite_float(self.y, "y")
        width = _finite_float(self.width, "width")
        height = _finite_float(self.height, "height")
        if width <= 0 or height <= 0:
            raise ValueError("BBox width and height must be positive.")
        if not math.isfinite(x + width) or not math.isfinite(y + height):
            raise ValueError("BBox right and bottom coordinates must be finite.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def contains_point(self, point: Point) -> bool:
        if not isinstance(point, Point):
            raise TypeError("point must be Point.")
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def is_within(self, image_size: ImageSize) -> bool:
        if not isinstance(image_size, ImageSize):
            raise TypeError("image_size must be ImageSize.")
        return (
            self.left >= 0
            and self.top >= 0
            and self.right <= image_size.width
            and self.bottom <= image_size.height
        )

    def intersection(self, other: BBox) -> BBox | None:
        if not isinstance(other, BBox):
            raise TypeError("other must be BBox.")
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return BBox(left, top, right - left, bottom - top)

    def clip_to(self, image_size: ImageSize) -> BBox | None:
        if not isinstance(image_size, ImageSize):
            raise TypeError("image_size must be ImageSize.")
        frame_bounds = BBox(0.0, 0.0, float(image_size.width), float(image_size.height))
        return self.intersection(frame_bounds)
