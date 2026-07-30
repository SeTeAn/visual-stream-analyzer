"""Dataset-agnostic review policy for fragmented binary instance masks.

This module belongs to the annotation/evaluation boundary.  It intentionally
contains no file I/O and makes no semantic or visual-type assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import struct
from typing import Iterable

import numpy as np


POLICY_ID = "largest_plus_reviewed_components_v1"
DEFAULT_SECONDARY_AREA_RATIO = 0.05
BINARY_MASK_DIGEST_ENCODING = "bool-mask-v1:shape-u64be+row-major-packbits-big"


class ComponentReviewError(ValueError):
    """Raised when a mask or a component-review decision is invalid."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned pixel box using an inclusive origin and exclusive extent."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Exclusive right coordinate."""

        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom coordinate."""

        return self.y + self.height

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class MaskComponent:
    """Deterministically identified 4-connected component."""

    component_id: str
    area: int
    bbox: BoundingBox
    centroid: tuple[float, float]
    border_touch: bool
    binary_mask_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.component_id,
            "area": self.area,
            "bbox": self.bbox.as_dict(),
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]},
            "border_touch": self.border_touch,
            "binary_mask_sha256": self.binary_mask_sha256,
        }


@dataclass(frozen=True, slots=True)
class ComponentAnalysis:
    """Component evidence and the automatic proposal for one source mask."""

    policy_id: str
    source_mask_sha256: str
    mask_shape: tuple[int, int]
    raw_bbox: BoundingBox
    largest_bbox: BoundingBox
    components: tuple[MaskComponent, ...]
    automatic_retained_component_ids: tuple[str, ...]
    review_required: bool
    secondary_area_ratio: float
    label_map: np.ndarray = field(repr=False, compare=False)

    def component_ids(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.components)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible part of the analysis evidence."""

        return {
            "policy_id": self.policy_id,
            "source_mask_sha256": self.source_mask_sha256,
            "mask_shape": list(self.mask_shape),
            "raw_bbox": self.raw_bbox.as_dict(),
            "largest_bbox": self.largest_bbox.as_dict(),
            "components": [component.as_dict() for component in self.components],
            "automatic_retained_component_ids": list(
                self.automatic_retained_component_ids
            ),
            "review_required": self.review_required,
            "secondary_area_ratio": self.secondary_area_ratio,
            "connectivity": 4,
            "binary_mask_digest_encoding": BINARY_MASK_DIGEST_ENCODING,
        }


@dataclass(frozen=True, slots=True)
class ComponentReviewDecision:
    """Auditable manual decision tied to an exact source binary mask."""

    source_mask_sha256: str
    retained_component_ids: tuple[str, ...]
    reason: str | None = None
    policy_id: str = POLICY_ID

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "source_mask_sha256": self.source_mask_sha256,
            "retained_component_ids": list(self.retained_component_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FinalizedComponentMask:
    """Validated retained-component union and its final bounding box."""

    policy_id: str
    source_mask_sha256: str
    retained_component_ids: tuple[str, ...]
    final_bbox: BoundingBox
    final_mask: np.ndarray = field(repr=False, compare=False)
    used_manual_decision: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "source_mask_sha256": self.source_mask_sha256,
            "retained_component_ids": list(self.retained_component_ids),
            "final_bbox": self.final_bbox.as_dict(),
            "used_manual_decision": self.used_manual_decision,
        }


@dataclass(frozen=True, slots=True)
class _RawComponent:
    coordinates: tuple[tuple[int, int], ...]
    area: int
    bbox: BoundingBox
    centroid: tuple[float, float]
    border_touch: bool
    binary_mask_sha256: str


def binary_mask_sha256(mask: np.ndarray) -> str:
    """Hash a 2-D bool mask with its shape using a canonical packed encoding."""

    checked = _validate_mask(mask, require_nonempty=False)
    height, width = checked.shape
    packed = np.packbits(checked.reshape(-1), bitorder="big").tobytes()
    payload = b"bool-mask-v1\0" + struct.pack(">QQ", height, width) + packed
    return sha256(payload).hexdigest()


def analyze_component_mask(
    mask: np.ndarray,
    *,
    secondary_area_ratio: float = DEFAULT_SECONDARY_AREA_RATIO,
) -> ComponentAnalysis:
    """Analyze one non-empty binary mask under the configured review policy."""

    checked = _validate_mask(mask, require_nonempty=True)
    if isinstance(secondary_area_ratio, bool) or not isinstance(
        secondary_area_ratio, (int, float)
    ):
        raise ComponentReviewError("secondary_area_ratio must be numeric.")
    ratio = float(secondary_area_ratio)
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ComponentReviewError(
            "secondary_area_ratio must be finite and within [0, 1]."
        )

    raw_components = _find_components(checked)
    raw_components.sort(
        key=lambda item: (
            -item.area,
            item.bbox.y,
            item.bbox.x,
            item.bbox.bottom,
            item.bbox.right,
            item.binary_mask_sha256,
            item.coordinates,
        )
    )

    label_map = np.zeros(checked.shape, dtype=np.int32)
    components: list[MaskComponent] = []
    for index, raw in enumerate(raw_components, start=1):
        component_id = f"c{index:03d}"
        ys = np.fromiter((point[0] for point in raw.coordinates), dtype=np.int64)
        xs = np.fromiter((point[1] for point in raw.coordinates), dtype=np.int64)
        label_map[ys, xs] = index
        components.append(
            MaskComponent(
                component_id=component_id,
                area=raw.area,
                bbox=raw.bbox,
                centroid=raw.centroid,
                border_touch=raw.border_touch,
                binary_mask_sha256=raw.binary_mask_sha256,
            )
        )

    label_map.setflags(write=False)
    largest_area = components[0].area
    automatic = tuple(
        component.component_id
        for component in components
        if component.component_id == "c001"
        or component.area / largest_area >= ratio
    )
    raw_bbox = _bbox_from_mask(checked)
    largest_bbox = components[0].bbox
    return ComponentAnalysis(
        policy_id=POLICY_ID,
        source_mask_sha256=binary_mask_sha256(checked),
        mask_shape=checked.shape,
        raw_bbox=raw_bbox,
        largest_bbox=largest_bbox,
        components=tuple(components),
        automatic_retained_component_ids=automatic,
        review_required=raw_bbox != largest_bbox,
        secondary_area_ratio=ratio,
        label_map=label_map,
    )


def component_union_mask(
    analysis: ComponentAnalysis, component_ids: Iterable[str]
) -> np.ndarray:
    """Return a read-only union mask after validating component IDs."""

    retained = _validate_component_ids(analysis, tuple(component_ids))
    numeric_ids = np.fromiter(
        (int(component_id[1:]) for component_id in retained), dtype=np.int32
    )
    union = np.isin(analysis.label_map, numeric_ids)
    union.setflags(write=False)
    return union


def finalize_component_review(
    analysis: ComponentAnalysis,
    decision: ComponentReviewDecision | None = None,
) -> FinalizedComponentMask:
    """Validate a review decision and derive the final component union."""

    if analysis.policy_id != POLICY_ID:
        raise ComponentReviewError(
            f"Unsupported component policy {analysis.policy_id!r}."
        )
    if analysis.review_required and decision is None:
        raise ComponentReviewError(
            "A manual decision is required because fragments change the raw bbox."
        )

    if decision is None:
        retained = analysis.automatic_retained_component_ids
    else:
        if decision.policy_id != analysis.policy_id:
            raise ComponentReviewError("Decision policy_id does not match the analysis.")
        if decision.source_mask_sha256 != analysis.source_mask_sha256:
            raise ComponentReviewError(
                "Decision source-mask sha256 does not match the analyzed mask."
            )
        retained = _validate_component_ids(
            analysis, decision.retained_component_ids
        )

    final_mask = component_union_mask(analysis, retained)
    return FinalizedComponentMask(
        policy_id=analysis.policy_id,
        source_mask_sha256=analysis.source_mask_sha256,
        retained_component_ids=retained,
        final_bbox=_bbox_from_mask(final_mask),
        final_mask=final_mask,
        used_manual_decision=decision is not None,
    )


def _validate_mask(mask: np.ndarray, *, require_nonempty: bool) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise ComponentReviewError("mask must be a numpy.ndarray.")
    if mask.ndim != 2:
        raise ComponentReviewError("mask must be two-dimensional.")
    if mask.dtype != np.bool_:
        raise ComponentReviewError("mask dtype must be bool.")
    if require_nonempty and not bool(mask.any()):
        raise ComponentReviewError("mask must contain at least one foreground pixel.")
    return np.ascontiguousarray(mask)


def _find_components(mask: np.ndarray) -> list[_RawComponent]:
    height, width = mask.shape
    remaining = set(int(index) for index in np.flatnonzero(mask))
    result: list[_RawComponent] = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        flat_coordinates: list[int] = []
        while stack:
            flat_index = stack.pop()
            flat_coordinates.append(flat_index)
            y, x = divmod(flat_index, width)
            neighbors: list[int] = []
            if y > 0:
                neighbors.append(flat_index - width)
            if x > 0:
                neighbors.append(flat_index - 1)
            if x + 1 < width:
                neighbors.append(flat_index + 1)
            if y + 1 < height:
                neighbors.append(flat_index + width)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        ordered_flat = np.asarray(sorted(flat_coordinates), dtype=np.int64)
        ys, xs = np.divmod(ordered_flat, width)
        ordered = tuple((int(y), int(x)) for y, x in zip(ys, xs, strict=True))
        component_mask = np.zeros(mask.shape, dtype=np.bool_)
        component_mask[ys, xs] = True
        bbox = _bbox_from_coordinates(ys, xs)
        result.append(
            _RawComponent(
                coordinates=ordered,
                area=len(ordered),
                bbox=bbox,
                centroid=(float(xs.mean()), float(ys.mean())),
                border_touch=bool(
                    (ys == 0).any()
                    or (ys == height - 1).any()
                    or (xs == 0).any()
                    or (xs == width - 1).any()
                ),
                binary_mask_sha256=binary_mask_sha256(component_mask),
            )
        )
    return result


def _validate_component_ids(
    analysis: ComponentAnalysis, component_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if not component_ids:
        raise ComponentReviewError("At least c001 must be retained.")
    if len(set(component_ids)) != len(component_ids):
        raise ComponentReviewError("Retained component IDs must be unique.")
    known = set(analysis.component_ids())
    unknown = [item for item in component_ids if item not in known]
    if unknown:
        raise ComponentReviewError(
            f"Unknown retained component IDs: {', '.join(unknown)}."
        )
    if "c001" not in component_ids:
        raise ComponentReviewError("Largest component c001 must be retained.")
    return tuple(
        component.component_id
        for component in analysis.components
        if component.component_id in component_ids
    )


def _bbox_from_mask(mask: np.ndarray) -> BoundingBox:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ComponentReviewError("Cannot derive a bbox from an empty mask.")
    return _bbox_from_coordinates(ys, xs)


def _bbox_from_coordinates(ys: np.ndarray, xs: np.ndarray) -> BoundingBox:
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


__all__ = [
    "BINARY_MASK_DIGEST_ENCODING",
    "DEFAULT_SECONDARY_AREA_RATIO",
    "POLICY_ID",
    "BoundingBox",
    "ComponentAnalysis",
    "ComponentReviewDecision",
    "ComponentReviewError",
    "FinalizedComponentMask",
    "MaskComponent",
    "analyze_component_mask",
    "binary_mask_sha256",
    "component_union_mask",
    "finalize_component_review",
]
