"""Compact, public contracts for the published analysis result.

The records in this module are independent of model-stage records. They permit
only public stream metadata, resolved visual types, and cleaned full-frame
masks. Candidate IDs, scores, confidence values, and model-specific
intermediate data have no field through which to enter the product result.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
from numpy.typing import NDArray

from .geometry import ImageSize


PRODUCT_RESULT_SCHEMA_VERSION = "visual-stream-product-result-1.0"
PUBLIC_PIPELINE_MODELS = ("Grounding DINO", "SAM2", "DINOv2")

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TYPE_ID_PATTERN = re.compile(r"^type_[A-Za-z0-9][A-Za-z0-9._:-]*$")
_OBJECT_ID_PATTERN = re.compile(r"^P[0-9]{2,}$")
_SHA256_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9",
}


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty portable identifier.")
    return value


def _require_portable_frame_id(value: str) -> str:
    _require_identifier(value, "frame_id")
    if value.endswith(".") or value.upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("frame_id must be portable as a directory and file name.")
    return value


def _require_type_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("type_id must be a string.")
    if not _TYPE_ID_PATTERN.fullmatch(value):
        raise ValueError("type_id must start with 'type_' and be a portable identifier.")
    return value


def _require_object_id(value: str, field_name: str = "object_id") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not _OBJECT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must use the public Pnn identifier format.")
    return value


def _require_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a SHA-256 hex digest, optionally prefixed by sha256:."
        )
    return value.lower()


def _require_version(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must be a non-empty token without whitespace.")
    return value


def _relative_posix_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("source_image must be a string.")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError("source_image must be a non-empty relative POSIX path.")
    windows_path = PureWindowsPath(value)
    if (
        "\\" in value
        or PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError("source_image must be a relative POSIX path.")
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(
            "source_image must not contain current-directory or parent traversal segments."
        )
    return value


def _canonical_binary_mask(value: object) -> NDArray[np.bool_]:
    if not isinstance(value, np.ndarray):
        raise TypeError("mask must be a numpy ndarray.")
    if value.ndim != 2:
        raise ValueError("mask must be a two-dimensional full-frame array.")
    if value.dtype != np.bool_ and not np.issubdtype(value.dtype, np.integer):
        raise TypeError("mask must have boolean or integer dtype.")
    if value.size == 0:
        raise ValueError("mask must not be empty.")
    if value.dtype != np.bool_ and not np.all((value == 0) | (value == 1)):
        raise ValueError("integer mask values must be binary (0 or 1).")
    canonical = np.array(value, dtype=np.bool_, copy=True, order="C")
    if not np.any(canonical):
        raise ValueError("mask must contain at least one foreground pixel.")
    canonical.setflags(write=False)
    return canonical


def _mask_sort_key(mask: "ProductMask") -> tuple[str, int, int, int, bytes]:
    rows, columns = np.nonzero(mask.mask)
    digest = hashlib.sha256(mask.mask.tobytes(order="C")).digest()
    return (mask.type_id, int(rows.min()), int(columns.min()), int(mask.mask.sum()), digest)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductStream:
    """Public identity and reproducibility metadata of the analysed stream."""

    stream_id: str
    input_schema_version: str
    manifest_sha256: str
    frame_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.stream_id, "stream_id")
        _require_version(self.input_schema_version, "input_schema_version")
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(self.manifest_sha256, "manifest_sha256"),
        )
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise TypeError("frame_count must be an integer.")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductPipeline:
    """Public, score-free identity of the analysis profile and models."""

    profile_id: str
    profile_sha256: str
    models: tuple[str, ...] = PUBLIC_PIPELINE_MODELS

    def __post_init__(self) -> None:
        _require_identifier(self.profile_id, "profile_id")
        object.__setattr__(
            self,
            "profile_sha256",
            _require_sha256(self.profile_sha256, "profile_sha256"),
        )
        if isinstance(self.models, (str, bytes)):
            raise TypeError("models must be an iterable of public model names.")
        models = tuple(self.models)
        if models != PUBLIC_PIPELINE_MODELS:
            raise ValueError("models must exactly equal the approved public model list.")
        object.__setattr__(self, "models", models)


class ProductStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    PARTIAL = "partial"
    FAILED = "failed"


class ProductEventKind(str, Enum):
    PERSISTED = "persisted"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    COUNT_CHANGED = "count_changed"
    POSITION_CHANGED = "position_changed"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductMask:
    """One resolved visual type and its cleaned full-frame binary mask.

    ``object_id`` is generated by :class:`ProductFrame` as a public ``Pnn``
    label.  Supplying a different ID is rejected so numbering stays deterministic.
    """

    type_id: str
    mask: NDArray[np.bool_] = field(repr=False, compare=False, hash=False)
    object_id: str | None = None

    def __post_init__(self) -> None:
        _require_type_id(self.type_id)
        if self.object_id is not None:
            _require_object_id(self.object_id)
        object.__setattr__(self, "mask", _canonical_binary_mask(self.mask))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductFrame:
    """Public per-frame output; masks are canonicalized to public Pnn objects."""

    frame_id: str
    frame_index: int
    source_image: str
    image_size: ImageSize
    masks: tuple[ProductMask, ...] = ()

    def __post_init__(self) -> None:
        _require_portable_frame_id(self.frame_id)
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        object.__setattr__(self, "source_image", _relative_posix_path(self.source_image))
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be ImageSize.")
        if isinstance(self.masks, (str, bytes)):
            raise TypeError("masks must be an iterable of ProductMask values.")
        masks = tuple(self.masks)
        if not all(isinstance(mask, ProductMask) for mask in masks):
            raise TypeError("masks must contain only ProductMask values.")
        ordered = tuple(sorted(masks, key=_mask_sort_key))
        numbered: list[ProductMask] = []
        for index, mask in enumerate(ordered):
            expected_object_id = f"P{index:02d}"
            if mask.object_id not in (None, expected_object_id):
                raise ValueError(
                    "object_id must equal the deterministic Pnn number within its frame."
                )
            if mask.mask.shape != (self.image_size.height, self.image_size.width):
                raise ValueError("Every mask must match the complete frame dimensions.")
            numbered.append(replace(mask, object_id=expected_object_id))
        object.__setattr__(self, "masks", tuple(numbered))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductObjectRef:
    """A public reference to one Pnn object in one frame."""

    frame_id: str
    object_id: str

    def __post_init__(self) -> None:
        _require_portable_frame_id(self.frame_id)
        _require_object_id(self.object_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductVisualType:
    """The deterministic list of public objects belonging to one visual type."""

    type_id: str
    objects: tuple[ProductObjectRef, ...]

    def __post_init__(self) -> None:
        _require_type_id(self.type_id)
        if isinstance(self.objects, (str, bytes)):
            raise TypeError("objects must be an iterable of ProductObjectRef values.")
        objects = tuple(self.objects)
        if not objects or not all(isinstance(item, ProductObjectRef) for item in objects):
            raise ValueError("objects must contain at least one ProductObjectRef value.")
        if len(objects) != len(set(objects)):
            raise ValueError("objects must not contain duplicate references.")
        object.__setattr__(
            self,
            "objects",
            tuple(sorted(objects, key=lambda item: (item.frame_id, item.object_id))),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductMatch:
    """A public, score-free link between two resolved objects."""

    from_object: ProductObjectRef
    to_object: ProductObjectRef
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.from_object, ProductObjectRef) or not isinstance(
            self.to_object, ProductObjectRef
        ):
            raise TypeError("from_object and to_object must be ProductObjectRef.")
        if self.from_object == self.to_object:
            raise ValueError("A match must connect two distinct objects.")
        _require_identifier(self.status, "status")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductEvent:
    """A public type-level event with references to its participating objects."""

    event_id: str
    kind: ProductEventKind
    type_id: str
    from_frame_id: str | None
    to_frame_id: str | None
    from_objects: tuple[ProductObjectRef, ...] = ()
    to_objects: tuple[ProductObjectRef, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.event_id, "event_id")
        if not isinstance(self.kind, ProductEventKind):
            raise TypeError("kind must be ProductEventKind.")
        _require_type_id(self.type_id)
        for field_name in ("from_frame_id", "to_frame_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_portable_frame_id(value)
        for field_name in ("from_objects", "to_objects"):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{field_name} must be an iterable of ProductObjectRef values.")
            refs = tuple(value)
            if not all(isinstance(item, ProductObjectRef) for item in refs):
                raise TypeError(f"{field_name} must contain only ProductObjectRef values.")
            if len(refs) != len(set(refs)):
                raise ValueError(f"{field_name} must not contain duplicate references.")
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(refs, key=lambda item: (item.frame_id, item.object_id))),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductResult:
    """A fully resolved, public result ready for deterministic serialization."""

    stream: ProductStream
    pipeline: ProductPipeline
    frames: tuple[ProductFrame, ...]
    status: ProductStatus = ProductStatus.COMPLETED
    matches: tuple[ProductMatch, ...] = ()
    events: tuple[ProductEvent, ...] = ()
    visual_types: tuple[ProductVisualType, ...] | None = None
    schema_version: str = PRODUCT_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCT_RESULT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PRODUCT_RESULT_SCHEMA_VERSION!r}.")
        if not isinstance(self.stream, ProductStream):
            raise TypeError("stream must be ProductStream.")
        if not isinstance(self.pipeline, ProductPipeline):
            raise TypeError("pipeline must be ProductPipeline.")
        if not isinstance(self.status, ProductStatus):
            raise TypeError("status must be ProductStatus.")
        if isinstance(self.frames, (str, bytes)):
            raise TypeError("frames must be an iterable of ProductFrame values.")
        frames = tuple(self.frames)
        if not frames or not all(isinstance(frame, ProductFrame) for frame in frames):
            raise ValueError("frames must contain at least one ProductFrame value.")
        frame_ids = tuple(frame.frame_id for frame in frames)
        frame_indices = tuple(frame.frame_index for frame in frames)
        if len(frame_ids) != len(set(frame_ids)) or len(frame_indices) != len(set(frame_indices)):
            raise ValueError("frame_id values and frame_index values must each be unique.")
        if self.stream.frame_count != len(frames):
            raise ValueError("stream.frame_count must equal the number of frames.")
        frames = tuple(sorted(frames, key=lambda frame: (frame.frame_index, frame.frame_id)))
        object.__setattr__(self, "frames", frames)

        object_types: dict[ProductObjectRef, str] = {}
        for frame in frames:
            for mask in frame.masks:
                assert mask.object_id is not None
                reference = ProductObjectRef(
                    frame_id=frame.frame_id,
                    object_id=mask.object_id,
                )
                object_types[reference] = mask.type_id
        expected_types = tuple(
            ProductVisualType(
                type_id=type_id,
                objects=tuple(
                    ref
                    for ref, object_type in object_types.items()
                    if object_type == type_id
                ),
            )
            for type_id in sorted(set(object_types.values()))
        )
        if self.visual_types is not None:
            if isinstance(self.visual_types, (str, bytes)):
                raise TypeError("visual_types must be an iterable of ProductVisualType values.")
            supplied_types = tuple(self.visual_types)
            if not all(isinstance(item, ProductVisualType) for item in supplied_types):
                raise TypeError("visual_types must contain only ProductVisualType values.")
            if tuple(sorted(supplied_types, key=lambda item: item.type_id)) != expected_types:
                raise ValueError("visual_types must exactly resolve the objects in frames.")
        object.__setattr__(self, "visual_types", expected_types)

        if isinstance(self.matches, (str, bytes)):
            raise TypeError("matches must be an iterable of ProductMatch values.")
        matches = tuple(self.matches)
        if not all(isinstance(match, ProductMatch) for match in matches):
            raise TypeError("matches must contain only ProductMatch values.")
        match_keys: set[tuple[ProductObjectRef, ProductObjectRef]] = set()
        for match in matches:
            if match.from_object not in object_types or match.to_object not in object_types:
                raise ValueError("Every match reference must resolve an object in frames.")
            if object_types[match.from_object] != object_types[match.to_object]:
                raise ValueError("A match can only link objects of the same type_id.")
            key = (match.from_object, match.to_object)
            if key in match_keys:
                raise ValueError("matches must not contain duplicate object links.")
            match_keys.add(key)
        object.__setattr__(self, "matches", tuple(sorted(matches, key=lambda item: (
            item.from_object.frame_id, item.from_object.object_id, item.to_object.frame_id,
            item.to_object.object_id, item.status,
        ))))

        if isinstance(self.events, (str, bytes)):
            raise TypeError("events must be an iterable of ProductEvent values.")
        events = tuple(self.events)
        if not all(isinstance(event, ProductEvent) for event in events):
            raise TypeError("events must contain only ProductEvent values.")
        event_ids = tuple(event.event_id for event in events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique.")
        known_frames = set(frame_ids)
        for event in events:
            for frame_id in (event.from_frame_id, event.to_frame_id):
                if frame_id is not None and frame_id not in known_frames:
                    raise ValueError("Every event frame reference must resolve a frame.")
            for ref in event.from_objects:
                if ref.frame_id != event.from_frame_id or object_types.get(ref) != event.type_id:
                    raise ValueError(
                        "from_objects must resolve same-type objects in from_frame_id."
                    )
            for ref in event.to_objects:
                if ref.frame_id != event.to_frame_id or object_types.get(ref) != event.type_id:
                    raise ValueError("to_objects must resolve same-type objects in to_frame_id.")
        object.__setattr__(self, "events", tuple(sorted(events, key=lambda event: event.event_id)))


__all__ = [
    "PRODUCT_RESULT_SCHEMA_VERSION",
    "PUBLIC_PIPELINE_MODELS",
    "ProductEvent",
    "ProductEventKind",
    "ProductFrame",
    "ProductMask",
    "ProductMatch",
    "ProductObjectRef",
    "ProductPipeline",
    "ProductResult",
    "ProductStatus",
    "ProductStream",
    "ProductVisualType",
]
