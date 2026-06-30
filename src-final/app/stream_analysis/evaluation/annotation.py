"""Strict loader for probe/final ``annotation.json`` files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..contracts import BBox, ImageSize

SUPPORTED_EVENT_TYPES = (
    "persisted",
    "appeared",
    "disappeared",
    "count_changed",
    "position_changed",
)


class AnnotationFormatError(ValueError):
    """Raised when an annotation file is not usable by the evaluator."""


@dataclass(frozen=True, slots=True)
class AnnotationInstance:
    instance_id: str
    frame_id: str
    visual_type_id: str
    bbox: BBox
    uncertainty: str = ""


@dataclass(frozen=True, slots=True)
class ExpectedChangeEvent:
    event_id: str
    event_type: str
    visual_type_id: str
    from_frame_id: str
    to_frame_id: str
    uncertainty: str = ""


@dataclass(frozen=True, slots=True)
class StreamAnnotation:
    path: Path
    digest_sha256: str
    schema_version: str
    stream_id: str
    manifest_ref: str
    annotation_scope: str
    visual_type_ids: tuple[str, ...]
    frame_ids: tuple[str, ...]
    frame_size: ImageSize
    instances: tuple[AnnotationInstance, ...]
    frame_comparisons: tuple[tuple[str, str, tuple[str, ...]], ...]
    change_events: tuple[ExpectedChangeEvent, ...]
    supported_event_types: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def instances_by_frame(self) -> dict[str, tuple[AnnotationInstance, ...]]:
        result: dict[str, list[AnnotationInstance]] = {frame_id: [] for frame_id in self.frame_ids}
        for instance in self.instances:
            result.setdefault(instance.frame_id, []).append(instance)
        return {key: tuple(value) for key, value in result.items()}

    @property
    def events_by_id(self) -> dict[str, ExpectedChangeEvent]:
        return {event.event_id: event for event in self.change_events}


def load_annotation(path: Path, *, manifest_path: Path | None = None) -> StreamAnnotation:
    """Load and validate an annotation file without touching analyze code."""

    annotation_path = Path(path).resolve()
    data = annotation_path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as error:
        raise AnnotationFormatError(f"Invalid annotation JSON: {error}") from error
    if not isinstance(payload, dict):
        raise AnnotationFormatError("annotation.json must contain a JSON object.")

    manifest = None
    if manifest_path is not None:
        manifest = _load_manifest(Path(manifest_path))

    schema_version = _text(payload, "schema_version")
    stream_id = _text(payload, "stream_id")
    manifest_ref = _text(payload, "manifest_ref")
    annotation_scope = _text(payload, "annotation_scope")

    visual_type_ids = tuple(_text(item, "visual_type_id") for item in _list(payload, "visual_types"))
    _require_unique(visual_type_ids, "visual_type_id")
    visual_type_set = set(visual_type_ids)

    frame_ids, frame_size = _frames_and_size(payload, manifest)
    frame_set = set(frame_ids)
    neighbor_pairs = _neighbor_pairs(frame_ids)

    instances = tuple(
        _parse_instance(item, visual_type_set, frame_set, frame_size)
        for item in _list(payload, "expected_element_instances")
    )
    _require_unique(tuple(item.instance_id for item in instances), "instance_id")

    events = tuple(
        _parse_event(item, visual_type_set, frame_set, neighbor_pairs)
        for item in _list(payload, "change_events")
    )
    _require_unique(tuple(item.event_id for item in events), "event_id")
    event_ids = {item.event_id for item in events}

    comparisons: list[tuple[str, str, tuple[str, ...]]] = []
    for item in _list(payload, "frame_comparisons"):
        if not isinstance(item, dict):
            raise AnnotationFormatError("frame_comparisons items must be objects.")
        left = _text(item, "from_frame_id")
        right = _text(item, "to_frame_id")
        if left not in frame_set or right not in frame_set:
            raise AnnotationFormatError("frame_comparison references an unknown frame.")
        if (left, right) not in neighbor_pairs:
            raise AnnotationFormatError(f"frame_comparison must reference neighboring frames, got {left!r}->{right!r}.")
        ids = tuple(_string(value, "expected_change_event_ids item") for value in _list(item, "expected_change_event_ids"))
        missing = set(ids) - event_ids
        if missing:
            raise AnnotationFormatError(f"frame_comparison references unknown event IDs: {sorted(missing)!r}.")
        comparisons.append((left, right, ids))

    supported = tuple(_string(value, "supported_event_types item") for value in payload.get("supported_event_types", SUPPORTED_EVENT_TYPES))
    if any(item not in SUPPORTED_EVENT_TYPES for item in supported):
        raise AnnotationFormatError("supported_event_types contains an unsupported event type.")

    if manifest is not None:
        if manifest.get("stream_id") != stream_id:
            raise AnnotationFormatError("annotation stream_id does not match manifest stream_id.")
        manifest_frames = tuple(_text(item, "frame_id") for item in _list(manifest, "frames"))
        if manifest_frames != frame_ids:
            raise AnnotationFormatError("annotation frames do not match manifest ordering.")

    return StreamAnnotation(
        path=annotation_path,
        digest_sha256=sha256(data).hexdigest(),
        schema_version=schema_version,
        stream_id=stream_id,
        manifest_ref=manifest_ref,
        annotation_scope=annotation_scope,
        visual_type_ids=visual_type_ids,
        frame_ids=frame_ids,
        frame_size=frame_size,
        instances=instances,
        frame_comparisons=tuple(comparisons),
        change_events=events,
        supported_event_types=supported,
        raw=payload,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationFormatError(f"Cannot load manifest for annotation validation: {error}") from error
    if not isinstance(loaded, dict):
        raise AnnotationFormatError("manifest.json must contain an object.")
    return loaded


def _frames_and_size(payload: dict[str, Any], manifest: dict[str, Any] | None) -> tuple[tuple[str, ...], ImageSize]:
    if manifest is not None:
        frames = tuple(_text(item, "frame_id") for item in _list(manifest, "frames"))
        metadata = manifest.get("metadata", {})
        size_data = metadata.get("frame_size") if isinstance(metadata, dict) else None
        if size_data is None:
            size_data = payload.get("frame_size")
    else:
        frames = tuple(
            dict.fromkeys(_text(item, "frame_id") for item in _list(payload, "expected_element_instances"))
        )
        size_data = payload.get("frame_size")
    if not frames:
        raise AnnotationFormatError("annotation must resolve at least one frame.")
    if not isinstance(size_data, dict):
        raise AnnotationFormatError("frame_size must be available from manifest metadata.")
    return frames, ImageSize(width=_int(size_data, "width"), height=_int(size_data, "height"))


def _parse_instance(
    item: Any,
    visual_type_ids: set[str],
    frame_ids: set[str],
    frame_size: ImageSize,
) -> AnnotationInstance:
    if not isinstance(item, dict):
        raise AnnotationFormatError("expected_element_instances items must be objects.")
    visual_type_id = _text(item, "visual_type_id")
    frame_id = _text(item, "frame_id")
    if visual_type_id not in visual_type_ids:
        raise AnnotationFormatError(f"Unknown visual_type_id {visual_type_id!r}.")
    if frame_id not in frame_ids:
        raise AnnotationFormatError(f"Unknown frame_id {frame_id!r}.")
    bbox_data = item.get("bbox")
    if not isinstance(bbox_data, dict):
        raise AnnotationFormatError("instance bbox must be an object.")
    bbox = BBox(
        x=_number(bbox_data, "x"),
        y=_number(bbox_data, "y"),
        width=_number(bbox_data, "width"),
        height=_number(bbox_data, "height"),
    )
    if not bbox.is_within(frame_size):
        raise AnnotationFormatError(f"Instance bbox is outside frame: {_text(item, 'instance_id')}.")
    return AnnotationInstance(
        instance_id=_text(item, "instance_id"),
        frame_id=frame_id,
        visual_type_id=visual_type_id,
        bbox=bbox,
        uncertainty=_optional_text(item, "uncertainty"),
    )


def _parse_event(
    item: Any,
    visual_type_ids: set[str],
    frame_ids: set[str],
    neighbor_pairs: set[tuple[str, str]],
) -> ExpectedChangeEvent:
    if not isinstance(item, dict):
        raise AnnotationFormatError("change_events items must be objects.")
    event_type = _text(item, "event_type")
    visual_type_id = _text(item, "visual_type_id")
    left = _text(item, "from_frame_id")
    right = _text(item, "to_frame_id")
    if event_type not in SUPPORTED_EVENT_TYPES:
        raise AnnotationFormatError(f"Unsupported event_type {event_type!r}.")
    if visual_type_id not in visual_type_ids:
        raise AnnotationFormatError(f"Unknown event visual_type_id {visual_type_id!r}.")
    if left not in frame_ids or right not in frame_ids:
        raise AnnotationFormatError("Event references an unknown frame.")
    if (left, right) not in neighbor_pairs:
        raise AnnotationFormatError(f"Event must reference neighboring frames, got {left!r}->{right!r}.")
    return ExpectedChangeEvent(
        event_id=_text(item, "event_id"),
        event_type=event_type,
        visual_type_id=visual_type_id,
        from_frame_id=left,
        to_frame_id=right,
        uncertainty=_optional_text(item, "uncertainty"),
    )


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise AnnotationFormatError(f"{key} must be a list.")
    return value


def _text(payload: dict[str, Any], key: str) -> str:
    return _string(payload.get(key), key)


def _optional_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise AnnotationFormatError(f"{key} must be a string without surrounding whitespace.")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise AnnotationFormatError(f"{name} must be a string without surrounding whitespace.")
    if not value:
        raise AnnotationFormatError(f"{name} must not be empty.")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationFormatError(f"{key} must be numeric.")
    return float(value)


def _int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnnotationFormatError(f"{key} must be an integer.")
    return value


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise AnnotationFormatError(f"{name} values must be unique.")


def _neighbor_pairs(frame_ids: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(frame_ids, frame_ids[1:]))
