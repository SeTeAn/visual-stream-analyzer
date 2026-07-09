"""Immutable input records for an ordered image stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from .common import RecordEnvelope, StageContext, _require_identifier
from .geometry import ImageSize


def _relative_stream_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("image_path must be a string.")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError("image_path must be a non-empty relative stream path.")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("image_path must be relative to the stream root.")
    if PureWindowsPath(value).drive:
        raise ValueError("image_path must not contain a drive prefix.")

    normalized = value.replace("\\", "/")
    raw_parts = normalized.split("/")
    if any(part == ".." for part in raw_parts):
        raise ValueError("image_path must not leave the stream root.")
    if normalized.endswith("/"):
        raise ValueError("image_path must identify a file-like relative path.")

    path = PurePosixPath(normalized)
    if not path.parts or path.name in ("", ".", ".."):
        raise ValueError("image_path must identify a file-like relative path.")
    return path.as_posix()


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameRecord:
    """A decoded-frame input record without filesystem or image-loading logic."""

    envelope: RecordEnvelope
    frame_id: str
    index: int
    image_path: str
    image_size: ImageSize

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise TypeError("envelope must be RecordEnvelope.")
        _require_identifier(self.frame_id, "frame_id")
        if self.envelope.context != StageContext(frame_id=self.frame_id):
            raise ValueError(
                "envelope.context must match StageContext(frame_id=frame_id) exactly "
                "and contain no downstream stage IDs."
            )
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("index must be an integer.")
        if self.index < 0:
            raise ValueError("index must be non-negative.")
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be ImageSize.")
        object.__setattr__(self, "image_path", _relative_stream_path(self.image_path))

    @property
    def stream_id(self) -> str:
        return self.envelope.stream_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageStream:
    """A non-empty deterministic sequence of frame records ordered by index."""

    envelope: RecordEnvelope
    frames: tuple[FrameRecord, ...]
    ordering: str = "manifest"

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise TypeError("envelope must be RecordEnvelope.")
        if self.envelope.context != StageContext():
            raise ValueError("Stream envelope context must not contain frame or stage-object IDs.")
        if self.ordering != "manifest":
            raise ValueError("ordering must be 'manifest'.")
        if isinstance(self.frames, (str, bytes)):
            raise TypeError("frames must be an iterable of FrameRecord values.")

        frames = tuple(self.frames)
        if not frames:
            raise ValueError("ImageStream must contain at least one frame.")
        if not all(isinstance(frame, FrameRecord) for frame in frames):
            raise TypeError("frames must contain only FrameRecord values.")

        frame_ids = [frame.frame_id for frame in frames]
        indices = [frame.index for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame_id values must be unique within a stream.")
        if len(indices) != len(set(indices)):
            raise ValueError("frame indices must be unique within a stream.")
        if any(frame.stream_id != self.stream_id for frame in frames):
            raise ValueError("Every frame stream_id must match the ImageStream stream_id.")

        object.__setattr__(self, "frames", tuple(sorted(frames, key=lambda frame: frame.index)))

    @property
    def stream_id(self) -> str:
        return self.envelope.stream_id
