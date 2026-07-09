"""Strict Pillow decoding into immutable in-memory RGB frame bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from ..contracts import (
    FrameRecord,
    ImageSize,
    ImageStream,
    RecordEnvelope,
    StageContext,
)
from ..provenance import DataProvenance, Fingerprint
from .manifest import (
    InputIssue,
    InputIssueCode,
    ManifestFrameEntry,
    ManifestLoadRequest,
    ManifestLoadResult,
    StreamInputError,
    _inside_root,
    _issue,
    load_manifest,
)

FRAME_RECORD_SCHEMA_VERSION = "frame-record-1.0"
IMAGE_STREAM_SCHEMA_VERSION = "image-stream-1.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedFrame:
    record: FrameRecord
    rgb_bytes: bytes
    raw_digest: Fingerprint
    image_format: str
    resolved_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.record, FrameRecord):
            raise TypeError("record must be FrameRecord.")
        if not isinstance(self.rgb_bytes, bytes):
            raise TypeError("rgb_bytes must be immutable bytes.")
        expected_length = self.record.image_size.width * self.record.image_size.height * 3
        if len(self.rgb_bytes) != expected_length:
            raise ValueError("rgb_bytes length must equal width * height * 3.")
        if not isinstance(self.raw_digest, Fingerprint):
            raise TypeError("raw_digest must be Fingerprint.")
        if self.raw_digest.algorithm != "sha256":
            raise ValueError("raw_digest must use sha256.")
        if self.image_format not in {"PNG", "JPEG"}:
            raise ValueError("image_format must be PNG or JPEG.")
        if not isinstance(self.resolved_path, Path) or not self.resolved_path.is_absolute():
            raise ValueError("resolved_path must be an absolute Path.")

    @property
    def frame_id(self) -> str:
        return self.record.frame_id

    @property
    def image_size(self) -> ImageSize:
        return self.record.image_size

    def to_pillow_image(self) -> Image.Image:
        """Return a new independent RGB image backed only by immutable bytes."""

        return Image.frombytes(
            "RGB",
            (self.image_size.width, self.image_size.height),
            self.rgb_bytes,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedStream:
    manifest: ManifestLoadResult
    stream: ImageStream
    frames: tuple[DecodedFrame, ...]
    data_provenance: DataProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ManifestLoadResult):
            raise TypeError("manifest must be ManifestLoadResult.")
        if not isinstance(self.stream, ImageStream):
            raise TypeError("stream must be ImageStream.")
        frames = tuple(self.frames)
        if not frames or not all(isinstance(frame, DecodedFrame) for frame in frames):
            raise ValueError("frames must contain DecodedFrame values.")
        if tuple(frame.record for frame in frames) != self.stream.frames:
            raise ValueError("Decoded frames must align exactly with ImageStream frames.")
        if tuple(frame.frame_id for frame in frames) != tuple(
            frame.frame_id for frame in self.manifest.frames
        ):
            raise ValueError("Decoded frame identity must match manifest identity.")
        if not isinstance(self.data_provenance, DataProvenance):
            raise TypeError("data_provenance must be DataProvenance.")
        if self.data_provenance.candidate_snapshot_digest is not None:
            raise ValueError("Input-stage DataProvenance must not contain a candidate snapshot.")
        if self.stream.stream_id != self.manifest.stream_id:
            raise ValueError("ImageStream identity must match manifest identity.")
        if self.data_provenance.manifest_digest != self.manifest.manifest_digest:
            raise ValueError("DataProvenance manifest digest must match the loaded manifest.")
        expected_digests = {frame.frame_id: frame.raw_digest for frame in frames}
        if dict(self.data_provenance.frame_content_digests) != expected_digests:
            raise ValueError("DataProvenance frame digests must match decoded frames.")
        object.__setattr__(self, "frames", frames)

    def frame(self, frame_id: str) -> DecodedFrame:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(frame_id)


def _fingerprint(data: bytes) -> Fingerprint:
    return Fingerprint(algorithm="sha256", value=hashlib.sha256(data).hexdigest())


def _current_resolved_path(
    entry: ManifestFrameEntry,
    manifest: ManifestLoadResult,
    issues: list[InputIssue],
    position: int,
) -> Path | None:
    source = manifest.stream_root.joinpath(*PurePosixPath(entry.image_path).parts)
    try:
        current = source.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        issues.append(
            _issue(
                InputIssueCode.PATH_RESOLUTION_ERROR,
                "Image path could not be resolved at decode time.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
                cause=error,
            )
        )
        return None
    if not _inside_root(current, manifest.stream_root):
        issues.append(
            _issue(
                InputIssueCode.PATH_OUTSIDE_STREAM_ROOT,
                "Resolved image path is outside stream root at decode time.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
            )
        )
        return None
    if current != entry.resolved_path:
        issues.append(
            _issue(
                InputIssueCode.IMAGE_PATH_CHANGED,
                "Resolved image path changed after manifest validation.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
            )
        )
        return None
    if not current.exists():
        issues.append(
            _issue(
                InputIssueCode.IMAGE_NOT_FOUND,
                "Image file does not exist at decode time.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
            )
        )
        return None
    if not current.is_file():
        issues.append(
            _issue(
                InputIssueCode.IMAGE_NOT_FILE,
                "Image path is not a file at decode time.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
            )
        )
        return None
    return current


def _expected_format(entry: ManifestFrameEntry) -> str:
    return "PNG" if PurePosixPath(entry.image_path).suffix.casefold() == ".png" else "JPEG"


def _terminal_marker_present(raw_bytes: bytes, image_format: str) -> bool:
    if image_format == "PNG":
        return b"IEND" in raw_bytes[-32:]
    if image_format == "JPEG":
        return raw_bytes.rstrip().endswith(b"\xff\xd9")
    return False


def _decode_entry(
    entry: ManifestFrameEntry,
    manifest: ManifestLoadResult,
    position: int,
    issues: list[InputIssue],
) -> tuple[bytes, bytes, str, ImageSize, Path] | None:
    resolved_path = _current_resolved_path(entry, manifest, issues, position)
    if resolved_path is None:
        return None
    try:
        raw_bytes = resolved_path.read_bytes()
    except OSError as error:
        issues.append(
            _issue(
                InputIssueCode.IMAGE_READ_ERROR,
                "Image file could not be read.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
                cause=error,
            )
        )
        return None

    expected_format = _expected_format(entry)
    try:
        with BytesIO(raw_bytes) as buffer:
            with Image.open(buffer) as image:
                actual_format = image.format
                if actual_format != expected_format:
                    issues.append(
                        _issue(
                            InputIssueCode.IMAGE_FORMAT_MISMATCH,
                            f"Image extension requires {expected_format}, decoded format is {actual_format or 'unknown'}.",
                            field_path=f"frames[{position}].image_path",
                            stream_id=manifest.stream_id,
                            frame_id=entry.frame_id,
                            image_path=entry.image_path,
                            manifest_position=position,
                        )
                    )
                    return None
                image.load()
                if not _terminal_marker_present(raw_bytes, actual_format):
                    raise OSError(f"{actual_format} terminal marker is missing")
                with image.convert("RGB") as rgb_image:
                    rgb_image.load()
                    size = ImageSize(width=rgb_image.width, height=rgb_image.height)
                    rgb_bytes = rgb_image.tobytes()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        issues.append(
            _issue(
                InputIssueCode.IMAGE_DECODE_ERROR,
                "Image could not be fully decoded as a non-truncated PNG or JPEG.",
                field_path=f"frames[{position}].image_path",
                stream_id=manifest.stream_id,
                frame_id=entry.frame_id,
                image_path=entry.image_path,
                manifest_position=position,
                cause=error,
            )
        )
        return None
    return raw_bytes, rgb_bytes, expected_format, size, resolved_path


def decode_stream(manifest: ManifestLoadResult) -> DecodedStream:
    if not isinstance(manifest, ManifestLoadResult):
        raise TypeError("manifest must be ManifestLoadResult.")
    issues: list[InputIssue] = []
    decoded: list[DecodedFrame] = []
    frame_digests: dict[str, Fingerprint] = {}
    frame_records: list[FrameRecord] = []

    for entry in manifest.frames:
        position = entry.manifest_position
        loaded = _decode_entry(entry, manifest, position, issues)
        if loaded is None:
            continue
        _raw_bytes, rgb_bytes, image_format, image_size, resolved_path = loaded
        raw_digest = _fingerprint(_raw_bytes)
        frame_record = FrameRecord(
            envelope=RecordEnvelope(
                record_id=entry.frame_id,
                schema_version=FRAME_RECORD_SCHEMA_VERSION,
                stream_id=manifest.stream_id,
                producer=manifest.producer,
                context=StageContext(frame_id=entry.frame_id),
            ),
            frame_id=entry.frame_id,
            index=entry.index,
            image_path=entry.image_path,
            image_size=image_size,
        )
        frame_records.append(frame_record)
        frame_digests[entry.frame_id] = raw_digest
        decoded.append(
            DecodedFrame(
                record=frame_record,
                rgb_bytes=rgb_bytes,
                raw_digest=raw_digest,
                image_format=image_format,
                resolved_path=resolved_path,
            )
        )

    if issues:
        raise StreamInputError(issues)
    stream = ImageStream(
        envelope=RecordEnvelope(
            record_id=manifest.stream_id,
            schema_version=IMAGE_STREAM_SCHEMA_VERSION,
            stream_id=manifest.stream_id,
            producer=manifest.producer,
            context=StageContext(),
        ),
        frames=tuple(frame_records),
        ordering="manifest",
    )
    provenance = DataProvenance(
        manifest_digest=manifest.manifest_digest,
        frame_content_digests=frame_digests,
        candidate_snapshot_digest=None,
    )
    return DecodedStream(
        manifest=manifest,
        stream=stream,
        frames=tuple(decoded),
        data_provenance=provenance,
    )


def load_decoded_stream(request: ManifestLoadRequest) -> DecodedStream:
    """Validate the manifest and decode every frame exactly once from raw bytes."""

    return decode_stream(load_manifest(request))


__all__ = [
    "FRAME_RECORD_SCHEMA_VERSION",
    "IMAGE_STREAM_SCHEMA_VERSION",
    "DecodedFrame",
    "DecodedStream",
    "decode_stream",
    "load_decoded_stream",
]
