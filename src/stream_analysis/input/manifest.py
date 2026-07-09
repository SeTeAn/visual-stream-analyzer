"""Validation and loading of the read-only ``stream-input-0.1`` manifest."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from ..config import ConfigValue, _freeze_mapping
from ..contracts import ProducerProvenance
from ..contracts.common import _require_identifier
from ..contracts.stream import _relative_stream_path
from ..provenance import Fingerprint

STREAM_INPUT_SCHEMA_VERSION = "stream-input-0.1"
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG"})

_ROOT_FIELDS = frozenset(
    {"schema_version", "stream_id", "ordering", "frames", "scene_description", "notes", "metadata"}
)
_FRAME_FIELDS = frozenset({"frame_id", "index", "image_path", "notes", "metadata"})


class InputIssueCode(str, Enum):
    STREAM_ROOT_NOT_FOUND = "STREAM_ROOT_NOT_FOUND"
    STREAM_ROOT_NOT_DIRECTORY = "STREAM_ROOT_NOT_DIRECTORY"
    MANIFEST_OUTSIDE_STREAM_ROOT = "MANIFEST_OUTSIDE_STREAM_ROOT"
    MANIFEST_NOT_FOUND = "MANIFEST_NOT_FOUND"
    MANIFEST_NOT_FILE = "MANIFEST_NOT_FILE"
    MANIFEST_READ_ERROR = "MANIFEST_READ_ERROR"
    MANIFEST_INVALID_UTF8 = "MANIFEST_INVALID_UTF8"
    MANIFEST_INVALID_JSON = "MANIFEST_INVALID_JSON"
    MANIFEST_ROOT_NOT_OBJECT = "MANIFEST_ROOT_NOT_OBJECT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FIELD_TYPE = "INVALID_FIELD_TYPE"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    FORBIDDEN_LABEL_FIELD = "FORBIDDEN_LABEL_FIELD"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    INVALID_STREAM_ID = "INVALID_STREAM_ID"
    INVALID_ORDERING = "INVALID_ORDERING"
    EMPTY_FRAMES = "EMPTY_FRAMES"
    INVALID_FRAME_ENTRY = "INVALID_FRAME_ENTRY"
    INVALID_FRAME_ID = "INVALID_FRAME_ID"
    INVALID_FRAME_INDEX = "INVALID_FRAME_INDEX"
    DUPLICATE_FRAME_ID = "DUPLICATE_FRAME_ID"
    DUPLICATE_FRAME_INDEX = "DUPLICATE_FRAME_INDEX"
    INVALID_IMAGE_PATH = "INVALID_IMAGE_PATH"
    PATH_RESOLUTION_ERROR = "PATH_RESOLUTION_ERROR"
    PATH_OUTSIDE_STREAM_ROOT = "PATH_OUTSIDE_STREAM_ROOT"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    IMAGE_NOT_FILE = "IMAGE_NOT_FILE"
    UNSUPPORTED_IMAGE_EXTENSION = "UNSUPPORTED_IMAGE_EXTENSION"
    IMAGE_PATH_CHANGED = "IMAGE_PATH_CHANGED"
    IMAGE_READ_ERROR = "IMAGE_READ_ERROR"
    IMAGE_FORMAT_MISMATCH = "IMAGE_FORMAT_MISMATCH"
    IMAGE_DECODE_ERROR = "IMAGE_DECODE_ERROR"


@dataclass(frozen=True, slots=True, kw_only=True)
class InputIssue:
    code: InputIssueCode
    message: str
    field_path: str | None = None
    stream_id: str | None = None
    frame_id: str | None = None
    image_path: str | None = None
    manifest_position: int | None = None
    cause_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, InputIssueCode):
            raise TypeError("code must be InputIssueCode.")
        if not isinstance(self.message, str) or not self.message or self.message != self.message.strip():
            raise ValueError("message must be a non-empty trimmed string.")
        for field_name in ("field_path", "stream_id", "frame_id", "image_path", "cause_type"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None.")
        if self.manifest_position is not None:
            if isinstance(self.manifest_position, bool) or not isinstance(self.manifest_position, int):
                raise TypeError("manifest_position must be an integer or None.")
            if self.manifest_position < 0:
                raise ValueError("manifest_position must be non-negative.")

    @property
    def sort_key(self) -> tuple[int, str, str, str, str, str]:
        return (
            -1 if self.manifest_position is None else self.manifest_position,
            self.field_path or "",
            self.code.value,
            self.frame_id or "",
            self.image_path or "",
            self.message,
        )


class StreamInputError(ValueError):
    """Fatal aggregate of deterministic structured stream-input issues."""

    def __init__(self, issues: tuple[InputIssue, ...] | list[InputIssue]) -> None:
        supplied = tuple(issues)
        if not supplied:
            raise ValueError("StreamInputError requires at least one issue.")
        if not all(isinstance(issue, InputIssue) for issue in supplied):
            raise TypeError("issues must contain only InputIssue values.")
        ordered = tuple(sorted(supplied, key=lambda issue: issue.sort_key))
        self.issues = ordered
        summary = "; ".join(
            f"{issue.code.value}"
            f"{f'[{issue.field_path}]' if issue.field_path else ''}: {issue.message}"
            for issue in ordered
        )
        super().__init__(f"Stream input failed: {summary}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestLoadRequest:
    stream_root: Path | str
    producer: ProducerProvenance
    manifest_name: str = "manifest.json"

    def __post_init__(self) -> None:
        try:
            root = Path(self.stream_root)
        except TypeError as error:
            raise TypeError("stream_root must be a path-like value.") from error
        if not isinstance(self.producer, ProducerProvenance):
            raise TypeError("producer must be ProducerProvenance.")
        if self.producer.producer_stage != "stream_input":
            raise ValueError("producer.producer_stage must be 'stream_input'.")
        if not isinstance(self.manifest_name, str) or not self.manifest_name.strip():
            raise ValueError("manifest_name must be a non-empty relative path.")
        posix = PurePosixPath(self.manifest_name.replace("\\", "/"))
        windows = PureWindowsPath(self.manifest_name)
        if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
            raise ValueError("manifest_name must remain within stream_root.")
        object.__setattr__(self, "stream_root", root)
        object.__setattr__(self, "manifest_name", posix.as_posix())


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestFrameEntry:
    frame_id: str
    index: int
    manifest_position: int
    image_path: str
    resolved_path: Path
    notes: str | None = None
    metadata: Mapping[str, ConfigValue] = field(
        default_factory=lambda: MappingProxyType({}),
        hash=False,
    )

    def __post_init__(self) -> None:
        _require_identifier(self.frame_id, "frame_id")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer.")
        if (
            isinstance(self.manifest_position, bool)
            or not isinstance(self.manifest_position, int)
            or self.manifest_position < 0
        ):
            raise ValueError("manifest_position must be a non-negative integer.")
        object.__setattr__(self, "image_path", _relative_stream_path(self.image_path))
        if not isinstance(self.resolved_path, Path) or not self.resolved_path.is_absolute():
            raise ValueError("resolved_path must be an absolute Path.")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be a string or None.")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestLoadResult:
    stream_root: Path
    manifest_path: Path
    manifest_digest: Fingerprint
    producer: ProducerProvenance
    stream_id: str
    ordering: str
    frames: tuple[ManifestFrameEntry, ...]
    scene_description: str | None = None
    notes: str | None = None
    metadata: Mapping[str, ConfigValue] = field(
        default_factory=lambda: MappingProxyType({}),
        hash=False,
    )

    def __post_init__(self) -> None:
        for field_name in ("stream_root", "manifest_path"):
            value = getattr(self, field_name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field_name} must be an absolute Path.")
        if not isinstance(self.manifest_digest, Fingerprint):
            raise TypeError("manifest_digest must be Fingerprint.")
        if self.manifest_digest.algorithm != "sha256":
            raise ValueError("manifest_digest must use sha256.")
        if not isinstance(self.producer, ProducerProvenance):
            raise TypeError("producer must be ProducerProvenance.")
        _require_identifier(self.stream_id, "stream_id")
        if self.ordering != "manifest":
            raise ValueError("ordering must be 'manifest'.")
        frames = tuple(self.frames)
        if not frames or not all(isinstance(frame, ManifestFrameEntry) for frame in frames):
            raise ValueError("frames must contain ManifestFrameEntry values.")
        if tuple(sorted(frames, key=lambda frame: frame.index)) != frames:
            raise ValueError("frames must be ordered by index.")
        frame_ids = [frame.frame_id for frame in frames]
        indices = [frame.index for frame in frames]
        positions = [frame.manifest_position for frame in frames]
        if len(frame_ids) != len(set(frame_ids)) or len(indices) != len(set(indices)):
            raise ValueError("Frame IDs and indices must be unique.")
        if len(positions) != len(set(positions)):
            raise ValueError("manifest_position values must be unique.")
        if not _inside_root(self.manifest_path, self.stream_root):
            raise ValueError("manifest_path must remain inside stream_root.")
        if any(not _inside_root(frame.resolved_path, self.stream_root) for frame in frames):
            raise ValueError("Every resolved frame path must remain inside stream_root.")
        for field_name in ("scene_description", "notes"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None.")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


def _fingerprint(data: bytes) -> Fingerprint:
    return Fingerprint(algorithm="sha256", value=hashlib.sha256(data).hexdigest())


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _issue(
    code: InputIssueCode,
    message: str,
    *,
    field_path: str | None = None,
    stream_id: str | None = None,
    frame_id: str | None = None,
    image_path: str | None = None,
    manifest_position: int | None = None,
    cause: BaseException | None = None,
) -> InputIssue:
    return InputIssue(
        code=code,
        message=message,
        field_path=field_path,
        stream_id=stream_id,
        frame_id=frame_id,
        image_path=image_path,
        manifest_position=manifest_position,
        cause_type=type(cause).__name__ if cause is not None else None,
    )


def _required(data: Mapping[str, Any], name: str, issues: list[InputIssue]) -> Any:
    if name not in data:
        issues.append(
            _issue(InputIssueCode.MISSING_FIELD, f"Required field {name!r} is missing.", field_path=name)
        )
        return None
    return data[name]


def _optional_text(
    value: Any,
    field_path: str,
    issues: list[InputIssue],
    *,
    frame_id: str | None = None,
    position: int | None = None,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        issues.append(
            _issue(
                InputIssueCode.INVALID_FIELD_TYPE,
                f"{field_path} must be a string when present.",
                field_path=field_path,
                frame_id=frame_id,
                manifest_position=position,
            )
        )
        return None
    return value


def _optional_metadata(
    value: Any,
    field_path: str,
    issues: list[InputIssue],
    *,
    frame_id: str | None = None,
    position: int | None = None,
) -> Mapping[str, ConfigValue]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                InputIssueCode.INVALID_FIELD_TYPE,
                f"{field_path} must be a JSON object when present.",
                field_path=field_path,
                frame_id=frame_id,
                manifest_position=position,
            )
        )
        return MappingProxyType({})
    _find_label_fields(value, field_path, issues, frame_id=frame_id, position=position)
    return _freeze_mapping(value, field_path)


def _label_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _find_label_fields(
    value: Any,
    path: str,
    issues: list[InputIssue],
    *,
    frame_id: str | None,
    position: int | None,
) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item).encode("utf-8")):
            token = _label_token(str(key))
            nested_path = f"{path}.{key}"
            if token.startswith(("annotation", "groundtruth", "gt")):
                issues.append(
                    _issue(
                        InputIssueCode.FORBIDDEN_LABEL_FIELD,
                        "Annotation and ground-truth metadata are not allowed in analyze input.",
                        field_path=nested_path,
                        frame_id=frame_id,
                        manifest_position=position,
                    )
                )
            _find_label_fields(value[key], nested_path, issues, frame_id=frame_id, position=position)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _find_label_fields(
                nested,
                f"{path}[{index}]",
                issues,
                frame_id=frame_id,
                position=position,
            )


def _resolve_stream_root(request: ManifestLoadRequest) -> Path:
    root = request.stream_root
    if not root.exists():
        raise StreamInputError(
            [_issue(InputIssueCode.STREAM_ROOT_NOT_FOUND, "Stream root does not exist.", field_path="stream_root")]
        )
    if not root.is_dir():
        raise StreamInputError(
            [_issue(InputIssueCode.STREAM_ROOT_NOT_DIRECTORY, "Stream root is not a directory.", field_path="stream_root")]
        )
    try:
        return root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise StreamInputError(
            [
                _issue(
                    InputIssueCode.PATH_RESOLUTION_ERROR,
                    "Stream root could not be resolved.",
                    field_path="stream_root",
                    cause=error,
                )
            ]
        ) from None


def _read_manifest(request: ManifestLoadRequest, root: Path) -> tuple[Path, bytes]:
    manifest_source = root.joinpath(*PurePosixPath(request.manifest_name).parts)
    try:
        manifest_path = manifest_source.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise StreamInputError(
            [
                _issue(
                    InputIssueCode.PATH_RESOLUTION_ERROR,
                    "Manifest path could not be resolved.",
                    field_path="manifest_name",
                    cause=error,
                )
            ]
        ) from None
    if not _inside_root(manifest_path, root):
        raise StreamInputError(
            [
                _issue(
                    InputIssueCode.MANIFEST_OUTSIDE_STREAM_ROOT,
                    "Resolved manifest path is outside stream root.",
                    field_path="manifest_name",
                )
            ]
        )
    if not manifest_path.exists():
        raise StreamInputError(
            [_issue(InputIssueCode.MANIFEST_NOT_FOUND, "Manifest file does not exist.", field_path="manifest_name")]
        )
    if not manifest_path.is_file():
        raise StreamInputError(
            [_issue(InputIssueCode.MANIFEST_NOT_FILE, "Manifest path is not a file.", field_path="manifest_name")]
        )
    try:
        return manifest_path, manifest_path.read_bytes()
    except OSError as error:
        raise StreamInputError(
            [
                _issue(
                    InputIssueCode.MANIFEST_READ_ERROR,
                    "Manifest file could not be read.",
                    field_path="manifest_name",
                    cause=error,
                )
            ]
        ) from None


def _validate_frame(
    raw: Any,
    position: int,
    root: Path,
    stream_id: str | None,
    issues: list[InputIssue],
) -> ManifestFrameEntry | None:
    prefix = f"frames[{position}]"
    if not isinstance(raw, Mapping):
        issues.append(
            _issue(
                InputIssueCode.INVALID_FRAME_ENTRY,
                "Frame entry must be a JSON object.",
                field_path=prefix,
                stream_id=stream_id,
                manifest_position=position,
            )
        )
        return None
    for name in sorted(set(raw) - _FRAME_FIELDS, key=lambda item: str(item).encode("utf-8")):
        issues.append(
            _issue(
                InputIssueCode.UNKNOWN_FIELD,
                f"Unknown frame field {name!r}.",
                field_path=f"{prefix}.{name}",
                stream_id=stream_id,
                manifest_position=position,
            )
        )

    frame_id_raw = raw.get("frame_id")
    frame_id: str | None = None
    if "frame_id" not in raw:
        issues.append(
            _issue(InputIssueCode.MISSING_FIELD, "Required field 'frame_id' is missing.", field_path=f"{prefix}.frame_id", stream_id=stream_id, manifest_position=position)
        )
    elif not isinstance(frame_id_raw, str):
        issues.append(
            _issue(InputIssueCode.INVALID_FIELD_TYPE, "frame_id must be a string.", field_path=f"{prefix}.frame_id", stream_id=stream_id, manifest_position=position)
        )
    else:
        try:
            _require_identifier(frame_id_raw, "frame_id")
            frame_id = frame_id_raw
        except (TypeError, ValueError):
            issues.append(
                _issue(InputIssueCode.INVALID_FRAME_ID, "frame_id is not a valid identifier.", field_path=f"{prefix}.frame_id", stream_id=stream_id, frame_id=frame_id_raw, manifest_position=position)
            )

    index_raw = raw.get("index")
    index: int | None = None
    if "index" not in raw:
        issues.append(
            _issue(InputIssueCode.MISSING_FIELD, "Required field 'index' is missing.", field_path=f"{prefix}.index", stream_id=stream_id, frame_id=frame_id, manifest_position=position)
        )
    elif isinstance(index_raw, bool) or not isinstance(index_raw, int):
        issues.append(
            _issue(InputIssueCode.INVALID_FRAME_INDEX, "index must be a non-negative integer and not bool.", field_path=f"{prefix}.index", stream_id=stream_id, frame_id=frame_id, manifest_position=position)
        )
    elif index_raw < 0:
        issues.append(
            _issue(InputIssueCode.INVALID_FRAME_INDEX, "index must be non-negative.", field_path=f"{prefix}.index", stream_id=stream_id, frame_id=frame_id, manifest_position=position)
        )
    else:
        index = index_raw

    image_raw = raw.get("image_path")
    image_path: str | None = None
    resolved_path: Path | None = None
    if "image_path" not in raw:
        issues.append(
            _issue(InputIssueCode.MISSING_FIELD, "Required field 'image_path' is missing.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, manifest_position=position)
        )
    elif not isinstance(image_raw, str):
        issues.append(
            _issue(InputIssueCode.INVALID_FIELD_TYPE, "image_path must be a string.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, manifest_position=position)
        )
    else:
        try:
            image_path = _relative_stream_path(image_raw)
        except (TypeError, ValueError):
            issues.append(
                _issue(InputIssueCode.INVALID_IMAGE_PATH, "image_path violates the relative stream-path contract.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, image_path=image_raw, manifest_position=position)
            )
        if image_path is not None:
            candidate = root.joinpath(*PurePosixPath(image_path).parts)
            try:
                resolved_path = candidate.resolve(strict=False)
            except (OSError, RuntimeError) as error:
                issues.append(
                    _issue(
                        InputIssueCode.PATH_RESOLUTION_ERROR,
                        "Image path could not be resolved.",
                        field_path=f"{prefix}.image_path",
                        stream_id=stream_id,
                        frame_id=frame_id,
                        image_path=image_path,
                        manifest_position=position,
                        cause=error,
                    )
                )
                resolved_path = None
            if resolved_path is not None and not _inside_root(resolved_path, root):
                issues.append(
                    _issue(InputIssueCode.PATH_OUTSIDE_STREAM_ROOT, "Resolved image path is outside stream root.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, image_path=image_path, manifest_position=position)
                )
                resolved_path = None
            suffix = PurePosixPath(image_path).suffix.casefold()
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                issues.append(
                    _issue(InputIssueCode.UNSUPPORTED_IMAGE_EXTENSION, "image_path extension is not supported.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, image_path=image_path, manifest_position=position)
                )
            if resolved_path is not None:
                if not resolved_path.exists():
                    issues.append(
                        _issue(InputIssueCode.IMAGE_NOT_FOUND, "Image file does not exist.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, image_path=image_path, manifest_position=position)
                    )
                elif not resolved_path.is_file():
                    issues.append(
                        _issue(InputIssueCode.IMAGE_NOT_FILE, "Image path is not a file.", field_path=f"{prefix}.image_path", stream_id=stream_id, frame_id=frame_id, image_path=image_path, manifest_position=position)
                    )

    notes = _optional_text(raw.get("notes"), f"{prefix}.notes", issues, frame_id=frame_id, position=position)
    metadata = _optional_metadata(raw.get("metadata"), f"{prefix}.metadata", issues, frame_id=frame_id, position=position)
    if frame_id is None or index is None or image_path is None or resolved_path is None:
        return None
    return ManifestFrameEntry(
        frame_id=frame_id,
        index=index,
        manifest_position=position,
        image_path=image_path,
        resolved_path=resolved_path,
        notes=notes,
        metadata=metadata,
    )


def load_manifest(request: ManifestLoadRequest) -> ManifestLoadResult:
    if not isinstance(request, ManifestLoadRequest):
        raise TypeError("request must be ManifestLoadRequest.")
    root = _resolve_stream_root(request)
    manifest_path, raw_bytes = _read_manifest(request, root)
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise StreamInputError(
            [_issue(InputIssueCode.MANIFEST_INVALID_UTF8, "Manifest must be UTF-8 with optional BOM.", field_path="manifest", cause=error)]
        ) from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise StreamInputError(
            [
                _issue(
                    InputIssueCode.MANIFEST_INVALID_JSON,
                    f"Manifest is not valid JSON at line {error.lineno}, column {error.colno}.",
                    field_path="manifest",
                    cause=error,
                )
            ]
        ) from None
    if not isinstance(data, Mapping):
        raise StreamInputError(
            [_issue(InputIssueCode.MANIFEST_ROOT_NOT_OBJECT, "Manifest root must be a JSON object.", field_path="manifest")]
        )

    issues: list[InputIssue] = []
    for name in sorted(set(data) - _ROOT_FIELDS, key=lambda item: str(item).encode("utf-8")):
        issues.append(
            _issue(InputIssueCode.UNKNOWN_FIELD, f"Unknown manifest field {name!r}.", field_path=str(name))
        )

    schema = _required(data, "schema_version", issues)
    if schema is not None:
        if not isinstance(schema, str):
            issues.append(_issue(InputIssueCode.INVALID_FIELD_TYPE, "schema_version must be a string.", field_path="schema_version"))
        elif schema != STREAM_INPUT_SCHEMA_VERSION:
            issues.append(_issue(InputIssueCode.UNSUPPORTED_SCHEMA_VERSION, f"Only {STREAM_INPUT_SCHEMA_VERSION!r} is supported.", field_path="schema_version"))

    stream_id_raw = _required(data, "stream_id", issues)
    stream_id: str | None = None
    if stream_id_raw is not None:
        if not isinstance(stream_id_raw, str):
            issues.append(_issue(InputIssueCode.INVALID_FIELD_TYPE, "stream_id must be a string.", field_path="stream_id"))
        else:
            try:
                _require_identifier(stream_id_raw, "stream_id")
                stream_id = stream_id_raw
            except (TypeError, ValueError):
                issues.append(_issue(InputIssueCode.INVALID_STREAM_ID, "stream_id is not a valid identifier.", field_path="stream_id"))

    ordering = _required(data, "ordering", issues)
    if ordering is not None:
        if not isinstance(ordering, str):
            issues.append(_issue(InputIssueCode.INVALID_FIELD_TYPE, "ordering must be a string.", field_path="ordering", stream_id=stream_id))
        elif ordering != "manifest":
            issues.append(_issue(InputIssueCode.INVALID_ORDERING, "ordering must equal 'manifest'.", field_path="ordering", stream_id=stream_id))

    frames_raw = _required(data, "frames", issues)
    frames: list[ManifestFrameEntry] = []
    if frames_raw is not None:
        if not isinstance(frames_raw, list):
            issues.append(_issue(InputIssueCode.INVALID_FIELD_TYPE, "frames must be an array.", field_path="frames", stream_id=stream_id))
        elif not frames_raw:
            issues.append(_issue(InputIssueCode.EMPTY_FRAMES, "frames must not be empty.", field_path="frames", stream_id=stream_id))
        else:
            for position, raw_frame in enumerate(frames_raw):
                frame = _validate_frame(raw_frame, position, root, stream_id, issues)
                if frame is not None:
                    frames.append(frame)

    seen_ids: dict[str, int] = {}
    seen_indices: dict[int, int] = {}
    for frame in frames:
        position = frame.manifest_position
        if frame.frame_id in seen_ids:
            issues.append(
                _issue(InputIssueCode.DUPLICATE_FRAME_ID, "frame_id values must be unique.", field_path=f"frames[{position}].frame_id", stream_id=stream_id, frame_id=frame.frame_id, manifest_position=position)
            )
        else:
            seen_ids[frame.frame_id] = position
        if frame.index in seen_indices:
            issues.append(
                _issue(InputIssueCode.DUPLICATE_FRAME_INDEX, "frame indices must be unique.", field_path=f"frames[{position}].index", stream_id=stream_id, frame_id=frame.frame_id, manifest_position=position)
            )
        else:
            seen_indices[frame.index] = position

    scene_description = _optional_text(data.get("scene_description"), "scene_description", issues)
    notes = _optional_text(data.get("notes"), "notes", issues)
    metadata = _optional_metadata(data.get("metadata"), "metadata", issues)
    if issues:
        raise StreamInputError(issues)
    if stream_id is None or ordering != "manifest" or not frames:
        raise RuntimeError("Manifest validation reached an inconsistent success state.")
    return ManifestLoadResult(
        stream_root=root,
        manifest_path=manifest_path,
        manifest_digest=_fingerprint(raw_bytes),
        producer=request.producer,
        stream_id=stream_id,
        ordering=ordering,
        frames=tuple(sorted(frames, key=lambda frame: frame.index)),
        scene_description=scene_description,
        notes=notes,
        metadata=metadata,
    )


__all__ = [
    "STREAM_INPUT_SCHEMA_VERSION",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_IMAGE_FORMATS",
    "InputIssue",
    "InputIssueCode",
    "ManifestFrameEntry",
    "ManifestLoadRequest",
    "ManifestLoadResult",
    "StreamInputError",
    "load_manifest",
]
