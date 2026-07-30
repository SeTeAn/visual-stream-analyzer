"""Load and verify the fixed Visual Stream Analyzer runtime profile."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

from ..representations import source_tree_fingerprint


PROFILE_SCHEMA_VERSION = "visual-stream-analyzer-profile-1.0"
DEFAULT_PROFILE_ID = "visual_stream_analyzer_v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class RuntimeConfigurationError(ValueError):
    """Raised before inference when the fixed runtime cannot be verified."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeProfile:
    path: Path
    sha256: str
    profile_id: str
    models: Mapping[str, Any]
    pipeline: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeAssets:
    models_root: Path
    grounding_dino_directory: Path
    sam2_directory: Path
    dinov2_source_directory: Path
    dinov2_checkpoint: Path


def repository_root() -> Path:
    """Return the Git repository root that contains the Python package."""

    return Path(__file__).resolve().parents[3]


def default_profile_path() -> Path:
    return repository_root() / "configs" / "visual_stream_analyzer_v1.json"


def default_models_root() -> Path:
    return repository_root() / "models"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise RuntimeConfigurationError(f"{context}.{key} must be an object")
    return result


def _text(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise RuntimeConfigurationError(f"{context}.{key} must be a non-empty string")
    return result


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_runtime_profile(path: Path | None = None) -> RuntimeProfile:
    try:
        profile_path = (default_profile_path() if path is None else Path(path)).resolve(strict=True)
        raw = profile_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeConfigurationError("runtime profile cannot be read") from error
    if not isinstance(payload, dict):
        raise RuntimeConfigurationError("runtime profile must be a JSON object")
    expected = {"schema_version", "profile_id", "models", "pipeline"}
    if set(payload) != expected:
        raise RuntimeConfigurationError(
            f"runtime profile fields must be exactly {sorted(expected)}"
        )
    if payload["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise RuntimeConfigurationError(
            f"runtime profile schema must be {PROFILE_SCHEMA_VERSION}"
        )
    profile_id = _text(payload, "profile_id", "profile")
    if profile_id != DEFAULT_PROFILE_ID:
        raise RuntimeConfigurationError(f"runtime profile_id must be {DEFAULT_PROFILE_ID}")
    models = _mapping(payload, "models", "profile")
    pipeline = _mapping(payload, "pipeline", "profile")
    if set(models) != {"candidate_extractor", "mask_refiner", "representation"}:
        raise RuntimeConfigurationError("runtime profile contains an unexpected model set")
    required_pipeline = {
        "candidate_extraction",
        "mask_refinement",
        "representation",
        "scoring",
        "matching",
        "grouping",
        "events",
    }
    if set(pipeline) != required_pipeline:
        raise RuntimeConfigurationError("runtime profile contains an unexpected pipeline stage set")
    return RuntimeProfile(
        path=profile_path,
        sha256=_canonical_sha256(payload),
        profile_id=profile_id,
        models=_freeze(models),
        pipeline=_freeze(pipeline),
    )


def runtime_profile_from_sections(
    *,
    profile_id: str,
    models: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    paths_include_models_prefix: bool = False,
) -> RuntimeProfile:
    """Build a runtime profile from already loaded fixed configuration sections."""

    copied_models = json.loads(json.dumps(models))
    copied_pipeline = json.loads(json.dumps(pipeline))
    if paths_include_models_prefix:
        for model_key in ("candidate_extractor", "mask_refiner"):
            value = copied_models[model_key]["directory"]
            prefix = "models/"
            if not isinstance(value, str) or not value.startswith(prefix):
                raise RuntimeConfigurationError(
                    f"{model_key}.directory must start with {prefix!r}"
                )
            copied_models[model_key]["directory"] = value[len(prefix):]
        representation = copied_models["representation"]
        for owner, key in (
            (representation, "source_directory"),
            (representation["checkpoint"], "path"),
        ):
            value = owner[key]
            if not isinstance(value, str) or not value.startswith("models/"):
                raise RuntimeConfigurationError(
                    f"representation {key} must start with 'models/'"
                )
            owner[key] = value[len("models/"):]
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "models": copied_models,
        "pipeline": copied_pipeline,
    }
    if set(copied_models) != {"candidate_extractor", "mask_refiner", "representation"}:
        raise RuntimeConfigurationError("runtime sections contain an unexpected model set")
    return RuntimeProfile(
        path=Path("in-memory-runtime-profile.json"),
        sha256=_canonical_sha256(payload),
        profile_id=profile_id,
        models=_freeze(copied_models),
        pipeline=_freeze(copied_pipeline),
    )


def _relative_asset(root: Path, value: str, label: str, *, directory: bool) -> Path:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or "." in posix.parts
    ):
        raise RuntimeConfigurationError(f"{label} must be a portable relative path")
    try:
        candidate = root.joinpath(*posix.parts).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeConfigurationError(f"{label} cannot be resolved") from error
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeConfigurationError(f"{label} escapes models root") from error
    if not candidate.exists():
        raise RuntimeConfigurationError(f"{label} does not exist")
    if directory and not candidate.is_dir():
        raise RuntimeConfigurationError(f"{label} must be a directory")
    if not directory and not candidate.is_file():
        raise RuntimeConfigurationError(f"{label} must be a file")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, specification: Mapping[str, Any], label: str) -> None:
    expected_size = specification.get("size_bytes")
    expected_digest = specification.get("sha256")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise RuntimeConfigurationError(f"{label}.size_bytes must be a non-negative integer")
    if not isinstance(expected_digest, str) or not _DIGEST_RE.fullmatch(expected_digest):
        raise RuntimeConfigurationError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeConfigurationError(f"{label} size does not match the fixed profile")
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        raise RuntimeConfigurationError(f"{label} SHA-256 does not match the fixed profile")


def resolve_runtime_assets(
    profile: RuntimeProfile,
    models_root: Path | None = None,
    *,
    verify: bool = True,
) -> RuntimeAssets:
    try:
        root = (default_models_root() if models_root is None else Path(models_root)).resolve(
            strict=True
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeConfigurationError("models root cannot be accessed") from error
    if not root.is_dir():
        raise RuntimeConfigurationError("models root must be a directory")
    candidate_spec = _mapping(profile.models, "candidate_extractor", "models")
    sam2_spec = _mapping(profile.models, "mask_refiner", "models")
    representation_spec = _mapping(profile.models, "representation", "models")
    candidate_directory = _relative_asset(
        root,
        _text(candidate_spec, "directory", "candidate_extractor"),
        "Grounding DINO directory",
        directory=True,
    )
    sam2_directory = _relative_asset(
        root,
        _text(sam2_spec, "directory", "mask_refiner"),
        "SAM2 directory",
        directory=True,
    )
    source_directory = _relative_asset(
        root,
        _text(representation_spec, "source_directory", "representation"),
        "DINOv2 source directory",
        directory=True,
    )
    checkpoint_spec = _mapping(representation_spec, "checkpoint", "representation")
    checkpoint = _relative_asset(
        root,
        _text(checkpoint_spec, "path", "representation.checkpoint"),
        "DINOv2 checkpoint",
        directory=False,
    )
    if verify:
        for directory, specification, label in (
            (candidate_directory, candidate_spec, "Grounding DINO"),
            (sam2_directory, sam2_spec, "SAM2"),
        ):
            assets = _mapping(specification, "assets", label)
            for filename, file_spec in assets.items():
                if not isinstance(filename, str) or not isinstance(file_spec, Mapping):
                    raise RuntimeConfigurationError(f"{label}.assets must contain file objects")
                asset = _relative_asset(directory, filename, f"{label} asset", directory=False)
                _verify_file(asset, file_spec, f"{label}/{filename}")
        _verify_file(checkpoint, checkpoint_spec, "DINOv2 checkpoint")
        expected_source = representation_spec.get("source_tree_fingerprint_sha256")
        if not isinstance(expected_source, str) or not _DIGEST_RE.fullmatch(expected_source):
            raise RuntimeConfigurationError(
                "representation.source_tree_fingerprint_sha256 must be a lowercase SHA-256 digest"
            )
        actual_source = source_tree_fingerprint(source_directory)
        if actual_source != expected_source:
            raise RuntimeConfigurationError(
                "DINOv2 source fingerprint does not match the fixed profile"
            )
    return RuntimeAssets(
        models_root=root,
        grounding_dino_directory=candidate_directory,
        sam2_directory=sam2_directory,
        dinov2_source_directory=source_directory,
        dinov2_checkpoint=checkpoint,
    )


def require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeConfigurationError("PyTorch is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeConfigurationError("the fixed runtime requires CUDA-enabled PyTorch")


def require_portable_component(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or value.endswith((" ", "."))
        or any(ord(character) < 32 for character in value)
        or any(character in '<>:"|?*' for character in value)
    ):
        raise RuntimeConfigurationError(f"{label} is not a portable file-name component")
    stem = value.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise RuntimeConfigurationError(f"{label} is reserved on Windows")


__all__ = [
    "DEFAULT_PROFILE_ID",
    "PROFILE_SCHEMA_VERSION",
    "RuntimeAssets",
    "RuntimeConfigurationError",
    "RuntimeProfile",
    "default_models_root",
    "default_profile_path",
    "load_runtime_profile",
    "repository_root",
    "require_cuda",
    "require_portable_component",
    "resolve_runtime_assets",
    "runtime_profile_from_sections",
]
