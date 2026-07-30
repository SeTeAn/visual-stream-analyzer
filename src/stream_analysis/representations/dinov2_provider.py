"""Strictly offline local DINOv2 batch provider with verified assets."""

from __future__ import annotations

import hashlib
import http.client
import importlib
import math
import os
import socket
import threading
import time
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

import numpy as np


DINO_MODEL_NAME: Final = "dinov2_vits14"
DINO_EMBEDDING_DIMENSION: Final = 384
DINO_MODEL_EMBEDDING_DIMENSIONS: Final = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
DINO_PROVIDER_ID: Final = "local_dinov2_torch_provider"
DINO_PROVIDER_VERSION: Final = "1.2"
DINO_MODEL_VERSION: Final = "dinov2-vits14-pretrained-1.0"
OFFLINE_GUARD_VERSION: Final = "python_network_guard_v1"

_OFFLINE_GUARD_LOCK = threading.RLock()

DevicePolicy = Literal["cpu", "cuda", "auto"]


class DinoV2ProviderError(RuntimeError):
    """Structured provider/readiness failure; never triggers a network fallback."""

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


@dataclass(frozen=True, slots=True)
class DinoV2ModelSpec:
    """Semantic identity of the frozen local model assets."""

    expected_checkpoint_sha256: str
    expected_source_tree_fingerprint: str
    expected_checkpoint_size_bytes: int | None = None
    model_name: str = DINO_MODEL_NAME
    embedding_dimension: int = DINO_EMBEDDING_DIMENSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_checkpoint_sha256",
            _validate_sha256(self.expected_checkpoint_sha256, "expected_checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "expected_source_tree_fingerprint",
            _validate_sha256(
                self.expected_source_tree_fingerprint,
                "expected_source_tree_fingerprint",
            ),
        )
        if self.expected_checkpoint_size_bytes is not None:
            size = self.expected_checkpoint_size_bytes
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("expected_checkpoint_size_bytes must be a positive integer or None.")
        expected_dimension = DINO_MODEL_EMBEDDING_DIMENSIONS.get(self.model_name)
        if expected_dimension is None:
            raise ValueError("model_name must identify a supported official DINOv2 backbone.")
        if self.embedding_dimension != expected_dimension:
            raise ValueError(
                f"embedding_dimension must be {expected_dimension} for {self.model_name!r}."
            )


@dataclass(frozen=True, slots=True, eq=False)
class DinoV2ProviderOutput:
    """Normalized embeddings and runtime details for one stable input batch."""

    embeddings: np.ndarray
    runtime_details: Mapping[str, object] = field(hash=False)
    warning_code: str | None = None
    embedding_dimension: int = DINO_EMBEDDING_DIMENSION

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings)
        if embeddings.dtype != np.dtype(np.float32):
            raise TypeError("embeddings dtype must be float32; implicit conversion is forbidden.")
        embeddings = np.ascontiguousarray(embeddings)
        if (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"embeddings must have shape (batch, {self.embedding_dimension})."
            )
        if embeddings.shape[0] <= 0:
            raise ValueError("embeddings batch must not be empty.")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values.")
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("embeddings must be L2-normalized.")
        if self.warning_code is not None and not self.warning_code:
            raise ValueError("warning_code must be non-empty or None.")
        embeddings.flags["WRITEABLE"] = False
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "runtime_details", MappingProxyType(dict(self.runtime_details)))


class DinoV2BatchProviderProtocol(Protocol):
    model_spec: DinoV2ModelSpec
    requested_device: DevicePolicy
    batch_size: int

    @property
    def resolved_device(self) -> str | None: ...

    def embed_batch(self, normalized_batch: np.ndarray) -> DinoV2ProviderOutput: ...

    def provider_metadata(self) -> Mapping[str, object]: ...

    def model_metadata(self) -> Mapping[str, object]: ...


def checkpoint_sha256(path: str | Path) -> str:
    """Return lowercase SHA-256 for one local checkpoint file."""

    checkpoint = Path(path)
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DinoV2ProviderError(
            "CHECKPOINT_READ_FAILED",
            "DINOv2 checkpoint could not be read.",
            {"path": checkpoint.as_posix(), "cause": type(error).__name__},
        ) from error
    return digest.hexdigest()


def source_tree_manifest_bytes(source_dir: str | Path) -> bytes:
    """Return the canonical LF-separated source-tree hash manifest.

    Relative paths use POSIX separators. ``__pycache__`` and ``*.pyc`` are
    excluded, paths are ordered by their UTF-8 bytes, and no trailing LF is
    emitted.
    """

    root = Path(source_dir)
    if not root.exists():
        raise DinoV2ProviderError(
            "MODEL_SOURCE_MISSING",
            "DINOv2 local source directory does not exist.",
            {"path": root.as_posix()},
        )
    if not root.is_dir():
        raise DinoV2ProviderError(
            "MODEL_SOURCE_NOT_DIRECTORY",
            "DINOv2 local source path is not a directory.",
            {"path": root.as_posix()},
        )
    resolved_root = root.resolve()
    files: list[tuple[str, Path]] = []
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix.casefold() == ".pyc":
                continue
            if path.is_symlink():
                raise DinoV2ProviderError(
                    "MODEL_SOURCE_SYMLINK_FORBIDDEN",
                    "DINOv2 source tree must not contain symbolic links.",
                    {"path": relative.as_posix()},
                )
            if not path.is_file():
                continue
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as error:
                raise DinoV2ProviderError(
                    "MODEL_SOURCE_PATH_ESCAPE",
                    "DINOv2 source file resolves outside the source tree.",
                    {"path": relative.as_posix()},
                ) from error
            files.append((relative.as_posix(), path))
    except DinoV2ProviderError:
        raise
    except OSError as error:
        raise DinoV2ProviderError(
            "MODEL_SOURCE_READ_FAILED",
            "DINOv2 source tree could not be enumerated.",
            {"path": root.as_posix(), "cause": type(error).__name__},
        ) from error
    if not files:
        raise DinoV2ProviderError(
            "MODEL_SOURCE_EMPTY",
            "DINOv2 source tree contains no fingerprinted files.",
            {"path": root.as_posix()},
        )

    entries: list[str] = []
    for relative, path in sorted(files, key=lambda item: item[0].encode("utf-8")):
        entries.append(f"{relative}:{checkpoint_sha256(path)}")
    return "\n".join(entries).encode("utf-8")


def source_tree_fingerprint(source_dir: str | Path) -> str:
    """Return lowercase SHA-256 of the canonical source-tree manifest."""

    return hashlib.sha256(source_tree_manifest_bytes(source_dir)).hexdigest()


def resolve_device(torch_module: Any, policy: DevicePolicy) -> tuple[str, str | None]:
    """Resolve the immutable run device and optional explicit fallback warning."""

    if policy not in {"cpu", "cuda", "auto"}:
        raise ValueError("device policy must be cpu, cuda or auto.")
    if policy == "cpu":
        return "cpu", None
    cuda_available = bool(torch_module.cuda.is_available())
    if policy == "cuda":
        if not cuda_available:
            raise DinoV2ProviderError(
                "DEVICE_UNAVAILABLE",
                "CUDA was explicitly requested but is unavailable.",
                {"requested_device": "cuda"},
            )
        return "cuda", None
    if cuda_available:
        return "cuda", None
    return "cpu", "DEVICE_FALLBACK_CPU"


@contextmanager
def _strict_offline_network_guard(torch_module: Any):
    """Block Python network and torch download APIs during local model loading.

    The patched APIs are process-global, so model loading is serialized for the
    short lifetime of the guard and every patch is restored in ``finally``.
    """

    def blocked(api_name: str):
        def deny(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise DinoV2ProviderError(
                "NETWORK_ACCESS_BLOCKED",
                "Network access is forbidden during local DINOv2 loading.",
                {"api": api_name, "guard_version": OFFLINE_GUARD_VERSION},
            )

        return deny

    patch_targets: list[tuple[object, str, str]] = [
        (urllib.request, "urlopen", "urllib.request.urlopen"),
        (urllib.request, "urlretrieve", "urllib.request.urlretrieve"),
        (socket, "create_connection", "socket.create_connection"),
        (socket, "getaddrinfo", "socket.getaddrinfo"),
        (socket.socket, "connect", "socket.socket.connect"),
        (socket.socket, "connect_ex", "socket.socket.connect_ex"),
        (socket.socket, "sendto", "socket.socket.sendto"),
        (http.client.HTTPConnection, "connect", "http.client.HTTPConnection.connect"),
        (http.client.HTTPSConnection, "connect", "http.client.HTTPSConnection.connect"),
    ]
    for attribute in ("download_url_to_file", "load_state_dict_from_url"):
        if hasattr(torch_module.hub, attribute):
            patch_targets.append(
                (torch_module.hub, attribute, f"torch.hub.{attribute}")
            )

    originals: list[tuple[object, str, object]] = []
    with _OFFLINE_GUARD_LOCK:
        try:
            for owner, attribute, api_name in patch_targets:
                originals.append((owner, attribute, getattr(owner, attribute)))
                setattr(owner, attribute, blocked(api_name))
            yield
        finally:
            for owner, attribute, original in reversed(originals):
                setattr(owner, attribute, original)


class LocalDinoV2Provider:
    """Lazy local-only official DINOv2 provider with stable batch ordering."""

    def __init__(
        self,
        *,
        source_dir: str | Path,
        checkpoint_path: str | Path,
        expected_checkpoint_sha256: str,
        expected_source_tree_fingerprint: str,
        expected_checkpoint_size_bytes: int | None = None,
        model_name: str = DINO_MODEL_NAME,
        embedding_dimension: int = DINO_EMBEDDING_DIMENSION,
        device_policy: DevicePolicy = "auto",
        batch_size: int = 32,
    ) -> None:
        if device_policy not in {"cpu", "cuda", "auto"}:
            raise ValueError("device_policy must be cpu, cuda or auto.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.source_dir = Path(source_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_spec = DinoV2ModelSpec(
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_source_tree_fingerprint=expected_source_tree_fingerprint,
            expected_checkpoint_size_bytes=expected_checkpoint_size_bytes,
            model_name=model_name,
            embedding_dimension=embedding_dimension,
        )
        self.requested_device: DevicePolicy = device_policy
        self.batch_size = batch_size
        self._resolved_device: str | None = None
        self._fallback_warning: str | None = None
        self._torch: Any | None = None
        self._model: Any | None = None
        self._model_load_seconds: float | None = None
        self._actual_checkpoint_sha256: str | None = None
        self._actual_source_fingerprint: str | None = None
        self._checkpoint_size_bytes: int | None = None

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    def validate_assets(self) -> None:
        """Validate local assets without importing torch or loading the model."""

        if not self.source_dir.exists():
            raise DinoV2ProviderError(
                "MODEL_SOURCE_MISSING",
                "DINOv2 local source directory does not exist.",
                {"path": self.source_dir.as_posix()},
            )
        if not self.source_dir.is_dir():
            raise DinoV2ProviderError(
                "MODEL_SOURCE_NOT_DIRECTORY",
                "DINOv2 local source path is not a directory.",
                {"path": self.source_dir.as_posix()},
            )
        hubconf = self.source_dir / "hubconf.py"
        if not hubconf.is_file():
            raise DinoV2ProviderError(
                "MODEL_SOURCE_INVALID",
                "DINOv2 local source tree must contain hubconf.py.",
                {"path": hubconf.as_posix()},
            )
        if not self.checkpoint_path.exists():
            raise DinoV2ProviderError(
                "MODEL_WEIGHTS_MISSING",
                "DINOv2 local checkpoint does not exist.",
                {"path": self.checkpoint_path.as_posix()},
            )
        if not self.checkpoint_path.is_file():
            raise DinoV2ProviderError(
                "MODEL_WEIGHTS_NOT_FILE",
                "DINOv2 checkpoint path is not a file.",
                {"path": self.checkpoint_path.as_posix()},
            )
        try:
            size = self.checkpoint_path.stat().st_size
        except OSError as error:
            raise DinoV2ProviderError(
                "CHECKPOINT_READ_FAILED",
                "DINOv2 checkpoint metadata could not be read.",
                {"path": self.checkpoint_path.as_posix()},
            ) from error
        expected_size = self.model_spec.expected_checkpoint_size_bytes
        if expected_size is not None and size != expected_size:
            raise DinoV2ProviderError(
                "CHECKPOINT_SIZE_MISMATCH",
                "DINOv2 checkpoint size does not match the expected asset.",
                {"expected": expected_size, "actual": size},
            )
        actual_checkpoint = checkpoint_sha256(self.checkpoint_path)
        if actual_checkpoint != self.model_spec.expected_checkpoint_sha256:
            raise DinoV2ProviderError(
                "CHECKPOINT_HASH_MISMATCH",
                "DINOv2 checkpoint SHA-256 does not match the expected fingerprint.",
                {
                    "expected": self.model_spec.expected_checkpoint_sha256,
                    "actual": actual_checkpoint,
                },
            )
        actual_source = source_tree_fingerprint(self.source_dir)
        if actual_source != self.model_spec.expected_source_tree_fingerprint:
            raise DinoV2ProviderError(
                "SOURCE_TREE_FINGERPRINT_MISMATCH",
                "DINOv2 source-tree fingerprint does not match the expected fingerprint.",
                {
                    "expected": self.model_spec.expected_source_tree_fingerprint,
                    "actual": actual_source,
                },
            )
        self._checkpoint_size_bytes = size
        self._actual_checkpoint_sha256 = actual_checkpoint
        self._actual_source_fingerprint = actual_source

    def prepare(self) -> None:
        """Validate and load the frozen model lazily, strictly from local paths."""

        if self._model is not None:
            return
        self.validate_assets()
        torch_module = _import_torch()
        resolved, warning = resolve_device(torch_module, self.requested_device)
        torch_home_was_present = "TORCH_HOME" in os.environ
        torch_home_before = os.environ.get("TORCH_HOME")
        started = time.perf_counter()
        try:
            with _strict_offline_network_guard(torch_module):
                model = torch_module.hub.load(
                    str(self.source_dir.resolve()),
                    self.model_spec.model_name,
                    source="local",
                    pretrained=False,
                )
                state = torch_module.load(
                    str(self.checkpoint_path.resolve()),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(state, Mapping):
                    raise TypeError("checkpoint payload is not a state-dict mapping")
                model.load_state_dict(state, strict=True)
                model.to(resolved)
                model.eval()
        except DinoV2ProviderError:
            raise
        except Exception as error:
            raise DinoV2ProviderError(
                "MODEL_LOAD_FAILED",
                "Local DINOv2 model could not be loaded.",
                {"cause": type(error).__name__, "resolved_device": resolved},
            ) from error
        finally:
            torch_home_changed = (
                ("TORCH_HOME" in os.environ) != torch_home_was_present
                or os.environ.get("TORCH_HOME") != torch_home_before
            )
            if torch_home_changed:
                if torch_home_was_present:
                    assert torch_home_before is not None
                    os.environ["TORCH_HOME"] = torch_home_before
                else:
                    os.environ.pop("TORCH_HOME", None)
                raise DinoV2ProviderError(
                    "TORCH_HOME_MUTATION_DETECTED",
                    "Local model loading mutated TORCH_HOME; its original state was restored.",
                    {"originally_present": torch_home_was_present, "restored": True},
                )
        self._torch = torch_module
        self._model = model
        self._resolved_device = resolved
        self._fallback_warning = warning
        self._model_load_seconds = time.perf_counter() - started

    def embed_batch(self, normalized_batch: np.ndarray) -> DinoV2ProviderOutput:
        """Return float32 L2-normalized CLS embeddings in input order."""

        batch = np.ascontiguousarray(normalized_batch)
        if batch.dtype != np.dtype(np.float32):
            raise DinoV2ProviderError(
                "INVALID_INPUT_DTYPE",
                "DINOv2 provider input must use float32.",
                {"dtype": str(batch.dtype)},
            )
        if batch.ndim != 4 or batch.shape[0] <= 0 or batch.shape[1] != 3:
            raise DinoV2ProviderError(
                "INVALID_INPUT_SHAPE",
                "DINOv2 provider input must have shape B x 3 x H x W.",
                {"shape": tuple(batch.shape)},
            )
        if not np.isfinite(batch).all():
            raise DinoV2ProviderError("NONFINITE_INPUT", "DINOv2 input contains NaN or infinity.")
        self.prepare()
        assert self._torch is not None and self._model is not None and self._resolved_device is not None
        torch_module = self._torch
        started = time.perf_counter()
        try:
            tensor = torch_module.from_numpy(batch).to(
                device=self._resolved_device,
                dtype=torch_module.float32,
            )
            with _strict_offline_network_guard(torch_module):
                with torch_module.inference_mode():
                    output = self._model(tensor)
            output = _model_tensor(output)
            if output.ndim != 2 or tuple(output.shape) != (
                batch.shape[0],
                self.model_spec.embedding_dimension,
            ):
                raise DinoV2ProviderError(
                    "INVALID_EMBEDDING_SHAPE",
                    "DINOv2 output shape does not match the selected backbone.",
                    {"shape": tuple(output.shape)},
                )
            if output.dtype != torch_module.float32:
                raise DinoV2ProviderError(
                    "INVALID_EMBEDDING_DTYPE",
                    "DINOv2 output must use float32.",
                    {"dtype": str(output.dtype)},
                )
            if not bool(torch_module.isfinite(output).all()):
                raise DinoV2ProviderError(
                    "NONFINITE_EMBEDDING",
                    "DINOv2 output contains NaN or infinity.",
                )
            norms = torch_module.linalg.vector_norm(output, ord=2, dim=1, keepdim=True)
            if bool((norms <= 0).any()) or not bool(torch_module.isfinite(norms).all()):
                raise DinoV2ProviderError(
                    "ZERO_NORM_EMBEDDING",
                    "DINOv2 output contains an invalid vector norm.",
                )
            normalized = output / norms
            normalized_norms = torch_module.linalg.vector_norm(normalized, ord=2, dim=1)
            if not bool(
                torch_module.allclose(
                    normalized_norms,
                    torch_module.ones_like(normalized_norms),
                    rtol=0.0,
                    atol=1e-5,
                )
            ):
                raise DinoV2ProviderError(
                    "L2_NORMALIZATION_FAILED",
                    "DINOv2 output could not be L2-normalized within tolerance.",
                )
            embeddings = normalized.detach().to(device="cpu", dtype=torch_module.float32).numpy().copy()
        except DinoV2ProviderError:
            raise
        except Exception as error:
            code = "CUDA_OUT_OF_MEMORY" if "out of memory" in str(error).casefold() else "INFERENCE_FAILED"
            raise DinoV2ProviderError(
                code,
                "DINOv2 batch inference failed without device fallback.",
                {"cause": type(error).__name__, "resolved_device": self._resolved_device},
            ) from error
        inference_seconds = time.perf_counter() - started
        details = {
            **self._runtime_details(),
            "batch_item_count": batch.shape[0],
            "inference_seconds": inference_seconds,
        }
        return DinoV2ProviderOutput(
            embeddings=embeddings,
            runtime_details=details,
            warning_code=self._fallback_warning,
            embedding_dimension=self.model_spec.embedding_dimension,
        )

    def provider_metadata(self) -> Mapping[str, object]:
        details: dict[str, object] = {
            "provider_id": DINO_PROVIDER_ID,
            "provider_version": DINO_PROVIDER_VERSION,
            "loading_mode": "torch_hub_local_source_pretrained_false",
            "network_access": "forbidden",
            "offline_guard_version": OFFLINE_GUARD_VERSION,
            "source_dir": self.source_dir.resolve().as_posix(),
            "checkpoint_path": self.checkpoint_path.resolve().as_posix(),
            "requested_device": self.requested_device,
            "resolved_device": self._resolved_device,
            "batch_size": self.batch_size,
            "dtype": "float32",
            "torch_home_modified": False,
        }
        if self._torch is not None:
            details.update(self._runtime_details())
        return details

    def model_metadata(self) -> Mapping[str, object]:
        model_version = f"{self.model_spec.model_name.replace('_', '-')}-pretrained-1.0"
        return {
            "model_name": self.model_spec.model_name,
            "model_version": model_version,
            "embedding_dimension": self.model_spec.embedding_dimension,
            "output_token": "cls",
            "frozen": True,
            "expected_checkpoint_sha256": self.model_spec.expected_checkpoint_sha256,
            "actual_checkpoint_sha256": self._actual_checkpoint_sha256,
            "expected_source_tree_fingerprint": self.model_spec.expected_source_tree_fingerprint,
            "actual_source_tree_fingerprint": self._actual_source_fingerprint,
            "checkpoint_size_bytes": self._checkpoint_size_bytes,
        }

    def _runtime_details(self) -> Mapping[str, object]:
        if self._torch is None:
            return {}
        torch_module = self._torch
        torchvision_version: str | None
        try:
            torchvision_version = importlib.import_module("torchvision").__version__
        except (ImportError, AttributeError):
            torchvision_version = None
        gpu_name = None
        if self._resolved_device == "cuda":
            gpu_name = torch_module.cuda.get_device_name(0)
        return {
            "requested_device": self.requested_device,
            "resolved_device": self._resolved_device,
            "dtype": "float32",
            "batch_size": self.batch_size,
            "torch_version": str(torch_module.__version__),
            "torchvision_version": torchvision_version,
            "cuda_version": torch_module.version.cuda,
            "gpu_name": gpu_name,
            "model_load_seconds": self._model_load_seconds,
            "deterministic_preprocessing": True,
            "model_eval": True,
            "inference_mode": True,
            "offline_guard_version": OFFLINE_GUARD_VERSION,
        }


def _model_tensor(output: Any) -> Any:
    if isinstance(output, Mapping):
        for key in ("x_norm_clstoken", "cls_token", "embedding"):
            if key in output:
                return output[key]
        raise DinoV2ProviderError(
            "UNSUPPORTED_MODEL_OUTPUT",
            "DINOv2 mapping output does not contain a supported CLS key.",
        )
    if isinstance(output, (tuple, list)):
        if not output:
            raise DinoV2ProviderError("UNSUPPORTED_MODEL_OUTPUT", "DINOv2 output is empty.")
        return output[0]
    if not hasattr(output, "ndim") or not hasattr(output, "dtype"):
        raise DinoV2ProviderError(
            "UNSUPPORTED_MODEL_OUTPUT",
            "DINOv2 model returned an unsupported output type.",
            {"type": type(output).__name__},
        )
    return output


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise DinoV2ProviderError(
            "TORCH_MISSING",
            "DINOv2 provider requires a compatible local PyTorch environment.",
        ) from error


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field_name} must contain exactly 64 hexadecimal digits.")
    return normalized


__all__ = [
    "DINO_EMBEDDING_DIMENSION",
    "DINO_MODEL_EMBEDDING_DIMENSIONS",
    "DINO_MODEL_NAME",
    "DINO_MODEL_VERSION",
    "DINO_PROVIDER_ID",
    "DINO_PROVIDER_VERSION",
    "OFFLINE_GUARD_VERSION",
    "DevicePolicy",
    "DinoV2BatchProviderProtocol",
    "DinoV2ModelSpec",
    "DinoV2ProviderError",
    "DinoV2ProviderOutput",
    "LocalDinoV2Provider",
    "checkpoint_sha256",
    "resolve_device",
    "source_tree_fingerprint",
    "source_tree_manifest_bytes",
]
