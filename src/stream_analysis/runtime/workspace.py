"""Private working artifacts used while the shared runtime is executing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredMask:
    raw_path: Path
    cleaned_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredEmbeddings:
    path: Path
    shape: tuple[int, int]


class RuntimeWorkspace:
    """A bounded workspace; callers decide whether it is temporary or auditable."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def _inside(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve(strict=False)
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("runtime artifact path escapes its workspace") from error
        return path

    @staticmethod
    def _candidate_name(candidate_id: str, source_index: int) -> str:
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]
        return f"candidate_{source_index:03d}_{digest}"

    @staticmethod
    def _write_mask(path: Path, mask: np.ndarray) -> None:
        value = np.asarray(mask, dtype=np.bool_)
        if value.ndim != 2 or not value.any():
            raise ValueError("runtime masks must be non-empty two-dimensional arrays")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".png", dir=path.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            Image.fromarray(np.where(value, 255, 0).astype(np.uint8), mode="L").save(
                temporary, format="PNG", optimize=False
            )
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def store_masks(
        self,
        *,
        stream_id: str,
        frame_id: str,
        candidate_id: str,
        source_index: int,
        raw_mask: np.ndarray,
        cleaned_mask: np.ndarray,
    ) -> StoredMask:
        name = self._candidate_name(candidate_id, source_index)
        raw_path = self._inside("masks", stream_id, frame_id, f"{name}.raw.png")
        cleaned_path = self._inside("masks", stream_id, frame_id, f"{name}.cleaned.png")
        if raw_path.exists() or cleaned_path.exists():
            raise FileExistsError(f"runtime mask artifact already exists for {candidate_id}")
        try:
            self._write_mask(raw_path, raw_mask)
            self._write_mask(cleaned_path, cleaned_mask)
        except BaseException:
            raw_path.unlink(missing_ok=True)
            cleaned_path.unlink(missing_ok=True)
            raise
        return StoredMask(raw_path=raw_path, cleaned_path=cleaned_path)

    @staticmethod
    def load_mask(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"))
        if value.ndim != 2 or not np.all((value == 0) | (value == 255)):
            raise ValueError(f"runtime mask is not binary: {path}")
        result = np.ascontiguousarray(value == 255, dtype=np.bool_)
        if not result.any():
            raise ValueError(f"runtime mask is empty: {path}")
        return result

    def store_embeddings(
        self,
        *,
        stream_id: str,
        embeddings: np.ndarray,
    ) -> StoredEmbeddings:
        value = np.asarray(embeddings, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("runtime embeddings must be a finite two-dimensional matrix")
        path = self._inside("analysis", stream_id, "embeddings.npz")
        if path.exists():
            raise FileExistsError(f"runtime embeddings already exist for {stream_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".npz", dir=path.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, embeddings=value)
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return StoredEmbeddings(path=path, shape=tuple(value.shape))


__all__ = ["RuntimeWorkspace", "StoredEmbeddings", "StoredMask"]
