"""In-memory mask records and deterministic mask digests for F05 extraction."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
)


MASK_DIGEST_ENCODING_ID = "stream_analysis.candidate_mask_bits.v1"


@dataclass(frozen=True, slots=True, eq=False)
class CandidateMaskRecord:
    """In-memory local mask owned by candidate extraction."""

    mask_ref: str
    mask_digest: str
    candidate_id: str
    frame_id: str
    coordinate_bbox: BBox
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.mask_ref, str) or not self.mask_ref.strip():
            raise ValueError("mask_ref must be a non-empty token.")
        if any(char.isspace() for char in self.mask_ref):
            raise ValueError("mask_ref must not contain whitespace.")
        if not isinstance(self.mask_digest, str) or not self.mask_digest.startswith("sha256:"):
            raise ValueError("mask_digest must be a sha256 token.")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string.")
        if not isinstance(self.coordinate_bbox, BBox):
            raise TypeError("coordinate_bbox must be BBox.")

        expected_width = _positive_integral_extent(self.coordinate_bbox.width, "bbox width")
        expected_height = _positive_integral_extent(self.coordinate_bbox.height, "bbox height")
        mask = _readonly_bool_mask(self.mask)
        if mask.shape != (expected_height, expected_width):
            raise ValueError("mask shape must match coordinate_bbox height and width.")
        if candidate_mask_digest(mask) != self.mask_digest:
            raise ValueError("mask_digest must match the local mask shape and contents.")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True, slots=True)
class CandidateExtractionSnapshot:
    """Canonical extraction result plus local in-memory masks."""

    result: CandidateExtractionResult
    masks: Mapping[str, CandidateMaskRecord]

    def __post_init__(self) -> None:
        if not isinstance(self.result, CandidateExtractionResult):
            raise TypeError("result must be CandidateExtractionResult.")
        if not isinstance(self.masks, Mapping):
            raise TypeError("masks must be a mapping.")

        frozen: dict[str, CandidateMaskRecord] = {}
        for mask_ref in sorted(self.masks, key=lambda value: value.encode("utf-8")):
            record = self.masks[mask_ref]
            if not isinstance(record, CandidateMaskRecord):
                raise TypeError("masks values must be CandidateMaskRecord.")
            if mask_ref != record.mask_ref:
                raise ValueError("masks keys must equal record.mask_ref values.")
            frozen[mask_ref] = record

        for candidate in self.result.candidates:
            _validate_candidate_mask(candidate, frozen)

        object.__setattr__(self, "masks", MappingProxyType(frozen))

    def mask_for_candidate(self, candidate_id: str) -> CandidateMaskRecord:
        for candidate in self.result.candidates:
            if candidate.candidate_id == candidate_id:
                if candidate.mask is None or candidate.mask.mask_ref is None:
                    raise KeyError(candidate_id)
                return self.masks[candidate.mask.mask_ref]
        raise KeyError(candidate_id)


def candidate_mask_digest(mask: np.ndarray) -> str:
    """Return a deterministic digest that includes mask shape and bit contents."""

    checked = _readonly_bool_mask(mask)
    height, width = checked.shape
    digest = hashlib.sha256()
    digest.update(MASK_DIGEST_ENCODING_ID.encode("ascii"))
    digest.update(b"\0")
    digest.update(height.to_bytes(8, byteorder="little", signed=False))
    digest.update(width.to_bytes(8, byteorder="little", signed=False))
    packed = np.packbits(checked.reshape(-1).astype(np.uint8), bitorder="little")
    digest.update(packed.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _validate_candidate_mask(
    candidate: CandidateRecord,
    masks: Mapping[str, CandidateMaskRecord],
) -> None:
    if candidate.mask is None or candidate.mask.mask_ref is None:
        raise ValueError("Every F05 candidate must carry a valid MaskReference.")
    record = masks.get(candidate.mask.mask_ref)
    if record is None:
        raise ValueError("Every candidate mask_ref must resolve in the snapshot mask store.")
    if record.candidate_id != candidate.candidate_id:
        raise ValueError("CandidateMaskRecord candidate_id must match CandidateRecord.")
    if record.frame_id != candidate.frame_id:
        raise ValueError("CandidateMaskRecord frame_id must match CandidateRecord.")
    if record.coordinate_bbox != candidate.bbox:
        raise ValueError("CandidateMaskRecord bbox must match CandidateRecord bbox.")
    if record.mask_digest != candidate.mask.mask_digest:
        raise ValueError("CandidateMaskRecord digest must match MaskReference digest.")


def _readonly_bool_mask(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("mask must be a NumPy ndarray.")
    if value.ndim != 2:
        raise ValueError("mask must have shape (height, width).")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("mask height and width must be positive.")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError("mask dtype must be bool.")
    result = np.ascontiguousarray(value.astype(np.bool_, copy=True))
    result.flags["WRITEABLE"] = False
    return result


def _positive_integral_extent(value: float, field_name: str) -> int:
    rounded = int(value)
    if rounded <= 0 or float(rounded) != float(value):
        raise ValueError(f"{field_name} must be a positive integer-valued extent.")
    return rounded


__all__ = [
    "MASK_DIGEST_ENCODING_ID",
    "CandidateExtractionSnapshot",
    "CandidateMaskRecord",
    "candidate_mask_digest",
]
