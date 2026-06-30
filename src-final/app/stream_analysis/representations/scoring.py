"""Canonical visual scorer adapters shared by both representation families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..contracts import (
    DinoEmbeddingPayload,
    MetricOrientation,
    RepresentationFamily,
    RepresentationRecord,
    ValidityStatus,
)
from .handcrafted import HandcraftedRepresentationConfig
from .handcrafted_distance import handcrafted_distance

DINO_CANONICAL_SCORER_ID = "dinov2_cosine_canonical_v1"
DINO_CANONICAL_SCORER_VERSION = "1.0.0"
HANDCRAFTED_CANONICAL_SCORER_ID = "handcrafted_bounded_canonical_v1"
HANDCRAFTED_CANONICAL_SCORER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class CanonicalScore:
    """Raw metric and monotonic bounded score; neither is a probability."""

    raw_metric_name: str
    raw_metric_value: float
    metric_orientation: MetricOrientation
    visual_score: float
    details: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_metric_name:
            raise ValueError("raw_metric_name must be non-empty.")
        for field_name in ("raw_metric_value", "visual_score"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite.")
        if not isinstance(self.metric_orientation, MetricOrientation):
            raise TypeError("metric_orientation must be MetricOrientation.")
        if not 0.0 <= float(self.visual_score) <= 1.0:
            raise ValueError("visual_score must be in [0, 1].")
        frozen_details = tuple(sorted((str(name), float(value)) for name, value in self.details))
        if any(not math.isfinite(value) for _name, value in frozen_details):
            raise ValueError("details values must be finite.")
        object.__setattr__(self, "raw_metric_value", float(self.raw_metric_value))
        object.__setattr__(self, "visual_score", float(self.visual_score))
        object.__setattr__(self, "details", frozen_details)


@runtime_checkable
class CanonicalScorer(Protocol):
    scorer_id: str
    scorer_version: str
    family: RepresentationFamily
    metric_orientation: MetricOrientation

    def score(self, left: RepresentationRecord, right: RepresentationRecord) -> CanonicalScore:
        """Return a canonical score for compatible valid records."""


@dataclass(frozen=True, slots=True)
class DinoV2CosineScorer:
    scorer_id: str = DINO_CANONICAL_SCORER_ID
    scorer_version: str = DINO_CANONICAL_SCORER_VERSION
    family: RepresentationFamily = RepresentationFamily.DINO_V2
    metric_orientation: MetricOrientation = MetricOrientation.HIGHER_IS_BETTER

    def score(self, left: RepresentationRecord, right: RepresentationRecord) -> CanonicalScore:
        _validate_pair(left, right, self.family)
        if not isinstance(left.payload, DinoEmbeddingPayload) or not isinstance(
            right.payload, DinoEmbeddingPayload
        ):
            raise TypeError("DINOv2 scorer requires DinoEmbeddingPayload records.")
        if left.payload.embedding_dimension != right.payload.embedding_dimension:
            raise ValueError("DINOv2 embedding dimensions are incompatible.")
        cosine = _cosine(left.payload.embedding, right.payload.embedding)
        cosine = max(-1.0, min(1.0, cosine))
        return CanonicalScore(
            raw_metric_name="cosine_similarity",
            raw_metric_value=cosine,
            metric_orientation=MetricOrientation.HIGHER_IS_BETTER,
            visual_score=(cosine + 1.0) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class HandcraftedCanonicalScorer:
    config: HandcraftedRepresentationConfig
    scorer_id: str = HANDCRAFTED_CANONICAL_SCORER_ID
    scorer_version: str = HANDCRAFTED_CANONICAL_SCORER_VERSION
    family: RepresentationFamily = RepresentationFamily.HANDCRAFTED
    metric_orientation: MetricOrientation = MetricOrientation.LOWER_IS_BETTER

    def __post_init__(self) -> None:
        if not isinstance(self.config, HandcraftedRepresentationConfig):
            raise TypeError("config must be HandcraftedRepresentationConfig.")

    def score(self, left: RepresentationRecord, right: RepresentationRecord) -> CanonicalScore:
        _validate_pair(left, right, self.family)
        result = handcrafted_distance(left, right, self.config)
        details = tuple(
            (f"group_{name}", value) for name, value in result.group_distances.items()
        )
        return CanonicalScore(
            raw_metric_name="handcrafted_distance",
            raw_metric_value=result.distance,
            metric_orientation=MetricOrientation.LOWER_IS_BETTER,
            visual_score=result.similarity,
            details=details,
        )


def _validate_pair(
    left: RepresentationRecord,
    right: RepresentationRecord,
    family: RepresentationFamily,
) -> None:
    if not isinstance(left, RepresentationRecord) or not isinstance(right, RepresentationRecord):
        raise TypeError("Scorer inputs must be RepresentationRecord values.")
    if left.envelope.validity_status is not ValidityStatus.VALID or left.payload is None:
        raise ValueError("left representation must be valid.")
    if right.envelope.validity_status is not ValidityStatus.VALID or right.payload is None:
        raise ValueError("right representation must be valid.")
    if left.family is not family or right.family is not family:
        raise ValueError("Representation family is incompatible with scorer.")
    if left.input_variant != right.input_variant:
        raise ValueError("Representation input variants are incompatible.")
    if left.representation_type != right.representation_type:
        raise ValueError("Representation types are incompatible.")
    if left.representation_version != right.representation_version:
        raise ValueError("Representation versions are incompatible.")
    if left.semantic_config_digest != right.semantic_config_digest:
        raise ValueError("Representation semantic config digests are incompatible.")


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("DINOv2 embedding norm must be positive.")
    return dot / (left_norm * right_norm)


__all__ = [
    "DINO_CANONICAL_SCORER_ID",
    "DINO_CANONICAL_SCORER_VERSION",
    "HANDCRAFTED_CANONICAL_SCORER_ID",
    "HANDCRAFTED_CANONICAL_SCORER_VERSION",
    "CanonicalScore",
    "CanonicalScorer",
    "DinoV2CosineScorer",
    "HandcraftedCanonicalScorer",
]
