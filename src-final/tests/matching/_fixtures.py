from __future__ import annotations

from dataclasses import dataclass

from stream_analysis import (
    BBox,
    CandidateRecord,
    DinoEmbeddingPayload,
    GeometryFeatureMetadata,
    ImageSize,
    InputQualityMetadata,
    MetricOrientation,
    ProducerProvenance,
    RecordEnvelope,
    RepresentationFamily,
    RepresentationRecord,
    RuntimeMetadata,
    StageContext,
    VersionedMetadata,
)
from stream_analysis.representations import CanonicalScore


def producer(stage: str = "fixture", digest: str = "sha256:fixture") -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0.0",
        config_version="1.0.0",
        config_digest=digest,
    )


def candidate(candidate_id: str, frame_index: int, x: int, y: int = 10, *, flags=()):
    frame_id = f"frame_{frame_index:03d}"
    bbox = BBox(x, y, 10, 10)
    return CandidateRecord(
        envelope=RecordEnvelope(
            record_id=candidate_id,
            schema_version="fixture-1.0",
            stream_id="stream",
            producer=producer(),
            context=StageContext(frame_id=frame_id, candidate_id=candidate_id),
        ),
        candidate_id=candidate_id,
        frame_id=frame_id,
        frame_index=frame_index,
        frame_size=ImageSize(100, 100),
        bbox=bbox,
        center=bbox.center,
        geometry=GeometryFeatureMetadata(
            feature_schema_id="geometry_v1",
            producer_version="1.0.0",
            config_digest="sha256:geometry",
            values={"area": 100.0},
        ),
        candidate_source="fixture",
        quality_flags=tuple(flags),
    )


def representation(item: CandidateRecord, vector=(1.0, 0.0)) -> RepresentationRecord:
    return RepresentationRecord(
        envelope=RecordEnvelope(
            record_id=f"representation:{item.candidate_id}",
            schema_version="fixture-1.0",
            stream_id="stream",
            producer=producer("representation", "sha256:representation"),
            context=StageContext(frame_id=item.frame_id, candidate_id=item.candidate_id),
        ),
        candidate_id=item.candidate_id,
        frame_id=item.frame_id,
        family=RepresentationFamily.DINO_V2,
        representation_type="dino_embedding",
        representation_version="1.0.0",
        input_variant="fixture_variant",
        semantic_config_digest="sha256:representation",
        payload=DinoEmbeddingPayload(
            embedding=tuple(vector),
            embedding_dimension=len(vector),
            l2_normalized=True,
        ),
        preprocessing_metadata=VersionedMetadata(identifier="fixture_preprocess", version="1.0.0"),
        provider_metadata=VersionedMetadata(identifier="fixture_provider", version="1.0.0"),
        model_metadata=VersionedMetadata(identifier="fixture_model", version="1.0.0"),
        runtime_metadata=RuntimeMetadata(runtime_id="fixture_runtime"),
        input_quality_metadata=InputQualityMetadata(),
    )


@dataclass(frozen=True)
class TableScorer:
    scores: dict[tuple[str, str], float]
    scorer_id: str = "table_scorer_v1"
    scorer_version: str = "1.0.0"
    family: RepresentationFamily = RepresentationFamily.DINO_V2
    metric_orientation: MetricOrientation = MetricOrientation.HIGHER_IS_BETTER

    def score(self, left, right):
        key = tuple(sorted((left.candidate_id, right.candidate_id)))
        value = self.scores.get(key, 0.0)
        return CanonicalScore(
            raw_metric_name="table_similarity",
            raw_metric_value=value,
            metric_orientation=MetricOrientation.HIGHER_IS_BETTER,
            visual_score=value,
        )
