from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, fields

import stream_analysis
from stream_analysis import (
    PRIMARY_STREAM_RESULT_SCHEMA_ID,
    CalibrationStatus,
    ChangeEvent,
    ConfidenceMetadata,
    EmissionStatus,
    EventEvidence,
    EventKind,
    EventStatus,
    FrameComparison,
    FramePair,
    ImageSize,
    ProducerProvenance,
    RecordEnvelope,
    RepresentationFamily,
    RepresentationSummary,
    RunStatus,
    RuntimeMetadata,
    StageContext,
    StreamAnalysisResult,
    TypePresenceSummary,
    ValidityStatus,
    VersionedMetadata,
    primary_result_payload,
    serialize_primary_result,
    to_json_compatible,
)


def _result(*, status_summary: dict | None = None) -> StreamAnalysisResult:
    run_id = "run_001"
    return StreamAnalysisResult(
        envelope=RecordEnvelope(
            record_id=run_id,
            schema_version="stream-analysis-result-1.0",
            stream_id="probe_01",
            producer=ProducerProvenance(
                producer_stage="pipeline",
                producer_version="1.0",
                config_version="analysis_config_v1",
                config_digest="sha256:analysis",
            ),
            context=StageContext(),
        ),
        run_id=run_id,
        pipeline_version="1.0",
        run_status=RunStatus.COMPLETED,
        frame_ids=("frame_001",),
        data_provenance=VersionedMetadata(
            identifier="probe_stream",
            version="1.0",
            details={"manifest": "sha256:manifest"},
        ),
        model_provenance=(
            VersionedMetadata(
                identifier="dinov2_model",
                version="source_revision_1",
                details={"checkpoint": "sha256:checkpoint"},
            ),
        ),
        runtime_summary=RuntimeMetadata(
            runtime_id="runtime_001",
            details={"device": "cpu"},
        ),
        candidate_extraction_result_id="candidate_result_001",
        candidate_record_ids=("candidate_001",),
        representation_summaries=(
            RepresentationSummary(
                representation_record_id="representation_001",
                candidate_id="candidate_001",
                frame_id="frame_001",
                family=RepresentationFamily.DINO_V2,
                representation_type="dino_embedding",
                representation_version="1.0",
                input_variant="bbox_crop",
                semantic_config_digest="sha256:representation",
                validity_status=ValidityStatus.VALID,
                embedding_dimension=7,
            ),
        ),
        status_summary=status_summary
        or {"precise_value": 0.12345678901234566, "message": "готово"},
    )


@dataclass(frozen=True)
class UnsupportedDataclass:
    value: int


class SerializationTest(unittest.TestCase):
    def test_json_compatible_conversion_is_deterministic_and_typed(self) -> None:
        first = to_json_compatible(
            {"zeta": (1, 2), "alpha": RunStatus.COMPLETED, "nested": {"b": 2, "a": 1}}
        )
        second = to_json_compatible(
            {"nested": {"a": 1, "b": 2}, "alpha": RunStatus.COMPLETED, "zeta": (1, 2)}
        )
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), ("alpha", "nested", "zeta"))
        self.assertEqual(first["alpha"], "completed")
        self.assertEqual(first["zeta"], [1, 2])

    def test_serialization_rejects_nonfinite_and_unsupported_values(self) -> None:
        invalid_values = (
            float("nan"),
            float("inf"),
            {"unordered"},
            b"bytes",
            object(),
            UnsupportedDataclass(1),
        )
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(
                (TypeError, ValueError)
            ):
                to_json_compatible(value)

    def test_primary_payload_is_compact_and_has_explicit_schema(self) -> None:
        payload = primary_result_payload(_result())
        self.assertEqual(payload["schema_id"], PRIMARY_STREAM_RESULT_SCHEMA_ID)
        representation = payload["result"]["representation_summaries"][0]
        self.assertEqual(representation["embedding_dimension"], 7)
        self.assertNotIn("embedding", representation)
        encoded = serialize_primary_result(_result())
        for forbidden in (
            "DinoEmbeddingPayload",
            "StreamAnalysisResult",
            '"annotation"',
            '"gt_mapping"',
            '"metrics"',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_primary_allowlist_serializes_structured_comparison_and_event(self) -> None:
        frame_pair = FramePair(
            from_frame_id="frame_001",
            from_frame_index=0,
            from_frame_size=ImageSize(100, 80),
            to_frame_id="frame_002",
            to_frame_index=1,
            to_frame_size=ImageSize(100, 80),
        )
        event = ChangeEvent(
            envelope=RecordEnvelope(
                record_id="event_001",
                schema_version="change-event-1.0",
                stream_id="probe_01",
                producer=ProducerProvenance(
                    producer_stage="events",
                    producer_version="1.0",
                    config_version="events_v1",
                    config_digest="sha256:events",
                ),
                context=StageContext(
                    pair_id="comparison_001",
                    type_id="type_001",
                    event_id="event_001",
                ),
            ),
            event_id="event_001",
            comparison_id="comparison_001",
            frame_pair=frame_pair,
            kind=EventKind.PERSISTED,
            predicted_type_id="type_001",
            status=EventStatus.CERTAIN,
            evidence=EventEvidence(
                from_count=1,
                to_count=1,
                from_firm_count=1,
                to_firm_count=1,
                from_member_ids=("candidate_001",),
                to_member_ids=("candidate_002",),
                details={
                    "predicate": "persisted",
                    "normalized_from_center": (0.1, 0.2),
                    "normalized_to_center": (0.2, 0.3),
                },
            ),
            confidence=ConfidenceMetadata(
                value=None,
                calibration_status=CalibrationStatus.NOT_CALIBRATED,
                components={
                    "from_firm_fraction": 1.0,
                    "to_firm_fraction": 1.0,
                },
            ),
            policy_id="event_policy",
            policy_version="1.0",
        )
        presence_from = TypePresenceSummary(
            type_id="type_001",
            candidate_ids=("candidate_001",),
            firm_count=1,
            possible_count=1,
        )
        presence_to = TypePresenceSummary(
            type_id="type_001",
            candidate_ids=("candidate_002",),
            firm_count=1,
            possible_count=1,
        )
        comparison = FrameComparison(
            envelope=RecordEnvelope(
                record_id="comparison_001",
                schema_version="frame-comparison-1.0",
                stream_id="probe_01",
                producer=ProducerProvenance(
                    producer_stage="events",
                    producer_version="1.0",
                    config_version="events_v1",
                    config_digest="sha256:events",
                ),
                context=StageContext(pair_id="comparison_001"),
            ),
            comparison_id="comparison_001",
            frame_pair=frame_pair,
            emission_status=EmissionStatus.PRODUCED,
            representation_variant_id="dino_bbox",
            scorer_id="cosine_scorer",
            scorer_version="1.0",
            event_policy_id="event_policy",
            event_policy_version="1.0",
            accepted_match_ids=(),
            uncertain_match_ids=(),
            unmatched_ids=(),
            from_type_presence={"type_001": presence_from},
            to_type_presence={"type_001": presence_to},
            change_event_ids=("event_001",),
        )
        base = _result()
        values = {item.name: getattr(base, item.name) for item in fields(StreamAnalysisResult)}
        values["frame_ids"] = ("frame_001", "frame_002")
        values["recurring_type_ids"] = ("type_001",)
        values["frame_comparisons"] = (comparison,)
        values["change_events"] = (event,)
        payload = primary_result_payload(StreamAnalysisResult(**values))
        self.assertEqual(
            payload["result"]["change_events"][0]["evidence"]["details"][
                "normalized_from_center"
            ],
            [0.1, 0.2],
        )
        self.assertEqual(
            payload["result"]["change_events"][0]["confidence"]["components"],
            {"from_firm_fraction": 1.0, "to_firm_fraction": 1.0},
        )

    def test_primary_serialization_rejects_metadata_leakage(self) -> None:
        forbidden_keys = (
            "embedding",
            "embedding_vector",
            "dino_embedding_values",
            "feature_vector",
            "working_features",
            "handcrafted_working_values",
            "totally_unrelated_values",
            "annotation_digest",
            "gt_mapping",
            "evaluation_metrics",
            "evaluator_rank_buckets",
            "metrics",
        )
        for key in forbidden_keys:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "must not contain"):
                primary_result_payload(_result(status_summary={key: [1.0, 2.0]}))

    def test_working_float_is_not_rounded(self) -> None:
        value = 0.12345678901234566
        encoded = serialize_primary_result(_result(status_summary={"precise_value": value}))
        decoded = json.loads(encoded)
        self.assertEqual(decoded["result"]["status_summary"]["precise_value"], value)
        self.assertIn(repr(value), encoded)

    def test_json_round_trip_preserves_compact_payload(self) -> None:
        result = _result()
        payload = primary_result_payload(result)
        encoded = serialize_primary_result(result)
        self.assertFalse(encoded.endswith("\n"))
        self.assertIn("готово", encoded)
        self.assertEqual(json.loads(encoded), payload)

    def test_primary_function_accepts_only_stream_analysis_result(self) -> None:
        with self.assertRaisesRegex(TypeError, "StreamAnalysisResult"):
            primary_result_payload({"run_id": "run_001"})  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "StreamAnalysisResult"):
            serialize_primary_result(object())  # type: ignore[arg-type]

    def test_serialization_does_not_use_default_string_fallback(self) -> None:
        unknown = object()
        with self.assertRaises(TypeError):
            to_json_compatible({"unknown": unknown})
        self.assertNotEqual(str(unknown), "")

    def test_serialization_package_exports(self) -> None:
        for name in (
            "PRIMARY_STREAM_RESULT_SCHEMA_ID",
            "to_json_compatible",
            "primary_result_payload",
            "serialize_primary_result",
        ):
            self.assertIn(name, stream_analysis.__all__)
            self.assertTrue(hasattr(stream_analysis, name))

    def test_primary_result_contract_remains_embedding_free(self) -> None:
        names = {item.name for item in fields(StreamAnalysisResult)}
        self.assertTrue(names.isdisjoint({"embedding", "embeddings", "metrics", "gt_mapping"}))


if __name__ == "__main__":
    unittest.main()
