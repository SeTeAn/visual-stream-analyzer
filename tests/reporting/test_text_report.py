from __future__ import annotations

import unittest
from dataclasses import fields

from stream_analysis import (
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
    RunStatus,
    RuntimeMetadata,
    StageContext,
    StreamAnalysisResult,
    TypePresenceSummary,
    VersionedMetadata,
)
from stream_analysis.reporting import build_text_report


def _producer(stage: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0",
        config_version=f"{stage}_v1",
        config_digest=f"sha256:{stage}",
    )


def _envelope(record_id: str, stage: str, context: StageContext = StageContext()) -> RecordEnvelope:
    return RecordEnvelope(
        record_id=record_id,
        schema_version="report-fixture-1.0",
        stream_id="stream_001",
        producer=_producer(stage),
        context=context,
    )


def _pair(from_id: str, from_index: int, to_id: str, to_index: int) -> FramePair:
    return FramePair(
        from_frame_id=from_id,
        from_frame_index=from_index,
        from_frame_size=ImageSize(64, 48),
        to_frame_id=to_id,
        to_frame_index=to_index,
        to_frame_size=ImageSize(64, 48),
    )


def _evidence(from_count: int, to_count: int, shift: float | None = None) -> EventEvidence:
    return EventEvidence(
        from_count=from_count,
        to_count=to_count,
        from_firm_count=from_count,
        to_firm_count=to_count,
        from_member_ids=tuple(f"candidate_from_{index}" for index in range(from_count)),
        to_member_ids=tuple(f"candidate_to_{index}" for index in range(to_count)),
        position_shift_norm=shift,
        details={"predicate": "report_fixture"},
    )


def _event(
    *,
    event_id: str,
    comparison_id: str,
    frame_pair: FramePair,
    kind: EventKind,
    type_id: str,
    evidence: EventEvidence,
) -> ChangeEvent:
    return ChangeEvent(
        envelope=_envelope(
            event_id,
            "events",
            StageContext(pair_id=comparison_id, type_id=type_id, event_id=event_id),
        ),
        event_id=event_id,
        comparison_id=comparison_id,
        frame_pair=frame_pair,
        kind=kind,
        predicted_type_id=type_id,
        status=EventStatus.CERTAIN,
        evidence=evidence,
        confidence=ConfidenceMetadata(
            value=None,
            calibration_status=CalibrationStatus.NOT_CALIBRATED,
        ),
        policy_id="event_policy",
        policy_version="1.0",
    )


def _comparison(comparison_id: str, frame_pair: FramePair, events: tuple[ChangeEvent, ...]) -> FrameComparison:
    type_ids = sorted({event.predicted_type_id for event in events})
    presence = {
        type_id: TypePresenceSummary(
            type_id=type_id,
            candidate_ids=(f"candidate_{type_id}",),
            firm_count=1,
            possible_count=1,
        )
        for type_id in type_ids
    }
    return FrameComparison(
        envelope=_envelope(comparison_id, "events", StageContext(pair_id=comparison_id)),
        comparison_id=comparison_id,
        frame_pair=frame_pair,
        emission_status=EmissionStatus.PRODUCED,
        representation_variant_id="handcrafted_bbox_v1",
        scorer_id="handcrafted_scorer",
        scorer_version="1.0",
        event_policy_id="event_policy",
        event_policy_version="1.0",
        accepted_match_ids=(),
        uncertain_match_ids=(),
        unmatched_ids=(),
        from_type_presence=presence,
        to_type_presence=presence,
        change_event_ids=tuple(event.event_id for event in events),
    )


class TextReportTest(unittest.TestCase):
    def _result(self) -> StreamAnalysisResult:
        first_pair = _pair("frame_001", 0, "frame_002", 1)
        second_pair = _pair("frame_002", 1, "frame_003", 2)
        first_events = (
            _event(
                event_id="event_003",
                comparison_id="comparison_001",
                frame_pair=first_pair,
                kind=EventKind.PERSISTED,
                type_id="type_002",
                evidence=_evidence(1, 1),
            ),
            _event(
                event_id="event_001",
                comparison_id="comparison_001",
                frame_pair=first_pair,
                kind=EventKind.APPEARED,
                type_id="type_003",
                evidence=_evidence(0, 1),
            ),
            _event(
                event_id="event_002",
                comparison_id="comparison_001",
                frame_pair=first_pair,
                kind=EventKind.PERSISTED,
                type_id="type_001",
                evidence=_evidence(1, 1),
            ),
        )
        second_events = (
            _event(
                event_id="event_004",
                comparison_id="comparison_002",
                frame_pair=second_pair,
                kind=EventKind.POSITION_CHANGED,
                type_id="type_001",
                evidence=_evidence(1, 1, shift=0.1171875),
            ),
            _event(
                event_id="event_005",
                comparison_id="comparison_002",
                frame_pair=second_pair,
                kind=EventKind.PERSISTED,
                type_id="type_001",
                evidence=_evidence(1, 1),
            ),
        )
        comparisons = (
            _comparison("comparison_001", first_pair, first_events),
            _comparison("comparison_002", second_pair, second_events),
        )
        return StreamAnalysisResult(
            envelope=_envelope("run_001", "pipeline"),
            run_id="run_001",
            pipeline_version="1.0",
            run_status=RunStatus.COMPLETED,
            frame_ids=("frame_001", "frame_002", "frame_003"),
            data_provenance=VersionedMetadata(identifier="stream", version="1.0"),
            model_provenance=(),
            runtime_summary=RuntimeMetadata(runtime_id="runtime_001"),
            candidate_record_ids=("candidate_001", "candidate_002"),
            recurring_type_ids=("type_001", "type_002", "type_003"),
            frame_comparisons=comparisons,
            change_events=(*first_events, *second_events),
            status_summary={"warning_count": 0, "error_count": 0},
        )

    def test_report_groups_events_by_transition_and_kind_order(self) -> None:
        report = build_text_report(self._result())
        self.assertIn("Отчёт по анализу потока изображений", report)
        self.assertIn(
            "Краткая сводка\n"
            "- Обработано кадров: 3.\n"
            "- Найдено областей-кандидатов: 2.",
            report,
        )
        self.assertIn("События по всему потоку\n- появились в следующем кадре: 1.", report)
        self.assertIn("Переходы между кадрами\n\nframe_001 → frame_002", report)
        first_transition = report.index("frame_001 → frame_002")
        second_transition = report.index("frame_002 → frame_003")
        self.assertLess(first_transition, second_transition)
        self.assertLess(
            report.index("  Появились в следующем кадре:", first_transition),
            report.index("  Продолжили присутствовать:", first_transition),
        )
        self.assertLess(report.index("type_001", first_transition), report.index("type_002", first_transition))
        self.assertIn(
            "  Изменилось положение:\n"
            "  - type_001: количество 1 → 1, нормализованный сдвиг: 0.1171875; "
            "статус: уверенно.",
            report,
        )
        second_block = report[second_transition:]
        self.assertNotIn("  Продолжили присутствовать:\n  - type_001: 1 → 1", second_block)

    def test_report_remains_prediction_only(self) -> None:
        report = build_text_report(self._result()).casefold()
        self.assertIn("не использует разметку (ground truth)", report)
        self.assertIn("не содержит метрики оценки качества", report)
        for forbidden in ("annotation", "gt_mapping"):
            self.assertNotIn(forbidden, report)
        self.assertNotIn("precision:", report)
        self.assertTrue({item.name for item in fields(StreamAnalysisResult)}.isdisjoint({"gt_mapping", "metrics"}))


if __name__ == "__main__":
    unittest.main()
