from __future__ import annotations

import unittest
from pathlib import Path

from stream_analysis.contracts import ImageSize
from stream_analysis.evaluation import ExpectedChangeEvent, StreamAnnotation
from stream_analysis.evaluation.runner import _evaluate_events


class EventMetricsTest(unittest.TestCase):
    def test_gt_events_without_predictions_have_zero_f1_not_null(self) -> None:
        annotation = _annotation((
            ExpectedChangeEvent(
                event_id="event_1",
                event_type="persisted",
                visual_type_id="type_a",
                from_frame_id="f1",
                to_frame_id="f2",
            ),
        ))

        result = _evaluate_events(annotation, {}, {"type_mapping": {}}, {"change_events": []})

        self.assertIsNone(result["micro_precision"])
        self.assertEqual(result["micro_recall"], 0.0)
        self.assertEqual(result["micro_f1"], 0.0)
        self.assertEqual(result["macro_f1"], 0.0)
        self.assertIsNone(result["per_event_type"]["persisted"]["precision"])
        self.assertEqual(result["per_event_type"]["persisted"]["recall"], 0.0)
        self.assertEqual(result["per_event_type"]["persisted"]["f1"], 0.0)
        self.assertIsNone(result["per_event_type"]["appeared"]["f1"])

    def test_empty_event_support_remains_not_applicable(self) -> None:
        result = _evaluate_events(_annotation(()), {}, {"type_mapping": {}}, {"change_events": []})

        self.assertIsNone(result["micro_precision"])
        self.assertIsNone(result["micro_recall"])
        self.assertIsNone(result["micro_f1"])
        self.assertIsNone(result["macro_f1"])
        self.assertIsNone(result["per_event_type"]["persisted"]["f1"])

    def test_false_positive_only_event_kind_has_zero_f1(self) -> None:
        result = _evaluate_events(
            _annotation(()),
            {},
            {"type_mapping": {"type_pred": "type_a"}},
            {
                "change_events": [
                    {
                        "event_id": "pred_1",
                        "kind": "appeared",
                        "predicted_type_id": "type_pred",
                        "frame_pair": {"from_frame_id": "f1", "to_frame_id": "f2"},
                        "status": "certain",
                    }
                ]
            },
        )

        self.assertEqual(result["micro_precision"], 0.0)
        self.assertIsNone(result["micro_recall"])
        self.assertEqual(result["micro_f1"], 0.0)
        self.assertEqual(result["macro_f1"], 0.0)
        self.assertEqual(result["per_event_type"]["appeared"]["precision"], 0.0)
        self.assertIsNone(result["per_event_type"]["appeared"]["recall"])
        self.assertEqual(result["per_event_type"]["appeared"]["f1"], 0.0)


def _annotation(events: tuple[ExpectedChangeEvent, ...]) -> StreamAnnotation:
    return StreamAnnotation(
        path=Path("annotation.json"),
        digest_sha256="0" * 64,
        schema_version="test",
        stream_id="test_stream",
        manifest_ref="manifest.json",
        annotation_scope="pilot_development",
        visual_type_ids=("type_a",),
        frame_ids=("f1", "f2"),
        frame_size=ImageSize(width=10, height=10),
        instances=(),
        frame_comparisons=(),
        change_events=events,
        supported_event_types=("persisted", "appeared", "disappeared", "count_changed", "position_changed"),
        raw={},
    )


if __name__ == "__main__":
    unittest.main()
