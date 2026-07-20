from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.cli import EXIT_OK, EXIT_OUTPUT_COLLISION, main
from stream_analysis.evaluation import (
    aggregate_candidate_metrics,
    EvaluationError,
    EvaluationRequest,
    PredictedCandidate,
    evaluate_candidate_predictions,
    evaluate_saved_run,
)
from stream_analysis.evaluation.annotation import AnnotationInstance, StreamAnnotation

try:
    from ..integration._helpers import config_document, create_tiny_stream
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from integration._helpers import config_document, create_tiny_stream


class EvaluationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.stream = create_tiny_stream(self.root / "stream")
        self.annotation = self.stream / "annotation.json"
        self.annotation.write_text(json.dumps(_tiny_annotation()), encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config_document()), encoding="utf-8")
        args = [
            "analyze",
            str(self.stream),
            "--config",
            str(self.config),
            "--representation-family",
            "handcrafted",
            "--variant",
            "handcrafted_bbox_v1",
            "--output-root",
            str(self.root / "outputs"),
            "--run-id",
            "run_1",
        ]
        self.assertEqual(main(args), EXIT_OK)
        self.run_dir = self.root / "outputs" / "runs" / "run_1"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_evaluates_saved_run_and_rejects_collision(self) -> None:
        result = evaluate_saved_run(EvaluationRequest(
            stream_directory=self.stream,
            run_directory=self.run_dir,
            output_root=self.root / "outputs",
            evaluation_id="eval_1",
            annotation_path=self.annotation,
        ))
        report = result.report
        self.assertEqual(report["metrics"]["candidate_extraction"]["status"], "supported")
        self.assertIn("events", report["metrics"])
        self.assertTrue((result.artifacts.evaluation_directory / "evaluation_report.json").exists())
        with self.assertRaises(FileExistsError):
            evaluate_saved_run(EvaluationRequest(
                stream_directory=self.stream,
                run_directory=self.run_dir,
                output_root=self.root / "outputs",
                evaluation_id="eval_1",
                annotation_path=self.annotation,
            ))
        self.assertEqual(
            report["metrics"]["representations"]["status"],
            "not_supported_by_saved_artifacts",
        )
        self.assertEqual(
            report["metrics"]["representations"]["required_optional_artifact"],
            "diagnostics/pair_scores.json",
        )

    def test_evaluator_uses_saved_pair_score_diagnostics(self) -> None:
        args = [
            "analyze",
            str(self.stream),
            "--config",
            str(self.config),
            "--representation-family",
            "handcrafted",
            "--variant",
            "handcrafted_bbox_v1",
            "--diagnostic-level",
            "standard",
            "--output-root",
            str(self.root / "outputs"),
            "--run-id",
            "run_with_pair_scores",
        ]
        self.assertEqual(main(args), EXIT_OK)
        diagnostic_run = self.root / "outputs" / "runs" / "run_with_pair_scores"
        self.assertTrue((diagnostic_run / "diagnostics" / "pair_scores.json").is_file())
        result = evaluate_saved_run(EvaluationRequest(
            stream_directory=self.stream,
            run_directory=diagnostic_run,
            output_root=self.root / "outputs",
            evaluation_id="eval_pair_scores",
            annotation_path=self.annotation,
        ))
        representation = result.report["metrics"]["representations"]
        self.assertEqual(representation["status"], "supported")
        self.assertGreater(representation["pair_count"], 0)
        self.assertIn("average_precision", representation)
        self.assertIn("auroc", representation)
        self.assertIn("retrieval", representation)
        self.assertIn("diagnostic_pair_scores", result.report["artifact_digests"])

    def test_candidate_only_annotation_skips_type_and_event_metrics(self) -> None:
        payload = _tiny_annotation()
        payload.update({
            "annotation_scope": "candidate_extraction_bbox_only",
            "frame_comparisons": [],
            "change_events": [],
            "supported_event_types": [],
        })
        annotation_path = self.root / "candidate_only_annotation.json"
        annotation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = evaluate_saved_run(EvaluationRequest(
            stream_directory=self.stream,
            run_directory=self.run_dir,
            output_root=self.root / "outputs",
            evaluation_id="eval_candidate_only",
            annotation_path=annotation_path,
        ))

        metrics = result.report["metrics"]
        self.assertEqual(metrics["candidate_extraction"]["status"], "supported")
        for stage in ("representations", "matching", "grouping", "events"):
            self.assertEqual(
                metrics[stage]["status"],
                "not_supported_by_annotation_scope",
            )
        saved_report = json.loads(
            (result.artifacts.evaluation_directory / "evaluation_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            saved_report["data"]["annotation_scope"],
            "candidate_extraction_bbox_only",
        )
        self.assertFalse(
            any("pair-score" in item for item in saved_report["limitations"])
        )

    def test_evaluator_rejects_unbound_or_malformed_pair_score_diagnostics(self) -> None:
        args = [
            "analyze",
            str(self.stream),
            "--config",
            str(self.config),
            "--representation-family",
            "handcrafted",
            "--variant",
            "handcrafted_bbox_v1",
            "--diagnostic-level",
            "standard",
            "--output-root",
            str(self.root / "outputs"),
            "--run-id",
            "run_tamper_check",
        ]
        self.assertEqual(main(args), EXIT_OK)
        run_dir = self.root / "outputs" / "runs" / "run_tamper_check"
        artifact_path = run_dir / "diagnostics" / "pair_scores.json"
        original = json.loads(artifact_path.read_text(encoding="utf-8"))

        cases = {
            "foreign_run": {**original, "run_id": "another_run"},
            "foreign_config": {
                **original,
                "analysis_config_digest": "sha256:" + "f" * 64,
            },
            "malformed_row": {**original, "pairs": ["not-an-object"]},
            "incomplete_matrix": {**original, "pairs": original["pairs"][1:]},
            "wrong_endpoint_frame": {
                **original,
                "pairs": [{
                    **original["pairs"][0],
                    "frame_pair": {
                        **original["pairs"][0]["frame_pair"],
                        "from_frame_id": original["pairs"][0]["frame_pair"]["to_frame_id"],
                    },
                }],
            },
            "wrong_representation_ids": {
                **original,
                "pairs": [{
                    **original["pairs"][0],
                    "representation_record_ids": [
                        "representation_foreign",
                        original["pairs"][0]["representation_record_ids"][1],
                    ],
                }],
            },
            "valid_without_score": {
                **original,
                "pairs": [{
                    **original["pairs"][0],
                    "visual_score": None,
                }],
            },
            "invalid_with_score": {
                **original,
                "pairs": [{
                    **original["pairs"][0],
                    "lineage": {
                        **original["pairs"][0]["lineage"],
                        "validity_status": "invalid",
                    },
                }],
            },
            "wrong_semantics": {
                **original,
                "score_semantics": {
                    **original["score_semantics"],
                    "visual_score": "lower_is_more_similar",
                },
            },
            "boolean_score": {
                **original,
                "pairs": [
                    {**original["pairs"][0], "visual_score": True}
                ],
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                artifact_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    evaluate_saved_run(EvaluationRequest(
                        stream_directory=self.stream,
                        run_directory=run_dir,
                        output_root=self.root / "outputs",
                        evaluation_id=f"eval_{name}",
                        annotation_path=self.annotation,
                    ))

        artifact_path.write_text(json.dumps(original), encoding="utf-8")
        primary_path = run_dir / "stream_analysis.json"
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        result_payload = primary.get("result", primary)
        result_payload["artifacts"] = [
            item for item in result_payload["artifacts"]
            if item.get("artifact_kind") != "diagnostic_pair_scores"
        ]
        primary_path.write_text(json.dumps(primary), encoding="utf-8")
        with self.assertRaises(EvaluationError):
            evaluate_saved_run(EvaluationRequest(
                stream_directory=self.stream,
                run_directory=run_dir,
                output_root=self.root / "outputs",
                evaluation_id="eval_unreferenced",
                annotation_path=self.annotation,
            ))

    def test_evaluator_uses_representation_summary_for_invalid_query_policy(self) -> None:
        args = [
            "analyze", str(self.stream), "--config", str(self.config),
            "--representation-family", "handcrafted",
            "--variant", "handcrafted_bbox_v1",
            "--diagnostic-level", "standard",
            "--output-root", str(self.root / "outputs"),
            "--run-id", "run_invalid_query",
        ]
        self.assertEqual(main(args), EXIT_OK)
        run_dir = self.root / "outputs" / "runs" / "run_invalid_query"
        pair_path = run_dir / "diagnostics" / "pair_scores.json"
        pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
        from_candidate_id = pair_payload["pairs"][0]["from_candidate_id"]
        for pair in pair_payload["pairs"]:
            if from_candidate_id in {
                pair["from_candidate_id"], pair["to_candidate_id"]
            }:
                pair["lineage"]["validity_status"] = "invalid"
                pair["visual_score"] = None
        pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")

        primary_path = run_dir / "stream_analysis.json"
        primary_payload = json.loads(primary_path.read_text(encoding="utf-8"))
        for summary in primary_payload["result"]["representation_summaries"]:
            if summary["candidate_id"] == from_candidate_id:
                summary["validity_status"] = "invalid"
        primary_path.write_text(json.dumps(primary_payload), encoding="utf-8")

        result = evaluate_saved_run(EvaluationRequest(
            stream_directory=self.stream,
            run_directory=run_dir,
            output_root=self.root / "outputs",
            evaluation_id="eval_invalid_query",
            annotation_path=self.annotation,
        ))
        retrieval = result.report["metrics"]["representations"]["retrieval"]
        self.assertEqual(retrieval["forward"]["invalid_query_count"], 1)
        self.assertEqual(retrieval["forward"]["map"], 0.0)
        self.assertEqual(retrieval["forward"]["mrr"], 0.0)
        self.assertEqual(retrieval["forward"]["recall@1"], 0.0)
        self.assertEqual(retrieval["reverse"]["invalid_query_count"], 0)

    def test_cli_evaluate(self) -> None:
        args = [
            "evaluate",
            str(self.stream),
            "--run-directory",
            str(self.run_dir),
            "--output-root",
            str(self.root / "outputs"),
            "--evaluation-id",
            "eval_cli",
            "--annotation",
            str(self.annotation),
        ]
        self.assertEqual(main(args), EXIT_OK)
        self.assertEqual(main(args), EXIT_OUTPUT_COLLISION)

    def test_candidate_assignment_maximizes_valid_pair_count_before_iou(self) -> None:
        annotation = _candidate_annotation(
            (
                AnnotationInstance("gt_1", "frame_001", "object_1", BBox(1, 0, 10, 10)),
                AnnotationInstance("gt_2", "frame_001", "object_2", BBox(4, 0, 10, 10)),
            ),
        )
        predicted = (
            PredictedCandidate("pred_1", "frame_001", BBox(1.5, 0, 10, 10), "valid", (), ()),
            PredictedCandidate("pred_2", "frame_001", BBox(0, 0, 10, 10), "valid", (), ()),
        )

        metrics = evaluate_candidate_predictions(
            annotation,
            predicted,
            iou_threshold=0.50,
        )

        # Raw-IoU Hungarian matching would select pred_1->gt_1 and
        # pred_2->gt_2, then lose the latter after thresholding.  The valid
        # matching pred_1->gt_2 and pred_2->gt_1 keeps both pairs.
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_candidate_evaluation_honors_caller_iou_threshold_for_metrics_and_diagnostics(self) -> None:
        annotation = _candidate_annotation((
            AnnotationInstance("gt_1", "frame_001", "object_1", BBox(0, 0, 10, 10)),
        ))
        predicted = (
            PredictedCandidate("pred_1", "frame_001", BBox(2.5, 0, 10, 10), "valid", (), ()),
        )

        at_050 = evaluate_candidate_predictions(annotation, predicted, iou_threshold=0.50)
        at_070 = evaluate_candidate_predictions(annotation, predicted, iou_threshold=0.70)

        self.assertEqual(at_050["primary_iou_threshold"], 0.50)
        self.assertEqual(at_050["tp"], 1)
        self.assertEqual(at_050["diagnostics"]["fragment"], 0)
        self.assertEqual(at_070["primary_iou_threshold"], 0.70)
        self.assertEqual(at_070["tp"], 0)
        self.assertEqual(at_070["diagnostics"]["fragment"], 1)
        for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    evaluate_candidate_predictions(
                        annotation,
                        predicted,
                        iou_threshold=threshold,
                    )["primary_iou_threshold"],
                    threshold,
                )

    def test_candidate_assignment_maximizes_total_iou_after_cardinality(self) -> None:
        annotation = _candidate_annotation(
            (
                AnnotationInstance("gt_1", "frame_001", "object_1", BBox(0, 0, 10, 10)),
                AnnotationInstance("gt_2", "frame_001", "object_2", BBox(2, 0, 10, 10)),
            ),
        )
        predicted = (
            PredictedCandidate("pred_1", "frame_001", BBox(0, 0, 10, 10), "valid", (), ()),
            PredictedCandidate("pred_2", "frame_001", BBox(2, 0, 10, 10), "valid", (), ()),
        )

        metrics = evaluate_candidate_predictions(annotation, predicted, iou_threshold=0.50)

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tp_iou"]["mean"], 1.0)

    def test_candidate_evaluation_rejects_invalid_iou_threshold(self) -> None:
        annotation = _candidate_annotation((
            AnnotationInstance("gt_1", "frame_001", "object_1", BBox(0, 0, 10, 10)),
        ))
        predicted = (
            PredictedCandidate("pred_1", "frame_001", BBox(0, 0, 10, 10), "valid", (), ()),
        )

        for threshold, error_type in ((True, TypeError), (float("nan"), ValueError), (1.1, ValueError)):
            with self.subTest(threshold=threshold):
                with self.assertRaises(error_type):
                    evaluate_candidate_predictions(
                        annotation,
                        predicted,
                        iou_threshold=threshold,
                    )

    def test_candidate_aggregate_handles_no_predictions(self) -> None:
        annotation = _candidate_annotation((
            AnnotationInstance("gt_1", "frame_001", "object_1", BBox(2, 2, 4, 4)),
        ))
        empty = evaluate_candidate_predictions(annotation, (), iou_threshold=0.50)

        pooled = aggregate_candidate_metrics((empty, empty))

        self.assertEqual((pooled["tp"], pooled["fp"], pooled["fn"]), (0, 0, 2))
        self.assertIsNone(pooled["precision"])
        self.assertEqual(pooled["recall"], 0.0)
        self.assertIsNone(pooled["f1"])


def _tiny_annotation() -> dict:
    return {
        "schema_version": "stream-pilot-annotation-0.1",
        "stream_id": "tiny_stream",
        "manifest_ref": "manifest.json",
        "annotation_scope": "pilot_development",
        "frame_size": {"width": 32, "height": 24},
        "visual_types": [{"visual_type_id": "square"}],
        "expected_element_instances": [
            {
                "instance_id": "gt_1",
                "frame_id": "frame_001",
                "visual_type_id": "square",
                "bbox": {"x": 5, "y": 6, "width": 12, "height": 12},
                "characteristic_regions": [],
                "uncertainty": "",
            },
            {
                "instance_id": "gt_2",
                "frame_id": "frame_002",
                "visual_type_id": "square",
                "bbox": {"x": 13, "y": 6, "width": 12, "height": 12},
                "characteristic_regions": [],
                "uncertainty": "",
            },
        ],
        "frame_comparisons": [
            {"from_frame_id": "frame_001", "to_frame_id": "frame_002", "expected_change_event_ids": ["event_1"]}
        ],
        "change_events": [
            {
                "event_id": "event_1",
                "event_type": "persisted",
                "visual_type_id": "square",
                "from_frame_id": "frame_001",
                "to_frame_id": "frame_002",
                "evidence": "",
                "uncertainty": "",
            }
        ],
        "supported_event_types": ["persisted", "appeared", "disappeared", "count_changed", "position_changed"],
        "uncertainty": [],
    }


def _candidate_annotation(instances: tuple[AnnotationInstance, ...]) -> StreamAnnotation:
    return StreamAnnotation(
        path=Path("annotation.json"),
        digest_sha256="0" * 64,
        schema_version="test",
        stream_id="stream",
        manifest_ref="manifest.json",
        annotation_scope="candidate_extraction_bbox_only",
        visual_type_ids=("object_1", "object_2"),
        frame_ids=("frame_001",),
        frame_size=ImageSize(32, 24),
        instances=instances,
        frame_comparisons=(),
        change_events=(),
        supported_event_types=(),
        raw={},
    )


if __name__ == "__main__":
    unittest.main()
