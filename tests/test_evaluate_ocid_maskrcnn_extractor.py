from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from stream_analysis.contracts import BBox
from tools.ocid_gate_c1_common import GateC1ContractError
from tools.evaluate_ocid_maskrcnn_extractor import (
    RawPrediction,
    DEFAULT_MATCH_IOU_THRESHOLDS,
    DEFAULT_SCORE_THRESHOLDS,
    aggregate_candidate_metrics,
    evaluate_profile_grid,
    profile_id,
    predictions_at_threshold,
    run_benchmark,
)
from tools.evaluate_ocid_maskrcnn_nms_postprocess import evaluate_saved_predictions


class MaskRCNNExtractorGateTest(unittest.TestCase):
    def test_threshold_and_geometry_variant_preserve_stable_prediction_ids(self) -> None:
        raw = (
            RawPrediction("frame_001", 7, 2, 0.8, 1, BBox(1, 2, 10, 12), BBox(2, 3, 8, 9)),
            RawPrediction("frame_001", 7, 5, 0.2, 1, BBox(20, 20, 5, 5), None),
        )

        bbox = predictions_at_threshold(raw, threshold=0.5, geometry_variant="model_bbox")
        mask = predictions_at_threshold(raw, threshold=0.5, geometry_variant="mask_tight_bbox")

        self.assertEqual(tuple(item.candidate_id for item in bbox), ("maskrcnn:frame_001:002",))
        self.assertEqual(tuple(item.candidate_id for item in mask), ("maskrcnn:frame_001:002",))
        self.assertEqual(bbox[0].bbox, BBox(1, 2, 10, 12))
        self.assertEqual(mask[0].bbox, BBox(2, 3, 8, 9))
        self.assertEqual(bbox[0].frame_index, 7)

    def test_aggregate_candidate_metrics_uses_micro_counts(self) -> None:
        base = {
            "tp_iou": {"count": 1, "mean": 0.75},
            "per_frame": [{"frame_id": "f"}],
            "diagnostics": {
                "duplicate": 0,
                "split": 0,
                "merge": 1,
                "fragment": 0,
                "noise": 0,
                "miss": 1,
            },
        }
        result = aggregate_candidate_metrics(
            (
                {**base, "tp": 1, "fp": 1, "fn": 1},
                {**base, "tp": 1, "fp": 0, "fn": 0},
            )
        )

        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["precision"], 2 / 3)
        self.assertEqual(result["recall"], 2 / 3)
        self.assertEqual(result["weighted_mean_tp_iou"], 0.75)
        self.assertEqual(result["diagnostics"]["merge"], 2)

    def test_class_agnostic_nms_suppresses_cross_label_duplicate(self) -> None:
        raw = (
            RawPrediction("frame_001", 0, 0, 0.9, 1, BBox(0, 0, 10, 10), None),
            RawPrediction("frame_001", 0, 1, 0.8, 47, BBox(1, 1, 10, 10), None),
            RawPrediction("frame_001", 0, 2, 0.7, 2, BBox(30, 30, 5, 5), None),
        )

        candidates = predictions_at_threshold(
            raw,
            threshold=0.1,
            geometry_variant="model_bbox",
            class_agnostic_nms_iou=0.5,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in candidates),
            ("maskrcnn:frame_001:000", "maskrcnn:frame_001:002"),
        )

    def test_pre_registered_defaults_and_profile_id_are_deterministic(self) -> None:
        self.assertEqual(DEFAULT_SCORE_THRESHOLDS, (0.05, 0.10, 0.25, 0.50))
        self.assertEqual(DEFAULT_MATCH_IOU_THRESHOLDS, (0.50, 0.60, 0.70, 0.80, 0.90))
        self.assertEqual(
            profile_id(
                geometry_variant="model_bbox",
                score_threshold=0.05,
                nms_iou_threshold=None,
            ),
            "geometry_model_bbox__score_0.05__nms_none",
        )

    def test_profile_grid_evaluates_every_iou_without_rerunning_predictions(self) -> None:
        raw = (RawPrediction("frame_001", 3, 0, 0.9, 1, BBox(0, 0, 10, 10), BBox(1, 1, 8, 8)),)
        calls: list[tuple[int, float]] = []

        def fake_evaluator(annotation, candidates, *, iou_threshold):
            calls.append((len(candidates), iou_threshold))
            return {"tp": len(candidates), "iou_threshold": iou_threshold}

        with patch(
            "tools.evaluate_ocid_maskrcnn_extractor.evaluate_candidate_predictions",
            side_effect=fake_evaluator,
        ):
            result = evaluate_profile_grid(
                object(),
                raw,
                thresholds=(0.05,),
                match_iou_thresholds=(0.50, 0.90),
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(calls, [(1, 0.50), (1, 0.90), (1, 0.50), (1, 0.90)])
        for identifier, value in result.items():
            self.assertIn("profile_id", value)
            self.assertNotIn("match_iou", identifier)
            self.assertIn("metrics_by_iou", value)
            self.assertEqual(set(value["metrics_by_iou"]), {"0.50", "0.90"})
            self.assertIsNone(value["class_agnostic_nms_iou"])

    def test_inference_firewall_rejects_before_checkpoint_hash_or_model_load(self) -> None:
        checkpoint_hash = Mock(side_effect=AssertionError("checkpoint hash must not run"))
        load_model = Mock(side_effect=AssertionError("model must not load"))
        with (
            patch(
                "tools.evaluate_ocid_maskrcnn_extractor.validate_development_input_pairs",
                side_effect=GateC1ContractError("held-out input"),
            ),
            patch("tools.evaluate_ocid_maskrcnn_extractor._sha256", checkpoint_hash),
            patch("tools.evaluate_ocid_maskrcnn_extractor._load_model", load_model),
        ):
            with self.assertRaisesRegex(GateC1ContractError, "held-out input"):
                run_benchmark(
                    stream_directories=(Path("heldout-stream"),),
                    annotation_paths=(Path("heldout-annotation.json"),),
                    checkpoint_path=Path("unused-checkpoint.pth"),
                    expected_checkpoint_sha256="0" * 64,
                    thresholds=(0.05,),
                    device="cpu",
                    output_path=Path("unused.json"),
                )
        checkpoint_hash.assert_not_called()
        load_model.assert_not_called()

    def test_nms_firewall_rejects_before_raw_report_or_annotation_load(self) -> None:
        raw_report_read = Mock(side_effect=AssertionError("raw report must not be read"))
        annotation_load = Mock(side_effect=AssertionError("annotation must not load"))
        with (
            patch(
                "tools.evaluate_ocid_maskrcnn_nms_postprocess.validate_development_input_pairs",
                side_effect=GateC1ContractError("held-out input"),
            ),
            patch.object(Path, "read_text", raw_report_read),
            patch("tools.evaluate_ocid_maskrcnn_nms_postprocess.load_annotation", annotation_load),
        ):
            with self.assertRaisesRegex(GateC1ContractError, "held-out input"):
                evaluate_saved_predictions(
                    raw_report_path=Path("heldout-raw.json"),
                    stream_directories=(Path("heldout-stream"),),
                    annotation_paths=(Path("heldout-annotation.json"),),
                    thresholds=(0.05,),
                    nms_iou_thresholds=(0.30,),
                    output_path=Path("unused.json"),
                )
        raw_report_read.assert_not_called()
        annotation_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
