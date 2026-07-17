from __future__ import annotations

import unittest

from stream_analysis.contracts import BBox
from tools.evaluate_ocid_maskrcnn_extractor import (
    RawPrediction,
    aggregate_candidate_metrics,
    predictions_at_threshold,
)


class MaskRCNNExtractorGateTest(unittest.TestCase):
    def test_threshold_and_geometry_variant_preserve_stable_prediction_ids(self) -> None:
        raw = (
            RawPrediction("frame_001", 2, 0.8, 1, BBox(1, 2, 10, 12), BBox(2, 3, 8, 9)),
            RawPrediction("frame_001", 5, 0.2, 1, BBox(20, 20, 5, 5), None),
        )

        bbox = predictions_at_threshold(raw, threshold=0.5, geometry_variant="model_bbox")
        mask = predictions_at_threshold(raw, threshold=0.5, geometry_variant="mask_tight_bbox")

        self.assertEqual(tuple(item.candidate_id for item in bbox), ("maskrcnn:frame_001:002",))
        self.assertEqual(tuple(item.candidate_id for item in mask), ("maskrcnn:frame_001:002",))
        self.assertEqual(bbox[0].bbox, BBox(1, 2, 10, 12))
        self.assertEqual(mask[0].bbox, BBox(2, 3, 8, 9))

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
            RawPrediction("frame_001", 0, 0.9, 1, BBox(0, 0, 10, 10), None),
            RawPrediction("frame_001", 1, 0.8, 47, BBox(1, 1, 10, 10), None),
            RawPrediction("frame_001", 2, 0.7, 2, BBox(30, 30, 5, 5), None),
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


if __name__ == "__main__":
    unittest.main()
