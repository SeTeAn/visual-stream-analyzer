from __future__ import annotations

import unittest

from stream_analysis.contracts import BBox
from tools.evaluate_ocid_mobilesam_extractor import (
    FilterProfile,
    RawPrediction,
    predictions_for_profile,
)


class MobileSAMExtractorGateTest(unittest.TestCase):
    def test_profile_filters_quality_and_area_without_changing_prediction_ids(self) -> None:
        profile = FilterProfile("gate", 0.8, 0.9, 0.01, 0.5)
        raw = (
            RawPrediction("frame_001", 3, 2, BBox(1, 2, 10, 10), 100, 10_000, 0.8, 0.9),
            RawPrediction("frame_001", 3, 4, BBox(2, 3, 5, 5), 25, 10_000, 0.9, 0.95),
            RawPrediction("frame_001", 3, 7, BBox(0, 0, 80, 80), 6_400, 10_000, 0.9, 0.95),
        )

        candidates = predictions_for_profile(raw, profile=profile)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, "mobilesam:frame_001:002")
        self.assertEqual(candidates[0].frame_index, 3)
        self.assertEqual(candidates[0].bbox, BBox(1, 2, 10, 10))

    def test_profile_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            FilterProfile("bad", 1.1, 0.9, 0.01, 0.5)
        with self.assertRaises(ValueError):
            FilterProfile("bad", 0.8, 0.9, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
