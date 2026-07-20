from __future__ import annotations

import unittest

import numpy as np

from stream_analysis.evaluation.component_review import (
    BoundingBox,
    ComponentReviewDecision,
    ComponentReviewError,
    analyze_component_mask,
    binary_mask_sha256,
    component_union_mask,
    finalize_component_review,
)


class ComponentReviewTests(unittest.TestCase):
    def test_one_component_has_complete_metadata_and_needs_no_review(self) -> None:
        mask = np.zeros((5, 7), dtype=bool)
        mask[0:2, 2:5] = True

        analysis = analyze_component_mask(mask)

        self.assertEqual(analysis.source_mask_sha256, binary_mask_sha256(mask))
        self.assertEqual(analysis.raw_bbox, BoundingBox(2, 0, 3, 2))
        self.assertEqual(analysis.largest_bbox, analysis.raw_bbox)
        self.assertFalse(analysis.review_required)
        self.assertEqual(analysis.automatic_retained_component_ids, ("c001",))
        component = analysis.components[0]
        self.assertEqual(component.area, 6)
        self.assertEqual(component.centroid, (3.0, 0.5))
        self.assertTrue(component.border_touch)
        self.assertEqual(len(component.binary_mask_sha256), 64)

        finalized = finalize_component_review(analysis)
        np.testing.assert_array_equal(finalized.final_mask, mask)
        self.assertFalse(finalized.used_manual_decision)

    def test_components_are_ordered_by_area_then_bbox_for_ties(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[5:7, 1:3] = True  # area 4, later bbox
        mask[1:3, 6:8] = True  # area 4, earlier top
        mask[3, 4] = True  # area 1

        analysis = analyze_component_mask(mask)

        self.assertEqual(
            [(item.component_id, item.area, item.bbox) for item in analysis.components],
            [
                ("c001", 4, BoundingBox(6, 1, 2, 2)),
                ("c002", 4, BoundingBox(1, 5, 2, 2)),
                ("c003", 1, BoundingBox(4, 3, 1, 1)),
            ],
        )
        second = analyze_component_mask(mask.copy())
        self.assertEqual(
            [item.binary_mask_sha256 for item in analysis.components],
            [item.binary_mask_sha256 for item in second.components],
        )
        np.testing.assert_array_equal(analysis.label_map, second.label_map)

    def test_diagonal_pixels_are_separate_under_four_connectivity(self) -> None:
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True

        analysis = analyze_component_mask(mask)

        self.assertEqual(len(analysis.components), 2)
        self.assertEqual(analysis.components[0].bbox, BoundingBox(0, 0, 1, 1))
        self.assertEqual(analysis.components[1].bbox, BoundingBox(1, 1, 1, 1))

    def test_five_percent_boundary_is_included(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[1:11, 1:11] = True  # 100 pixels
        mask[15, 20:25] = True  # exactly 5 pixels
        mask[17, 20:24] = True  # 4 pixels

        analysis = analyze_component_mask(mask)

        self.assertEqual(
            analysis.automatic_retained_component_ids, ("c001", "c002")
        )

    def test_manual_override_can_retain_legitimate_4_88_percent_part(self) -> None:
        mask = np.zeros((60, 80), dtype=bool)
        mask[1:48, 1:48] = True  # 2209 pixels
        mask[50:59, 60:72] = True  # 108 pixels = 4.889...%
        analysis = analyze_component_mask(mask)
        self.assertEqual(analysis.automatic_retained_component_ids, ("c001",))
        self.assertTrue(analysis.review_required)

        decision = ComponentReviewDecision(
            source_mask_sha256=analysis.source_mask_sha256,
            retained_component_ids=("c001", "c002"),
            reason="Visible occluded part confirmed during annotation review.",
        )
        finalized = finalize_component_review(analysis, decision)

        np.testing.assert_array_equal(finalized.final_mask, mask)
        self.assertEqual(finalized.final_bbox, analysis.raw_bbox)
        self.assertTrue(finalized.used_manual_decision)

    def test_review_is_required_only_when_fragments_change_bbox(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[1, 1:6] = True
        mask[5, 1:6] = True
        mask[1:6, 1] = True
        mask[1:6, 5] = True
        mask[3, 3] = True  # disconnected but inside largest component bbox

        analysis = analyze_component_mask(mask)

        self.assertEqual(len(analysis.components), 2)
        self.assertEqual(analysis.raw_bbox, analysis.largest_bbox)
        self.assertFalse(analysis.review_required)

    def test_rejects_unknown_duplicate_missing_largest_and_stale_hash(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:3, 1:3] = True
        mask[6, 6] = True
        analysis = analyze_component_mask(mask)

        cases = (
            (("c001", "c999"), analysis.source_mask_sha256, "Unknown"),
            (("c001", "c001"), analysis.source_mask_sha256, "unique"),
            (("c002",), analysis.source_mask_sha256, "c001"),
            (("c001",), "0" * 64, "does not match"),
        )
        for retained, digest, message in cases:
            with self.subTest(retained=retained, digest=digest):
                decision = ComponentReviewDecision(
                    source_mask_sha256=digest,
                    retained_component_ids=retained,
                )
                with self.assertRaisesRegex(ComponentReviewError, message):
                    finalize_component_review(analysis, decision)

    def test_rejects_missing_decision_for_required_review(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:3, 1:3] = True
        mask[4, 4] = True
        analysis = analyze_component_mask(mask)

        with self.assertRaisesRegex(ComponentReviewError, "manual decision"):
            finalize_component_review(analysis)

    def test_union_mask_and_bbox_use_only_retained_components(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        mask[3, 7:9] = True
        mask[7, 0] = True
        analysis = analyze_component_mask(mask)
        decision = ComponentReviewDecision(
            source_mask_sha256=analysis.source_mask_sha256,
            retained_component_ids=("c002", "c001"),
        )

        finalized = finalize_component_review(analysis, decision)
        expected = np.zeros_like(mask)
        expected[2:5, 2:5] = True
        expected[3, 7:9] = True

        np.testing.assert_array_equal(finalized.final_mask, expected)
        np.testing.assert_array_equal(
            component_union_mask(analysis, ("c001", "c002")), expected
        )
        self.assertEqual(finalized.retained_component_ids, ("c001", "c002"))
        self.assertEqual(finalized.final_bbox, BoundingBox(2, 2, 7, 3))
        self.assertFalse(finalized.final_mask.flags.writeable)

    def test_rejects_invalid_mask_inputs(self) -> None:
        with self.assertRaisesRegex(ComponentReviewError, "dtype"):
            analyze_component_mask(np.ones((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ComponentReviewError, "foreground"):
            analyze_component_mask(np.zeros((2, 2), dtype=bool))
        with self.assertRaisesRegex(ComponentReviewError, "two-dimensional"):
            analyze_component_mask(np.ones((1, 2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
