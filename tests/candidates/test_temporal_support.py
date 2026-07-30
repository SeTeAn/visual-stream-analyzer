import itertools
import unittest

from stream_analysis.candidates.temporal_support import (
    TEMPORAL_ADJACENT_RADIUS,
    TEMPORAL_BASE_SCORE,
    TEMPORAL_CONTAINMENT_REJECT,
    TEMPORAL_LOW_SCORE,
    TEMPORAL_SUPPORT_IOU,
    Proposal,
    class_agnostic_nms,
    geometry_rejects,
    temporal_support_selection,
)
from stream_analysis.contracts import BBox


FRAME_IDS = ("frame_000", "frame_001", "frame_002")
FRAME_SIZES = {
    frame_id: {"width": 100, "height": 100} for frame_id in FRAME_IDS
}


def _proposal(
    frame_id: str,
    prediction_index: int,
    score: float,
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 20.0, 20.0),
) -> Proposal:
    return Proposal(
        frame_id=frame_id,
        frame_index=FRAME_IDS.index(frame_id),
        prediction_index=prediction_index,
        score=score,
        phrase="object",
        bbox=BBox(*bbox),
    )


class TemporalSupportSelectionTest(unittest.TestCase):
    def test_constants_match_configured_temporal_policy(self) -> None:
        self.assertEqual(TEMPORAL_BASE_SCORE, 0.15)
        self.assertEqual(TEMPORAL_LOW_SCORE, 0.10)
        self.assertEqual(TEMPORAL_SUPPORT_IOU, 0.50)
        self.assertEqual(TEMPORAL_CONTAINMENT_REJECT, 0.70)
        self.assertEqual(TEMPORAL_ADJACENT_RADIUS, 1)

    def test_accepts_low_score_proposal_with_adjacent_support(self) -> None:
        base = _proposal("frame_000", 0, TEMPORAL_BASE_SCORE)
        low = _proposal("frame_001", 1, TEMPORAL_LOW_SCORE)

        result = temporal_support_selection(
            (low, base),
            frame_ids=FRAME_IDS,
            frame_sizes=FRAME_SIZES,
            nms_iou=0.30,
            min_area_ratio=None,
            min_span_ratio=None,
        )

        self.assertEqual(result.final_proposals, (base, low))
        self.assertEqual(result.supplemental_proposals, (low,))
        self.assertEqual(len(result.audit_rows), 1)
        audit = result.audit_rows[0]
        self.assertEqual(audit["neighbor_frame_ids"], ["frame_000", "frame_002"])
        self.assertEqual(audit["best_support_iou"], 1.0)
        self.assertEqual(
            audit["best_support"],
            {"frame_id": "frame_000", "prediction_index": 0, "score": 0.15},
        )
        self.assertTrue(audit["temporal_supported"])
        self.assertFalse(audit["containment_rejected"])
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["decision"], "accepted_supplemental")

    def test_rejects_low_score_proposal_without_adjacent_support(self) -> None:
        base = _proposal("frame_000", 0, 0.90)
        unsupported = _proposal(
            "frame_001",
            1,
            0.12,
            bbox=(60.0, 60.0, 10.0, 10.0),
        )

        result = temporal_support_selection(
            (base, unsupported),
            frame_ids=FRAME_IDS,
            frame_sizes=FRAME_SIZES,
            nms_iou=0.30,
            min_area_ratio=None,
            min_span_ratio=None,
        )

        self.assertEqual(result.final_proposals, (base,))
        self.assertEqual(result.supplemental_proposals, ())
        self.assertEqual(result.audit_rows[0]["best_support_iou"], 0.0)
        self.assertFalse(result.audit_rows[0]["accepted"])
        self.assertEqual(
            result.audit_rows[0]["decision"],
            "rejected_no_temporal_support",
        )

    def test_rejects_supported_low_score_proposal_contained_by_base(self) -> None:
        neighbor = _proposal(
            "frame_000",
            0,
            0.20,
            bbox=(12.0, 12.0, 5.0, 5.0),
        )
        primary = _proposal("frame_001", 1, 0.90)
        contained = _proposal(
            "frame_001",
            2,
            0.12,
            bbox=(12.0, 12.0, 5.0, 5.0),
        )

        result = temporal_support_selection(
            (contained, primary, neighbor),
            frame_ids=FRAME_IDS,
            frame_sizes=FRAME_SIZES,
            nms_iou=0.30,
            min_area_ratio=None,
            min_span_ratio=None,
        )

        self.assertEqual(result.final_proposals, (neighbor, primary))
        self.assertEqual(result.supplemental_proposals, ())
        audit = result.audit_rows[0]
        self.assertEqual(audit["best_support_iou"], 1.0)
        self.assertEqual(audit["maximum_base_containment"], 1.0)
        self.assertTrue(audit["temporal_supported"])
        self.assertTrue(audit["containment_rejected"])
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["decision"], "rejected_containment")

    def test_class_agnostic_nms_is_deterministic(self) -> None:
        tied_later = _proposal("frame_000", 2, 0.80)
        tied_first = _proposal("frame_000", 1, 0.80)
        separate = _proposal(
            "frame_000", 3, 0.70, bbox=(60.0, 60.0, 10.0, 10.0)
        )
        other_frame = _proposal("frame_001", 4, 0.60)
        proposals = (tied_later, tied_first, separate, other_frame)
        expected = (tied_first, separate, other_frame)

        for permutation in itertools.permutations(proposals):
            with self.subTest(
                order=tuple(item.prediction_index for item in permutation)
            ):
                self.assertEqual(class_agnostic_nms(permutation, 0.30), expected)

    def test_geometry_filter_applies_area_and_span_thresholds(self) -> None:
        large = _proposal(
            "frame_000",
            0,
            0.90,
            bbox=(0.0, 0.0, 80.0, 80.0),
        )

        self.assertFalse(
            geometry_rejects(
                large,
                FRAME_SIZES["frame_000"],
                min_area_ratio=None,
                min_span_ratio=None,
            )
        )
        self.assertTrue(
            geometry_rejects(
                large,
                FRAME_SIZES["frame_000"],
                min_area_ratio=0.50,
                min_span_ratio=0.75,
            )
        )
        self.assertFalse(
            geometry_rejects(
                large,
                FRAME_SIZES["frame_000"],
                min_area_ratio=0.50,
                min_span_ratio=0.90,
            )
        )

    def test_validates_nms_threshold_and_frame_metadata(self) -> None:
        proposal = _proposal("frame_000", 0, 0.90)
        for threshold in (-0.01, 1.01, float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError,
                    "iou_threshold must be in \\[0, 1\\]",
                ):
                    class_agnostic_nms((proposal,), threshold)

        for frame_ids in ((), ("frame_000", "frame_000")):
            with self.subTest(frame_ids=frame_ids):
                with self.assertRaisesRegex(ValueError, "non-empty unique"):
                    temporal_support_selection(
                        (proposal,),
                        frame_ids=frame_ids,
                        frame_sizes={},
                        nms_iou=0.30,
                        min_area_ratio=None,
                        min_span_ratio=None,
                    )

        with self.assertRaisesRegex(ValueError, "exactly match"):
            temporal_support_selection(
                (proposal,),
                frame_ids=FRAME_IDS,
                frame_sizes={"frame_000": FRAME_SIZES["frame_000"]},
                nms_iou=0.30,
                min_area_ratio=None,
                min_span_ratio=None,
            )

        invalid_size = dict(FRAME_SIZES)
        invalid_size["frame_000"] = {"width": 0, "height": 100}
        with self.assertRaisesRegex(ValueError, "width must be a positive number"):
            temporal_support_selection(
                (proposal,),
                frame_ids=FRAME_IDS,
                frame_sizes=invalid_size,
                nms_iou=0.30,
                min_area_ratio=0.50,
                min_span_ratio=0.75,
            )


if __name__ == "__main__":
    unittest.main()
