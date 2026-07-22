from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation import (
    evaluate_candidate_predictions,
    evaluate_physical_instance_continuity,
)
from stream_analysis.evaluation.annotation import AnnotationInstance, StreamAnnotation
from stream_analysis.evaluation.runner import (
    EvaluationError,
    PredictedCandidate,
    _candidate_overlap_diagnostics,
    _load_json,
)


def _pred(candidate_id: str, bbox: BBox) -> PredictedCandidate:
    return PredictedCandidate(candidate_id, "f", bbox, "valid", (), ())


def _gt(instance_id: str, bbox: BBox) -> AnnotationInstance:
    return AnnotationInstance(instance_id, "f", "t", bbox)


class CandidateDiagnosticsTest(unittest.TestCase):
    def test_physical_continuity_evaluates_matches_without_enabling_grouping_claims(self) -> None:
        annotation = StreamAnnotation(
            path=Path("annotation.json"),
            digest_sha256="0" * 64,
            schema_version="test",
            stream_id="stream",
            manifest_ref="manifest.json",
            annotation_scope="ocid_candidate_extraction_bbox_from_instance_masks",
            visual_type_ids=("physical_1",),
            frame_ids=("f1", "f2"),
            frame_size=ImageSize(100, 100),
            instances=(
                AnnotationInstance("g1", "f1", "physical_1", BBox(10, 10, 20, 20)),
                AnnotationInstance("g2", "f2", "physical_1", BBox(10, 10, 20, 20)),
            ),
            frame_comparisons=(),
            change_events=(),
            supported_event_types=(),
            raw={},
        )
        predicted = (
            PredictedCandidate("p:left", "f1", BBox(10, 10, 20, 20), "valid", (), (), 0),
            PredictedCandidate("p:right", "f2", BBox(10, 10, 20, 20), "valid", (), (), 1),
        )
        primary = {
            "frame_comparisons": [{
                "comparison_id": "comparison:f1:f2",
                "frame_pair": {"from_frame_id": "f1", "to_frame_id": "f2"},
                "emission_status": "produced",
                "accepted_match_ids": ["match:comparison:f1:f2:p:left:p:right"],
                "uncertain_match_ids": [],
            }],
        }

        result = evaluate_physical_instance_continuity(annotation, predicted, primary)

        self.assertEqual(result["accepted_strict"]["f1"], 1.0)
        self.assertEqual(
            result["identity_semantics"],
            "physical_instance_continuity_not_visual_type_ground_truth",
        )

    def test_public_candidate_evaluation_returns_json_ready_metrics(self) -> None:
        annotation = StreamAnnotation(
            path=Path("annotation.json"),
            digest_sha256="0" * 64,
            schema_version="test",
            stream_id="stream",
            manifest_ref="manifest.json",
            annotation_scope="candidate_extraction_bbox_only",
            visual_type_ids=("t",),
            frame_ids=("f",),
            frame_size=ImageSize(100, 100),
            instances=(_gt("g", BBox(10, 10, 20, 20)),),
            frame_comparisons=(),
            change_events=(),
            supported_event_types=(),
            raw={},
        )

        result = evaluate_candidate_predictions(
            annotation,
            (_pred("p", BBox(10, 10, 20, 20)),),
        )

        self.assertEqual(result["f1"], 1.0)
        self.assertNotIn("assignments", result)

    def test_duplicate_split_merge_fragment_noise_diagnostics_are_distinct(self) -> None:
        gt = (
            _gt("g_duplicate", BBox(0, 0, 10, 10)),
            _gt("g_split", BBox(30, 0, 20, 10)),
            _gt("g_merge_a", BBox(0, 30, 10, 10)),
            _gt("g_merge_b", BBox(12, 30, 10, 10)),
            _gt("g_fragment", BBox(60, 0, 20, 20)),
        )
        pred = (
            _pred("p_dup_1", BBox(0, 0, 10, 10)),
            _pred("p_dup_2", BBox(0, 0, 10, 10)),
            _pred("p_split_1", BBox(30, 0, 8, 10)),
            _pred("p_split_2", BBox(42, 0, 8, 10)),
            _pred("p_merge", BBox(0, 30, 22, 10)),
            _pred("p_fragment", BBox(60, 0, 8, 8)),
            _pred("p_noise", BBox(100, 100, 5, 5)),
        )
        diagnostics = _candidate_overlap_diagnostics(pred, gt, matched_pred=set(), matched_gt=set())
        self.assertEqual(diagnostics["duplicate"], 1)
        self.assertEqual(diagnostics["split"], 1)
        self.assertEqual(diagnostics["merge"], 1)
        self.assertEqual(diagnostics["fragment"], 1)
        self.assertEqual(diagnostics["noise"], 1)
        self.assertEqual(diagnostics["miss"], len(gt))

    def test_missing_required_artifact_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(EvaluationError):
                _load_json(Path(temporary) / "missing.json")


if __name__ == "__main__":
    unittest.main()
