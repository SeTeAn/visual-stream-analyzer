from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream_analysis.contracts import BBox
from stream_analysis.evaluation.annotation import AnnotationInstance
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
