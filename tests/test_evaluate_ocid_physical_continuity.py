from __future__ import annotations

import unittest

from stream_analysis.contracts import BBox
from stream_analysis.evaluation import PredictedCandidate
from tools.evaluate_ocid_physical_continuity import _bound_run_id


def _candidate_manifest(stream_id: str) -> dict:
    return {
        "result": {
            "envelope": {"stream_id": stream_id},
            "candidates": [],
        }
    }


def _primary_result(stream_id: str) -> dict:
    return {
        "run_id": "run_1",
        "envelope": {"stream_id": stream_id, "record_id": "run_1"},
        "candidate_record_ids": ["candidate_1"],
    }


def _predicted() -> tuple[PredictedCandidate, ...]:
    return (
        PredictedCandidate(
            "candidate_1",
            "frame_1",
            BBox(0, 0, 10, 10),
            "valid",
            (),
            (),
            1,
        ),
    )


class PhysicalContinuityBindingTest(unittest.TestCase):
    def test_rejects_cross_stream_candidate_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream_id values must match"):
            _bound_run_id(
                expected_stream_id="stream_a",
                candidate_manifest=_candidate_manifest("stream_b"),
                primary_result=_primary_result("stream_a"),
                predicted=_predicted(),
            )

    def test_rejects_candidate_set_not_bound_to_primary_result(self) -> None:
        primary = _primary_result("stream_a")
        primary["candidate_record_ids"] = ["candidate_other"]
        with self.assertRaisesRegex(ValueError, "candidate_record_ids"):
            _bound_run_id(
                expected_stream_id="stream_a",
                candidate_manifest=_candidate_manifest("stream_a"),
                primary_result=primary,
                predicted=_predicted(),
            )

    def test_returns_parent_run_id_for_bound_inputs(self) -> None:
        self.assertEqual(
            _bound_run_id(
                expected_stream_id="stream_a",
                candidate_manifest=_candidate_manifest("stream_a"),
                primary_result=_primary_result("stream_a"),
                predicted=_predicted(),
            ),
            "run_1",
        )


if __name__ == "__main__":
    unittest.main()
