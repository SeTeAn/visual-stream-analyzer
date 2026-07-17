from __future__ import annotations

import unittest

import numpy as np

from tools.evaluate_ocid_oracle_dinov2 import (
    OracleInstance,
    PreparedOracleInstance,
    anonymize_oracle_candidates,
    evaluate_oracle_embeddings,
)
from stream_analysis import BBox
from tests.matching._fixtures import candidate


class OracleDinoEvaluationTest(unittest.TestCase):
    def test_matcher_candidate_ids_are_geometry_ordered_and_hide_oracle_identity(self) -> None:
        first = PreparedOracleInstance(
            OracleInstance(
                "stream", "frame_000", 0, "instance_label_9", "identity_9", BBox(30, 10, 5, 5)
            ),
            None,
            candidate("instance_label_9", 0, 30),
            None,
        )
        second = PreparedOracleInstance(
            OracleInstance(
                "stream", "frame_000", 0, "instance_label_2", "identity_2", BBox(10, 10, 5, 5)
            ),
            None,
            candidate("instance_label_2", 0, 10),
            None,
        )

        candidates, identities = anonymize_oracle_candidates((first, second))

        self.assertEqual(candidates["instance_label_2"].candidate_id, "candidate:frame_000:000")
        self.assertEqual(candidates["instance_label_9"].candidate_id, "candidate:frame_000:001")
        self.assertNotIn("identity", " ".join(item.candidate_id for item in candidates.values()))
        self.assertEqual(identities["candidate:frame_000:000"], "identity_2")

    def test_perfect_adjacent_identity_embeddings_retrieve_at_top_one(self) -> None:
        instances = (
            OracleInstance("stream", "frame_001", 0, "a_1", "identity_a", BBox(0, 0, 2, 2)),
            OracleInstance("stream", "frame_001", 0, "b_1", "identity_b", BBox(3, 0, 2, 2)),
            OracleInstance("stream", "frame_002", 1, "a_2", "identity_a", BBox(0, 0, 2, 2)),
            OracleInstance("stream", "frame_002", 1, "b_2", "identity_b", BBox(3, 0, 2, 2)),
        )
        embeddings = {
            "a_1": np.asarray((1.0, 0.0), dtype=np.float32),
            "a_2": np.asarray((1.0, 0.0), dtype=np.float32),
            "b_1": np.asarray((0.0, 1.0), dtype=np.float32),
            "b_2": np.asarray((0.0, 1.0), dtype=np.float32),
        }

        result, records = evaluate_oracle_embeddings(
            instances,
            embeddings,
            namespace="stream",
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(result["ranking"]["retrieval"]["aggregate"]["recall@1"], 1.0)
        self.assertEqual(result["ranking"]["auroc"], 1.0)
        self.assertEqual(result["hard_negative_margin"]["positive_margin_rate"], 1.0)

    def test_adjacent_pairs_do_not_bridge_missing_frame_positions(self) -> None:
        instances = (
            OracleInstance("stream", "frame_001", 0, "a_1", "identity_a", BBox(0, 0, 2, 2)),
            OracleInstance("stream", "frame_003", 2, "a_3", "identity_a", BBox(0, 0, 2, 2)),
        )
        embeddings = {
            "a_1": np.asarray((1.0, 0.0), dtype=np.float32),
            "a_3": np.asarray((1.0, 0.0), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "no adjacent-frame pairs"):
            evaluate_oracle_embeddings(instances, embeddings, namespace="stream")


if __name__ == "__main__":
    unittest.main()
