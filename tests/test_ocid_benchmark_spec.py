from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ocid"
    / "benchmark"
    / "ocid_candidate_benchmark_v1.json"
)
COMPONENT_DECISIONS_PATH = (
    SPEC_PATH.parent / "ocid_candidate_component_decisions_v1.json"
)


class OcidBenchmarkSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_frozen_grouped_split_matches_declared_totals(self) -> None:
        roles = self.spec["roles"]
        scene_ids: list[str] = []
        streams: list[str] = []
        frame_totals: dict[str, int] = {}
        for role in ("development", "heldout"):
            groups = roles[role]
            frame_totals[role] = sum(
                group["frames_per_stream"] * len(group["member_streams"])
                for group in groups
            )
            for group in groups:
                scene_ids.append(group["scene_group_id"])
                streams.extend(group["member_streams"])

        expected = self.spec["expected_totals"]
        self.assertEqual(len(roles["development"]), expected["development_scene_groups"])
        self.assertEqual(len(roles["heldout"]), expected["heldout_scene_groups"])
        self.assertEqual(
            sum(len(group["member_streams"]) for group in roles["development"]),
            expected["development_streams"],
        )
        self.assertEqual(
            sum(len(group["member_streams"]) for group in roles["heldout"]),
            expected["heldout_streams"],
        )
        self.assertEqual(frame_totals["development"], expected["development_frames"])
        self.assertEqual(frame_totals["heldout"], expected["heldout_frames"])
        self.assertEqual(sum(frame_totals.values()), expected["benchmark_frames"])
        self.assertEqual(len(scene_ids), len(set(scene_ids)))
        self.assertEqual(len(streams), len(set(streams)))

    def test_each_scene_group_keeps_top_and_bottom_together(self) -> None:
        for groups in self.spec["roles"].values():
            for group in groups:
                streams = group["member_streams"]
                self.assertEqual(len(streams), 2)
                cameras = {stream.split("/")[2] for stream in streams}
                self.assertEqual(cameras, {"bottom", "top"})
                camera_removed = {
                    "/".join(parts[:2] + parts[3:])
                    for parts in (stream.split("/") for stream in streams)
                }
                self.assertEqual(camera_removed, {group["scene_group_key"]})

    def test_ground_truth_and_heldout_firewalls_are_explicit(self) -> None:
        boundary = self.spec["ground_truth_boundary"]
        self.assertEqual(boundary["forbidden_consumer"], "analyze")
        self.assertNotIn("analyze", boundary["allowed_consumers"])
        self.assertFalse(self.spec["heldout_lock"]["predictions_unlocked"])
        self.assertIn(
            "model_prediction", self.spec["heldout_lock"]["forbidden_before_unlock"]
        )

    def test_component_review_policy_and_frozen_decisions_are_explicit(self) -> None:
        annotation = self.spec["annotation_contract"]
        self.assertEqual(
            annotation["bbox_derivation"],
            "tight_axis_aligned_bbox_from_reviewed_instance_mask_components",
        )
        self.assertEqual(
            annotation["component_review_policy"],
            "largest_plus_reviewed_components_v1",
        )
        self.assertEqual(annotation["component_connectivity"], 4)
        self.assertEqual(annotation["secondary_area_ratio"], 0.05)
        self.assertEqual(
            annotation["review_candidate_rule"],
            "raw_bbox_differs_from_largest_component_bbox",
        )
        self.assertEqual(
            self.spec["review_contract"]["component_review_expected_observations"],
            261,
        )
        self.assertEqual(
            self.spec["source"]["component_decisions_status"],
            "frozen_complete",
        )
        self.assertTrue(COMPONENT_DECISIONS_PATH.is_file())
        self.assertEqual(
            self.spec["source"]["component_decisions_sha256"],
            hashlib.sha256(COMPONENT_DECISIONS_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            self.spec["annotation_contract"]["annotation_scope"],
            "ocid_candidate_extraction_bbox_from_instance_masks",
        )
        self.assertEqual(
            self.spec["annotation_contract"]["visual_type_claim"], "not_assessed"
        )


if __name__ == "__main__":
    unittest.main()
