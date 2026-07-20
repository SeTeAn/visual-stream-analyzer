from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_ocid_sequences.py"
SPEC = importlib.util.spec_from_file_location("audit_ocid_sequences", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load tool module from {TOOL_PATH}.")
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


class AuditOcidSequencesTest(unittest.TestCase):
    WIDTH = 8
    HEIGHT = 6

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ocid_root = self.root / "OCID-dataset"
        self.ocid_root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def sequence(self, relative: str) -> Path:
        path = self.ocid_root.joinpath(*relative.split("/"))
        (path / "rgb").mkdir(parents=True)
        (path / "label").mkdir()
        return path

    def mask(self, support_label: int, assignments: dict[tuple[int, int], int] | None = None) -> np.ndarray:
        labels = np.full((self.HEIGHT, self.WIDTH), support_label, dtype=np.uint16)
        for (x, y), label in (assignments or {}).items():
            labels[y, x] = label
        return labels

    def write_frame(
        self,
        sequence: Path,
        filename: str,
        labels: np.ndarray,
        *,
        rgb_size: tuple[int, int] | None = None,
    ) -> None:
        Image.new("RGB", rgb_size or (self.WIDTH, self.HEIGHT), (20, 30, 40)).save(
            sequence / "rgb" / filename
        )
        Image.fromarray(labels).save(sequence / "label" / filename)

    def run_audit(
        self,
        output_name: str = "audit",
        *,
        expected_sequences: int | None = None,
        expected_frames: int | None = None,
        expected_scene_groups: int | None = None,
        severe_area_ratio: float = 0.5,
        occluder_coverage: float = 0.5,
    ):
        return TOOL.audit_ocid_sequences(
            ocid_root=self.ocid_root,
            output_directory=self.root / output_name,
            expected_sequences=expected_sequences,
            expected_frames=expected_frames,
            expected_scene_groups=expected_scene_groups,
            severe_area_ratio=severe_area_ratio,
            occluder_coverage=occluder_coverage,
        )

    def payload(self, output_name: str = "audit") -> dict[str, object]:
        return json.loads(
            (self.root / output_name / "structural_audit.json").read_text(encoding="utf-8")
        )

    def test_discovers_deterministically_groups_views_and_hashes_artifacts(self) -> None:
        top = self.sequence("ARID10/table/top/box/seq01")
        bottom = self.sequence("ARID10/table/bottom/box/seq01")
        self.write_frame(top, "state.png", self.mask(2, {(3, 2): 3}))
        self.write_frame(bottom, "state.png", self.mask(2, {(4, 2): 3}))

        first = self.run_audit(
            "audit_a", expected_sequences=2, expected_frames=2, expected_scene_groups=1
        )
        second = self.run_audit(
            "audit_b", expected_sequences=2, expected_frames=2, expected_scene_groups=1
        )

        self.assertEqual(first.source_sha256, second.source_sha256)
        self.assertEqual(first.scene_group_count, 1)
        for filename in (
            "artifact_manifest.json",
            "structural_audit.json",
            "sequence_inventory.csv",
            "suspicious_transitions.csv",
        ):
            self.assertEqual(
                (self.root / "audit_a" / filename).read_bytes(),
                (self.root / "audit_b" / filename).read_bytes(),
            )

        payload = self.payload("audit_a")
        self.assertEqual(payload["scope"], "evaluation_only_dataset_audit")
        sequences = payload["sequences"]
        self.assertEqual(
            [item["source_sequence"] for item in sequences],
            ["ARID10/table/bottom/box/seq01", "ARID10/table/top/box/seq01"],
        )
        self.assertEqual(sequences[0]["paired_scene_key"], sequences[1]["paired_scene_key"])
        self.assertEqual(sequences[0]["scene_group_id"], sequences[1]["scene_group_id"])
        self.assertNotEqual(sequences[0]["scene_group_id"], sequences[0]["sequence_id"])
        self.assertEqual(sequences[0]["support_label"], 2)
        self.assertEqual(sequences[0]["object_label_minimum"], 3)
        bottom_id = sequences[0]["physical_instance_ids"][0]
        top_id = sequences[1]["physical_instance_ids"][0]
        self.assertNotEqual(bottom_id, top_id)
        self.assertTrue(bottom_id.startswith(sequences[0]["sequence_id"]))
        self.assertEqual(payload["signal_semantics"]["physical_instance_id_scope"], "sequence_local")
        self.assertEqual(payload["summary"]["family_counts"]["ARID10"], {"sequences": 2, "frames": 2})
        self.assertEqual(payload["summary"]["camera_counts"]["bottom"]["sequences"], 1)
        self.assertEqual(payload["summary"]["scene_group_count"], 1)
        self.assertEqual(payload["summary"]["complete_top_bottom_scene_group_count"], 1)
        self.assertEqual(payload["summary"]["incomplete_scene_group_count"], 0)
        self.assertEqual(payload["summary"]["stream_local_physical_instance_count"], 2)
        self.assertEqual(payload["summary"]["physical_observation_count"], 2)
        scene_group = payload["scene_groups"][0]
        self.assertEqual(scene_group["scene_group_id"], sequences[0]["scene_group_id"])
        self.assertEqual(scene_group["member_sequence_ids"], [item["sequence_id"] for item in sequences])
        self.assertEqual(scene_group["cameras"], ["bottom", "top"])
        self.assertTrue(scene_group["complete_top_bottom_pair"])
        self.assertTrue(scene_group["matching_frame_count"])
        self.assertNotIn("visual_type_id", json.dumps(payload))

        manifest = json.loads(
            (self.root / "audit_a" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source_fingerprint"]["entry_count"], 4)
        for artifact in manifest["artifacts"]:
            data = (self.root / "audit_a" / artifact["path"]).read_bytes()
            self.assertEqual(artifact["size_bytes"], len(data))
            self.assertEqual(artifact["sha256"], hashlib.sha256(data).hexdigest())

    def test_offset_support_label_uses_objects_above_256(self) -> None:
        sequence = self.sequence("ARID10/floor/top/curved/seq05")
        labels = self.mask(256, {(2, 2): 257, (3, 2): 257, (0, 0): 0})
        self.write_frame(sequence, "state.png", labels)

        self.run_audit(expected_sequences=1, expected_frames=1)

        payload = self.payload()
        audited = payload["sequences"][0]
        self.assertEqual(audited["support_label"], 256)
        self.assertEqual(audited["object_label_minimum"], 257)
        self.assertEqual(audited["expected_normal_support_label"], 1)
        self.assertFalse(audited["support_label_matches_expected"])
        self.assertTrue(audited["unexpected_label_offset"])
        self.assertEqual(audited["frames"][0]["object_count"], 1)
        self.assertEqual(audited["frames"][0]["physical_instances"][0]["source_label"], 257)
        self.assertEqual(payload["summary"]["unexpected_support_offset_sequence_count"], 1)
        self.assertEqual(payload["summary"]["scene_group_count"], 1)
        self.assertEqual(payload["summary"]["complete_top_bottom_scene_group_count"], 0)
        self.assertEqual(payload["summary"]["incomplete_scene_group_count"], 1)
        self.assertFalse(payload["scene_groups"][0]["complete_top_bottom_pair"])

        with (self.root / "audit" / "sequence_inventory.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["support_label"], "256")
        self.assertEqual(row["object_label_minimum"], "257")
        self.assertEqual(row["unexpected_label_offset"], "true")
        self.assertEqual(row["scene_group_id"], audited["scene_group_id"])

    def test_records_added_removed_returned_and_occlusion_proxies(self) -> None:
        sequence = self.sequence("ARID10/floor/top/mixed/seq01")
        object_two = {(2, 1), (3, 1), (2, 2), (3, 2)}
        object_three = {(5, 1), (6, 1), (5, 2), (6, 2), (5, 3), (6, 3)}
        frame_one = self.mask(
            1,
            {**{point: 2 for point in object_two}, **{point: 3 for point in object_three}},
        )
        frame_two = self.mask(
            1,
            {
                **{point: 3 for point in object_two | object_three},
                (1, 4): 4,
            },
        )
        frame_three = self.mask(
            1,
            {
                **{point: 2 for point in object_two},
                (5, 1): 3,
                (6, 1): 3,
                (1, 4): 4,
            },
        )
        self.write_frame(sequence, "frame_01.png", frame_one)
        self.write_frame(sequence, "frame_02.png", frame_two)
        self.write_frame(sequence, "frame_03.png", frame_three)

        self.run_audit(
            expected_sequences=1,
            expected_frames=3,
            severe_area_ratio=0.5,
            occluder_coverage=0.3,
        )

        payload = self.payload()
        audited = payload["sequences"][0]
        id_two = f"{audited['sequence_id']}__label_002"
        id_three = f"{audited['sequence_id']}__label_003"
        id_four = f"{audited['sequence_id']}__label_004"
        first_transition, second_transition = audited["transitions"]
        self.assertEqual(first_transition["added_ids"], [id_four])
        self.assertEqual(first_transition["removed_ids"], [id_two])
        self.assertEqual(first_transition["retained_ids"], [id_three])
        removed = first_transition["removed_observation_candidates"][0]
        self.assertEqual(removed["removed_object_occluder_coverage"], 1.0)
        self.assertIn("occlusion_candidate", removed["candidate_labels"])

        self.assertEqual(second_transition["returned_ids"], [id_two])
        drop = second_transition["visibility_drops"][0]
        self.assertEqual(drop["physical_instance_id"], id_three)
        self.assertEqual(drop["visible_area_ratio"], 0.2)
        self.assertEqual(drop["lost_area_reassigned_to_objects"], 0.5)
        self.assertEqual(
            drop["candidate_labels"],
            ["severe_visibility_drop_candidate", "occlusion_candidate"],
        )
        with (self.root / "audit" / "suspicious_transitions.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            signal_types = {row["signal_type"] for row in csv.DictReader(stream)}
        self.assertIn("annotation_absence_candidate", signal_types)
        self.assertIn("annotation_return_candidate", signal_types)
        self.assertIn("severe_visibility_drop_candidate", signal_types)
        self.assertIn("occlusion_candidate", signal_types)
        self.assertEqual(payload["summary"]["strict_add_one_sequence_count"], 0)
        self.assertEqual(
            payload["summary"]["observable_removal_sequence_ids"],
            [audited["sequence_id"]],
        )
        self.assertEqual(
            payload["summary"]["observable_return_sequence_ids"],
            [audited["sequence_id"]],
        )
        self.assertEqual(payload["summary"]["stream_local_physical_instance_count"], 3)
        self.assertEqual(payload["summary"]["physical_observation_count"], 7)

    def test_counts_strict_add_one_sequences(self) -> None:
        sequence = self.sequence("ARID10/table/top/box/seq07")
        self.write_frame(sequence, "frame_01.png", self.mask(2))
        self.write_frame(sequence, "frame_02.png", self.mask(2, {(2, 2): 3}))
        self.write_frame(
            sequence,
            "frame_03.png",
            self.mask(2, {(2, 2): 3, (5, 3): 4}),
        )

        self.run_audit(expected_scene_groups=1)

        payload = self.payload()
        audited = payload["sequences"][0]
        self.assertTrue(audited["structural_signals"]["strict_add_one"])
        self.assertEqual(payload["summary"]["strict_add_one_sequence_count"], 1)
        self.assertEqual(payload["summary"]["observable_removal_sequence_ids"], [])
        self.assertEqual(payload["summary"]["observable_return_sequence_ids"], [])
        self.assertEqual(payload["summary"]["stream_local_physical_instance_count"], 2)
        self.assertEqual(payload["summary"]["physical_observation_count"], 3)

    def test_records_connected_components_and_border_touch(self) -> None:
        sequence = self.sequence("YCB10/floor/bottom/box/seq02")
        labels = self.mask(1, {(0, 0): 2, (7, 5): 2})
        self.write_frame(sequence, "state.png", labels)

        self.run_audit()

        audited = self.payload()["sequences"][0]
        instance = audited["frames"][0]["physical_instances"][0]
        self.assertEqual(instance["connected_component_count"], 2)
        self.assertTrue(instance["border_touch"])
        self.assertEqual(audited["structural_signals"]["multi_component_observation_count"], 1)
        self.assertEqual(audited["structural_signals"]["border_touch_observation_count"], 1)

    def test_rejects_filename_mismatch(self) -> None:
        sequence = self.sequence("ARID10/table/top/box/seq01")
        self.write_frame(sequence, "rgb_name.png", self.mask(2))
        (sequence / "label" / "rgb_name.png").rename(sequence / "label" / "label_name.png")

        with self.assertRaisesRegex(ValueError, "filenames do not align"):
            self.run_audit()
        self.assertFalse((self.root / "audit").exists())

    def test_rejects_dimension_and_integer_label_errors(self) -> None:
        dimension_sequence = self.sequence("ARID10/floor/top/box/seq01")
        wrong_size = np.full((self.HEIGHT, self.WIDTH - 1), 1, dtype=np.uint16)
        self.write_frame(dimension_sequence, "state.png", wrong_size)
        with self.assertRaisesRegex(ValueError, "dimensions do not align"):
            self.run_audit("dimension_error")
        self.write_frame(dimension_sequence, "state.png", self.mask(1))

        shutil_sequence = self.sequence("YCB10/floor/top/box/seq02")
        float_labels = np.full((self.HEIGHT, self.WIDTH), 1.0, dtype=np.float32)
        Image.new("RGB", (self.WIDTH, self.HEIGHT), (0, 0, 0)).save(
            shutil_sequence / "rgb" / "state.png"
        )
        Image.fromarray(float_labels, mode="F").save(
            shutil_sequence / "label" / "state.png", format="TIFF"
        )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            self.run_audit("integer_error")

    def test_rejects_support_label_that_stops_being_dominant(self) -> None:
        sequence = self.sequence("ARID10/floor/top/box/seq01")
        self.write_frame(sequence, "frame_01.png", self.mask(1, {(2, 2): 2}))
        second = np.full((self.HEIGHT, self.WIDTH), 2, dtype=np.uint16)
        second[0, 0] = 1
        self.write_frame(sequence, "frame_02.png", second)

        with self.assertRaisesRegex(ValueError, "not dominant over every"):
            self.run_audit()
        self.assertFalse((self.root / "audit").exists())

    def test_source_fingerprint_changes_with_source_content(self) -> None:
        sequence = self.sequence("ARID10/floor/top/box/seq01")
        self.write_frame(sequence, "state.png", self.mask(1, {(2, 2): 2}))
        first = self.run_audit("first")

        self.write_frame(sequence, "state.png", self.mask(1, {(3, 2): 2}))
        second = self.run_audit("second")

        self.assertNotEqual(first.source_sha256, second.source_sha256)

    def test_refuses_overwrite_and_cleans_staging_after_write_failure(self) -> None:
        sequence = self.sequence("ARID10/floor/top/box/seq01")
        self.write_frame(sequence, "state.png", self.mask(1, {(2, 2): 2}))
        destination = self.root / "audit"

        with mock.patch.object(TOOL, "_write_csv", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.run_audit()
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".audit-*")), [])

        self.run_audit()
        original_manifest = (destination / "artifact_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual((destination / "artifact_manifest.json").read_bytes(), original_manifest)

    def test_expected_counts_fail_before_publishing_output(self) -> None:
        sequence = self.sequence("ARID10/floor/top/box/seq01")
        self.write_frame(sequence, "state.png", self.mask(1))

        with self.assertRaisesRegex(ValueError, "Expected 2 OCID sequences"):
            self.run_audit(expected_sequences=2)
        self.assertFalse((self.root / "audit").exists())

        with self.assertRaisesRegex(ValueError, "Expected 2 OCID frames"):
            self.run_audit(expected_frames=2)
        self.assertFalse((self.root / "audit").exists())

        with self.assertRaisesRegex(ValueError, "Expected 2 OCID scene groups"):
            self.run_audit(expected_scene_groups=2)
        self.assertFalse((self.root / "audit").exists())


if __name__ == "__main__":
    unittest.main()
