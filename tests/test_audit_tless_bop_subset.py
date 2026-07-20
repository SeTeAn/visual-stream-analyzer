from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_tless_bop_subset.py"
SPEC = importlib.util.spec_from_file_location("audit_tless_bop_subset", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load tool module from {TOOL_PATH}.")
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
CAMERA_MATRIX = [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0]


class AuditTlessBopSubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def json_bytes(payload: object) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def scene_members(
        self,
        scene_id: str,
        views: list[dict[str, object]],
    ) -> list[tuple[str, bytes]]:
        scene_gt: dict[str, object] = {}
        scene_gt_info: dict[str, object] = {}
        scene_camera: dict[str, object] = {}
        members: list[tuple[str, bytes]] = []
        base = f"test_primesense/{scene_id}"

        for view_index, view in enumerate(views):
            image_id = int(view["image_id"])
            camera_translation = [
                10.0 * view_index,
                -4.0 * view_index,
                3.0 * view_index,
            ]
            objects = list(view["objects"])
            visibility = list(view.get("visibility", [0.9] * len(objects)))
            gt_records: list[dict[str, object]] = []
            info_records: list[dict[str, object]] = []
            for gt_id, ((obj_id, world_position), visible_fraction) in enumerate(
                zip(objects, visibility, strict=True)
            ):
                cam_t_m2c = [
                    float(world_position[axis]) + camera_translation[axis]
                    for axis in range(3)
                ]
                gt_records.append(
                    {
                        "obj_id": int(obj_id),
                        "cam_R_m2c": IDENTITY_ROTATION,
                        "cam_t_m2c": cam_t_m2c,
                    }
                )
                info_records.append(
                    {
                        "bbox_obj": [10, 20, 30, 40],
                        "bbox_visib": [11, 21, 28, 38],
                        "px_count_all": 100,
                        "px_count_valid": 100,
                        "px_count_visib": int(round(100 * float(visible_fraction))),
                        "visib_fract": float(visible_fraction),
                    }
                )
                members.append(
                    (f"{base}/mask/{image_id:06d}_{gt_id:06d}.png", b"mask")
                )
                members.append(
                    (
                        f"{base}/mask_visib/{image_id:06d}_{gt_id:06d}.png",
                        b"visible-mask",
                    )
                )

            scene_gt[str(image_id)] = gt_records
            scene_gt_info[str(image_id)] = info_records
            scene_camera[str(image_id)] = {
                "cam_K": CAMERA_MATRIX,
                "cam_R_w2c": IDENTITY_ROTATION,
                "cam_t_w2c": camera_translation,
                "depth_scale": 0.1,
            }
            members.append((f"{base}/rgb/{image_id:06d}.png", b"rgb"))

        members.extend(
            [
                (f"{base}/scene_gt.json", self.json_bytes(scene_gt)),
                (f"{base}/scene_gt_info.json", self.json_bytes(scene_gt_info)),
                (f"{base}/scene_camera.json", self.json_bytes(scene_camera)),
            ]
        )
        return members

    def write_archive(
        self,
        name: str,
        members: list[tuple[str, bytes]],
    ) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member_name, payload in members:
                info = zipfile.ZipInfo(member_name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 << 16)
                archive.writestr(info, payload)
        return path

    @staticmethod
    def stable_duplicate_views() -> list[dict[str, object]]:
        objects = [
            (1, [100.0, 0.0, 700.0]),
            (1, [180.0, 0.0, 700.0]),
            (2, [260.0, 0.0, 700.0]),
        ]
        return [
            {"image_id": 0, "objects": objects, "visibility": [0.9, 0.7, 0.8]},
            {"image_id": 5, "objects": objects, "visibility": [0.8, 0.6, 0.75]},
        ]

    def run_audit(
        self,
        archive: Path,
        output_name: str = "audit",
        *,
        expected_scenes: int | None = None,
        expected_images: int | None = None,
        expected_archive_sha256: str | None = None,
        tolerance_mm: float = 0.01,
    ):
        return TOOL.audit_tless_bop_subset(
            archive=archive,
            output_directory=self.root / output_name,
            expected_scenes=expected_scenes,
            expected_images=expected_images,
            expected_archive_sha256=expected_archive_sha256,
            world_position_tolerance_mm=tolerance_mm,
        )

    def payload(self, output_name: str = "audit") -> dict[str, object]:
        return json.loads(
            (self.root / output_name / "tless_structural_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_stable_duplicate_visual_type_is_validated_and_hashed(self) -> None:
        archive = self.write_archive(
            "stable.zip", self.scene_members("000009", self.stable_duplicate_views())
        )
        archive_sha256 = self.sha256(archive)

        first = self.run_audit(
            archive,
            "audit_a",
            expected_scenes=1,
            expected_images=2,
            expected_archive_sha256=archive_sha256.upper(),
        )
        second = self.run_audit(
            archive,
            "audit_b",
            expected_scenes=1,
            expected_images=2,
            expected_archive_sha256=archive_sha256.upper(),
        )

        self.assertEqual(first.archive_sha256, archive_sha256)
        self.assertEqual(first.strict_visual_type_scene_count, 1)
        self.assertEqual(second.strict_visual_type_scene_count, 1)
        for filename in (
            "artifact_manifest.json",
            "tless_structural_audit.json",
            "scene_inventory.csv",
        ):
            self.assertEqual(
                (self.root / "audit_a" / filename).read_bytes(),
                (self.root / "audit_b" / filename).read_bytes(),
            )

        payload = self.payload("audit_a")
        self.assertEqual(payload["scope"], "evaluation_only_dataset_audit")
        self.assertFalse(payload["semantics"]["natural_video"])
        self.assertFalse(payload["semantics"]["events_assessed"])
        scene = payload["scenes"][0]
        self.assertEqual(scene["scene_id"], "000009")
        self.assertEqual(
            scene["obj_id_multiplicities"],
            [{"count": 2, "obj_id": 1}, {"count": 1, "obj_id": 2}],
        )
        self.assertEqual(scene["repeated_visual_type_obj_ids"], [1])
        self.assertEqual(scene["strict_visual_type_obj_ids"], [1])
        self.assertTrue(scene["obj_id_order_stable_across_images"])
        self.assertTrue(scene["strict_visual_type_scene_candidate"])
        validation = scene["physical_instance_candidate_validation"]
        self.assertEqual(validation["validated_candidate_count"], 3)
        self.assertTrue(validation["all_gt_ids_validated_as_physical_instances"])
        self.assertEqual(validation["maximum_world_position_delta_mm"], 0.0)
        self.assertTrue(all(item["physical_instance_id"] for item in validation["candidates"]))
        self.assertEqual(scene["visibility_fraction"]["minimum"], 0.6)
        self.assertEqual(scene["visibility_fraction"]["median"], 0.775)
        self.assertEqual(scene["member_coverage"]["rgb"]["coverage"], 1.0)
        self.assertEqual(scene["member_coverage"]["mask"]["member_count"], 6)

        manifest = json.loads(
            (self.root / "audit_a" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source_archive_fingerprint"]["sha256"], archive_sha256)
        self.assertEqual(manifest["source_archive_fingerprint"]["crc_validation"], "passed_all_members")
        for artifact in manifest["artifacts"]:
            data = (self.root / "audit_a" / artifact["path"]).read_bytes()
            self.assertEqual(artifact["size_bytes"], len(data))
            self.assertEqual(artifact["sha256"], hashlib.sha256(data).hexdigest())

        with (self.root / "audit_a" / "scene_inventory.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["strict_visual_type_scene_candidate"], "true")
        self.assertEqual(row["repeated_visual_type_obj_ids"], "[1]")

    def test_same_obj_id_swap_is_caught_by_world_position_validation(self) -> None:
        first = [
            (1, [100.0, 0.0, 700.0]),
            (1, [250.0, 0.0, 700.0]),
        ]
        swapped = [first[1], first[0]]
        views = [
            {"image_id": 0, "objects": first},
            {"image_id": 1, "objects": swapped},
        ]
        archive = self.write_archive("swapped.zip", self.scene_members("000020", views))

        self.run_audit(archive, tolerance_mm=1.0)

        scene = self.payload()["scenes"][0]
        self.assertTrue(scene["obj_id_order_stable_across_images"])
        self.assertEqual(scene["repeated_visual_type_obj_ids"], [1])
        self.assertFalse(scene["strict_visual_type_scene_candidate"])
        validation = scene["physical_instance_candidate_validation"]
        self.assertEqual(validation["validated_candidate_count"], 0)
        self.assertFalse(validation["all_gt_ids_validated_as_physical_instances"])
        for candidate in validation["candidates"]:
            self.assertTrue(candidate["obj_id_stable_across_images"])
            self.assertFalse(candidate["world_position_stable_within_tolerance"])
            self.assertEqual(candidate["maximum_pairwise_world_position_delta_mm"], 150.0)
            self.assertIsNone(candidate["physical_instance_id"])

    def test_reports_unstable_obj_id_order_without_promoting_gt_ids(self) -> None:
        views = [
            {
                "image_id": 0,
                "objects": [
                    (1, [100.0, 0.0, 700.0]),
                    (2, [200.0, 0.0, 700.0]),
                ],
            },
            {
                "image_id": 1,
                "objects": [
                    (2, [100.0, 0.0, 700.0]),
                    (1, [200.0, 0.0, 700.0]),
                ],
            },
        ]
        archive = self.write_archive("order.zip", self.scene_members("000003", views))

        self.run_audit(archive)

        payload = self.payload()
        scene = payload["scenes"][0]
        self.assertFalse(scene["obj_id_order_stable_across_images"])
        self.assertEqual(scene["different_obj_id_order_image_ids"], [1])
        self.assertEqual(payload["summary"]["scene_ids_with_unstable_obj_id_order"], ["000003"])
        candidates = scene["physical_instance_candidate_validation"]["candidates"]
        self.assertTrue(all(not item["obj_id_stable_across_images"] for item in candidates))
        self.assertTrue(all(item["physical_instance_id"] is None for item in candidates))

    def test_rejects_unsafe_and_duplicate_zip_paths(self) -> None:
        unsafe_cases = (
            "../escape.txt",
            "/absolute.txt",
            "C:/drive.txt",
            "test_primesense//000001/scene_gt.json",
        )
        for index, member_name in enumerate(unsafe_cases):
            with self.subTest(member_name=member_name):
                archive = self.write_archive(
                    f"unsafe_{index}.zip", [(member_name, b"unsafe")]
                )
                with self.assertRaisesRegex(ValueError, "ZIP member path|forward slashes"):
                    self.run_audit(archive, f"unsafe_output_{index}")
                self.assertFalse((self.root / f"unsafe_output_{index}").exists())

        backslash_info = zipfile.ZipInfo("placeholder")
        backslash_info.filename = "test_primesense\\000001\\scene_gt.json"
        with self.assertRaisesRegex(ValueError, "forward slashes"):
            TOOL._safe_member_name(backslash_info)

        duplicate_path = "test_primesense/000001/scene_gt.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate_archive = self.write_archive(
                "duplicate.zip",
                [(duplicate_path, b"{}"), (duplicate_path, b"{}")],
            )
        with self.assertRaisesRegex(ValueError, "duplicate member path"):
            self.run_audit(duplicate_archive, "duplicate_output")
        self.assertFalse((self.root / "duplicate_output").exists())

    def test_rejects_incomplete_member_coverage_and_malformed_json(self) -> None:
        complete = self.scene_members("000009", self.stable_duplicate_views())
        missing_mask_members = [
            item
            for item in complete
            if item[0] != "test_primesense/000009/mask/000005_000001.png"
        ]
        missing_mask_archive = self.write_archive("missing_mask.zip", missing_mask_members)
        with self.assertRaisesRegex(ValueError, "Incomplete or malformed.*mask"):
            self.run_audit(missing_mask_archive, "missing_mask_output")
        self.assertFalse((self.root / "missing_mask_output").exists())

        malformed_members = [
            (name, b"{" if name.endswith("/scene_gt.json") else payload)
            for name, payload in complete
        ]
        malformed_archive = self.write_archive("malformed_json.zip", malformed_members)
        with self.assertRaisesRegex(ValueError, "Malformed JSON"):
            self.run_audit(malformed_archive, "malformed_json_output")
        self.assertFalse((self.root / "malformed_json_output").exists())

        missing_info_field: list[tuple[str, bytes]] = []
        for name, payload in complete:
            if name.endswith("/scene_gt_info.json"):
                info = json.loads(payload)
                del info["0"][0]["visib_fract"]
                payload = self.json_bytes(info)
            missing_info_field.append((name, payload))
        incomplete_json_archive = self.write_archive(
            "incomplete_json.zip", missing_info_field
        )
        with self.assertRaisesRegex(ValueError, "missing required field 'visib_fract'"):
            self.run_audit(incomplete_json_archive, "incomplete_json_output")
        self.assertFalse((self.root / "incomplete_json_output").exists())

    def test_rejects_misaligned_json_image_keys_and_parallel_records(self) -> None:
        complete = self.scene_members("000009", self.stable_duplicate_views())
        mismatched_keys: list[tuple[str, bytes]] = []
        for name, payload in complete:
            if name.endswith("/scene_camera.json"):
                cameras = json.loads(payload)
                del cameras["5"]
                payload = self.json_bytes(cameras)
            mismatched_keys.append((name, payload))
        key_archive = self.write_archive("key_mismatch.zip", mismatched_keys)
        with self.assertRaisesRegex(ValueError, "JSON image keys do not align"):
            self.run_audit(key_archive, "key_mismatch_output")

        parallel_mismatch: list[tuple[str, bytes]] = []
        for name, payload in complete:
            if name.endswith("/scene_gt_info.json"):
                info = json.loads(payload)
                info["0"].pop()
                payload = self.json_bytes(info)
            parallel_mismatch.append((name, payload))
        parallel_archive = self.write_archive("parallel_mismatch.zip", parallel_mismatch)
        with self.assertRaisesRegex(ValueError, "scene_gt records but"):
            self.run_audit(parallel_archive, "parallel_mismatch_output")

    def test_accepts_bop_visible_count_above_depth_valid_count(self) -> None:
        members = self.scene_members("000001", self.stable_duplicate_views())
        adjusted: list[tuple[str, bytes]] = []
        for name, payload in members:
            if name.endswith("/scene_gt_info.json"):
                info = json.loads(payload)
                info["0"][0]["px_count_valid"] = 80
                info["0"][0]["px_count_visib"] = 90
                payload = self.json_bytes(info)
            adjusted.append((name, payload))
        archive = self.write_archive("bop_counts.zip", adjusted)

        result = self.run_audit(archive)

        self.assertEqual(result.scene_count, 1)
        self.assertTrue(self.payload()["scenes"][0]["strict_visual_type_scene_candidate"])

    def test_expected_fingerprint_and_counts_fail_before_publication(self) -> None:
        archive = self.write_archive(
            "expected.zip", self.scene_members("000009", self.stable_duplicate_views())
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            self.run_audit(
                archive,
                "bad_sha",
                expected_archive_sha256="0" * 64,
            )
        self.assertFalse((self.root / "bad_sha").exists())

        with self.assertRaisesRegex(ValueError, "Expected 2 T-LESS scenes"):
            self.run_audit(archive, "bad_scenes", expected_scenes=2)
        self.assertFalse((self.root / "bad_scenes").exists())

        with self.assertRaisesRegex(ValueError, "Expected 3 total T-LESS images"):
            self.run_audit(archive, "bad_images", expected_images=3)
        self.assertFalse((self.root / "bad_images").exists())

    def test_refuses_overwrite_and_cleans_staging_after_write_failure(self) -> None:
        archive = self.write_archive(
            "atomic.zip", self.scene_members("000009", self.stable_duplicate_views())
        )
        destination = self.root / "audit"

        with mock.patch.object(TOOL, "_write_csv", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.run_audit(archive)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".audit-*")), [])

        self.run_audit(archive)
        original_manifest = (destination / "artifact_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_audit(archive)
        self.assertEqual((destination / "artifact_manifest.json").read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
