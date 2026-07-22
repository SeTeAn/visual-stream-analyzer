from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ocid_pipeline_contract import (
    DEFAULT_PROTOCOL,
    ACCESS_DECISION,
    ACCESS_CONSUMPTION_SCHEMA,
    ACCESS_SCHEMA,
    ACCESS_STATUS,
    OcidPipelineError,
    GitState,
    claim_evaluation_access,
    inventory_for_role,
    load_pipeline_protocol,
    build_check_report,
    validate_analysis_inputs,
    validate_evaluation_access,
    validate_evaluation_access_consumption,
    validate_evaluation_inputs,
    validate_run_id,
    write_json_exclusive,
)


_REPOSITORY_ROOT = DEFAULT_PROTOCOL.parents[3]
_LOCAL_OCID_PIPELINE_ASSETS_AVAILABLE = all(
    path.exists()
    for path in (
        _REPOSITORY_ROOT / "models" / "extractors" / "grounding-dino-tiny-hf-a2bb814" / "model.safetensors",
        _REPOSITORY_ROOT / "models" / "extractors" / "sam2.1-hiera-tiny-hf-de431c4" / "model.safetensors",
        _REPOSITORY_ROOT / "models" / "dinov2" / "runtime-source-7764ea0",
        _REPOSITORY_ROOT / "models" / "dinov2" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        _REPOSITORY_ROOT
        / "data"
        / "ocid"
        / "derived"
        / "ocid_candidate_benchmark_v1_reviewed"
        / "benchmark_manifest.json",
    )
)


@unittest.skipUnless(_LOCAL_OCID_PIPELINE_ASSETS_AVAILABLE, "requires local OCID pipeline assets")
class OcidPipelineContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_pipeline_protocol(DEFAULT_PROTOCOL)

    def test_exact_configured_metadata_inventory(self) -> None:
        self.assertEqual(len(self.protocol.development), 10)
        self.assertEqual(sum(item.frame_count for item in self.protocol.development), 148)
        self.assertEqual(len({item.scene_group_id for item in self.protocol.development}), 5)
        self.assertEqual(len(self.protocol.heldout), 6)
        self.assertEqual(sum(item.frame_count for item in self.protocol.heldout), 82)
        self.assertEqual(len({item.scene_group_id for item in self.protocol.heldout}), 3)

    def test_sample_inventory_is_one_fixed_stream(self) -> None:
        inventory = inventory_for_role(self.protocol, "sample")
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0].stream_id, "ocid_arid10_table_bottom_box_seq05")
        self.assertEqual(inventory[0].frame_count, 11)

    def test_build_check_report_does_not_claim_stream_or_ground_truth_access(self) -> None:
        receipt = build_check_report(self.protocol)
        self.assertFalse(receipt["benchmark"]["heldout_rgb_opened"])
        self.assertFalse(receipt["benchmark"]["heldout_annotation_opened"])
        self.assertFalse(receipt["heldout_lock"]["predictions_unlocked"])
        self.assertTrue(receipt["heldout_lock"]["explicit_confirmation_required"])

    def test_development_analysis_input_hashes_do_not_include_annotations(self) -> None:
        inventory = inventory_for_role(self.protocol, "sample")
        receipt = validate_analysis_inputs(self.protocol, inventory)
        self.assertEqual(receipt["verified_rgb_count"], 11)
        self.assertEqual(receipt["verified_manifest_count"], 1)
        self.assertFalse(receipt["ground_truth_opened"])
        self.assertTrue(
            all(row["path"].startswith("analysis_streams/") for row in receipt["artifacts"])
        )
        self.assertFalse(any("evaluation_annotations" in row["path"] for row in receipt["artifacts"]))

    def test_development_evaluation_inputs_pin_annotations_and_source_labels(self) -> None:
        inventory = inventory_for_role(self.protocol, "sample")
        receipt = validate_evaluation_inputs(self.protocol, inventory)
        self.assertEqual(receipt["verified_annotation_count"], 1)
        self.assertEqual(receipt["verified_source_label_count"], 11)
        self.assertEqual(
            receipt["source_label_inventory_sha256"],
            "7a36ac7cc82a1d26226e60dcc6dcb85b5279fe494daccc88b18f3ea2bcaaa38a",
        )
        self.assertTrue(
            all(
                row["path"].startswith("ARID10/table/bottom/box/seq05/label/")
                for row in receipt["source_labels"]
            )
        )

    def test_evaluation_access_is_bound_to_protocol_commit_and_exact_inventory(self) -> None:
        clean_state = GitState("a" * 40, "jmlc", True)
        payload = {
            "schema_version": ACCESS_SCHEMA,
            "status": ACCESS_STATUS,
            "decision": ACCESS_DECISION,
            "protocol_sha256": self.protocol.sha256,
            "execution_git_commit": clean_state.commit,
            "execution_git_branch": clean_state.branch,
            "run_id": "ocid-pipeline-heldout-v1",
            "heldout_stream_ids": [item.stream_id for item in self.protocol.heldout],
            "authorized_at_utc": "2026-07-20T20:00:00Z",
            "single_use": True,
            "automatic_retry": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "access.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.ocid_pipeline_contract.git_state", return_value=clean_state):
                accepted = validate_evaluation_access(
                    path,
                    self.protocol,
                    expected_run_id="ocid-pipeline-heldout-v1",
                )
                self.assertEqual(accepted["decision"], ACCESS_DECISION)
                payload["heldout_stream_ids"] = payload["heldout_stream_ids"][:-1]
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(OcidPipelineError, "exact configured stream"):
                    validate_evaluation_access(path, self.protocol)

    def test_evaluation_access_rejects_dirty_worktree(self) -> None:
        state = GitState("a" * 40, "jmlc", False)
        payload = {
            "schema_version": ACCESS_SCHEMA,
            "status": ACCESS_STATUS,
            "decision": ACCESS_DECISION,
            "protocol_sha256": self.protocol.sha256,
            "execution_git_commit": state.commit,
            "execution_git_branch": state.branch,
            "run_id": "ocid-pipeline-heldout-v1",
            "heldout_stream_ids": [item.stream_id for item in self.protocol.heldout],
            "authorized_at_utc": "2026-07-20T20:00:00Z",
            "single_use": True,
            "automatic_retry": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "access.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.ocid_pipeline_contract.git_state", return_value=state):
                with self.assertRaisesRegex(OcidPipelineError, "clean Git worktree"):
                    validate_evaluation_access(path, self.protocol)

    def test_evaluation_access_can_be_claimed_only_once(self) -> None:
        state = GitState("a" * 40, "jmlc", True)
        payload = {
            "schema_version": ACCESS_SCHEMA,
            "status": ACCESS_STATUS,
            "decision": ACCESS_DECISION,
            "protocol_sha256": self.protocol.sha256,
            "execution_git_commit": state.commit,
            "execution_git_branch": state.branch,
            "run_id": "ocid-pipeline-heldout-v1",
            "heldout_stream_ids": [item.stream_id for item in self.protocol.heldout],
            "authorized_at_utc": "2026-07-20T20:00:00Z",
            "single_use": True,
            "automatic_retry": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "access.json"
            run_directory = Path(temporary) / "run"
            run_directory.mkdir()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.ocid_pipeline_contract.git_state", return_value=state):
                claimed = claim_evaluation_access(
                    path,
                    self.protocol,
                    expected_run_id="ocid-pipeline-heldout-v1",
                    run_directory=run_directory,
                )
                self.assertEqual(
                    claimed["consumption_receipt"]["schema_version"],
                    ACCESS_CONSUMPTION_SCHEMA,
                )
                validated = validate_evaluation_access_consumption(
                    path,
                    self.protocol,
                    expected_run_id="ocid-pipeline-heldout-v1",
                    expected_run_directory=run_directory,
                )
                self.assertEqual(
                    validated["consumption_receipt"]["run_id"],
                    "ocid-pipeline-heldout-v1",
                )
                with self.assertRaisesRegex(OcidPipelineError, "refusing to overwrite"):
                    claim_evaluation_access(
                        path,
                        self.protocol,
                        expected_run_id="ocid-pipeline-heldout-v1",
                        run_directory=Path(temporary) / "another-run",
                    )

    def test_run_id_must_be_one_portable_path_segment(self) -> None:
        self.assertEqual(validate_run_id("ocid-pipeline-heldout-v1"), "ocid-pipeline-heldout-v1")
        for invalid in ("../escape", "nested/run", "CON", "trailing."):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(OcidPipelineError, "portable path segment"):
                    validate_run_id(invalid)

    def test_exclusive_json_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            write_json_exclusive(path, {"status": "first"})
            with self.assertRaisesRegex(OcidPipelineError, "refusing to overwrite"):
                write_json_exclusive(path, {"status": "second"})


if __name__ == "__main__":
    unittest.main()
