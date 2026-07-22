from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tools.run_ocid_pipeline as ocid_pipeline
from tools.ocid_pipeline_contract import (
    DEFAULT_PROTOCOL,
    OcidPipelineError,
    git_state,
    load_pipeline_protocol,
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
class OcidPipelineRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_pipeline_protocol(DEFAULT_PROTOCOL)

    def test_protocol_constructs_exact_selected_detector_and_matching_configs(self) -> None:
        prompt, geometry = ocid_pipeline._detector_profiles(self.protocol)
        self.assertEqual(prompt.text, "object.")
        self.assertEqual(geometry.geometry_id, "g_surface_span_055_090")
        self.assertEqual(geometry.min_area_ratio, 0.55)
        self.assertEqual(geometry.min_span_ratio, 0.90)
        scoring, matching, grouping, events = ocid_pipeline._matching_configs(self.protocol)
        self.assertEqual(scoring.visual_gate, 0.80)
        self.assertIsNone(scoring.spatial_gate)
        self.assertEqual(matching.unmatched_pair_cost, 0.70)
        self.assertEqual(matching.local_margin_gate, 0.01)
        self.assertEqual(grouping.representation_variant_id, "mask_neutral_letterbox_v1")
        self.assertEqual(grouping.coframe_visual_gate, 0.963)
        self.assertTrue(grouping.coframe_margin_bypass)
        self.assertEqual(events.position_threshold_norm, 0.10)
        self.assertEqual(ocid_pipeline._evaluation_iou_grid(self.protocol), (0.5, 0.6, 0.7, 0.8, 0.9))

    def test_heldout_inference_cannot_start_without_access_or_create_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            with self.assertRaisesRegex(OcidPipelineError, "requires an evaluation access"):
                ocid_pipeline.run_inference(
                    protocol_path=DEFAULT_PROTOCOL,
                    role="heldout",
                    output_root=output,
                    run_id="ocid-pipeline-heldout-v1",
                )
            self.assertFalse((output / "ocid-pipeline-heldout-v1").exists())

    def test_sample_rejects_access_argument_before_stream_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            access = Path(temporary) / "access.json"
            access.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OcidPipelineError, "does not accept"):
                ocid_pipeline.run_inference(
                    protocol_path=DEFAULT_PROTOCOL,
                    role="sample",
                    output_root=Path(temporary) / "runs",
                    run_id="development-smoke",
                    access_path=access,
                )

    def test_sample_rejects_unsafe_run_id_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "runs"
            with self.assertRaisesRegex(OcidPipelineError, "portable path segment"):
                ocid_pipeline.run_inference(
                    protocol_path=DEFAULT_PROTOCOL,
                    role="sample",
                    output_root=output,
                    run_id="../escape",
                )
            self.assertFalse(output.exists())

    def test_fixed_development_stream_uses_canonical_manifest_indices(self) -> None:
        item = self.protocol.development[
            next(
                index
                for index, row in enumerate(self.protocol.development)
                if row.stream_id == "ocid_arid10_table_bottom_box_seq05"
            )
        ]
        decoded = ocid_pipeline._load_stream(item, self.protocol.sha256)
        self.assertEqual(decoded.stream.envelope.producer.producer_stage, "stream_input")
        self.assertEqual([frame.record.index for frame in decoded.frames], list(range(1, 12)))

    def test_artifact_inventory_excludes_mutable_boundary_files_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "attempt.json").write_text("{}", encoding="utf-8")
            (root / "inference_manifest.json").write_text("{}", encoding="utf-8")
            artifact = root / "stage" / "artifact.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"value":1}', encoding="utf-8")
            rows = ocid_pipeline._artifact_inventory(root)
            self.assertEqual([row["path"] for row in rows], ["stage/artifact.json"])
            ocid_pipeline._verify_artifact_inventory(root, rows)
            artifact.write_text('{"value":2}', encoding="utf-8")
            with self.assertRaisesRegex(OcidPipelineError, "artifact changed"):
                ocid_pipeline._verify_artifact_inventory(root, rows)

    def test_binary_mask_and_embedding_artifacts_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((8, 9), dtype=np.bool_)
            mask[2:6, 3:8] = True
            path = root / "mask.png"
            ocid_pipeline._save_binary_mask(path, mask)
            np.testing.assert_array_equal(ocid_pipeline._load_binary_mask(path), mask)

            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            embedding_path = root / "embeddings.npz"
            ocid_pipeline._save_embeddings(embedding_path, embeddings)
            with np.load(embedding_path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"embeddings"})
                np.testing.assert_array_equal(archive["embeddings"], embeddings)

    def test_check_output_is_structured_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "check.json"
            receipt = ocid_pipeline.run_check(protocol_path=DEFAULT_PROTOCOL, output_path=output)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted, receipt)
            self.assertEqual(persisted["protocol"]["sha256"], self.protocol.sha256)
            self.assertFalse(persisted["heldout_lock"]["predictions_unlocked"])
            with self.assertRaisesRegex(OcidPipelineError, "refusing to overwrite"):
                ocid_pipeline.run_check(protocol_path=DEFAULT_PROTOCOL, output_path=output)

    def test_stage_failure_is_detected_before_evaluation_inputs_are_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_minimal_inference(root)
            with (
                patch.object(
                    ocid_pipeline,
                    "_verify_artifact_inventory",
                ),
                patch.object(
                    ocid_pipeline,
                    "_verified_stage_manifest",
                    side_effect=OcidPipelineError("broken stage"),
                ),
                patch.object(ocid_pipeline, "validate_evaluation_inputs") as evaluation_inputs,
            ):
                with self.assertRaisesRegex(OcidPipelineError, "broken stage"):
                    self._run_minimal_development_evaluation(root)
            evaluation_inputs.assert_not_called()
            attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
            self.assertFalse(attempt["ground_truth_opened"])
            self.assertEqual(attempt["status"], "inference_completed_before_ground_truth")

    def test_evaluation_failure_after_boundary_is_not_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_minimal_inference(root)
            with (
                patch.object(ocid_pipeline, "_verify_artifact_inventory"),
                patch.object(ocid_pipeline, "_verified_stage_manifest", return_value={}),
                patch.object(ocid_pipeline, "_verify_analysis_inputs_unchanged"),
                patch.object(ocid_pipeline, "_verify_stage_contracts"),
                patch.object(ocid_pipeline, "_snapshots_from_persisted_masks", return_value={}),
                patch.object(ocid_pipeline, "_verify_analysis_embeddings", return_value={}),
                patch.object(
                    ocid_pipeline,
                    "validate_evaluation_inputs",
                    side_effect=OcidPipelineError("frozen GT unavailable"),
                ) as evaluation_inputs,
            ):
                with self.assertRaisesRegex(OcidPipelineError, "frozen GT unavailable"):
                    self._run_minimal_development_evaluation(root)
            attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
            self.assertTrue(attempt["ground_truth_opened"])
            self.assertEqual(attempt["status"], "evaluation_failed_after_ground_truth_boundary")
            self.assertEqual(evaluation_inputs.call_count, 1)

            with (
                patch.object(ocid_pipeline, "_verify_artifact_inventory"),
                patch.object(ocid_pipeline, "_verified_stage_manifest", return_value={}),
                patch.object(ocid_pipeline, "_verify_analysis_inputs_unchanged"),
                patch.object(ocid_pipeline, "_verify_stage_contracts"),
                patch.object(ocid_pipeline, "_snapshots_from_persisted_masks", return_value={}),
                patch.object(ocid_pipeline, "_verify_analysis_embeddings", return_value={}),
                patch.object(ocid_pipeline, "validate_evaluation_inputs") as second_open,
            ):
                with self.assertRaisesRegex(OcidPipelineError, "not eligible"):
                    self._run_minimal_development_evaluation(root)
            second_open.assert_not_called()

    def _write_minimal_inference(self, root: Path) -> None:
        item = next(
            row
            for row in self.protocol.development
            if row.stream_id == "ocid_arid10_table_bottom_box_seq05"
        )
        state = git_state(self.protocol.repository_root)
        inference = {
            "schema_version": ocid_pipeline.INFERENCE_SCHEMA,
            "status": "inference_completed_before_ground_truth",
            "run_id": "development-smoke-test",
            "role": "sample",
            "protocol": {"sha256": self.protocol.sha256},
            "git": {"commit": state.commit, "branch": state.branch},
            "inventory": {"stream_ids": [item.stream_id]},
            "ground_truth_boundary": {
                "ground_truth_opened": False,
                "all_prediction_artifacts_persisted": True,
            },
            "artifact_inventory": [{"path": "placeholder"}],
        }
        attempt = {
            "schema_version": ocid_pipeline.ATTEMPT_SCHEMA,
            "run_id": inference["run_id"],
            "role": inference["role"],
            "protocol_sha256": self.protocol.sha256,
            "status": "inference_completed_before_ground_truth",
            "ground_truth_opened": False,
        }
        (root / "inference_manifest.json").write_text(json.dumps(inference), encoding="utf-8")
        (root / "attempt.json").write_text(json.dumps(attempt), encoding="utf-8")

    def _run_minimal_development_evaluation(self, root: Path) -> dict[str, object]:
        item = next(
            row
            for row in self.protocol.development
            if row.stream_id == "ocid_arid10_table_bottom_box_seq05"
        )
        return ocid_pipeline._run_evaluation(
            protocol=self.protocol,
            run_directory=root,
            expected_role="sample",
            inventory=(item,),
            output_filename="sample_evaluation.json",
            completion_status="completed_sample_evaluation",
            attempt_status="sample_evaluation_completed",
            access_path=None,
        )


if __name__ == "__main__":
    unittest.main()
