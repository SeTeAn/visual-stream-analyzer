from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tools.run_ocid_pipeline as ocid_pipeline
from stream_analysis.contracts import BBox, ImageSize
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


class OcidPipelinePureUnitTest(unittest.TestCase):
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
            with self.assertRaisesRegex(OcidPipelineError, "non-empty"):
                ocid_pipeline._verify_artifact_inventory(root, [])
            extra = root / "stage" / "unlisted.json"
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OcidPipelineError, "exact immutable file set"):
                ocid_pipeline._verify_artifact_inventory(root, rows)
            extra.unlink()
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

    def test_aggregate_resolution_filters_only_the_global_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = {
                "global": np.ones((20, 20), dtype=np.bool_),
                "a": np.pad(np.ones((4, 4), dtype=np.bool_), ((1, 15), (1, 15))),
                "b": np.pad(np.ones((4, 4), dtype=np.bool_), ((1, 15), (8, 8))),
                "c": np.pad(np.ones((4, 4), dtype=np.bool_), ((8, 8), (1, 15))),
            }
            rows = []
            for candidate_id, mask in masks.items():
                path = root / "masks" / f"{candidate_id}.png"
                ocid_pipeline._save_binary_mask(path, mask)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "frame_id": "frame_0001",
                        "source_bbox": {"x": 0, "y": 0, "width": 20, "height": 20},
                        "cleaned_mask": {"path": path.relative_to(root).as_posix()},
                    }
                )
            kept, report = ocid_pipeline._resolve_stream_aggregate_masks(
                run_directory=root,
                stream_id="stream",
                records=rows,
            )
            self.assertEqual(set(kept), {"a", "b", "c"})
            self.assertEqual(report["removed_candidate_count"], 1)
            self.assertEqual(report["frames"][0]["removed_candidate_ids"], ["global"])

    def test_temporal_adapter_adds_supported_low_score_candidate(self) -> None:
        frame_size = ImageSize(width=100, height=100)
        decoded = SimpleNamespace(
            frames=(
                SimpleNamespace(frame_id="frame_0001", image_size=frame_size),
                SimpleNamespace(frame_id="frame_0002", image_size=frame_size),
            )
        )
        raw = (
            SimpleNamespace(
                frame_id="frame_0001",
                frame_index=1,
                prediction_index=0,
                score=0.10,
                phrase="object",
                bbox=BBox(x=10, y=10, width=20, height=20),
            ),
            SimpleNamespace(
                frame_id="frame_0002",
                frame_index=2,
                prediction_index=0,
                score=0.20,
                phrase="object",
                bbox=BBox(x=10, y=10, width=20, height=20),
            ),
        )
        baseline = (
            {
                "frame_id": "frame_0002",
                "frame_index": 2,
                "prediction_index": 0,
                "candidate_id": "base",
                "score": 0.20,
                "phrase": "object",
                "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                "geometry_rejected": False,
            },
        )
        accepted, report = ocid_pipeline._temporal_detector_records(
            raw=raw,
            decoded=decoded,
            prompt=SimpleNamespace(prompt_id="p_object"),
            geometry=SimpleNamespace(min_area_ratio=None, min_span_ratio=None),
            baseline_accepted=baseline,
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(report["base_candidate_count"], 1)
        self.assertEqual(report["supplemental_candidate_count"], 1)
        self.assertFalse(report["ground_truth_used_for_selection"])

    def test_mask_assignment_counts_false_positives_and_misses(self) -> None:
        matrix = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.0, 0.0]])
        assignments = ocid_pipeline._maximum_cardinality_mask_assignment(matrix, 0.5)
        self.assertEqual(assignments, ((0, 0, 0.9),))
        metrics = ocid_pipeline._detection_count_metrics(
            {"tp": len(assignments), "fp": 2, "fn": 1}
        )
        self.assertEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["recall"], 1 / 2)
        self.assertEqual(metrics["f1"], 0.4)

    def test_candidate_overlay_label_contains_only_frame_local_ordinal(self) -> None:
        self.assertEqual(ocid_pipeline._candidate_overlay_label(0), "P00")
        self.assertEqual(ocid_pipeline._candidate_overlay_label(29), "P29")
        self.assertNotIn(".", ocid_pipeline._candidate_overlay_label(1))
        with self.assertRaises(ValueError):
            ocid_pipeline._candidate_overlay_label(-1)


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

    def test_development_rejects_access_argument_before_stream_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            access = Path(temporary) / "access.json"
            access.write_text("{}", encoding="utf-8")
            output = Path(temporary) / "runs"
            with self.assertRaisesRegex(OcidPipelineError, "does not accept"):
                ocid_pipeline.run_inference(
                    protocol_path=DEFAULT_PROTOCOL,
                    role="development",
                    output_root=output,
                    run_id="development-full",
                    access_path=access,
                )
            self.assertFalse(output.exists())

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

    def test_evaluation_rejects_modified_inference_manifest_before_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_minimal_inference(root)
            inference_path = root / "inference_manifest.json"
            inference = json.loads(inference_path.read_text(encoding="utf-8"))
            inference["role"] = "development"
            inference_path.write_text(json.dumps(inference), encoding="utf-8")
            with patch.object(ocid_pipeline, "validate_evaluation_inputs") as evaluation_inputs:
                with self.assertRaisesRegex(OcidPipelineError, "manifest changed"):
                    self._run_minimal_development_evaluation(root)
            evaluation_inputs.assert_not_called()
            attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
            self.assertFalse(attempt["ground_truth_opened"])

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
                patch.object(ocid_pipeline, "_verify_implementation_receipt"),
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
                patch.object(ocid_pipeline, "_verify_implementation_receipt"),
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
        inference_path = root / "inference_manifest.json"
        inference_path.write_text(json.dumps(inference), encoding="utf-8")
        attempt["inference_manifest_sha256"] = ocid_pipeline.sha256_file(inference_path)
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
