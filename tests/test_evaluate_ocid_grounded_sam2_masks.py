from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from stream_analysis.contracts import BBox
from stream_analysis.evaluation.component_review import analyze_component_mask
from tools import evaluate_ocid_grounded_sam2_masks as subject
from tools.ocid_evaluation_common import DevelopmentStream, OcidEvaluationError


def _inventory(root: Path) -> tuple[DevelopmentStream, ...]:
    return (
        DevelopmentStream(
            stream_id="stream_a",
            scene_group_id="scene_a",
            frame_count=1,
            stream_directory=root / "stream_a",
            annotation_path=root / "annotation.json",
        ),
    )


class GroundedSam2MaskTest(unittest.TestCase):
    def test_selected_manifest_is_exact_and_development_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "schema_version": subject.SELECTED_CANDIDATE_SCHEMA,
                "scope": "development_only_selected_candidates",
                "heldout_access": "none",
                "selected_profile_id": "p_object__g_none",
                "streams": {
                    "stream_a": {
                        "frame_sizes": {"frame_0001": {"width": 8, "height": 6}},
                        "candidates": [
                            {
                                "candidate_id": "candidate:a",
                                "frame_id": "frame_0001",
                                "frame_index": 0,
                                "score": 0.8,
                                "phrase": "object",
                                "bbox": {"x": 1, "y": 1, "width": 3, "height": 2},
                                "geometry_rejected": False,
                            }
                        ],
                    }
                },
            }
            path = root / "selected.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parsed, loaded = subject.load_selected_candidates(path, _inventory(root))
            self.assertEqual(parsed["stream_a"][0].bbox, BBox(1, 1, 3, 2))
            self.assertEqual(loaded["selected_profile_id"], "p_object__g_none")

            payload["heldout_access"] = "accessed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(OcidEvaluationError, "development-only"):
                subject.load_selected_candidates(path, _inventory(root))

    def test_pixel_metrics_and_bbox_mask_are_half_open(self) -> None:
        expected = np.zeros((4, 5), dtype=bool)
        expected[1:3, 1:3] = True
        predicted = np.zeros((4, 5), dtype=bool)
        predicted[1:3, 2:4] = True
        metrics = subject._pixel_metrics(predicted, expected)
        self.assertEqual(metrics["intersection_pixels"], 2)
        self.assertEqual(metrics["iou"], 2 / 6)
        box = subject._bbox_mask(BBox(1, 1, 2, 2), subject.ImageSize(width=5, height=4))
        self.assertTrue(np.array_equal(box, expected))

    def test_profile_selection_obeys_frozen_safety_rule(self) -> None:
        rows = [
            {
                "raw": {"iou": 0.70, "precision": 0.75},
                "cleaned": {"iou": 0.71, "precision": 0.77},
            },
            {
                "raw": {"iou": 0.80, "precision": 0.82},
                "cleaned": {"iou": 0.80, "precision": 0.83},
            },
        ]
        selected = subject._mask_profile_selection(rows, removed_count=2)
        self.assertEqual(selected["selected_profile"], "m_conservative_components")
        retained = subject._mask_profile_selection(rows, removed_count=0)
        self.assertEqual(retained["selected_profile"], "m_raw")

        unsafe = [
            {"raw": {"iou": 0.70, "precision": 0.75}, "cleaned": {"iou": 0.68, "precision": 0.80}}
        ]
        retained = subject._mask_profile_selection(unsafe, removed_count=1)
        self.assertIn("individual_iou_safety_failed", retained["ineligibility_reasons"])

    def test_mask_artifact_round_trip_preserves_binary_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "mask.png"
            mask = np.zeros((6, 7), dtype=bool)
            mask[1:5, 2:6] = True
            subject._save_mask(path, mask)
            loaded = subject._load_mask(path)
            self.assertTrue(np.array_equal(mask, loaded))

    def test_model_asset_provenance_hashes_every_required_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(subject.SAM2_CONFIG_FILES):
                (root / name).write_bytes(f"asset-{index}".encode("ascii"))
            result = subject._model_asset_provenance(root)
            self.assertEqual(set(result), set(subject.SAM2_CONFIG_FILES))
            self.assertTrue(all(len(row["sha256"]) == 64 for row in result.values()))
            (root / subject.SAM2_CONFIG_FILES[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                subject._model_asset_provenance(root)

    def test_prepared_mask_uses_configured_retained_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_path = root / "ARID" / "table" / "bottom" / "seq" / "label" / "source.png"
            label_path.parent.mkdir(parents=True)
            labels = np.zeros((6, 8), dtype=np.uint8)
            labels[1:4, 1:4] = 3
            labels[5, 7] = 3
            Image.fromarray(labels, mode="L").save(label_path)
            analysis = analyze_component_mask(labels == 3)
            case_id = "stream_a__frame_0001__label_003"
            mask, provenance = subject._reviewed_mask(
                stream_id="stream_a",
                frame_id="frame_0001",
                instance_payload={"source_label": 3},
                source_sequence="ARID/table/bottom/seq",
                source_filename="source.png",
                ocid_root=root,
                decisions={
                    case_id: {
                        "analysis": {"source_mask_sha256": analysis.source_mask_sha256},
                        "final_retained_component_ids": ["c001"],
                    }
                },
            )
            self.assertEqual(int(mask.sum()), 9)
            self.assertEqual(provenance["retained_component_ids"], ["c001"])

    def test_contract_failure_happens_before_selected_or_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    subject,
                    "load_development_inventory",
                    side_effect=OcidEvaluationError("held-out remains locked"),
                ),
                mock.patch.object(subject, "load_selected_candidates") as selected,
                mock.patch.object(subject, "run_rgb_inference") as inference,
            ):
                with self.assertRaisesRegex(OcidEvaluationError, "held-out"):
                    subject.run_benchmark(
                        selected_candidates_path=root / "selected.json",
                        model_directory=root / "model",
                        expected_model_sha256="0" * 64,
                        benchmark_spec_path=root / "spec.json",
                        reviewed_root=root / "reviewed",
                        ocid_root=root / "ocid",
                        device="cpu",
                        artifact_root=root / "artifacts",
                        output_path=root / "report.json",
                    )
            selected.assert_not_called()
            inference.assert_not_called()

    def test_help_does_not_run_benchmark(self) -> None:
        with mock.patch.object(subject, "run_benchmark") as runner:
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stdout(io.StringIO()):
                subject.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
