from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ocid_evaluation_common import (  # noqa: E402
    IOU_GRID,
    OcidEvaluationError,
    evaluate_profiles,
    load_development_inventory,
    validate_development_input_pairs,
    write_canonical_json_atomic,
)
from stream_analysis.evaluation import PredictedCandidate  # noqa: E402
from stream_analysis.evaluation.annotation import load_annotation  # noqa: E402


SPEC = ROOT / "data" / "ocid" / "benchmark" / "ocid_candidate_benchmark_v1.json"
REVIEWED = ROOT / "data" / "ocid" / "derived" / "ocid_candidate_benchmark_v1_reviewed"


class OcidEvaluationCommonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = load_development_inventory(SPEC, REVIEWED)

    def test_exact_configured_development_inventory(self) -> None:
        self.assertEqual(len(self.inventory), 10)
        self.assertEqual(sum(item.frame_count for item in self.inventory), 148)
        self.assertEqual(len({item.scene_group_id for item in self.inventory}), 5)
        self.assertFalse(any("seq02" in item.stream_id for item in self.inventory))
        self.assertTrue(all(item.annotation_path.is_file() for item in self.inventory))

    def test_rejects_role_overlap_and_wrong_total(self) -> None:
        baseline = json.loads(SPEC.read_text(encoding="utf-8"))
        for mutate, message in (
            (lambda payload: payload["roles"]["development"][0]["member_streams"].__setitem__(0, payload["roles"]["heldout"][0]["member_streams"][0]), "held-out"),
            (lambda payload: payload["expected_totals"].update(development_frames=1), "development_frames"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                payload = json.loads(json.dumps(baseline))
                mutate(payload)
                path = Path(temporary) / "spec.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(OcidEvaluationError, message):
                    load_development_inventory(path, REVIEWED)

    def test_rejects_manifest_role_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._metadata_only_fixture(Path(temporary))
            item = self.inventory[0]
            manifest_path = fixture / "analysis_streams" / item.stream_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["role"] = "heldout"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(OcidEvaluationError, "role mismatch"):
                load_development_inventory(SPEC, fixture)

    def test_rejects_frame_path_escaping_stream_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._metadata_only_fixture(Path(temporary))
            item = self.inventory[0]
            manifest_path = fixture / "analysis_streams" / item.stream_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frames"][0]["image_path"] = "../not_a_frame.png"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(OcidEvaluationError, "escapes reviewed root"):
                load_development_inventory(SPEC, fixture)

    def test_iou_loop_and_scene_group_macro(self) -> None:
        predictions = {}
        for stream in self.inventory:
            annotation = load_annotation(stream.annotation_path, manifest_path=stream.stream_directory / "manifest.json")
            predictions[stream.stream_id] = tuple(
                PredictedCandidate(
                    candidate_id=f"pred:{instance.instance_id}", frame_id=instance.frame_id,
                    bbox=instance.bbox, validity_status="valid", warning_ids=(), error_ids=(),
                ) for instance in annotation.instances
            )
        report = evaluate_profiles(self.inventory, {"oracle_boxes": predictions})
        self.assertEqual(report["iou_grid"], list(IOU_GRID))
        summary = report["profiles"]["oracle_boxes"]["summary_by_iou"]["0.90"]
        self.assertEqual(summary["pooled"]["fp"], 0)
        self.assertEqual(summary["pooled"]["fn"], 0)
        self.assertEqual(summary["pooled"]["f1"], 1.0)
        self.assertEqual(summary["scene_group_macro"]["f1"], 1.0)
        self.assertEqual(len(summary["scene_groups"]), 5)
        self.assertEqual(summary["worst_scene_group"]["metrics"]["f1"], 1.0)

    def test_rejects_missing_or_heldout_prediction_keys(self) -> None:
        with self.assertRaisesRegex(OcidEvaluationError, "exactly the development streams"):
            evaluate_profiles(self.inventory, {"bad": {"ocid_arid20_table_bottom_seq02": ()}})

    def test_no_predictions_count_as_zero_in_macro_selection_metrics(self) -> None:
        predictions = {stream.stream_id: () for stream in self.inventory}
        report = evaluate_profiles(self.inventory, {"no_predictions": predictions})
        summary = report["profiles"]["no_predictions"]["summary_by_iou"]["0.50"]
        self.assertEqual(summary["pooled"]["fn"], 1170)
        self.assertEqual(summary["scene_group_macro"]["precision"], 0.0)
        self.assertEqual(summary["scene_group_macro"]["recall"], 0.0)
        self.assertEqual(summary["scene_group_macro"]["f1"], 0.0)
        # The raw precision/F1 stay undefined when no positives were predicted;
        # only the macro/selection interpretation maps this failed detection to 0.
        self.assertIsNone(summary["worst_scene_group"]["metrics"]["f1"])

    def test_exact_input_pairs_reject_subset_and_cross_paired_annotations(self) -> None:
        streams = tuple(item.stream_directory for item in self.inventory)
        annotations = tuple(item.annotation_path for item in self.inventory)
        self.assertEqual(
            validate_development_input_pairs(
                streams,
                annotations,
                benchmark_spec_path=SPEC,
                reviewed_root=REVIEWED,
            ),
            self.inventory,
        )
        with self.assertRaisesRegex(OcidEvaluationError, "exactly the canonical development"):
            validate_development_input_pairs(
                streams[:-1],
                annotations[:-1],
                benchmark_spec_path=SPEC,
                reviewed_root=REVIEWED,
            )
        cross_paired = (annotations[1], annotations[0], *annotations[2:])
        with self.assertRaisesRegex(OcidEvaluationError, "exactly the canonical development"):
            validate_development_input_pairs(
                streams,
                cross_paired,
                benchmark_spec_path=SPEC,
                reviewed_root=REVIEWED,
            )

    def test_atomic_canonical_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "report.json"
            write_canonical_json_atomic(path, {"z": 1, "a": ["x"]})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": [\n    "x"\n  ],\n  "z": 1\n}\n')
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def _metadata_only_fixture(self, target: Path) -> Path:
        for item in self.inventory:
            stream = target / "analysis_streams" / item.stream_id
            annotation = target / "evaluation_annotations" / item.stream_id
            stream.mkdir(parents=True, exist_ok=True)
            annotation.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.stream_directory / "manifest.json", stream / "manifest.json")
            shutil.copy2(item.annotation_path, annotation / "annotation.json")
        return target


if __name__ == "__main__":
    unittest.main()
