from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation.annotation import AnnotationInstance
from tools.ocid_gate_c1_common import DevelopmentStream, GateC1ContractError
from tools import prepare_ocid_gate_c1_review_pack as subject


def _inventory(root: Path) -> tuple[DevelopmentStream, ...]:
    streams: list[DevelopmentStream] = []
    for number in range(5):
        group = f"scene_{number:02d}"
        for view in ("bottom", "top"):
            stream_id = f"ocid_{group}_{view}"
            directory = root / stream_id
            (directory / "frames").mkdir(parents=True)
            (directory / "annotation.json").write_text("{}", encoding="utf-8")
            (directory / "manifest.json").write_text(
                json.dumps({"frames": [{"frame_id": "frame_0001", "image_path": "frames/frame_0001.png"}]}),
                encoding="utf-8",
            )
            Image.new("RGB", (48, 32), (90, 90, 90)).save(directory / "frames" / "frame_0001.png")
            streams.append(DevelopmentStream(stream_id, group, 1, directory, directory / "annotation.json"))
    return tuple(streams)


def _report(inventory: tuple[DevelopmentStream, ...]) -> dict:
    profile_id = "score_0.15_nms_0.30"
    streams = {}
    for index, stream in enumerate(inventory):
        fp = index % 3
        fn = (index + 1) % 2
        streams[stream.stream_id] = {
            "raw_predictions": [
                {"frame_id": "frame_0001", "frame_index": 0, "prediction_index": 0, "score": 0.90, "phrase": "object", "bbox": {"x": 2, "y": 3, "width": 10, "height": 8}},
                {"frame_id": "frame_0001", "frame_index": 0, "prediction_index": 1, "score": 0.80, "phrase": "", "bbox": {"x": 3, "y": 3, "width": 10, "height": 8}},
            ],
            "profiles": {profile_id: {"candidate_count": 1, "iou_metrics": {"0.50": {"per_frame": [{"frame_id": "frame_0001", "tp": 1, "fp": fp, "fn": fn}]}}}},
        }
    return {"status": "completed", "profiles": [{"profile_id": profile_id, "score_threshold": 0.15, "class_agnostic_nms_iou": 0.30}], "streams": streams}


class ReviewPackTest(unittest.TestCase):
    def test_exact_stream_firewall_rejects_report_before_annotation_or_image_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inventory = _inventory(Path(temporary))
            report = _report(inventory)
            report["streams"].pop(inventory[-1].stream_id)
            with mock.patch.object(subject, "load_development_inventory", return_value=inventory), \
                 mock.patch.object(subject, "load_annotation") as annotations, \
                 mock.patch.object(subject.Image, "open") as images:
                with self.assertRaisesRegex(GateC1ContractError, "exactly the frozen"):
                    subject.prepare_review_pack(
                        report_path=_write_report(Path(temporary), report), benchmark_spec_path=Path("ignored.json"),
                        reviewed_root=Path("ignored"), profile_id="score_0.15_nms_0.30", output_dir=Path(temporary) / "out"
                    )
            annotations.assert_not_called()
            images.assert_not_called()

    def test_selection_is_deterministic_by_error_then_stream_then_frame(self) -> None:
        root = Path("C:/review-pack-fixture")
        inventory = (
            DevelopmentStream("stream_b", "scene", 1, root, root / "b.json"),
            DevelopmentStream("stream_a", "scene", 1, root, root / "a.json"),
        )
        rows = {"stream_a": [{"frame_id": "frame_0002", "errors": {"tp": 0, "fp": 1, "fn": 2}}, {"frame_id": "frame_0001", "errors": {"tp": 0, "fp": 1, "fn": 2}}], "stream_b": [{"frame_id": "frame_0001", "errors": {"tp": 0, "fp": 4, "fn": 0}}]}
        selected = subject._select_error_frames(inventory, rows)
        self.assertEqual([(row["stream"].stream_id, row["frame_id"]) for row in selected], [("stream_b", "frame_0001"), ("stream_a", "frame_0001")])

    def test_profile_reconstruction_uses_existing_score_filter_and_nms(self) -> None:
        raw = subject._raw_predictions([
            {"frame_id": "f", "frame_index": 0, "prediction_index": 0, "score": 0.90, "phrase": "object", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"frame_id": "f", "frame_index": 0, "prediction_index": 1, "score": 0.80, "phrase": "object", "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}},
            {"frame_id": "f", "frame_index": 0, "prediction_index": 2, "score": 0.10, "phrase": "object", "bbox": {"x": 30, "y": 1, "width": 5, "height": 5}},
        ], "stream")
        selected = subject.predictions_for_profile(raw, profile=subject.FilterProfile(0.15, 0.30))
        self.assertEqual([item.candidate_id for item in selected], ["grounding-dino:f:000"])

    def test_generated_index_is_canonical_and_contains_bbox_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _inventory(root / "data")
            report = _report(inventory)
            report_path = _write_report(root, report)
            annotation = mock.Mock()
            annotation.instances_by_frame = {"frame_0001": (AnnotationInstance("gt_1", "frame_0001", "visual_type_1", BBox(4, 5, 8, 7)),)}
            with mock.patch.object(subject, "load_development_inventory", return_value=inventory), \
                 mock.patch.object(subject, "load_annotation", return_value=annotation):
                payload = subject.prepare_review_pack(
                    report_path=report_path, benchmark_spec_path=root / "benchmark.json", reviewed_root=root / "data",
                    profile_id="score_0.15_nms_0.30", output_dir=root / "out"
                )
            index = json.loads((root / "out" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index, payload)
            self.assertEqual(len(index["frames"]), 10)
            self.assertEqual(index["profile"]["profile_id"], "score_0.15_nms_0.30")
            self.assertEqual(index["input_report"]["sha256"], hashlib.sha256(report_path.read_bytes()).hexdigest())
            self.assertTrue((root / "out" / "contact_sheet.png").is_file())
            self.assertEqual(len(index["frames"][0]["ground_truth_bboxes"]), 1)
            self.assertEqual(len(index["frames"][0]["predicted_bboxes"]), 1)


def _write_report(root: Path, report: dict) -> Path:
    path = root / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
