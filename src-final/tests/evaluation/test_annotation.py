from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stream_analysis.evaluation import AnnotationFormatError, load_annotation


class AnnotationLoaderTest(unittest.TestCase):
    def test_loads_probe_annotation_strictly(self) -> None:
        root = Path("src-final/data/streams/probe_01_stationery")
        annotation = load_annotation(root / "annotation.json", manifest_path=root / "manifest.json")
        self.assertEqual(annotation.stream_id, "probe_01_stationery")
        self.assertEqual(len(annotation.frame_ids), 10)
        self.assertEqual(len(annotation.visual_type_ids), 7)
        self.assertGreater(len(annotation.instances), 0)
        self.assertIn("persisted", annotation.supported_event_types)

    def test_rejects_unknown_event_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "stream_id": "s",
                "frames": [{"frame_id": "f1", "index": 1, "image_path": "a.png"}],
                "metadata": {"frame_size": {"width": 10, "height": 10}},
            }
            annotation = {
                "schema_version": "x",
                "stream_id": "s",
                "manifest_ref": "manifest.json",
                "annotation_scope": "pilot_development",
                "visual_types": [{"visual_type_id": "t"}],
                "expected_element_instances": [],
                "frame_comparisons": [],
                "change_events": [
                    {"event_id": "e", "event_type": "rotated", "visual_type_id": "t", "from_frame_id": "f1", "to_frame_id": "f1"}
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
            with self.assertRaises(AnnotationFormatError):
                load_annotation(root / "annotation.json", manifest_path=root / "manifest.json")

    def test_rejects_non_neighboring_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_three_frame_annotation(
                root,
                change_events=[
                    {
                        "event_id": "e",
                        "event_type": "persisted",
                        "visual_type_id": "t",
                        "from_frame_id": "f1",
                        "to_frame_id": "f3",
                    }
                ],
                frame_comparisons=[],
            )

            with self.assertRaisesRegex(AnnotationFormatError, "neighboring"):
                load_annotation(root / "annotation.json", manifest_path=root / "manifest.json")

    def test_rejects_non_neighboring_frame_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_three_frame_annotation(
                root,
                change_events=[
                    {
                        "event_id": "e",
                        "event_type": "persisted",
                        "visual_type_id": "t",
                        "from_frame_id": "f1",
                        "to_frame_id": "f2",
                    }
                ],
                frame_comparisons=[
                    {"from_frame_id": "f1", "to_frame_id": "f3", "expected_change_event_ids": ["e"]}
                ],
            )

            with self.assertRaisesRegex(AnnotationFormatError, "neighboring"):
                load_annotation(root / "annotation.json", manifest_path=root / "manifest.json")


def _write_three_frame_annotation(
    root: Path,
    *,
    change_events: list[dict],
    frame_comparisons: list[dict],
) -> None:
    manifest = {
        "stream_id": "s",
        "frames": [
            {"frame_id": "f1", "index": 1, "image_path": "f1.png"},
            {"frame_id": "f2", "index": 2, "image_path": "f2.png"},
            {"frame_id": "f3", "index": 3, "image_path": "f3.png"},
        ],
        "metadata": {"frame_size": {"width": 10, "height": 10}},
    }
    annotation = {
        "schema_version": "x",
        "stream_id": "s",
        "manifest_ref": "manifest.json",
        "annotation_scope": "pilot_development",
        "visual_types": [{"visual_type_id": "t"}],
        "expected_element_instances": [],
        "frame_comparisons": frame_comparisons,
        "change_events": change_events,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "annotation.json").write_text(json.dumps(annotation), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
