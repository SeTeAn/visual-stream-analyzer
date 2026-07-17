from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stream_analysis.evaluation import load_annotation


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "prepare_ocid_candidate_annotations.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_ocid_candidate_annotations", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load tool module from {TOOL_PATH}.")
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


class PrepareOcidCandidateAnnotationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ocid_root = self.root / "OCID-dataset"
        self.sequence_name = "ARID10/table/top/box/seq05"
        self.sequence = self.ocid_root.joinpath(*self.sequence_name.split("/"))
        (self.sequence / "label").mkdir(parents=True)
        self.stream = self.root / "stream"
        self.stream.mkdir()
        self.output = self.root / "annotations" / "annotation.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, filenames: list[str]) -> None:
        payload = {
            "schema_version": "stream-input-0.1",
            "stream_id": "ocid_test_stream",
            "ordering": "manifest",
            "frames": [
                {
                    "frame_id": f"frame_{index:03d}",
                    "index": index,
                    "image_path": f"frames/frame_{index:03d}.png",
                    "metadata": {"ocid_source_filename": filename},
                }
                for index, filename in enumerate(filenames, start=1)
            ],
            "metadata": {
                "source_sequence": self.sequence_name,
                "frame_size": {"width": 8, "height": 6},
            },
        }
        (self.stream / "manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def write_label(self, name: str, pixels: list[int]) -> None:
        image = Image.new("I;16", (8, 6), 0)
        image.putdata(pixels)
        image.save(self.sequence / "label" / name)

    def prepare(self):
        return TOOL.prepare_candidate_annotation(
            ocid_root=self.ocid_root,
            stream_directory=self.stream,
            output_path=self.output,
        )

    def test_builds_tight_table_object_bboxes_and_loadable_annotation(self) -> None:
        filenames = ["state_01.png", "state_02.png"]
        self.write_manifest(filenames)
        first = [2] * 48
        second = [2] * 48
        for y in range(1, 3):
            for x in range(2, 5):
                first[y * 8 + x] = 3
                second[y * 8 + x] = 3
        second[4 * 8 + 6] = 4
        self.write_label(filenames[0], first)
        self.write_label(filenames[1], second)

        result = self.prepare()

        self.assertEqual(result.instance_count, 3)
        self.assertEqual(result.physical_instance_count, 2)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["expected_element_instances"][0]["bbox"],
            {"x": 2, "y": 1, "width": 3, "height": 2},
        )
        self.assertFalse(
            any(item["visual_type_id"].endswith("002") for item in payload["visual_types"])
        )
        annotation = load_annotation(
            self.output, manifest_path=self.stream / "manifest.json"
        )
        self.assertEqual(annotation.annotation_scope, "candidate_extraction_bbox_only")
        self.assertEqual(annotation.supported_event_types, ())

    def test_floor_scene_treats_label_two_as_an_object(self) -> None:
        self.sequence_name = "ARID10/floor/top/curved/seq01"
        self.sequence = self.ocid_root.joinpath(*self.sequence_name.split("/"))
        (self.sequence / "label").mkdir(parents=True)
        self.write_manifest(["state.png"])
        pixels = [0] * 48
        pixels[2 * 8 + 3] = 2
        self.write_label("state.png", pixels)

        result = self.prepare()

        self.assertEqual(result.instance_count, 1)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["object_label_minimum"], 2)

    def test_rejects_wrong_label_dimensions(self) -> None:
        self.write_manifest(["state.png"])
        Image.new("I;16", (7, 6), 3).save(self.sequence / "label" / "state.png")
        with self.assertRaisesRegex(ValueError, "must have shape"):
            self.prepare()

    def test_refuses_overwrite_and_unsafe_source_filename(self) -> None:
        self.write_manifest(["state.png"])
        self.write_label("state.png", [2] * 48)
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

        self.output.unlink()
        self.write_manifest(["../state.png"])
        with self.assertRaisesRegex(ValueError, "filename, not a path"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
