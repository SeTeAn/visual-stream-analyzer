from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_ocid_streams.py"
SPEC = importlib.util.spec_from_file_location("prepare_ocid_streams", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load tool module from {TOOL_PATH}.")
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


class PrepareOcidStreamsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ocid_root = self.root / "OCID-dataset"
        self.sequence = self.ocid_root / "ARID10" / "table" / "top" / "box" / "seq05"
        (self.sequence / "rgb").mkdir(parents=True)
        (self.sequence / "label").mkdir()
        for name, color in (
            ("result_2020-01-01-00-00-03.png", (30, 0, 0)),
            ("result_2020-01-01-00-00-01.png", (10, 0, 0)),
            ("result_2020-01-01-00-00-02.png", (20, 0, 0)),
        ):
            Image.new("RGB", (8, 6), color).save(self.sequence / "rgb" / name)
            Image.new("I;16", (8, 6), 0).save(self.sequence / "label" / name)
        self.output_root = self.root / "prepared"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self):
        return TOOL.prepare_sequence(
            ocid_root=self.ocid_root,
            output_root=self.output_root,
            sequence_name="ARID10/table/top/box/seq05",
        )

    def test_prepares_sorted_rgb_only_stream(self) -> None:
        result = self.prepare()
        self.assertEqual(result.stream_id, "ocid_arid10_table_top_box_seq05")
        self.assertEqual(result.frame_count, 3)
        manifest = json.loads((result.stream_directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "stream-input-0.1")
        self.assertEqual([frame["index"] for frame in manifest["frames"]], [1, 2, 3])
        self.assertEqual(
            [frame["metadata"]["ocid_source_filename"] for frame in manifest["frames"]],
            [
                "result_2020-01-01-00-00-01.png",
                "result_2020-01-01-00-00-02.png",
                "result_2020-01-01-00-00-03.png",
            ],
        )
        self.assertFalse(any(result.stream_directory.rglob("*label*")))
        def keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from keys(nested)

        manifest_keys = {key.casefold() for key in keys(manifest)}
        self.assertFalse(any("ground_truth" in key for key in manifest_keys))
        self.assertFalse(any("annotation" in key for key in manifest_keys))

    def test_rejects_rgb_label_filename_mismatch(self) -> None:
        (self.sequence / "label" / "result_2020-01-01-00-00-03.png").unlink()
        with self.assertRaisesRegex(ValueError, "do not align"):
            self.prepare()

    def test_refuses_to_overwrite_prepared_stream(self) -> None:
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_rejects_sequence_outside_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the OCID root"):
            TOOL.prepare_sequence(
                ocid_root=self.ocid_root,
                output_root=self.output_root,
                sequence_name="../outside/seq01",
            )


if __name__ == "__main__":
    unittest.main()
