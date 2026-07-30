from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from stream_analysis.contracts import (
    ImageSize,
    ProductEvent,
    ProductEventKind,
    ProductFrame,
    ProductMask,
    ProductMatch,
    ProductObjectRef,
    ProductPipeline,
    ProductResult,
    ProductStatus,
    ProductStream,
)
from stream_analysis.reporting import ProductOutputCollisionError, write_product_output
from stream_analysis.reporting import product as product_reporting


_DIGEST = "a" * 64


def _result() -> ProductResult:
    first = ProductFrame(
        frame_id="frame_001",
        frame_index=0,
        source_image="frames/frame_001.png",
        image_size=ImageSize(4, 3),
    )
    second = ProductFrame(
        frame_id="frame_002",
        frame_index=1,
        source_image="frames/frame_002.png",
        image_size=ImageSize(4, 3),
        masks=(
            ProductMask(
                type_id="type_blue",
                mask=np.array(
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    dtype=np.uint8,
                ),
            ),
        ),
    )
    ref = ProductObjectRef(frame_id="frame_002", object_id="P00")
    return ProductResult(
        stream=ProductStream(
            stream_id="stream_001", input_schema_version="image-stream-1.0",
            manifest_sha256=_DIGEST, frame_count=2,
        ),
        pipeline=ProductPipeline(profile_id="profile_001", profile_sha256=_DIGEST),
        frames=(second, first), status=ProductStatus.COMPLETED,
        matches=(),
        events=(ProductEvent(
            event_id="event_001", kind=ProductEventKind.APPEARED, type_id="type_blue",
            from_frame_id=None, to_frame_id="frame_002", to_objects=(ref,),
        ),),
    )


class ProductOutputTest(unittest.TestCase):
    def test_writer_creates_exact_full_result_with_resolving_references(self) -> None:
        result = _result()
        images = {frame.frame_id: Image.new("RGB", (4, 3), "white") for frame in result.frames}
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "bundle"
            output = write_product_output(output_root, result, images)
            self.assertEqual(output.result_path, output_root / "result.json")
            self.assertTrue((output_root / "masks" / "frame_002" / "P00.png").is_file())
            self.assertTrue((output_root / "overlays" / "frame_001.png").is_file())
            payload = json.loads(output.result_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "schema_version",
                    "status",
                    "stream",
                    "pipeline",
                    "summary",
                    "frames",
                    "visual_types",
                    "matches",
                    "events",
                },
            )
            self.assertEqual(
                payload["summary"],
                {
                    "frame_count": 2,
                    "object_count": 1,
                    "visual_type_count": 1,
                    "match_count": 0,
                    "event_count": 1,
                },
            )
            frame = payload["frames"][1]
            self.assertEqual(frame["overlay"], "overlays/frame_002.png")
            self.assertEqual(frame["objects"][0], {
                "object_id": "P00",
                "type_id": "type_blue",
                "bbox": {"x": 1, "y": 1, "width": 2, "height": 1},
                "mask": "masks/frame_002/P00.png",
            })
            self.assertEqual(payload["pipeline"]["models"], ["Grounding DINO", "SAM2", "DINOv2"])
            self.assertEqual(
                payload["visual_types"],
                [
                    {
                        "type_id": "type_blue",
                        "objects": [{"frame_id": "frame_002", "object_id": "P00"}],
                    }
                ],
            )
            self.assertEqual(
                payload["events"][0]["to_objects"],
                [{"frame_id": "frame_002", "object_id": "P00"}],
            )
            text = output.result_path.read_text(encoding="utf-8")
            self.assertNotIn("confidence", text)
            self.assertNotIn("candidate", text)
            self.assertNotIn("score", text)
            with Image.open(output_root / frame["objects"][0]["mask"]) as saved_mask:
                self.assertEqual(set(np.asarray(saved_mask).ravel()), {0, 255})
            with self.assertRaises(ProductOutputCollisionError):
                write_product_output(output_root, result, images)

    def test_overlay_is_rendered_from_saved_mask_and_uses_type_label_only(self) -> None:
        result = _result()
        images = {frame.frame_id: Image.new("RGB", (4, 3), "white") for frame in result.frames}
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "bundle"
            write_product_output(output_root, result, images)
            with Image.open(output_root / "overlays" / "frame_002.png") as overlay:
                self.assertNotEqual(overlay.getpixel((1, 1)), (255, 255, 255))
            self.assertFalse(
                any("P00" in path.name for path in (output_root / "overlays").iterdir())
            )

    def test_writer_rejects_missing_or_wrong_sized_source_images_before_publication(self) -> None:
        result = _result()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "bundle"
            with self.assertRaisesRegex(ValueError, "exactly resolve"):
                write_product_output(output_root, result, {})
            self.assertFalse(output_root.exists())
            images = {frame.frame_id: Image.new("RGB", (1, 1), "white") for frame in result.frames}
            with self.assertRaisesRegex(ValueError, "dimensions"):
                write_product_output(output_root, result, images)
            self.assertFalse(output_root.exists())

    def test_writer_does_not_replace_a_destination_created_during_staging(self) -> None:
        result = _result()
        images = {frame.frame_id: Image.new("RGB", (4, 3), "white") for frame in result.frames}
        original = product_reporting._write_bundle
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "bundle"

            def create_collision(staging_root, value, sources):
                staged = original(staging_root, value, sources)
                output_root.mkdir()
                return staged

            with patch.object(product_reporting, "_write_bundle", side_effect=create_collision):
                with self.assertRaises(ProductOutputCollisionError):
                    write_product_output(output_root, result, images)
            self.assertTrue(output_root.is_dir())
            self.assertEqual(tuple(output_root.iterdir()), ())
            self.assertFalse(any(Path(directory).glob(".bundle.staging-*")))

    def test_writer_cleans_staging_after_interruption(self) -> None:
        result = _result()
        images = {frame.frame_id: Image.new("RGB", (4, 3), "white") for frame in result.frames}
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "bundle"
            with patch.object(product_reporting, "_write_bundle", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    write_product_output(output_root, result, images)
            self.assertFalse(output_root.exists())
            self.assertFalse(any(Path(directory).glob(".bundle.staging-*")))


if __name__ == "__main__":
    unittest.main()
