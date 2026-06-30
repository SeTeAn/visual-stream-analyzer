from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image, ImageChops

from stream_analysis import BBox, ImageSize
from stream_analysis.reporting import render_primary_overlays


class _Frame:
    frame_id = "frame_001"
    image_size = ImageSize(8, 6)

    def to_pillow_image(self):
        return Image.new("RGB", (8, 6), "white")


class _LargeFrame:
    frame_id = "frame_002"
    image_size = ImageSize(80, 60)

    def to_pillow_image(self):
        return Image.new("RGB", (80, 60), "white")


class OverlayTest(unittest.TestCase):
    def test_border_bbox_is_clipped_and_source_image_is_not_shared(self) -> None:
        candidate = SimpleNamespace(
            candidate_id="candidate_001", frame_id="frame_001",
            bbox=BBox(x=0, y=0, width=8, height=6),
        )
        grouping = SimpleNamespace(assignments=(SimpleNamespace(
            candidate_id="candidate_001", primary_type_id="type_001",
        ),))
        decoded = SimpleNamespace(frames=(_Frame(),))
        overlays = render_primary_overlays(decoded, (candidate,), grouping)
        image = overlays["frame_001"]
        self.assertEqual(image.size, (8, 6))
        self.assertNotEqual(image.getpixel((7, 5)), (255, 255, 255))
        image.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(_Frame().to_pillow_image().getpixel((0, 0)), (255, 255, 255))

    def test_events_do_not_render_upper_left_text_block_but_labels_remain(self) -> None:
        candidate = SimpleNamespace(
            candidate_id="candidate_001", frame_id="frame_002",
            bbox=BBox(x=30, y=20, width=20, height=16),
        )
        grouping = SimpleNamespace(assignments=(SimpleNamespace(
            candidate_id="candidate_001", primary_type_id="type_001",
        ),))
        decoded = SimpleNamespace(frames=(_LargeFrame(),))
        event = SimpleNamespace(
            event_id="event_001",
            predicted_type_id="type_001",
            frame_pair=SimpleNamespace(to_frame_id="frame_002"),
            kind=SimpleNamespace(value="appeared"),
            status=SimpleNamespace(value="certain"),
        )
        without_events = render_primary_overlays(decoded, (candidate,), grouping)["frame_002"]
        with_events = render_primary_overlays(decoded, (candidate,), grouping, (event,))["frame_002"]
        self.assertIsNone(ImageChops.difference(without_events, with_events).getbbox())
        self.assertEqual(with_events.getpixel((4, 4)), (255, 255, 255))
        self.assertNotEqual(with_events.getpixel((30, 20)), (255, 255, 255))
        label_area_above_bbox = with_events.crop((28, 0, 70, 20))
        self.assertIsNotNone(
            ImageChops.difference(
                label_area_above_bbox,
                Image.new("RGB", label_area_above_bbox.size, "white"),
            ).getbbox()
        )
        bbox_inner = with_events.crop((32, 22, 48, 34))
        self.assertIsNone(
            ImageChops.difference(
                bbox_inner,
                Image.new("RGB", bbox_inner.size, "white"),
            ).getbbox()
        )


if __name__ == "__main__":
    unittest.main()
