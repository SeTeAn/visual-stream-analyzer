from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from stream_analysis import BBox, ImageSize, Point


class GeometryContractsTest(unittest.TestCase):
    def test_half_open_bbox_properties(self) -> None:
        bbox = BBox(10, 20, 30, 40)

        self.assertEqual((bbox.left, bbox.top, bbox.right, bbox.bottom), (10, 20, 40, 60))
        self.assertEqual(bbox.area, 1200)
        self.assertEqual(bbox.center, Point(25, 40))
        self.assertTrue(bbox.contains_point(Point(10, 20)))
        self.assertTrue(bbox.contains_point(Point(39.999, 59.999)))
        self.assertFalse(bbox.contains_point(Point(40, 20)))
        self.assertFalse(bbox.contains_point(Point(10, 60)))

    def test_point_requires_finite_coordinates(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Point(value, 0)

    def test_bbox_rejects_non_finite_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BBox(value, 0, 1, 1)
            with self.subTest(width=value), self.assertRaises(ValueError):
                BBox(0, 0, value, 1)

    def test_bbox_rejects_zero_and_negative_sizes(self) -> None:
        for width, height in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                BBox(0, 0, width, height)

    def test_bbox_rejects_overflowed_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "right and bottom"):
            BBox(1.7e308, 0, 1.7e308, 1)

    def test_image_size_is_positive_integer(self) -> None:
        self.assertEqual(ImageSize(640, 480).area, 307200)
        for width, height in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                ImageSize(width, height)
        with self.assertRaises(TypeError):
            ImageSize(10.5, 20)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ImageSize(True, 20)  # type: ignore[arg-type]

    def test_bbox_within_frame_bounds(self) -> None:
        size = ImageSize(100, 80)
        self.assertTrue(BBox(0, 0, 100, 80).is_within(size))
        self.assertTrue(BBox(10, 10, 20, 20).is_within(size))
        self.assertFalse(BBox(-1, 0, 10, 10).is_within(size))
        self.assertFalse(BBox(90, 70, 11, 10).is_within(size))

    def test_intersection_uses_half_open_boundaries(self) -> None:
        left = BBox(0, 0, 10, 10)
        self.assertEqual(left.intersection(BBox(5, 5, 10, 10)), BBox(5, 5, 5, 5))
        self.assertIsNone(left.intersection(BBox(10, 0, 2, 2)))
        self.assertIsNone(left.intersection(BBox(0, 10, 2, 2)))

    def test_clipping_returns_valid_intersection(self) -> None:
        size = ImageSize(100, 80)
        self.assertEqual(BBox(-10, 5, 30, 20).clip_to(size), BBox(0, 5, 20, 20))
        self.assertEqual(BBox(90, 70, 20, 20).clip_to(size), BBox(90, 70, 10, 10))

    def test_clipping_non_intersecting_or_boundary_box_returns_none(self) -> None:
        size = ImageSize(100, 80)
        self.assertIsNone(BBox(100, 0, 10, 10).clip_to(size))
        self.assertIsNone(BBox(-10, 0, 10, 10).clip_to(size))
        self.assertIsNone(BBox(0, 80, 10, 10).clip_to(size))

    def test_geometry_contracts_are_frozen(self) -> None:
        bbox = BBox(0, 0, 10, 10)
        with self.assertRaises(FrozenInstanceError):
            bbox.width = 20  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
