import math
import unittest
from dataclasses import replace

import numpy as np

from stream_analysis import HandcraftedFeatureGroup, HandcraftedPayload
from stream_analysis.representations import (
    HANDCRAFTED_BBOX_VARIANT,
    HANDCRAFTED_MASK_VARIANT,
    HandcraftedRepresentationConfig,
    handcrafted_distance,
)

try:
    from .test_handcrafted import build_record, groups
except ImportError:  # direct discovery with tests/representations as top-level
    from test_handcrafted import build_record, groups


def _replace_group(record, group_name: str, replacement: HandcraftedFeatureGroup):
    payload = record.payload
    new_groups = tuple(
        replacement if group.group_name == group_name else group
        for group in payload.feature_groups
    )
    return replace(
        record,
        payload=HandcraftedPayload(
            feature_schema_id=payload.feature_schema_id,
            feature_groups=new_groups,
        ),
    )


class HandcraftedDistanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((24, 30), dtype=bool)
        self.mask[4:20, 6:24] = True

    def test_distance_is_symmetric_bounded_and_identity_is_zero(self) -> None:
        config = HandcraftedRepresentationConfig()
        left = build_record(self.mask, config=config, foreground_color=(220, 40, 30))[0]
        right = build_record(self.mask, config=config, foreground_color=(30, 170, 210))[0]
        identity = handcrafted_distance(left, left, config)
        forward = handcrafted_distance(left, right, config)
        reverse = handcrafted_distance(right, left, config)
        self.assertEqual(identity.distance, 0.0)
        self.assertEqual(identity.similarity, 1.0)
        self.assertAlmostEqual(forward.distance, reverse.distance, places=15)
        self.assertAlmostEqual(forward.similarity, 1.0 - forward.distance, places=15)
        self.assertGreaterEqual(forward.distance, 0.0)
        self.assertLessEqual(forward.distance, 1.0)
        for value in forward.group_distances.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_position_does_not_influence_visual_distance(self) -> None:
        config = HandcraftedRepresentationConfig()
        left = build_record(
            self.mask,
            config=config,
            bbox_x=2,
            bbox_y=3,
            frame_width=80,
            frame_height=70,
        )[0]
        right = build_record(
            self.mask,
            config=config,
            bbox_x=41,
            bbox_y=35,
            frame_width=80,
            frame_height=70,
        )[0]
        result = handcrafted_distance(left, right, config)
        self.assertEqual(result.distance, 0.0)
        self.assertNotEqual(left.preprocessing_metadata.details["crop_bbox"], right.preprocessing_metadata.details["crop_bbox"])

    def test_color_variation_remains_soft_for_both_variants(self) -> None:
        for variant in (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT):
            with self.subTest(variant=variant):
                config = HandcraftedRepresentationConfig(variant=variant)
                left = build_record(
                    self.mask,
                    config=config,
                    foreground_color=(220, 45, 35),
                )[0]
                right = build_record(
                    self.mask,
                    config=config,
                    foreground_color=(35, 160, 210),
                )[0]
                result = handcrafted_distance(left, right, config)

                self.assertEqual(
                    set(result.group_distances),
                    {"shape", "structure", "color", "size"},
                )
                self.assertEqual(result.group_distances["shape"], 0.0)
                self.assertEqual(result.group_distances["size"], 0.0)
                self.assertLess(result.group_distances["structure"], 0.05)
                self.assertLess(result.distance, 0.20)

    def test_rotation_and_scale_variation_remain_close_for_mask_variant(self) -> None:
        config = HandcraftedRepresentationConfig(variant=HANDCRAFTED_MASK_VARIANT)
        base = np.zeros((28, 36), dtype=bool)
        base[7:21, 8:28] = True
        rotated = np.rot90(base)
        scaled = np.kron(base, np.ones((2, 2), dtype=bool))
        base_record = build_record(base, config=config)[0]
        rotated_record = build_record(rotated, config=config)[0]
        scaled_record = build_record(
            scaled,
            config=config,
            frame_width=96,
            frame_height=96,
        )[0]

        self.assertLess(handcrafted_distance(base_record, rotated_record, config).distance, 0.12)
        self.assertLess(handcrafted_distance(base_record, scaled_record, config).distance, 0.20)

    def test_shape_and_structure_difference_exceeds_color_only_difference(self) -> None:
        disk_y, disk_x = np.ogrid[:40, :40]
        disk = (disk_x - 19.5) ** 2 + (disk_y - 19.5) ** 2 <= 13 ** 2
        rectangle = np.zeros((40, 40), dtype=bool)
        rectangle[10:30, 5:35] = True
        for variant in (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT):
            with self.subTest(variant=variant):
                config = HandcraftedRepresentationConfig(variant=variant)
                base = build_record(
                    rectangle,
                    config=config,
                    foreground_color=(210, 140, 80),
                )[0]
                recolored = build_record(
                    rectangle,
                    config=config,
                    foreground_color=(90, 190, 210),
                )[0]
                different_geometry = build_record(
                    disk,
                    config=config,
                    foreground_color=(210, 140, 80),
                )[0]

                color_only = handcrafted_distance(base, recolored, config)
                geometry = handcrafted_distance(base, different_geometry, config)
                self.assertGreater(
                    geometry.distance,
                    color_only.distance,
                )

    def test_size_has_soft_capped_contribution(self) -> None:
        config = HandcraftedRepresentationConfig()
        record = build_record(self.mask, config=config)[0]
        size_group = groups(record)["size"]
        name = "hc_area_ratio_to_frame"
        changed_size = min(size_group.values[name] * config.size_ratio_cap, 0.99)
        changed = _replace_group(
            record,
            "size",
            HandcraftedFeatureGroup(group_name="size", values={name: changed_size}),
        )
        result = handcrafted_distance(record, changed, config)
        self.assertLessEqual(result.distance, config.group_weights["size"] + 1e-12)
        self.assertEqual(set(result.group_distances), {"shape", "structure", "color", "size"})

    def test_size_distance_saturates_at_cap_for_both_variants(self) -> None:
        cases = (
            (HANDCRAFTED_MASK_VARIANT, "hc_area_ratio_to_frame"),
            (HANDCRAFTED_BBOX_VARIANT, "hcb_bbox_area_ratio_to_frame"),
        )
        for variant, name in cases:
            config = HandcraftedRepresentationConfig(variant=variant)
            record = build_record(self.mask, config=config)[0]
            base_value = 0.1
            base = _replace_group(
                record,
                "size",
                HandcraftedFeatureGroup(group_name="size", values={name: base_value}),
            )
            identity = handcrafted_distance(base, base, config)
            self.assertEqual(identity.group_distances["size"], 0.0)
            self.assertEqual(identity.distance, 0.0)

            for ratio, expected in (
                (config.size_ratio_cap / 2.0, math.log(config.size_ratio_cap / 2.0) / math.log(config.size_ratio_cap)),
                (config.size_ratio_cap, 1.0),
                (config.size_ratio_cap * 2.0, 1.0),
            ):
                with self.subTest(variant=variant, ratio=ratio):
                    compared_value = ratio * (base_value + config.epsilon) - config.epsilon
                    compared = _replace_group(
                        record,
                        "size",
                        HandcraftedFeatureGroup(
                            group_name="size",
                            values={name: compared_value},
                        ),
                    )
                    forward = handcrafted_distance(base, compared, config)
                    reverse = handcrafted_distance(compared, base, config)
                    self.assertAlmostEqual(
                        forward.group_distances["size"],
                        expected,
                        places=14,
                    )
                    self.assertAlmostEqual(
                        reverse.group_distances["size"],
                        expected,
                        places=14,
                    )
                    self.assertAlmostEqual(forward.distance, reverse.distance, places=15)

    def test_incompatible_variants_and_configs_are_rejected(self) -> None:
        mask_config = HandcraftedRepresentationConfig()
        bbox_config = HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT)
        mask_record = build_record(self.mask, config=mask_config)[0]
        bbox_record = build_record(self.mask, config=bbox_config)[0]
        with self.assertRaisesRegex(ValueError, "input_variant"):
            handcrafted_distance(mask_record, bbox_record, mask_config)

        changed_config = HandcraftedRepresentationConfig(canny_low_threshold=40.0)
        with self.assertRaisesRegex(ValueError, "digest"):
            handcrafted_distance(mask_record, mask_record, changed_config)

    def test_mask_invalid_structure_group_is_omitted_and_weights_renormalize(self) -> None:
        config = HandcraftedRepresentationConfig()
        record = build_record(self.mask, config=config)[0]
        invalid = _replace_group(
            record,
            "structure",
            HandcraftedFeatureGroup(
                group_name="structure",
                values={},
                valid=False,
                invalid_reason="empty_edge_set",
            ),
        )
        result = handcrafted_distance(record, invalid, config)
        self.assertEqual(result.omitted_groups, ("structure",))
        self.assertNotIn("structure", result.group_distances)
        self.assertEqual(result.distance, 0.0)

    def test_bbox_empty_edge_profile_policy_is_explicit(self) -> None:
        config = HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT)
        uniform = np.empty((*self.mask.shape, 3), dtype=np.uint8)
        uniform[:, :] = (80, 80, 80)
        empty_left = build_record(
            np.ones_like(self.mask),
            config=config,
            custom_crop=uniform,
        )[0]
        empty_right = build_record(
            np.ones_like(self.mask),
            config=config,
            custom_crop=uniform,
        )[0]
        self.assertFalse(empty_left.preprocessing_metadata.details["edge_profile_valid"])
        self.assertEqual(
            handcrafted_distance(empty_left, empty_right, config).group_distances["structure"],
            0.0,
        )

        split = uniform.copy()
        split[:, self.mask.shape[1] // 2:] = (230, 230, 230)
        edged = build_record(
            np.ones_like(self.mask),
            config=config,
            custom_crop=split,
        )[0]
        self.assertTrue(edged.preprocessing_metadata.details["edge_profile_valid"])
        structure = handcrafted_distance(empty_left, edged, config).group_distances["structure"]
        self.assertGreaterEqual(structure, config.structure_component_weights[1])

    def test_bbox_distance_uses_separate_schema_and_formula(self) -> None:
        config = HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT)
        left = build_record(self.mask, config=config, foreground_color=(220, 40, 30))[0]
        right = build_record(np.rot90(self.mask), config=config, foreground_color=(30, 170, 210))[0]
        result = handcrafted_distance(left, right, config)
        self.assertEqual(result.variant, HANDCRAFTED_BBOX_VARIANT)
        self.assertEqual(set(result.group_distances), {"shape", "structure", "color", "size"})
        self.assertEqual(result.omitted_groups, ())

    def test_distance_has_no_threshold_or_type_assignment_output(self) -> None:
        config = HandcraftedRepresentationConfig(variant=HANDCRAFTED_MASK_VARIANT)
        record = build_record(self.mask, config=config)[0]
        result = handcrafted_distance(record, record, config)
        self.assertFalse(hasattr(result, "threshold"))
        self.assertFalse(hasattr(result, "visual_type_id"))
        self.assertFalse(hasattr(result, "position"))


if __name__ == "__main__":
    unittest.main()
