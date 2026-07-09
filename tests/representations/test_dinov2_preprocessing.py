import unittest

import numpy as np

from stream_analysis import ValidityStatus
from stream_analysis.representations import (
    DINO_BBOX_VARIANT,
    DINO_IMAGENET_MEAN,
    DINO_IMAGENET_STD,
    DINO_MASK_NEUTRAL_VARIANT,
    DINO_NEUTRAL_RGB,
    DinoV2PreprocessingError,
    model_bbox_with_context,
    preprocess_dinov2_candidate,
)

try:
    from .test_handcrafted import make_fixture
except ImportError:
    from test_handcrafted import make_fixture


class DinoV2PreprocessingTest(unittest.TestCase):
    def _parts(self, mask: np.ndarray, **kwargs):
        decoded, snapshot = make_fixture(mask, **kwargs)
        candidate = snapshot.result.candidates[0]
        frame = decoded.frames[0]
        mask_record = snapshot.mask_for_candidate(candidate.candidate_id)
        return frame, candidate, mask_record

    def test_letterbox_preserves_aspect_ratio_without_stretching(self) -> None:
        mask = np.ones((10, 20), dtype=bool)
        frame, candidate, _mask = self._parts(mask)

        result = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_BBOX_VARIANT,
            input_size=40,
        )

        self.assertEqual(
            dict(result.details["resized_content_size"]),
            {"width": 40, "height": 20},
        )
        self.assertEqual(dict(result.details["letterbox_offset"]), {"x": 0, "y": 10})
        self.assertEqual(result.letterboxed_rgb.shape, (40, 40, 3))
        self.assertEqual(result.normalized_chw.shape, (3, 40, 40))

    def test_imagenet_normalization_matches_reference(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        white = np.full((8, 8, 3), 255, dtype=np.uint8)
        frame, candidate, _mask = self._parts(mask, custom_crop=white)

        result = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_BBOX_VARIANT,
            input_size=8,
        )

        expected = np.array(
            [(1.0 - mean) / std for mean, std in zip(DINO_IMAGENET_MEAN, DINO_IMAGENET_STD)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result.normalized_chw[:, 0, 0], expected, atol=1e-6)
        self.assertEqual(result.normalized_chw.dtype, np.float32)

    def test_context_padding_is_clipped_to_frame_bounds(self) -> None:
        mask = np.ones((10, 12), dtype=bool)
        frame, candidate, _mask = self._parts(
            mask,
            bbox_x=0,
            bbox_y=1,
            frame_width=18,
            frame_height=16,
        )

        model_bbox = model_bbox_with_context(candidate, 0.5)
        result = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_BBOX_VARIANT,
            input_size=32,
            context_padding_ratio=0.5,
        )

        self.assertEqual(model_bbox.x, 0.0)
        self.assertEqual(model_bbox.y, 0.0)
        self.assertTrue(model_bbox.is_within(candidate.frame_size))
        self.assertEqual(result.model_bbox, model_bbox)
        self.assertEqual(dict(result.details["model_bbox"])["x"], 0.0)

    def test_mask_neutral_replaces_only_pixels_outside_mask(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        crop = np.empty((8, 8, 3), dtype=np.uint8)
        crop[:, :] = (10, 30, 220)
        crop[mask] = (230, 40, 20)
        frame, candidate, mask_record = self._parts(mask, custom_crop=crop)

        result = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_MASK_NEUTRAL_VARIANT,
            input_size=8,
            mask_record=mask_record,
        )

        np.testing.assert_array_equal(
            result.letterboxed_rgb[~mask],
            np.tile(DINO_NEUTRAL_RGB, (int(np.count_nonzero(~mask)), 1)),
        )
        np.testing.assert_array_equal(
            result.letterboxed_rgb[mask],
            np.tile((230, 40, 20), (int(np.count_nonzero(mask)), 1)),
        )
        self.assertEqual(result.details["mask_policy"], "mask_required")
        self.assertEqual(result.details["mask_digest"], mask_record.mask_digest)

    def test_invalid_required_mask_does_not_fallback_to_bbox(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        frame, candidate, mask_record = self._parts(
            mask,
            mask_validity_status=ValidityStatus.INVALID,
            mask_error_ids=("err_mask_invalid",),
        )

        with self.assertRaises(DinoV2PreprocessingError) as raised:
            preprocess_dinov2_candidate(
                frame,
                candidate,
                variant=DINO_MASK_NEUTRAL_VARIANT,
                input_size=8,
                mask_record=mask_record,
            )

        self.assertEqual(raised.exception.code, "MASK_REQUIRED")

    def test_preprocessing_is_deterministic_and_read_only(self) -> None:
        mask = np.zeros((12, 18), dtype=bool)
        mask[2:10, 4:14] = True
        frame, candidate, mask_record = self._parts(mask)
        before_mask = mask_record.mask.copy()

        first = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_MASK_NEUTRAL_VARIANT,
            input_size=32,
            mask_record=mask_record,
        )
        second = preprocess_dinov2_candidate(
            frame,
            candidate,
            variant=DINO_MASK_NEUTRAL_VARIANT,
            input_size=32,
            mask_record=mask_record,
        )

        np.testing.assert_array_equal(first.normalized_chw, second.normalized_chw)
        np.testing.assert_array_equal(first.letterboxed_rgb, second.letterboxed_rgb)
        np.testing.assert_array_equal(mask_record.mask, before_mask)
        with self.assertRaises(ValueError):
            first.normalized_chw[0, 0, 0] = 0.0


if __name__ == "__main__":
    unittest.main()
