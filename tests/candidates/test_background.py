from __future__ import annotations

import unittest

import numpy as np

import stream_analysis
import stream_analysis.candidates as candidates
from stream_analysis.candidates import (
    BackgroundModelConfig,
    HysteresisMaskConfig,
    MorphologyCleanupConfig,
    ResidualMaskConfig,
    cleanup_morphology,
    compute_residual,
    estimate_background_model,
    hysteresis_threshold_mask,
    threshold_residual_mask,
)


def _smooth_background(height: int = 8, width: int = 9) -> np.ndarray:
    y = np.arange(height, dtype=np.uint8).reshape(height, 1)
    x = np.arange(width, dtype=np.uint8).reshape(1, width)
    red = 20 + y * 2
    green = 40 + x * 3
    blue = np.full((height, width), 70, dtype=np.uint8)
    return np.stack(
        [
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            blue,
        ],
        axis=2,
    ).astype(np.uint8)


class BackgroundPrimitiveTest(unittest.TestCase):
    def test_background_estimation_uses_multiple_same_size_rgb_frames(self) -> None:
        base = _smooth_background()
        frames = [base.copy() for _ in range(3)]
        frames[0][2:4, 3:5] = (240, 10, 10)

        model = estimate_background_model(
            frames,
            BackgroundModelConfig(aggregation="median"),
        )

        self.assertEqual(model.frame_count, 3)
        self.assertEqual(model.rgb.shape, base.shape)
        self.assertEqual(model.rgb.dtype, np.float32)
        self.assertFalse(model.rgb.flags.writeable)
        np.testing.assert_allclose(model.rgb, base.astype(np.float32), atol=0.0)
        self.assertEqual(model.lab.shape, base.shape)
        self.assertEqual(model.lab.dtype, np.float32)

    def test_residual_mask_distinguishes_foreground_from_smooth_background(self) -> None:
        base = _smooth_background()
        model = estimate_background_model(
            [base, base.copy(), base.copy()],
            BackgroundModelConfig(aggregation="median"),
        )
        frame = base.copy()
        frame[2:5, 3:6] = (230, 20, 20)

        residual = compute_residual(frame, model)
        mask = threshold_residual_mask(residual, ResidualMaskConfig(threshold=15.0))

        self.assertEqual(residual.shape, base.shape[:2])
        self.assertEqual(residual.dtype, np.float32)
        self.assertFalse(residual.flags.writeable)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, base.shape[:2])
        self.assertTrue(mask[3, 4])
        self.assertFalse(mask[0, 0])
        self.assertTrue(set(np.unique(mask).tolist()).issubset({False, True}))

    def test_hysteresis_keeps_weak_regions_connected_to_strong_regions(self) -> None:
        residual = np.zeros((6, 7), dtype=np.float32)
        residual[2, 2] = 10.0
        residual[2, 3] = 5.0
        residual[2, 4] = 5.0
        residual[4, 5] = 5.0

        mask = hysteresis_threshold_mask(
            residual,
            HysteresisMaskConfig(weak_threshold=4.0, strong_threshold=8.0, connectivity=8),
        )

        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[2, 3])
        self.assertTrue(mask[2, 4])
        self.assertFalse(mask[4, 5])
        self.assertEqual(mask.dtype, np.bool_)

    def test_hysteresis_discards_isolated_weak_noise(self) -> None:
        residual = np.zeros((5, 5), dtype=np.float32)
        residual[1, 1] = 4.5
        residual[3, 3] = 7.0

        mask = hysteresis_threshold_mask(
            residual,
            HysteresisMaskConfig(weak_threshold=4.0, strong_threshold=8.0, connectivity=8),
        )

        self.assertFalse(mask.any())
        self.assertEqual(mask.dtype, np.bool_)

    def test_morphology_cleanup_is_deterministic_for_noise_and_holes(self) -> None:
        noisy = np.zeros((7, 7), dtype=np.bool_)
        noisy[2:5, 2:5] = True
        noisy[0, 0] = True
        first = cleanup_morphology(
            noisy,
            MorphologyCleanupConfig(open_kernel_size=3, kernel_shape="rect"),
        )
        second = cleanup_morphology(
            noisy,
            MorphologyCleanupConfig(open_kernel_size=3, kernel_shape="rect"),
        )

        np.testing.assert_array_equal(first, second)
        self.assertFalse(first[0, 0])
        self.assertTrue(first[3, 3])
        self.assertEqual(first.dtype, np.bool_)

        with_hole = np.zeros((7, 7), dtype=np.bool_)
        with_hole[1:6, 1:6] = True
        with_hole[3, 3] = False
        filled = cleanup_morphology(
            with_hole,
            MorphologyCleanupConfig(fill_holes=True),
        )
        self.assertTrue(filled[3, 3])
        self.assertFalse(filled[0, 0])

    def test_invalid_shapes_dtypes_thresholds_and_kernel_sizes_are_rejected(self) -> None:
        valid = _smooth_background()
        with self.assertRaises(ValueError):
            estimate_background_model([], BackgroundModelConfig())
        with self.assertRaises(ValueError):
            estimate_background_model([np.zeros((4, 4), dtype=np.uint8)], BackgroundModelConfig())
        with self.assertRaises(ValueError):
            estimate_background_model(
                [valid, np.zeros((valid.shape[0] + 1, valid.shape[1], 3), dtype=np.uint8)],
                BackgroundModelConfig(),
            )
        with self.assertRaises(TypeError):
            estimate_background_model([valid.astype(np.int16)], BackgroundModelConfig())
        float_frame = valid.astype(np.float32)
        float_frame[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            estimate_background_model([float_frame], BackgroundModelConfig())

        with self.assertRaises(ValueError):
            ResidualMaskConfig(threshold=-1.0)
        with self.assertRaises(ValueError):
            HysteresisMaskConfig(weak_threshold=9.0, strong_threshold=8.0)
        with self.assertRaises(ValueError):
            HysteresisMaskConfig(weak_threshold=1.0, strong_threshold=2.0, connectivity=6)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MorphologyCleanupConfig(open_kernel_size=2)
        with self.assertRaises(ValueError):
            MorphologyCleanupConfig(close_kernel_size=-1)
        with self.assertRaises(TypeError):
            cleanup_morphology(np.zeros((3, 3), dtype=np.uint8), MorphologyCleanupConfig())
        with self.assertRaises(TypeError):
            threshold_residual_mask(
                np.zeros((3, 3), dtype=np.uint8),
                ResidualMaskConfig(threshold=1.0),
            )

    def test_primitives_do_not_mutate_input_arrays(self) -> None:
        base = _smooth_background()
        frames = [base.copy() for _ in range(3)]
        frames_before = [frame.copy() for frame in frames]
        model = estimate_background_model(frames, BackgroundModelConfig())
        for actual, expected in zip(frames, frames_before, strict=True):
            np.testing.assert_array_equal(actual, expected)

        frame = base.copy()
        frame[1:3, 1:3] = (240, 30, 30)
        frame_before = frame.copy()
        residual = compute_residual(frame, model)
        np.testing.assert_array_equal(frame, frame_before)

        residual_before = residual.copy()
        _ = threshold_residual_mask(residual, ResidualMaskConfig(threshold=12.0))
        _ = hysteresis_threshold_mask(
            residual,
            HysteresisMaskConfig(weak_threshold=8.0, strong_threshold=12.0),
        )
        np.testing.assert_array_equal(residual, residual_before)

        mask = np.zeros((6, 6), dtype=np.bool_)
        mask[1:5, 1:5] = True
        mask_before = mask.copy()
        _ = cleanup_morphology(
            mask,
            MorphologyCleanupConfig(close_kernel_size=3, kernel_shape="rect"),
        )
        np.testing.assert_array_equal(mask, mask_before)

    def test_candidate_package_exports_are_available(self) -> None:
        public_names = (
            "BackgroundModelConfig",
            "ResidualMaskConfig",
            "HysteresisMaskConfig",
            "MorphologyCleanupConfig",
            "estimate_background_model",
            "compute_residual",
            "threshold_residual_mask",
            "hysteresis_threshold_mask",
            "cleanup_morphology",
        )
        for name in public_names:
            with self.subTest(name=name):
                self.assertIn(name, candidates.__all__)
                self.assertTrue(hasattr(candidates, name))
                self.assertIn(name, stream_analysis.__all__)
                self.assertTrue(hasattr(stream_analysis, name))


if __name__ == "__main__":
    unittest.main()
