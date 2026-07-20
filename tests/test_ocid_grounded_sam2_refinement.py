from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from stream_analysis.contracts import BBox
from tools.ocid_grounded_sam2_refinement import (
    MaskCleanupConfig,
    SAM2BBoxRefiner,
    clean_mask_components,
    refine_grounding_boxes,
)


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    def inference_mode(self):
        return _InferenceMode()


class _Batch(dict):
    def to(self, device):
        self["moved_to"] = device
        return self


class _FakeProcessor:
    def __init__(self, processed_masks):
        self.processed_masks = processed_masks
        self.calls = []
        self.post_calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _Batch(original_sizes=np.asarray([[6, 8]], dtype=np.int64))

    def post_process_masks(self, masks, original_sizes, **kwargs):
        self.post_calls.append((masks, original_sizes, kwargs))
        return [self.processed_masks]


class _FakeModel:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.output


def _refiner(processed_masks, *, iou_scores, object_scores=None, error=None):
    processor = _FakeProcessor(processed_masks)
    model = _FakeModel(
        output=SimpleNamespace(
            pred_masks=np.zeros((1, 1, 3, 2, 2), dtype=np.float32),
            iou_scores=iou_scores,
            object_score_logits=object_scores,
        ),
        error=error,
    )
    return SAM2BBoxRefiner(processor=processor, model=model, device="cpu", torch_module=_FakeTorch())


class GroundedSam2RefinementTest(unittest.TestCase):
    def test_refines_all_boxes_in_one_prompt_batch_and_preserves_components(self) -> None:
        masks = np.zeros((2, 3, 6, 8), dtype=bool)
        # Box 0: mask option 1 has better quality and two visible components.
        masks[0, 1, 1:4, 1:4] = True
        masks[0, 1, 4:6, 6:8] = True
        # Box 1: best option is 2.
        masks[1, 2, 0:2, 5:8] = True
        refiner = _refiner(
            masks,
            iou_scores=np.asarray([[[0.10, 0.90, 0.30], [0.20, 0.40, 0.80]]]),
            object_scores=np.asarray([[[1.25], [0.75]]]),
        )
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)

        results = refine_grounding_boxes(
            rgb,
            (BBox(1, 1, 3, 3), BBox(5, 0, 3, 2)),
            refiner=refiner,
            cleanup=MaskCleanupConfig(min_component_pixels=2, min_component_area_ratio=0.0),
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(item.status == "valid" for item in results))
        self.assertEqual([item.quality.selected_mask_index for item in results], [1, 2])
        self.assertEqual(results[0].quality.object_score_logit, 1.25)
        self.assertEqual(results[0].quality.removed_component_count, 0)
        self.assertTrue(results[0].cleaned_mask[4, 6])
        self.assertEqual(len(refiner.processor.calls), 1)
        self.assertEqual(len(refiner.model.calls), 1)
        self.assertEqual(
            refiner.processor.calls[0]["input_boxes"],
            [[[1.0, 1.0, 4.0, 4.0], [5.0, 0.0, 8.0, 2.0]]],
        )
        self.assertEqual(refiner.model.calls[0]["multimask_output"], True)

    def test_cleanup_removes_only_small_island_and_keeps_largest_component(self) -> None:
        raw = np.zeros((6, 7), dtype=bool)
        raw[0:3, 0:3] = True
        raw[4, 4] = True
        raw[5, 6] = True
        cleaned, details = clean_mask_components(
            raw,
            cleanup=MaskCleanupConfig(min_component_pixels=2, min_component_area_ratio=1.0),
        )
        self.assertEqual(int(cleaned.sum()), 9)
        self.assertEqual(details["removed_component_count"], 2)
        self.assertEqual(details["removed_foreground_pixels"], 2)
        self.assertEqual(details["absolute_threshold_pixels"], 2)
        self.assertEqual(details["relative_threshold_pixels"], 11)

        # Even an intentionally harsh threshold cannot remove the major region.
        largest_only, details = clean_mask_components(
            raw,
            cleanup=MaskCleanupConfig(min_component_pixels=50, min_component_area_ratio=1.0),
        )
        self.assertEqual(int(largest_only.sum()), 9)
        self.assertEqual(details["removed_component_count"], 2)

    def test_invalid_box_and_inference_failure_return_ordered_bbox_fallbacks(self) -> None:
        refiner = _refiner(
            np.zeros((1, 3, 6, 8), dtype=bool),
            iou_scores=np.asarray([[[0.9, 0.2, 0.1]]]),
            error=RuntimeError("synthetic failure"),
        )
        results = refine_grounding_boxes(
            np.zeros((6, 8, 3), dtype=np.uint8),
            (BBox(-10, -10, 2, 2), BBox(1, 1, 2, 2)),
            refiner=refiner,
        )
        self.assertEqual([item.status for item in results], ["bbox_fallback", "bbox_fallback"])
        self.assertEqual(results[0].fallback_reason, "INVALID_BBOX")
        self.assertEqual(results[1].fallback_reason, "INFERENCE_FAILED:RuntimeError")
        self.assertEqual(len(refiner.model.calls), 1)

    def test_bad_quality_for_one_box_does_not_drop_other_box(self) -> None:
        masks = np.zeros((2, 2, 6, 8), dtype=bool)
        masks[0, 0, 1:3, 1:3] = True
        masks[1, 0, 3:5, 3:5] = True
        refiner = _refiner(
            masks,
            iou_scores=np.asarray([[[0.8, 0.1], [np.nan, np.nan]]]),
        )
        results = refine_grounding_boxes(
            np.zeros((6, 8, 3), dtype=np.uint8),
            (BBox(1, 1, 2, 2), BBox(3, 3, 2, 2)),
            refiner=refiner,
        )
        self.assertEqual(results[0].status, "valid")
        self.assertEqual(results[1].status, "bbox_fallback")
        self.assertEqual(results[1].fallback_reason, "QUALITY_FAILED:ValueError")

    def test_input_contracts(self) -> None:
        with self.assertRaises(ValueError):
            MaskCleanupConfig(min_component_pixels=-1)
        with self.assertRaises(ValueError):
            MaskCleanupConfig(min_component_area_ratio=1.1)
        with self.assertRaises(ValueError):
            refine_grounding_boxes(
                np.zeros((4, 4, 4), dtype=np.uint8),
                (),
                refiner=_refiner(np.zeros((0, 3, 4, 4), dtype=bool), iou_scores=np.zeros((1, 0, 3))),
            )


if __name__ == "__main__":
    unittest.main()
