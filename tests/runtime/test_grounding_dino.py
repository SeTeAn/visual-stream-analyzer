from __future__ import annotations

import unittest

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.runtime import grounding_dino as runtime
from tools import evaluate_ocid_grounding_dino_extractor as extractor
from tools import evaluate_ocid_grounding_dino_hardening as hardening


class GroundingDinoRuntimeTest(unittest.TestCase):
    def test_tool_types_and_helpers_delegate_to_the_shared_runtime(self) -> None:
        self.assertIs(extractor.RawPrediction, runtime.RawPrediction)
        self.assertIs(extractor.FilterProfile, runtime.FilterProfile)
        self.assertIs(extractor._load_model_processor, runtime.load_model_processor)
        self.assertIs(
            extractor._normalise_processor_predictions,
            runtime.normalise_processor_predictions,
        )
        self.assertIs(hardening.GeometryProfile, runtime.GeometryProfile)
        self.assertIs(hardening.geometry_rejects, runtime.geometry_rejects)

    def test_profile_selection_preserves_threshold_nms_geometry_order(self) -> None:
        raw = (
            runtime.RawPrediction("f", 0, 0, 0.90, "object", BBox(0, 0, 95, 70)),
            runtime.RawPrediction("f", 0, 1, 0.80, "object", BBox(1, 1, 95, 70)),
            runtime.RawPrediction("f", 0, 2, 0.14, "object", BBox(50, 50, 5, 5)),
            runtime.RawPrediction("f", 0, 3, 0.70, "object", BBox(70, 70, 5, 5)),
        )
        surface = runtime.GeometryProfile(
            "g_surface_span_055_090",
            min_area_ratio=0.55,
            min_span_ratio=0.90,
        )
        selected = runtime.select_predictions_for_profile(
            raw,
            filter_profile=runtime.FilterProfile(0.15, 0.30),
            geometry=surface,
            image_sizes={"f": ImageSize(width=100, height=100)},
            surface_geometries=(surface,),
        )

        self.assertEqual([item.prediction.prediction_index for item in selected], [0, 3])
        self.assertEqual([item.geometry_rejected for item in selected], [True, False])
        self.assertEqual([item.surface_like for item in selected], [True, False])


if __name__ == "__main__":
    unittest.main()
