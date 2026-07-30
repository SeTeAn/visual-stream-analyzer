from __future__ import annotations

import unittest

from stream_analysis.runtime import sam2 as runtime
from tools import ocid_grounded_sam2_refinement as ocid_interface


class SAM2RuntimeTest(unittest.TestCase):
    def test_tool_module_uses_the_shared_runtime_implementation(self) -> None:
        for name in ocid_interface.__all__:
            self.assertIs(getattr(ocid_interface, name), getattr(runtime, name))
        for name in (
            "GroundedMaskResult",
            "MaskCleanupConfig",
            "MaskQuality",
            "SAM2BBoxRefiner",
            "clean_mask_components",
            "load_local_sam2_bbox_refiner",
            "mask_bbox_from_full_frame",
            "refine_grounding_boxes",
        ):
            self.assertEqual(getattr(runtime, name).__module__, runtime.__name__)


if __name__ == "__main__":
    unittest.main()
