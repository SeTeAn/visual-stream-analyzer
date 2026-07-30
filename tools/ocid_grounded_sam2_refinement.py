"""BBox-prompted SAM2 interface used by OCID utilities."""

from stream_analysis.runtime.sam2 import (
    GroundedMaskResult,
    MaskCleanupConfig,
    MaskQuality,
    RefinementStatus,
    SAM2BBoxRefiner,
    clean_mask_components,
    load_local_sam2_bbox_refiner,
    mask_bbox_from_full_frame,
    refine_grounding_boxes,
)

__all__ = [
    "GroundedMaskResult",
    "MaskCleanupConfig",
    "MaskQuality",
    "RefinementStatus",
    "SAM2BBoxRefiner",
    "clean_mask_components",
    "load_local_sam2_bbox_refiner",
    "mask_bbox_from_full_frame",
    "refine_grounding_boxes",
]
