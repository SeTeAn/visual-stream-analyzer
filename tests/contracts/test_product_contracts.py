from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

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


_DIGEST = "a" * 64


def _stream(frame_count: int = 2) -> ProductStream:
    return ProductStream(
        stream_id="stream_001",
        input_schema_version="image-stream-1.0",
        manifest_sha256=_DIGEST,
        frame_count=frame_count,
    )


def _pipeline() -> ProductPipeline:
    return ProductPipeline(
        profile_id="profile_001",
        profile_sha256=_DIGEST,
    )


def _frame(frame_id: str, index: int, masks: tuple[ProductMask, ...] = ()) -> ProductFrame:
    return ProductFrame(
        frame_id=frame_id,
        frame_index=index,
        source_image=f"frames/{frame_id}.png",
        image_size=ImageSize(2, 2),
        masks=masks,
    )


class ProductContractsTest(unittest.TestCase):
    def test_result_assigns_deterministic_public_object_ids_and_visual_types(self) -> None:
        type_z = ProductMask(type_id="type_z", mask=np.array([[0, 1], [0, 0]], dtype=np.uint8))
        type_a = ProductMask(type_id="type_a", mask=np.array([[1, 0], [0, 0]], dtype=np.uint8))
        result = ProductResult(
            stream=_stream(),
            pipeline=_pipeline(),
            frames=(_frame("frame_b", 1, (type_z, type_a)), _frame("frame_a", 0)),
        )
        self.assertEqual([frame.frame_id for frame in result.frames], ["frame_a", "frame_b"])
        self.assertEqual([mask.object_id for mask in result.frames[1].masks], ["P00", "P01"])
        self.assertEqual([mask.type_id for mask in result.frames[1].masks], ["type_a", "type_z"])
        self.assertEqual([item.type_id for item in result.visual_types], ["type_a", "type_z"])

    def test_masks_are_full_frame_binary_and_immutable(self) -> None:
        source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        mask = ProductMask(type_id="type_001", mask=source)
        source[0, 0] = 1
        self.assertFalse(mask.mask[0, 0])
        self.assertFalse(mask.mask.flags.writeable)
        with self.assertRaises(ValueError):
            _frame("frame_001", 0, (mask,)).__class__(
                frame_id="frame_001",
                frame_index=0,
                source_image="frames/frame_001.png",
                image_size=ImageSize(3, 2),
                masks=(mask,),
            )
        with self.assertRaises(ValueError):
            ProductMask(type_id="type_001", mask=np.array([[0, 255]], dtype=np.uint8))

    def test_matches_and_events_require_resolving_public_same_type_references(self) -> None:
        first = _frame(
            "frame_001",
            0,
            (
                ProductMask(
                    type_id="type_001",
                    mask=np.array([[1, 0], [0, 0]], dtype=np.uint8),
                ),
            ),
        )
        second = _frame(
            "frame_002",
            1,
            (
                ProductMask(
                    type_id="type_001",
                    mask=np.array([[0, 0], [0, 1]], dtype=np.uint8),
                ),
            ),
        )
        first_ref = ProductObjectRef(frame_id="frame_001", object_id="P00")
        second_ref = ProductObjectRef(frame_id="frame_002", object_id="P00")
        result = ProductResult(
            stream=_stream(),
            pipeline=_pipeline(),
            frames=(second, first),
            status=ProductStatus.COMPLETED_WITH_WARNINGS,
            matches=(
                ProductMatch(
                    from_object=first_ref,
                    to_object=second_ref,
                    status="accepted",
                ),
            ),
            events=(
                ProductEvent(
                    event_id="event_001",
                    kind=ProductEventKind.PERSISTED,
                    type_id="type_001",
                    from_frame_id="frame_001",
                    to_frame_id="frame_002",
                    from_objects=(first_ref,),
                    to_objects=(second_ref,),
                ),
            ),
        )
        self.assertEqual(result.matches[0].status, "accepted")
        self.assertEqual(result.events[0].event_id, "event_001")
        with self.assertRaisesRegex(ValueError, "resolve"):
            ProductResult(
                stream=_stream(),
                pipeline=_pipeline(),
                frames=(first, second),
                matches=(
                    ProductMatch(
                        from_object=first_ref,
                        to_object=ProductObjectRef(
                            frame_id="frame_002",
                            object_id="P01",
                        ),
                        status="accepted",
                    ),
                ),
            )

    def test_contract_cannot_carry_candidate_scores_or_confidence(self) -> None:
        forbidden = {"candidate_id", "score", "confidence", "raw_id", "internal_id"}
        for contract in (ProductMask, ProductFrame, ProductResult, ProductMatch, ProductEvent):
            self.assertTrue(forbidden.isdisjoint(contract.__dataclass_fields__))
        result = ProductResult(
            stream=_stream(1),
            pipeline=_pipeline(),
            frames=(_frame("frame_001", 0),),
        )
        with self.assertRaises(FrozenInstanceError):
            result.status = ProductStatus.FAILED  # type: ignore[misc]

    def test_invalid_portable_paths_and_metadata_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "POSIX"):
            ProductFrame(
                frame_id="frame_001", frame_index=0, source_image="frames\\frame.png",
                image_size=ImageSize(1, 1),
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ProductStream(
                stream_id="stream",
                input_schema_version="stream-1",
                manifest_sha256="bad",
                frame_count=1,
            )
        with self.assertRaisesRegex(ValueError, "approved public model"):
            ProductPipeline(
                profile_id="profile_001",
                profile_sha256=_DIGEST,
                models=("SAM2",),
            )


if __name__ == "__main__":
    unittest.main()
