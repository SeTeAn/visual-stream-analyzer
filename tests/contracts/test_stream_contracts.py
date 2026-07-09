from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

from stream_analysis import (
    FrameRecord,
    ImageSize,
    ImageStream,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:abc123",
    )


def _stream_envelope(stream_id: str = "probe_01") -> RecordEnvelope:
    return RecordEnvelope(
        record_id=f"stream_record_{stream_id}",
        schema_version="image-stream-1.0",
        stream_id=stream_id,
        producer=_producer(),
    )


def _frame(frame_id: str, index: int, stream_id: str = "probe_01") -> FrameRecord:
    return FrameRecord(
        envelope=RecordEnvelope(
            record_id=f"frame_record_{frame_id}",
            schema_version="frame-record-1.0",
            stream_id=stream_id,
            producer=_producer(),
            context=StageContext(frame_id=frame_id),
        ),
        frame_id=frame_id,
        index=index,
        image_path=f"frames/{frame_id}.png",
        image_size=ImageSize(640, 480),
    )


class StreamContractsTest(unittest.TestCase):
    def test_frame_record_contains_known_image_size_and_relative_path(self) -> None:
        frame = _frame("frame_001", 1)
        self.assertEqual(frame.stream_id, "probe_01")
        self.assertEqual(frame.image_path, "frames/frame_001.png")
        self.assertEqual(frame.image_size, ImageSize(640, 480))

    def test_backslash_path_is_canonicalized_to_posix(self) -> None:
        frame = FrameRecord(
            envelope=RecordEnvelope(
                record_id="frame_record_001",
                schema_version="frame-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                context=StageContext(frame_id="frame_001"),
            ),
            frame_id="frame_001",
            index=1,
            image_path="frames\\frame_001.png",
            image_size=ImageSize(640, 480),
        )
        self.assertEqual(frame.image_path, "frames/frame_001.png")

    def test_frame_context_must_match_frame_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            FrameRecord(
                envelope=RecordEnvelope(
                    record_id="frame_record_001",
                    schema_version="frame-record-1.0",
                    stream_id="probe_01",
                    producer=_producer(),
                    context=StageContext(frame_id="frame_002"),
                ),
                frame_id="frame_001",
                index=1,
                image_path="frames/frame_001.png",
                image_size=ImageSize(640, 480),
            )

    def test_frame_context_rejects_downstream_stage_ids(self) -> None:
        downstream_context_fields = (
            "candidate_id",
            "pair_id",
            "type_id",
            "event_id",
        )
        for field_name in downstream_context_fields:
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(ValueError, "no downstream stage IDs"),
            ):
                FrameRecord(
                    envelope=RecordEnvelope(
                        record_id="frame_record_001",
                        schema_version="frame-record-1.0",
                        stream_id="probe_01",
                        producer=_producer(),
                        context=StageContext(
                            frame_id="frame_001",
                            **{field_name: f"{field_name}_001"},
                        ),
                    ),
                    frame_id="frame_001",
                    index=1,
                    image_path="frames/frame_001.png",
                    image_size=ImageSize(640, 480),
                )

    def test_invalid_frame_id_and_index_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _frame("bad frame", 1)
        with self.assertRaises(ValueError):
            _frame("frame_001", -1)
        with self.assertRaises(TypeError):
            _frame("frame_001", True)  # type: ignore[arg-type]

    def test_absolute_paths_are_rejected_syntactically(self) -> None:
        for path in ("/tmp/frame.png", "C:/tmp/frame.png", "C:\\tmp\\frame.png"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative"):
                FrameRecord(
                    envelope=RecordEnvelope(
                        record_id="frame_record_001",
                        schema_version="frame-record-1.0",
                        stream_id="probe_01",
                        producer=_producer(),
                        context=StageContext(frame_id="frame_001"),
                    ),
                    frame_id="frame_001",
                    index=1,
                    image_path=path,
                    image_size=ImageSize(640, 480),
                )

    def test_parent_traversal_paths_are_rejected_syntactically(self) -> None:
        for path in ("../frame.png", "frames/../../frame.png", "frames\\..\\frame.png"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "must not leave"):
                FrameRecord(
                    envelope=RecordEnvelope(
                        record_id="frame_record_001",
                        schema_version="frame-record-1.0",
                        stream_id="probe_01",
                        producer=_producer(),
                        context=StageContext(frame_id="frame_001"),
                    ),
                    frame_id="frame_001",
                    index=1,
                    image_path=path,
                    image_size=ImageSize(640, 480),
                )

    def test_stream_sorts_frames_deterministically_by_unique_index(self) -> None:
        stream = ImageStream(
            envelope=_stream_envelope(),
            frames=(_frame("frame_003", 3), _frame("frame_001", 1), _frame("frame_002", 2)),
        )
        self.assertEqual([frame.index for frame in stream.frames], [1, 2, 3])
        self.assertEqual(
            [frame.frame_id for frame in stream.frames],
            ["frame_001", "frame_002", "frame_003"],
        )

    def test_stream_rejects_duplicate_frame_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_id values must be unique"):
            ImageStream(
                envelope=_stream_envelope(),
                frames=(_frame("frame_001", 1), _frame("frame_001", 2)),
            )

    def test_stream_rejects_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "indices must be unique"):
            ImageStream(
                envelope=_stream_envelope(),
                frames=(_frame("frame_001", 1), _frame("frame_002", 1)),
            )

    def test_stream_rejects_mixed_stream_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            ImageStream(
                envelope=_stream_envelope("probe_01"),
                frames=(_frame("frame_001", 1, "probe_02"),),
            )

    def test_stream_is_non_empty_and_manifest_ordered(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            ImageStream(envelope=_stream_envelope(), frames=())
        with self.assertRaisesRegex(ValueError, "manifest"):
            ImageStream(
                envelope=_stream_envelope(),
                frames=(_frame("frame_001", 1),),
                ordering="filename",
            )

    def test_stream_contracts_have_no_annotation_or_gt_fields(self) -> None:
        forbidden = {"annotation", "annotation_path", "gt_labels", "visual_type_id", "metrics"}
        self.assertTrue(forbidden.isdisjoint({field.name for field in fields(FrameRecord)}))
        self.assertTrue(forbidden.isdisjoint({field.name for field in fields(ImageStream)}))

    def test_stream_and_frames_are_immutable(self) -> None:
        stream = ImageStream(envelope=_stream_envelope(), frames=(_frame("frame_001", 1),))
        self.assertIsInstance(stream.frames, tuple)
        with self.assertRaises(FrozenInstanceError):
            stream.ordering = "other"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            stream.frames[0].index = 2  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
