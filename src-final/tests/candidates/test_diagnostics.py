import unittest

import numpy as np

from stream_analysis import (
    BBox,
    CandidateExtractionResult,
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    FrameCandidateDiagnostics,
    ImageSize,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
)
from stream_analysis.candidates import candidate_mask_digest


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="candidate_extraction",
        producer_version="1.0",
        config_version="1.0",
        config_digest="sha256:diagnostics",
    )


class CandidateDiagnosticsTest(unittest.TestCase):
    def test_candidate_mask_digest_includes_shape_and_contents(self) -> None:
        first = np.zeros((3, 4), dtype=np.bool_)
        second = first.copy()
        second[1, 2] = True
        same_bits_different_shape = np.zeros((2, 6), dtype=np.bool_)

        self.assertEqual(candidate_mask_digest(first), candidate_mask_digest(first.copy()))
        self.assertNotEqual(candidate_mask_digest(first), candidate_mask_digest(second))
        self.assertNotEqual(candidate_mask_digest(first), candidate_mask_digest(same_bits_different_shape))

    def test_candidate_mask_record_validates_bbox_shape_and_digest(self) -> None:
        mask = np.zeros((3, 4), dtype=np.bool_)
        mask[1, 1] = True
        record = CandidateMaskRecord(
            mask_ref="mask:cand_frame_001_001",
            mask_digest=candidate_mask_digest(mask),
            candidate_id="cand_frame_001_001",
            frame_id="frame_001",
            coordinate_bbox=BBox(2, 3, 4, 3),
            mask=mask,
        )
        self.assertFalse(record.mask.flags["WRITEABLE"])

        with self.assertRaises(ValueError):
            CandidateMaskRecord(
                mask_ref="mask:cand_frame_001_001",
                mask_digest=candidate_mask_digest(mask),
                candidate_id="cand_frame_001_001",
                frame_id="frame_001",
                coordinate_bbox=BBox(2, 3, 5, 3),
                mask=mask,
            )
        with self.assertRaises(ValueError):
            CandidateMaskRecord(
                mask_ref="mask:cand_frame_001_001",
                mask_digest="sha256:" + "0" * 64,
                candidate_id="cand_frame_001_001",
                frame_id="frame_001",
                coordinate_bbox=BBox(2, 3, 4, 3),
                mask=mask,
            )

    def test_empty_snapshot_accepts_empty_mask_store(self) -> None:
        result = CandidateExtractionResult(
            envelope=RecordEnvelope(
                record_id="candidate_result_synthetic_stream",
                schema_version="candidate-extraction-result-1.0",
                stream_id="synthetic_stream",
                producer=_producer(),
                context=StageContext(),
            ),
            extractor_source="controlled_background_components_v1",
            candidates=(),
            frame_diagnostics=(
                FrameCandidateDiagnostics(
                    envelope=RecordEnvelope(
                        record_id="candidate_diag_frame_001",
                        schema_version="frame-candidate-diagnostics-1.0",
                        stream_id="synthetic_stream",
                        producer=_producer(),
                        context=StageContext(frame_id="frame_001"),
                    ),
                    frame_id="frame_001",
                    frame_index=0,
                    image_size=ImageSize(12, 10),
                    summary={"accepted_candidate_count": 0},
                ),
            ),
        )

        snapshot = CandidateExtractionSnapshot(result=result, masks={})
        self.assertIs(snapshot.result, result)
        self.assertEqual(dict(snapshot.masks), {})


if __name__ == "__main__":
    unittest.main()
