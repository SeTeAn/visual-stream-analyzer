from __future__ import annotations

import unittest
from dataclasses import replace

from stream_analysis import (
    AssignmentProvenance,
    CandidateEndpoint,
    FrameMatchingResult,
    FramePair,
    ImageSize,
    MetricOrientation,
    PairEligibility,
    PairwiseScoreRecord,
    ProducerProvenance,
    RecordEnvelope,
    SpatialEvidence,
    StageContext,
    ValidityStatus,
    build_pair_scores_payload,
)


class PairScoreArtifactTest(unittest.TestCase):
    def test_pair_score_payload_preserves_invalid_warning_error_lineage(self) -> None:
        producer = ProducerProvenance(
            producer_stage="matching",
            producer_version="1.0.0",
            config_version="1.0.0",
            config_digest="sha256:abc",
        )
        frame_pair = FramePair(
            from_frame_id="frame_001",
            from_frame_index=0,
            from_frame_size=ImageSize(width=32, height=24),
            to_frame_id="frame_002",
            to_frame_index=1,
            to_frame_size=ImageSize(width=32, height=24),
        )
        left = CandidateEndpoint(
            candidate_id="candidate_left",
            frame_id="frame_001",
            frame_index=0,
        )
        right = CandidateEndpoint(
            candidate_id="candidate_right",
            frame_id="frame_002",
            frame_index=1,
        )
        score = PairwiseScoreRecord(
            envelope=RecordEnvelope(
                record_id="pair_001",
                schema_version="pair-score-1.0",
                stream_id="stream_1",
                producer=producer,
                context=StageContext(pair_id="pair_001"),
                validity_status=ValidityStatus.INVALID,
                warning_ids=("warning_1",),
                error_ids=("error_1",),
                provenance_refs=("representation_left", "representation_right"),
            ),
            pair_id="pair_001",
            comparison_id="comparison_001",
            frame_pair=frame_pair,
            from_endpoint=left,
            to_endpoint=right,
            representation_record_ids=("representation_left", "representation_right"),
            representation_variant_id="handcrafted_bbox_v1",
            scorer_id="handcrafted_canonical_scorer",
            scorer_version="1.0.0",
            raw_metric_name=None,
            raw_metric_value=None,
            metric_orientation=MetricOrientation.LOWER_IS_BETTER,
            visual_score=None,
            real_cost=None,
            spatial_evidence=SpatialEvidence(normalized_distance=None),
            eligibility=PairEligibility.INELIGIBLE,
            eligibility_reason="INVALID_REPRESENTATION",
        )
        result = FrameMatchingResult(
            envelope=RecordEnvelope(
                record_id="matching_result_001",
                schema_version="frame-matching-result-1.0",
                stream_id="stream_1",
                producer=producer,
                context=StageContext(pair_id="comparison_001"),
                validity_status=ValidityStatus.INVALID,
                error_ids=("error_1",),
            ),
            comparison_id="comparison_001",
            frame_pair=frame_pair,
            representation_variant_id="handcrafted_bbox_v1",
            scorer_id="handcrafted_canonical_scorer",
            scorer_version="1.0.0",
            assignment=AssignmentProvenance(
                assignment_id="assignment_001",
                policy_id="test_assignment",
                policy_version="1.0.0",
                config_digest="sha256:def",
            ),
            from_candidates=(left,),
            to_candidates=(right,),
            pairwise_scores=(score,),
            selected_matches=(),
            unmatched=(),
        )
        payload = build_pair_scores_payload(
            run_id="run_1",
            stream_id="stream_1",
            analysis_config_digest="sha256:analysis",
            scoring_config_digest="sha256:scoring",
            representation_variant_id="handcrafted_bbox_v1",
            scorer_id="handcrafted_canonical_scorer",
            scorer_version="1.0.0",
            matching_results=(result,),
        )
        row = payload["pairs"][0]
        self.assertEqual(row["lineage"]["validity_status"], "invalid")
        self.assertEqual(row["lineage"]["warning_ids"], ["warning_1"])
        self.assertEqual(row["lineage"]["error_ids"], ["error_1"])
        self.assertEqual(
            row["lineage"]["provenance_refs"],
            ["representation_left", "representation_right"],
        )
        self.assertNotIn("embedding", str(payload).casefold())
        self.assertNotIn("feature_vector", str(payload).casefold())

        mixed_score = replace(
            score,
            representation_variant_id="different_variant",
        )
        mixed_result = replace(
            result,
            representation_variant_id="different_variant",
            pairwise_scores=(mixed_score,),
        )
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            build_pair_scores_payload(
                run_id="run_1",
                stream_id="stream_1",
                analysis_config_digest="sha256:analysis",
                scoring_config_digest="sha256:scoring",
                representation_variant_id="handcrafted_bbox_v1",
                scorer_id="handcrafted_canonical_scorer",
                scorer_version="1.0.0",
                matching_results=(result, mixed_result),
            )

        empty = build_pair_scores_payload(
            run_id="run_empty",
            stream_id="stream_1",
            analysis_config_digest="sha256:analysis",
            scoring_config_digest="sha256:scoring",
            representation_variant_id="handcrafted_bbox_v1",
            scorer_id="handcrafted_canonical_scorer",
            scorer_version="1.0.0",
            matching_results=(),
        )
        self.assertEqual(empty["pairs"], [])
        self.assertEqual(empty["representation_variant_id"], "handcrafted_bbox_v1")
        self.assertEqual(empty["scorer_id"], "handcrafted_canonical_scorer")


if __name__ == "__main__":
    unittest.main()
