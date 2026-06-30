from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, fields

import stream_analysis
import stream_analysis.contracts as contracts
from stream_analysis import (
    ArtifactReference,
    AssignmentProvenance,
    BBox,
    CalibrationStatus,
    CandidateEndpoint,
    CandidateExtractionResult,
    CandidateRecord,
    ChangeEvent,
    ConfidenceMetadata,
    DinoEmbeddingPayload,
    EmissionStatus,
    EventEvidence,
    EventKind,
    EventStatus,
    FrameCandidateDiagnostics,
    FrameComparison,
    FrameMatchingResult,
    FramePair,
    GeometryFeatureMetadata,
    GroupingResult,
    HandcraftedFeatureGroup,
    HandcraftedPayload,
    ImageSize,
    InputQualityMetadata,
    MaskReference,
    MatchDecisionStatus,
    MatchRecord,
    MembershipStatus,
    MetricOrientation,
    PairEligibility,
    PairwiseScoreRecord,
    ProducerProvenance,
    RecordEnvelope,
    RecurringVisualType,
    RepresentationFamily,
    RepresentationRecord,
    RepresentationSummary,
    RunStatus,
    RuntimeMetadata,
    SpatialEvidence,
    StageContext,
    StreamAnalysisResult,
    TypeAssignmentRecord,
    TypePresenceSummary,
    UnmatchedRecord,
    UnmatchedSide,
    ValidityStatus,
    VersionedMetadata,
    VisualTypeStatus,
)


STREAM_ID = "probe_01"


def _producer(stage: str, digest: str | None = None) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0",
        config_version=f"{stage}_v1",
        config_digest=digest or f"sha256:{stage}",
    )


def _envelope(
    record_id: str,
    stage: str,
    context: StageContext = StageContext(),
    *,
    digest: str | None = None,
    validity: ValidityStatus = ValidityStatus.VALID,
    warning_ids: tuple[str, ...] = (),
    error_ids: tuple[str, ...] = (),
) -> RecordEnvelope:
    return RecordEnvelope(
        record_id=record_id,
        schema_version="stage-record-1.0",
        stream_id=STREAM_ID,
        producer=_producer(stage, digest),
        context=context,
        validity_status=validity,
        warning_ids=warning_ids,
        error_ids=error_ids,
    )


def _frame_pair() -> FramePair:
    return FramePair(
        from_frame_id="frame_001",
        from_frame_index=0,
        from_frame_size=ImageSize(100, 80),
        to_frame_id="frame_002",
        to_frame_index=1,
        to_frame_size=ImageSize(100, 80),
    )


def _endpoint(candidate_id: str, side: UnmatchedSide) -> CandidateEndpoint:
    pair = _frame_pair()
    if side is UnmatchedSide.FROM:
        return CandidateEndpoint(
            candidate_id=candidate_id,
            frame_id=pair.from_frame_id,
            frame_index=pair.from_frame_index,
        )
    return CandidateEndpoint(
        candidate_id=candidate_id,
        frame_id=pair.to_frame_id,
        frame_index=pair.to_frame_index,
    )


def _assignment() -> AssignmentProvenance:
    return AssignmentProvenance(
        assignment_id="assignment_001",
        policy_id="global_linear_assignment",
        policy_version="1.0",
        config_digest="sha256:matching",
        details={"backend": "contract_fixture"},
    )


def _score(
    *,
    pair_id: str = "pair_001",
    from_candidate_id: str = "candidate_001",
    to_candidate_id: str = "candidate_002",
    visual_score: float = 0.8,
) -> PairwiseScoreRecord:
    return PairwiseScoreRecord(
        envelope=_envelope(
            pair_id,
            "pair_scoring",
            StageContext(pair_id=pair_id),
        ),
        pair_id=pair_id,
        comparison_id="comparison_001",
        frame_pair=_frame_pair(),
        from_endpoint=_endpoint(from_candidate_id, UnmatchedSide.FROM),
        to_endpoint=_endpoint(to_candidate_id, UnmatchedSide.TO),
        representation_record_ids=(f"repr_{from_candidate_id}", f"repr_{to_candidate_id}"),
        representation_variant_id="dino_crop_bbox",
        scorer_id="cosine_scorer",
        scorer_version="1.0",
        raw_metric_name="cosine_similarity",
        raw_metric_value=visual_score,
        metric_orientation=MetricOrientation.HIGHER_IS_BETTER,
        visual_score=visual_score,
        real_cost=1.0 - visual_score,
        spatial_evidence=SpatialEvidence(
            normalized_distance=0.25,
            details={"source": "bbox_centers"},
        ),
        eligibility=PairEligibility.ELIGIBLE,
        gate_results={"visual": "passed"},
        candidate_quality_refs=("quality_001",),
    )


def _match(
    *,
    score: PairwiseScoreRecord | None = None,
    match_id: str = "match_001",
    status: MatchDecisionStatus = MatchDecisionStatus.UNCERTAIN,
) -> MatchRecord:
    score = score or _score()
    return MatchRecord(
        envelope=_envelope(
            match_id,
            "matching",
            StageContext(pair_id=score.pair_id),
        ),
        match_id=match_id,
        comparison_id=score.comparison_id,
        pair_id=score.pair_id,
        frame_pair=score.frame_pair,
        from_endpoint=score.from_endpoint,
        to_endpoint=score.to_endpoint,
        pairwise_score_id=score.pair_id,
        assignment=_assignment(),
        status=status,
        visual_score=score.visual_score,
        real_cost=score.real_cost,
        spatial_evidence=score.spatial_evidence,
        confidence_components={"local_support": 0.6},
        delta_global=None,
    )


def _unmatched(candidate_id: str, side: UnmatchedSide, suffix: str) -> UnmatchedRecord:
    endpoint = _endpoint(candidate_id, side)
    return UnmatchedRecord(
        envelope=_envelope(
            f"unmatched_{suffix}",
            "matching",
            StageContext(
                frame_id=endpoint.frame_id,
                candidate_id=endpoint.candidate_id,
                pair_id="comparison_001",
            ),
        ),
        unmatched_id=f"unmatched_{suffix}",
        comparison_id="comparison_001",
        frame_pair=_frame_pair(),
        endpoint=endpoint,
        side=side,
        reason_code="LOST_GLOBAL_ASSIGNMENT",
        assignment=_assignment(),
        selected_dummy_id=f"dummy_{suffix}",
        unmatched_cost=0.5,
        global_unmatched_margin=None,
    )


def _confidence(value: float | None = None) -> ConfidenceMetadata:
    return ConfidenceMetadata(
        value=value,
        calibration_status=(
            CalibrationStatus.CALIBRATED
            if value is not None
            else CalibrationStatus.NOT_CALIBRATED
        ),
        components={"support": 0.75},
    )


class CandidateContractsTest(unittest.TestCase):
    def _candidate(self) -> CandidateRecord:
        bbox = BBox(10, 20, 30, 20)
        return CandidateRecord(
            envelope=_envelope(
                "candidate_001",
                "candidate_extraction",
                StageContext(frame_id="frame_001", candidate_id="candidate_001"),
            ),
            candidate_id="candidate_001",
            frame_id="frame_001",
            frame_index=0,
            frame_size=ImageSize(100, 80),
            bbox=bbox,
            center=bbox.center,
            geometry=GeometryFeatureMetadata(
                feature_schema_id="geometry_v1",
                producer_version="1.0",
                config_digest="sha256:geometry",
                values={"area": 600, "solidity": 0.9},
            ),
            candidate_source="connected_components",
            mask=MaskReference(
                mask_ref="artifact:masks/candidate_001",
                mask_digest="sha256:mask",
                producer_version="1.0",
                coordinate_bbox=bbox,
                validity_status=ValidityStatus.VALID,
            ),
            candidate_confidence=0.85,
            quality_flags=("high_contrast",),
        )

    def test_candidate_and_extraction_result_are_valid_and_ordered(self) -> None:
        candidate = self._candidate()
        diagnostics = FrameCandidateDiagnostics(
            envelope=_envelope(
                "candidate_diag_001",
                "candidate_extraction",
                StageContext(frame_id="frame_001"),
            ),
            frame_id="frame_001",
            frame_index=0,
            image_size=ImageSize(100, 80),
            summary={"candidate_count": 1},
        )
        result = CandidateExtractionResult(
            envelope=_envelope("candidate_result_001", "candidate_extraction"),
            extractor_source="connected_components",
            candidates=(candidate,),
            frame_diagnostics=(diagnostics,),
        )
        self.assertEqual(result.candidates, (candidate,))
        self.assertEqual(result.frame_diagnostics, (diagnostics,))

    def test_candidate_metadata_is_immutable_and_has_no_downstream_fields(self) -> None:
        candidate = self._candidate()
        with self.assertRaises(FrozenInstanceError):
            candidate.candidate_source = "other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            candidate.geometry.values["area"] = 1  # type: ignore[index]
        field_names = {item.name for item in fields(CandidateRecord)}
        self.assertTrue(field_names.isdisjoint({"representation", "visual_type", "gt_label"}))

    def test_candidate_rejects_nonfinite_confidence_and_mismatched_mask(self) -> None:
        candidate = self._candidate()
        values = {item.name: getattr(candidate, item.name) for item in fields(CandidateRecord)}
        values["candidate_confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            CandidateRecord(**values)
        values["candidate_confidence"] = 0.5
        values["mask"] = MaskReference(
            mask_ref="mask:other",
            mask_digest="sha256:other",
            producer_version="1.0",
            coordinate_bbox=BBox(0, 0, 5, 5),
            validity_status=ValidityStatus.VALID,
        )
        with self.assertRaisesRegex(ValueError, "coordinate_bbox"):
            CandidateRecord(**values)

    def test_extraction_result_rejects_candidate_diagnostics_frame_mismatch(self) -> None:
        candidate = self._candidate()
        mismatches = (
            (99, candidate.frame_size, "frame_index"),
            (candidate.frame_index, ImageSize(99, 80), "frame_size"),
        )
        for frame_index, image_size, message in mismatches:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                CandidateExtractionResult(
                    envelope=_envelope("candidate_result_001", "candidate_extraction"),
                    extractor_source="connected_components",
                    candidates=(candidate,),
                    frame_diagnostics=(
                        FrameCandidateDiagnostics(
                            envelope=_envelope(
                                "candidate_diag_001",
                                "candidate_extraction",
                                StageContext(frame_id="frame_001"),
                            ),
                            frame_id="frame_001",
                            frame_index=frame_index,
                            image_size=image_size,
                        ),
                    ),
                )


class RepresentationContractsTest(unittest.TestCase):
    def _record(self, payload: DinoEmbeddingPayload) -> RepresentationRecord:
        return RepresentationRecord(
            envelope=_envelope(
                "representation_001",
                "representation",
                StageContext(frame_id="frame_001", candidate_id="candidate_001"),
                digest="sha256:semantic",
            ),
            candidate_id="candidate_001",
            frame_id="frame_001",
            family=RepresentationFamily.DINO_V2,
            representation_type="dino_embedding",
            representation_version="1.0",
            input_variant="bbox_crop",
            semantic_config_digest="sha256:semantic",
            payload=payload,
            preprocessing_metadata=VersionedMetadata(
                identifier="dino_preprocess",
                version="1.0",
                details={"resize_mode": "short_side"},
            ),
            provider_metadata=VersionedMetadata(
                identifier="local_torch_provider",
                version="1.0",
            ),
            model_metadata=VersionedMetadata(
                identifier="dinov2_model",
                version="source_revision_1",
                details={"checkpoint": "external"},
            ),
            runtime_metadata=RuntimeMetadata(
                runtime_id="runtime_001",
                details={"device": "cpu", "elapsed_ms": 3.2},
            ),
            input_quality_metadata=InputQualityMetadata(candidate_confidence=0.8),
        )

    def test_embedding_dimension_is_dynamic_and_vectors_are_not_rounded(self) -> None:
        for vector in ((0.123456789, 0.2, 0.3), (0.1, 0.2, 0.3, 0.4, 0.5)):
            with self.subTest(dimension=len(vector)):
                payload = DinoEmbeddingPayload(
                    embedding=vector,
                    embedding_dimension=len(vector),
                    l2_normalized=False,
                )
                record = self._record(payload)
                self.assertEqual(record.payload.embedding, vector)
                self.assertEqual(record.payload.embedding_dimension, len(vector))

    def test_embedding_rejects_nonfinite_and_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            DinoEmbeddingPayload(
                embedding=(0.1, float("inf")),
                embedding_dimension=2,
                l2_normalized=False,
            )
        with self.assertRaisesRegex(ValueError, "embedding_dimension"):
            DinoEmbeddingPayload(
                embedding=(0.1, 0.2),
                embedding_dimension=3,
                l2_normalized=False,
            )

    def test_handcrafted_payload_is_typed_and_immutable(self) -> None:
        group = HandcraftedFeatureGroup(
            group_name="color",
            values={"histogram_distance": 0.125},
        )
        payload = HandcraftedPayload(feature_schema_id="baseline_v1", feature_groups=(group,))
        record = RepresentationRecord(
            envelope=_envelope(
                "representation_002",
                "representation",
                StageContext(frame_id="frame_001", candidate_id="candidate_001"),
                digest="sha256:baseline",
            ),
            candidate_id="candidate_001",
            frame_id="frame_001",
            family=RepresentationFamily.HANDCRAFTED,
            representation_type="handcrafted_features",
            representation_version="1.0",
            input_variant="bbox_masked",
            semantic_config_digest="sha256:baseline",
            payload=payload,
            preprocessing_metadata=VersionedMetadata(identifier="baseline_preprocess", version="1.0"),
            provider_metadata=VersionedMetadata(identifier="stdlib_provider", version="1.0"),
            model_metadata=None,
            runtime_metadata=RuntimeMetadata(runtime_id="runtime_002"),
            input_quality_metadata=InputQualityMetadata(),
        )
        self.assertIsInstance(record.payload, HandcraftedPayload)
        with self.assertRaises(TypeError):
            group.values["new"] = 1.0  # type: ignore[index]

    def test_representation_family_rejects_mixed_payload(self) -> None:
        group = HandcraftedFeatureGroup(group_name="shape", values={"compactness": 0.5})
        with self.assertRaisesRegex(TypeError, "DinoEmbeddingPayload"):
            self._record(  # type: ignore[arg-type]
                HandcraftedPayload(feature_schema_id="baseline_v1", feature_groups=(group,))
            )

    def test_semantic_and_runtime_metadata_are_separate(self) -> None:
        record = self._record(
            DinoEmbeddingPayload(
                embedding=(0.1, 0.2),
                embedding_dimension=2,
                l2_normalized=False,
            )
        )
        self.assertEqual(record.semantic_config_digest, "sha256:semantic")
        self.assertEqual(record.runtime_metadata.details["device"], "cpu")
        self.assertNotIn("device", record.preprocessing_metadata.details)
        with self.assertRaises(TypeError):
            record.runtime_metadata.details["device"] = "cuda"  # type: ignore[index]


class MatchingContractsTest(unittest.TestCase):
    def test_pair_score_keeps_raw_visual_quality_and_spatial_axes_separate(self) -> None:
        score = _score(visual_score=0.8)
        self.assertEqual(score.raw_metric_value, 0.8)
        self.assertEqual(score.visual_score, 0.8)
        self.assertEqual(score.spatial_evidence.normalized_distance, 0.25)
        self.assertEqual(score.candidate_quality_refs, ("quality_001",))
        with self.assertRaises(TypeError):
            score.gate_results["new"] = True  # type: ignore[index]

    def test_pair_score_rejects_duplicate_or_wrong_frame_endpoints(self) -> None:
        score = _score()
        values = {item.name: getattr(score, item.name) for item in fields(PairwiseScoreRecord)}
        values["to_endpoint"] = CandidateEndpoint(
            candidate_id="candidate_001",
            frame_id="frame_002",
            frame_index=1,
        )
        with self.assertRaisesRegex(ValueError, "different endpoints"):
            PairwiseScoreRecord(**values)
        values["to_endpoint"] = CandidateEndpoint(
            candidate_id="candidate_002",
            frame_id="frame_099",
            frame_index=1,
        )
        with self.assertRaisesRegex(ValueError, "to frame"):
            PairwiseScoreRecord(**values)

    def test_pair_score_rejects_invalid_visual_score_and_nonfinite_metric(self) -> None:
        score = _score()
        values = {item.name: getattr(score, item.name) for item in fields(PairwiseScoreRecord)}
        values["visual_score"] = 1.1
        with self.assertRaises(ValueError):
            PairwiseScoreRecord(**values)
        values["visual_score"] = 0.8
        values["raw_metric_value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            PairwiseScoreRecord(**values)

    def test_spatial_evidence_rejects_distance_above_normalized_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 1.0"):
            SpatialEvidence(normalized_distance=1.5)

    def _valid_result(self) -> FrameMatchingResult:
        score = _score()
        return FrameMatchingResult(
            envelope=_envelope(
                "matching_result_001",
                "matching",
                StageContext(pair_id="comparison_001"),
            ),
            comparison_id="comparison_001",
            frame_pair=_frame_pair(),
            representation_variant_id="dino_crop_bbox",
            scorer_id="cosine_scorer",
            scorer_version="1.0",
            assignment=_assignment(),
            from_candidates=(
                _endpoint("candidate_003", UnmatchedSide.FROM),
                score.from_endpoint,
            ),
            to_candidates=(
                _endpoint("candidate_004", UnmatchedSide.TO),
                score.to_endpoint,
            ),
            pairwise_scores=(score,),
            selected_matches=(_match(score=score),),
            unmatched=(
                _unmatched("candidate_003", UnmatchedSide.FROM, "003"),
                _unmatched("candidate_004", UnmatchedSide.TO, "004"),
            ),
            matrix_summary={"shape": [2, 2]},
        )

    def test_matching_result_enforces_one_final_decision_per_endpoint(self) -> None:
        result = self._valid_result()
        self.assertEqual(
            tuple(item.candidate_id for item in result.from_candidates),
            ("candidate_001", "candidate_003"),
        )
        values = {item.name: getattr(result, item.name) for item in fields(FrameMatchingResult)}
        values["unmatched"] = result.unmatched[:1]
        with self.assertRaisesRegex(ValueError, "Every endpoint"):
            FrameMatchingResult(**values)

    def test_uncertain_selected_match_cannot_be_unmatched(self) -> None:
        result = self._valid_result()
        self.assertIs(result.selected_matches[0].status, MatchDecisionStatus.UNCERTAIN)
        values = {item.name: getattr(result, item.name) for item in fields(FrameMatchingResult)}
        values["unmatched"] = result.unmatched + (
            _unmatched("candidate_001", UnmatchedSide.FROM, "001"),
        )
        with self.assertRaisesRegex(ValueError, "including an uncertain match"):
            FrameMatchingResult(**values)

    def test_matching_result_rejects_mixed_variant_or_scorer_matrix(self) -> None:
        result = self._valid_result()
        original_score = result.pairwise_scores[0]
        mutations = (
            ("representation_variant_id", "handcrafted_bbox", "representation_variant_id"),
            ("scorer_id", "other_scorer", "scorer_id"),
            ("scorer_version", "2.0", "scorer_version"),
        )
        for field_name, value, message in mutations:
            score_values = {
                item.name: getattr(original_score, item.name)
                for item in fields(PairwiseScoreRecord)
            }
            score_values[field_name] = value
            mixed_score = PairwiseScoreRecord(**score_values)
            result_values = {
                item.name: getattr(result, item.name)
                for item in fields(FrameMatchingResult)
            }
            result_values["pairwise_scores"] = (mixed_score,)
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                FrameMatchingResult(**result_values)

    def test_match_must_preserve_pairwise_visual_score(self) -> None:
        score = _score()
        match = _match(score=score)
        values = {item.name: getattr(match, item.name) for item in fields(MatchRecord)}
        values["visual_score"] = 0.7
        values["real_cost"] = 0.3
        altered = MatchRecord(**values)
        result = self._valid_result()
        result_values = {item.name: getattr(result, item.name) for item in fields(FrameMatchingResult)}
        result_values["selected_matches"] = (altered,)
        with self.assertRaisesRegex(ValueError, "preserve"):
            FrameMatchingResult(**result_values)


class GroupingContractsTest(unittest.TestCase):
    def _assignment_record(
        self,
        candidate_id: str,
        frame_id: str,
        assignment_id: str,
        primary_type_id: str = "type_001",
    ) -> TypeAssignmentRecord:
        return TypeAssignmentRecord(
            envelope=_envelope(
                assignment_id,
                "grouping",
                StageContext(
                    frame_id=frame_id,
                    candidate_id=candidate_id,
                    type_id=primary_type_id,
                ),
            ),
            assignment_id=assignment_id,
            candidate_id=candidate_id,
            frame_id=frame_id,
            primary_type_id=primary_type_id,
            membership_status=MembershipStatus.FIRM,
            evidence_kind="connected_component",
            scores={"support": 0.9},
            confidence=_confidence(0.8),
            alternative_type_ids=(),
        )

    def _visual_type(self) -> RecurringVisualType:
        return RecurringVisualType(
            envelope=_envelope(
                "type_001",
                "grouping",
                StageContext(type_id="type_001"),
            ),
            type_id="type_001",
            representation_variant_id="dino_crop_bbox",
            grouping_policy_id="stream_components",
            grouping_policy_version="1.0",
            member_candidate_ids=("candidate_003", "candidate_001"),
            instances_by_frame={"frame_001": ("candidate_003", "candidate_001")},
            count_by_frame={"frame_001": 2},
            representative_policy="medoid",
            representative_candidate_id="candidate_001",
            seed_match_ids=("match_001",),
            presence_summary={"frames": 1},
            internal_similarity_summary={"minimum": 0.7},
            confidence=_confidence(),
            status=VisualTypeStatus.ESTABLISHED,
            alternative_type_ids=(),
        )

    def test_grouping_allows_multiple_same_frame_members_and_orders_them(self) -> None:
        visual_type = self._visual_type()
        assignments = (
            self._assignment_record("candidate_003", "frame_001", "type_assignment_003"),
            self._assignment_record("candidate_001", "frame_001", "type_assignment_001"),
        )
        result = GroupingResult(
            envelope=_envelope("grouping_result_001", "grouping"),
            representation_variant_id="dino_crop_bbox",
            grouping_policy_id="stream_components",
            grouping_policy_version="1.0",
            candidate_ids=("candidate_003", "candidate_001"),
            assignments=assignments,
            recurring_types=(visual_type,),
        )
        self.assertEqual(
            visual_type.instances_by_frame["frame_001"],
            ("candidate_001", "candidate_003"),
        )
        self.assertEqual(
            tuple(item.candidate_id for item in result.assignments),
            ("candidate_001", "candidate_003"),
        )

    def test_grouping_rejects_multiple_primary_assignments_for_candidate(self) -> None:
        first = self._assignment_record("candidate_001", "frame_001", "assignment_001")
        second = self._assignment_record(
            "candidate_001",
            "frame_001",
            "assignment_002",
            primary_type_id="type_002",
        )
        with self.assertRaisesRegex(ValueError, "exactly one primary"):
            GroupingResult(
                envelope=_envelope("grouping_result_001", "grouping"),
                representation_variant_id="dino_crop_bbox",
                grouping_policy_id="stream_components",
                grouping_policy_version="1.0",
                candidate_ids=("candidate_001",),
                assignments=(first, second),
                recurring_types=(self._visual_type(),),
            )

    def test_grouping_rejects_physical_instance_semantics(self) -> None:
        assignment = self._assignment_record("candidate_001", "frame_001", "assignment_001")
        values = {item.name: getattr(assignment, item.name) for item in fields(TypeAssignmentRecord)}
        values["physical_instance_claim"] = True
        with self.assertRaisesRegex(ValueError, "physical-instance"):
            TypeAssignmentRecord(**values)

    def test_grouping_rejects_dangling_alternative_type_ids(self) -> None:
        assignments = (
            self._assignment_record("candidate_001", "frame_001", "type_assignment_001"),
            self._assignment_record("candidate_003", "frame_001", "type_assignment_003"),
        )
        visual_type = self._visual_type()
        base_values = {
            "envelope": _envelope("grouping_result_001", "grouping"),
            "representation_variant_id": "dino_crop_bbox",
            "grouping_policy_id": "stream_components",
            "grouping_policy_version": "1.0",
            "candidate_ids": ("candidate_001", "candidate_003"),
        }

        assignment_values = {
            item.name: getattr(assignments[0], item.name)
            for item in fields(TypeAssignmentRecord)
        }
        assignment_values["alternative_type_ids"] = ("type_missing",)
        dangling_assignment = TypeAssignmentRecord(**assignment_values)
        with self.assertRaisesRegex(ValueError, "assignment alternative_type_id"):
            GroupingResult(
                **base_values,
                assignments=(dangling_assignment, assignments[1]),
                recurring_types=(visual_type,),
            )

        type_values = {
            item.name: getattr(visual_type, item.name)
            for item in fields(RecurringVisualType)
        }
        type_values["alternative_type_ids"] = ("type_missing",)
        dangling_type = RecurringVisualType(**type_values)
        with self.assertRaisesRegex(ValueError, "recurring type alternative_type_id"):
            GroupingResult(
                **base_values,
                assignments=assignments,
                recurring_types=(dangling_type,),
            )


class EventAndTopLevelContractsTest(unittest.TestCase):
    def _evidence(self) -> EventEvidence:
        return EventEvidence(
            from_count=1,
            to_count=1,
            from_firm_count=1,
            to_firm_count=0,
            from_member_ids=("candidate_001",),
            to_member_ids=("candidate_002",),
            uncertain_match_ids=("match_001",),
            position_shift_norm=0.1,
            details={"predicate": "contract_fixture"},
        )

    def _event(self, kind: EventKind = EventKind.PERSISTED, suffix: str = "001") -> ChangeEvent:
        event_id = f"event_{suffix}"
        return ChangeEvent(
            envelope=_envelope(
                event_id,
                "events",
                StageContext(
                    pair_id="comparison_001",
                    type_id="type_001",
                    event_id=event_id,
                ),
            ),
            event_id=event_id,
            comparison_id="comparison_001",
            frame_pair=_frame_pair(),
            kind=kind,
            predicted_type_id="type_001",
            status=EventStatus.UNCERTAIN,
            evidence=self._evidence(),
            confidence=_confidence(),
            policy_id="type_change_rules",
            policy_version="1.0",
        )

    def _comparison(self, event: ChangeEvent) -> FrameComparison:
        return FrameComparison(
            envelope=_envelope(
                "comparison_001",
                "events",
                StageContext(pair_id="comparison_001"),
            ),
            comparison_id="comparison_001",
            frame_pair=_frame_pair(),
            emission_status=EmissionStatus.PRODUCED,
            representation_variant_id="dino_crop_bbox",
            scorer_id="cosine_scorer",
            scorer_version="1.0",
            event_policy_id="type_change_rules",
            event_policy_version="1.0",
            accepted_match_ids=(),
            uncertain_match_ids=("match_001",),
            unmatched_ids=(),
            from_type_presence={
                "type_001": TypePresenceSummary(
                    type_id="type_001",
                    candidate_ids=("candidate_001",),
                    firm_count=1,
                    possible_count=1,
                )
            },
            to_type_presence={
                "type_001": TypePresenceSummary(
                    type_id="type_001",
                    candidate_ids=("candidate_002",),
                    firm_count=0,
                    possible_count=1,
                )
            },
            change_event_ids=(event.event_id,),
        )

    def _minimal_top_result(
        self,
        event: ChangeEvent,
        comparison: FrameComparison,
    ) -> StreamAnalysisResult:
        return StreamAnalysisResult(
            envelope=_envelope("run_reference_check", "pipeline"),
            run_id="run_reference_check",
            pipeline_version="1.0",
            run_status=RunStatus.COMPLETED,
            frame_ids=("frame_001", "frame_002"),
            data_provenance=VersionedMetadata(identifier="probe_stream", version="1.0"),
            model_provenance=(),
            runtime_summary=RuntimeMetadata(runtime_id="runtime_reference_check"),
            recurring_type_ids=("type_001",),
            frame_comparisons=(comparison,),
            change_events=(event,),
        )

    def test_all_five_event_kinds_are_distinct_and_constructible(self) -> None:
        self.assertEqual(len(EventKind), 5)
        for index, kind in enumerate(EventKind, start=1):
            with self.subTest(kind=kind):
                event = self._event(kind, f"{index:03d}")
                self.assertIs(event.kind, kind)
                self.assertIsNone(event.confidence.value)
                self.assertIs(
                    event.confidence.calibration_status,
                    CalibrationStatus.NOT_CALIBRATED,
                )

    def test_event_position_shift_uses_normalized_plane_sqrt2_bound(self) -> None:
        evidence = self._evidence()
        values = {item.name: getattr(evidence, item.name) for item in fields(EventEvidence)}
        values["position_shift_norm"] = 1.2
        self.assertEqual(EventEvidence(**values).position_shift_norm, 1.2)
        values["position_shift_norm"] = 1.5
        with self.assertRaisesRegex(ValueError, f"at most {math.sqrt(2.0)}"):
            EventEvidence(**values)

    def test_invalid_comparison_must_be_withheld(self) -> None:
        invalid_envelope = _envelope(
            "comparison_002",
            "events",
            StageContext(pair_id="comparison_002"),
            validity=ValidityStatus.INVALID,
            error_ids=("error_comparison_002",),
        )
        comparison = FrameComparison(
            envelope=invalid_envelope,
            comparison_id="comparison_002",
            frame_pair=_frame_pair(),
            emission_status=EmissionStatus.WITHHELD,
            representation_variant_id="dino_crop_bbox",
            scorer_id="cosine_scorer",
            scorer_version="1.0",
            event_policy_id="type_change_rules",
            event_policy_version="1.0",
            accepted_match_ids=(),
            uncertain_match_ids=(),
            unmatched_ids=(),
            from_type_presence={},
            to_type_presence={},
            withheld_diagnostic_ids=("error_comparison_002",),
        )
        self.assertIs(comparison.emission_status, EmissionStatus.WITHHELD)
        values = {item.name: getattr(comparison, item.name) for item in fields(FrameComparison)}
        values["emission_status"] = EmissionStatus.PRODUCED
        values["withheld_diagnostic_ids"] = ()
        with self.assertRaisesRegex(ValueError, "must withhold"):
            FrameComparison(**values)

    def test_status_axes_are_separate_types(self) -> None:
        axes = (
            ValidityStatus,
            MatchDecisionStatus,
            MembershipStatus,
            EventStatus,
            EmissionStatus,
            RunStatus,
            MetricOrientation,
        )
        self.assertEqual(len(axes), len(set(axes)))
        self.assertIsNot(ValidityStatus.VALID, MatchDecisionStatus.ACCEPTED)
        self.assertIsNot(EventStatus.CERTAIN, EmissionStatus.PRODUCED)

    def test_top_level_rejects_unknown_presence_type(self) -> None:
        event = self._event()
        comparison = self._comparison(event)
        comparison_values = {
            item.name: getattr(comparison, item.name)
            for item in fields(FrameComparison)
        }
        comparison_values["from_type_presence"] = {
            **dict(comparison.from_type_presence),
            "type_999": TypePresenceSummary(
                type_id="type_999",
                candidate_ids=("candidate_999",),
                firm_count=1,
                possible_count=1,
            ),
        }
        invalid_comparison = FrameComparison(**comparison_values)
        with self.assertRaisesRegex(ValueError, "presence summaries"):
            self._minimal_top_result(event, invalid_comparison)

    def test_top_level_rejects_unresolved_event_match_evidence(self) -> None:
        event = self._event()
        evidence_values = {
            item.name: getattr(event.evidence, item.name)
            for item in fields(EventEvidence)
        }
        evidence_values["uncertain_match_ids"] = ("match_999",)
        event_values = {item.name: getattr(event, item.name) for item in fields(ChangeEvent)}
        event_values["evidence"] = EventEvidence(**evidence_values)
        invalid_event = ChangeEvent(**event_values)
        comparison = self._comparison(invalid_event)
        with self.assertRaisesRegex(ValueError, "uncertain-match evidence"):
            self._minimal_top_result(invalid_event, comparison)

    def test_compact_top_level_result_has_no_embedding_gt_or_metrics(self) -> None:
        event = self._event()
        comparison = self._comparison(event)
        summary = RepresentationSummary(
            representation_record_id="representation_001",
            candidate_id="candidate_001",
            frame_id="frame_001",
            family=RepresentationFamily.DINO_V2,
            representation_type="dino_embedding",
            representation_version="1.0",
            input_variant="bbox_crop",
            semantic_config_digest="sha256:semantic",
            validity_status=ValidityStatus.VALID,
            embedding_dimension=7,
        )
        result = StreamAnalysisResult(
            envelope=_envelope("run_001", "pipeline"),
            run_id="run_001",
            pipeline_version="1.0",
            run_status=RunStatus.COMPLETED_WITH_WARNINGS,
            frame_ids=("frame_001", "frame_002"),
            data_provenance=VersionedMetadata(identifier="probe_stream", version="1.0"),
            model_provenance=(
                VersionedMetadata(identifier="dinov2_model", version="source_revision_1"),
            ),
            runtime_summary=RuntimeMetadata(runtime_id="runtime_001", details={"device": "cpu"}),
            candidate_extraction_result_id="candidate_result_001",
            candidate_record_ids=("candidate_002", "candidate_001"),
            representation_summaries=(summary,),
            frame_matching_result_ids=("matching_result_001",),
            grouping_result_id="grouping_result_001",
            recurring_type_ids=("type_001",),
            type_assignment_ids=("assignment_001", "assignment_002"),
            frame_comparisons=(comparison,),
            change_events=(event,),
            artifacts=(
                ArtifactReference(
                    artifact_id="artifact_001",
                    artifact_kind="overlay",
                    reference="outputs/overlay.png",
                    producer_record_id="run_001",
                ),
            ),
            status_summary={"warning_count": 1},
        )
        self.assertEqual(result.representation_summaries[0].embedding_dimension, 7)
        self.assertFalse(hasattr(result.representation_summaries[0], "embedding"))
        top_fields = {item.name for item in fields(StreamAnalysisResult)}
        forbidden = {"annotation", "gt_mapping", "evaluation_metrics", "metrics", "embeddings"}
        self.assertTrue(top_fields.isdisjoint(forbidden))
        with self.assertRaises(TypeError):
            StreamAnalysisResult(  # type: ignore[call-arg]
                envelope=_envelope("run_002", "pipeline"),
                run_id="run_002",
                pipeline_version="1.0",
                run_status=RunStatus.FAILED,
                frame_ids=(),
                data_provenance=VersionedMetadata(identifier="probe_stream", version="1.0"),
                model_provenance=(),
                runtime_summary=RuntimeMetadata(runtime_id="runtime_002"),
                metrics={},
            )

    def test_primary_stage_contracts_have_no_annotation_or_gt_fields(self) -> None:
        primary_contracts = (
            CandidateRecord,
            RepresentationRecord,
            PairwiseScoreRecord,
            MatchRecord,
            TypeAssignmentRecord,
            ChangeEvent,
            StreamAnalysisResult,
        )
        for contract_type in primary_contracts:
            with self.subTest(contract=contract_type.__name__):
                names = {item.name.lower() for item in fields(contract_type)}
                self.assertFalse(any("annotation" in name or name.startswith("gt") for name in names))

    def test_public_imports_are_available_from_both_package_boundaries(self) -> None:
        public_names = (
            "CandidateRecord",
            "RepresentationRecord",
            "PairwiseScoreRecord",
            "FrameMatchingResult",
            "GroupingResult",
            "ChangeEvent",
            "StreamAnalysisResult",
            "RunStatus",
        )
        for name in public_names:
            with self.subTest(name=name):
                self.assertIs(getattr(stream_analysis, name), getattr(contracts, name))
                self.assertIn(name, stream_analysis.__all__)
                self.assertIn(name, contracts.__all__)


if __name__ == "__main__":
    unittest.main()
