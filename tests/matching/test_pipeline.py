import unittest
from dataclasses import replace

from stream_analysis import (
    BBox, EmissionStatus, EventKind, EventStatus, FramePair, ImageSize,
    HandcraftedFeatureGroup, HandcraftedPayload, InputQualityMetadata,
    MatchDecisionStatus, MembershipStatus, RecordEnvelope, RepresentationFamily,
    ValidityStatus,
)
from stream_analysis.matching import (
    EventConfig,
    GroupingConfig,
    MatchingConfig,
    PairScoringConfig,
    build_change_events,
    group_recurring_visual_types,
    match_neighboring_frames,
)

try:
    from ._fixtures import TableScorer, candidate, representation
except ImportError:
    from _fixtures import TableScorer, candidate, representation


def pair(index):
    return FramePair(
        from_frame_id=f"frame_{index:03d}",
        from_frame_index=index,
        from_frame_size=ImageSize(100, 100),
        to_frame_id=f"frame_{index + 1:03d}",
        to_frame_index=index + 1,
        to_frame_size=ImageSize(100, 100),
    )


def scoring_config(scorer):
    return PairScoringConfig(
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.scorer_version,
        visual_gate=0.8,
        spatial_gate=None,
    )


def matching_config(local=0.01, global_=0.01):
    return MatchingConfig(
        unmatched_pair_cost=0.7,
        local_margin_gate=local,
        global_margin_gate=global_,
        severe_quality_flags=("POSSIBLE_SPLITTING", "POSSIBLE_MERGING"),
    )


def candidate_box(
    candidate_id: str,
    frame_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill_ratio: float = 1.0,
    hole_count: int = 0,
):
    item = candidate(candidate_id, frame_index, x, y)
    bbox = BBox(x, y, width, height)
    return replace(
        item,
        bbox=bbox,
        center=bbox.center,
        geometry=replace(
            item.geometry,
            values={
                "area": width * height * fill_ratio,
                "bbox_area": width * height,
                "aspect_ratio": width / height,
                "foreground_fill_ratio": fill_ratio,
                "hole_count": hole_count,
            },
        ),
    )


def score_table(*rows):
    return {tuple(sorted((left, right))): value for left, right, value in rows}


def handcrafted_representation(item):
    record = representation(item)
    return replace(
        record,
        family=RepresentationFamily.HANDCRAFTED,
        representation_type="fixture_handcrafted",
        payload=HandcraftedPayload(
            feature_schema_id="fixture_handcrafted_v1",
            feature_groups=(
                HandcraftedFeatureGroup(group_name="shape", values={"value": 1.0}),
            ),
        ),
        model_metadata=None,
    )


def grouping_config(*, coframe_visual_gate=None, coframe_margin_bypass=False):
    return GroupingConfig(
        medoid_gate=0.8,
        support_quantile=0.25,
        quantile_gate=0.8,
        support_pair_gate=0.8,
        support_ratio_gate=0.75,
        visual_gate=0.8,
        medoid_weight=0.4,
        quantile_weight=0.4,
        support_ratio_weight=0.2,
        second_best_margin=0.01,
        representation_variant_id="fixture_variant",
        scorer_id="table_scorer_v1",
        scorer_version="1.0.0",
        coframe_visual_gate=coframe_visual_gate,
        coframe_margin_bypass=coframe_margin_bypass,
    )


class NeighboringMatchingTest(unittest.TestCase):
    def test_empty_comparison_preserves_explicit_stream_identity(self):
        scorer = TableScorer({})
        result = match_neighboring_frames(
            stream_id="stream", frame_pair=pair(0), from_candidates=(), to_candidates=(),
            representations=(), representation_variant_id="fixture_variant", scorer=scorer,
            scoring_config=scoring_config(scorer), matching_config=matching_config(),
        ).result
        self.assertEqual(result.envelope.stream_id, "stream")
        self.assertEqual(result.matrix_summary["matrix_order"], 0)

    def test_invalid_endpoint_is_detected_when_opposite_frame_is_empty(self):
        item = candidate("invalid", 0, 10)
        record = representation(item)
        record = replace(
            record,
            envelope=RecordEnvelope(
                record_id=record.envelope.record_id,
                schema_version=record.envelope.schema_version,
                stream_id="stream",
                producer=record.envelope.producer,
                context=record.envelope.context,
                validity_status=ValidityStatus.INVALID,
                error_ids=("fixture_representation_error",),
            ),
            payload=None,
        )
        scorer = TableScorer({})
        batch = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0), from_candidates=(item,), to_candidates=(),
            representations=(record,), representation_variant_id="fixture_variant",
            scorer=scorer, scoring_config=scoring_config(scorer),
            matching_config=matching_config(),
        )
        self.assertEqual(batch.result.envelope.validity_status, ValidityStatus.INVALID)
        self.assertEqual(batch.result.pairwise_scores, ())
        self.assertEqual(batch.result.unmatched[0].envelope.validity_status, ValidityStatus.INVALID)
        self.assertEqual(batch.result.unmatched[0].reason_code, "INVALID_REPRESENTATION")
        self.assertEqual(batch.errors[0].code, "REPRESENTATION_PREREQUISITE_INVALID")

    def test_uncertain_match_reserves_endpoints_and_is_not_unmatched(self):
        left = candidate("left", 0, 10)
        right_a = candidate("right_a", 1, 10)
        right_b = candidate("right_b", 1, 20)
        scorer = TableScorer({("left", "right_a"): 0.95, ("left", "right_b"): 0.95})
        batch = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(left,),
            to_candidates=(right_a, right_b),
            representations=tuple(representation(item) for item in (left, right_a, right_b)),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(),
        )
        self.assertEqual(batch.result.selected_matches[0].status, MatchDecisionStatus.UNCERTAIN)
        reserved = {
            batch.result.selected_matches[0].from_endpoint.candidate_id,
            batch.result.selected_matches[0].to_endpoint.candidate_id,
        }
        self.assertTrue(reserved.isdisjoint(item.endpoint.candidate_id for item in batch.result.unmatched))
        self.assertEqual(len(batch.result.unmatched), 1)

    def test_deterministic_result_is_independent_of_input_order(self):
        left = (candidate("a", 0, 10), candidate("b", 0, 30))
        right = (candidate("c", 1, 10), candidate("d", 1, 30))
        scorer = TableScorer({("a", "c"): 0.95, ("b", "d"): 0.95})
        kwargs = dict(
            frame_pair=pair(0), representation_variant_id="fixture_variant", scorer=scorer,
            scoring_config=scoring_config(scorer), matching_config=matching_config(),
        )
        first = match_neighboring_frames(
            stream_id="stream",
            from_candidates=left, to_candidates=right,
            representations=tuple(representation(item) for item in (*left, *right)), **kwargs,
        ).result
        second = match_neighboring_frames(
            stream_id="stream",
            from_candidates=tuple(reversed(left)), to_candidates=tuple(reversed(right)),
            representations=tuple(reversed(tuple(representation(item) for item in (*left, *right)))), **kwargs,
        ).result
        self.assertEqual(first, second)

    def test_representation_quality_flag_makes_selected_match_uncertain(self):
        left = candidate("left", 0, 10)
        right = candidate("right", 1, 10)
        left_rep = replace(
            representation(left),
            input_quality_metadata=InputQualityMetadata(
                quality_flags=("POSSIBLE_SPLITTING",)
            ),
        )
        scorer = TableScorer({("left", "right"): 0.99})
        result = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0), from_candidates=(left,), to_candidates=(right,),
            representations=(left_rep, representation(right)),
            representation_variant_id="fixture_variant", scorer=scorer,
            scoring_config=scoring_config(scorer), matching_config=matching_config(),
        ).result
        self.assertEqual(result.selected_matches[0].status, MatchDecisionStatus.UNCERTAIN)


class GroupingTest(unittest.TestCase):
    def test_grouping_digest_and_validation_include_scorer_identity(self):
        base = grouping_config()
        changed = replace(base, scorer_version="9.9.9")
        self.assertNotEqual(base.config_digest, changed.config_digest)

        item = candidate("single", 0, 10)
        incompatible = replace(
            TableScorer({}), family=RepresentationFamily.HANDCRAFTED
        )
        incompatible_config = replace(
            base,
            scorer_id=incompatible.scorer_id,
            scorer_version=incompatible.scorer_version,
        )
        with self.assertRaisesRegex(ValueError, "family"):
            group_recurring_visual_types(
                stream_id="stream", candidates=(item,),
                representations=(representation(item),), matching_results=(),
                representation_variant_id="fixture_variant", scorer=incompatible,
                config=incompatible_config,
            )

    def test_cross_member_support_blocks_single_link_bridge(self):
        items = (
            candidate("a", 0, 10), candidate("b", 1, 10), candidate("c", 2, 10),
        )
        scorer = TableScorer({("a", "b"): 0.95, ("b", "c"): 0.90, ("a", "c"): 0.1})
        result = group_recurring_visual_types(
            stream_id="stream", candidates=iter(items),
            representations=tuple(representation(item) for item in items), matching_results=(),
            representation_variant_id="fixture_variant", scorer=scorer, config=grouping_config(),
        )
        member_sets = {item.member_candidate_ids for item in result.recurring_types}
        self.assertIn(("a", "b"), member_sets)
        self.assertIn(("c",), member_sets)

    def test_reappearance_can_merge_without_physical_instance_tracking(self):
        items = (candidate("cup_first", 0, 10), candidate("cup_return", 2, 70))
        scorer = TableScorer({("cup_first", "cup_return"): 0.98})
        result = group_recurring_visual_types(
            stream_id="stream", candidates=items,
            representations=tuple(representation(item) for item in items), matching_results=(),
            representation_variant_id="fixture_variant", scorer=scorer, config=grouping_config(),
        )
        self.assertEqual(len(result.recurring_types), 1)
        self.assertEqual(result.recurring_types[0].member_candidate_ids, ("cup_first", "cup_return"))
        self.assertFalse(hasattr(result.recurring_types[0], "physical_instance_id"))

    def test_ambiguous_singletons_remain_provisional(self):
        items = (
            candidate("a", 0, 10), candidate("b", 1, 10), candidate("c", 2, 10),
        )
        scorer = TableScorer({("a", "b"): 0.95, ("a", "c"): 0.94, ("b", "c"): 0.93})
        config = replace(grouping_config(), second_best_margin=0.1)
        result = group_recurring_visual_types(
            stream_id="stream", candidates=items,
            representations=tuple(representation(item) for item in items), matching_results=(),
            representation_variant_id="fixture_variant", scorer=scorer, config=config,
        )
        self.assertTrue(
            all(item.membership_status is MembershipStatus.PROVISIONAL for item in result.assignments)
        )

    def test_unmerged_recurrent_components_do_not_expose_assignment_alternatives_by_default(self):
        items = (
            candidate("a0", 0, 10), candidate("a1", 1, 10),
            candidate("b0", 0, 60), candidate("b1", 1, 60),
            candidate("c0", 0, 80), candidate("c1", 1, 80),
        )
        scorer = TableScorer({
            ("a0", "a1"): 0.99,
            ("b0", "b1"): 0.99,
            ("c0", "c1"): 0.99,
            ("a0", "b0"): 0.94, ("a0", "b1"): 0.94,
            ("a1", "b0"): 0.94, ("a1", "b1"): 0.94,
            ("a0", "c0"): 0.936, ("a0", "c1"): 0.936,
            ("a1", "c0"): 0.936, ("a1", "c1"): 0.936,
            ("b0", "c0"): 0.932, ("b0", "c1"): 0.932,
            ("b1", "c0"): 0.932, ("b1", "c1"): 0.932,
        })
        reps = tuple(representation(item) for item in items)
        match = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(items[0], items[2], items[4]),
            to_candidates=(items[1], items[3], items[5]),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        result = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(match,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(),
        )
        self.assertEqual(len(result.recurring_types), 3)
        self.assertTrue(
            all(item.membership_status is MembershipStatus.FIRM for item in result.assignments)
        )
        self.assertTrue(
            all(item.alternative_type_ids == () for item in result.assignments)
        )
        self.assertTrue(
            all(item.scores["diagnostic_alternative_type_count"] >= 1 for item in result.assignments)
        )

    def test_unmerged_alternatives_can_be_exposed_as_uncertain_diagnostic_mode(self):
        items = (
            candidate("a0", 0, 10), candidate("a1", 1, 10),
            candidate("b0", 0, 60), candidate("b1", 1, 60),
            candidate("c0", 0, 80), candidate("c1", 1, 80),
        )
        scorer = TableScorer({
            ("a0", "a1"): 0.99,
            ("b0", "b1"): 0.99,
            ("c0", "c1"): 0.99,
            ("a0", "b0"): 0.94, ("a0", "b1"): 0.94,
            ("a1", "b0"): 0.94, ("a1", "b1"): 0.94,
            ("a0", "c0"): 0.936, ("a0", "c1"): 0.936,
            ("a1", "c0"): 0.936, ("a1", "c1"): 0.936,
            ("b0", "c0"): 0.932, ("b0", "c1"): 0.932,
            ("b1", "c0"): 0.932, ("b1", "c1"): 0.932,
        })
        reps = tuple(representation(item) for item in items)
        match = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(items[0], items[2], items[4]),
            to_candidates=(items[1], items[3], items[5]),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        result = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(match,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=replace(grouping_config(), expose_unmerged_alternatives=True),
        )
        self.assertTrue(
            all(item.membership_status is MembershipStatus.UNCERTAIN for item in result.assignments)
        )
        self.assertTrue(
            all(len(item.alternative_type_ids) >= 1 for item in result.assignments)
        )

    def test_severe_representation_quality_survives_robust_group_merge(self):
        items = (candidate("a", 0, 10), candidate("b", 2, 10))
        records = (
            replace(
                representation(items[0]),
                input_quality_metadata=InputQualityMetadata(
                    quality_flags=("POSSIBLE_MERGING",)
                ),
            ),
            representation(items[1]),
        )
        scorer = TableScorer({("a", "b"): 0.98})
        config = replace(grouping_config(), severe_quality_flags=("POSSIBLE_MERGING",))
        result = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=records, matching_results=(),
            representation_variant_id="fixture_variant", scorer=scorer, config=config,
        )
        self.assertTrue(
            all(item.membership_status is MembershipStatus.UNCERTAIN for item in result.assignments)
        )

    def test_coframe_geometry_blocks_square_rectangle_overmerge(self):
        square0 = candidate_box("square0", 0, 10, 10, 20, 20)
        square1 = candidate_box("square1", 1, 10, 10, 20, 20)
        rect0 = candidate_box("rect0", 0, 50, 10, 40, 20)
        rect1 = candidate_box("rect1", 1, 50, 10, 40, 20)
        items = (square0, square1, rect0, rect1)
        scorer = TableScorer(score_table(
            ("square0", "square1", 0.99),
            ("rect0", "rect1", 0.99),
            ("square0", "rect0", 0.98),
            ("square0", "rect1", 0.98),
            ("square1", "rect0", 0.98),
            ("square1", "rect1", 0.98),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(square0, rect0),
            to_candidates=(square1, rect1),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(matching,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(),
        )
        self.assertEqual(
            {item.member_candidate_ids for item in grouping.recurring_types},
            {("square0", "square1"), ("rect0", "rect1")},
        )

    def test_dino_coframe_visual_hard_negative_blocks_h02_like_overmerge(self):
        round0 = candidate_box("round0", 0, 10, 10, 20, 20)
        round1 = candidate_box("round1", 1, 10, 10, 20, 20)
        tapered0 = candidate_box("tapered0", 0, 50, 10, 24, 18)
        tapered1 = candidate_box("tapered1", 1, 50, 10, 24, 18)
        items = (round0, round1, tapered0, tapered1)
        scorer = TableScorer(score_table(
            ("round0", "round1", 0.99),
            ("tapered0", "tapered1", 0.99),
            ("round0", "tapered0", 0.94),
            ("round0", "tapered1", 0.94),
            ("round1", "tapered0", 0.94),
            ("round1", "tapered1", 0.94),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(round0, tapered0),
            to_candidates=(round1, tapered1),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result

        legacy = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=reps,
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(),
        )
        corrected = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=reps,
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(coframe_visual_gate=0.95),
        )

        self.assertEqual(len(legacy.recurring_types), 1)
        self.assertEqual(
            {item.member_candidate_ids for item in corrected.recurring_types},
            {("round0", "round1"), ("tapered0", "tapered1")},
        )
        self.assertNotEqual(
            grouping_config().config_digest,
            grouping_config(
                coframe_visual_gate=0.95,
                coframe_margin_bypass=True,
            ).config_digest,
        )

    def test_dino_hard_negative_keeps_appearance_scale_rotation_reappearance_stable(self):
        original = candidate_box("produce_original", 0, 10, 10, 20, 30)
        changed = candidate_box("produce_changed", 2, 60, 10, 30, 20)
        scorer = TableScorer({("produce_changed", "produce_original"): 0.97})
        result = group_recurring_visual_types(
            stream_id="stream",
            candidates=(original, changed),
            representations=(representation(original), representation(changed)),
            matching_results=(),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(
                coframe_visual_gate=0.963,
                coframe_margin_bypass=True,
            ),
        )
        self.assertEqual(
            {item.member_candidate_ids for item in result.recurring_types},
            {("produce_changed", "produce_original")},
        )

    def test_dino_opt_in_policy_leaves_handcrafted_family_on_legacy_path(self):
        a0 = candidate_box("a0", 0, 10, 10, 20, 20)
        a1 = candidate_box("a1", 1, 10, 10, 20, 20)
        b0 = candidate_box("b0", 0, 50, 10, 24, 18)
        b1 = candidate_box("b1", 1, 50, 10, 24, 18)
        items = (a0, a1, b0, b1)
        scorer = replace(
            TableScorer(score_table(
                ("a0", "a1", 0.99), ("b0", "b1", 0.99),
                ("a0", "b0", 0.94), ("a0", "b1", 0.94),
                ("a1", "b0", 0.94), ("a1", "b1", 0.94),
            )),
            family=RepresentationFamily.HANDCRAFTED,
        )
        reps = tuple(handcrafted_representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream", frame_pair=pair(0),
            from_candidates=(a0, b0), to_candidates=(a1, b1),
            representations=reps, representation_variant_id="fixture_variant",
            scorer=scorer, scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        result = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=reps,
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(),
        )
        self.assertEqual(len(result.recurring_types), 1)

    def test_uncertain_spatially_stable_multi_instance_matches_seed_one_visual_type(self):
        w10 = candidate_box("washer1_f0", 0, 10, 10, 20, 20)
        w11 = candidate_box("washer1_f1", 1, 10, 10, 20, 20)
        w20 = candidate_box("washer2_f0", 0, 60, 10, 20, 20)
        w21 = candidate_box("washer2_f1", 1, 60, 10, 20, 20)
        items = (w10, w11, w20, w21)
        scorer = TableScorer(score_table(
            ("washer1_f0", "washer1_f1", 0.99),
            ("washer2_f0", "washer2_f1", 0.99),
            ("washer1_f0", "washer2_f1", 0.99),
            ("washer2_f0", "washer1_f1", 0.99),
            ("washer1_f0", "washer2_f0", 0.99),
            ("washer1_f1", "washer2_f1", 0.99),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(w10, w20),
            to_candidates=(w11, w21),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        self.assertTrue(all(item.status is MatchDecisionStatus.UNCERTAIN for item in matching.selected_matches))
        grouping = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(matching,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(
                coframe_visual_gate=0.963,
                coframe_margin_bypass=True,
            ),
        )
        self.assertEqual(len(grouping.recurring_types), 1)
        self.assertEqual(grouping.recurring_types[0].count_by_frame["frame_000"], 2)
        self.assertEqual(grouping.recurring_types[0].count_by_frame["frame_001"], 2)
        self.assertTrue(all(item.membership_status is MembershipStatus.FIRM for item in grouping.assignments))

    def test_reappearance_and_new_instance_do_not_create_uncontrolled_type_proliferation(self):
        w0 = candidate_box("washer_f0", 0, 10, 10, 20, 20)
        w2a = candidate_box("washer_a_f2", 2, 10, 10, 20, 20)
        w2b = candidate_box("washer_b_f2", 2, 60, 10, 20, 20)
        w3a = candidate_box("washer_a_f3", 3, 10, 10, 20, 20)
        w3b = candidate_box("washer_b_f3", 3, 60, 10, 20, 20)
        items = (w0, w2a, w2b, w3a, w3b)
        scorer = TableScorer(score_table(
            ("washer_f0", "washer_a_f2", 0.99),
            ("washer_f0", "washer_b_f2", 0.90),
            ("washer_a_f2", "washer_a_f3", 0.99),
            ("washer_b_f2", "washer_b_f3", 0.99),
            ("washer_a_f2", "washer_b_f2", 0.99),
            ("washer_a_f3", "washer_b_f3", 0.99),
            ("washer_a_f2", "washer_b_f3", 0.99),
            ("washer_b_f2", "washer_a_f3", 0.99),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(2),
            from_candidates=(w2a, w2b),
            to_candidates=(w3a, w3b),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        result = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(matching,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(
                coframe_visual_gate=0.963,
                coframe_margin_bypass=True,
            ),
        )
        self.assertLessEqual(len(result.recurring_types), 3)
        member_sets = {item.member_candidate_ids for item in result.recurring_types}
        self.assertIn(
            ("washer_a_f2", "washer_a_f3", "washer_b_f2", "washer_b_f3"),
            member_sets,
        )

    def test_p04_like_washer_plate_and_gear_case_keeps_stable_visual_types(self):
        square0 = candidate_box("square0", 0, 10, 10, 20, 20, fill_ratio=0.94, hole_count=4)
        square1 = candidate_box("square1", 1, 10, 10, 20, 20, fill_ratio=0.94, hole_count=4)
        rect0 = candidate_box("rect0", 0, 50, 10, 40, 20)
        rect1 = candidate_box("rect1", 1, 50, 10, 40, 20)
        washer0 = candidate_box("washer0", 0, 10, 60, 20, 20)
        washer1a = candidate_box("washer1a", 1, 10, 60, 20, 20)
        washer1b = candidate_box("washer1b", 1, 60, 60, 20, 20)
        gear0 = candidate_box("gear0", 0, 65, 60, 32, 32)
        gear1 = candidate_box("gear1", 1, 65, 60, 32, 32)
        triangle0 = candidate_box("triangle0", 0, 35, 60, 22, 20, fill_ratio=0.44, hole_count=1)
        triangle1 = candidate_box("triangle1", 1, 35, 60, 22, 20, fill_ratio=0.44, hole_count=1)
        items = (square0, square1, rect0, rect1, washer0, washer1a, washer1b, gear0, gear1, triangle0, triangle1)
        scorer = TableScorer(score_table(
            ("square0", "square1", 0.99),
            ("triangle0", "triangle1", 0.99),
            ("square0", "triangle0", 0.98), ("square0", "triangle1", 0.98),
            ("square1", "triangle0", 0.98), ("square1", "triangle1", 0.98),
            ("rect0", "rect1", 0.99),
            ("square0", "rect0", 0.98), ("square0", "rect1", 0.98),
            ("square1", "rect0", 0.98), ("square1", "rect1", 0.98),
            ("washer0", "washer1a", 1.0),
            ("washer0", "washer1b", 1.0),
            ("washer1a", "washer1b", 1.0),
            ("gear0", "gear1", 0.99),
            ("washer0", "gear0", 0.97), ("washer0", "gear1", 0.97),
            ("washer1a", "gear0", 0.97), ("washer1a", "gear1", 0.97),
            ("washer1b", "gear0", 0.97), ("washer1b", "gear1", 0.97),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(square0, rect0, washer0, gear0, triangle0),
            to_candidates=(square1, rect1, washer1a, washer1b, gear1, triangle1),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(matching,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(
                coframe_visual_gate=0.963,
                coframe_margin_bypass=True,
            ),
        )
        member_sets = {item.member_candidate_ids for item in grouping.recurring_types}
        self.assertIn(("square0", "square1"), member_sets)
        self.assertIn(("triangle0", "triangle1"), member_sets)
        self.assertIn(("rect0", "rect1"), member_sets)
        self.assertIn(("gear0", "gear1"), member_sets)
        washer_type = next(item for item in grouping.recurring_types if "washer0" in item.member_candidate_ids)
        self.assertEqual(washer_type.count_by_frame["frame_001"], 2)


class EventIntegrationTest(unittest.TestCase):
    def test_dino_coframe_hard_negative_preserves_count_changed(self):
        a0 = candidate_box("a0", 0, 10, 10, 20, 20)
        b0 = candidate_box("b0", 0, 70, 10, 24, 18)
        a1 = candidate_box("a1", 1, 10, 10, 20, 20)
        a1_extra = candidate_box("a1_extra", 1, 35, 10, 20, 20)
        b1 = candidate_box("b1", 1, 70, 10, 24, 18)
        items = (a0, b0, a1, a1_extra, b1)
        scorer = TableScorer(score_table(
            ("a0", "a1", 0.99), ("a0", "a1_extra", 0.99),
            ("a1", "a1_extra", 0.99), ("b0", "b1", 0.99),
            ("a0", "b0", 0.94), ("a0", "b1", 0.94),
            ("a1", "b0", 0.94), ("a1", "b1", 0.94),
            ("a1_extra", "b0", 0.94), ("a1_extra", "b1", 0.94),
        ))
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream", frame_pair=pair(0),
            from_candidates=(a0, b0), to_candidates=(a1, a1_extra, b1),
            representations=reps, representation_variant_id="fixture_variant",
            scorer=scorer, scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=reps,
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(coframe_visual_gate=0.95),
        )
        batch = build_change_events(
            stream_id="stream", candidates=items, grouping=grouping,
            matching_results=(matching,),
            config=EventConfig(position_threshold_norm=0.2, severe_quality_flags=()),
        )
        count_events = [item for item in batch.events if item.kind is EventKind.COUNT_CHANGED]
        self.assertEqual(len(count_events), 1)
        self.assertEqual(
            (count_events[0].evidence.from_count, count_events[0].evidence.to_count),
            (1, 2),
        )

    def test_all_five_events_multiple_instances_and_position_policy(self):
        a0 = candidate("a0", 0, 10)
        a1 = candidate("a1", 1, 70)
        b1 = candidate("b1", 1, 10)
        b2a = candidate("b2a", 2, 10)
        b2b = candidate("b2b", 2, 30)
        items = (a0, a1, b1, b2a, b2b)
        scores = {
            ("a0", "a1"): 0.99, ("a0", "b1"): 0.1,
            ("a1", "b2a"): 0.1, ("a1", "b2b"): 0.1,
            ("b1", "b2a"): 0.99, ("b1", "b2b"): 0.97,
            ("b2a", "b2b"): 0.98,
            ("a0", "b2a"): 0.1, ("a0", "b2b"): 0.1,
            ("a1", "b1"): 0.1,
        }
        scorer = TableScorer(scores)
        reps = tuple(representation(item) for item in items)
        match01 = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0), from_candidates=(a0,), to_candidates=(a1, b1),
            representations=reps, representation_variant_id="fixture_variant", scorer=scorer,
            scoring_config=scoring_config(scorer), matching_config=matching_config(0.005, 0.005),
        ).result
        match12 = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(1), from_candidates=(a1, b1), to_candidates=(b2a, b2b),
            representations=reps, representation_variant_id="fixture_variant", scorer=scorer,
            scoring_config=scoring_config(scorer), matching_config=matching_config(0.005, 0.005),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream", candidates=items, representations=reps,
            matching_results=(match01, match12), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(),
        )
        batch = build_change_events(
            stream_id="stream", candidates=items, grouping=grouping,
            matching_results=(match01, match12),
            config=EventConfig(position_threshold_norm=0.2, severe_quality_flags=("POSSIBLE_SPLITTING",)),
        )
        kinds = {item.kind for item in batch.events}
        self.assertEqual(kinds, set(EventKind))
        count_event = next(item for item in batch.events if item.kind is EventKind.COUNT_CHANGED)
        self.assertEqual((count_event.evidence.from_count, count_event.evidence.to_count), (1, 2))
        position = next(item for item in batch.events if item.kind is EventKind.POSITION_CHANGED)
        self.assertAlmostEqual(position.evidence.position_shift_norm, 0.6)
        self.assertEqual(position.evidence.details["position_threshold_norm"], 0.2)
        self.assertEqual(position.evidence.details["from_centroid_norm"], (0.15, 0.15))
        self.assertEqual(position.evidence.details["to_centroid_norm"], (0.75, 0.15))
        self.assertTrue(all(item.status is EventStatus.CERTAIN for item in batch.events))

    def test_provisional_membership_counts_but_makes_event_uncertain(self):
        item = candidate("single", 1, 10)
        scorer = TableScorer({})
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0), from_candidates=(), to_candidates=(item,),
            representations=(representation(item),), representation_variant_id="fixture_variant",
            scorer=scorer, scoring_config=scoring_config(scorer), matching_config=matching_config(),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream", candidates=(item,), representations=(representation(item),),
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(),
        )
        batch = build_change_events(
            stream_id="stream", candidates=(item,), grouping=grouping, matching_results=(matching,),
            config=EventConfig(position_threshold_norm=0.2, severe_quality_flags=()),
        )
        self.assertEqual(batch.events[0].kind, EventKind.APPEARED)
        self.assertEqual(batch.events[0].status, EventStatus.UNCERTAIN)
        self.assertEqual(batch.events[0].evidence.to_count, 1)

    def test_recurrent_near_merge_diagnostics_do_not_suppress_event_recall(self):
        a0 = candidate("a0", 0, 10)
        a1 = candidate("a1", 1, 10)
        b0 = candidate("b0", 0, 60)
        b1 = candidate("b1", 1, 60)
        c0 = candidate("c0", 0, 80)
        c1 = candidate("c1", 1, 80)
        items = (a0, a1, b0, b1, c0, c1)
        scorer = TableScorer({
            ("a0", "a1"): 0.99,
            ("b0", "b1"): 0.99,
            ("c0", "c1"): 0.99,
            ("a0", "b0"): 0.94, ("a0", "b1"): 0.94,
            ("a1", "b0"): 0.94, ("a1", "b1"): 0.94,
            ("a0", "c0"): 0.936, ("a0", "c1"): 0.936,
            ("a1", "c0"): 0.936, ("a1", "c1"): 0.936,
            ("b0", "c0"): 0.932, ("b0", "c1"): 0.932,
            ("b1", "c0"): 0.932, ("b1", "c1"): 0.932,
        })
        reps = tuple(representation(item) for item in items)
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0),
            from_candidates=(a0, b0, c0),
            to_candidates=(a1, b1, c1),
            representations=reps,
            representation_variant_id="fixture_variant",
            scorer=scorer,
            scoring_config=scoring_config(scorer),
            matching_config=matching_config(0.005, 0.005),
        ).result
        grouping = group_recurring_visual_types(
            stream_id="stream",
            candidates=items,
            representations=reps,
            matching_results=(matching,),
            representation_variant_id="fixture_variant",
            scorer=scorer,
            config=grouping_config(),
        )
        batch = build_change_events(
            stream_id="stream",
            candidates=items,
            grouping=grouping,
            matching_results=(matching,),
            config=EventConfig(position_threshold_norm=0.2, severe_quality_flags=()),
        )
        self.assertEqual({item.kind for item in batch.events}, {EventKind.PERSISTED})
        self.assertTrue(all(item.status is EventStatus.CERTAIN for item in batch.events))
        self.assertTrue(
            all(
                summary.candidate_ids
                for comparison in batch.comparisons
                for summary in comparison.from_type_presence.values()
                if summary.possible_count
            )
        )

    def test_invalid_representation_withholds_events_instead_of_false_absence(self):
        left = candidate("left", 0, 10)
        right = candidate("right", 1, 10)
        left_rep = representation(left)
        invalid_rep = representation(right)
        invalid_rep = replace(
            invalid_rep,
            envelope=RecordEnvelope(
                record_id=invalid_rep.envelope.record_id,
                schema_version=invalid_rep.envelope.schema_version,
                stream_id="stream",
                producer=invalid_rep.envelope.producer,
                context=invalid_rep.envelope.context,
                validity_status=ValidityStatus.INVALID,
                error_ids=("fixture_representation_error",),
            ),
            payload=None,
        )
        scorer = TableScorer({("left", "right"): 0.99})
        matching = match_neighboring_frames(
            stream_id="stream",
            frame_pair=pair(0), from_candidates=(left,), to_candidates=(right,),
            representations=(left_rep, invalid_rep), representation_variant_id="fixture_variant",
            scorer=scorer, scoring_config=scoring_config(scorer), matching_config=matching_config(),
        ).result
        self.assertEqual(matching.envelope.validity_status, ValidityStatus.INVALID)
        grouping = group_recurring_visual_types(
            stream_id="stream", candidates=(left, right), representations=(left_rep, invalid_rep),
            matching_results=(matching,), representation_variant_id="fixture_variant",
            scorer=scorer, config=grouping_config(),
        )
        batch = build_change_events(
            stream_id="stream", candidates=(left, right), grouping=grouping,
            matching_results=(matching,),
            config=EventConfig(position_threshold_norm=0.2, severe_quality_flags=()),
        )
        self.assertEqual(batch.events, ())
        self.assertEqual(batch.comparisons[0].emission_status, EmissionStatus.WITHHELD)
        self.assertEqual(len(batch.errors), 1)


if __name__ == "__main__":
    unittest.main()
