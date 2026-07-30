import itertools
import unittest

import numpy as np

from stream_analysis.candidates.aggregate_resolution import (
    AggregateResolutionReason,
    PositionedCandidateMask,
    resolve_aggregate_masks,
)


def _mask(
    candidate_id: str,
    shape: tuple[int, int],
    *,
    origin: tuple[int, int] = (0, 0),
    filled: bool = True,
) -> PositionedCandidateMask:
    return PositionedCandidateMask(
        candidate_id=candidate_id,
        frame_id="frame_001",
        mask=np.full(shape, filled, dtype=np.bool_),
        origin_x=origin[0],
        origin_y=origin[1],
    )


class AggregateResolutionTest(unittest.TestCase):
    def test_removes_global_aggregate_and_keeps_ordinary_children(self) -> None:
        aggregate = _mask("aggregate", (10, 10))
        child_a = _mask("child_a", (2, 2), origin=(1, 1))
        child_b = _mask("child_b", (2, 2), origin=(5, 5))
        child_half_covered = _mask("child_c", (2, 4), origin=(8, 2))

        result = resolve_aggregate_masks(
            (child_b, aggregate, child_half_covered, child_a)
        )

        self.assertEqual(result.removed_candidate_ids, ("aggregate",))
        self.assertEqual(result.kept_candidate_ids, ("child_a", "child_b", "child_c"))
        decision = result.decision_for("aggregate")
        self.assertEqual(
            decision.reason,
            AggregateResolutionReason.REMOVED_AGGREGATE_MASK,
        )
        self.assertEqual(decision.covered_mask_count, 3)
        self.assertEqual(
            tuple(evidence.candidate_id for evidence in decision.covered_masks),
            ("child_a", "child_b", "child_c"),
        )
        self.assertEqual(decision.covered_masks[-1].containment, 0.5)

    def test_removes_parent_and_nested_aggregate_simultaneously(self) -> None:
        root = _mask("root", (12, 12))
        nested = _mask("nested", (8, 8), origin=(2, 2))
        children = (
            _mask("child_a", (2, 2), origin=(3, 3)),
            _mask("child_b", (2, 2), origin=(6, 3)),
            _mask("child_c", (2, 2), origin=(3, 6)),
        )

        result = resolve_aggregate_masks((children[1], nested, root, children[2], children[0]))

        self.assertEqual(result.removed_candidate_ids, ("nested", "root"))
        self.assertEqual(result.kept_candidate_ids, ("child_a", "child_b", "child_c"))
        self.assertEqual(result.decision_for("nested").covered_mask_count, 3)
        self.assertEqual(result.decision_for("root").covered_mask_count, 4)

    def test_default_minimum_three_keeps_candidate_that_covers_only_two(self) -> None:
        aggregate_candidate = _mask("candidate", (8, 8))
        child_a = _mask("child_a", (2, 2), origin=(1, 1))
        child_b = _mask("child_b", (2, 2), origin=(5, 5))

        safe_result = resolve_aggregate_masks((aggregate_candidate, child_a, child_b))
        explicit_two_result = resolve_aggregate_masks(
            (aggregate_candidate, child_a, child_b),
            minimum_covered_masks=2,
        )

        safe_decision = safe_result.decision_for("candidate")
        self.assertFalse(safe_decision.removed)
        self.assertEqual(
            safe_decision.reason,
            AggregateResolutionReason.KEPT_BELOW_MINIMUM,
        )
        self.assertEqual(safe_decision.covered_mask_count, 2)
        self.assertEqual(explicit_two_result.removed_candidate_ids, ("candidate",))

    def test_missing_and_empty_masks_are_kept_and_never_counted_as_covered(self) -> None:
        candidate = _mask("candidate", (8, 8))
        child = _mask("child", (2, 2), origin=(1, 1))
        missing = PositionedCandidateMask(
            candidate_id="missing",
            frame_id="frame_001",
            mask=None,
        )
        empty = _mask("empty", (3, 3), origin=(2, 2), filled=False)

        result = resolve_aggregate_masks((missing, child, empty, candidate))

        self.assertEqual(result.removed_candidate_ids, ())
        self.assertEqual(result.skipped_candidate_ids, ("empty", "missing"))
        self.assertEqual(
            result.decision_for("empty").reason,
            AggregateResolutionReason.KEPT_EMPTY_MASK,
        )
        self.assertEqual(
            result.decision_for("missing").reason,
            AggregateResolutionReason.KEPT_MISSING_MASK,
        )
        self.assertEqual(result.decision_for("candidate").covered_mask_count, 1)

    def test_result_is_independent_of_candidate_input_order(self) -> None:
        candidates = (
            _mask("root", (10, 10)),
            _mask("child_a", (2, 2), origin=(1, 1)),
            _mask("child_b", (2, 2), origin=(4, 4)),
            _mask("child_c", (2, 2), origin=(7, 7)),
        )
        expected = resolve_aggregate_masks(candidates)

        for permutation in itertools.permutations(candidates):
            with self.subTest(order=tuple(item.candidate_id for item in permutation)):
                self.assertEqual(resolve_aggregate_masks(permutation), expected)


if __name__ == "__main__":
    unittest.main()
