import itertools
import math
import unittest

import numpy as np

from stream_analysis.matching import MatchingConfig, solve_augmented_assignment


def exhaustive_cost(costs, unmatched_pair_cost):
    n, m = costs.shape
    best = float("inf")
    for count in range(min(n, m) + 1):
        for left in itertools.combinations(range(n), count):
            for right in itertools.combinations(range(m), count):
                for permutation in itertools.permutations(right):
                    values = [costs[i, j] for i, j in zip(left, permutation)]
                    if not all(np.isfinite(value) for value in values):
                        continue
                    total = sum(values) + (n - count + m - count) * unmatched_pair_cost / 2.0
                    best = min(best, total)
    return best


def exhaustive_lexicographic(costs, spatial, unmatched_pair_cost):
    """Reference enumeration with no scalar epsilon or assignment backend."""

    n, m = costs.shape
    order = n + m
    candidates = []
    for count in range(min(n, m) + 1):
        for left in itertools.combinations(range(n), count):
            for right in itertools.combinations(range(m), count):
                for permutation in itertools.permutations(right):
                    matches = tuple(sorted(zip(left, permutation)))
                    if not all(np.isfinite(costs[i, j]) for i, j in matches):
                        continue
                    primary = math.fsum(
                        [float(costs[i, j]) for i, j in matches]
                        + [unmatched_pair_cost / 2.0] * (n + m - 2 * count)
                    )
                    spatial_total = math.fsum(float(spatial[i, j]) for i, j in matches)
                    by_left = dict(matches)
                    canonical = tuple(by_left.get(i, m + i) for i in range(n))
                    candidates.append((primary, spatial_total, canonical, matches))
    primary_min = min(item[0] for item in candidates)
    tolerance = 8.0 * np.finfo(np.float64).eps * max(1, order) * max(1.0, abs(primary_min))
    primary_ties = [item for item in candidates if abs(item[0] - primary_min) <= tolerance]
    spatial_min = min(item[1] for item in primary_ties)
    spatial_tolerance = (
        8.0 * np.finfo(np.float64).eps * max(1, order) * max(1.0, abs(spatial_min))
    )
    spatial_ties = [item for item in primary_ties if abs(item[1] - spatial_min) <= spatial_tolerance]
    return min(spatial_ties, key=lambda item: item[2])


class AugmentedAssignmentTest(unittest.TestCase):
    def solve(self, costs, unmatched=0.8):
        values = np.asarray(costs, dtype=float)
        return solve_augmented_assignment(values, np.zeros_like(values), unmatched)

    def test_equal_cardinality_can_leave_every_endpoint_unmatched(self):
        result = self.solve([[np.inf, np.inf], [np.inf, np.inf]])
        self.assertEqual(result.matrix_order, 4)
        self.assertEqual(result.real_matches, ())
        self.assertEqual(len(result.unmatched_left), 2)
        self.assertEqual(len(result.unmatched_right), 2)

    def test_rectangular_empty_forbidden_dummy_and_tie_cases(self):
        cases = (
            np.empty((0, 0)),
            np.empty((0, 2)),
            np.empty((3, 0)),
            np.array([[0.1, np.inf, 0.2], [np.inf, 0.1, 0.2]]),
            np.array([[0.2, 0.2], [0.2, 0.2]]),
        )
        for costs in cases:
            with self.subTest(shape=costs.shape):
                result = solve_augmented_assignment(costs, np.zeros_like(costs), 0.8)
                self.assertAlmostEqual(result.primary_total_cost, exhaustive_cost(costs, 0.8))

    def test_equal_primary_and_spatial_costs_use_canonical_assignment(self):
        result = self.solve([[0.2, 0.2], [0.2, 0.2]])
        self.assertEqual(
            tuple((item.left_index, item.right_index) for item in result.real_matches),
            ((0, 0), (1, 1)),
        )

    def test_spatial_total_precedes_canonical_ids(self):
        costs = np.full((2, 2), 0.2)
        spatial = np.array([[0.8, 0.1], [0.1, 0.8]])
        result = solve_augmented_assignment(costs, spatial, 0.8)
        self.assertEqual(
            tuple((item.left_index, item.right_index) for item in result.real_matches),
            ((0, 1), (1, 0)),
        )

    def test_reported_decimal_tie_uses_lower_spatial_assignment(self):
        costs = np.array([[0.3, 0.2], [np.inf, 0.3]])
        spatial = np.array([[0.9, 0.5], [0.0, 0.5]])
        result = solve_augmented_assignment(costs, spatial, 0.4)
        self.assertEqual(
            tuple((item.left_index, item.right_index) for item in result.real_matches),
            ((0, 1),),
        )
        self.assertEqual(tuple(item.index for item in result.unmatched_left), (1,))
        self.assertEqual(tuple(item.index for item in result.unmatched_right), (0,))

    def test_scipy_backend_matches_exhaustive_oracle_on_small_matrices(self):
        rng = np.random.default_rng(20260621)
        for n in range(4):
            for m in range(4):
                for _ in range(8):
                    costs = rng.random((n, m))
                    if n and m:
                        costs[rng.random((n, m)) < 0.25] = np.inf
                    spatial = rng.random((n, m))
                    result = solve_augmented_assignment(costs, spatial, 0.7)
                    reference = exhaustive_lexicographic(costs, spatial, 0.7)
                    self.assertAlmostEqual(result.primary_total_cost, reference[0], places=12)
                    self.assertEqual(
                        tuple((item.left_index, item.right_index) for item in result.real_matches),
                        reference[3],
                    )

    def test_global_margin_includes_dummy_alternative(self):
        result = self.solve([[0.39]], unmatched=0.8)
        self.assertAlmostEqual(result.real_matches[0].delta_global, 0.41)
        self.assertAlmostEqual(result.real_matches[0].delta_row, 0.01)
        self.assertAlmostEqual(result.real_matches[0].delta_col, 0.01)

    def test_semantic_config_digest_is_deterministic_and_parameter_sensitive(self):
        base = MatchingConfig(
            unmatched_pair_cost=0.8,
            local_margin_gate=0.1,
            global_margin_gate=0.2,
            severe_quality_flags=("POSSIBLE_SPLITTING",),
        )
        same = MatchingConfig(
            unmatched_pair_cost=0.8,
            local_margin_gate=0.1,
            global_margin_gate=0.2,
            severe_quality_flags=("POSSIBLE_SPLITTING",),
        )
        changed = MatchingConfig(
            unmatched_pair_cost=0.7,
            local_margin_gate=0.1,
            global_margin_gate=0.2,
            severe_quality_flags=("POSSIBLE_SPLITTING",),
        )
        self.assertEqual(base.config_digest, same.config_digest)
        self.assertNotEqual(base.config_digest, changed.config_digest)


if __name__ == "__main__":
    unittest.main()
