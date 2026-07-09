"""SciPy-backed augmented global one-to-one assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class RealMatchDecision:
    left_index: int
    right_index: int
    delta_row: float | None
    delta_col: float | None
    delta_global: float | None


@dataclass(frozen=True, slots=True)
class UnmatchedDecision:
    side: str
    index: int
    delta_global: float | None


@dataclass(frozen=True, slots=True)
class AugmentedAssignmentResult:
    real_matches: tuple[RealMatchDecision, ...]
    unmatched_left: tuple[UnmatchedDecision, ...]
    unmatched_right: tuple[UnmatchedDecision, ...]
    primary_total_cost: float
    matrix_order: int
    forbidden_cost: float
    unmatched_left_cost: float
    unmatched_right_cost: float
    primary_tie_tolerance: float
    spatial_tie_tolerance: float


def solve_augmented_assignment(
    real_costs: np.ndarray,
    spatial_distances: np.ndarray,
    unmatched_pair_cost: float,
) -> AugmentedAssignmentResult:
    """Solve the architecture-defined `(n+m) x (n+m)` assignment.

    Non-finite real costs are forbidden. Spatial evidence is a secondary
    lexicographic tie-break and never changes the primary optimum.
    """

    costs = np.asarray(real_costs, dtype=np.float64)
    spatial = np.asarray(spatial_distances, dtype=np.float64)
    if costs.ndim != 2 or spatial.shape != costs.shape:
        raise ValueError("real_costs and spatial_distances must be equal 2-D matrices.")
    if isinstance(unmatched_pair_cost, bool) or not isinstance(unmatched_pair_cost, (int, float)):
        raise TypeError("unmatched_pair_cost must be a real number.")
    unmatched_pair_cost = float(unmatched_pair_cost)
    if not math.isfinite(unmatched_pair_cost) or unmatched_pair_cost <= 0.0:
        raise ValueError("unmatched_pair_cost must be finite and positive.")
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size and (np.min(finite_costs) < 0.0 or np.max(finite_costs) > 1.0):
        raise ValueError("Finite real costs must be in [0, 1].")
    finite_spatial = spatial[np.isfinite(costs)]
    if finite_spatial.size and (
        np.any(~np.isfinite(finite_spatial))
        or np.min(finite_spatial) < 0.0
        or np.max(finite_spatial) > 1.0
    ):
        raise ValueError("Spatial distances for finite real costs must be in [0, 1].")

    n, m = costs.shape
    order = n + m
    left_unmatched = unmatched_pair_cost / 2.0
    right_unmatched = unmatched_pair_cost / 2.0
    if order == 0:
        return AugmentedAssignmentResult(
            (), (), (), 0.0, 0, 0.0, left_unmatched, right_unmatched, 0.0, 0.0
        )

    finite_max = max(
        [left_unmatched, right_unmatched, 0.0]
        + finite_costs.tolist()
    )
    forbidden = (order + 1.0) * (finite_max + 1.0)
    base = np.full((order, order), forbidden, dtype=np.float64)
    base[:n, :m] = np.where(np.isfinite(costs), costs, forbidden)
    for index in range(n):
        base[index, m + index] = left_unmatched
    for index in range(m):
        base[n + index, index] = right_unmatched
    base[n:, m:] = 0.0

    spatial_tie = np.zeros_like(base)
    if n and m:
        for i in range(n):
            for j in range(m):
                if math.isfinite(costs[i, j]):
                    spatial_tie[i, j] = spatial[i, j]

    canonical_tie = np.zeros_like(base)
    radix = order + 1.0
    for i in range(n):
        row_weight = radix ** (-i)
        for j in range(order):
            if base[i, j] < forbidden:
                canonical_tie[i, j] = (j + 1.0) * row_weight

    rows, cols, primary_total, primary_tolerance, spatial_tolerance = _lexicographic_solve(
        base, spatial_tie, canonical_tie, forbidden
    )
    real_matches: list[RealMatchDecision] = []
    unmatched_left_decisions: list[UnmatchedDecision] = []
    unmatched_right_decisions: list[UnmatchedDecision] = []
    selected = tuple(zip(rows.tolist(), cols.tolist()))

    for row, col in selected:
        if row < n and col < m:
            row_alternatives = [left_unmatched]
            row_alternatives.extend(costs[row, k] for k in range(m) if k != col and math.isfinite(costs[row, k]))
            col_alternatives = [right_unmatched]
            col_alternatives.extend(costs[h, col] for h in range(n) if h != row and math.isfinite(costs[h, col]))
            real_matches.append(
                RealMatchDecision(
                    left_index=row,
                    right_index=col,
                    delta_row=min(row_alternatives) - costs[row, col],
                    delta_col=min(col_alternatives) - costs[row, col],
                    delta_global=_decision_margin(base, row, col, forbidden, primary_total),
                )
            )
        elif row < n and col == m + row:
            unmatched_left_decisions.append(
                UnmatchedDecision(
                    side="from",
                    index=row,
                    delta_global=_decision_margin(base, row, col, forbidden, primary_total),
                )
            )
        elif row == n + col and col < m:
            unmatched_right_decisions.append(
                UnmatchedDecision(
                    side="to",
                    index=col,
                    delta_global=_decision_margin(base, row, col, forbidden, primary_total),
                )
            )

    return AugmentedAssignmentResult(
        real_matches=tuple(sorted(real_matches, key=lambda item: (item.left_index, item.right_index))),
        unmatched_left=tuple(sorted(unmatched_left_decisions, key=lambda item: item.index)),
        unmatched_right=tuple(sorted(unmatched_right_decisions, key=lambda item: item.index)),
        primary_total_cost=primary_total,
        matrix_order=order,
        forbidden_cost=forbidden,
        unmatched_left_cost=left_unmatched,
        unmatched_right_cost=right_unmatched,
        primary_tie_tolerance=primary_tolerance,
        spatial_tie_tolerance=spatial_tolerance,
    )


def _primary_solve(matrix: np.ndarray, forbidden: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    rows, cols = linear_sum_assignment(matrix)
    selected = matrix[rows, cols]
    feasible = bool(np.all(selected < forbidden))
    return rows, cols, float(np.sum(selected)), feasible


def _lexicographic_solve(
    base: np.ndarray,
    spatial_tie: np.ndarray,
    canonical_tie: np.ndarray,
    forbidden: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    base_rows, base_cols, optimum, feasible = _primary_solve(base, forbidden)
    if not feasible:
        raise RuntimeError("Augmented assignment unexpectedly has no feasible solution.")
    primary_tolerance = _numerical_tie_tolerance(optimum, optimum, base.shape[0])
    epsilon = 1e-8 / max(1, base.shape[0])
    spatial_rows, spatial_cols = base_rows, base_cols
    spatial_optimum = float(np.sum(spatial_tie[base_rows, base_cols]))
    for _attempt in range(16):
        perturbed = base + epsilon * spatial_tie
        rows, cols = linear_sum_assignment(perturbed)
        primary = float(np.sum(base[rows, cols]))
        if _numerically_equal(primary, optimum, base.shape[0]):
            spatial_rows, spatial_cols = rows, cols
            spatial_optimum = float(np.sum(spatial_tie[rows, cols]))
            break
        epsilon *= 0.1

    canonical_epsilon = epsilon * 1e-4
    for _attempt in range(16):
        perturbed = base + epsilon * spatial_tie + canonical_epsilon * canonical_tie
        rows, cols = linear_sum_assignment(perturbed)
        primary = float(np.sum(base[rows, cols]))
        spatial_value = float(np.sum(spatial_tie[rows, cols]))
        if (
            _numerically_equal(primary, optimum, base.shape[0])
            and _numerically_equal(spatial_value, spatial_optimum, base.shape[0])
        ):
            return (
                rows,
                cols,
                optimum,
                primary_tolerance,
                _numerical_tie_tolerance(
                    spatial_value, spatial_optimum, base.shape[0]
                ),
            )
        canonical_epsilon *= 0.1
    return (
        spatial_rows,
        spatial_cols,
        optimum,
        primary_tolerance,
        _numerical_tie_tolerance(spatial_optimum, spatial_optimum, base.shape[0]),
    )


def _numerical_tie_tolerance(left: float, right: float, order: int) -> float:
    """Forward-error bound for sums of at most ``order`` binary64 costs."""

    scale = max(1.0, abs(left), abs(right))
    return 8.0 * np.finfo(np.float64).eps * max(1, order) * scale


def _numerically_equal(left: float, right: float, order: int) -> bool:
    return abs(left - right) <= _numerical_tie_tolerance(left, right, order)


def _decision_margin(
    base: np.ndarray,
    row: int,
    col: int,
    forbidden: float,
    optimum: float,
) -> float | None:
    alternative = base.copy()
    alternative[row, col] = forbidden
    _rows, _cols, cost, feasible = _primary_solve(alternative, forbidden)
    if not feasible:
        return None
    return max(0.0, cost - optimum)


__all__ = [
    "AugmentedAssignmentResult",
    "RealMatchDecision",
    "UnmatchedDecision",
    "solve_augmented_assignment",
]
