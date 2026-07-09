"""Canonical scoring, assignment, matching, grouping and change events."""

from .assignment import (
    AugmentedAssignmentResult,
    RealMatchDecision,
    UnmatchedDecision,
    solve_augmented_assignment,
)
from .config import (
    EVENT_POLICY_ID,
    EVENT_POLICY_VERSION,
    GLOBAL_ASSIGNMENT_POLICY_ID,
    GLOBAL_ASSIGNMENT_POLICY_VERSION,
    GROUPING_POLICY_ID,
    GROUPING_POLICY_VERSION,
    EventConfig,
    GroupingConfig,
    MatchingConfig,
    PairScoringConfig,
)
from .events import EventGenerationBatch, build_change_events
from .grouping import GroupEvidence, group_recurring_visual_types
from .neighboring import NeighboringMatchingBatch, match_neighboring_frames

__all__ = [
    "EVENT_POLICY_ID",
    "EVENT_POLICY_VERSION",
    "GLOBAL_ASSIGNMENT_POLICY_ID",
    "GLOBAL_ASSIGNMENT_POLICY_VERSION",
    "GROUPING_POLICY_ID",
    "GROUPING_POLICY_VERSION",
    "AugmentedAssignmentResult",
    "EventConfig",
    "EventGenerationBatch",
    "GroupEvidence",
    "GroupingConfig",
    "MatchingConfig",
    "NeighboringMatchingBatch",
    "PairScoringConfig",
    "RealMatchDecision",
    "UnmatchedDecision",
    "build_change_events",
    "group_recurring_visual_types",
    "match_neighboring_frames",
    "solve_augmented_assignment",
]
