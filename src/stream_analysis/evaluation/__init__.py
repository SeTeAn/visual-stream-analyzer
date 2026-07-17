"""Annotation-only evaluation layer for saved stream analysis runs."""

from .annotation import (
    AnnotationFormatError,
    AnnotationInstance,
    ExpectedChangeEvent,
    StreamAnnotation,
    load_annotation,
)
from .artifacts import (
    EVALUATION_ERROR_LEDGER_SCHEMA_ID,
    EVALUATION_MANIFEST_SCHEMA_ID,
    EVALUATION_REPORT_SCHEMA_ID,
    EvaluationArtifacts,
    EvaluationCollisionError,
    write_evaluation_artifacts,
)
from .ranking import PairRankingRecord, evaluate_pair_ranking
from .runner import (
    EVALUATOR_VERSION,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    PredictedCandidate,
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
    evaluate_physical_instance_continuity,
    evaluate_saved_run,
)

__all__ = [
    "EVALUATION_ERROR_LEDGER_SCHEMA_ID",
    "EVALUATION_MANIFEST_SCHEMA_ID",
    "EVALUATION_REPORT_SCHEMA_ID",
    "EVALUATOR_VERSION",
    "AnnotationFormatError",
    "AnnotationInstance",
    "EvaluationArtifacts",
    "EvaluationCollisionError",
    "EvaluationError",
    "EvaluationRequest",
    "EvaluationResult",
    "ExpectedChangeEvent",
    "PairRankingRecord",
    "PredictedCandidate",
    "StreamAnnotation",
    "aggregate_candidate_metrics",
    "evaluate_candidate_predictions",
    "evaluate_physical_instance_continuity",
    "evaluate_pair_ranking",
    "evaluate_saved_run",
    "load_annotation",
    "write_evaluation_artifacts",
]
