"""Evaluation runner for saved analyze artifacts and annotations."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import (
    EvaluationConfig,
    EvaluationDataRole,
    EvaluationPolicy,
    evaluation_config_digest,
)
from ..contracts import BBox
from .annotation import (
    SUPPORTED_EVENT_TYPES,
    AnnotationInstance,
    ExpectedChangeEvent,
    StreamAnnotation,
    load_annotation,
)
from .artifacts import EvaluationArtifacts, write_evaluation_artifacts
from .ranking import PairRankingRecord, evaluate_pair_ranking

EVALUATOR_VERSION = "1.0.0"
METRIC_SPEC_VERSION = "stream_analysis_evaluation_protocol.v1"
CANDIDATE_ONLY_ANNOTATION_SCOPES = frozenset({
    "candidate_extraction_bbox_only",
    "ocid_candidate_extraction_bbox_from_instance_masks",
})

PRIMARY_IOU_THRESHOLD = 0.50
DIAGNOSTIC_IOU_LEVELS = (0.25, 0.75)
TAU_GT_PART = 0.10
TAU_PRED_PART = 0.10
TAU_UNION = 0.50


class EvaluationError(RuntimeError):
    """Fatal evaluator error that does not modify prediction artifacts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRequest:
    stream_directory: Path
    run_directory: Path
    output_root: Path
    evaluation_id: str
    annotation_path: Path | None = None
    data_role: EvaluationDataRole = EvaluationDataRole.PROBE_DEVELOPMENT


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    report: dict[str, Any]
    error_ledger: list[dict[str, Any]]
    artifacts: EvaluationArtifacts


@dataclass(frozen=True, slots=True)
class PredictedCandidate:
    candidate_id: str
    frame_id: str
    bbox: BBox
    validity_status: str
    warning_ids: tuple[str, ...]
    error_ids: tuple[str, ...]
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class CandidateAssignment:
    candidate_id: str
    instance_id: str
    frame_id: str
    iou: float
    visual_type_id: str


def evaluate_candidate_predictions(
    annotation: StreamAnnotation,
    predicted: tuple[PredictedCandidate, ...],
) -> dict[str, Any]:
    """Evaluate in-memory candidates for an isolated extractor gate."""

    if not isinstance(annotation, StreamAnnotation):
        raise TypeError("annotation must be StreamAnnotation.")
    values = tuple(predicted)
    if not all(isinstance(item, PredictedCandidate) for item in values):
        raise TypeError("predicted must contain PredictedCandidate values.")
    return _strip_internal(_evaluate_candidates(annotation, values))


def evaluate_physical_instance_continuity(
    annotation: StreamAnnotation,
    predicted: tuple[PredictedCandidate, ...],
    primary_result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate neighboring matches against an evaluation-only identity proxy.

    This metric is intentionally separate from visual-type grouping and event
    evaluation. It is valid only for candidate-only annotations whose IDs mean
    physical continuity across neighboring frames.
    """

    if annotation.annotation_scope not in CANDIDATE_ONLY_ANNOTATION_SCOPES:
        raise ValueError("Physical continuity requires a candidate-only annotation scope.")
    if not isinstance(primary_result, dict):
        raise TypeError("primary_result must be a dictionary.")
    values = tuple(predicted)
    if not all(isinstance(item, PredictedCandidate) for item in values):
        raise TypeError("predicted must contain PredictedCandidate values.")
    candidate_evaluation = _evaluate_candidates(annotation, values)
    assignment_by_candidate = {
        item.candidate_id: item for item in candidate_evaluation["assignments"]
    }
    candidate_ids_by_frame = _candidate_ids_by_frame(values)
    gt_counts = _gt_counts_by_frame(annotation)
    policy_rows: dict[str, list[dict[str, Any]]] = {
        "accepted_strict": [],
        "selected_including_uncertain": [],
    }
    withheld = 0
    uncertain_count = 0
    for comparison in _frame_comparisons(primary_result):
        pair = _mapping(comparison, "frame_pair")
        left = _text(pair, "from_frame_id")
        right = _text(pair, "to_frame_id")
        expected = sum(
            min(gt_counts[left].get(identity_id, 0), gt_counts[right].get(identity_id, 0))
            for identity_id in annotation.visual_type_ids
        )
        if comparison.get("emission_status") == "withheld":
            withheld += 1
            for rows in policy_rows.values():
                rows.append({"tp": 0, "fp": 0, "fn": expected})
            continue
        accepted = tuple(str(item) for item in (comparison.get("accepted_match_ids") or ()))
        uncertain = tuple(str(item) for item in (comparison.get("uncertain_match_ids") or ()))
        uncertain_count += len(uncertain)
        accepted_tp, accepted_fp = _physical_match_counts(
            accepted,
            assignment_by_candidate,
            comparison,
            candidate_ids_by_frame,
        )
        selected_tp, selected_fp = _physical_match_counts(
            accepted + uncertain,
            assignment_by_candidate,
            comparison,
            candidate_ids_by_frame,
        )
        policy_rows["accepted_strict"].append({
            "tp": accepted_tp,
            "fp": accepted_fp + len(uncertain),
            "fn": max(0, expected - accepted_tp),
        })
        policy_rows["selected_including_uncertain"].append({
            "tp": selected_tp,
            "fp": selected_fp,
            "fn": max(0, expected - selected_tp),
        })

    def aggregate(rows: list[dict[str, int]]) -> dict[str, Any]:
        tp = sum(item["tp"] for item in rows)
        fp = sum(item["fp"] for item in rows)
        fn = sum(item["fn"] for item in rows)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
        }

    return {
        "status": "supported_physical_instance_proxy_evaluation_only",
        "identity_semantics": "physical_instance_continuity_not_visual_type_ground_truth",
        "candidate_iou_threshold": PRIMARY_IOU_THRESHOLD,
        "bbox_matched_candidate_count": len(assignment_by_candidate),
        "frame_pair_count": len(_frame_comparisons(primary_result)),
        "uncertain_selected_count": uncertain_count,
        "withheld_comparison_count": withheld,
        **{name: aggregate(rows) for name, rows in policy_rows.items()},
        "limitations": [
            "Includes candidate-extraction misses in continuity false negatives.",
            "Uses compact match IDs because full MatchRecord artifacts are not saved.",
            "Must not be interpreted as visual-type grouping or event quality.",
        ],
    }


def _physical_match_counts(
    match_ids: tuple[str, ...],
    assignment_by_candidate: dict[str, CandidateAssignment],
    comparison: dict[str, Any],
    candidate_ids_by_frame: dict[str, tuple[str, ...]],
) -> tuple[int, int]:
    true_positive = 0
    false_positive = 0
    for match_id in match_ids:
        parsed = _resolve_match_endpoints(
            match_id,
            comparison,
            candidate_ids_by_frame,
        )
        if parsed is None:
            false_positive += 1
            continue
        left = assignment_by_candidate.get(parsed[0])
        right = assignment_by_candidate.get(parsed[1])
        if left is not None and right is not None and left.visual_type_id == right.visual_type_id:
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive


def aggregate_candidate_metrics(metrics: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Micro-aggregate candidate-only reports from disjoint streams."""

    values = tuple(metrics)
    if not values:
        raise ValueError("metrics must not be empty.")
    tp = sum(int(item["tp"]) for item in values)
    fp = sum(int(item["fp"]) for item in values)
    fn = sum(int(item["fn"]) for item in values)
    frame_count = sum(len(item["per_frame"]) for item in values)
    iou_count = sum(int(item["tp_iou"]["count"]) for item in values)
    iou_total = sum(
        int(item["tp_iou"]["count"]) * float(item["tp_iou"]["mean"] or 0.0)
        for item in values
    )
    diagnostic_keys = ("duplicate", "split", "merge", "fragment", "noise", "miss")
    diagnostics = {
        key: sum(int(item["diagnostics"][key]) for item in values)
        for key in diagnostic_keys
    }
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _harmonic(precision, recall),
        "false_per_frame": None if frame_count == 0 else fp / frame_count,
        "weighted_mean_tp_iou": None if iou_count == 0 else iou_total / iou_count,
        "diagnostics": diagnostics,
    }


def evaluate_saved_run(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate existing files under ``outputs/runs/<run_id>``."""

    stream_root = Path(request.stream_directory).resolve()
    run_dir = Path(request.run_directory).resolve()
    if not run_dir.is_dir():
        raise EvaluationError(f"Run directory does not exist: {run_dir}")
    annotation_path = (
        Path(request.annotation_path).resolve()
        if request.annotation_path is not None
        else stream_root / "annotation.json"
    )
    manifest_path = stream_root / "manifest.json"
    annotation = load_annotation(annotation_path, manifest_path=manifest_path)
    candidate_manifest = _load_json(run_dir / "candidate_manifest.json")
    stream_result_payload = _load_json(run_dir / "stream_analysis.json")
    run_manifest = _load_json(run_dir / "run_manifest.json")
    primary_result = _primary_result(stream_result_payload)
    run_id = _text(primary_result, "run_id")
    if primary_result.get("envelope", {}).get("stream_id") != annotation.stream_id:
        raise EvaluationError("Prediction stream_id does not match annotation stream_id.")

    predicted = _predicted_candidates(candidate_manifest)
    candidate_eval = _evaluate_candidates(annotation, predicted)
    assignment_by_candidate = {
        item.candidate_id: item for item in candidate_eval["assignments"]
    }
    if annotation.annotation_scope in CANDIDATE_ONLY_ANNOTATION_SCOPES:
        representation_eval = _unsupported_by_annotation_scope("representations")
        matching_eval = _unsupported_by_annotation_scope("matching")
        grouping_eval = _unsupported_by_annotation_scope("grouping")
        event_eval = {
            **_unsupported_by_annotation_scope("events"),
            "false_negative_keys": [],
            "false_positive_keys": [],
        }
    else:
        representation_eval = _evaluate_representations(
            run_dir,
            annotation,
            assignment_by_candidate,
            primary_result,
            run_manifest,
            predicted,
        )
        matching_eval = _evaluate_matching(
            annotation,
            assignment_by_candidate,
            primary_result,
            predicted,
        )
        grouping_eval = _evaluate_grouping(
            annotation, assignment_by_candidate, primary_result
        )
        event_eval = _evaluate_events(
            annotation, assignment_by_candidate, grouping_eval, primary_result
        )
    error_ledger = _error_ledger(candidate_eval, event_eval)
    now = datetime.now(timezone.utc).isoformat()
    evaluation_config = _default_evaluation_config(request.data_role)
    evaluated_files = {
        "run_manifest": _digest_file(run_dir / "run_manifest.json"),
        "candidate_manifest": _digest_file(run_dir / "candidate_manifest.json"),
        "stream_analysis": _digest_file(run_dir / "stream_analysis.json"),
        "annotation": {"algorithm": "sha256", "value": annotation.digest_sha256},
    }
    pair_scores_path = _pair_scores_path(run_dir)
    if pair_scores_path.exists():
        evaluated_files["diagnostic_pair_scores"] = _digest_file(pair_scores_path)
    report: dict[str, Any] = {
        "evaluator_version": EVALUATOR_VERSION,
        "metric_spec_version": METRIC_SPEC_VERSION,
        "evaluation_id": request.evaluation_id,
        "evaluated_run": {
            "run_id": run_id,
            "run_directory": str(run_dir),
            "run_status": primary_result.get("run_status"),
            "candidate_manifest_ref": primary_result.get("candidate_manifest_ref"),
        },
        "data": {
            "stream_id": annotation.stream_id,
            "annotation_scope": annotation.annotation_scope,
            "data_role": request.data_role.value,
            "frame_count": len(annotation.frame_ids),
            "gt_instance_count": len(annotation.instances),
            "gt_event_count": len(annotation.change_events),
            "is_final_claim": request.data_role is EvaluationDataRole.FINAL_HELD_OUT,
        },
        "support": {
            "event_types": dict(Counter(event.event_type for event in annotation.change_events)),
            "visual_type_count": len(annotation.visual_type_ids),
        },
        "metrics": {
            "candidate_extraction": _strip_internal(candidate_eval),
            "representations": representation_eval,
            "matching": matching_eval,
            "grouping": _strip_internal(grouping_eval),
            "events": _strip_internal(event_eval),
        },
        "artifact_digests": evaluated_files,
        "limitations": _limitations(representation_eval, grouping_eval),
    }
    summary = _summary_text(report)
    manifest = {
        "evaluation_id": request.evaluation_id,
        "created_at": now,
        "evaluator_version": EVALUATOR_VERSION,
        "metric_spec_version": METRIC_SPEC_VERSION,
        "parent_prediction_run_id": run_id,
        "evaluation_config_digest": evaluation_config_digest(evaluation_config),
        "evaluated_files": evaluated_files,
        "output_files": (
            "evaluation_manifest.json",
            "evaluation_report.json",
            "error_ledger.json",
            "summary.txt",
        ),
    }
    artifacts = write_evaluation_artifacts(
        output_root=request.output_root,
        evaluation_id=request.evaluation_id,
        report=report,
        error_ledger=error_ledger,
        summary_text=summary,
        manifest=manifest,
    )
    return EvaluationResult(report=report, error_ledger=error_ledger, artifacts=artifacts)


def _unsupported_by_annotation_scope(stage: str) -> dict[str, Any]:
    return {
        "status": "not_supported_by_annotation_scope",
        "reason": (
            f"{stage} metrics require type/event ground truth; this annotation "
            "contains physical-instance boxes for candidate evaluation only."
        ),
        "limitations": [
            "OCID numeric instance labels are not ground-truth visual types."
        ],
    }


def _default_evaluation_config(role: EvaluationDataRole) -> EvaluationConfig:
    return EvaluationConfig(
        config_version="1.0.0",
        data_role=role,
        metric_policy=EvaluationPolicy(
            policy_id="stream_analysis_primary_metrics",
            policy_version="1.0.0",
            parameters={
                "primary_iou_threshold": PRIMARY_IOU_THRESHOLD,
                "diagnostic_iou_levels": DIAGNOSTIC_IOU_LEVELS,
                "significant_overlap": {
                    "tau_gt_part": TAU_GT_PART,
                    "tau_pred_part": TAU_PRED_PART,
                    "tau_union": TAU_UNION,
                },
            },
        ),
        invalid_status_policy=EvaluationPolicy(
            policy_id="strict_lower_bound_invalid_uncertain",
            policy_version="1.0.0",
            parameters={"invalid_representation_ranking": "rank_buckets"},
        ),
        mapping_policy=EvaluationPolicy(
            policy_id="one_to_one_maximum_overlap_type_mapping",
            policy_version="1.0.0",
        ),
        artifact_policy=EvaluationPolicy(
            policy_id="immutable_evaluation_directory",
            policy_version="1.0.0",
        ),
        selection_freeze_metadata={},
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot load required artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"Artifact must contain a JSON object: {path}")
    return value


def _primary_result(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_id") != "stream_analysis.primary_stream_result.v1":
        raise EvaluationError("stream_analysis.json has an unsupported schema_id.")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise EvaluationError("stream_analysis.json misses result object.")
    return result


def _predicted_candidates(candidate_manifest: dict[str, Any]) -> tuple[PredictedCandidate, ...]:
    if candidate_manifest.get("schema_id") != "stream_analysis.candidate_manifest.v1":
        raise EvaluationError("candidate_manifest.json has an unsupported schema_id.")
    result = candidate_manifest.get("result")
    candidates = result.get("candidates") if isinstance(result, dict) else None
    if not isinstance(candidates, list):
        raise EvaluationError("candidate_manifest.json misses result.candidates.")
    parsed: list[PredictedCandidate] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise EvaluationError("candidate entries must be objects.")
        bbox_data = _mapping(item, "bbox")
        envelope = _mapping(item, "envelope")
        parsed.append(
            PredictedCandidate(
                candidate_id=_text(item, "candidate_id"),
                frame_id=_text(item, "frame_id"),
                frame_index=_integer(item, "frame_index"),
                bbox=BBox(
                    x=_number(bbox_data, "x"),
                    y=_number(bbox_data, "y"),
                    width=_number(bbox_data, "width"),
                    height=_number(bbox_data, "height"),
                ),
                validity_status=str(envelope.get("validity_status", "valid")),
                warning_ids=tuple(envelope.get("warning_ids") or ()),
                error_ids=tuple(envelope.get("error_ids") or ()),
            )
        )
    return tuple(sorted(parsed, key=lambda item: (item.frame_id, item.candidate_id)))


def _evaluate_candidates(annotation: StreamAnnotation, predicted: tuple[PredictedCandidate, ...]) -> dict[str, Any]:
    gt_by_frame = annotation.instances_by_frame
    pred_by_frame: dict[str, list[PredictedCandidate]] = {frame_id: [] for frame_id in annotation.frame_ids}
    for candidate in predicted:
        pred_by_frame.setdefault(candidate.frame_id, []).append(candidate)

    assignments: list[CandidateAssignment] = []
    rows: list[dict[str, Any]] = []
    diagnostics = {"duplicate": 0, "split": 0, "merge": 0, "fragment": 0, "noise": 0, "miss": 0}
    fp_ids: list[str] = []
    fn_ids: list[str] = []
    tp_ious: list[float] = []

    for frame_id in annotation.frame_ids:
        gt = gt_by_frame.get(frame_id, ())
        pred = tuple(pred_by_frame.get(frame_id, ()))
        matched_pred, matched_gt, frame_assignments = _assign_frame_candidates(pred, gt, PRIMARY_IOU_THRESHOLD)
        assignments.extend(frame_assignments)
        tp_ious.extend(item.iou for item in frame_assignments)
        frame_fp = [item.candidate_id for index, item in enumerate(pred) if index not in matched_pred]
        frame_fn = [item.instance_id for index, item in enumerate(gt) if index not in matched_gt]
        fp_ids.extend(frame_fp)
        fn_ids.extend(frame_fn)
        frame_diag = _candidate_overlap_diagnostics(pred, gt, matched_pred, matched_gt)
        for key in diagnostics:
            diagnostics[key] += frame_diag[key]
        rows.append(
            {
                "frame_id": frame_id,
                "gt_count": len(gt),
                "predicted_count": len(pred),
                "tp": len(frame_assignments),
                "fp": len(frame_fp),
                "fn": len(frame_fn),
                "count_error": len(pred) - len(gt),
                "empty_frame_correct": len(gt) == 0 and len(pred) == 0,
            }
        )
    tp = len(assignments)
    fp = len(fp_ids)
    fn = len(fn_ids)
    summary = {
        "status": "supported",
        "primary_iou_threshold": PRIMARY_IOU_THRESHOLD,
        "diagnostic_iou_levels": DIAGNOSTIC_IOU_LEVELS,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "false_per_frame": fp / len(annotation.frame_ids),
        "tp_iou": _distribution(tp_ious),
        "completion_rate": 1.0,
        "diagnostics": diagnostics,
        "per_frame": rows,
        "false_positive_candidate_ids": fp_ids,
        "missed_instance_ids": fn_ids,
        "assignments": assignments,
    }
    return summary


def _assign_frame_candidates(
    predicted: tuple[PredictedCandidate, ...],
    gt: tuple[AnnotationInstance, ...],
    threshold: float,
) -> tuple[set[int], set[int], list[CandidateAssignment]]:
    if not predicted or not gt:
        return set(), set(), []
    matrix = np.zeros((len(predicted), len(gt)), dtype=np.float64)
    for i, pred in enumerate(predicted):
        for j, expected in enumerate(gt):
            matrix[i, j] = _iou(pred.bbox, expected.bbox)
    rows, cols = linear_sum_assignment(-matrix)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    assignments: list[CandidateAssignment] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        value = float(matrix[row, col])
        if value >= threshold:
            matched_pred.add(row)
            matched_gt.add(col)
            expected = gt[col]
            assignments.append(
                CandidateAssignment(
                    candidate_id=predicted[row].candidate_id,
                    instance_id=expected.instance_id,
                    frame_id=expected.frame_id,
                    iou=value,
                    visual_type_id=expected.visual_type_id,
                )
            )
    return matched_pred, matched_gt, assignments


def _candidate_overlap_diagnostics(
    predicted: tuple[PredictedCandidate, ...],
    gt: tuple[AnnotationInstance, ...],
    matched_pred: set[int],
    matched_gt: set[int],
) -> dict[str, int]:
    duplicate_gt: set[int] = set()
    split_gt: set[int] = set()
    split_pred: set[int] = set()
    merge_pred: set[int] = set()
    fragment_pred: set[int] = set()
    noise_pred: set[int] = set()
    edges: dict[int, list[int]] = defaultdict(list)
    reverse_edges: dict[int, list[int]] = defaultdict(list)
    for i, pred in enumerate(predicted):
        for j, expected in enumerate(gt):
            if _significant_overlap(pred.bbox, expected.bbox):
                edges[j].append(i)
                reverse_edges[i].append(j)
    for j, expected in enumerate(gt):
        iou_hits = [i for i, pred in enumerate(predicted) if _iou(pred.bbox, expected.bbox) >= PRIMARY_IOU_THRESHOLD]
        if len(iou_hits) >= 2:
            duplicate_gt.add(j)
            continue
        significant = edges.get(j, [])
        if len(significant) >= 2 and _union_coverage(expected.bbox, tuple(predicted[i].bbox for i in significant)) >= TAU_UNION:
            split_gt.add(j)
            split_pred.update(significant)
    for i, pred in enumerate(predicted):
        if len(reverse_edges.get(i, [])) >= 2:
            merge_pred.add(i)
    for i, pred in enumerate(predicted):
        if i in matched_pred or i in merge_pred or i in split_pred:
            continue
        related = reverse_edges.get(i, [])
        if len(related) == 1 and _iou(pred.bbox, gt[related[0]].bbox) < PRIMARY_IOU_THRESHOLD:
            fragment_pred.add(i)
        elif not related:
            noise_pred.add(i)
    return {
        "duplicate": len(duplicate_gt),
        "split": len(split_gt),
        "merge": len(merge_pred),
        "fragment": len(fragment_pred),
        "noise": len(noise_pred),
        "miss": sum(1 for index in range(len(gt)) if index not in matched_gt),
    }


def _evaluate_representations(
    run_dir: Path,
    annotation: StreamAnnotation,
    assignment_by_candidate: dict[str, CandidateAssignment],
    primary_result: dict[str, Any],
    run_manifest: dict[str, Any],
    predicted_candidates: tuple[PredictedCandidate, ...],
) -> dict[str, Any]:
    summaries = primary_result.get("representation_summaries")
    invalid_summary_count = 0
    if isinstance(summaries, list):
        invalid_summary_count = sum(
            1 for item in summaries
            if isinstance(item, dict) and item.get("validity_status") != "valid"
        )
    optional = _pair_scores_path(run_dir)
    declared_reference = _declared_pair_scores_reference(primary_result)
    if not optional.exists():
        if declared_reference is not None:
            raise EvaluationError(
                "stream_analysis.json declares diagnostics/pair_scores.json, but the file is missing."
            )
        return {
            "status": "not_supported_by_saved_artifacts",
            "reason": "Saved primary artifacts do not include full pairwise visual_score records or full representation payloads.",
            "implemented_policy": "strict_invalid_rank_buckets_v1",
            "candidate_assignment_coverage": _safe_div(len(assignment_by_candidate), len(annotation.instances)),
            "representation_summary_count": len(summaries) if isinstance(summaries, list) else 0,
            "invalid_representation_summary_count": invalid_summary_count,
            "required_optional_artifact": "diagnostics/pair_scores.json",
        }
    if declared_reference != "diagnostics/pair_scores.json":
        raise EvaluationError(
            "diagnostics/pair_scores.json exists without its exact typed artifact reference."
        )
    payload = _load_json(optional)
    if payload.get("schema_id") != "stream_analysis.diagnostic_pair_scores.v1":
        raise EvaluationError("diagnostics/pair_scores.json has an unsupported schema_id.")
    run_id = _text(primary_result, "run_id")
    if payload.get("run_id") != run_id:
        raise EvaluationError("diagnostics/pair_scores.json run_id does not match the prediction run.")
    if payload.get("stream_id") != annotation.stream_id:
        raise EvaluationError("diagnostics/pair_scores.json stream_id does not match annotation stream_id.")
    producer = _mapping(_mapping(primary_result, "envelope"), "producer")
    if payload.get("analysis_config_digest") != _text(producer, "config_digest"):
        raise EvaluationError(
            "diagnostics/pair_scores.json analysis_config_digest does not match the prediction run."
        )
    scorer_digest = _stage_semantic_digest(run_manifest, "scorer")
    if payload.get("scoring_config_digest") != scorer_digest:
        raise EvaluationError(
            "diagnostics/pair_scores.json scoring_config_digest does not match run provenance."
        )
    for field_name in ("representation_variant_id", "scorer_id", "scorer_version"):
        if not isinstance(payload.get(field_name), str) or not payload[field_name].strip():
            raise EvaluationError(f"diagnostics/pair_scores.json {field_name} must be a non-empty string.")
    score_semantics = _mapping(payload, "score_semantics")
    if score_semantics.get("visual_score") != "bounded_higher_is_more_similar_not_probability":
        raise EvaluationError(
            "diagnostics/pair_scores.json uses unsupported visual_score semantics."
        )
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise EvaluationError("diagnostics/pair_scores.json pairs must be an array.")
    candidate_by_id = {item.candidate_id: item for item in predicted_candidates}
    if len(candidate_by_id) != len(predicted_candidates):
        raise EvaluationError("candidate_manifest.json contains duplicate candidate IDs.")
    primary_candidate_ids = primary_result.get("candidate_record_ids")
    if not isinstance(primary_candidate_ids, list) or any(
        not isinstance(item, str) for item in primary_candidate_ids
    ):
        raise EvaluationError("stream_analysis.json candidate_record_ids must be an array of strings.")
    if set(primary_candidate_ids) != set(candidate_by_id):
        raise EvaluationError(
            "candidate_manifest.json candidates do not match stream_analysis.json candidate_record_ids."
        )
    summary_by_candidate = _representation_summaries_by_candidate(
        primary_result,
        candidate_by_id,
    )
    frame_ids = primary_result.get("frame_ids")
    if not isinstance(frame_ids, list) or any(not isinstance(item, str) for item in frame_ids):
        raise EvaluationError("stream_analysis.json frame_ids must be an array of strings.")
    if len(frame_ids) != len(set(frame_ids)):
        raise EvaluationError("stream_analysis.json frame_ids must be unique.")
    frame_position = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    candidates_by_frame: dict[str, list[PredictedCandidate]] = {
        frame_id: [] for frame_id in frame_ids
    }
    for candidate in predicted_candidates:
        if candidate.frame_id not in candidates_by_frame:
            raise EvaluationError("A candidate references a frame outside stream_analysis.json.")
        candidates_by_frame[candidate.frame_id].append(candidate)
    expected_endpoint_pairs = {
        (from_frame_id, to_frame_id, left.candidate_id, right.candidate_id)
        for from_frame_id, to_frame_id in zip(frame_ids, frame_ids[1:])
        for left in candidates_by_frame[from_frame_id]
        for right in candidates_by_frame[to_frame_id]
    }
    pairs = []
    seen_pair_ids: set[str] = set()
    seen_endpoints: set[tuple[str, str, str, str]] = set()
    comparison_by_frame_pair: dict[tuple[str, str], str] = {}
    frame_pair_by_comparison: dict[str, tuple[str, str]] = {}
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise EvaluationError("diagnostics/pair_scores.json pair rows must be objects.")
        pair_id = _text(item, "pair_id")
        comparison_id = _text(item, "comparison_id")
        left_candidate_id = _text(item, "from_candidate_id")
        right_candidate_id = _text(item, "to_candidate_id")
        if pair_id in seen_pair_ids:
            raise EvaluationError("diagnostics/pair_scores.json contains duplicate pair_id values.")
        frame_pair = _mapping(item, "frame_pair")
        from_frame_id = _text(frame_pair, "from_frame_id")
        to_frame_id = _text(frame_pair, "to_frame_id")
        from_frame_index = _integer(frame_pair, "from_frame_index")
        to_frame_index = _integer(frame_pair, "to_frame_index")
        if (
            from_frame_id not in frame_position
            or to_frame_id not in frame_position
            or frame_position[to_frame_id] != frame_position[from_frame_id] + 1
            or to_frame_index != from_frame_index + 1
        ):
            raise EvaluationError(
                "diagnostics/pair_scores.json rows must compare neighboring frames in forward order."
            )
        left_candidate = candidate_by_id.get(left_candidate_id)
        right_candidate = candidate_by_id.get(right_candidate_id)
        if left_candidate is None or right_candidate is None:
            raise EvaluationError(
                "diagnostics/pair_scores.json references an unknown candidate endpoint."
            )
        if (left_candidate.frame_id, left_candidate.frame_index) != (
            from_frame_id,
            from_frame_index,
        ) or (right_candidate.frame_id, right_candidate.frame_index) != (
            to_frame_id,
            to_frame_index,
        ):
            raise EvaluationError(
                "A pair-score candidate endpoint does not belong to its declared frame."
            )
        frame_pair_key = (from_frame_id, to_frame_id)
        prior_comparison = comparison_by_frame_pair.setdefault(
            frame_pair_key,
            comparison_id,
        )
        prior_frame_pair = frame_pair_by_comparison.setdefault(
            comparison_id,
            frame_pair_key,
        )
        if prior_comparison != comparison_id or prior_frame_pair != frame_pair_key:
            raise EvaluationError(
                "Pair-score comparison IDs and neighboring frame pairs must be one-to-one."
            )
        endpoint_key = (
            from_frame_id,
            to_frame_id,
            left_candidate_id,
            right_candidate_id,
        )
        if endpoint_key in seen_endpoints:
            raise EvaluationError(
                "diagnostics/pair_scores.json contains duplicate comparison endpoint pairs."
            )
        seen_pair_ids.add(pair_id)
        seen_endpoints.add(endpoint_key)
        visual_score = item.get("visual_score")
        if visual_score is not None:
            if (
                isinstance(visual_score, bool)
                or not isinstance(visual_score, (int, float))
                or not math.isfinite(float(visual_score))
                or not 0.0 <= float(visual_score) <= 1.0
            ):
                raise EvaluationError(
                    "diagnostics/pair_scores.json visual_score must be null or finite in [0, 1]."
                )
        validity = _pair_validity(item)
        if validity not in {"valid", "invalid"}:
            raise EvaluationError(
                "diagnostics/pair_scores.json lineage validity_status must be valid or invalid."
            )
        if validity == "valid" and visual_score is None:
            raise EvaluationError(
                "A valid pair score requires finite visual_score in [0, 1]."
            )
        if validity == "invalid" and visual_score is not None:
            raise EvaluationError("An invalid pair score must have visual_score=null.")
        representation_ids = item.get("representation_record_ids")
        if (
            not isinstance(representation_ids, list)
            or len(representation_ids) != 2
            or any(not isinstance(value, str) for value in representation_ids)
            or len(set(representation_ids)) != 2
        ):
            raise EvaluationError(
                "diagnostics/pair_scores.json representation_record_ids must contain two unique strings."
            )
        left_summary = summary_by_candidate[left_candidate_id]
        right_summary = summary_by_candidate[right_candidate_id]
        expected_representation_ids = {
            left_summary["representation_record_id"],
            right_summary["representation_record_id"],
        }
        if set(representation_ids) != expected_representation_ids:
            raise EvaluationError(
                "Pair-score representation IDs do not correspond to its candidate endpoints."
            )
        query_valid = left_summary["validity_status"] == "valid"
        gallery_valid = right_summary["validity_status"] == "valid"
        if validity == "valid" and not (query_valid and gallery_valid):
            raise EvaluationError(
                "A pair score with an invalid representation endpoint cannot be valid."
            )
        left_type = assignment_by_candidate.get(left_candidate_id)
        right_type = assignment_by_candidate.get(right_candidate_id)
        pairs.append(
            PairRankingRecord(
                query_id=left_candidate_id,
                gallery_id=right_candidate_id,
                label=(
                    left_type is not None
                    and right_type is not None
                    and left_type.visual_type_id == right_type.visual_type_id
                ),
                visual_score=None if visual_score is None else float(visual_score),
                valid=validity == "valid",
                query_valid=query_valid,
                gallery_valid=gallery_valid,
            )
        )
    if seen_endpoints != expected_endpoint_pairs:
        missing = len(expected_endpoint_pairs - seen_endpoints)
        extra = len(seen_endpoints - expected_endpoint_pairs)
        raise EvaluationError(
            "diagnostics/pair_scores.json must contain the complete Cartesian score "
            f"matrix for every neighboring frame pair (missing={missing}, extra={extra})."
        )
    return evaluate_pair_ranking(
        tuple(pairs),
        candidate_validity={
            candidate_id: summary["validity_status"] == "valid"
            for candidate_id, summary in summary_by_candidate.items()
        },
    )


def _representation_summaries_by_candidate(
    primary_result: dict[str, Any],
    candidate_by_id: dict[str, PredictedCandidate],
) -> dict[str, dict[str, Any]]:
    summaries = primary_result.get("representation_summaries")
    if not isinstance(summaries, list):
        raise EvaluationError("stream_analysis.json representation_summaries must be an array.")
    result: dict[str, dict[str, Any]] = {}
    representation_ids: set[str] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            raise EvaluationError("Representation summaries must be objects.")
        candidate_id = _text(summary, "candidate_id")
        representation_id = _text(summary, "representation_record_id")
        frame_id = _text(summary, "frame_id")
        validity = str(summary.get("validity_status"))
        if validity not in {"valid", "invalid"}:
            raise EvaluationError("Representation summary validity_status must be valid or invalid.")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None or candidate.frame_id != frame_id:
            raise EvaluationError(
                "A representation summary does not correspond to its candidate frame."
            )
        if candidate_id in result or representation_id in representation_ids:
            raise EvaluationError(
                "Each candidate and representation ID must occur in exactly one summary."
            )
        result[candidate_id] = {
            "representation_record_id": representation_id,
            "validity_status": validity,
        }
        representation_ids.add(representation_id)
    if set(result) != set(candidate_by_id):
        raise EvaluationError(
            "Every saved candidate must have exactly one representation summary."
        )
    return result


def _declared_pair_scores_reference(primary_result: dict[str, Any]) -> str | None:
    artifacts = primary_result.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvaluationError("stream_analysis.json artifacts must be an array.")
    references = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise EvaluationError("stream_analysis.json artifact references must be objects.")
        if item.get("artifact_kind") == "diagnostic_pair_scores":
            metadata = _mapping(item, "metadata")
            if metadata.get("artifact_format") != "stream_analysis.diagnostic_pair_scores.v1":
                raise EvaluationError(
                    "Pair-score artifact reference has an unsupported artifact_format."
                )
            if metadata.get("mimetype") != "application/json":
                raise EvaluationError(
                    "Pair-score artifact reference must declare application/json."
                )
            references.append(_text(item, "reference"))
    if len(references) > 1:
        raise EvaluationError("stream_analysis.json declares duplicate pair-score artifacts.")
    return references[0] if references else None


def _stage_semantic_digest(run_manifest: dict[str, Any], stage_id: str) -> str:
    provenance = _mapping(run_manifest, "provenance")
    stages = provenance.get("stage_configs")
    if not isinstance(stages, list):
        raise EvaluationError("run_manifest.json provenance.stage_configs must be an array.")
    matches = [
        item for item in stages
        if isinstance(item, dict) and item.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise EvaluationError(
            f"run_manifest.json must contain exactly one {stage_id} stage provenance record."
        )
    fingerprint = _mapping(matches[0], "semantic_digest")
    if fingerprint.get("algorithm") != "sha256":
        raise EvaluationError(f"{stage_id} semantic digest must use sha256.")
    return f"sha256:{_text(fingerprint, 'value')}"


def _pair_scores_path(run_dir: Path) -> Path:
    return run_dir / "diagnostics" / "pair_scores.json"


def _pair_validity(item: dict[str, Any]) -> str:
    lineage = item.get("lineage")
    if isinstance(lineage, dict):
        return str(lineage.get("validity_status", "valid"))
    return str(item.get("validity_status", "valid"))


def _evaluate_matching(
    annotation: StreamAnnotation,
    assignment_by_candidate: dict[str, CandidateAssignment],
    primary_result: dict[str, Any],
    predicted: tuple[PredictedCandidate, ...],
) -> dict[str, Any]:
    comparisons = _frame_comparisons(primary_result)
    gt_counts = _gt_counts_by_frame(annotation)
    candidate_ids_by_frame = _candidate_ids_by_frame(predicted)
    totals = Counter()
    rows = []
    for comparison in comparisons:
        pair = _mapping(comparison, "frame_pair")
        left = _text(pair, "from_frame_id")
        right = _text(pair, "to_frame_id")
        expected_capacity = sum(
            min(gt_counts[left].get(type_id, 0), gt_counts[right].get(type_id, 0))
            for type_id in annotation.visual_type_ids
        )
        if comparison.get("emission_status") == "withheld":
            totals.update({"fn": expected_capacity, "withheld": 1})
            rows.append({"comparison_id": comparison.get("comparison_id"), "tp": 0, "fp": 0, "fn": expected_capacity, "withheld": True})
            continue
        accepted = tuple(comparison.get("accepted_match_ids") or ())
        uncertain = tuple(comparison.get("uncertain_match_ids") or ())
        tp = 0
        fp = 0
        for match_id in accepted:
            parsed = _resolve_match_endpoints(
                str(match_id),
                comparison,
                candidate_ids_by_frame,
            )
            if parsed is None:
                fp += 1
                continue
            left_assignment = assignment_by_candidate.get(parsed[0])
            right_assignment = assignment_by_candidate.get(parsed[1])
            if left_assignment is not None and right_assignment is not None and left_assignment.visual_type_id == right_assignment.visual_type_id:
                tp += 1
            else:
                fp += 1
        fp += len(uncertain)
        fn = max(0, expected_capacity - tp)
        totals.update({"tp": tp, "fp": fp, "fn": fn, "uncertain": len(uncertain)})
        rows.append({"comparison_id": comparison.get("comparison_id"), "tp": tp, "fp": fp, "fn": fn, "uncertain": len(uncertain), "withheld": False})
    return {
        "status": "supported_from_compact_match_ids",
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": _safe_div(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": _safe_div(totals["tp"], totals["tp"] + totals["fn"]),
        "f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
        "uncertain_selected_count": totals["uncertain"],
        "withheld_comparison_count": totals["withheld"],
        "per_comparison": rows,
        "limitations": [
            "Endpoints are resolved against candidates from the declared frame pair because full MatchRecord artifacts are not saved."
        ],
    }


def _evaluate_grouping(
    annotation: StreamAnnotation,
    assignment_by_candidate: dict[str, CandidateAssignment],
    primary_result: dict[str, Any],
) -> dict[str, Any]:
    clusters, ambiguous = _predicted_clusters_from_presence(primary_result)
    matched_candidates = tuple(sorted(set(assignment_by_candidate) & set().union(*clusters.values()) if clusters else ()))
    if not matched_candidates:
        return {
            "status": "not_supported",
            "reason": "No bbox-matched candidate can be linked to predicted type presence.",
            "predicted_clusters": clusters,
            "type_mapping": {},
            "ambiguous_candidate_ids": ambiguous,
        }
    precision_sum = 0.0
    recall_sum = 0.0
    for candidate_id in matched_candidates:
        pred_cluster = next(members for members in clusters.values() if candidate_id in members)
        gt_type = assignment_by_candidate[candidate_id].visual_type_id
        gt_cluster = {
            other_id for other_id, assignment in assignment_by_candidate.items()
            if assignment.visual_type_id == gt_type and other_id in matched_candidates
        }
        intersection = len(set(pred_cluster) & gt_cluster)
        precision_sum += intersection / len(pred_cluster)
        recall_sum += intersection / len(gt_cluster)
    bcubed_p = precision_sum / len(matched_candidates)
    bcubed_r = recall_sum / len(matched_candidates)
    mapping = _type_mapping(clusters, assignment_by_candidate, annotation.visual_type_ids)
    return {
        "status": "supported_from_compact_type_presence",
        "candidate_coverage": _safe_div(len(matched_candidates), len(annotation.instances)),
        "bbox_matched_candidate_count": len(matched_candidates),
        "ambiguous_candidate_ids": ambiguous,
        "bcubed_precision": bcubed_p,
        "bcubed_recall": bcubed_r,
        "bcubed_f1": _harmonic(bcubed_p, bcubed_r),
        "predicted_type_count": len(clusters),
        "gt_type_count": len(annotation.visual_type_ids),
        "type_count_error": len(clusters) - len(annotation.visual_type_ids),
        "weighted_over_split_loss": _over_split_loss(matched_candidates, clusters, assignment_by_candidate),
        "weighted_over_merge_loss": _over_merge_loss(matched_candidates, clusters, assignment_by_candidate),
        "type_mapping": mapping,
        "predicted_clusters": {key: sorted(value) for key, value in clusters.items()},
        "limitations": ["Compact FrameComparison stores possible type presence; ambiguous alternatives are evaluated as singleton strict penalties."],
    }


def _evaluate_events(
    annotation: StreamAnnotation,
    assignment_by_candidate: dict[str, CandidateAssignment],
    grouping_eval: dict[str, Any],
    primary_result: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        (event.from_frame_id, event.to_frame_id, event.event_type, event.visual_type_id)
        for event in annotation.change_events
    }
    mapping = grouping_eval.get("type_mapping") if isinstance(grouping_eval.get("type_mapping"), dict) else {}
    predicted_events = _change_events(primary_result)
    strict_predicted: set[tuple[str, str, str, str]] = set()
    uncertain_events: list[str] = []
    unmapped_events: list[str] = []
    for event in predicted_events:
        kind = str(event.get("kind"))
        frame_pair = _mapping(event, "frame_pair")
        predicted_type_id = str(event.get("predicted_type_id"))
        mapped = mapping.get(predicted_type_id)
        if mapped is None:
            mapped = f"unmapped:{predicted_type_id}"
            unmapped_events.append(str(event.get("event_id")))
        key = (_text(frame_pair, "from_frame_id"), _text(frame_pair, "to_frame_id"), kind, mapped)
        if event.get("status") == "uncertain":
            uncertain_events.append(str(event.get("event_id")))
        else:
            strict_predicted.add(key)
    per_kind = {}
    for kind in SUPPORTED_EVENT_TYPES:
        exp = {item for item in expected if item[2] == kind}
        pred = {item for item in strict_predicted if item[2] == kind}
        tp = len(exp & pred)
        fp = len(pred - exp)
        fn = len(exp - pred)
        per_kind[kind] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "f1": _event_f1(tp, fp, fn),
            "support": len(exp),
            "predicted_support": len(pred),
        }
    tp_total = sum(row["tp"] for row in per_kind.values())
    fp_total = sum(row["fp"] for row in per_kind.values())
    fn_total = sum(row["fn"] for row in per_kind.values())
    f1_values = [
        row["f1"] for row in per_kind.values()
        if row["support"] or row["predicted_support"]
    ]
    false_negative_keys = sorted(expected - strict_predicted)
    false_positive_keys = sorted(strict_predicted - expected)
    return {
        "status": "supported",
        "strict_policy": "uncertain_events_are_fp_and_matching_gt_remains_fn",
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "micro_precision": _safe_div(tp_total, tp_total + fp_total),
        "micro_recall": _safe_div(tp_total, tp_total + fn_total),
        "micro_f1": _event_f1(tp_total, fp_total, fn_total),
        "macro_f1": None if not f1_values else sum(value or 0.0 for value in f1_values) / len(f1_values),
        "per_event_type": per_kind,
        "uncertain_predicted_event_ids": uncertain_events,
        "unmapped_predicted_event_ids": unmapped_events,
        "false_negative_keys": false_negative_keys,
        "false_positive_keys": false_positive_keys,
        "assignment_by_candidate": assignment_by_candidate,
    }


def _predicted_clusters_from_presence(primary_result: dict[str, Any]) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    by_candidate: dict[str, set[str]] = defaultdict(set)
    for comparison in _frame_comparisons(primary_result):
        for side in ("from_type_presence", "to_type_presence"):
            presence = comparison.get(side) or {}
            if not isinstance(presence, dict):
                continue
            for type_id, summary in presence.items():
                if not isinstance(summary, dict):
                    continue
                for candidate_id in summary.get("candidate_ids") or ():
                    by_candidate[str(candidate_id)].add(str(type_id))
    clusters: dict[str, set[str]] = defaultdict(set)
    ambiguous = []
    for candidate_id, type_ids in by_candidate.items():
        if len(type_ids) == 1:
            clusters[next(iter(type_ids))].add(candidate_id)
        else:
            synthetic = f"uncertain:{candidate_id}"
            clusters[synthetic].add(candidate_id)
            ambiguous.append(candidate_id)
    return dict(clusters), tuple(sorted(ambiguous))


def _type_mapping(
    clusters: dict[str, set[str]],
    assignment_by_candidate: dict[str, CandidateAssignment],
    gt_type_ids: tuple[str, ...],
) -> dict[str, str]:
    pred_ids = tuple(sorted(clusters))
    if not pred_ids or not gt_type_ids:
        return {}
    overlap = np.zeros((len(pred_ids), len(gt_type_ids)), dtype=np.float64)
    for i, pred_id in enumerate(pred_ids):
        for candidate_id in clusters[pred_id]:
            assignment = assignment_by_candidate.get(candidate_id)
            if assignment is None:
                continue
            overlap[i, gt_type_ids.index(assignment.visual_type_id)] += 1.0
    rows, cols = linear_sum_assignment(-overlap)
    result: dict[str, str] = {}
    for row, col in zip(rows.tolist(), cols.tolist()):
        if overlap[row, col] > 0:
            result[pred_ids[row]] = gt_type_ids[col]
    return result


def _error_ledger(candidate_eval: dict[str, Any], event_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate_id in candidate_eval["false_positive_candidate_ids"]:
        rows.append({"scope": "candidate", "record_id": candidate_id, "error_type": "false_positive", "primary_root_cause": "EXTRACTION_FP"})
    for instance_id in candidate_eval["missed_instance_ids"]:
        rows.append({"scope": "candidate", "record_id": instance_id, "error_type": "false_negative", "primary_root_cause": "EXTRACTION_MISS"})
    for key in event_eval["false_negative_keys"]:
        rows.append({"scope": "event", "record_id": "|".join(key), "error_type": "false_negative", "primary_root_cause": "GROUPING_OR_EVENT_RULE"})
    for key in event_eval["false_positive_keys"]:
        root = "GROUPING_OVERMERGE" if str(key[3]).startswith("unmapped:") else "EVENT_RULE"
        rows.append({"scope": "event", "record_id": "|".join(key), "error_type": "false_positive", "primary_root_cause": root})
    return rows


def _frame_comparisons(primary_result: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    values = primary_result.get("frame_comparisons")
    if not isinstance(values, list):
        return ()
    return tuple(item for item in values if isinstance(item, dict))


def _change_events(primary_result: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    values = primary_result.get("change_events")
    if not isinstance(values, list):
        return ()
    return tuple(item for item in values if isinstance(item, dict))


def _candidate_ids_by_frame(
    predicted: tuple[PredictedCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for candidate in predicted:
        if candidate.candidate_id in seen:
            raise EvaluationError(
                f"Duplicate candidate_id in candidate manifest: {candidate.candidate_id}."
            )
        seen.add(candidate.candidate_id)
        values[candidate.frame_id].append(candidate.candidate_id)
    return {
        frame_id: tuple(sorted(candidate_ids))
        for frame_id, candidate_ids in values.items()
    }


def _resolve_match_endpoints(
    match_id: str,
    comparison: dict[str, Any],
    candidate_ids_by_frame: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    pair = _mapping(comparison, "frame_pair")
    from_frame_id = _text(pair, "from_frame_id")
    to_frame_id = _text(pair, "to_frame_id")
    comparison_id = _text(comparison, "comparison_id")
    prefix = f"match:{comparison_id}:"
    if not match_id.startswith(prefix):
        return None
    right_ids = set(candidate_ids_by_frame.get(to_frame_id, ()))
    possibilities: list[tuple[str, str]] = []
    for left_id in candidate_ids_by_frame.get(from_frame_id, ()):
        left_prefix = f"{prefix}{left_id}:"
        if not match_id.startswith(left_prefix):
            continue
        right_id = match_id[len(left_prefix):]
        if right_id in right_ids:
            possibilities.append((left_id, right_id))
    if len(possibilities) != 1:
        return None
    return possibilities[0]


def _gt_counts_by_frame(annotation: StreamAnnotation) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {frame_id: Counter() for frame_id in annotation.frame_ids}
    for instance in annotation.instances:
        result[instance.frame_id][instance.visual_type_id] += 1
    return result


def _over_split_loss(
    candidates: tuple[str, ...],
    clusters: dict[str, set[str]],
    assignment_by_candidate: dict[str, CandidateAssignment],
) -> float:
    by_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for pred_id, members in clusters.items():
        for candidate_id in members:
            if candidate_id in candidates:
                by_gt[assignment_by_candidate[candidate_id].visual_type_id][pred_id] += 1
    outside = sum(sum(counter.values()) - max(counter.values()) for counter in by_gt.values() if counter)
    return outside / len(candidates) if candidates else 0.0


def _over_merge_loss(
    candidates: tuple[str, ...],
    clusters: dict[str, set[str]],
    assignment_by_candidate: dict[str, CandidateAssignment],
) -> float:
    outside = 0
    for members in clusters.values():
        counter = Counter(
            assignment_by_candidate[candidate_id].visual_type_id
            for candidate_id in members
            if candidate_id in candidates
        )
        if counter:
            outside += sum(counter.values()) - max(counter.values())
    return outside / len(candidates) if candidates else 0.0


def _strip_internal(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(item)
        for key, item in value.items()
        if key not in {"assignments", "assignment_by_candidate"}
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, CandidateAssignment):
        return {
            "candidate_id": value.candidate_id,
            "instance_id": value.instance_id,
            "frame_id": value.frame_id,
            "iou": value.iou,
            "visual_type_id": value.visual_type_id,
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _summary_text(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    candidate = metrics["candidate_extraction"]
    matching = metrics["matching"]
    grouping = metrics["grouping"]
    events = metrics["events"]
    return (
        "Stream analysis evaluation\n"
        f"Evaluation: {report['evaluation_id']}\n"
        f"Run: {report['evaluated_run']['run_id']}\n"
        f"Data role: {report['data']['data_role']}\n"
        f"Candidate F1: {candidate.get('f1')}\n"
        f"Representation status: {metrics['representations'].get('status')}\n"
        f"Matching F1: {matching.get('f1')}\n"
        f"Grouping B-cubed F1: {grouping.get('bcubed_f1')}\n"
        f"Event macro F1: {events.get('macro_f1')}\n"
    )


def _limitations(representation_eval: dict[str, Any], grouping_eval: dict[str, Any]) -> list[str]:
    rows = []
    if representation_eval.get("status") == "not_supported_by_saved_artifacts":
        rows.append("Representation ranking metrics require saved pair-score diagnostics; compact primary JSON is insufficient.")
    elif representation_eval.get("status") == "not_supported_by_annotation_scope":
        rows.append(str(representation_eval.get("reason")))
    elif representation_eval.get("status") != "supported":
        rows.append("Representation metrics are unavailable for this evaluation.")
    rows.extend(grouping_eval.get("limitations") or [])
    rows.append("Probe/development metrics are not final held-out claims unless data_role=final_held_out and freeze rules were followed.")
    return rows


def _digest_file(path: Path) -> dict[str, str]:
    return {"algorithm": "sha256", "value": sha256(path.read_bytes()).hexdigest()}


def _iou(left: BBox, right: BBox) -> float:
    intersection = left.intersection(right)
    if intersection is None:
        return 0.0
    union = left.area + right.area - intersection.area
    return 0.0 if union <= 0.0 else intersection.area / union


def _significant_overlap(pred: BBox, gt: BBox) -> bool:
    intersection = pred.intersection(gt)
    if intersection is None:
        return False
    coverage_gt = intersection.area / gt.area
    purity_pred = intersection.area / pred.area
    return coverage_gt >= TAU_GT_PART and purity_pred >= TAU_PRED_PART


def _union_coverage(gt: BBox, fragments: tuple[BBox, ...]) -> float:
    if not fragments:
        return 0.0
    # Exact rectangle union within GT by sweep over x-coordinates. Small n in probes.
    clipped = [fragment.intersection(gt) for fragment in fragments]
    boxes = [box for box in clipped if box is not None]
    xs = sorted({gt.left, gt.right, *(x for box in boxes for x in (box.left, box.right))})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((box.top, box.bottom) for box in boxes if box.left < right and box.right > left)
        covered = 0.0
        current_top = None
        current_bottom = None
        for top, bottom in intervals:
            if current_top is None:
                current_top, current_bottom = top, bottom
            elif top <= current_bottom:
                current_bottom = max(current_bottom, bottom)
            else:
                covered += current_bottom - current_top
                current_top, current_bottom = top, bottom
        if current_top is not None and current_bottom is not None:
            covered += current_bottom - current_top
        area += covered * (right - left)
    return area / gt.area


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": ordered[len(ordered) // 2],
        "p10": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
    }


def _f1(tp: int, fp: int, fn: int) -> float | None:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision is None or recall is None:
        return None
    return _harmonic(precision, recall)


def _event_f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    if tp == 0:
        return 0.0
    return _f1(tp, fp, fn)


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right == 0.0 else 2.0 * left * right / (left + right)


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvaluationError(f"{key} must be an object.")
    return value


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise EvaluationError(f"{key} must be a string.")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{key} must be numeric.")
    return float(value)


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{key} must be an integer.")
    return value
