"""Run the configured Grounding DINO, SAM2, and DINOv2 OCID pipeline.

The command has separate check, RGB-only inference, and ground-truth
evaluation phases.  There are no CLI parameters for model or threshold
selection. The configured pipeline applies the same prediction-only object
refinement to sample streams and complete dataset runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from stream_analysis.candidates import (
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    DEFAULT_MASK_CONTAINMENT,
    DEFAULT_MINIMUM_COVERED_MASKS,
    PositionedCandidateMask,
    candidate_mask_digest,
    resolve_aggregate_masks,
)
from stream_analysis.contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    FrameCandidateDiagnostics,
    GeometryFeatureMetadata,
    MaskReference,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
    ValidityStatus,
)
from stream_analysis.evaluation import (
    PredictedCandidate,
    aggregate_candidate_metrics,
    binary_mask_sha256,
    evaluate_candidate_predictions,
    load_annotation,
)
from stream_analysis.input import ManifestLoadRequest, load_decoded_stream
from stream_analysis.matching import (
    EventConfig,
    GroupingConfig,
    MatchingConfig,
    PairScoringConfig,
)
from stream_analysis.representations import (
    DINO_MASK_NEUTRAL_VARIANT,
)
from stream_analysis.serialization import to_json_compatible
from stream_analysis.runtime.config import (
    resolve_runtime_assets,
    runtime_profile_from_sections,
)
from stream_analysis.runtime.engine import (
    detector_profiles as runtime_detector_profiles,
    execute_pipeline,
    matching_configs as runtime_matching_configs,
    temporal_detector_records as runtime_temporal_detector_records,
)
from stream_analysis.runtime.workspace import RuntimeWorkspace

from tools import evaluate_ocid_grounding_dino_hardening as hardening
from tools.evaluate_ocid_masked_dinov2 import (
    CandidateInput,
    _assignment_identity,
)
from tools.evaluate_ocid_grounded_sam2_masks import (
    OVERLAY_PALETTE,
    _reviewed_mask,
    _same_bbox,
    evaluate_masks,
    mask_bbox_from_full_frame,
)
from tools.evaluate_ocid_oracle_dinov2 import (
    MatcherPolicy,
    OracleInstance,
    PreparedOracleInstance,
    aggregate_oracle_matcher_reports,
    evaluate_oracle_embeddings,
    evaluate_oracle_matcher,
)
from tools.ocid_pipeline_contract import (
    DEFAULT_PROTOCOL,
    OcidPipelineProtocol,
    OcidPipelineError,
    OcidStream,
    claim_evaluation_access,
    git_state,
    inventory_for_role,
    load_pipeline_protocol,
    build_check_report,
    sha256_file,
    validate_analysis_inputs,
    validate_evaluation_access,
    validate_evaluation_access_consumption,
    validate_evaluation_inputs,
    validate_run_id,
    write_json_atomic,
    write_json_exclusive,
)
INFERENCE_SCHEMA = "ocid-pipeline-inference-manifest-1.0"
DETECTOR_SCHEMA = "ocid-pipeline-detector-artifacts-1.0"
SAM2_SCHEMA = "ocid-pipeline-sam2-artifacts-1.0"
ANALYSIS_SCHEMA = "ocid-pipeline-analysis-artifacts-1.0"
EVALUATION_SCHEMA = "ocid-pipeline-evaluation-1.0"
ATTEMPT_SCHEMA = "ocid-pipeline-attempt-1.0"
INPUT_RECEIPT_SCHEMA = "ocid-pipeline-analysis-input-receipt-1.0"
IMPLEMENTATION_RECEIPT_SCHEMA = "ocid-pipeline-implementation-receipt-1.0"
DISPLAY_BUNDLE_SCHEMA = "ocid-pipeline-candidate-overlay-bundle-1.0"
_MUTABLE_FILENAMES = frozenset({"attempt.json", "inference_manifest.json"})


def run_check(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_pipeline_protocol(protocol_path)
    receipt = build_check_report(protocol)
    write_json_exclusive(output_path, receipt)
    return receipt


def run_inference(
    *,
    protocol_path: Path,
    role: str,
    output_root: Path,
    run_id: str,
    access_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_pipeline_protocol(protocol_path)
    if role not in {"sample", "development", "heldout"}:
        raise OcidPipelineError("inference role must be sample, development, or heldout")
    run_id = validate_run_id(run_id)
    inventory = inventory_for_role(protocol, role)  # type: ignore[arg-type]
    access: Mapping[str, Any] | None = None
    if role == "heldout":
        if access_path is None:
            raise OcidPipelineError("held-out inference requires an evaluation access artifact")
        access = validate_evaluation_access(access_path, protocol, expected_run_id=run_id)
    elif access_path is not None:
        raise OcidPipelineError("non-heldout analysis does not accept evaluation access")

    output_directory = Path(output_root).resolve(strict=False)
    run_directory = (output_directory / run_id).resolve(strict=False)
    try:
        run_directory.relative_to(output_directory)
    except ValueError as error:
        raise OcidPipelineError("OCID pipeline run directory escapes the output root") from error
    if run_directory.exists():
        raise OcidPipelineError(f"OCID pipeline run directory already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    state = git_state(protocol.repository_root)
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "run_id": run_id,
        "role": role,
        "protocol_sha256": protocol.sha256,
        "execution_git_commit": state.commit,
        "execution_git_branch": state.branch,
        "status": "started_before_rgb_access",
        "started_at_utc": started_at,
        "automatic_retry": False,
        "ground_truth_opened": False,
    }
    if access_path is not None:
        attempt["evaluation_access"] = {
            "sha256": sha256_file(Path(access_path).resolve(strict=True)),
            "decision": access["decision"] if access is not None else None,
        }
    write_json_exclusive(run_directory / "attempt.json", attempt)

    try:
        if role == "heldout":
            state_after_attempt = git_state(protocol.repository_root)
            if state_after_attempt != state or not state_after_attempt.worktree_clean:
                raise OcidPipelineError(
                    "evaluation output and access artifacts must remain outside the Git worktree"
                )
            assert access_path is not None
            claimed = claim_evaluation_access(
                access_path,
                protocol,
                expected_run_id=run_id,
                run_directory=run_directory,
            )
            consumption_path = Path(claimed["consumption_path"])
            attempt["evaluation_access"]["consumption_receipt_sha256"] = sha256_file(
                consumption_path
            )
            attempt["evaluation_access"]["consumed_before_rgb_access"] = True
            attempt["status"] = "evaluation_access_consumed_before_rgb_access"
            write_json_atomic(run_directory / "attempt.json", attempt)
        analysis_inputs = validate_analysis_inputs(protocol, inventory)
        input_receipt = {
            "schema_version": INPUT_RECEIPT_SCHEMA,
            "run_id": run_id,
            "role": role,
            "protocol_sha256": protocol.sha256,
            **analysis_inputs,
        }
        write_json_exclusive(run_directory / "analysis_input_receipt.json", input_receipt)
        implementation_receipt = _implementation_receipt(protocol)
        write_json_exclusive(
            run_directory / "implementation_receipt.json",
            implementation_receipt,
        )
        attempt["status"] = "rgb_inputs_verified"
        attempt["rgb_input_receipt_sha256"] = sha256_file(run_directory / "analysis_input_receipt.json")
        write_json_atomic(run_directory / "attempt.json", attempt)

        detector_manifest, sam2_manifest, analysis_manifest = _run_runtime_pipeline(
            protocol,
            inventory,
            role,
            run_directory,
        )
        artifacts = _artifact_inventory(run_directory)
        inference = {
            "schema_version": INFERENCE_SCHEMA,
            "status": "inference_completed_before_ground_truth",
            "run_id": run_id,
            "role": role,
            "protocol": {
                "path": protocol.path.relative_to(protocol.repository_root).as_posix(),
                "sha256": protocol.sha256,
                "protocol_id": protocol.payload["protocol_id"],
            },
            "git": {"commit": state.commit, "branch": state.branch},
            "evaluation_access": attempt.get("evaluation_access") if role == "heldout" else None,
            "inventory": {
                "stream_ids": [item.stream_id for item in inventory],
                "stream_count": len(inventory),
                "frame_count": sum(item.frame_count for item in inventory),
            },
            "stage_manifests": {
                "analysis_inputs": _file_reference(run_directory, run_directory / "analysis_input_receipt.json"),
                "implementation": _file_reference(run_directory, run_directory / "implementation_receipt.json"),
                "detector": _file_reference(run_directory, run_directory / "detector_manifest.json"),
                "sam2": _file_reference(run_directory, run_directory / "sam2_manifest.json"),
                "analysis": _file_reference(run_directory, run_directory / "analysis_manifest.json"),
            },
            "coverage": {
                "detector_candidate_count": detector_manifest["candidate_count"],
                "candidate_count": sam2_manifest["resolved_candidate_count"],
                "valid_mask_count": sam2_manifest["valid_mask_count"],
                "resolved_mask_count": sam2_manifest["resolved_candidate_count"],
                "removed_aggregate_count": sam2_manifest["removed_aggregate_count"],
                "embedding_count": analysis_manifest["embedding_count"],
                "mask_fallback_count": sam2_manifest["fallback_count"],
                "embedding_failure_count": analysis_manifest["embedding_failure_count"],
            },
            "artifact_inventory": artifacts,
            "ground_truth_boundary": {
                "ground_truth_opened": False,
                "evaluation_annotation_paths_persisted": False,
                "all_prediction_artifacts_persisted": True,
            },
            "completed_at_utc": _utc_now(),
        }
        write_json_exclusive(run_directory / "inference_manifest.json", inference)
        attempt["status"] = "inference_completed_before_ground_truth"
        attempt["completed_at_utc"] = inference["completed_at_utc"]
        attempt["inference_manifest_sha256"] = sha256_file(run_directory / "inference_manifest.json")
        write_json_atomic(run_directory / "attempt.json", attempt)
        return inference
    except BaseException as error:
        attempt["status"] = "failed_after_attempt_started"
        attempt["failed_at_utc"] = _utc_now()
        attempt["failure"] = {
            "error_type": type(error).__name__,
            "message": str(error).strip() or type(error).__name__,
        }
        write_json_atomic(run_directory / "attempt.json", attempt)
        raise


def run_evaluation(
    *,
    protocol_path: Path,
    run_directory: Path,
    access_path: Path,
) -> dict[str, Any]:
    protocol = load_pipeline_protocol(protocol_path)
    return _run_evaluation(
        protocol=protocol,
        run_directory=run_directory,
        expected_role="heldout",
        inventory=protocol.heldout,
        output_filename="evaluation.json",
        completion_status="completed_one_time_heldout_evaluation",
        attempt_status="heldout_evaluation_completed",
        access_path=access_path,
    )


def run_sample_evaluation(
    *,
    protocol_path: Path,
    run_directory: Path,
) -> dict[str, Any]:
    protocol = load_pipeline_protocol(protocol_path)
    return _run_evaluation(
        protocol=protocol,
        run_directory=run_directory,
        expected_role="sample",
        inventory=inventory_for_role(protocol, "sample"),
        output_filename="sample_evaluation.json",
        completion_status="completed_sample_evaluation",
        attempt_status="sample_evaluation_completed",
        access_path=None,
    )


def run_development_evaluation(
    *,
    protocol_path: Path,
    run_directory: Path,
) -> dict[str, Any]:
    """Evaluate a completed full-dataset run from persisted inference artifacts."""

    protocol = load_pipeline_protocol(protocol_path)
    return _run_evaluation(
        protocol=protocol,
        run_directory=run_directory,
        expected_role="development",
        inventory=inventory_for_role(protocol, "development"),
        output_filename="development_evaluation.json",
        completion_status="completed_development_evaluation",
        attempt_status="development_evaluation_completed",
        access_path=None,
    )


def render_development_candidate_overlays(
    *,
    protocol_path: Path,
    run_directory: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Render labels-only overlays from one completed development inference."""

    protocol = load_pipeline_protocol(protocol_path)
    run_root = Path(run_directory).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    try:
        destination.relative_to(protocol.repository_root)
    except ValueError as error:
        raise OcidPipelineError("overlay output must stay inside the repository") from error
    forbidden = {part.casefold().replace("-", "_") for part in destination.parts}
    if any("heldout" in part.replace("_", "") for part in forbidden):
        raise OcidPipelineError("overlay output points to a forbidden held-out location")
    if destination.exists():
        raise OcidPipelineError(f"overlay output already exists: {destination}")

    inference_path = run_root / "inference_manifest.json"
    attempt = _json_object(run_root / "attempt.json", "OCID pipeline attempt")
    if attempt.get("inference_manifest_sha256") != sha256_file(inference_path):
        raise OcidPipelineError("inference manifest changed after inference completion")
    inference = _json_object(inference_path, "OCID pipeline inference manifest")
    inventory = inventory_for_role(protocol, "development")
    if (
        inference.get("schema_version") != INFERENCE_SCHEMA
        or inference.get("role") != "development"
        or _mapping(inference, "protocol", "inference manifest").get("sha256")
        != protocol.sha256
        or _mapping(inference, "inventory", "inference manifest").get("stream_ids")
        != [item.stream_id for item in inventory]
    ):
        raise OcidPipelineError("overlay source is not the complete development inference")
    detector = _verified_stage_manifest(
        run_root, inference, "detector", DETECTOR_SCHEMA
    )
    sam2 = _verified_stage_manifest(run_root, inference, "sam2", SAM2_SCHEMA)
    analysis = _verified_stage_manifest(
        run_root, inference, "analysis", ANALYSIS_SCHEMA
    )
    _verify_stage_contracts(
        protocol,
        "development",
        inventory,
        inference,
        detector,
        sam2,
        analysis,
    )
    snapshots = _snapshots_from_persisted_masks(
        protocol,
        inventory,
        run_root,
        detector,
        sam2,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        streams: dict[str, Any] = {}
        for item in inventory:
            decoded = _load_stream(item, protocol.sha256)
            streams[item.stream_id] = _render_candidate_overlays(
                decoded,
                snapshots[item.stream_id],
                sam2["records_by_stream"][item.stream_id],
                run_root,
                output_root=staging,
            )
        bundle = {
            "schema_version": DISPLAY_BUNDLE_SCHEMA,
            "status": "completed_from_persisted_development_inference",
            "source_run": run_root.relative_to(protocol.repository_root).as_posix(),
            "source_inference_sha256": sha256_file(inference_path),
            "renderer_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "stream_count": len(inventory),
            "frame_count": sum(item.frame_count for item in inventory),
            "confidence_displayed": False,
            "label_format": "Pnn_frame_local_ordinal_only",
            "streams": streams,
            "artifact_inventory": _artifact_inventory(staging),
        }
        write_json_exclusive(staging / "manifest.json", bundle)
        if destination.exists():
            raise OcidPipelineError("overlay output appeared during rendering")
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return bundle


def _run_evaluation(
    *,
    protocol: OcidPipelineProtocol,
    run_directory: Path,
    expected_role: str,
    inventory: Sequence[OcidStream],
    output_filename: str,
    completion_status: str,
    attempt_status: str,
    access_path: Path | None,
) -> dict[str, Any]:
    run_root = Path(run_directory).resolve(strict=True)
    inference_path = run_root / "inference_manifest.json"
    attempt_path = run_root / "attempt.json"
    attempt = _json_object(attempt_path, "OCID pipeline attempt")
    if attempt.get("inference_manifest_sha256") != sha256_file(inference_path):
        raise OcidPipelineError("inference manifest changed after inference completion")
    inference = _json_object(inference_path, "OCID pipeline inference manifest")
    if inference.get("schema_version") != INFERENCE_SCHEMA:
        raise OcidPipelineError("unsupported OCID pipeline inference manifest schema")
    if inference.get("status") != "inference_completed_before_ground_truth":
        raise OcidPipelineError("OCID pipeline evaluation requires completed persisted inference")
    if inference.get("role") != expected_role:
        raise OcidPipelineError(f"OCID pipeline evaluation expected role {expected_role}")
    run_id = _text(inference, "run_id", "inference manifest")
    authorization: Mapping[str, Any] | None = None
    if expected_role == "heldout":
        if access_path is None:
            raise OcidPipelineError("held-out evaluation requires the evaluation access")
        authorization = validate_evaluation_access_consumption(
            access_path,
            protocol,
            expected_run_id=run_id,
            expected_run_directory=run_root,
        )
    elif access_path is not None:
        raise OcidPipelineError("non-heldout evaluation does not accept an access file")
    if _mapping(inference, "protocol", "inference manifest").get("sha256") != protocol.sha256:
        raise OcidPipelineError("inference manifest protocol digest mismatch")
    state = git_state(protocol.repository_root)
    if _mapping(inference, "git", "inference manifest").get("commit") != state.commit:
        raise OcidPipelineError("evaluation Git commit differs from inference")
    expected_streams = [item.stream_id for item in inventory]
    if _mapping(inference, "inventory", "inference manifest").get("stream_ids") != expected_streams:
        raise OcidPipelineError("inference manifest evaluation inventory mismatch")
    boundary = _mapping(inference, "ground_truth_boundary", "inference manifest")
    if (
        boundary.get("ground_truth_opened") is not False
        or boundary.get("all_prediction_artifacts_persisted") is not True
    ):
        raise OcidPipelineError("persisted inference does not satisfy the ground-truth boundary")
    _verify_artifact_inventory(run_root, inference.get("artifact_inventory"))
    output_path = run_root / output_filename
    if output_path.exists():
        raise OcidPipelineError("refusing to overwrite or rerun a completed OCID pipeline evaluation")

    analysis_input_receipt = _verified_stage_manifest(
        run_root,
        inference,
        "analysis_inputs",
        INPUT_RECEIPT_SCHEMA,
    )
    implementation_receipt = _verified_stage_manifest(
        run_root,
        inference,
        "implementation",
        IMPLEMENTATION_RECEIPT_SCHEMA,
    )
    _verify_implementation_receipt(protocol, implementation_receipt)
    _verify_analysis_inputs_unchanged(
        protocol,
        inventory,
        run_id,
        expected_role,
        analysis_input_receipt,
    )
    detector_manifest = _verified_stage_manifest(run_root, inference, "detector", DETECTOR_SCHEMA)
    sam2_manifest = _verified_stage_manifest(run_root, inference, "sam2", SAM2_SCHEMA)
    analysis_manifest = _verified_stage_manifest(run_root, inference, "analysis", ANALYSIS_SCHEMA)
    _verify_stage_contracts(
        protocol,
        expected_role,
        inventory,
        inference,
        detector_manifest,
        sam2_manifest,
        analysis_manifest,
    )
    snapshots = _snapshots_from_persisted_masks(
        protocol,
        inventory,
        run_root,
        detector_manifest,
        sam2_manifest,
    )
    embedding_matrices = _verify_analysis_embeddings(
        protocol,
        inventory,
        run_root,
        snapshots,
        analysis_manifest,
    )
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("run_id") != run_id
        or attempt.get("role") != expected_role
        or attempt.get("protocol_sha256") != protocol.sha256
        or attempt.get("status") != "inference_completed_before_ground_truth"
        or attempt.get("ground_truth_opened") is not False
    ):
        raise OcidPipelineError("OCID pipeline attempt is not eligible to cross the ground-truth boundary")
    if expected_role == "heldout":
        assert authorization is not None
        access_file = Path(access_path).resolve(strict=True)  # type: ignore[arg-type]
        consumption_file = Path(authorization["consumption_path"])
        expected_authorization = {
            "sha256": sha256_file(access_file),
            "decision": authorization["access"]["decision"],
            "consumption_receipt_sha256": sha256_file(consumption_file),
            "consumed_before_rgb_access": True,
        }
        if (
            attempt.get("evaluation_access") != expected_authorization
            or inference.get("evaluation_access") != expected_authorization
        ):
            raise OcidPipelineError("evaluation access record differs from inference")
    attempt["status"] = "evaluation_started_before_ground_truth_access"
    attempt["ground_truth_opened"] = True
    attempt["ground_truth_boundary_crossed_at_utc"] = _utc_now()
    write_json_atomic(attempt_path, attempt)
    try:
        evaluation_inputs = validate_evaluation_inputs(protocol, inventory)
        candidate_metrics = _evaluate_candidate_metrics(protocol, inventory, snapshots)
        mask_metrics = _evaluate_mask_diagnostics(
            protocol,
            inventory,
            run_root,
            snapshots,
            sam2_manifest,
        )
        continuity = _evaluate_physical_continuity(
            protocol,
            inventory,
            snapshots,
            embedding_matrices,
        )
        event_support_path = protocol.reviewed_root / "event_support_matrix.json"
        event_support = _json_object(event_support_path, "event support matrix")
        report = {
            "schema_version": EVALUATION_SCHEMA,
            "status": completion_status,
            "role": expected_role,
            "run_id": run_id,
            "protocol_sha256": protocol.sha256,
            "inference_manifest": {
                "path": "inference_manifest.json",
                "sha256": sha256_file(inference_path),
            },
            "evaluation_inputs": evaluation_inputs,
            "candidate_metrics": candidate_metrics,
            "mask_diagnostics": mask_metrics,
            "physical_continuity": continuity,
            "event_support": {
                "matrix_sha256": sha256_file(event_support_path),
                "matrix": event_support,
                "model_event_accuracy": "not_scored_without_real_image_visual_type_ground_truth",
            },
            "limitations": [
                "OCID supplies physical-instance proxies, not real-image visual-type ground truth.",
                "Grouping and event artifacts are qualitative model outputs and are not visual-type accuracy metrics.",
                "Physical removal, return after removal, and object motion are not supported by this benchmark.",
                "Conditional mask diagnostics include only bbox-matched detections; use the adjacent end-to-end mask assignment for full detection-inclusive quality.",
                "Continuity metrics are conditional on candidates assigned to physical-instance proxies.",
                "Evaluation reads persisted predictions and does not modify model, threshold, or post-processing configuration.",
            ],
            "completed_at_utc": _utc_now(),
        }
        write_json_exclusive(output_path, report)
    except BaseException as error:
        attempt["status"] = "evaluation_failed_after_ground_truth_boundary"
        attempt["evaluation_failed_at_utc"] = _utc_now()
        attempt["evaluation_failure"] = {
            "error_type": type(error).__name__,
            "message": str(error).strip() or type(error).__name__,
        }
        write_json_atomic(attempt_path, attempt)
        raise
    attempt["status"] = attempt_status
    attempt["evaluation_completed_at_utc"] = report["completed_at_utc"]
    attempt["evaluation_artifact"] = output_filename
    attempt["evaluation_sha256"] = sha256_file(output_path)
    write_json_atomic(attempt_path, attempt)
    return report


def _run_runtime_pipeline(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    role: str,
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute the shared runtime and persist the detailed analysis artifacts."""

    profile = runtime_profile_from_sections(
        profile_id="visual_stream_analyzer_v1",
        models=_mapping(protocol.payload, "models", "protocol"),
        pipeline=_mapping(protocol.payload, "pipeline", "protocol"),
        paths_include_models_prefix=True,
    )
    assets = resolve_runtime_assets(
        profile,
        protocol.repository_root / "models",
        verify=False,
    )
    decoded_streams = tuple(_load_stream(item, protocol.sha256) for item in inventory)
    outcome = execute_pipeline(
        decoded_streams,
        profile=profile,
        assets=assets,
        workspace=RuntimeWorkspace(run_directory),
        record_namespace="ocid-pipeline",
        candidate_source="grounding_dino_sam2_ocid_pipeline_v1",
    )
    inventory_by_stream = {item.stream_id: item for item in inventory}

    detector_streams: dict[str, Any] = {}
    total_raw = 0
    total_candidates = 0
    total_rejected = 0
    total_supplemental = 0
    for stream in outcome.streams:
        detector = stream.detector
        stream_id = detector.decoded.stream.stream_id
        item = inventory_by_stream[stream_id]
        payload = {
            "schema_version": "ocid-pipeline-detector-stream-1.0",
            "stream_id": stream_id,
            "scene_group_id": item.scene_group_id,
            "frame_count": len(detector.decoded.frames),
            "raw_predictions": [hardening._raw_payload(row) for row in detector.raw_predictions],
            "postprocess_records": [dict(row) for row in detector.postprocess_records],
            "accepted_candidates": [dict(row) for row in detector.accepted_candidates],
            "temporal_support": {
                **dict(detector.temporal_support),
                "ground_truth_used_for_selection": False,
            },
            "runtime": dict(detector.runtime),
        }
        path = run_directory / "detector" / f"{stream_id}.json"
        write_json_exclusive(path, payload)
        detector_streams[stream_id] = _file_reference(run_directory, path)
        total_raw += len(detector.raw_predictions)
        total_candidates += len(detector.accepted_candidates)
        total_rejected += sum(
            1 for row in detector.postprocess_records if row["geometry_rejected"] is True
        )
        total_supplemental += int(detector.temporal_support["supplemental_candidate_count"])
    detector_model = _mapping(
        _mapping(protocol.payload, "models", "protocol"),
        "candidate_extractor",
        "models",
    )
    detector_manifest = {
        "schema_version": DETECTOR_SCHEMA,
        "status": "detector_inference_completed_before_ground_truth",
        "role": role,
        "protocol_sha256": protocol.sha256,
        "configuration": protocol.payload["pipeline"]["candidate_extraction"],
        "model": {
            "family": detector_model["family"],
            "hf_revision": detector_model["hf_revision"],
            "model_safetensors_sha256": detector_model["assets"]["model.safetensors"]["sha256"],
            "device": "cuda",
            "dtype": "float32",
            "local_files_only": True,
        },
        "streams": detector_streams,
        "stream_count": len(detector_streams),
        "frame_count": sum(item.frame_count for item in inventory),
        "raw_prediction_count": total_raw,
        "candidate_count": total_candidates,
        "base_candidate_count": total_candidates - total_supplemental,
        "supplemental_candidate_count": total_supplemental,
        "geometry_rejected_count": total_rejected,
        "elapsed_seconds": outcome.detector_elapsed_seconds,
        "ground_truth_opened": False,
    }
    write_json_exclusive(run_directory / "detector_manifest.json", detector_manifest)

    records_by_stream: dict[str, list[dict[str, Any]]] = {}
    resolved_by_stream: dict[str, list[str]] = {}
    resolution_by_stream: dict[str, Any] = {}
    valid_count = 0
    resolved_count = 0
    removed_count = 0
    for stream in outcome.streams:
        mask_outcome = stream.masks
        stream_id = mask_outcome.decoded.stream.stream_id
        rows: list[dict[str, Any]] = []
        for record in mask_outcome.records:
            result = record.result
            assert (
                result.raw_mask is not None
                and result.cleaned_mask is not None
                and result.mask_bbox is not None
                and result.quality is not None
            )
            raw_path = record.artifacts.raw_path
            cleaned_path = record.artifacts.cleaned_path
            quality = result.quality
            rows.append(
                {
                    "stream_id": stream_id,
                    "candidate_id": record.candidate_id,
                    "frame_id": record.frame_id,
                    "frame_index": record.frame_index,
                    "score": record.score,
                    "phrase": record.phrase,
                    "source_bbox": _bbox_payload(record.source_bbox),
                    "status": "valid",
                    "fallback_reason": None,
                    "mask_bbox": _bbox_payload(result.mask_bbox),
                    "raw_mask": {
                        "path": raw_path.relative_to(run_directory).as_posix(),
                        "file_sha256": sha256_file(raw_path),
                        "binary_mask_sha256": binary_mask_sha256(result.raw_mask),
                    },
                    "cleaned_mask": {
                        "path": cleaned_path.relative_to(run_directory).as_posix(),
                        "file_sha256": sha256_file(cleaned_path),
                        "binary_mask_sha256": binary_mask_sha256(result.cleaned_mask),
                    },
                    "quality": {
                        "selected_mask_index": quality.selected_mask_index,
                        "predicted_iou": quality.predicted_iou,
                        "object_score_logit": quality.object_score_logit,
                        "raw_foreground_pixels": quality.raw_foreground_pixels,
                        "cleaned_foreground_pixels": quality.cleaned_foreground_pixels,
                        "removed_component_count": quality.removed_component_count,
                        "removed_foreground_pixels": quality.removed_foreground_pixels,
                    },
                    "details": dict(result.details),
                }
            )
        rows.sort(key=lambda row: (int(row["frame_index"]), str(row["candidate_id"])))
        records_by_stream[stream_id] = rows
        resolved_by_stream[stream_id] = list(mask_outcome.resolved_candidate_ids)
        resolution_by_stream[stream_id] = {
            **dict(mask_outcome.aggregate_resolution),
            "ground_truth_used_for_selection": False,
        }
        valid_count += len(rows)
        resolved_count += len(mask_outcome.resolved_candidate_ids)
        removed_count += int(mask_outcome.aggregate_resolution["removed_candidate_count"])
    sam2_model = _mapping(
        _mapping(protocol.payload, "models", "protocol"),
        "mask_refiner",
        "models",
    )
    sam2_manifest = {
        "schema_version": SAM2_SCHEMA,
        "status": "rgb_inference_completed_before_ground_truth",
        "scope": role,
        "heldout_access": "evaluation_access_granted" if role == "heldout" else "none",
        "protocol_sha256": protocol.sha256,
        "configuration": protocol.payload["pipeline"]["mask_refinement"],
        "model": {
            "family": sam2_model["family"],
            "model_safetensors_sha256": sam2_model["assets"]["model.safetensors"]["sha256"],
            "device": "cuda",
            "dtype": "float32",
            "local_files_only": True,
        },
        "records_by_stream": records_by_stream,
        "resolved_candidate_ids_by_stream": resolved_by_stream,
        "aggregate_resolution_by_stream": resolution_by_stream,
        "candidate_count": total_candidates,
        "valid_mask_count": valid_count,
        "resolved_candidate_count": resolved_count,
        "removed_aggregate_count": removed_count,
        "fallback_count": 0,
        "elapsed_seconds": outcome.mask_elapsed_seconds,
        "ground_truth_opened": False,
    }
    write_json_exclusive(run_directory / "sam2_manifest.json", sam2_manifest)

    analysis_streams: dict[str, Any] = {}
    total_embeddings = 0
    representation_model = _mapping(
        _mapping(protocol.payload, "models", "protocol"),
        "representation",
        "models",
    )
    for stream in outcome.streams:
        analysis = stream.analysis
        decoded = analysis.decoded
        stream_id = decoded.stream.stream_id
        candidate_ids = [
            candidate.candidate_id for candidate in analysis.snapshot.result.candidates
        ]
        embedding_path = analysis.embedding_artifact.path
        embedding_reference = _file_reference(run_directory, embedding_path)
        first_record = analysis.representation_batch.records[0] if candidate_ids else None
        representation = {
            "variant": DINO_MASK_NEUTRAL_VARIANT,
            "semantic_config_digest": analysis.representation_batch.semantic_config_digest,
            "requested_device": analysis.representation_batch.requested_device,
            "resolved_device": analysis.representation_batch.resolved_device,
            "provider": (
                _portable_runtime_metadata(first_record.provider_metadata.details)
                if first_record is not None
                else {}
            ),
            "model": (
                _portable_runtime_metadata(first_record.model_metadata.details)
                if first_record is not None and first_record.model_metadata is not None
                else {}
            ),
            "warning_count": len(analysis.representation_batch.warnings),
            "error_count": len(analysis.representation_batch.errors),
        }
        stream_payload = {
            "schema_version": "ocid-pipeline-stream-analysis-1.0",
            "stream_id": stream_id,
            "candidate_ids": candidate_ids,
            "embedding_artifact": embedding_reference,
            "embedding_shape": list(analysis.embeddings.shape),
            "representation": representation,
            "matching_results": to_json_compatible(analysis.matching_results),
            "grouping": to_json_compatible(analysis.grouping),
            "frame_comparisons": to_json_compatible(analysis.event_batch.comparisons),
            "change_events": to_json_compatible(analysis.event_batch.events),
            "event_warning_count": len(analysis.event_batch.warnings),
            "event_error_count": len(analysis.event_batch.errors),
        }
        path = run_directory / "analysis" / stream_id / "analysis.json"
        write_json_exclusive(path, stream_payload)
        overlays = _render_mask_type_overlays(
            decoded,
            analysis.snapshot,
            analysis.grouping,
            records_by_stream[stream_id],
            run_directory,
        )
        candidate_overlays = _render_candidate_overlays(
            decoded,
            analysis.snapshot,
            records_by_stream[stream_id],
            run_directory,
        )
        analysis_streams[stream_id] = {
            "analysis": _file_reference(run_directory, path),
            "embedding_artifact": embedding_reference,
            "overlay_manifest": overlays,
            "candidate_overlay_manifest": candidate_overlays,
            "candidate_count": len(candidate_ids),
            "embedding_count": len(candidate_ids),
            "matching_comparison_count": len(analysis.matching_results),
            "group_count": len(analysis.grouping.recurring_types),
            "event_count": len(analysis.event_batch.events),
        }
        total_embeddings += len(candidate_ids)
    analysis_manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "annotation_free_analysis_completed_before_ground_truth",
        "role": role,
        "protocol_sha256": protocol.sha256,
        "configuration": {
            "representation": protocol.payload["pipeline"]["representation"],
            "scoring": protocol.payload["pipeline"]["scoring"],
            "matching": protocol.payload["pipeline"]["matching"],
            "grouping": protocol.payload["pipeline"]["grouping"],
            "events": protocol.payload["pipeline"]["events"],
        },
        "model": {
            "model_name": representation_model["model_name"],
            "checkpoint_sha256": representation_model["checkpoint"]["sha256"],
            "source_tree_fingerprint_sha256": representation_model[
                "source_tree_fingerprint_sha256"
            ],
            "embedding_dimension": representation_model["embedding_dimension"],
        },
        "streams": analysis_streams,
        "input_candidate_count": total_candidates,
        "candidate_count": resolved_count,
        "embedding_count": total_embeddings,
        "embedding_failure_count": 0,
        "elapsed_seconds": outcome.analysis_elapsed_seconds,
        "ground_truth_opened": False,
    }
    if total_embeddings != resolved_count:
        raise OcidPipelineError("runtime embedding coverage differs from resolved coverage")
    write_json_exclusive(run_directory / "analysis_manifest.json", analysis_manifest)
    return detector_manifest, sam2_manifest, analysis_manifest


def _portable_runtime_metadata(details: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-local paths and hardware names from persisted metadata."""

    private_keys = {"checkpoint_path", "gpu_name", "source_dir"}
    return {str(key): value for key, value in details.items() if key not in private_keys}


def _temporal_detector_records(
    *,
    raw: Sequence[Any],
    decoded: Any,
    prompt: hardening.PromptProfile,
    geometry: hardening.GeometryProfile,
    baseline_accepted: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt shared temporal-support results to the detailed artifact schema."""

    accepted, audit = runtime_temporal_detector_records(
        raw=raw,
        decoded=decoded,
        prompt=prompt,
        geometry=geometry,
        baseline_accepted=baseline_accepted,
    )
    return [dict(row) for row in accepted], {
        **dict(audit),
        "ground_truth_used_for_selection": False,
    }


def _resolve_stream_aggregate_masks(
    *,
    run_directory: Path,
    stream_id: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Resolve global aggregate masks per frame and persist full evidence."""

    by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_frame[_text(row, "frame_id", "SAM2 record")].append(row)
    kept_ids: list[str] = []
    frame_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        rows = sorted(by_frame[frame_id], key=lambda row: str(row["candidate_id"]))
        positioned: list[PositionedCandidateMask] = []
        for row in rows:
            candidate_id = _text(row, "candidate_id", "SAM2 record")
            cleaned = _mapping(row, "cleaned_mask", "SAM2 record")
            full_mask = _load_binary_mask(
                _run_artifact_path(
                    run_directory,
                    _text(cleaned, "path", "cleaned mask"),
                )
            )
            positioned.append(
                PositionedCandidateMask(
                    candidate_id=candidate_id,
                    frame_id=frame_id,
                    mask=np.ascontiguousarray(full_mask, dtype=np.bool_),
                )
            )
        resolution = resolve_aggregate_masks(positioned)
        frame_kept = resolution.kept_candidate_ids
        kept_ids.extend(frame_kept)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "input_candidate_count": len(rows),
                "kept_candidate_ids": list(frame_kept),
                "removed_candidate_ids": list(resolution.removed_candidate_ids),
                "decisions": [
                    {
                        "candidate_id": decision.candidate_id,
                        "reason": decision.reason.value,
                        "mask_area_pixels": decision.mask_area_pixels,
                        "covered_masks": [
                            {
                                "candidate_id": evidence.candidate_id,
                                "intersection_pixels": evidence.intersection_pixels,
                                "candidate_mask_area_pixels": (
                                    evidence.candidate_mask_area_pixels
                                ),
                                "containment": evidence.containment,
                            }
                            for evidence in decision.covered_masks
                        ],
                    }
                    for decision in resolution.decisions
                ],
            }
        )
    all_ids = {str(row["candidate_id"]) for row in records}
    if len(kept_ids) != len(set(kept_ids)) or not set(kept_ids) <= all_ids:
        raise OcidPipelineError("aggregate resolution produced an invalid candidate set")
    removed_count = len(all_ids) - len(kept_ids)
    return tuple(kept_ids), {
        "enabled": True,
        "policy_id": "aggregate_mask_containment_v1",
        "mask_containment_at_least": DEFAULT_MASK_CONTAINMENT,
        "minimum_covered_masks": DEFAULT_MINIMUM_COVERED_MASKS,
        "ground_truth_used_for_selection": False,
        "input_candidate_count": len(records),
        "resolved_candidate_count": len(kept_ids),
        "removed_candidate_count": removed_count,
        "frames": frame_rows,
    }


def _snapshots_from_persisted_masks(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    run_directory: Path,
    detector_manifest: Mapping[str, Any],
    sam2_manifest: Mapping[str, Any],
) -> dict[str, CandidateExtractionSnapshot]:
    records_by_stream = sam2_manifest.get("records_by_stream")
    if not isinstance(records_by_stream, Mapping):
        raise OcidPipelineError("SAM2 manifest lacks records_by_stream")
    resolved_by_stream = sam2_manifest.get("resolved_candidate_ids_by_stream")
    if not isinstance(resolved_by_stream, Mapping):
        raise OcidPipelineError("SAM2 manifest lacks resolved candidate membership")
    resolution_by_stream = sam2_manifest.get("aggregate_resolution_by_stream")
    if not isinstance(resolution_by_stream, Mapping):
        raise OcidPipelineError("SAM2 manifest lacks aggregate resolution evidence")
    result: dict[str, CandidateExtractionSnapshot] = {}
    for item in inventory:
        decoded = _load_stream(item, protocol.sha256)
        detector = _load_referenced_json(run_directory, detector_manifest, item.stream_id)
        accepted = detector.get("accepted_candidates")
        mask_rows = records_by_stream.get(item.stream_id)
        resolved_ids = resolved_by_stream.get(item.stream_id)
        if (
            not isinstance(accepted, list)
            or not isinstance(mask_rows, list)
            or not isinstance(resolved_ids, list)
            or not all(isinstance(value, str) for value in resolved_ids)
        ):
            raise OcidPipelineError(f"persisted candidates or masks missing for {item.stream_id}")
        masks_by_id = {
            str(row["candidate_id"]): row
            for row in mask_rows
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        }
        if len(masks_by_id) != len(mask_rows):
            raise OcidPipelineError(f"duplicate or malformed mask rows for {item.stream_id}")
        accepted_by_id = {
            str(row["candidate_id"]): row
            for row in accepted
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        }
        if (
            len(accepted_by_id) != len(accepted)
            or len(resolved_ids) != len(set(resolved_ids))
            or not set(resolved_ids) <= set(accepted_by_id)
            or set(accepted_by_id) != set(masks_by_id)
        ):
            raise OcidPipelineError(
                f"resolved candidate membership is inconsistent for {item.stream_id}"
            )
        _verify_aggregate_resolution_membership(
            stream_id=item.stream_id,
            accepted_ids=set(accepted_by_id),
            resolved_ids=tuple(resolved_ids),
            payload=resolution_by_stream.get(item.stream_id),
        )
        accepted = [accepted_by_id[candidate_id] for candidate_id in resolved_ids]
        frame_lookup = {frame.frame_id: frame for frame in decoded.frames}
        producer = _producer("ocid_pipeline_candidate_extraction", protocol.sha256)
        candidates: list[CandidateRecord] = []
        mask_store: dict[str, CandidateMaskRecord] = {}
        per_frame_counts: dict[str, int] = defaultdict(int)
        for row in sorted(accepted, key=lambda value: (int(value["frame_index"]), str(value["candidate_id"]))):
            candidate_id = _text(row, "candidate_id", "detector candidate")
            frame_id = _text(row, "frame_id", "detector candidate")
            frame = frame_lookup.get(frame_id)
            if frame is None:
                raise OcidPipelineError(f"candidate references unknown frame: {candidate_id}")
            mask_row = masks_by_id.get(candidate_id)
            if mask_row is None or mask_row.get("status") != "valid":
                raise OcidPipelineError(f"candidate lacks a valid persisted mask: {candidate_id}")
            bbox = _bbox(_mapping(row, "bbox", "detector candidate"))
            if bbox != _bbox(_mapping(mask_row, "source_bbox", "SAM2 mask row")):
                raise OcidPipelineError(f"candidate/mask bbox mismatch: {candidate_id}")
            cleaned = _mapping(mask_row, "cleaned_mask", "SAM2 mask row")
            full_mask_path = _run_artifact_path(run_directory, _text(cleaned, "path", "cleaned mask"))
            if cleaned.get("file_sha256") != sha256_file(full_mask_path):
                raise OcidPipelineError(f"cleaned mask file digest mismatch: {candidate_id}")
            full_mask = _load_binary_mask(full_mask_path)
            if binary_mask_sha256(full_mask) != cleaned.get("binary_mask_sha256"):
                raise OcidPipelineError(f"cleaned mask binary digest mismatch: {candidate_id}")
            x, y, width, height = _integral_bbox(bbox)
            local_mask = np.ascontiguousarray(full_mask[y : y + height, x : x + width], dtype=np.bool_)
            if local_mask.shape != (height, width) or not local_mask.any():
                raise OcidPipelineError(f"invalid local mask crop: {candidate_id}")
            mask_ref = f"ocid-pipeline-mask:{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:24]}"
            mask_digest = candidate_mask_digest(local_mask)
            mask_record = CandidateMaskRecord(
                mask_ref=mask_ref,
                mask_digest=mask_digest,
                candidate_id=candidate_id,
                frame_id=frame_id,
                coordinate_bbox=bbox,
                mask=local_mask,
            )
            mask_store[mask_ref] = mask_record
            geometry = GeometryFeatureMetadata(
                feature_schema_id="grounding_dino_ocid_pipeline_geometry_v1",
                producer_version="1.0.0",
                config_digest=f"sha256:{protocol.sha256}",
                values={
                    "area": bbox.area,
                    "area_ratio": bbox.area / (frame.image_size.width * frame.image_size.height),
                    "width_ratio": bbox.width / frame.image_size.width,
                    "height_ratio": bbox.height / frame.image_size.height,
                },
            )
            candidate = CandidateRecord(
                envelope=RecordEnvelope(
                    record_id=candidate_id,
                    schema_version="ocid-pipeline-candidate-record-1.0",
                    stream_id=item.stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame_id, candidate_id=candidate_id),
                ),
                candidate_id=candidate_id,
                frame_id=frame_id,
                frame_index=frame.record.index,
                frame_size=frame.image_size,
                bbox=bbox,
                center=bbox.center,
                geometry=geometry,
                candidate_source="grounding_dino_sam2_ocid_pipeline_v1",
                mask=MaskReference(
                    mask_ref=mask_ref,
                    mask_digest=mask_digest,
                    producer_version="1.0.0",
                    coordinate_bbox=bbox,
                    validity_status=ValidityStatus.VALID,
                ),
                candidate_confidence=float(row["score"]),
            )
            candidates.append(candidate)
            per_frame_counts[frame_id] += 1
        diagnostics = tuple(
            FrameCandidateDiagnostics(
                envelope=RecordEnvelope(
                    record_id=f"ocid-pipeline-candidate-diagnostics:{item.stream_id}:{frame.frame_id}",
                    schema_version="ocid-pipeline-frame-candidate-diagnostics-1.0",
                    stream_id=item.stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame.frame_id),
                ),
                frame_id=frame.frame_id,
                frame_index=frame.record.index,
                image_size=frame.image_size,
                summary={"candidate_count": per_frame_counts.get(frame.frame_id, 0)},
            )
            for frame in decoded.frames
        )
        extraction = CandidateExtractionResult(
            envelope=RecordEnvelope(
                record_id=f"ocid-pipeline-candidate-extraction:{item.stream_id}",
                schema_version="ocid-pipeline-candidate-extraction-result-1.0",
                stream_id=item.stream_id,
                producer=producer,
                context=StageContext(),
            ),
            extractor_source="grounding_dino_sam2_ocid_pipeline_v1",
            candidates=tuple(candidates),
            frame_diagnostics=diagnostics,
        )
        result[item.stream_id] = CandidateExtractionSnapshot(result=extraction, masks=mask_store)
    return result


def _verify_aggregate_resolution_membership(
    *,
    stream_id: str,
    accepted_ids: set[str],
    resolved_ids: Sequence[str],
    payload: Any,
) -> None:
    if not isinstance(payload, Mapping):
        raise OcidPipelineError(f"aggregate resolution is missing for {stream_id}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise OcidPipelineError(f"aggregate resolution frames are malformed for {stream_id}")
    input_ids: list[str] = []
    kept_ids: list[str] = []
    removed_ids: list[str] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise OcidPipelineError("aggregate resolution frame must be an object")
        frame_kept = frame.get("kept_candidate_ids")
        frame_removed = frame.get("removed_candidate_ids")
        if (
            not isinstance(frame_kept, list)
            or not isinstance(frame_removed, list)
            or not all(isinstance(value, str) for value in (*frame_kept, *frame_removed))
            or set(frame_kept) & set(frame_removed)
            or int(frame.get("input_candidate_count", -1))
            != len(frame_kept) + len(frame_removed)
        ):
            raise OcidPipelineError("aggregate resolution frame partition is invalid")
        input_ids.extend((*frame_kept, *frame_removed))
        kept_ids.extend(frame_kept)
        removed_ids.extend(frame_removed)
    if (
        len(input_ids) != len(set(input_ids))
        or set(input_ids) != accepted_ids
        or set(kept_ids) != set(resolved_ids)
        or len(kept_ids) != len(resolved_ids)
        or int(payload.get("input_candidate_count", -1)) != len(accepted_ids)
        or int(payload.get("resolved_candidate_count", -1)) != len(resolved_ids)
        or int(payload.get("removed_candidate_count", -1)) != len(removed_ids)
    ):
        raise OcidPipelineError(
            f"aggregate resolution does not exactly partition candidates for {stream_id}"
        )


def _verify_stage_contracts(
    protocol: OcidPipelineProtocol,
    expected_role: str,
    inventory: Sequence[OcidStream],
    inference: Mapping[str, Any],
    detector: Mapping[str, Any],
    sam2: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    expected_stream_ids = {item.stream_id for item in inventory}
    stages = (
        (detector, "detector", "detector_inference_completed_before_ground_truth"),
        (sam2, "sam2", "rgb_inference_completed_before_ground_truth"),
        (analysis, "analysis", "annotation_free_analysis_completed_before_ground_truth"),
    )
    for payload, label, status in stages:
        if (
            payload.get("protocol_sha256") != protocol.sha256
            or payload.get("status") != status
            or payload.get("ground_truth_opened") is not False
        ):
            raise OcidPipelineError(f"{label} stage violates the persisted inference boundary")
    if (
        detector.get("role") != expected_role
        or sam2.get("scope") != expected_role
        or analysis.get("role") != expected_role
    ):
        raise OcidPipelineError("stage role differs from the inference role")
    detector_streams = _mapping(detector, "streams", "detector manifest")
    sam2_streams = _mapping(sam2, "records_by_stream", "SAM2 manifest")
    analysis_streams = _mapping(analysis, "streams", "analysis manifest")
    if any(
        set(streams) != expected_stream_ids
        for streams in (detector_streams, sam2_streams, analysis_streams)
    ):
        raise OcidPipelineError("stage stream inventory differs from the configured inference inventory")
    coverage = _mapping(inference, "coverage", "inference manifest")
    candidate_count = coverage.get("candidate_count")
    detector_candidate_count = coverage.get("detector_candidate_count")
    removed_aggregate_count = coverage.get("removed_aggregate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
        or isinstance(detector_candidate_count, bool)
        or not isinstance(detector_candidate_count, int)
        or detector_candidate_count < candidate_count
        or isinstance(removed_aggregate_count, bool)
        or not isinstance(removed_aggregate_count, int)
        or removed_aggregate_count != detector_candidate_count - candidate_count
        or detector.get("candidate_count") != detector_candidate_count
        or sam2.get("candidate_count") != detector_candidate_count
        or sam2.get("valid_mask_count") != detector_candidate_count
        or sam2.get("resolved_candidate_count") != candidate_count
        or sam2.get("removed_aggregate_count") != removed_aggregate_count
        or sam2.get("fallback_count") != 0
        or analysis.get("candidate_count") != candidate_count
        or analysis.get("input_candidate_count") != detector_candidate_count
        or analysis.get("embedding_count") != candidate_count
        or analysis.get("embedding_failure_count") != 0
        or coverage.get("valid_mask_count") != detector_candidate_count
        or coverage.get("resolved_mask_count") != candidate_count
        or coverage.get("embedding_count") != candidate_count
        or coverage.get("mask_fallback_count") != 0
        or coverage.get("embedding_failure_count") != 0
    ):
        raise OcidPipelineError("stage coverage differs from the strict OCID pipeline contract")


def _implementation_receipt(protocol: OcidPipelineProtocol) -> dict[str, Any]:
    """Record executable Python sources for later integrity validation."""

    package_root = protocol.repository_root / "src" / "stream_analysis"
    tool_names = (
        "run_ocid_pipeline.py",
        "ocid_pipeline_contract.py",
        "evaluate_ocid_grounding_dino_extractor.py",
        "evaluate_ocid_grounding_dino_hardening.py",
        "evaluate_ocid_grounded_sam2_masks.py",
        "ocid_grounded_sam2_refinement.py",
        "evaluate_ocid_masked_dinov2.py",
        "evaluate_ocid_oracle_dinov2.py",
    )
    paths = list(package_root.rglob("*.py")) + [
        protocol.repository_root / "tools" / name for name in tool_names
    ]
    state = git_state(protocol.repository_root)
    files = []
    for path in sorted(set(paths), key=lambda row: row.relative_to(protocol.repository_root).as_posix()):
        full = path.resolve(strict=True)
        files.append(
            {
                "path": full.relative_to(protocol.repository_root).as_posix(),
                "size_bytes": full.stat().st_size,
                "sha256": sha256_file(full),
            }
        )
    return {
        "schema_version": IMPLEMENTATION_RECEIPT_SCHEMA,
        "git_commit": state.commit,
        "git_branch": state.branch,
        "git_worktree_clean": state.worktree_clean,
        "file_count": len(files),
        "files": files,
        "ground_truth_opened": False,
    }


def _verify_implementation_receipt(
    protocol: OcidPipelineProtocol,
    persisted: Mapping[str, Any],
) -> None:
    current = _implementation_receipt(protocol)
    if dict(persisted) != current:
        raise OcidPipelineError("executable implementation changed after inference")


def _verify_analysis_inputs_unchanged(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    run_id: str,
    expected_role: str,
    persisted: Mapping[str, Any],
) -> None:
    if (
        persisted.get("run_id") != run_id
        or persisted.get("role") != expected_role
        or persisted.get("protocol_sha256") != protocol.sha256
        or persisted.get("ground_truth_opened") is not False
    ):
        raise OcidPipelineError("analysis input receipt differs from the inference contract")
    current = validate_analysis_inputs(protocol, inventory)
    comparable_keys = (
        "artifact_manifest_sha256",
        "verified_artifact_count",
        "verified_rgb_count",
        "verified_manifest_count",
        "artifacts",
        "ground_truth_opened",
    )
    if any(persisted.get(key) != current.get(key) for key in comparable_keys):
        raise OcidPipelineError("analysis inputs changed after inference")


def _verify_analysis_embeddings(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    run_directory: Path,
    snapshots: Mapping[str, CandidateExtractionSnapshot],
    analysis_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    model_spec = _mapping(
        _mapping(protocol.payload, "models", "protocol"),
        "representation",
        "models",
    )
    embedding_dimension = int(model_spec["embedding_dimension"])
    streams = _mapping(analysis_manifest, "streams", "analysis manifest")
    result: dict[str, np.ndarray] = {}
    for item in inventory:
        stream_reference = _mapping(streams, item.stream_id, "analysis streams")
        analysis_reference = _mapping(stream_reference, "analysis", "analysis stream")
        analysis_path = _run_artifact_path(
            run_directory,
            _text(analysis_reference, "path", "analysis artifact"),
        )
        if analysis_reference.get("sha256") != sha256_file(analysis_path):
            raise OcidPipelineError(f"analysis artifact digest mismatch: {item.stream_id}")
        stream_analysis = _json_object(analysis_path, "persisted stream analysis")
        candidates = snapshots[item.stream_id].result.candidates
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if (
            stream_analysis.get("schema_version") != "ocid-pipeline-stream-analysis-1.0"
            or stream_analysis.get("stream_id") != item.stream_id
            or stream_analysis.get("candidate_ids") != candidate_ids
            or stream_reference.get("candidate_count") != len(candidate_ids)
            or stream_reference.get("embedding_count") != len(candidate_ids)
        ):
            raise OcidPipelineError(f"analysis candidate index differs: {item.stream_id}")
        embedding_reference = _mapping(
            stream_analysis,
            "embedding_artifact",
            "stream analysis",
        )
        outer_embedding_reference = _mapping(
            stream_reference,
            "embedding_artifact",
            "analysis stream",
        )
        if dict(embedding_reference) != dict(outer_embedding_reference):
            raise OcidPipelineError(f"embedding references differ: {item.stream_id}")
        embedding_path = _run_artifact_path(
            run_directory,
            _text(embedding_reference, "path", "embedding artifact"),
        )
        if embedding_reference.get("sha256") != sha256_file(embedding_path):
            raise OcidPipelineError(f"embedding artifact digest mismatch: {item.stream_id}")
        with np.load(embedding_path, allow_pickle=False) as archive:
            if set(archive.files) != {"embeddings"}:
                raise OcidPipelineError("OCID pipeline embedding NPZ must contain only embeddings")
            matrix = np.asarray(archive["embeddings"], dtype=np.float32).copy()
        expected_shape = (len(candidate_ids), embedding_dimension)
        if (
            matrix.shape != expected_shape
            or stream_analysis.get("embedding_shape") != list(expected_shape)
            or not np.isfinite(matrix).all()
        ):
            raise OcidPipelineError(f"invalid persisted embedding matrix for {item.stream_id}")
        if len(matrix):
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
                raise OcidPipelineError(f"embeddings are not normalized for {item.stream_id}")
        result[item.stream_id] = matrix
    return result


def _evaluate_candidate_metrics(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    snapshots: Mapping[str, CandidateExtractionSnapshot],
) -> dict[str, Any]:
    iou_grid = _evaluation_iou_grid(protocol)
    per_stream: dict[str, Any] = {}
    for item in inventory:
        annotation = load_annotation(item.annotation_path, manifest_path=item.stream_directory / "manifest.json")
        predicted = _predicted_candidates(snapshots[item.stream_id])
        per_stream[item.stream_id] = {
            "scene_group_id": item.scene_group_id,
            "frame_count": item.frame_count,
            "metrics_by_iou": {
                f"{threshold:.2f}": evaluate_candidate_predictions(
                    annotation,
                    predicted,
                    iou_threshold=threshold,
                )
                for threshold in iou_grid
            },
        }
    summaries = _candidate_summaries(inventory, per_stream, iou_grid)
    return {"iou_grid": list(iou_grid), "per_stream": per_stream, "summary_by_iou": summaries}


def _evaluate_mask_diagnostics(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    run_directory: Path,
    snapshots: Mapping[str, CandidateExtractionSnapshot],
    sam2_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selected = {
        item.stream_id: snapshots[item.stream_id].result.candidates
        for item in inventory
    }
    records = sam2_manifest.get("records_by_stream")
    if not isinstance(records, Mapping):
        raise OcidPipelineError("SAM2 manifest lacks records for mask evaluation")
    final_records: dict[str, list[Mapping[str, Any]]] = {}
    for item in inventory:
        rows = records.get(item.stream_id)
        if not isinstance(rows, list):
            raise OcidPipelineError(f"SAM2 records missing for {item.stream_id}")
        final_ids = {candidate.candidate_id for candidate in selected[item.stream_id]}
        filtered = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("candidate_id") in final_ids
        ]
        if len(filtered) != len(final_ids):
            raise OcidPipelineError(f"final SAM2 membership differs for {item.stream_id}")
        final_records[item.stream_id] = filtered
    evaluation_manifest = dict(sam2_manifest)
    final_count = sum(len(rows) for rows in final_records.values())
    evaluation_manifest["records_by_stream"] = final_records
    evaluation_manifest["candidate_count"] = final_count
    evaluation_manifest["valid_mask_count"] = final_count
    ocid_root = protocol.repository_root / "data" / "ocid" / "raw" / "OCID-dataset"
    summary, rows, _reviewed_masks = evaluate_masks(
        inventory=inventory,  # type: ignore[arg-type]
        selected=selected,  # type: ignore[arg-type]
        inference_manifest=evaluation_manifest,
        reviewed_root=protocol.reviewed_root,
        ocid_root=ocid_root,
        artifact_root=run_directory,
    )
    return {
        "conditional_bbox_matched": {
            "summary": summary,
            "matched_observation_count": len(rows),
            "rows": rows,
            "conditional_on_bbox_match": True,
        },
        "end_to_end_assignment": _evaluate_end_to_end_masks(
            protocol=protocol,
            inventory=inventory,
            run_directory=run_directory,
            snapshots=snapshots,
            records_by_stream=final_records,
            ocid_root=ocid_root,
        ),
    }


def _evaluate_end_to_end_masks(
    *,
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    run_directory: Path,
    snapshots: Mapping[str, CandidateExtractionSnapshot],
    records_by_stream: Mapping[str, Sequence[Mapping[str, Any]]],
    ocid_root: Path,
) -> dict[str, Any]:
    """Score every final prediction and every reference by full mask IoU."""

    benchmark_path = protocol.reviewed_root / "benchmark_manifest.json"
    benchmark = _json_object(benchmark_path, "reviewed benchmark manifest")
    if _mapping(benchmark, "heldout_lock", "reviewed benchmark").get(
        "predictions_unlocked"
    ) is not False:
        raise OcidPipelineError("held-out benchmark lock must remain closed")
    stream_metadata = {
        str(row["stream_id"]): row
        for row in benchmark.get("streams", [])
        if isinstance(row, Mapping) and isinstance(row.get("stream_id"), str)
    }
    component_review = _mapping(benchmark, "component_review", "reviewed benchmark")
    decisions_path = protocol.reviewed_root / _text(
        component_review,
        "decisions_artifact",
        "component review",
    )
    if sha256_file(decisions_path) != component_review.get("decisions_sha256"):
        raise OcidPipelineError("component-review decision digest mismatch")
    decisions_payload = _json_object(decisions_path, "component review decisions")
    decisions = {
        str(row["case_id"]): row
        for row in decisions_payload.get("decisions", [])
        if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
    }
    thresholds = (0.50, 0.75)
    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": [], "frames": []}
        for threshold in thresholds
    }
    per_stream: dict[float, dict[str, Any]] = {
        threshold: {} for threshold in thresholds
    }
    expected_frame_count = sum(item.frame_count for item in inventory)
    for item in inventory:
        annotation = load_annotation(
            item.annotation_path,
            manifest_path=item.stream_directory / "manifest.json",
        )
        raw_instances = {
            str(row["instance_id"]): row
            for row in annotation.raw["expected_element_instances"]
        }
        manifest = _json_object(item.stream_directory / "manifest.json", "analysis manifest")
        source_filenames = {
            str(row["frame_id"]): str(row["metadata"]["ocid_source_filename"])
            for row in manifest["frames"]
        }
        metadata = stream_metadata.get(item.stream_id)
        if not isinstance(metadata, Mapping):
            raise OcidPipelineError(f"reviewed stream metadata missing: {item.stream_id}")
        source_sequence = _text(metadata, "source_sequence", "reviewed stream")
        candidates_by_frame: dict[str, list[CandidateRecord]] = defaultdict(list)
        for candidate in snapshots[item.stream_id].result.candidates:
            candidates_by_frame[candidate.frame_id].append(candidate)
        mask_rows = {
            str(row["candidate_id"]): row for row in records_by_stream[item.stream_id]
        }
        stream_counts = {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "frame_count": 0}
            for threshold in thresholds
        }
        for frame_id in annotation.frame_ids:
            predictions = tuple(
                sorted(
                    candidates_by_frame.get(frame_id, ()),
                    key=lambda row: row.candidate_id.encode("utf-8"),
                )
            )
            references = tuple(
                sorted(
                    annotation.instances_by_frame.get(frame_id, ()),
                    key=lambda row: row.instance_id.encode("utf-8"),
                )
            )
            predicted_masks = tuple(
                _load_binary_mask(
                    _run_artifact_path(
                        run_directory,
                        _text(
                            _mapping(mask_rows[candidate.candidate_id], "cleaned_mask", "mask row"),
                            "path",
                            "cleaned mask",
                        ),
                    )
                )
                for candidate in predictions
            )
            expected_masks: list[np.ndarray] = []
            for reference in references:
                expected, _provenance = _reviewed_mask(
                    stream_id=item.stream_id,
                    frame_id=frame_id,
                    instance_payload=raw_instances[reference.instance_id],
                    source_sequence=source_sequence,
                    source_filename=source_filenames[frame_id],
                    ocid_root=ocid_root,
                    decisions=decisions,
                )
                expected_bbox = mask_bbox_from_full_frame(expected)
                if expected_bbox is None or not _same_bbox(expected_bbox, reference.bbox):
                    raise OcidPipelineError(
                        f"reviewed mask bbox differs from annotation: {reference.instance_id}"
                    )
                expected_masks.append(expected)
            matrix = _mask_iou_matrix(predicted_masks, tuple(expected_masks))
            for threshold in thresholds:
                assignments = _maximum_cardinality_mask_assignment(matrix, threshold)
                matched_predictions = {row[0] for row in assignments}
                matched_references = {row[1] for row in assignments}
                tp = len(assignments)
                fp = len(predictions) - tp
                fn = len(references) - tp
                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn
                totals[threshold]["matched_ious"].extend(row[2] for row in assignments)
                stream_counts[threshold]["tp"] += tp
                stream_counts[threshold]["fp"] += fp
                stream_counts[threshold]["fn"] += fn
                stream_counts[threshold]["frame_count"] += 1
                totals[threshold]["frames"].append(
                    {
                        "stream_id": item.stream_id,
                        "scene_group_id": item.scene_group_id,
                        "frame_id": frame_id,
                        "prediction_count": len(predictions),
                        "reference_count": len(references),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "assignments": [
                            {
                                "candidate_id": predictions[left].candidate_id,
                                "instance_id": references[right].instance_id,
                                "mask_iou": iou,
                            }
                            for left, right, iou in assignments
                        ],
                        "false_positive_candidate_ids": [
                            row.candidate_id
                            for index, row in enumerate(predictions)
                            if index not in matched_predictions
                        ],
                        "missed_instance_ids": [
                            row.instance_id
                            for index, row in enumerate(references)
                            if index not in matched_references
                        ],
                    }
                )
        for threshold in thresholds:
            per_stream[threshold][item.stream_id] = _detection_count_metrics(
                stream_counts[threshold]
            ) | {"frame_count": stream_counts[threshold]["frame_count"]}
    output: dict[str, Any] = {}
    for threshold in thresholds:
        if len(totals[threshold]["frames"]) != expected_frame_count:
            raise OcidPipelineError("end-to-end mask evaluation missed inventory frames")
        output[f"{threshold:.2f}"] = {
            "mask_iou_at_least": threshold,
            "assignment_policy": "maximum_cardinality_then_maximum_summed_mask_iou",
            "frame_count": expected_frame_count,
            "micro": _detection_count_metrics(totals[threshold])
            | {
                "matched_mask_iou": _numeric_distribution(
                    totals[threshold]["matched_ious"]
                )
            },
            "per_stream": per_stream[threshold],
            "per_frame": totals[threshold]["frames"],
        }
    return {
        "status": "computed",
        "reference_semantics": "ocid_derived_component_reviewed",
        "prediction_universe": "all_final_candidates_including_false_positives",
        "reference_universe": "all_reference_instances_in_evaluated_frames",
        "metrics_by_mask_iou": output,
    }


def _mask_iou_matrix(
    predicted_masks: Sequence[np.ndarray],
    expected_masks: Sequence[np.ndarray],
) -> np.ndarray:
    matrix = np.zeros((len(predicted_masks), len(expected_masks)), dtype=np.float64)
    if not predicted_masks or not expected_masks:
        return matrix
    shapes = {np.asarray(mask).shape for mask in (*predicted_masks, *expected_masks)}
    if len(shapes) != 1:
        raise OcidPipelineError("prediction and reference mask shapes differ")
    left = np.stack(predicted_masks).reshape(len(predicted_masks), -1)
    right = np.stack(expected_masks).reshape(len(expected_masks), -1)
    intersections = left.astype(np.float32) @ right.astype(np.float32).T
    left_areas = np.count_nonzero(left, axis=1)[:, None]
    right_areas = np.count_nonzero(right, axis=1)[None, :]
    unions = left_areas + right_areas - intersections
    np.divide(intersections, unions, out=matrix, where=unions > 0)
    return matrix


def _maximum_cardinality_mask_assignment(
    matrix: np.ndarray,
    threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or (values.size and not np.all(np.isfinite(values))):
        raise OcidPipelineError("mask IoU matrix is malformed")
    prediction_count, reference_count = values.shape
    if prediction_count == 0 or reference_count == 0:
        return ()
    valid = values >= threshold
    bonus = float(min(prediction_count, reference_count) + 1)
    rewards = np.zeros(
        (prediction_count + reference_count, prediction_count + reference_count),
        dtype=np.float64,
    )
    rewards[:prediction_count, :reference_count] = np.where(
        valid,
        bonus + values,
        0.0,
    )
    rows, columns = linear_sum_assignment(-rewards)
    return tuple(
        sorted(
            (
                (row, column, float(values[row, column]))
                for row, column in zip(rows.tolist(), columns.tolist())
                if row < prediction_count
                and column < reference_count
                and valid[row, column]
            ),
            key=lambda row: (row[0], row[1]),
        )
    )


def _detection_count_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
    tp = int(values["tp"])
    fp = int(values["fp"])
    fn = int(values["fn"])
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "f1": None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
    }


def _numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(array),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _evaluate_physical_continuity(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
    snapshots: Mapping[str, CandidateExtractionSnapshot],
    embedding_matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    model_spec = _mapping(_mapping(protocol.payload, "models", "protocol"), "representation", "models")
    pipeline = _mapping(protocol.payload, "pipeline", "protocol")
    representation = _mapping(pipeline, "representation", "pipeline")
    evaluation = _mapping(protocol.payload, "evaluation", "protocol")
    assignment_iou = float(evaluation["physical_continuity_assignment_iou"])
    scoring_config, matching_config, _grouping, _events = _matching_configs(protocol)
    matcher_policy = MatcherPolicy(
        str(matching_config.assignment_policy_id),
        float(scoring_config.visual_gate),
        float(matching_config.unmatched_pair_cost),
        float(matching_config.local_margin_gate),
        float(matching_config.global_margin_gate),
    )
    representation_variant = str(representation["variant"])
    model_name = str(model_spec["model_name"])
    per_stream: dict[str, Any] = {}
    matcher_reports: dict[str, dict[str, Any]] = {}
    for item in inventory:
        snapshot = snapshots[item.stream_id]
        candidates = snapshot.result.candidates
        matrix = embedding_matrices[item.stream_id]
        embeddings = {
            candidate.candidate_id: matrix[index]
            for index, candidate in enumerate(candidates)
        }
        annotation = load_annotation(item.annotation_path, manifest_path=item.stream_directory / "manifest.json")
        candidate_inputs = tuple(
            CandidateInput(item.stream_id, candidate, snapshot.mask_for_candidate(candidate.candidate_id))
            for candidate in candidates
        )
        identity, _bbox_iou, detector_evaluation = _assignment_identity(
            annotation,
            candidate_inputs,
            iou_threshold=assignment_iou,
        )
        oracle = tuple(
            OracleInstance(
                item.stream_id,
                candidate.frame_id,
                candidate.frame_index,
                candidate.candidate_id,
                identity[candidate.candidate_id],
                candidate.bbox,
            )
            for candidate in candidates
        )
        prepared = tuple(
            PreparedOracleInstance(
                oracle_row,
                None,
                candidate_input.candidate,
                candidate_input.mask_record,
            )
            for oracle_row, candidate_input in zip(oracle, candidate_inputs, strict=True)
        )
        decoded = _load_stream(item, protocol.sha256)
        ranking, _pairs = evaluate_oracle_embeddings(oracle, embeddings, namespace=item.stream_id)
        matcher = evaluate_oracle_matcher(
            decoded,
            prepared,
            embeddings,
            variant=representation_variant,
            model_name=model_name,
            policy=matcher_policy,
        )
        matcher_reports[item.stream_id] = matcher
        per_stream[item.stream_id] = {
            "scene_group_id": item.scene_group_id,
            "detector_assignment": _candidate_evaluation_payload(detector_evaluation),
            "ranking": ranking,
            "matcher": matcher,
        }
    pooled = aggregate_oracle_matcher_reports(tuple(matcher_reports.values()))
    scene_groups: dict[str, Any] = {}
    for scene_group_id in sorted({item.scene_group_id for item in inventory}):
        stream_ids = sorted(item.stream_id for item in inventory if item.scene_group_id == scene_group_id)
        scene_groups[scene_group_id] = {
            "stream_ids": stream_ids,
            "matcher": aggregate_oracle_matcher_reports(
                tuple(matcher_reports[stream_id] for stream_id in stream_ids)
            ),
        }
    accepted_f1 = [
        row["matcher"]["accepted"]["f1"]
        for row in scene_groups.values()
        if row["matcher"]["accepted"]["f1"] is not None
    ]
    macro_f1 = None if not accepted_f1 else sum(float(value) for value in accepted_f1) / len(accepted_f1)
    worst_id = min(
        scene_groups,
        key=lambda key: (
            -1.0
            if scene_groups[key]["matcher"]["accepted"]["f1"] is None
            else float(scene_groups[key]["matcher"]["accepted"]["f1"]),
            key,
        ),
    )
    return {
        "identity_proxy": "sequence_local_physical_instance_id_evaluation_only",
        "bbox_assignment_iou": assignment_iou,
        "per_stream": per_stream,
        "pooled_matcher": pooled,
        "scene_groups": scene_groups,
        "scene_group_macro_accepted_f1": macro_f1,
        "worst_scene_group": {
            "scene_group_id": worst_id,
            "accepted_f1": scene_groups[worst_id]["matcher"]["accepted"]["f1"],
        },
        "visual_type_accuracy_claim": False,
    }


def _candidate_summaries(
    inventory: Sequence[OcidStream],
    per_stream: Mapping[str, Any],
    iou_grid: Sequence[float],
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in inventory:
        groups[item.scene_group_id].append(item.stream_id)
    result: dict[str, Any] = {}
    for threshold in iou_grid:
        key = f"{threshold:.2f}"
        all_metrics = [per_stream[item.stream_id]["metrics_by_iou"][key] for item in inventory]
        scene_groups = {
            scene_id: {
                "stream_ids": sorted(stream_ids),
                "metrics": aggregate_candidate_metrics(
                    tuple(per_stream[stream_id]["metrics_by_iou"][key] for stream_id in sorted(stream_ids))
                ),
            }
            for scene_id, stream_ids in sorted(groups.items())
        }
        macro = {
            metric: _mean_defined(
                [_macro_metric(row["metrics"], metric) for row in scene_groups.values()]
            )
            for metric in ("precision", "recall", "f1", "false_per_frame", "weighted_mean_tp_iou")
        }
        worst_id = min(
            scene_groups,
            key=lambda scene_id: (
                -1.0
                if scene_groups[scene_id]["metrics"].get("f1") is None
                else scene_groups[scene_id]["metrics"]["f1"],
                scene_id,
            ),
        )
        result[key] = {
            "pooled": aggregate_candidate_metrics(tuple(all_metrics)),
            "scene_groups": scene_groups,
            "scene_group_macro": macro,
            "worst_scene_group": {
                "scene_group_id": worst_id,
                "metrics": scene_groups[worst_id]["metrics"],
            },
        }
    return result


def _evaluation_iou_grid(protocol: OcidPipelineProtocol) -> tuple[float, ...]:
    evaluation = _mapping(protocol.payload, "evaluation", "protocol")
    values = evaluation.get("candidate_bbox_iou_grid")
    if not isinstance(values, list) or not values:
        raise OcidPipelineError("protocol evaluation IoU grid must be a non-empty list")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise OcidPipelineError("protocol evaluation IoU grid must be numeric") from error
    if any(not 0.0 < value <= 1.0 for value in result) or tuple(sorted(set(result))) != result:
        raise OcidPipelineError("protocol evaluation IoU grid must be unique and increasing in (0, 1]")
    return result


def _candidate_evaluation_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    assignments = value.get("assignments", ())
    if not isinstance(assignments, (list, tuple)):
        raise OcidPipelineError("candidate evaluation assignments must be a sequence")
    result["assignments"] = [
        {
            "candidate_id": row.candidate_id,
            "instance_id": row.instance_id,
            "frame_id": row.frame_id,
            "iou": row.iou,
            "visual_type_id": row.visual_type_id,
        }
        for row in assignments
    ]
    return result


def _matching_configs(
    protocol: OcidPipelineProtocol,
) -> tuple[PairScoringConfig, MatchingConfig, GroupingConfig, EventConfig]:
    profile = runtime_profile_from_sections(
        profile_id="visual_stream_analyzer_v1",
        models=_mapping(protocol.payload, "models", "protocol"),
        pipeline=_mapping(protocol.payload, "pipeline", "protocol"),
        paths_include_models_prefix=True,
    )
    return runtime_matching_configs(profile.pipeline)


def _detector_profiles(
    protocol: OcidPipelineProtocol,
) -> tuple[hardening.PromptProfile, hardening.GeometryProfile]:
    profile = runtime_profile_from_sections(
        profile_id="visual_stream_analyzer_v1",
        models=_mapping(protocol.payload, "models", "protocol"),
        pipeline=_mapping(protocol.payload, "pipeline", "protocol"),
        paths_include_models_prefix=True,
    )
    prompt, geometry = runtime_detector_profiles(profile)
    return (
        hardening.PromptProfile(
            prompt.prompt_id,
            prompt.text,
            prompt.selectable,
            "ocid_evaluation_raw_reuse",
        ),
        geometry,
    )


def _load_stream(item: OcidStream, protocol_sha256: str):
    producer = _producer("stream_input", protocol_sha256)
    decoded = load_decoded_stream(
        ManifestLoadRequest(stream_root=item.stream_directory.resolve(strict=True), producer=producer)
    )
    if decoded.stream.stream_id != item.stream_id or len(decoded.frames) != item.frame_count:
        raise OcidPipelineError(f"decoded stream differs from configured inventory: {item.stream_id}")
    return decoded


def _producer(stage: str, protocol_sha256: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0.0",
        config_version="1.0.0",
        config_digest=f"sha256:{protocol_sha256}",
    )


def _render_candidate_overlays(
    decoded: Any,
    snapshot: CandidateExtractionSnapshot,
    mask_rows: Sequence[Mapping[str, Any]],
    run_directory: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Render one audit-friendly overlay for every frame, including empty ones."""

    destination_root = (
        run_directory / "candidate_overlays"
        if output_root is None
        else Path(output_root)
    )
    reference_root = run_directory if output_root is None else destination_root
    mask_by_id = {str(row["candidate_id"]): row for row in mask_rows}
    candidates_by_frame: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in snapshot.result.candidates:
        candidates_by_frame[candidate.frame_id].append(candidate)
    frames: list[dict[str, Any]] = []
    font = ImageFont.load_default()
    for frame in decoded.frames:
        canvas = _frame_array(frame).astype(np.float32)
        candidates = sorted(
            candidates_by_frame.get(frame.frame_id, ()),
            key=lambda row: row.candidate_id.encode("utf-8"),
        )
        labels: list[tuple[CandidateRecord, str, tuple[int, int, int]]] = []
        prediction_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            row = mask_by_id.get(candidate.candidate_id)
            if row is None:
                raise OcidPipelineError(
                    f"overlay candidate lacks mask: {candidate.candidate_id}"
                )
            cleaned = _mapping(row, "cleaned_mask", "mask row")
            mask = _load_binary_mask(
                _run_artifact_path(run_directory, _text(cleaned, "path", "cleaned mask"))
            )
            color = OVERLAY_PALETTE[index % len(OVERLAY_PALETTE)]
            color_array = np.asarray(color, dtype=np.float32)
            canvas[mask] = canvas[mask] * 0.55 + color_array * 0.45
            canvas[_mask_boundary(mask)] = color_array
            label = _candidate_overlay_label(index)
            labels.append((candidate, label, color))
            prediction_rows.append(
                {
                    "label": f"P{index:02d}",
                    "candidate_id": candidate.candidate_id,
                    "score": candidate.candidate_confidence,
                    "color_rgb": list(color),
                    "bbox": _bbox_payload(candidate.bbox),
                }
            )
        image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        for candidate, label, color in labels:
            bbox = candidate.bbox
            draw.rectangle(
                (bbox.left, bbox.top, bbox.right - 1, bbox.bottom - 1),
                outline=color,
                width=2,
            )
            draw.text((bbox.left + 2, max(0, bbox.top - 12)), label, fill=color, font=font)
        path = (
            destination_root
            / decoded.stream.stream_id
            / f"{frame.frame_id}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", optimize=True)
        frames.append(
            {
                "frame_id": frame.frame_id,
                "prediction_count": len(candidates),
                "predictions": prediction_rows,
                "artifact": _file_reference(reference_root, path),
            }
        )
    manifest_path = (
        destination_root / decoded.stream.stream_id / "manifest.json"
    )
    manifest = {
        "schema_version": "ocid-pipeline-candidate-overlays-1.0",
        "stream_id": decoded.stream.stream_id,
        "rendering": "cleaned_mask_alpha_contour_bbox_and_frame_local_ordinal_label",
        "confidence_displayed": False,
        "palette_size": len(OVERLAY_PALETTE),
        "ground_truth_opened": False,
        "frames": frames,
    }
    write_json_exclusive(manifest_path, manifest)
    return _file_reference(reference_root, manifest_path)


def _candidate_overlay_label(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("candidate overlay index must be a nonnegative integer")
    return f"P{index:02d}"


def _render_mask_type_overlays(
    decoded: Any,
    snapshot: CandidateExtractionSnapshot,
    grouping: Any,
    mask_rows: Sequence[Mapping[str, Any]],
    run_directory: Path,
) -> dict[str, Any]:
    assignment = {row.candidate_id: row.primary_type_id for row in grouping.assignments}
    mask_by_id = {str(row["candidate_id"]): row for row in mask_rows}
    candidates_by_frame: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in snapshot.result.candidates:
        candidates_by_frame[candidate.frame_id].append(candidate)
    paths: list[dict[str, Any]] = []
    font = ImageFont.load_default()
    for frame in decoded.frames:
        canvas = _frame_array(frame).astype(np.float32)
        labels: list[tuple[BBox, str, tuple[int, int, int]]] = []
        for candidate in candidates_by_frame.get(frame.frame_id, ()):
            row = mask_by_id[candidate.candidate_id]
            cleaned = _mapping(row, "cleaned_mask", "mask row")
            mask = _load_binary_mask(_run_artifact_path(run_directory, str(cleaned["path"])))
            type_id = assignment.get(candidate.candidate_id, "unassigned")
            color = _type_color(type_id)
            color_array = np.asarray(color, dtype=np.float32)
            canvas[mask] = canvas[mask] * 0.58 + color_array * 0.42
            canvas[_mask_boundary(mask)] = color_array
            labels.append((candidate.bbox, type_id, color))
        image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        for bbox, type_id, color in labels:
            draw.text((bbox.left + 2, max(0, bbox.top - 12)), type_id, fill=color, font=font)
        path = run_directory / "overlays" / decoded.stream.stream_id / f"{frame.frame_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", optimize=True)
        paths.append(_file_reference(run_directory, path))
    manifest_path = run_directory / "overlays" / decoded.stream.stream_id / "manifest.json"
    manifest = {
        "schema_version": "ocid-pipeline-mask-type-overlays-1.0",
        "stream_id": decoded.stream.stream_id,
        "rendering": "cleaned_mask_alpha_contour_and_predicted_group_label",
        "ground_truth_opened": False,
        "frames": paths,
    }
    write_json_exclusive(manifest_path, manifest)
    return _file_reference(run_directory, manifest_path)


def _artifact_inventory(run_directory: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_directory.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_directory).as_posix(),
    ):
        if path.name in _MUTABLE_FILENAMES:
            continue
        artifacts.append(_file_reference(run_directory, path))
    return artifacts


def _verify_artifact_inventory(run_directory: Path, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise OcidPipelineError("inference artifact inventory must be a non-empty list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise OcidPipelineError("malformed inference artifact entry")
    expected_path_rows = [row.get("path") for row in rows]
    if not all(isinstance(path, str) and path for path in expected_path_rows):
        raise OcidPipelineError("malformed inference artifact path")
    current_rows = _artifact_inventory(run_directory)
    expected_paths = set(expected_path_rows)
    current_paths = {row["path"] for row in current_rows}
    if (
        len(expected_paths) != len(rows)
        or expected_paths != current_paths
    ):
        raise OcidPipelineError(
            "inference artifact inventory differs from the exact immutable file set"
        )
    current_by_path = {str(row["path"]): row for row in current_rows}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise OcidPipelineError("malformed inference artifact entry")
        relative = _text(row, "path", "inference artifact")
        if relative in seen:
            raise OcidPipelineError("duplicate inference artifact path")
        seen.add(relative)
        current = current_by_path[relative]
        if (
            current.get("size_bytes") != row.get("size_bytes")
            or current.get("sha256") != row.get("sha256")
        ):
            raise OcidPipelineError(f"inference artifact changed: {relative}")


def _verified_stage_manifest(
    run_directory: Path,
    inference: Mapping[str, Any],
    key: str,
    schema: str,
) -> dict[str, Any]:
    stages = _mapping(inference, "stage_manifests", "inference manifest")
    reference = _mapping(stages, key, "stage manifests")
    path = _run_artifact_path(run_directory, _text(reference, "path", f"{key} manifest"))
    if reference.get("sha256") != sha256_file(path):
        raise OcidPipelineError(f"{key} stage manifest digest mismatch")
    payload = _json_object(path, f"{key} stage manifest")
    if payload.get("schema_version") != schema:
        raise OcidPipelineError(f"{key} stage manifest schema mismatch")
    return payload


def _load_referenced_json(
    run_directory: Path,
    manifest: Mapping[str, Any],
    stream_id: str,
) -> dict[str, Any]:
    streams = _mapping(manifest, "streams", "stage manifest")
    reference = _mapping(streams, stream_id, "stage streams")
    path = _run_artifact_path(run_directory, _text(reference, "path", "stream artifact"))
    if reference.get("sha256") != sha256_file(path):
        raise OcidPipelineError(f"stream artifact digest mismatch: {stream_id}")
    return _json_object(path, f"stream artifact {stream_id}")


def _file_reference(root: Path, path: Path) -> dict[str, Any]:
    full = path.resolve(strict=True)
    try:
        relative = full.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise OcidPipelineError("artifact path escapes the OCID pipeline run directory") from error
    return {"path": relative, "size_bytes": full.stat().st_size, "sha256": sha256_file(full)}


def _run_artifact_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise OcidPipelineError("run artifact must be a safe relative path")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise OcidPipelineError("run artifact escapes the run directory") from error
    if not path.is_file():
        raise OcidPipelineError(f"run artifact is not a file: {path}")
    return path


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    value = np.asarray(mask, dtype=np.bool_)
    if value.ndim != 2 or not value.any():
        raise OcidPipelineError("refusing to persist an empty or malformed mask")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path, format="PNG", optimize=True)


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source).copy()
    if value.ndim != 2 or not np.isin(value, (0, 255)).all():
        raise OcidPipelineError(f"persisted mask is not binary: {path}")
    return np.ascontiguousarray(value > 0, dtype=np.bool_)


def _save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    value = np.asarray(embeddings, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise OcidPipelineError("embedding matrix must be finite and two-dimensional")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, embeddings=value)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _predicted_candidates(snapshot: CandidateExtractionSnapshot) -> tuple[PredictedCandidate, ...]:
    return tuple(
        PredictedCandidate(
            candidate_id=item.candidate_id,
            frame_id=item.frame_id,
            frame_index=item.frame_index,
            bbox=item.bbox,
            validity_status="valid",
            warning_ids=(),
            error_ids=(),
        )
        for item in snapshot.result.candidates
    )


def _bbox(payload: Mapping[str, Any]) -> BBox:
    try:
        return BBox(
            float(payload["x"]),
            float(payload["y"]),
            float(payload["width"]),
            float(payload["height"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OcidPipelineError(f"malformed bbox: {payload!r}") from error


def _bbox_payload(value: BBox | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value.x, "y": value.y, "width": value.width, "height": value.height}


def _integral_bbox(value: BBox) -> tuple[int, int, int, int]:
    numbers = (value.x, value.y, value.width, value.height)
    integers = tuple(int(number) for number in numbers)
    if any(float(number) != float(integer) for number, integer in zip(numbers, integers, strict=True)):
        raise OcidPipelineError("OCID pipeline candidate bbox must have integral coordinates")
    return integers  # type: ignore[return-value]


def _frame_array(frame: Any) -> np.ndarray:
    return np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
        frame.image_size.height,
        frame.image_size.width,
        3,
    ).copy()


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    interior = value.copy()
    interior[1:, :] &= value[:-1, :]
    interior[:-1, :] &= value[1:, :]
    interior[:, 1:] &= value[:, :-1]
    interior[:, :-1] &= value[:, 1:]
    return value & ~interior


def _type_color(type_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(type_id.encode("utf-8")).digest()
    return tuple(64 + value % 160 for value in digest[:3])  # type: ignore[return-value]


def _macro_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    if name == "false_per_frame":
        frame_count = int(metrics.get("frame_count", 0))
        return None if frame_count <= 0 else float(metrics["fp"]) / frame_count
    value = metrics.get(name)
    return None if value is None else float(value)


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return None if not defined else sum(defined) / len(defined)


def _mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise OcidPipelineError(f"{label}.{key} must be an object")
    return result


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise OcidPipelineError(f"{label}.{key} must be a non-empty string")
    return result


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcidPipelineError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise OcidPipelineError(f"{label} must be a JSON object")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate configured metadata without stream access")
    check.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser("analyze-sample", help="run one configured RGB sample without ground truth")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--run-id", required=True)

    development = subparsers.add_parser(
        "analyze-development",
        help="run the current pipeline on the complete configured dataset split",
    )
    development.add_argument("--output-root", type=Path, required=True)
    development.add_argument("--run-id", required=True)

    inference = subparsers.add_parser("analyze-evaluation", help="run the exact evaluation-accessed held-out inference once")
    inference.add_argument("--output-root", type=Path, required=True)
    inference.add_argument("--run-id", required=True)
    inference.add_argument("--access", type=Path, required=True)

    smoke_evaluation = subparsers.add_parser(
        "evaluate-sample",
        help="evaluate a completed sample run",
    )
    smoke_evaluation.add_argument("--run-directory", type=Path, required=True)

    development_evaluation = subparsers.add_parser(
        "evaluate-development",
        help="evaluate a completed full development run",
    )
    development_evaluation.add_argument("--run-directory", type=Path, required=True)

    display = subparsers.add_parser(
        "render-development-overlays",
        help="render frame-local ordinal labels without confidence values",
    )
    display.add_argument("--run-directory", type=Path, required=True)
    display.add_argument("--output-root", type=Path, required=True)

    evaluation = subparsers.add_parser("evaluate-results", help="open GT only after validating persisted inference")
    evaluation.add_argument("--run-directory", type=Path, required=True)
    evaluation.add_argument("--access", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = run_check(protocol_path=args.protocol, output_path=args.output)
        elif args.command == "analyze-sample":
            result = run_inference(
                protocol_path=args.protocol,
                role="sample",
                output_root=args.output_root,
                run_id=args.run_id,
            )
        elif args.command == "analyze-development":
            result = run_inference(
                protocol_path=args.protocol,
                role="development",
                output_root=args.output_root,
                run_id=args.run_id,
            )
        elif args.command == "analyze-evaluation":
            result = run_inference(
                protocol_path=args.protocol,
                role="heldout",
                output_root=args.output_root,
                run_id=args.run_id,
                access_path=args.access,
            )
        elif args.command == "evaluate-sample":
            result = run_sample_evaluation(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
            )
        elif args.command == "evaluate-development":
            result = run_development_evaluation(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
            )
        elif args.command == "render-development-overlays":
            result = render_development_candidate_overlays(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
                output_root=args.output_root,
            )
        else:
            result = run_evaluation(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
                access_path=args.access,
            )
    except (OcidPipelineError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
