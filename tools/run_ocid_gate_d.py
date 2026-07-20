"""Freeze, execute, and evaluate the one-run OCID Gate D protocol.

The command has separate preflight, RGB-only inference, and ground-truth
evaluation phases.  There are no CLI parameters for model or threshold
selection; every semantic value comes from the reviewed protocol JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stream_analysis.candidates import (
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    candidate_mask_digest,
)
from stream_analysis.contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    FrameCandidateDiagnostics,
    FramePair,
    GeometryFeatureMetadata,
    ImageSize,
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
    build_change_events,
    group_recurring_visual_types,
    match_neighboring_frames,
)
from stream_analysis.representations import (
    DINO_MASK_NEUTRAL_VARIANT,
    DinoV2CosineScorer,
    DinoV2RepresentationConfig,
    LocalDinoV2Provider,
    build_dinov2_representations,
)
from stream_analysis.serialization import to_json_compatible

from tools import evaluate_ocid_grounding_dino_extractor as grounding
from tools import evaluate_ocid_grounding_dino_hardening as hardening
from tools.evaluate_ocid_gate_c2_masked_dinov2 import (
    CandidateInput,
    _assignment_identity,
)
from tools.evaluate_ocid_grounded_sam2_masks import evaluate_masks
from tools.evaluate_ocid_oracle_dinov2 import (
    MatcherPolicy,
    OracleInstance,
    PreparedOracleInstance,
    aggregate_oracle_matcher_reports,
    evaluate_oracle_embeddings,
    evaluate_oracle_matcher,
)
from tools.ocid_gate_d_contract import (
    DEFAULT_PROTOCOL,
    FrozenGateDProtocol,
    GateDContractError,
    GateDStream,
    claim_author_unlock,
    git_state,
    inventory_for_role,
    load_frozen_protocol,
    preflight_receipt,
    sha256_file,
    validate_analysis_inputs,
    validate_author_unlock,
    validate_author_unlock_consumption,
    validate_evaluation_inputs,
    validate_run_id,
    write_json_atomic,
    write_json_exclusive,
)
from tools.ocid_grounded_sam2_refinement import (
    MaskCleanupConfig,
    load_local_sam2_bbox_refiner,
)


INFERENCE_SCHEMA = "ocid-gate-d-inference-manifest-1.0"
DETECTOR_SCHEMA = "ocid-gate-d-detector-artifacts-1.0"
SAM2_SCHEMA = "ocid-gate-d-sam2-artifacts-1.0"
ANALYSIS_SCHEMA = "ocid-gate-d-analysis-artifacts-1.0"
EVALUATION_SCHEMA = "ocid-gate-d-evaluation-1.0"
ATTEMPT_SCHEMA = "ocid-gate-d-attempt-1.0"
INPUT_RECEIPT_SCHEMA = "ocid-gate-d-analysis-input-receipt-1.0"
_MUTABLE_FILENAMES = frozenset({"attempt.json", "inference_manifest.json"})


def run_preflight(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    receipt = preflight_receipt(protocol)
    write_json_exclusive(output_path, receipt)
    return receipt


def run_inference(
    *,
    protocol_path: Path,
    role: str,
    output_root: Path,
    run_id: str,
    unlock_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    if role not in {"development_smoke", "heldout"}:
        raise GateDContractError("inference role must be development_smoke or heldout")
    run_id = validate_run_id(run_id)
    inventory = inventory_for_role(protocol, role)  # type: ignore[arg-type]
    unlock: Mapping[str, Any] | None = None
    if role == "heldout":
        if unlock_path is None:
            raise GateDContractError("held-out inference requires an author unlock artifact")
        unlock = validate_author_unlock(unlock_path, protocol, expected_run_id=run_id)
    elif unlock_path is not None:
        raise GateDContractError("development smoke does not accept a held-out unlock")

    output_directory = Path(output_root).resolve(strict=False)
    run_directory = (output_directory / run_id).resolve(strict=False)
    try:
        run_directory.relative_to(output_directory)
    except ValueError as error:
        raise GateDContractError("Gate D run directory escapes the output root") from error
    if run_directory.exists():
        raise GateDContractError(f"Gate D run directory already exists: {run_directory}")
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
    if unlock_path is not None:
        attempt["author_unlock"] = {
            "sha256": sha256_file(Path(unlock_path).resolve(strict=True)),
            "decision": unlock["decision"] if unlock is not None else None,
        }
    write_json_exclusive(run_directory / "attempt.json", attempt)

    try:
        if role == "heldout":
            state_after_attempt = git_state(protocol.repository_root)
            if state_after_attempt != state or not state_after_attempt.worktree_clean:
                raise GateDContractError(
                    "held-out output and unlock artifacts must not dirty the frozen Git worktree"
                )
            assert unlock_path is not None
            claimed = claim_author_unlock(
                unlock_path,
                protocol,
                expected_run_id=run_id,
                run_directory=run_directory,
            )
            consumption_path = Path(claimed["consumption_path"])
            attempt["author_unlock"]["consumption_receipt_sha256"] = sha256_file(
                consumption_path
            )
            attempt["author_unlock"]["consumed_before_rgb_access"] = True
            attempt["status"] = "author_unlock_consumed_before_rgb_access"
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
        attempt["status"] = "rgb_inputs_verified"
        attempt["rgb_input_receipt_sha256"] = sha256_file(run_directory / "analysis_input_receipt.json")
        write_json_atomic(run_directory / "attempt.json", attempt)

        detector_manifest = _run_detector_stage(protocol, inventory, role, run_directory)
        sam2_manifest = _run_sam2_stage(protocol, inventory, role, run_directory, detector_manifest)
        analysis_manifest = _run_analysis_stage(
            protocol,
            inventory,
            role,
            run_directory,
            detector_manifest,
            sam2_manifest,
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
            "author_unlock": attempt.get("author_unlock") if role == "heldout" else None,
            "inventory": {
                "stream_ids": [item.stream_id for item in inventory],
                "stream_count": len(inventory),
                "frame_count": sum(item.frame_count for item in inventory),
            },
            "stage_manifests": {
                "analysis_inputs": _file_reference(run_directory, run_directory / "analysis_input_receipt.json"),
                "detector": _file_reference(run_directory, run_directory / "detector_manifest.json"),
                "sam2": _file_reference(run_directory, run_directory / "sam2_manifest.json"),
                "analysis": _file_reference(run_directory, run_directory / "analysis_manifest.json"),
            },
            "coverage": {
                "candidate_count": detector_manifest["candidate_count"],
                "valid_mask_count": sam2_manifest["valid_mask_count"],
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
    unlock_path: Path,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    return _run_evaluation(
        protocol=protocol,
        run_directory=run_directory,
        expected_role="heldout",
        inventory=protocol.heldout,
        output_filename="evaluation.json",
        completion_status="completed_one_time_heldout_evaluation",
        attempt_status="heldout_evaluation_completed",
        unlock_path=unlock_path,
    )


def run_development_smoke_evaluation(
    *,
    protocol_path: Path,
    run_directory: Path,
) -> dict[str, Any]:
    protocol = load_frozen_protocol(protocol_path)
    return _run_evaluation(
        protocol=protocol,
        run_directory=run_directory,
        expected_role="development_smoke",
        inventory=inventory_for_role(protocol, "development_smoke"),
        output_filename="development_smoke_evaluation.json",
        completion_status="completed_development_smoke_evaluation",
        attempt_status="development_smoke_evaluation_completed",
        unlock_path=None,
    )


def _run_evaluation(
    *,
    protocol: FrozenGateDProtocol,
    run_directory: Path,
    expected_role: str,
    inventory: Sequence[GateDStream],
    output_filename: str,
    completion_status: str,
    attempt_status: str,
    unlock_path: Path | None,
) -> dict[str, Any]:
    run_root = Path(run_directory).resolve(strict=True)
    inference_path = run_root / "inference_manifest.json"
    inference = _json_object(inference_path, "Gate D inference manifest")
    if inference.get("schema_version") != INFERENCE_SCHEMA:
        raise GateDContractError("unsupported Gate D inference manifest schema")
    if inference.get("status") != "inference_completed_before_ground_truth":
        raise GateDContractError("Gate D evaluation requires completed persisted inference")
    if inference.get("role") != expected_role:
        raise GateDContractError(f"Gate D evaluation expected role {expected_role}")
    run_id = _text(inference, "run_id", "inference manifest")
    authorization: Mapping[str, Any] | None = None
    if expected_role == "heldout":
        if unlock_path is None:
            raise GateDContractError("held-out evaluation requires the author unlock")
        authorization = validate_author_unlock_consumption(
            unlock_path,
            protocol,
            expected_run_id=run_id,
            expected_run_directory=run_root,
        )
    elif unlock_path is not None:
        raise GateDContractError("development smoke evaluation does not accept an unlock")
    if _mapping(inference, "protocol", "inference manifest").get("sha256") != protocol.sha256:
        raise GateDContractError("inference manifest protocol digest mismatch")
    state = git_state(protocol.repository_root)
    if _mapping(inference, "git", "inference manifest").get("commit") != state.commit:
        raise GateDContractError("evaluation Git commit differs from inference")
    expected_streams = [item.stream_id for item in inventory]
    if _mapping(inference, "inventory", "inference manifest").get("stream_ids") != expected_streams:
        raise GateDContractError("inference manifest evaluation inventory mismatch")
    boundary = _mapping(inference, "ground_truth_boundary", "inference manifest")
    if (
        boundary.get("ground_truth_opened") is not False
        or boundary.get("all_prediction_artifacts_persisted") is not True
    ):
        raise GateDContractError("persisted inference does not satisfy the ground-truth boundary")
    _verify_artifact_inventory(run_root, inference.get("artifact_inventory"))
    output_path = run_root / output_filename
    if output_path.exists():
        raise GateDContractError("refusing to overwrite or rerun a completed Gate D evaluation")

    analysis_input_receipt = _verified_stage_manifest(
        run_root,
        inference,
        "analysis_inputs",
        INPUT_RECEIPT_SCHEMA,
    )
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
    attempt_path = run_root / "attempt.json"
    attempt = _json_object(attempt_path, "Gate D attempt")
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("run_id") != run_id
        or attempt.get("role") != expected_role
        or attempt.get("protocol_sha256") != protocol.sha256
        or attempt.get("status") != "inference_completed_before_ground_truth"
        or attempt.get("ground_truth_opened") is not False
    ):
        raise GateDContractError("Gate D attempt is not eligible to cross the ground-truth boundary")
    if expected_role == "heldout":
        assert authorization is not None
        unlock_file = Path(unlock_path).resolve(strict=True)  # type: ignore[arg-type]
        consumption_file = Path(authorization["consumption_path"])
        expected_authorization = {
            "sha256": sha256_file(unlock_file),
            "decision": authorization["unlock"]["decision"],
            "consumption_receipt_sha256": sha256_file(consumption_file),
            "consumed_before_rgb_access": True,
        }
        if (
            attempt.get("author_unlock") != expected_authorization
            or inference.get("author_unlock") != expected_authorization
        ):
            raise GateDContractError("held-out authorization evidence differs from inference")
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
            "claim_scope": protocol.payload["claim_scope"],
            "limitations": [
                "OCID supplies physical-instance proxies, not real-image visual-type ground truth.",
                "Grouping and event artifacts are qualitative model outputs and are not visual-type accuracy metrics.",
                "Physical removal, return after removal, and object motion are not supported by this benchmark.",
                "Mask diagnostics are conditional on bbox-matched detections and do not replace candidate metrics.",
                "Continuity metrics are conditional on candidates assigned to physical-instance proxies.",
                "No model, prompt, threshold, or post-processing choice was changed for this evaluation.",
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


def _run_detector_stage(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
    role: str,
    run_directory: Path,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise GateDContractError("Gate D protocol requires CUDA, but CUDA is unavailable")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.backends.cudnn.benchmark = False
    models = _mapping(protocol.payload, "models", "protocol")
    model_spec = _mapping(models, "candidate_extractor", "models")
    model_directory = _repo_path(protocol, _text(model_spec, "directory", "candidate extractor"), directory=True)
    processor, model = grounding._load_model_processor(model_directory, torch_module=torch, device="cuda")
    prompt, geometry = _detector_profiles(protocol)
    streams: dict[str, Any] = {}
    total_candidates = 0
    total_raw = 0
    total_rejected = 0
    started = time.perf_counter()
    try:
        for item in inventory:
            decoded = _load_stream(item, protocol.sha256)
            raw, runtime = grounding._infer_stream(
                decoded,
                processor=processor,
                model=model,
                torch_module=torch,
                device="cuda",
                prompt=prompt.text,
            )
            _candidates, records = hardening.candidates_for_profile(
                raw,
                prompt=prompt,
                geometry=geometry,
                image_sizes={frame.frame_id: frame.image_size for frame in decoded.frames},
            )
            accepted = [dict(record) for record in records if record["geometry_rejected"] is False]
            stream_payload = {
                "schema_version": "ocid-gate-d-detector-stream-1.0",
                "stream_id": item.stream_id,
                "scene_group_id": item.scene_group_id,
                "frame_count": len(decoded.frames),
                "raw_predictions": [hardening._raw_payload(row) for row in raw],
                "postprocess_records": [dict(record) for record in records],
                "accepted_candidates": accepted,
                "runtime": runtime,
            }
            path = run_directory / "detector" / f"{item.stream_id}.json"
            write_json_exclusive(path, stream_payload)
            streams[item.stream_id] = _file_reference(run_directory, path)
            total_raw += len(raw)
            total_candidates += len(accepted)
            total_rejected += len(records) - len(accepted)
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": DETECTOR_SCHEMA,
        "status": "detector_inference_completed_before_ground_truth",
        "role": role,
        "protocol_sha256": protocol.sha256,
        "configuration": protocol.payload["pipeline"]["candidate_extraction"],
        "model": {
            "family": model_spec["family"],
            "hf_revision": model_spec["hf_revision"],
            "model_safetensors_sha256": model_spec["assets"]["model.safetensors"]["sha256"],
            "device": "cuda",
            "dtype": "float32",
            "local_files_only": True,
        },
        "streams": streams,
        "stream_count": len(streams),
        "frame_count": sum(item.frame_count for item in inventory),
        "raw_prediction_count": total_raw,
        "candidate_count": total_candidates,
        "geometry_rejected_count": total_rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "ground_truth_opened": False,
    }
    write_json_exclusive(run_directory / "detector_manifest.json", manifest)
    return manifest


def _run_sam2_stage(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
    role: str,
    run_directory: Path,
    detector_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    models = _mapping(protocol.payload, "models", "protocol")
    model_spec = _mapping(models, "mask_refiner", "models")
    model_directory = _repo_path(protocol, _text(model_spec, "directory", "mask refiner"), directory=True)
    cleanup_payload = _mapping(
        _mapping(_mapping(protocol.payload, "pipeline", "protocol"), "mask_refinement", "pipeline"),
        "cleanup",
        "mask refinement",
    )
    cleanup = MaskCleanupConfig(
        min_component_pixels=int(cleanup_payload["min_component_pixels"]),
        min_component_area_ratio=float(cleanup_payload["min_component_area_ratio"]),
        connectivity=int(cleanup_payload["connectivity"]),  # type: ignore[arg-type]
    )
    refiner = load_local_sam2_bbox_refiner(model_directory, device="cuda")
    records_by_stream: dict[str, list[dict[str, Any]]] = {}
    valid_count = 0
    fallback_count = 0
    started = time.perf_counter()
    try:
        for item in inventory:
            decoded = _load_stream(item, protocol.sha256)
            detector = _load_referenced_json(run_directory, detector_manifest, item.stream_id)
            rows = detector.get("accepted_candidates")
            if not isinstance(rows, list):
                raise GateDContractError(f"detector candidates missing for {item.stream_id}")
            by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                if not isinstance(row, Mapping):
                    raise GateDContractError("malformed persisted detector candidate")
                by_frame[_text(row, "frame_id", "detector candidate")].append(row)
            stream_records: list[dict[str, Any]] = []
            for frame in decoded.frames:
                candidates = sorted(
                    by_frame.get(frame.frame_id, ()),
                    key=lambda row: str(row["candidate_id"]),
                )
                if not candidates:
                    continue
                boxes = tuple(_bbox(_mapping(row, "bbox", "detector candidate")) for row in candidates)
                results = refiner.refine(_frame_array(frame), boxes, cleanup=cleanup)
                if len(results) != len(candidates):
                    raise GateDContractError("SAM2 result count differs from candidate count")
                for index, (candidate, result) in enumerate(zip(candidates, results, strict=True)):
                    if result.status != "valid" or result.raw_mask is None or result.cleaned_mask is None:
                        fallback_count += 1
                        raise GateDContractError(
                            f"SAM2 mask fallback is forbidden: {candidate['candidate_id']} "
                            f"({result.fallback_reason})"
                        )
                    safe_name = _safe_candidate_artifact_name(str(candidate["candidate_id"]), index)
                    raw_path = run_directory / "masks" / item.stream_id / frame.frame_id / f"{safe_name}.raw.png"
                    cleaned_path = (
                        run_directory / "masks" / item.stream_id / frame.frame_id / f"{safe_name}.cleaned.png"
                    )
                    _save_binary_mask(raw_path, result.raw_mask)
                    _save_binary_mask(cleaned_path, result.cleaned_mask)
                    quality = result.quality
                    assert quality is not None
                    stream_records.append(
                        {
                            "stream_id": item.stream_id,
                            "candidate_id": candidate["candidate_id"],
                            "frame_id": candidate["frame_id"],
                            "frame_index": candidate["frame_index"],
                            "score": candidate["score"],
                            "phrase": candidate["phrase"],
                            "source_bbox": candidate["bbox"],
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
                    valid_count += 1
            records_by_stream[item.stream_id] = sorted(
                stream_records,
                key=lambda row: (int(row["frame_index"]), str(row["candidate_id"])),
            )
    finally:
        del refiner
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    candidate_count = int(detector_manifest["candidate_count"])
    if valid_count != candidate_count or fallback_count:
        raise GateDContractError(
            f"Gate D requires one valid mask per candidate: {valid_count}/{candidate_count}"
        )
    manifest = {
        "schema_version": SAM2_SCHEMA,
        "status": "rgb_inference_completed_before_ground_truth",
        "scope": role,
        "heldout_access": "author_unlocked_once" if role == "heldout" else "none",
        "protocol_sha256": protocol.sha256,
        "configuration": protocol.payload["pipeline"]["mask_refinement"],
        "model": {
            "family": model_spec["family"],
            "model_safetensors_sha256": model_spec["assets"]["model.safetensors"]["sha256"],
            "device": "cuda",
            "dtype": "float32",
            "local_files_only": True,
        },
        "records_by_stream": records_by_stream,
        "candidate_count": candidate_count,
        "valid_mask_count": valid_count,
        "fallback_count": fallback_count,
        "elapsed_seconds": time.perf_counter() - started,
        "ground_truth_opened": False,
    }
    write_json_exclusive(run_directory / "sam2_manifest.json", manifest)
    return manifest


def _run_analysis_stage(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
    role: str,
    run_directory: Path,
    detector_manifest: Mapping[str, Any],
    sam2_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    models = _mapping(protocol.payload, "models", "protocol")
    model_spec = _mapping(models, "representation", "models")
    checkpoint = _repo_path(
        protocol,
        _text(_mapping(model_spec, "checkpoint", "representation"), "path", "DINOv2 checkpoint"),
        directory=False,
    )
    source = _repo_path(protocol, _text(model_spec, "source_directory", "representation"), directory=True)
    pipeline = _mapping(protocol.payload, "pipeline", "protocol")
    representation_payload = _mapping(pipeline, "representation", "pipeline")
    provider = LocalDinoV2Provider(
        source_dir=source,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=_mapping(model_spec, "checkpoint", "representation")["sha256"],
        expected_checkpoint_size_bytes=int(_mapping(model_spec, "checkpoint", "representation")["size_bytes"]),
        expected_source_tree_fingerprint=model_spec["source_tree_fingerprint_sha256"],
        model_name=model_spec["model_name"],
        embedding_dimension=int(model_spec["embedding_dimension"]),
        device_policy="cuda",
        batch_size=int(representation_payload["batch_size"]),
    )
    representation_config = DinoV2RepresentationConfig(
        expected_checkpoint_sha256=_mapping(model_spec, "checkpoint", "representation")["sha256"],
        expected_checkpoint_size_bytes=int(_mapping(model_spec, "checkpoint", "representation")["size_bytes"]),
        expected_source_tree_fingerprint=model_spec["source_tree_fingerprint_sha256"],
        model_name=model_spec["model_name"],
        embedding_dimension=int(model_spec["embedding_dimension"]),
        variant=DINO_MASK_NEUTRAL_VARIANT,
        input_size=int(representation_payload["input_size"]),
        context_padding_ratio=float(representation_payload["context_padding_ratio"]),
        device_policy="cuda",
        batch_size=int(representation_payload["batch_size"]),
    )
    scoring, matching, grouping_config, events_config = _matching_configs(protocol)
    scorer = DinoV2CosineScorer()
    streams: dict[str, Any] = {}
    total_embeddings = 0
    embedding_failures = 0
    started = time.perf_counter()
    try:
        snapshots = _snapshots_from_persisted_masks(
            protocol,
            inventory,
            run_directory,
            detector_manifest,
            sam2_manifest,
        )
        for item in inventory:
            decoded = _load_stream(item, protocol.sha256)
            snapshot = snapshots[item.stream_id]
            batch = build_dinov2_representations(decoded, snapshot, provider, representation_config)
            embedding_failures += len(batch.errors)
            if batch.errors or len(batch.records) != len(snapshot.result.candidates):
                raise GateDContractError(
                    f"DINOv2 representation coverage failed for {item.stream_id}: "
                    f"records={len(batch.records)}, candidates={len(snapshot.result.candidates)}, "
                    f"errors={len(batch.errors)}"
                )
            representation_by_id = {record.candidate_id: record for record in batch.records}
            candidate_ids = [candidate.candidate_id for candidate in snapshot.result.candidates]
            embeddings = np.stack(
                [
                    np.asarray(representation_by_id[candidate_id].payload.embedding, dtype=np.float32)
                    for candidate_id in candidate_ids
                ],
                axis=0,
            ) if candidate_ids else np.empty((0, int(model_spec["embedding_dimension"])), dtype=np.float32)
            npz_path = run_directory / "analysis" / item.stream_id / "embeddings.npz"
            _save_embeddings(npz_path, embeddings)
            matching_results = []
            candidates = snapshot.result.candidates
            for left, right in zip(decoded.frames, decoded.frames[1:]):
                frame_pair = FramePair(
                    from_frame_id=left.frame_id,
                    from_frame_index=left.record.index,
                    from_frame_size=left.image_size,
                    to_frame_id=right.frame_id,
                    to_frame_index=right.record.index,
                    to_frame_size=right.image_size,
                )
                batch_match = match_neighboring_frames(
                    stream_id=item.stream_id,
                    frame_pair=frame_pair,
                    from_candidates=tuple(
                        candidate for candidate in candidates if candidate.frame_id == left.frame_id
                    ),
                    to_candidates=tuple(candidate for candidate in candidates if candidate.frame_id == right.frame_id),
                    representations=batch.records,
                    representation_variant_id=representation_config.variant,
                    scorer=scorer,
                    scoring_config=scoring,
                    matching_config=matching,
                )
                if batch_match.errors:
                    raise GateDContractError(f"neighbor matching failed for {item.stream_id}")
                matching_results.append(batch_match.result)
            matching_tuple = tuple(matching_results)
            grouping = group_recurring_visual_types(
                stream_id=item.stream_id,
                candidates=candidates,
                representations=batch.records,
                matching_results=matching_tuple,
                representation_variant_id=representation_config.variant,
                scorer=scorer,
                config=grouping_config,
            )
            event_batch = build_change_events(
                stream_id=item.stream_id,
                candidates=candidates,
                grouping=grouping,
                matching_results=matching_tuple,
                config=events_config,
            )
            stream_payload = {
                "schema_version": "ocid-gate-d-stream-analysis-1.0",
                "stream_id": item.stream_id,
                "candidate_ids": candidate_ids,
                "embedding_artifact": _file_reference(run_directory, npz_path),
                "embedding_shape": list(embeddings.shape),
                "representation": {
                    "variant": representation_config.variant,
                    "semantic_config_digest": batch.semantic_config_digest,
                    "requested_device": batch.requested_device,
                    "resolved_device": batch.resolved_device,
                    "provider": dict(provider.provider_metadata()),
                    "model": dict(provider.model_metadata()),
                    "warning_count": len(batch.warnings),
                    "error_count": len(batch.errors),
                },
                "matching_results": to_json_compatible(matching_tuple),
                "grouping": to_json_compatible(grouping),
                "frame_comparisons": to_json_compatible(event_batch.comparisons),
                "change_events": to_json_compatible(event_batch.events),
                "event_warning_count": len(event_batch.warnings),
                "event_error_count": len(event_batch.errors),
            }
            path = run_directory / "analysis" / item.stream_id / "analysis.json"
            write_json_exclusive(path, stream_payload)
            overlays = _render_mask_type_overlays(
                decoded,
                snapshot,
                grouping,
                sam2_manifest["records_by_stream"][item.stream_id],
                run_directory,
            )
            streams[item.stream_id] = {
                "analysis": _file_reference(run_directory, path),
                "embedding_artifact": _file_reference(run_directory, npz_path),
                "overlay_manifest": overlays,
                "candidate_count": len(candidate_ids),
                "embedding_count": len(candidate_ids),
                "matching_comparison_count": len(matching_tuple),
                "group_count": len(grouping.recurring_types),
                "event_count": len(event_batch.events),
            }
            total_embeddings += len(candidate_ids)
    finally:
        del provider
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    manifest = {
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
            "model_name": model_spec["model_name"],
            "checkpoint_sha256": model_spec["checkpoint"]["sha256"],
            "source_tree_fingerprint_sha256": model_spec["source_tree_fingerprint_sha256"],
            "embedding_dimension": model_spec["embedding_dimension"],
        },
        "streams": streams,
        "candidate_count": int(detector_manifest["candidate_count"]),
        "embedding_count": total_embeddings,
        "embedding_failure_count": embedding_failures,
        "elapsed_seconds": time.perf_counter() - started,
        "ground_truth_opened": False,
    }
    if total_embeddings != int(detector_manifest["candidate_count"]):
        raise GateDContractError("Gate D embedding coverage differs from detector coverage")
    write_json_exclusive(run_directory / "analysis_manifest.json", manifest)
    return manifest


def _snapshots_from_persisted_masks(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
    run_directory: Path,
    detector_manifest: Mapping[str, Any],
    sam2_manifest: Mapping[str, Any],
) -> dict[str, CandidateExtractionSnapshot]:
    records_by_stream = sam2_manifest.get("records_by_stream")
    if not isinstance(records_by_stream, Mapping):
        raise GateDContractError("SAM2 manifest lacks records_by_stream")
    result: dict[str, CandidateExtractionSnapshot] = {}
    for item in inventory:
        decoded = _load_stream(item, protocol.sha256)
        detector = _load_referenced_json(run_directory, detector_manifest, item.stream_id)
        accepted = detector.get("accepted_candidates")
        mask_rows = records_by_stream.get(item.stream_id)
        if not isinstance(accepted, list) or not isinstance(mask_rows, list):
            raise GateDContractError(f"persisted candidates or masks missing for {item.stream_id}")
        masks_by_id = {
            str(row["candidate_id"]): row
            for row in mask_rows
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        }
        if len(masks_by_id) != len(mask_rows):
            raise GateDContractError(f"duplicate or malformed mask rows for {item.stream_id}")
        frame_lookup = {frame.frame_id: frame for frame in decoded.frames}
        producer = _producer("gate_d_candidate_extraction", protocol.sha256)
        candidates: list[CandidateRecord] = []
        mask_store: dict[str, CandidateMaskRecord] = {}
        per_frame_counts: dict[str, int] = defaultdict(int)
        for row in sorted(accepted, key=lambda value: (int(value["frame_index"]), str(value["candidate_id"]))):
            candidate_id = _text(row, "candidate_id", "detector candidate")
            frame_id = _text(row, "frame_id", "detector candidate")
            frame = frame_lookup.get(frame_id)
            if frame is None:
                raise GateDContractError(f"candidate references unknown frame: {candidate_id}")
            mask_row = masks_by_id.get(candidate_id)
            if mask_row is None or mask_row.get("status") != "valid":
                raise GateDContractError(f"candidate lacks a valid persisted mask: {candidate_id}")
            bbox = _bbox(_mapping(row, "bbox", "detector candidate"))
            if bbox != _bbox(_mapping(mask_row, "source_bbox", "SAM2 mask row")):
                raise GateDContractError(f"candidate/mask bbox mismatch: {candidate_id}")
            cleaned = _mapping(mask_row, "cleaned_mask", "SAM2 mask row")
            full_mask_path = _run_artifact_path(run_directory, _text(cleaned, "path", "cleaned mask"))
            if cleaned.get("file_sha256") != sha256_file(full_mask_path):
                raise GateDContractError(f"cleaned mask file digest mismatch: {candidate_id}")
            full_mask = _load_binary_mask(full_mask_path)
            if binary_mask_sha256(full_mask) != cleaned.get("binary_mask_sha256"):
                raise GateDContractError(f"cleaned mask binary digest mismatch: {candidate_id}")
            x, y, width, height = _integral_bbox(bbox)
            local_mask = np.ascontiguousarray(full_mask[y : y + height, x : x + width], dtype=np.bool_)
            if local_mask.shape != (height, width) or not local_mask.any():
                raise GateDContractError(f"invalid local mask crop: {candidate_id}")
            mask_ref = f"gate-d-mask:{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:24]}"
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
                feature_schema_id="grounding_dino_gate_d_geometry_v1",
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
                    schema_version="ocid-gate-d-candidate-record-1.0",
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
                candidate_source="grounding_dino_sam2_gate_d_v1",
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
                    record_id=f"gate-d-candidate-diagnostics:{item.stream_id}:{frame.frame_id}",
                    schema_version="ocid-gate-d-frame-candidate-diagnostics-1.0",
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
                record_id=f"gate-d-candidate-extraction:{item.stream_id}",
                schema_version="ocid-gate-d-candidate-extraction-result-1.0",
                stream_id=item.stream_id,
                producer=producer,
                context=StageContext(),
            ),
            extractor_source="grounding_dino_sam2_gate_d_v1",
            candidates=tuple(candidates),
            frame_diagnostics=diagnostics,
        )
        result[item.stream_id] = CandidateExtractionSnapshot(result=extraction, masks=mask_store)
    return result


def _verify_stage_contracts(
    protocol: FrozenGateDProtocol,
    expected_role: str,
    inventory: Sequence[GateDStream],
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
            raise GateDContractError(f"{label} stage violates the frozen inference boundary")
    if (
        detector.get("role") != expected_role
        or sam2.get("scope") != expected_role
        or analysis.get("role") != expected_role
    ):
        raise GateDContractError("stage role differs from the inference role")
    detector_streams = _mapping(detector, "streams", "detector manifest")
    sam2_streams = _mapping(sam2, "records_by_stream", "SAM2 manifest")
    analysis_streams = _mapping(analysis, "streams", "analysis manifest")
    if any(
        set(streams) != expected_stream_ids
        for streams in (detector_streams, sam2_streams, analysis_streams)
    ):
        raise GateDContractError("stage stream inventory differs from the frozen inference inventory")
    coverage = _mapping(inference, "coverage", "inference manifest")
    candidate_count = coverage.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
        or detector.get("candidate_count") != candidate_count
        or sam2.get("candidate_count") != candidate_count
        or sam2.get("valid_mask_count") != candidate_count
        or sam2.get("fallback_count") != 0
        or analysis.get("candidate_count") != candidate_count
        or analysis.get("embedding_count") != candidate_count
        or analysis.get("embedding_failure_count") != 0
        or coverage.get("valid_mask_count") != candidate_count
        or coverage.get("embedding_count") != candidate_count
        or coverage.get("mask_fallback_count") != 0
        or coverage.get("embedding_failure_count") != 0
    ):
        raise GateDContractError("stage coverage differs from the strict Gate D contract")


def _verify_analysis_inputs_unchanged(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
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
        raise GateDContractError("analysis input receipt differs from the inference contract")
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
        raise GateDContractError("analysis inputs changed after inference")


def _verify_analysis_embeddings(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
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
            raise GateDContractError(f"analysis artifact digest mismatch: {item.stream_id}")
        stream_analysis = _json_object(analysis_path, "persisted stream analysis")
        candidates = snapshots[item.stream_id].result.candidates
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if (
            stream_analysis.get("schema_version") != "ocid-gate-d-stream-analysis-1.0"
            or stream_analysis.get("stream_id") != item.stream_id
            or stream_analysis.get("candidate_ids") != candidate_ids
            or stream_reference.get("candidate_count") != len(candidate_ids)
            or stream_reference.get("embedding_count") != len(candidate_ids)
        ):
            raise GateDContractError(f"analysis candidate index differs: {item.stream_id}")
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
            raise GateDContractError(f"embedding references differ: {item.stream_id}")
        embedding_path = _run_artifact_path(
            run_directory,
            _text(embedding_reference, "path", "embedding artifact"),
        )
        if embedding_reference.get("sha256") != sha256_file(embedding_path):
            raise GateDContractError(f"embedding artifact digest mismatch: {item.stream_id}")
        with np.load(embedding_path, allow_pickle=False) as archive:
            if set(archive.files) != {"embeddings"}:
                raise GateDContractError("Gate D embedding NPZ must contain only embeddings")
            matrix = np.asarray(archive["embeddings"], dtype=np.float32).copy()
        expected_shape = (len(candidate_ids), embedding_dimension)
        if (
            matrix.shape != expected_shape
            or stream_analysis.get("embedding_shape") != list(expected_shape)
            or not np.isfinite(matrix).all()
        ):
            raise GateDContractError(f"invalid persisted embedding matrix for {item.stream_id}")
        if len(matrix):
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
                raise GateDContractError(f"embeddings are not normalized for {item.stream_id}")
        result[item.stream_id] = matrix
    return result


def _evaluate_candidate_metrics(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
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
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
    run_directory: Path,
    snapshots: Mapping[str, CandidateExtractionSnapshot],
    sam2_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selected = {
        item.stream_id: snapshots[item.stream_id].result.candidates
        for item in inventory
    }
    ocid_root = protocol.repository_root / "data" / "ocid" / "raw" / "OCID-dataset"
    summary, rows, _reviewed_masks = evaluate_masks(
        inventory=inventory,  # type: ignore[arg-type]
        selected=selected,  # type: ignore[arg-type]
        inference_manifest=sam2_manifest,
        reviewed_root=protocol.reviewed_root,
        ocid_root=ocid_root,
        artifact_root=run_directory,
    )
    return {
        "summary": summary,
        "matched_observation_count": len(rows),
        "rows": rows,
        "conditional_on_bbox_match": True,
    }


def _evaluate_physical_continuity(
    protocol: FrozenGateDProtocol,
    inventory: Sequence[GateDStream],
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
    inventory: Sequence[GateDStream],
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


def _evaluation_iou_grid(protocol: FrozenGateDProtocol) -> tuple[float, ...]:
    evaluation = _mapping(protocol.payload, "evaluation", "protocol")
    values = evaluation.get("candidate_bbox_iou_grid")
    if not isinstance(values, list) or not values:
        raise GateDContractError("protocol evaluation IoU grid must be a non-empty list")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise GateDContractError("protocol evaluation IoU grid must be numeric") from error
    if any(not 0.0 < value <= 1.0 for value in result) or tuple(sorted(set(result))) != result:
        raise GateDContractError("protocol evaluation IoU grid must be unique and increasing in (0, 1]")
    return result


def _candidate_evaluation_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    assignments = value.get("assignments", ())
    if not isinstance(assignments, (list, tuple)):
        raise GateDContractError("candidate evaluation assignments must be a sequence")
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
    protocol: FrozenGateDProtocol,
) -> tuple[PairScoringConfig, MatchingConfig, GroupingConfig, EventConfig]:
    pipeline = _mapping(protocol.payload, "pipeline", "protocol")
    score = _mapping(pipeline, "scoring", "pipeline")
    match = _mapping(pipeline, "matching", "pipeline")
    group = _mapping(pipeline, "grouping", "pipeline")
    event = _mapping(pipeline, "events", "pipeline")
    scoring = PairScoringConfig(
        scorer_id=score["scorer_id"],
        scorer_version=score["scorer_version"],
        visual_gate=float(score["visual_gate"]),
        spatial_gate=score["spatial_gate"],
    )
    matching = MatchingConfig(
        unmatched_pair_cost=float(match["unmatched_pair_cost"]),
        local_margin_gate=float(match["local_margin_gate"]),
        global_margin_gate=float(match["global_margin_gate"]),
        severe_quality_flags=tuple(match["severe_quality_flags"]),
        assignment_policy_id=match["assignment_policy_id"],
        assignment_policy_version=match["assignment_policy_version"],
    )
    grouping = GroupingConfig(
        medoid_gate=float(group["medoid_gate"]),
        support_quantile=float(group["support_quantile"]),
        quantile_gate=float(group["quantile_gate"]),
        support_pair_gate=float(group["support_pair_gate"]),
        support_ratio_gate=float(group["support_ratio_gate"]),
        visual_gate=float(group["visual_gate"]),
        medoid_weight=float(group["medoid_weight"]),
        quantile_weight=float(group["quantile_weight"]),
        support_ratio_weight=float(group["support_ratio_weight"]),
        second_best_margin=float(group["second_best_margin"]),
        representation_variant_id=DINO_MASK_NEUTRAL_VARIANT,
        scorer_id=score["scorer_id"],
        scorer_version=score["scorer_version"],
        expose_unmerged_alternatives=bool(group["expose_unmerged_alternatives"]),
        use_uncertain_temporal_seeds=bool(group["use_uncertain_temporal_seeds"]),
        uncertain_seed_visual_gate=float(group["uncertain_seed_visual_gate"]),
        uncertain_seed_spatial_gate=float(group["uncertain_seed_spatial_gate"]),
        coframe_aspect_log_gate=float(group["coframe_aspect_log_gate"]),
        coframe_area_log_gate=float(group["coframe_area_log_gate"]),
        coframe_fill_ratio_gate=float(group["coframe_fill_ratio_gate"]),
        coframe_hole_count_gate=int(group["coframe_hole_count_gate"]),
        coframe_visual_gate=float(group["coframe_visual_gate"]),
        coframe_margin_bypass=bool(group["coframe_margin_bypass"]),
        severe_quality_flags=tuple(group["severe_quality_flags"]),
        grouping_policy_id=group["grouping_policy_id"],
        grouping_policy_version=group["grouping_policy_version"],
    )
    events = EventConfig(
        position_threshold_norm=float(event["position_threshold_norm"]),
        severe_quality_flags=tuple(event["severe_quality_flags"]),
        event_policy_id=event["event_policy_id"],
        event_policy_version=event["event_policy_version"],
    )
    return scoring, matching, grouping, events


def _detector_profiles(
    protocol: FrozenGateDProtocol,
) -> tuple[hardening.PromptProfile, hardening.GeometryProfile]:
    payload = _mapping(
        _mapping(protocol.payload, "pipeline", "protocol"),
        "candidate_extraction",
        "pipeline",
    )
    if (
        float(payload["raw_score_floor"]) != grounding.RAW_POSTPROCESS_FLOOR
        or float(payload["score_threshold"]) != hardening.FROZEN_SCORE_THRESHOLD
        or float(payload["class_agnostic_nms_iou"]) != hardening.FROZEN_NMS_IOU
    ):
        raise GateDContractError("protocol detector gates differ from selected implementation constants")
    geometry = _mapping(payload, "geometry_filter", "candidate extraction")
    return (
        hardening.PromptProfile("p_object", str(payload["prompt"]), True, "gate_c1_raw_reuse"),
        hardening.GeometryProfile(
            str(geometry["policy_id"]),
            min_area_ratio=float(geometry["reject_when_area_ratio_at_least"]),
            min_span_ratio=float(geometry["and_either_span_ratio_at_least"]),
        ),
    )


def _load_stream(item: GateDStream, protocol_sha256: str):
    producer = _producer("stream_input", protocol_sha256)
    decoded = load_decoded_stream(
        ManifestLoadRequest(stream_root=item.stream_directory.resolve(strict=True), producer=producer)
    )
    if decoded.stream.stream_id != item.stream_id or len(decoded.frames) != item.frame_count:
        raise GateDContractError(f"decoded stream differs from frozen inventory: {item.stream_id}")
    return decoded


def _producer(stage: str, protocol_sha256: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0.0",
        config_version="1.0.0",
        config_digest=f"sha256:{protocol_sha256}",
    )


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
        "schema_version": "ocid-gate-d-mask-type-overlays-1.0",
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
        raise GateDContractError("inference artifact inventory must be a non-empty list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise GateDContractError("malformed inference artifact entry")
        relative = _text(row, "path", "inference artifact")
        if relative in seen:
            raise GateDContractError("duplicate inference artifact path")
        seen.add(relative)
        path = _run_artifact_path(run_directory, relative)
        if path.stat().st_size != row.get("size_bytes") or sha256_file(path) != row.get("sha256"):
            raise GateDContractError(f"inference artifact changed: {relative}")


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
        raise GateDContractError(f"{key} stage manifest digest mismatch")
    payload = _json_object(path, f"{key} stage manifest")
    if payload.get("schema_version") != schema:
        raise GateDContractError(f"{key} stage manifest schema mismatch")
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
        raise GateDContractError(f"stream artifact digest mismatch: {stream_id}")
    return _json_object(path, f"stream artifact {stream_id}")


def _file_reference(root: Path, path: Path) -> dict[str, Any]:
    full = path.resolve(strict=True)
    try:
        relative = full.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise GateDContractError("artifact path escapes the Gate D run directory") from error
    return {"path": relative, "size_bytes": full.stat().st_size, "sha256": sha256_file(full)}


def _run_artifact_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise GateDContractError("run artifact must be a safe relative path")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise GateDContractError("run artifact escapes the run directory") from error
    if not path.is_file():
        raise GateDContractError(f"run artifact is not a file: {path}")
    return path


def _repo_path(protocol: FrozenGateDProtocol, value: str, *, directory: bool) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise GateDContractError("protocol repository path must be relative")
    path = (protocol.repository_root / Path(*relative.parts)).resolve(strict=True)
    try:
        path.relative_to(protocol.repository_root)
    except ValueError as error:
        raise GateDContractError("protocol repository path escapes the repository") from error
    if directory and not path.is_dir():
        raise GateDContractError(f"protocol directory is missing: {path}")
    if not directory and not path.is_file():
        raise GateDContractError(f"protocol file is missing: {path}")
    return path


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    value = np.asarray(mask, dtype=np.bool_)
    if value.ndim != 2 or not value.any():
        raise GateDContractError("refusing to persist an empty or malformed mask")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path, format="PNG", optimize=True)


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        value = np.asarray(source).copy()
    if value.ndim != 2 or not np.isin(value, (0, 255)).all():
        raise GateDContractError(f"persisted mask is not binary: {path}")
    return np.ascontiguousarray(value > 0, dtype=np.bool_)


def _save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    value = np.asarray(embeddings, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise GateDContractError("embedding matrix must be finite and two-dimensional")
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
        raise GateDContractError(f"malformed bbox: {payload!r}") from error


def _bbox_payload(value: BBox | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value.x, "y": value.y, "width": value.width, "height": value.height}


def _integral_bbox(value: BBox) -> tuple[int, int, int, int]:
    numbers = (value.x, value.y, value.width, value.height)
    integers = tuple(int(number) for number in numbers)
    if any(float(number) != float(integer) for number, integer in zip(numbers, integers, strict=True)):
        raise GateDContractError("Gate D candidate bbox must have integral coordinates")
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


def _safe_candidate_artifact_name(candidate_id: str, source_index: int) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]
    return f"candidate_{source_index:03d}_{digest}"


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
        raise GateDContractError(f"{label}.{key} must be an object")
    return result


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise GateDContractError(f"{label}.{key} must be a non-empty string")
    return result


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateDContractError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise GateDContractError(f"{label} must be a JSON object")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate frozen metadata without stream access")
    preflight.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser("smoke-development", help="run one frozen development RGB stream without GT")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--run-id", required=True)

    inference = subparsers.add_parser("infer-heldout", help="run the exact author-unlocked held-out inference once")
    inference.add_argument("--output-root", type=Path, required=True)
    inference.add_argument("--run-id", required=True)
    inference.add_argument("--unlock", type=Path, required=True)

    smoke_evaluation = subparsers.add_parser(
        "evaluate-development-smoke",
        help="verify the evaluation phase on the fixed development smoke run",
    )
    smoke_evaluation.add_argument("--run-directory", type=Path, required=True)

    evaluation = subparsers.add_parser("evaluate-heldout", help="open GT only after validating persisted inference")
    evaluation.add_argument("--run-directory", type=Path, required=True)
    evaluation.add_argument("--unlock", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = run_preflight(protocol_path=args.protocol, output_path=args.output)
        elif args.command == "smoke-development":
            result = run_inference(
                protocol_path=args.protocol,
                role="development_smoke",
                output_root=args.output_root,
                run_id=args.run_id,
            )
        elif args.command == "infer-heldout":
            result = run_inference(
                protocol_path=args.protocol,
                role="heldout",
                output_root=args.output_root,
                run_id=args.run_id,
                unlock_path=args.unlock,
            )
        elif args.command == "evaluate-development-smoke":
            result = run_development_smoke_evaluation(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
            )
        else:
            result = run_evaluation(
                protocol_path=args.protocol,
                run_directory=args.run_directory,
                unlock_path=args.unlock,
            )
    except (GateDContractError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
