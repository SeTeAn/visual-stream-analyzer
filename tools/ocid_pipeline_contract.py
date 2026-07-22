"""Validate the local assets and data boundaries of the OCID pipeline.

The check path deliberately validates only source-controlled metadata and
model assets.  Stream manifests, RGB frames, evaluation annotations, and OCID
source masks are opened only by the explicitly separated runner phases.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "data" / "ocid" / "benchmark" / "ocid_pipeline_protocol_v1.json"
)
PROTOCOL_SCHEMA = "ocid-pipeline-protocol-1.0"
SOURCE_LABEL_INVENTORY_SCHEMA = "ocid-pipeline-source-label-inventory-1.0"
ACCESS_SCHEMA = "ocid-pipeline-evaluation-access-1.0"
ACCESS_STATUS = "evaluation_access_granted"
ACCESS_DECISION = "approved_for_controlled_evaluation_run"
ACCESS_CONSUMPTION_SCHEMA = "ocid-pipeline-evaluation-access-consumption-1.0"
CHECK_REPORT_SCHEMA = "ocid-pipeline-check-receipt-1.0"
RunRole = Literal["sample", "heldout"]


class OcidPipelineError(ValueError):
    """Raised when the OCID pipeline contract is invalid."""


@dataclass(frozen=True, slots=True)
class OcidStream:
    stream_id: str
    role: Literal["development", "heldout"]
    scene_group_id: str
    scene_group_key: str
    source_sequence: str
    frame_count: int
    stream_directory: Path
    annotation_path: Path


@dataclass(frozen=True, slots=True)
class OcidPipelineProtocol:
    path: Path
    repository_root: Path
    reviewed_root: Path
    sha256: str
    payload: Mapping[str, Any]
    development: tuple[OcidStream, ...]
    heldout: tuple[OcidStream, ...]


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    branch: str
    worktree_clean: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline_protocol(path: Path = DEFAULT_PROTOCOL) -> OcidPipelineProtocol:
    """Validate configured metadata without opening any stream or annotation."""

    return _load_pipeline_protocol(path)


def _load_pipeline_protocol(
    path: Path,
) -> OcidPipelineProtocol:

    protocol_path = Path(path).resolve(strict=True)
    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    _require_under(repository_root, protocol_path, "protocol")
    payload = _json_object(protocol_path, "OCID pipeline protocol")
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise OcidPipelineError(f"unsupported OCID pipeline protocol schema: {payload.get('schema_version')!r}")
    if payload.get("status") != "fixed_configuration":
        raise OcidPipelineError("OCID pipeline protocol must remain fixed_configuration")
    lock = _mapping(payload, "heldout_lock", "OCID pipeline protocol")
    if lock.get("predictions_unlocked") is not False:
        raise OcidPipelineError("source-controlled OCID pipeline held-out lock must remain false")
    if lock.get("access_schema_version") != ACCESS_SCHEMA:
        raise OcidPipelineError("OCID pipeline access schema changed")
    if lock.get("access_decision") != ACCESS_DECISION:
        raise OcidPipelineError("OCID pipeline access decision token changed")
    if lock.get("automatic_retry") is not False or lock.get("overwrite_completed_run") is not False:
        raise OcidPipelineError("OCID pipeline must forbid automatic retry and completed-run overwrite")

    benchmark = _mapping(payload, "benchmark", "OCID pipeline protocol")
    _validate_file_descriptor(repository_root, _mapping(benchmark, "spec", "benchmark"), "benchmark spec")
    reviewed_artifacts = _mapping(benchmark, "reviewed_artifacts", "benchmark")
    for key in ("artifact_manifest", "benchmark_manifest", "event_support_matrix"):
        _validate_file_descriptor(
            repository_root,
            _mapping(reviewed_artifacts, key, "reviewed_artifacts"),
            key,
        )
    reviewed_root = _resolve_relative(
        repository_root,
        _text(benchmark, "reviewed_root", "benchmark"),
        "reviewed root",
        kind="directory",
    )

    _validate_model_assets(repository_root, _mapping(payload, "models", "OCID pipeline protocol"))
    _validate_execution_contract(payload)

    spec_path = _descriptor_path(repository_root, _mapping(benchmark, "spec", "benchmark"))
    spec = _json_object(spec_path, "benchmark spec")
    if spec.get("benchmark_id") != benchmark.get("benchmark_id"):
        raise OcidPipelineError("protocol and benchmark spec IDs differ")
    totals = _mapping(spec, "expected_totals", "benchmark spec")
    split = _mapping(benchmark, "split", "benchmark")
    _validate_split_counts(split, totals)

    benchmark_manifest_path = _descriptor_path(
        repository_root,
        _mapping(reviewed_artifacts, "benchmark_manifest", "reviewed_artifacts"),
    )
    reviewed_manifest = _json_object(benchmark_manifest_path, "reviewed benchmark manifest")
    reviewed_lock = _mapping(reviewed_manifest, "heldout_lock", "reviewed benchmark manifest")
    if reviewed_lock.get("predictions_unlocked") is not False:
        raise OcidPipelineError("reviewed benchmark held-out lock must remain false")
    development, heldout = _metadata_inventory(spec, reviewed_manifest, reviewed_root)
    _require_inventory_totals(development, heldout, totals)
    return OcidPipelineProtocol(
        path=protocol_path,
        repository_root=repository_root,
        reviewed_root=reviewed_root,
        sha256=sha256_file(protocol_path),
        payload=payload,
        development=development,
        heldout=heldout,
    )


def inventory_for_role(
    protocol: OcidPipelineProtocol,
    role: RunRole,
) -> tuple[OcidStream, ...]:
    if role == "heldout":
        return protocol.heldout
    if role == "sample":
        execution = _mapping(protocol.payload, "execution", "OCID pipeline protocol")
        stream_id = _text(execution, "sample_stream_id", "execution")
        selected = tuple(item for item in protocol.development if item.stream_id == stream_id)
        if len(selected) != 1:
            raise OcidPipelineError("sample stream is absent or ambiguous")
        return selected
    raise OcidPipelineError(f"unsupported OCID pipeline role: {role!r}")


def git_state(repository_root: Path = REPOSITORY_ROOT) -> GitState:
    root = Path(repository_root).resolve(strict=True)
    safe = root.as_posix()
    commit = _git(root, "rev-parse", "HEAD", safe=safe)
    branch = _git(root, "branch", "--show-current", safe=safe)
    status = _git(root, "status", "--porcelain", safe=safe)
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.casefold()):
        raise OcidPipelineError("Git returned an invalid commit digest")
    if not branch:
        raise OcidPipelineError("OCID pipeline requires a named Git branch")
    return GitState(commit.casefold(), branch, not bool(status.strip()))


def build_check_report(protocol: OcidPipelineProtocol) -> dict[str, Any]:
    """Build a metadata-only receipt; callers decide where to persist it."""

    state = git_state(protocol.repository_root)
    models = _mapping(protocol.payload, "models", "OCID pipeline protocol")
    benchmark = _mapping(protocol.payload, "benchmark", "OCID pipeline protocol")
    return {
        "schema_version": CHECK_REPORT_SCHEMA,
        "status": "ready" if state.worktree_clean else "working_tree_has_changes",
        "protocol": {
            "path": protocol.path.relative_to(protocol.repository_root).as_posix(),
            "sha256": protocol.sha256,
            "protocol_id": protocol.payload["protocol_id"],
        },
        "git": {
            "commit": state.commit,
            "branch": state.branch,
            "worktree_clean": state.worktree_clean,
        },
        "benchmark": {
            "benchmark_id": benchmark["benchmark_id"],
            "development_stream_count": len(protocol.development),
            "development_frame_count": sum(item.frame_count for item in protocol.development),
            "heldout_stream_count": len(protocol.heldout),
            "heldout_frame_count": sum(item.frame_count for item in protocol.heldout),
            "heldout_stream_ids": [item.stream_id for item in protocol.heldout],
            "heldout_rgb_opened": False,
            "heldout_annotation_opened": False,
        },
        "models": {
            "candidate_extractor": models["candidate_extractor"]["family"],
            "mask_refiner": models["mask_refiner"]["family"],
            "representation": models["representation"]["model_name"],
            "assets_verified": True,
        },
        "heldout_lock": {
            "predictions_unlocked": False,
            "explicit_confirmation_required": True,
            "automatic_retry": False,
        },
    }


def validate_evaluation_access(
    path: Path,
    protocol: OcidPipelineProtocol,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    access_path = Path(path).resolve(strict=True)
    try:
        access_path.relative_to(protocol.repository_root.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise OcidPipelineError("evaluation access must remain outside the Git repository")
    payload = _json_object(access_path, "OCID pipeline evaluation access")
    if payload.get("schema_version") != ACCESS_SCHEMA:
        raise OcidPipelineError("unsupported OCID pipeline evaluation access schema")
    if payload.get("status") != ACCESS_STATUS or payload.get("decision") != ACCESS_DECISION:
        raise OcidPipelineError("OCID pipeline evaluation access decision is absent")
    if payload.get("protocol_sha256") != protocol.sha256:
        raise OcidPipelineError("evaluation access is bound to a different protocol digest")
    run_id = _text(payload, "run_id", "evaluation access")
    _portable_segment(run_id, "run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise OcidPipelineError("evaluation access run_id does not match the requested run")
    expected_streams = [item.stream_id for item in protocol.heldout]
    if payload.get("heldout_stream_ids") != expected_streams:
        raise OcidPipelineError("evaluation access must name the exact configured stream inventory")
    state = git_state(protocol.repository_root)
    if payload.get("execution_git_commit") != state.commit:
        raise OcidPipelineError("evaluation access is bound to a different execution Git commit")
    if payload.get("execution_git_branch") != state.branch:
        raise OcidPipelineError("evaluation access is bound to a different Git branch")
    if not state.worktree_clean:
        raise OcidPipelineError("held-out execution requires a clean Git worktree")
    if not isinstance(payload.get("authorized_at_utc"), str) or not payload["authorized_at_utc"].endswith("Z"):
        raise OcidPipelineError("evaluation access requires an authorized_at_utc timestamp ending in Z")
    if payload.get("single_use") is not True or payload.get("automatic_retry") is not False:
        raise OcidPipelineError("evaluation access must be single-use and forbid automatic retry")
    return payload


def claim_evaluation_access(
    path: Path,
    protocol: OcidPipelineProtocol,
    *,
    expected_run_id: str,
    run_directory: Path,
) -> dict[str, Any]:
    """Consume one evaluation access before any held-out RGB is opened."""

    access_path = Path(path).resolve(strict=True)
    access = validate_evaluation_access(
        access_path,
        protocol,
        expected_run_id=expected_run_id,
    )
    consumption_path = access_path.with_name(f"{access_path.name}.consumed.json")
    receipt = {
        "schema_version": ACCESS_CONSUMPTION_SCHEMA,
        "status": "consumed_before_rgb_access",
        "access_sha256": sha256_file(access_path),
        "protocol_sha256": protocol.sha256,
        "execution_git_commit": access["execution_git_commit"],
        "execution_git_branch": access["execution_git_branch"],
        "run_id": access["run_id"],
        "heldout_stream_ids": access["heldout_stream_ids"],
        "run_directory": Path(run_directory).resolve(strict=False).as_posix(),
        "consumed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "single_use": True,
        "automatic_retry": False,
    }
    write_json_exclusive(consumption_path, receipt)
    return {
        "access": access,
        "consumption_path": consumption_path,
        "consumption_receipt": receipt,
    }


def validate_evaluation_access_consumption(
    path: Path,
    protocol: OcidPipelineProtocol,
    *,
    expected_run_id: str,
    expected_run_directory: Path,
) -> dict[str, Any]:
    access_path = Path(path).resolve(strict=True)
    access = validate_evaluation_access(
        access_path,
        protocol,
        expected_run_id=expected_run_id,
    )
    consumption_path = access_path.with_name(f"{access_path.name}.consumed.json")
    receipt = _json_object(consumption_path, "evaluation access consumption receipt")
    if (
        receipt.get("schema_version") != ACCESS_CONSUMPTION_SCHEMA
        or receipt.get("status") != "consumed_before_rgb_access"
        or receipt.get("access_sha256") != sha256_file(access_path)
        or receipt.get("protocol_sha256") != protocol.sha256
        or receipt.get("execution_git_commit") != access["execution_git_commit"]
        or receipt.get("execution_git_branch") != access["execution_git_branch"]
        or receipt.get("run_id") != expected_run_id
        or receipt.get("heldout_stream_ids") != access["heldout_stream_ids"]
        or receipt.get("run_directory")
        != Path(expected_run_directory).resolve(strict=True).as_posix()
        or receipt.get("single_use") is not True
        or receipt.get("automatic_retry") is not False
    ):
        raise OcidPipelineError("evaluation access consumption receipt differs from the held-out run")
    consumed_at = receipt.get("consumed_at_utc")
    if not isinstance(consumed_at, str) or not consumed_at.endswith("Z"):
        raise OcidPipelineError("evaluation access consumption receipt lacks a UTC timestamp")
    return {
        "access": access,
        "consumption_path": consumption_path,
        "consumption_receipt": receipt,
    }


def validate_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise OcidPipelineError("run_id must be one portable path segment")
    _portable_segment(value, "run_id")
    return value


def validate_analysis_inputs(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
) -> dict[str, Any]:
    """Hash only analysis manifests and RGB files for the selected inventory.

    This function crosses the RGB boundary and must therefore be called only
    after a valid evaluation access for held-out.  It never hashes annotations.
    """

    benchmark = _mapping(protocol.payload, "benchmark", "OCID pipeline protocol")
    artifacts = _mapping(benchmark, "reviewed_artifacts", "benchmark")
    manifest_path = _descriptor_path(
        protocol.repository_root,
        _mapping(artifacts, "artifact_manifest", "reviewed_artifacts"),
    )
    artifact_manifest = _json_object(manifest_path, "reviewed artifact manifest")
    rows = artifact_manifest.get("artifacts")
    if not isinstance(rows, list):
        raise OcidPipelineError("reviewed artifact manifest lacks artifacts")
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise OcidPipelineError("malformed reviewed artifact entry")
        if row["path"] in index:
            raise OcidPipelineError("duplicate reviewed artifact path")
        index[row["path"]] = row

    verified: list[dict[str, Any]] = []
    for item in inventory:
        prefix = f"analysis_streams/{item.stream_id}/"
        selected = sorted(path for path in index if path.startswith(prefix))
        if not selected or any("evaluation_annotations/" in path for path in selected):
            raise OcidPipelineError(f"invalid analysis artifact inventory for {item.stream_id}")
        for relative in selected:
            descriptor = index[relative]
            full = _resolve_relative(protocol.reviewed_root, relative, "analysis artifact", kind="file")
            _validate_expected_file(full, descriptor, f"analysis artifact {relative}")
            verified.append(
                {
                    "path": relative,
                    "size_bytes": int(descriptor["size_bytes"]),
                    "sha256": str(descriptor["sha256"]),
                }
            )
    expected_frames = sum(item.frame_count for item in inventory)
    rgb_count = sum(1 for row in verified if "/frames/" in row["path"])
    manifest_count = sum(1 for row in verified if row["path"].endswith("/manifest.json"))
    if rgb_count != expected_frames or manifest_count != len(inventory):
        raise OcidPipelineError(
            f"analysis artifact counts differ: rgb={rgb_count}/{expected_frames}, "
            f"manifests={manifest_count}/{len(inventory)}"
        )
    return {
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "verified_artifact_count": len(verified),
        "verified_rgb_count": rgb_count,
        "verified_manifest_count": manifest_count,
        "artifacts": verified,
        "ground_truth_opened": False,
    }


def validate_evaluation_inputs(
    protocol: OcidPipelineProtocol,
    inventory: Sequence[OcidStream],
) -> dict[str, Any]:
    """Hash configured evaluation annotations after the inference boundary."""

    benchmark = _mapping(protocol.payload, "benchmark", "OCID pipeline protocol")
    artifacts = _mapping(benchmark, "reviewed_artifacts", "benchmark")
    manifest_path = _descriptor_path(
        protocol.repository_root,
        _mapping(artifacts, "artifact_manifest", "reviewed_artifacts"),
    )
    artifact_manifest = _json_object(manifest_path, "reviewed artifact manifest")
    rows = artifact_manifest.get("artifacts")
    if not isinstance(rows, list):
        raise OcidPipelineError("reviewed artifact manifest lacks artifacts")
    index = {
        str(row["path"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    verified: list[dict[str, Any]] = []
    for item in inventory:
        relative = f"evaluation_annotations/{item.stream_id}/annotation.json"
        descriptor = index.get(relative)
        if descriptor is None:
            raise OcidPipelineError(f"missing configured annotation artifact: {relative}")
        full = _resolve_relative(protocol.reviewed_root, relative, "evaluation annotation", kind="file")
        _validate_expected_file(full, descriptor, f"evaluation annotation {relative}")
        verified.append(
            {
                "path": relative,
                "size_bytes": int(descriptor["size_bytes"]),
                "sha256": str(descriptor["sha256"]),
            }
        )
    label_inventory_path = _resolve_relative(
        protocol.repository_root,
        _text(benchmark, "source_label_inventory_path", "benchmark"),
        "source label inventory",
        kind="file",
    )
    label_inventory = _json_object(label_inventory_path, "source label inventory")
    _validate_source_label_inventory_metadata(
        benchmark,
        label_inventory,
        protocol.development,
        protocol.heldout,
    )
    label_streams = label_inventory.get("streams")
    if not isinstance(label_streams, list):
        raise OcidPipelineError("source label inventory lacks streams")
    label_index = {
        str(row["stream_id"]): row
        for row in label_streams
        if isinstance(row, Mapping) and isinstance(row.get("stream_id"), str)
    }
    source_root = _resolve_relative(
        protocol.repository_root,
        _text(label_inventory, "source_root", "source label inventory"),
        "OCID source root",
        kind="directory",
    )
    verified_labels: list[dict[str, Any]] = []
    for item in inventory:
        stream = label_index.get(item.stream_id)
        if not isinstance(stream, Mapping):
            raise OcidPipelineError(f"source label inventory lacks {item.stream_id}")
        labels = stream.get("labels")
        if not isinstance(labels, list) or len(labels) != item.frame_count:
            raise OcidPipelineError(f"source label count differs for {item.stream_id}")
        for descriptor in labels:
            if not isinstance(descriptor, Mapping):
                raise OcidPipelineError(f"malformed source label for {item.stream_id}")
            relative = _text(descriptor, "path", "source label")
            full = _resolve_relative(source_root, relative, "source label", kind="file")
            _validate_expected_file(full, descriptor, f"source label {relative}")
            verified_labels.append(
                {
                    "stream_id": item.stream_id,
                    "frame_id": _text(descriptor, "frame_id", "source label"),
                    "path": relative,
                    "size_bytes": int(descriptor["size_bytes"]),
                    "sha256": str(descriptor["sha256"]),
                }
            )
    return {
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "verified_annotation_count": len(verified),
        "annotations": verified,
        "source_label_inventory_sha256": sha256_file(label_inventory_path),
        "verified_source_label_count": len(verified_labels),
        "source_labels": verified_labels,
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise OcidPipelineError(f"refusing to overwrite existing artifact: {destination}") from error
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, payload)
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_execution_contract(payload: Mapping[str, Any]) -> None:
    execution = _mapping(payload, "execution", "OCID pipeline protocol")
    if execution.get("device") != "cuda" or execution.get("dtype") != "float32":
        raise OcidPipelineError("OCID pipeline execution must remain CUDA float32")
    if execution.get("network_access") != "forbidden":
        raise OcidPipelineError("OCID pipeline model loading must remain offline")
    phases = execution.get("phases")
    if phases != [
        "check",
        "inference_before_ground_truth",
        "evaluation_after_persisted_inference",
    ]:
        raise OcidPipelineError("OCID pipeline phase order changed")
    pipeline = _mapping(payload, "pipeline", "OCID pipeline protocol")
    refinement = _mapping(pipeline, "mask_refinement", "pipeline")
    if refinement.get("fallback_policy") != "fail_run_if_any_candidate_has_no_valid_mask":
        raise OcidPipelineError("OCID pipeline must not hide a mask fallback")
    representation = _mapping(pipeline, "representation", "pipeline")
    if representation.get("variant") != "mask_neutral_letterbox_v1":
        raise OcidPipelineError("OCID pipeline representation variant changed")


def _validate_model_assets(repository_root: Path, models: Mapping[str, Any]) -> None:
    for key in ("candidate_extractor", "mask_refiner"):
        model = _mapping(models, key, "models")
        directory = _resolve_relative(
            repository_root,
            _text(model, "directory", key),
            f"{key} directory",
            kind="directory",
        )
        assets = _mapping(model, "assets", key)
        if not assets:
            raise OcidPipelineError(f"{key} assets must not be empty")
        for filename, descriptor in assets.items():
            if not isinstance(filename, str) or not isinstance(descriptor, Mapping):
                raise OcidPipelineError(f"malformed {key} asset descriptor")
            _validate_expected_file(
                _resolve_relative(directory, filename, f"{key} asset", kind="file"),
                descriptor,
                f"{key} asset {filename}",
            )

    representation = _mapping(models, "representation", "models")
    source_directory = _resolve_relative(
        repository_root,
        _text(representation, "source_directory", "representation"),
        "DINOv2 source directory",
        kind="directory",
    )
    expected_source = _sha_text(
        representation,
        "source_tree_fingerprint_sha256",
        "DINOv2 representation",
    )
    from stream_analysis.representations import source_tree_fingerprint

    actual_source = source_tree_fingerprint(source_directory)
    if actual_source != expected_source:
        raise OcidPipelineError(
            f"DINOv2 source fingerprint mismatch: expected {expected_source}, got {actual_source}"
        )
    checkpoint = _mapping(representation, "checkpoint", "representation")
    _validate_file_descriptor(repository_root, checkpoint, "DINOv2 checkpoint")


def _validate_file_descriptor(root: Path, descriptor: Mapping[str, Any], label: str) -> None:
    path = _descriptor_path(root, descriptor)
    _validate_expected_file(path, descriptor, label)


def _validate_expected_file(path: Path, descriptor: Mapping[str, Any], label: str) -> None:
    expected_size = descriptor.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise OcidPipelineError(f"{label} size_bytes must be a positive integer")
    expected_sha = _sha_text(descriptor, "sha256", label)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise OcidPipelineError(f"{label} size mismatch: expected {expected_size}, got {actual_size}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise OcidPipelineError(f"{label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")


def _descriptor_path(root: Path, descriptor: Mapping[str, Any]) -> Path:
    return _resolve_relative(root, _text(descriptor, "path", "file descriptor"), "artifact", kind="file")


def _metadata_inventory(
    spec: Mapping[str, Any],
    reviewed_manifest: Mapping[str, Any],
    reviewed_root: Path,
) -> tuple[tuple[OcidStream, ...], tuple[OcidStream, ...]]:
    roles = _mapping(spec, "roles", "benchmark spec")
    expected_by_stream: dict[str, tuple[str, str, str, int]] = {}
    for role in ("development", "heldout"):
        groups = roles.get(role)
        if not isinstance(groups, list):
            raise OcidPipelineError(f"benchmark role {role} must be a list")
        for group in groups:
            if not isinstance(group, Mapping):
                raise OcidPipelineError(f"malformed {role} scene group")
            scene_group_id = _text(group, "scene_group_id", f"{role} scene group")
            scene_group_key = _text(group, "scene_group_key", f"{role} scene group")
            frame_count = group.get("frames_per_stream")
            members = group.get("member_streams")
            if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
                raise OcidPipelineError(f"invalid frame count in {role} scene group")
            if not isinstance(members, list) or not members:
                raise OcidPipelineError(f"invalid stream list in {role} scene group")
            for source in members:
                if not isinstance(source, str):
                    raise OcidPipelineError(f"invalid source stream in {role} scene group")
                stream_id = "ocid_" + "_".join(part.casefold() for part in source.split("/"))
                if stream_id in expected_by_stream:
                    raise OcidPipelineError(f"duplicate benchmark stream: {stream_id}")
                expected_by_stream[stream_id] = (
                    role,
                    scene_group_id,
                    scene_group_key,
                    frame_count,
                )

    rows = reviewed_manifest.get("streams")
    if not isinstance(rows, list):
        raise OcidPipelineError("reviewed benchmark manifest streams must be a list")
    result: dict[str, list[OcidStream]] = {"development": [], "heldout": []}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise OcidPipelineError("malformed reviewed benchmark stream row")
        stream_id = _text(row, "stream_id", "reviewed benchmark stream")
        if stream_id in seen or stream_id not in expected_by_stream:
            raise OcidPipelineError(f"unexpected or duplicate reviewed stream: {stream_id}")
        seen.add(stream_id)
        role, scene_group_id, scene_group_key, frame_count = expected_by_stream[stream_id]
        if (
            row.get("role") != role
            or row.get("scene_group_id") != scene_group_id
            or row.get("scene_group_key") != scene_group_key
            or row.get("frame_count") != frame_count
        ):
            raise OcidPipelineError(f"reviewed stream metadata differs for {stream_id}")
        analysis_manifest = _text(row, "analysis_manifest", "reviewed benchmark stream")
        evaluation_annotation = _text(row, "evaluation_annotation", "reviewed benchmark stream")
        expected_analysis = f"analysis_streams/{stream_id}/manifest.json"
        expected_annotation = f"evaluation_annotations/{stream_id}/annotation.json"
        if analysis_manifest != expected_analysis or evaluation_annotation != expected_annotation:
            raise OcidPipelineError(f"non-canonical reviewed paths for {stream_id}")
        source_sequence = _text(row, "source_sequence", "reviewed benchmark stream")
        stream_directory = (reviewed_root / "analysis_streams" / stream_id).resolve(strict=False)
        annotation_path = (reviewed_root / "evaluation_annotations" / stream_id / "annotation.json").resolve(
            strict=False
        )
        _require_under(reviewed_root, stream_directory, "analysis stream")
        _require_under(reviewed_root, annotation_path, "evaluation annotation")
        result[role].append(
            OcidStream(
                stream_id=stream_id,
                role=role,  # type: ignore[arg-type]
                scene_group_id=scene_group_id,
                scene_group_key=scene_group_key,
                source_sequence=source_sequence,
                frame_count=frame_count,
                stream_directory=stream_directory,
                annotation_path=annotation_path,
            )
        )
    if seen != set(expected_by_stream):
        raise OcidPipelineError("benchmark manifest does not cover the configured inventory")
    return (
        tuple(sorted(result["development"], key=lambda item: item.stream_id)),
        tuple(sorted(result["heldout"], key=lambda item: item.stream_id)),
    )


def _validate_split_counts(split: Mapping[str, Any], totals: Mapping[str, Any]) -> None:
    expected = {
        "development": {
            "scene_groups": totals.get("development_scene_groups"),
            "streams": totals.get("development_streams"),
            "frames": totals.get("development_frames"),
        },
        "heldout": {
            "scene_groups": totals.get("heldout_scene_groups"),
            "streams": totals.get("heldout_streams"),
            "frames": totals.get("heldout_frames"),
        },
    }
    for role, wanted in expected.items():
        actual = _mapping(split, role, "benchmark split")
        if dict(actual) != wanted:
            raise OcidPipelineError(f"protocol {role} split counts differ from benchmark spec")


def _require_inventory_totals(
    development: Sequence[OcidStream],
    heldout: Sequence[OcidStream],
    totals: Mapping[str, Any],
) -> None:
    actual = {
        "development_streams": len(development),
        "development_frames": sum(item.frame_count for item in development),
        "development_scene_groups": len({item.scene_group_id for item in development}),
        "heldout_streams": len(heldout),
        "heldout_frames": sum(item.frame_count for item in heldout),
        "heldout_scene_groups": len({item.scene_group_id for item in heldout}),
    }
    for key, value in actual.items():
        if totals.get(key) != value:
            raise OcidPipelineError(f"configured inventory total differs for {key}: {value}")


def _validate_source_label_inventory_metadata(
    benchmark: Mapping[str, Any],
    payload: Mapping[str, Any],
    development: Sequence[OcidStream],
    heldout: Sequence[OcidStream],
) -> None:
    if payload.get("schema_version") != SOURCE_LABEL_INVENTORY_SCHEMA:
        raise OcidPipelineError("unsupported source label inventory schema")
    if payload.get("benchmark_id") != benchmark.get("benchmark_id"):
        raise OcidPipelineError("source label inventory benchmark differs")
    if payload.get("benchmark_spec_sha256") != _mapping(benchmark, "spec", "benchmark").get(
        "sha256"
    ):
        raise OcidPipelineError("source label inventory benchmark digest differs")
    if payload.get("source_root") != "data/ocid/raw/OCID-dataset":
        raise OcidPipelineError("source label inventory root changed")
    expected = {item.stream_id: item for item in (*development, *heldout)}
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != len(expected):
        raise OcidPipelineError("source label inventory stream count differs")
    seen: set[str] = set()
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise OcidPipelineError("malformed source label stream")
        stream_id = _text(stream, "stream_id", "source label stream")
        item = expected.get(stream_id)
        if item is None or stream_id in seen:
            raise OcidPipelineError(f"unexpected source label stream: {stream_id}")
        seen.add(stream_id)
        labels = stream.get("labels")
        if (
            stream.get("role") != item.role
            or stream.get("source_sequence") != item.source_sequence
            or stream.get("frame_count") != item.frame_count
            or not isinstance(labels, list)
            or len(labels) != item.frame_count
        ):
            raise OcidPipelineError(f"source label stream metadata differs: {stream_id}")
        frame_ids: set[str] = set()
        frame_indices: list[int] = []
        for label in labels:
            if not isinstance(label, Mapping):
                raise OcidPipelineError(f"malformed source label row: {stream_id}")
            frame_id = _text(label, "frame_id", "source label")
            source_filename = _text(label, "source_filename", "source label")
            frame_index = label.get("frame_index")
            if (
                frame_id in frame_ids
                or isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise OcidPipelineError(f"invalid source label frame identity: {stream_id}")
            frame_ids.add(frame_id)
            frame_indices.append(frame_index)
            expected_path = (
                PurePosixPath(item.source_sequence) / "label" / source_filename
            ).as_posix()
            if label.get("path") != expected_path:
                raise OcidPipelineError(f"non-canonical source label path: {stream_id}")
            size = label.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise OcidPipelineError(f"invalid source label size: {stream_id}")
            _sha_text(label, "sha256", "source label")
        if frame_indices != sorted(frame_indices) or len(set(frame_indices)) != len(frame_indices):
            raise OcidPipelineError(f"source label frame order differs: {stream_id}")
    totals = payload.get("totals")
    expected_totals = {
        "development_streams": len(development),
        "development_label_files": sum(item.frame_count for item in development),
        "heldout_streams": len(heldout),
        "heldout_label_files": sum(item.frame_count for item in heldout),
        "label_files": sum(item.frame_count for item in (*development, *heldout)),
    }
    if not isinstance(totals, Mapping) or dict(totals) != expected_totals:
        raise OcidPipelineError("source label inventory totals differ")


def _resolve_relative(
    root: Path,
    value: str,
    label: str,
    *,
    kind: Literal["file", "directory"],
) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise OcidPipelineError(f"{label} must be a safe relative path")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    _require_under(root, path, label)
    if kind == "file" and not path.is_file():
        raise OcidPipelineError(f"{label} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise OcidPipelineError(f"{label} is not a directory: {path}")
    return path


def _require_under(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise OcidPipelineError(f"{label} escapes the repository boundary") from error


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcidPipelineError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise OcidPipelineError(f"{label} must be a JSON object")
    return payload


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


def _sha_text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = _text(value, key, label).casefold()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise OcidPipelineError(f"{label}.{key} must be a SHA-256 digest")
    return result


def _git(root: Path, *arguments: str, safe: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={safe}", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OcidPipelineError(f"cannot inspect Git state: {error}") from error
    return completed.stdout.strip()


def _portable_segment(value: str, label: str) -> None:
    forbidden = frozenset('<>:"/\\|?*')
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (
        not value
        or any(char in forbidden for char in value)
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in reserved
        or len(value.encode("utf-8")) > 255
    ):
        raise OcidPipelineError(f"{label} must be one portable path segment")


__all__ = [
    "DEFAULT_PROTOCOL",
    "CHECK_REPORT_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SOURCE_LABEL_INVENTORY_SCHEMA",
    "REPOSITORY_ROOT",
    "ACCESS_DECISION",
    "ACCESS_CONSUMPTION_SCHEMA",
    "ACCESS_SCHEMA",
    "ACCESS_STATUS",
    "OcidPipelineProtocol",
    "OcidPipelineError",
    "OcidStream",
    "GitState",
    "git_state",
    "claim_evaluation_access",
    "inventory_for_role",
    "load_pipeline_protocol",
    "build_check_report",
    "sha256_file",
    "validate_analysis_inputs",
    "validate_evaluation_access",
    "validate_evaluation_access_consumption",
    "validate_evaluation_inputs",
    "validate_run_id",
    "write_json_atomic",
    "write_json_exclusive",
]
