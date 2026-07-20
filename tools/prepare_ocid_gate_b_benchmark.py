"""Build the frozen Gate B OCID candidate-extraction benchmark.

The builder verifies the approved selection spec, the Gate A structural audit,
and the audited OCID source bytes before publishing a benchmark.  RGB inputs
for ``analyze`` are isolated under ``analysis_streams``; every artifact derived
from OCID instance masks is written elsewhere and is rejected by a recursive
ground-truth firewall if it leaks into an analysis stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))

from stream_analysis.evaluation.component_review import (
    DEFAULT_SECONDARY_AREA_RATIO,
    POLICY_ID as COMPONENT_REVIEW_POLICY_ID,
    ComponentAnalysis,
    ComponentReviewDecision,
    analyze_component_mask,
    finalize_component_review,
)
from stream_analysis.evaluation.review_contract import canonical_reviewer_id


SPEC_SCHEMA_VERSION = "ocid-candidate-benchmark-spec-1.0"
AUDIT_SCHEMA_ID = "ocid_structural_audit"
AUDIT_SCHEMA_VERSION = "1.0.0"
ANALYSIS_MANIFEST_SCHEMA_VERSION = "stream-input-0.1"
ANNOTATION_SCHEMA_VERSION = "stream-pilot-annotation-0.1"
ANNOTATION_SCOPE = "ocid_candidate_extraction_bbox_from_instance_masks"
COMPONENT_DECISIONS_SCHEMA_VERSION = "ocid-component-review-decisions-1.0"
COMPONENT_BBOX_DERIVATION = (
    "tight_axis_aligned_bbox_from_reviewed_instance_mask_components"
)
COMPONENT_REVIEW_CANDIDATE_RULE = (
    "raw_bbox_differs_from_largest_component_bbox"
)
FINAL_COMPONENT_DECISIONS = frozenset({"KEEP", "DROP"})
FINAL_COMPONENT_CONFIDENCES = frozenset({"HIGH", "MEDIUM"})
PRIMARY_COMPONENT_DECISIONS = frozenset(
    {"KEEP", "DROP", "UNRESOLVED", "INTEGRITY_ALERT"}
)
CASE_INTEGRITY_DECISIONS = frozenset({"OK", "INTEGRITY_ALERT"})
FINAL_CASE_INTEGRITY_RESOLUTION = "OK"
FINAL_FORBIDDEN_OUTCOMES = frozenset({"UNRESOLVED", "INTEGRITY_ALERT"})
RATIONALE_CODES_BY_DECISION = {
    "KEEP": frozenset({"K_TEMPORAL", "K_OCCLUSION", "K_CONTINUITY"}),
    "DROP": frozenset(
        {
            "D_OTHER_REGION",
            "D_TEMPORAL_OTHER",
            "D_TRANSIENT_ARTIFACT",
            "D_BOUNDARY_BLEED",
        }
    ),
    "UNRESOLVED": frozenset(
        {"A_OCCLUDED", "A_NO_TEMPORAL", "A_CONFLICT", "A_IMAGE_QUALITY"}
    ),
    "INTEGRITY_ALERT": frozenset(
        {"I_C001_INVALID", "I_LABEL_SWITCH", "I_MULTIPLE_OBJECTS"}
    ),
}
SOURCE_FINGERPRINT_METHOD = (
    "sha256 over lexicographically sorted UTF-8 records: "
    "relative_posix_path\\0size_bytes\\0sha256(content)\\n"
)
ROLE_ORDER = ("development", "heldout")
HELDOUT_ALLOWED_BEFORE_UNLOCK = (
    "structural_audit",
    "annotation_generation",
    "ground_truth_preview",
    "annotation_correction",
    "artifact_freeze",
)
HELDOUT_FORBIDDEN_BEFORE_UNLOCK = (
    "model_prediction",
    "prediction_overlay",
    "metric_computation",
    "model_selection",
    "threshold_selection",
    "postprocessing_selection",
)
AUTOMATIC_FLAGS = (
    "relative_visible_area_below_0_10",
    "connected_component_count_above_1",
    "border_touch",
    "visible_to_absent",
    "returned_visible",
    "severe_interframe_visibility_drop",
)
FIREWALL_MARKERS = (
    "annotation",
    "ground_truth",
    "groundtruth",
    "label",
    "mask",
    "visual_type",
    "visualtype",
    "metric",
    "prediction",
)
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

OBSERVATION_COLUMNS = (
    "role",
    "scene_group_id",
    "stream_id",
    "source_sequence",
    "frame_id",
    "frame_index",
    "source_filename",
    "physical_instance_id",
    "source_label",
    "visible_pixels",
    "frame_area_pixels",
    "visible_fraction_of_frame",
    "stream_instance_max_visible_pixels",
    "relative_to_stream_instance_max_area",
    "raw_bbox_x",
    "raw_bbox_y",
    "raw_bbox_width",
    "raw_bbox_height",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "connected_component_count",
    "border_touch",
    "source_mask_sha256",
    "component_review_required",
    "automatic_retained_component_ids",
    "final_retained_component_ids",
    "removed_component_ids",
    "final_component_decisions",
    "final_confidence",
    "resolved_by",
    "author_attention",
    "automatic_flags",
    "outgoing_transition_flags",
    "diagnostic_only",
    "retained_in_annotation",
)
FRAME_REVIEW_COLUMNS = (
    "role",
    "scene_group_id",
    "stream_id",
    "source_sequence",
    "frame_id",
    "frame_index",
    "source_filename",
    "visible_instance_count",
    "visible_instance_ids",
    "automatic_flags",
    "automatic_flag_details",
    "mandatory_review_reasons",
    "review_required",
    "author_review_status",
    "author_review_decision",
    "author_review_notes",
    "author_reviewed_by",
)
TIMELINE_COLUMNS = (
    "role",
    "scene_group_id",
    "stream_id",
    "source_sequence",
    "frame_id",
    "frame_index",
    "source_filename",
    "visible_instance_count",
    "visible_instance_ids",
    "first_visible_or_added_ids",
    "visible_to_absent_ids",
    "returned_visible_ids",
    "severe_interframe_visibility_drop_ids",
    "count_delta",
    "automatic_flags",
)


class GroundTruthFirewallError(ValueError):
    """Raised when an analysis input contains a ground-truth leak marker."""


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_root: Path
    scene_group_count: int
    stream_count: int
    frame_count: int
    observation_count: int
    review_queue_count: int


@dataclass(frozen=True, slots=True)
class StreamSelection:
    role: str
    scene_group_id: str
    scene_group_key: str
    reason: str
    frames_per_stream: int
    source_sequence: str


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourcePair:
    filename: str
    rgb_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class Observation:
    physical_instance_id: str
    source_label: int
    visible_pixels: int
    visible_fraction: float
    raw_bbox: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    connected_component_count: int
    border_touch: bool
    source_mask_sha256: str
    component_review_required: bool
    automatic_retained_component_ids: tuple[str, ...]
    retained_component_ids: tuple[str, ...]
    removed_component_ids: tuple[str, ...]
    final_component_decisions: tuple[dict[str, object], ...]
    final_confidence: str
    resolved_by: str
    author_attention: bool
    retained_mask: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FrameData:
    frame_id: str
    frame_index: int
    source_filename: str
    width: int
    height: int
    rgb_path: Path
    labels: np.ndarray
    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class StreamReview:
    observation_rows: tuple[dict[str, object], ...]
    frame_rows: tuple[dict[str, object], ...]
    timeline_rows: tuple[dict[str, object], ...]
    queue_items: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ComponentDecisionLedger:
    path: Path
    sha256: str
    raw_bytes: bytes
    decisions_by_case_id: dict[str, dict[str, Any]]


def _load_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}.")
    return value, data


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace.")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array.")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _digest(value: object, name: str) -> str:
    digest = _text(value, name).casefold()
    if not SHA256_HEX.fullmatch(digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _canonical_reviewer_id(value: object, name: str) -> str:
    return canonical_reviewer_id(value, name)


def _reviewer_id(review: dict[str, Any], context: str) -> str:
    present = [name for name in ("reviewer_id", "reviewer") if name in review]
    if not present:
        raise ValueError(f"{context} must identify reviewer_id.")
    values = [
        _canonical_reviewer_id(review[name], f"{context}.{name}") for name in present
    ]
    if len(set(values)) != 1:
        raise ValueError(f"{context} contains contradictory reviewer identities.")
    return values[0]


def _relative_posix(value: object, name: str) -> PurePosixPath:
    raw = _text(value, name)
    posix = PurePosixPath(raw.replace("\\", "/"))
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or not posix.parts
    ):
        raise ValueError(f"{name} must remain inside the OCID root.")
    if posix.as_posix() != raw.replace("\\", "/"):
        raise ValueError(f"{name} must be a normalized relative path.")
    return posix


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _stream_id(source_sequence: str) -> str:
    path = PurePosixPath(source_sequence)
    value = "ocid_" + "_".join(part.casefold().replace("-", "_") for part in path.parts)
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"OCID sequence produces an unsafe stream_id: {source_sequence!r}.")
    return value


def _paired_scene_key(source_sequence: str) -> str:
    parts = list(PurePosixPath(source_sequence).parts)
    camera_positions = [index for index, part in enumerate(parts) if part in {"top", "bottom"}]
    if len(camera_positions) != 1:
        raise ValueError(
            f"OCID stream must contain exactly one top/bottom camera component: "
            f"{source_sequence!r}."
        )
    del parts[camera_positions[0]]
    return PurePosixPath(*parts).as_posix()


def _camera(source_sequence: str) -> str:
    cameras = [part for part in PurePosixPath(source_sequence).parts if part in {"top", "bottom"}]
    if len(cameras) != 1:
        raise ValueError(f"Cannot resolve OCID camera for {source_sequence!r}.")
    return cameras[0]


def _validate_spec(
    spec: dict[str, Any], *, require_component_decisions: bool = True
) -> tuple[list[StreamSelection], dict[str, int]]:
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Gate B spec schema_version: {spec.get('schema_version')!r}.")
    _text(spec.get("benchmark_id"), "spec.benchmark_id")
    if spec.get("status") != "approved_gate_a_ocid_scope":
        raise ValueError("Gate B spec must have approved_gate_a_ocid_scope status.")

    source = _mapping(spec.get("source"), "spec.source")
    _text(source.get("dataset"), "spec.source.dataset")
    _text(source.get("source_url"), "spec.source.source_url")
    _digest(source.get("source_fingerprint_sha256"), "spec.source.source_fingerprint_sha256")
    _digest(source.get("structural_audit_sha256"), "spec.source.structural_audit_sha256")
    component_status = source.get("component_decisions_status")
    if require_component_decisions:
        if component_status != "frozen_complete":
            raise ValueError(
                "spec.source.component_decisions_status must be 'frozen_complete' "
                "for a final Gate B build."
            )
        _digest(
            source.get("component_decisions_sha256"),
            "spec.source.component_decisions_sha256",
        )
    else:
        if component_status not in {"pending_double_blind_review", "frozen_complete"}:
            raise ValueError(
                "Bootstrap Gate B validation requires component_decisions_status "
                "pending_double_blind_review or frozen_complete."
            )
        if "component_decisions_sha256" in source:
            _digest(
                source.get("component_decisions_sha256"),
                "spec.source.component_decisions_sha256",
            )

    boundary = _mapping(spec.get("ground_truth_boundary"), "spec.ground_truth_boundary")
    if boundary.get("forbidden_consumer") != "analyze":
        raise ValueError("Gate B spec must forbid analyze from consuming ground truth.")
    allowed_consumers = {
        _text(item, "spec.ground_truth_boundary.allowed_consumers item")
        for item in _list(
            boundary.get("allowed_consumers"),
            "spec.ground_truth_boundary.allowed_consumers",
        )
    }
    if "analyze" in allowed_consumers:
        raise ValueError(
            "Gate B spec cannot list analyze as an allowed ground-truth consumer."
        )
    if not {"dataset_audit", "annotation", "evaluation"}.issubset(
        allowed_consumers
    ):
        raise ValueError(
            "Gate B ground-truth consumers must include dataset_audit, annotation, "
            "and evaluation."
        )
    policy = _text(boundary.get("analysis_input_policy"), "analysis_input_policy")
    if "RGB" not in policy or "stream-input-0.1" not in policy:
        raise ValueError("Gate B analysis_input_policy must require RGB stream-input-0.1 inputs.")

    split_policy = _mapping(spec.get("split_policy"), "spec.split_policy")
    if split_policy.get("model_selection_role") != "development_only":
        raise ValueError(
            "spec.split_policy.model_selection_role must be 'development_only'."
        )

    annotation = _mapping(spec.get("annotation_contract"), "spec.annotation_contract")
    expected_annotation = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_scope": ANNOTATION_SCOPE,
        "object_label_rule": "source_label_greater_than_detected_support_label",
        "support_label_rule": "unique_dominant_nonzero_label_in_first_mask",
        "bbox_derivation": COMPONENT_BBOX_DERIVATION,
        "component_review_policy": COMPONENT_REVIEW_POLICY_ID,
        "component_connectivity": 4,
        "secondary_area_ratio": DEFAULT_SECONDARY_AREA_RATIO,
        "review_candidate_rule": COMPONENT_REVIEW_CANDIDATE_RULE,
        "identity_proxy": "sequence_local_physical_instance_id",
        "visual_type_claim": "not_assessed",
    }
    for key, expected in expected_annotation.items():
        if annotation.get(key) != expected:
            raise ValueError(
                f"spec.annotation_contract.{key} must be {expected!r}, "
                f"got {annotation.get(key)!r}."
            )
    exclusion = _text(annotation.get("object_exclusion_policy"), "object_exclusion_policy")
    if "none" not in exclusion.casefold() or "retained" not in exclusion.casefold():
        raise ValueError("Gate B spec must retain low-visibility and fragmented observations.")

    review = _mapping(spec.get("review_contract"), "spec.review_contract")
    configured_flags = tuple(
        _text(item, "spec.review_contract.automatic_flags item")
        for item in _list(review.get("automatic_flags"), "spec.review_contract.automatic_flags")
    )
    if configured_flags != AUTOMATIC_FLAGS:
        raise ValueError(
            "spec.review_contract.automatic_flags does not match the frozen Gate B flag set."
        )
    if review.get("author_review_status") != "pending":
        raise ValueError("Gate B author review must start pending.")
    _integer(
        review.get("component_review_expected_observations"),
        "spec.review_contract.component_review_expected_observations",
    )

    heldout_lock = _mapping(spec.get("heldout_lock"), "spec.heldout_lock")
    if heldout_lock.get("predictions_unlocked") is not False:
        raise ValueError("Gate B held-out predictions must remain locked.")
    allowed = tuple(
        _text(item, "allowed_before_unlock item")
        for item in _list(
            heldout_lock.get("allowed_before_unlock"), "allowed_before_unlock"
        )
    )
    forbidden = tuple(
        _text(item, "forbidden_before_unlock item")
        for item in _list(
            heldout_lock.get("forbidden_before_unlock"), "forbidden_before_unlock"
        )
    )
    if allowed != HELDOUT_ALLOWED_BEFORE_UNLOCK:
        raise ValueError(
            "spec.heldout_lock.allowed_before_unlock does not match the frozen "
            "Gate B policy."
        )
    if forbidden != HELDOUT_FORBIDDEN_BEFORE_UNLOCK:
        raise ValueError(
            "spec.heldout_lock.forbidden_before_unlock does not match the frozen "
            "Gate B policy."
        )
    overlap = set(allowed).intersection(forbidden)
    if overlap:
        raise ValueError(
            "Held-out allowed and forbidden operations must be disjoint; overlap: "
            + ", ".join(sorted(overlap))
        )

    event_support = _mapping(spec.get("event_support"), "spec.event_support")
    for key in (
        "first_visible_or_added",
        "persisted_visible",
        "count_changed_visible",
        "visibility_reduced",
        "visibility_lost",
        "physical_removed",
        "returned_after_physical_removal",
        "object_motion",
        "interpretation",
    ):
        _text(event_support.get(key), f"spec.event_support.{key}")

    roles = _mapping(spec.get("roles"), "spec.roles")
    selections: list[StreamSelection] = []
    seen_groups: set[str] = set()
    seen_group_keys: set[str] = set()
    seen_streams: set[str] = set()
    derived = {
        "development_scene_groups": 0,
        "development_streams": 0,
        "development_frames": 0,
        "heldout_scene_groups": 0,
        "heldout_streams": 0,
        "heldout_frames": 0,
    }
    for role in ROLE_ORDER:
        groups = _list(roles.get(role), f"spec.roles.{role}")
        for position, raw_group in enumerate(groups):
            group = _mapping(raw_group, f"spec.roles.{role}[{position}]")
            scene_group_id = _text(group.get("scene_group_id"), "scene_group_id")
            scene_group_key = _relative_posix(
                group.get("scene_group_key"), "scene_group_key"
            ).as_posix()
            frames_per_stream = _integer(
                group.get("frames_per_stream"), "frames_per_stream", minimum=1
            )
            reason = _text(group.get("reason"), "reason")
            if scene_group_id in seen_groups or scene_group_key in seen_group_keys:
                raise ValueError(f"Duplicate Gate B scene group: {scene_group_id!r}.")
            seen_groups.add(scene_group_id)
            seen_group_keys.add(scene_group_key)
            members = [
                _relative_posix(item, "member_streams item").as_posix()
                for item in _list(group.get("member_streams"), "member_streams")
            ]
            if len(members) != 2 or {_camera(item) for item in members} != {"bottom", "top"}:
                raise ValueError(
                    f"Gate B scene group {scene_group_id!r} must contain one top and one "
                    "bottom stream."
                )
            if any(_paired_scene_key(item) != scene_group_key for item in members):
                raise ValueError(
                    f"Gate B scene group {scene_group_id!r} member paths do not match "
                    f"scene_group_key {scene_group_key!r}."
                )
            for member in members:
                if member in seen_streams:
                    raise ValueError(f"Duplicate Gate B stream membership: {member!r}.")
                seen_streams.add(member)
                selections.append(
                    StreamSelection(
                        role=role,
                        scene_group_id=scene_group_id,
                        scene_group_key=scene_group_key,
                        reason=reason,
                        frames_per_stream=frames_per_stream,
                        source_sequence=member,
                    )
                )
            derived[f"{role}_scene_groups"] += 1
            derived[f"{role}_streams"] += len(members)
            derived[f"{role}_frames"] += frames_per_stream * len(members)

    derived.update(
        {
            "benchmark_scene_groups": derived["development_scene_groups"]
            + derived["heldout_scene_groups"],
            "benchmark_streams": derived["development_streams"]
            + derived["heldout_streams"],
            "benchmark_frames": derived["development_frames"] + derived["heldout_frames"],
        }
    )
    expected_totals = _mapping(spec.get("expected_totals"), "spec.expected_totals")
    normalized_totals: dict[str, int] = {}
    if set(expected_totals) != set(derived):
        raise ValueError(
            "spec.expected_totals must contain exactly the development, heldout, and "
            "benchmark scene/stream/frame totals."
        )
    for key, actual in derived.items():
        expected = _integer(expected_totals.get(key), f"expected_totals.{key}")
        normalized_totals[key] = expected
        if expected != actual:
            raise ValueError(
                f"Gate B spec total mismatch for {key}: declared {expected}, derived {actual}."
            )
    expected_frame_coverage = f"all_{derived['benchmark_frames']}_frames"
    if review.get("frame_coverage") != expected_frame_coverage:
        raise ValueError(
            "spec.review_contract.frame_coverage must cover every selected frame: "
            f"expected {expected_frame_coverage!r}, got "
            f"{review.get('frame_coverage')!r}."
        )
    if not selections:
        raise ValueError("Gate B spec selects no streams.")
    return selections, normalized_totals


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_component_decision_ledger(
    path: Path,
    *,
    spec: dict[str, Any],
) -> ComponentDecisionLedger:
    decision_path = Path(path).resolve(strict=True)
    if not decision_path.is_file():
        raise FileNotFoundError(f"Component decisions must be a file: {decision_path}.")
    payload, raw_bytes = _load_json_object(decision_path, "component decisions")
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    source = _mapping(spec.get("source"), "spec.source")
    expected_sha = _digest(
        source.get("component_decisions_sha256"),
        "spec.source.component_decisions_sha256",
    )
    if actual_sha != expected_sha:
        raise ValueError(
            "Component decisions SHA-256 mismatch: "
            f"spec={expected_sha}, actual={actual_sha}."
        )
    if payload.get("schema_version") != COMPONENT_DECISIONS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported component decisions schema_version: "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("benchmark_id") != spec.get("benchmark_id"):
        raise ValueError("Component decisions benchmark_id does not match the Gate B spec.")
    if payload.get("policy_id") != COMPONENT_REVIEW_POLICY_ID:
        raise ValueError("Component decisions policy_id does not match the frozen policy.")
    expected_source_sha = _digest(
        source.get("source_fingerprint_sha256"),
        "spec.source.source_fingerprint_sha256",
    )
    if payload.get("source_fingerprint_sha256") != expected_source_sha:
        raise ValueError(
            "Component decisions source_fingerprint_sha256 does not match the Gate B spec."
        )
    if payload.get("review_status") != "complete":
        raise ValueError("Component decisions review_status must be 'complete'.")

    review_contract = _mapping(spec.get("review_contract"), "spec.review_contract")
    expected_count = _integer(
        review_contract.get("component_review_expected_observations"),
        "spec.review_contract.component_review_expected_observations",
    )
    decisions = _list(payload.get("decisions"), "component decisions.decisions")
    if len(decisions) != expected_count:
        raise ValueError(
            "Component decision count mismatch: "
            f"expected {expected_count}, found {len(decisions)}."
        )

    by_case_id: dict[str, dict[str, Any]] = {}
    for position, raw_entry in enumerate(decisions):
        context = f"component decisions.decisions[{position}]"
        entry = _mapping(raw_entry, context)
        case_id = _text(entry.get("case_id"), f"{context}.case_id")
        if case_id in by_case_id:
            raise ValueError(f"Duplicate component decision case_id: {case_id!r}.")
        if entry.get("review_status") != "complete":
            raise ValueError(f"{context}.review_status must be 'complete'.")
        _digest(entry.get("source_mask_sha256"), f"{context}.source_mask_sha256")
        _mapping(entry.get("analysis"), f"{context}.analysis")
        automatic = [
            _text(item, f"{context}.automatic_retained_component_ids item")
            for item in _list(
                entry.get("automatic_retained_component_ids"),
                f"{context}.automatic_retained_component_ids",
            )
        ]
        if len(automatic) != len(set(automatic)) or "c001" not in automatic:
            raise ValueError(
                f"{context}.automatic_retained_component_ids must be unique and include c001."
            )

        valid_frame_ids = _valid_frame_ids_for_entry(spec, entry)
        reviews = _list(entry.get("reviews"), f"{context}.reviews")
        if len(reviews) != 2:
            raise ValueError(f"{context}.reviews must contain exactly two blind reviews.")
        reviewer_ids: list[str] = []
        for review_position, raw_review in enumerate(reviews):
            review_context = f"{context}.reviews[{review_position}]"
            review = _mapping(raw_review, review_context)
            if review.get("blind_to_automatic_proposal") is not True:
                raise ValueError(
                    f"{review_context} must set "
                    "blind_to_automatic_proposal=true."
                )
            reviewer_ids.append(_reviewer_id(review, review_context))
            _case_integrity_decision(
                review,
                review_context,
                valid_frame_ids=valid_frame_ids,
            )
            _primary_review_component_rows(
                review, review_context, valid_frame_ids=valid_frame_ids
            )
        if len({item.casefold() for item in reviewer_ids}) != 2:
            raise ValueError(f"{context}.reviews must identify two distinct reviewers.")

        adjudication = entry.get("adjudication")
        if adjudication is not None:
            _mapping(adjudication, f"{context}.adjudication")

        final_rows = _list(
            entry.get("final_component_decisions"),
            f"{context}.final_component_decisions",
        )
        seen_components: set[str] = set()
        for final_position, raw_final in enumerate(final_rows):
            final_context = f"{context}.final_component_decisions[{final_position}]"
            final = _mapping(raw_final, final_context)
            component_id = _text(final.get("component_id"), f"{final_context}.component_id")
            if component_id == "c001":
                raise ValueError(f"{final_context} must not repeat implicit KEEP component c001.")
            if component_id in seen_components:
                raise ValueError(f"Duplicate final component decision for {component_id!r}.")
            seen_components.add(component_id)
            if final.get("decision") not in FINAL_COMPONENT_DECISIONS:
                raise ValueError(f"{final_context}.decision must be KEEP or DROP.")
            if final.get("confidence") not in FINAL_COMPONENT_CONFIDENCES:
                raise ValueError(f"{final_context}.confidence must be HIGH or MEDIUM.")
            rationale_codes = [
                _text(item, f"{final_context}.rationale_codes item")
                for item in _list(final.get("rationale_codes"), f"{final_context}.rationale_codes")
            ]
            if not rationale_codes or len(rationale_codes) != len(set(rationale_codes)):
                raise ValueError(
                    f"{final_context}.rationale_codes must be a non-empty unique list."
                )
            invalid_codes = [
                code
                for code in rationale_codes
                if code not in RATIONALE_CODES_BY_DECISION[str(final["decision"])]
            ]
            if invalid_codes:
                raise ValueError(
                    f"{final_context}.rationale_codes contains invalid final codes: "
                    + ", ".join(invalid_codes)
                )
            _evidence_frame_ids(
                final, final_context, valid_frame_ids=valid_frame_ids
            )
            if not isinstance(final.get("notes"), str):
                raise ValueError(f"{final_context}.notes must be a string.")

        final_retained = [
            _text(item, f"{context}.final_retained_component_ids item")
            for item in _list(
                entry.get("final_retained_component_ids"),
                f"{context}.final_retained_component_ids",
            )
        ]
        if (
            not final_retained
            or final_retained[0] != "c001"
            or len(final_retained) != len(set(final_retained))
        ):
            raise ValueError(
                f"{context}.final_retained_component_ids must start with c001 and be unique."
            )
        if entry.get("final_confidence") not in FINAL_COMPONENT_CONFIDENCES:
            raise ValueError(f"{context}.final_confidence must be HIGH or MEDIUM.")
        _text(entry.get("resolved_by"), f"{context}.resolved_by")
        if not isinstance(entry.get("author_attention"), bool):
            raise ValueError(f"{context}.author_attention must be boolean.")
        if _contains_exact_outcome(
            {
                "final_component_decisions": final_rows,
                "final_retained_component_ids": final_retained,
                "final_confidence": entry.get("final_confidence"),
                "resolved_by": entry.get("resolved_by"),
            },
            FINAL_FORBIDDEN_OUTCOMES,
        ):
            raise ValueError(f"{context} contains a forbidden unresolved final outcome.")
        by_case_id[case_id] = entry

    return ComponentDecisionLedger(
        path=decision_path,
        sha256=actual_sha,
        raw_bytes=raw_bytes,
        decisions_by_case_id=by_case_id,
    )


def _contains_exact_outcome(value: object, forbidden: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return any(_contains_exact_outcome(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_exact_outcome(item, forbidden) for item in value)
    return isinstance(value, str) and value in forbidden


def _case_integrity_decision(
    container: dict[str, Any],
    context: str,
    *,
    valid_frame_ids: set[str],
    field_name: str = "case_integrity_decision",
    final_only: bool = False,
) -> str:
    row_context = f"{context}.{field_name}"
    row = _mapping(container.get(field_name), row_context)
    decision = row.get("decision")
    if decision not in CASE_INTEGRITY_DECISIONS:
        raise ValueError(f"{row_context}.decision must be OK or INTEGRITY_ALERT.")

    confidence = row.get("confidence")
    rationale_codes = [
        _text(item, f"{row_context}.rationale_codes item")
        for item in _list(row.get("rationale_codes"), f"{row_context}.rationale_codes")
    ]
    if len(rationale_codes) != len(set(rationale_codes)):
        raise ValueError(f"{row_context}.rationale_codes must be unique.")
    if decision == "OK":
        if confidence not in FINAL_COMPONENT_CONFIDENCES or rationale_codes:
            raise ValueError(
                f"{row_context} OK requires HIGH or MEDIUM confidence and no "
                "integrity rationale codes."
            )
    else:
        valid_codes = RATIONALE_CODES_BY_DECISION["INTEGRITY_ALERT"]
        if (
            confidence != "LOW"
            or not rationale_codes
            or any(code not in valid_codes for code in rationale_codes)
        ):
            raise ValueError(
                f"{row_context} INTEGRITY_ALERT requires LOW confidence and valid "
                "integrity rationale codes."
            )

    evidence = _evidence_frame_ids(
        row,
        row_context,
        valid_frame_ids=valid_frame_ids,
    )
    if not evidence:
        raise ValueError(f"{row_context}.evidence_frame_ids must be non-empty.")
    if not isinstance(row.get("notes"), str) or not row["notes"].strip():
        raise ValueError(f"{row_context}.notes must be a non-empty string.")
    if final_only and decision != FINAL_CASE_INTEGRITY_RESOLUTION:
        raise ValueError(
            f"{context}.{field_name} must be OK; unresolved or "
            "author_decision_required is forbidden in a final build."
        )
    return str(decision)


def _evidence_frame_ids(
    row: dict[str, Any], context: str, *, valid_frame_ids: set[str] | None
) -> list[str]:
    frame_ids = [
        _text(item, f"{context}.evidence_frame_ids item")
        for item in _list(row.get("evidence_frame_ids"), f"{context}.evidence_frame_ids")
    ]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{context}.evidence_frame_ids must be unique.")
    if valid_frame_ids is not None:
        invalid = [frame_id for frame_id in frame_ids if frame_id not in valid_frame_ids]
        if invalid:
            raise ValueError(
                f"{context}.evidence_frame_ids contains frame IDs outside its stream: "
                + ", ".join(invalid)
            )
    return frame_ids


def _primary_review_component_rows(
    review: dict[str, Any],
    context: str,
    *,
    valid_frame_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    present_fields = [
        field_name
        for field_name in ("component_decisions", "components")
        if field_name in review
    ]
    if len(present_fields) != 1:
        raise ValueError(
            f"{context} must contain exactly one of component_decisions or components."
        )
    field_name = present_fields[0]
    rows = [
        _mapping(item, f"{context}.{field_name} item")
        for item in _list(review.get(field_name), f"{context}.{field_name}")
    ]
    seen: set[str] = set()
    for position, row in enumerate(rows):
        row_context = f"{context}.{field_name}[{position}]"
        component_id = _text(row.get("component_id"), f"{row_context}.component_id")
        if component_id == "c001":
            raise ValueError(f"{row_context} must not repeat implicit KEEP component c001.")
        if component_id in seen:
            raise ValueError(
                f"{context} contains duplicate primary decision for {component_id!r}."
            )
        seen.add(component_id)
        decision = row.get("decision")
        if decision not in PRIMARY_COMPONENT_DECISIONS:
            raise ValueError(
                f"{row_context}.decision must be KEEP, DROP, UNRESOLVED, or "
                "INTEGRITY_ALERT."
            )
        confidence = row.get("confidence")
        if decision in FINAL_COMPONENT_DECISIONS:
            if confidence not in FINAL_COMPONENT_CONFIDENCES:
                raise ValueError(
                    f"{row_context}.confidence for KEEP/DROP must be HIGH or MEDIUM."
                )
        elif confidence != "LOW":
            raise ValueError(
                f"{row_context}.confidence for UNRESOLVED/INTEGRITY_ALERT must be LOW."
            )
        rationale_codes = [
            _text(item, f"{row_context}.rationale_codes item")
            for item in _list(row.get("rationale_codes"), f"{row_context}.rationale_codes")
        ]
        if not rationale_codes or len(rationale_codes) != len(set(rationale_codes)):
            raise ValueError(
                f"{row_context}.rationale_codes must be a non-empty unique list."
            )
        invalid_codes = [
            code
            for code in rationale_codes
            if code not in RATIONALE_CODES_BY_DECISION[str(decision)]
        ]
        if invalid_codes:
            raise ValueError(
                f"{row_context}.rationale_codes contains invalid codes: "
                + ", ".join(invalid_codes)
            )
        _evidence_frame_ids(row, row_context, valid_frame_ids=valid_frame_ids)
        if not isinstance(row.get("notes"), str):
            raise ValueError(f"{row_context}.notes must be a string.")
    return rows


def _final_component_row(
    raw_row: object,
    context: str,
    *,
    valid_frame_ids: set[str] | None,
) -> dict[str, Any]:
    row = _mapping(raw_row, context)
    component_id = _text(row.get("component_id"), f"{context}.component_id")
    if component_id == "c001":
        raise ValueError(f"{context} must not repeat implicit KEEP component c001.")
    decision = row.get("decision")
    if decision not in FINAL_COMPONENT_DECISIONS:
        raise ValueError(f"{context}.decision must be KEEP or DROP.")
    if row.get("confidence") not in FINAL_COMPONENT_CONFIDENCES:
        raise ValueError(f"{context}.confidence must be HIGH or MEDIUM.")
    rationale_codes = [
        _text(item, f"{context}.rationale_codes item")
        for item in _list(row.get("rationale_codes"), f"{context}.rationale_codes")
    ]
    if not rationale_codes or len(rationale_codes) != len(set(rationale_codes)):
        raise ValueError(f"{context}.rationale_codes must be a non-empty unique list.")
    invalid_codes = [
        code
        for code in rationale_codes
        if code not in RATIONALE_CODES_BY_DECISION[str(decision)]
    ]
    if invalid_codes:
        raise ValueError(
            f"{context}.rationale_codes contains invalid final codes: "
            + ", ".join(invalid_codes)
        )
    _evidence_frame_ids(row, context, valid_frame_ids=valid_frame_ids)
    if not isinstance(row.get("notes"), str):
        raise ValueError(f"{context}.notes must be a string.")
    return row


def _valid_frame_ids_for_entry(spec: dict[str, Any], entry: dict[str, Any]) -> set[str]:
    source_sequence = _text(entry.get("source_sequence"), "component decision.source_sequence")
    roles = _mapping(spec.get("roles"), "spec.roles")
    matches: list[int] = []
    for role in ROLE_ORDER:
        for raw_group in _list(roles.get(role), f"spec.roles.{role}"):
            group = _mapping(raw_group, f"spec.roles.{role} item")
            members = _list(group.get("member_streams"), "member_streams")
            if source_sequence in members:
                matches.append(_integer(group.get("frames_per_stream"), "frames_per_stream", minimum=1))
    if len(matches) != 1:
        raise ValueError(
            "Component decision source_sequence must identify exactly one selected stream: "
            f"{source_sequence!r}."
        )
    return {f"frame_{index:04d}" for index in range(1, matches[0] + 1)}


def _validate_component_decision_provenance(
    entry: dict[str, Any],
    *,
    secondary_component_ids: Sequence[str],
    valid_frame_ids: set[str],
    context: str,
) -> None:
    reviews = [
        _mapping(item, f"{context}.reviews item")
        for item in _list(entry.get("reviews"), f"{context}.reviews")
    ]
    if len(reviews) != 2:
        raise ValueError(f"{context}.reviews must contain exactly two blind reviews.")

    reviewer_ids: list[str] = []
    review_maps: list[dict[str, dict[str, Any]]] = []
    integrity_decisions: list[str] = []
    expected_ids = set(secondary_component_ids)
    for position, review in enumerate(reviews):
        review_context = f"{context}.reviews[{position}]"
        if review.get("blind_to_automatic_proposal") is not True:
            raise ValueError(f"{review_context} must set blind_to_automatic_proposal=true.")
        reviewer_ids.append(_reviewer_id(review, review_context))
        integrity_decisions.append(
            _case_integrity_decision(
                review,
                review_context,
                valid_frame_ids=valid_frame_ids,
            )
        )
        rows = _primary_review_component_rows(
            review, review_context, valid_frame_ids=valid_frame_ids
        )
        row_map = {str(row["component_id"]): row for row in rows}
        if len(rows) != len(expected_ids) or set(row_map) != expected_ids:
            raise ValueError(
                f"{review_context} must cover every current secondary component exactly once."
            )
        review_maps.append(row_map)
    if reviewer_ids[0].casefold() == reviewer_ids[1].casefold():
        raise ValueError(f"{context}.reviews must identify two distinct reviewers.")

    final_rows = [
        _final_component_row(
            item,
            f"{context}.final_component_decisions[{position}]",
            valid_frame_ids=valid_frame_ids,
        )
        for position, item in enumerate(
            _list(entry.get("final_component_decisions"), f"{context}.final_component_decisions")
        )
    ]
    final_by_id = {str(row["component_id"]): row for row in final_rows}
    if len(final_rows) != len(expected_ids) or set(final_by_id) != expected_ids:
        raise ValueError(
            f"{context}.final_component_decisions must cover every current secondary "
            "component exactly once."
        )

    disputed: set[str] = set()
    consensus: dict[str, str] = {}
    for component_id in secondary_component_ids:
        decisions = [str(rows[component_id]["decision"]) for rows in review_maps]
        if decisions[0] == decisions[1] and decisions[0] in FINAL_COMPONENT_DECISIONS:
            consensus[component_id] = decisions[0]
        else:
            disputed.add(component_id)

    integrity_requires_adjudication = any(
        decision != FINAL_CASE_INTEGRITY_RESOLUTION for decision in integrity_decisions
    )
    adjudication_required = bool(disputed) or integrity_requires_adjudication
    raw_adjudication = entry.get("adjudication")
    if adjudication_required and raw_adjudication is None:
        raise ValueError(
            f"{context} requires structured adjudication for disagreement, UNRESOLVED, "
            "or INTEGRITY_ALERT."
        )
    if not adjudication_required and raw_adjudication is not None:
        raise ValueError(
            f"{context}.adjudication must be null when both blind reviews fully agree."
        )

    adjudicated: dict[str, dict[str, Any]] = {}
    if raw_adjudication is not None:
        adjudication = _mapping(raw_adjudication, f"{context}.adjudication")
        adjudicator_id = _canonical_reviewer_id(
            adjudication.get("reviewer_id"), f"{context}.adjudication.reviewer_id"
        )
        exception = adjudication.get("reviewer_identity_exception")
        if exception not in (None, ""):
            raise ValueError(
                f"{context}.adjudication reviewer identity exceptions are not permitted."
            )
        if adjudicator_id.casefold() in {item.casefold() for item in reviewer_ids}:
            raise ValueError(
                f"{context}.adjudication reviewer must be distinct from both primary "
                "reviewers."
            )
        if adjudication.get("review_status") != "complete":
            raise ValueError(
                f"{context}.adjudication.review_status must be complete."
            )
        resolution = _case_integrity_decision(
            adjudication,
            f"{context}.adjudication",
            field_name="case_integrity_resolution",
            valid_frame_ids=valid_frame_ids,
            final_only=True,
        )
        if not isinstance(adjudication.get("case_notes"), str):
            raise ValueError(f"{context}.adjudication.case_notes must be a string.")
        adjudication_rows = [
            _final_component_row(
                item,
                f"{context}.adjudication.component_decisions[{position}]",
                valid_frame_ids=valid_frame_ids,
            )
            for position, item in enumerate(
                _list(
                    adjudication.get("component_decisions"),
                    f"{context}.adjudication.component_decisions",
                )
            )
        ]
        adjudicated = {str(row["component_id"]): row for row in adjudication_rows}
        if len(adjudication_rows) != len(disputed) or set(adjudicated) != disputed:
            raise ValueError(
                f"{context}.adjudication must cover exactly the disputed secondary components."
            )

    for component_id, consensus_decision in consensus.items():
        if final_by_id[component_id]["decision"] != consensus_decision:
            raise ValueError(
                f"{context}.{component_id} final decision overrides blind-review consensus."
            )
    for component_id in disputed:
        if final_by_id[component_id]["decision"] != adjudicated[component_id]["decision"]:
            raise ValueError(
                f"{context}.{component_id} final decision does not match adjudication."
            )

    final_integrity = _case_integrity_decision(
        entry,
        context,
        field_name="case_integrity_resolution",
        valid_frame_ids=valid_frame_ids,
        final_only=True,
    )


def _source_descriptor(path: Path, root: Path) -> SourceDescriptor:
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ValueError(f"OCID source file resolves outside the dataset root: {path}.")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return SourceDescriptor(
        relative_path=resolved.relative_to(root).as_posix(),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _fingerprint(descriptors: Iterable[SourceDescriptor]) -> dict[str, object]:
    entries = sorted(descriptors, key=lambda item: item.relative_path)
    if len({item.relative_path for item in entries}) != len(entries):
        raise ValueError("OCID source fingerprint contains duplicate relative paths.")
    digest = hashlib.sha256()
    total_size = 0
    for item in entries:
        record = f"{item.relative_path}\0{item.size_bytes}\0{item.sha256}\n"
        digest.update(record.encode("utf-8"))
        total_size += item.size_bytes
    return {
        "algorithm": "sha256",
        "method": SOURCE_FINGERPRINT_METHOD,
        "scope": "paired_rgb_and_label_png_files",
        "entry_count": len(entries),
        "total_size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def _validate_fingerprint(
    recorded: object,
    actual: dict[str, object],
    name: str,
) -> None:
    value = _mapping(recorded, name)
    expected_keys = {
        "algorithm",
        "method",
        "scope",
        "entry_count",
        "total_size_bytes",
        "sha256",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{name} has an unexpected shape.")
    normalized = {
        "algorithm": _text(value.get("algorithm"), f"{name}.algorithm"),
        "method": _text(value.get("method"), f"{name}.method"),
        "scope": _text(value.get("scope"), f"{name}.scope"),
        "entry_count": _integer(value.get("entry_count"), f"{name}.entry_count"),
        "total_size_bytes": _integer(
            value.get("total_size_bytes"), f"{name}.total_size_bytes"
        ),
        "sha256": _digest(value.get("sha256"), f"{name}.sha256"),
    }
    if normalized != actual:
        raise ValueError(
            f"{name} does not match the current OCID source fingerprint: "
            f"recorded={normalized['sha256']}, actual={actual['sha256']}."
        )


def _png_files(directory: Path, description: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing OCID {description} directory: {directory}.")
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() == ".png"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not files:
        raise ValueError(f"OCID sequence contains no {description} PNG files: {directory}.")
    folded = [path.name.casefold() for path in files]
    if len(folded) != len(set(folded)):
        raise ValueError(f"OCID {description} filenames are not case-insensitively unique.")
    return files


def _source_pairs(root: Path, source_sequence: str) -> list[SourcePair]:
    sequence = _relative_posix(source_sequence, "audit source_sequence")
    directory = root.joinpath(*sequence.parts).resolve(strict=False)
    if not _inside(directory, root):
        raise ValueError(f"Resolved OCID sequence is outside the dataset root: {source_sequence}.")
    if not directory.is_dir():
        raise FileNotFoundError(f"OCID sequence directory does not exist: {directory}.")
    rgb_files = _png_files(directory / "rgb", "RGB")
    label_files = _png_files(directory / "label", "label")
    rgb_names = [path.name for path in rgb_files]
    label_names = [path.name for path in label_files]
    if rgb_names != label_names:
        raise ValueError(
            f"RGB and label filenames do not align for {source_sequence}: "
            f"RGB={rgb_names}, label={label_names}."
        )
    return [
        SourcePair(filename=rgb.name, rgb_path=rgb, label_path=label)
        for rgb, label in zip(rgb_files, label_files, strict=True)
    ]


def _audit_sequence_maps(
    audit: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if audit.get("schema_id") != AUDIT_SCHEMA_ID:
        raise ValueError(f"Unsupported structural audit schema_id: {audit.get('schema_id')!r}.")
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION or audit.get("status") != "complete":
        raise ValueError("Structural audit must be complete schema version 1.0.0.")
    if audit.get("scope") != "evaluation_only_dataset_audit":
        raise ValueError("Structural audit must have evaluation-only scope.")
    boundary = _mapping(audit.get("ground_truth_boundary"), "audit.ground_truth_boundary")
    if boundary.get("forbidden_consumer") != "analyze" or boundary.get("uses_ground_truth") is not True:
        raise ValueError("Structural audit ground-truth boundary is incompatible with Gate B.")
    parameters = _mapping(audit.get("parameters"), "audit.parameters")
    if parameters.get("support_label_detection") != "unique_dominant_nonzero_label_in_first_mask":
        raise ValueError("Structural audit uses an unsupported support-label detector.")
    if parameters.get("object_label_rule") != "source_label_greater_than_support_label":
        raise ValueError("Structural audit uses an unsupported object-label rule.")
    if parameters.get("expected_normal_values_used_for_object_parsing") is not False:
        raise ValueError("Structural audit must not parse objects from expected floor/table offsets.")

    by_source: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(_list(audit.get("sequences"), "audit.sequences")):
        sequence = _mapping(raw, f"audit.sequences[{position}]")
        source = _relative_posix(
            sequence.get("source_sequence"), f"audit.sequences[{position}].source_sequence"
        ).as_posix()
        sequence_id = _text(sequence.get("sequence_id"), "audit sequence_id")
        if not SAFE_ID.fullmatch(sequence_id):
            raise ValueError(f"Unsafe structural-audit sequence_id: {sequence_id!r}.")
        if source in by_source or sequence_id in by_id:
            raise ValueError("Structural audit contains duplicate sequence identity fields.")
        by_source[source] = sequence
        by_id[sequence_id] = sequence
    if not by_source:
        raise ValueError("Structural audit contains no sequences.")

    summary = _mapping(audit.get("summary"), "audit.summary")
    if _integer(summary.get("sequence_count"), "audit.summary.sequence_count") != len(by_source):
        raise ValueError("Structural audit sequence_count does not match its sequence inventory.")

    groups: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(_list(audit.get("scene_groups"), "audit.scene_groups")):
        group = _mapping(raw, f"audit.scene_groups[{position}]")
        group_id = _text(group.get("scene_group_id"), "audit scene_group_id")
        if group_id in groups:
            raise ValueError(f"Duplicate structural-audit scene_group_id: {group_id!r}.")
        groups[group_id] = group
    return by_source, groups


def _verify_complete_audit_source(
    *,
    root: Path,
    audit: dict[str, Any],
    sequences: dict[str, dict[str, Any]],
) -> dict[str, list[SourcePair]]:
    pairs_by_source: dict[str, list[SourcePair]] = {}
    all_descriptors: list[SourceDescriptor] = []
    expected_paths: set[str] = set()
    for source_sequence in sorted(sequences):
        pairs = _source_pairs(root, source_sequence)
        pairs_by_source[source_sequence] = pairs
        descriptors: list[SourceDescriptor] = []
        for pair in pairs:
            descriptors.append(_source_descriptor(pair.rgb_path, root))
            descriptors.append(_source_descriptor(pair.label_path, root))
        sequence_fingerprint = _fingerprint(descriptors)
        _validate_fingerprint(
            sequences[source_sequence].get("source_fingerprint"),
            sequence_fingerprint,
            f"audit sequence {source_sequence!r} source_fingerprint",
        )
        all_descriptors.extend(descriptors)
        expected_paths.update(item.relative_path for item in descriptors)

    discovered_paths: set[str] = set()
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.casefold() == ".png"
            and path.parent.name.casefold() in {"rgb", "label"}
        ):
            resolved = path.resolve(strict=True)
            if not _inside(resolved, root):
                raise ValueError(f"OCID PNG resolves outside the dataset root: {path}.")
            discovered_paths.add(resolved.relative_to(root).as_posix())
    if discovered_paths != expected_paths:
        missing = sorted(expected_paths - discovered_paths)
        extra = sorted(discovered_paths - expected_paths)
        raise ValueError(
            "Current OCID RGB/label inventory differs from the structural audit: "
            f"missing={missing}, extra={extra}."
        )

    full_fingerprint = _fingerprint(all_descriptors)
    provenance = _mapping(audit.get("provenance"), "audit.provenance")
    _validate_fingerprint(
        provenance.get("source_fingerprint"),
        full_fingerprint,
        "audit.provenance.source_fingerprint",
    )
    return pairs_by_source


def _validate_selected_membership(
    *,
    selections: Sequence[StreamSelection],
    audit_sequences: dict[str, dict[str, Any]],
    audit_groups: dict[str, dict[str, Any]],
) -> None:
    by_group: dict[str, list[StreamSelection]] = {}
    for selection in selections:
        by_group.setdefault(selection.scene_group_id, []).append(selection)
        sequence = audit_sequences.get(selection.source_sequence)
        if sequence is None:
            raise ValueError(
                f"Gate B stream is absent from the structural audit: "
                f"{selection.source_sequence}."
            )
        expected_stream_id = _stream_id(selection.source_sequence)
        if sequence.get("sequence_id") != expected_stream_id:
            raise ValueError(
                f"Structural-audit sequence_id mismatch for {selection.source_sequence}."
            )
        if sequence.get("scene_group_id") != selection.scene_group_id:
            raise ValueError(
                f"Structural-audit scene_group_id mismatch for {selection.source_sequence}."
            )
        if sequence.get("paired_scene_key") != selection.scene_group_key:
            raise ValueError(
                f"Structural-audit paired scene mismatch for {selection.source_sequence}."
            )
        if _integer(sequence.get("frame_count"), "audit frame_count", minimum=1) != selection.frames_per_stream:
            raise ValueError(
                f"Frame-count mismatch for {selection.source_sequence}: spec expects "
                f"{selection.frames_per_stream}, audit has {sequence.get('frame_count')}."
            )

    for group_id, group_selections in by_group.items():
        group = audit_groups.get(group_id)
        if group is None:
            raise ValueError(f"Gate B scene group is absent from the structural audit: {group_id}.")
        expected_ids = sorted(_stream_id(item.source_sequence) for item in group_selections)
        recorded_ids = sorted(
            _text(item, "audit member_sequence_ids item")
            for item in _list(group.get("member_sequence_ids"), "member_sequence_ids")
        )
        if recorded_ids != expected_ids:
            raise ValueError(f"Exact stream membership mismatch for scene group {group_id}.")
        if group.get("scene_group_key") != group_selections[0].scene_group_key:
            raise ValueError(f"Scene-group key mismatch for {group_id}.")
        if group.get("complete_top_bottom_pair") is not True or group.get("matching_frame_count") is not True:
            raise ValueError(f"Gate B scene group {group_id} is not a complete matched camera pair.")


def _load_label_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"OCID label source is not PNG: {path}.")
            labels = np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f"Cannot decode OCID label mask {path}: {error}") from error
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"OCID label mask must be a 2D integer array: {path}.")
    if np.any(labels < 0):
        raise ValueError(f"OCID label mask contains a negative value: {path}.")
    return labels


def _rgb_properties(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"OCID RGB input must be an RGB PNG: {path}.")
            image.load()
            width, height = image.size
    except OSError as error:
        raise ValueError(f"Cannot decode OCID RGB image {path}: {error}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"OCID RGB image has invalid dimensions: {path}.")
    return int(width), int(height)


def _detect_support_label(labels: np.ndarray, source: str) -> int:
    values, counts = np.unique(labels, return_counts=True)
    nonzero = [
        (int(value), int(count))
        for value, count in zip(values, counts, strict=True)
        if int(value) != 0
    ]
    if not nonzero:
        raise ValueError(f"Cannot detect a support label from an all-zero first mask: {source}.")
    maximum = max(count for _, count in nonzero)
    candidates = sorted(value for value, count in nonzero if count == maximum)
    if len(candidates) != 1:
        raise ValueError(
            f"First mask has no unique dominant nonzero support label in {source}: "
            f"candidates={candidates}."
        )
    return candidates[0]


def _validate_support_label(labels: np.ndarray, support_label: int, source: str) -> None:
    values, counts = np.unique(labels, return_counts=True)
    by_label = {
        int(value): int(count) for value, count in zip(values, counts, strict=True)
    }
    support_count = by_label.get(support_label, 0)
    if support_count == 0:
        raise ValueError(f"Audited support label {support_label} is absent from {source}.")
    competing = {
        label: count
        for label, count in by_label.items()
        if label > support_label and count >= support_count
    }
    if competing:
        raise ValueError(
            f"Audited support label {support_label} is not dominant over objects in "
            f"{source}: competing={competing}."
        )


def _connected_component_count(mask: np.ndarray) -> int:
    """Count 4-connected components via row runs without SciPy."""

    parents: list[int] = []

    def add() -> int:
        identifier = len(parents)
        parents.append(identifier)
        return identifier

    def find(identifier: int) -> int:
        root = identifier
        while parents[root] != root:
            root = parents[root]
        while parents[identifier] != identifier:
            parent = parents[identifier]
            parents[identifier] = root
            identifier = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parents[right_root] = left_root
            else:
                parents[left_root] = right_root

    previous: list[tuple[int, int, int]] = []
    for raw_row in mask:
        row = np.asarray(raw_row, dtype=bool)
        padded = np.pad(row, (1, 1), constant_values=False)
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
        current: list[tuple[int, int, int]] = []
        for start_raw, end_raw in zip(starts, ends, strict=True):
            start = int(start_raw)
            end = int(end_raw)
            identifier = add()
            for prior_start, prior_end, prior_id in previous:
                if prior_end < start:
                    continue
                if prior_start > end:
                    break
                union(identifier, prior_id)
            current.append((start, end, identifier))
        previous = current
    return len({find(identifier) for identifier in range(len(parents))})


def _physical_instance_id(stream_id: str, source_label: int) -> str:
    return f"{stream_id}__label_{source_label:03d}"


def _component_case_id(stream_id: str, frame_id: str, source_label: int) -> str:
    return f"{stream_id}__{frame_id}__label_{source_label:03d}"


def _finalize_observation_components(
    *,
    analysis: ComponentAnalysis,
    ledger: ComponentDecisionLedger,
    consumed_case_ids: set[str],
    selection: StreamSelection,
    stream_id: str,
    frame_id: str,
    frame_index: int,
    source_filename: str,
    source_label: int,
) -> tuple[
    tuple[int, int, int, int],
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[dict[str, object], ...],
    str,
    str,
    bool,
]:
    case_id = _component_case_id(stream_id, frame_id, source_label)
    entry = ledger.decisions_by_case_id.get(case_id)
    if not analysis.review_required:
        if entry is not None:
            raise ValueError(
                f"Extra component decision {case_id!r}: current analysis does not require review."
            )
        all_component_ids = analysis.component_ids()
        finalized = finalize_component_review(
            analysis,
            ComponentReviewDecision(
                source_mask_sha256=analysis.source_mask_sha256,
                retained_component_ids=all_component_ids,
                reason="raw_union_bbox_unchanged_no_manual_review",
            ),
        )
        final_rows = tuple(
            {
                "component_id": component.component_id,
                "decision": "KEEP",
                "source": "raw_union_bbox_unchanged_no_manual_review",
            }
            for component in analysis.components[1:]
        )
        return (
            (
                finalized.final_bbox.x,
                finalized.final_bbox.y,
                finalized.final_bbox.width,
                finalized.final_bbox.height,
            ),
            finalized.final_mask,
            finalized.retained_component_ids,
            tuple(
                component_id
                for component_id in analysis.component_ids()
                if component_id not in finalized.retained_component_ids
            ),
            final_rows,
            "HIGH",
            "raw_union_bbox_unchanged_no_manual_review",
            False,
        )

    if entry is None:
        raise ValueError(f"Missing required component decision for {case_id!r}.")
    expected_metadata = {
        "case_id": case_id,
        "role": selection.role,
        "stream_id": stream_id,
        "source_sequence": selection.source_sequence,
        "frame_id": frame_id,
        "frame_index": frame_index,
        "source_filename": source_filename,
        "source_label": source_label,
        "source_mask_sha256": analysis.source_mask_sha256,
        "analysis": analysis.as_dict(),
        "automatic_retained_component_ids": list(
            analysis.automatic_retained_component_ids
        ),
    }
    for key, expected in expected_metadata.items():
        if entry.get(key) != expected:
            raise ValueError(
                f"Stale or mismatched component decision evidence at {case_id}.{key}: "
                f"expected={expected!r}, found={entry.get(key)!r}."
            )

    secondary_ids = [component.component_id for component in analysis.components[1:]]
    valid_frame_ids = {
        f"frame_{index:04d}" for index in range(1, selection.frames_per_stream + 1)
    }
    _validate_component_decision_provenance(
        entry,
        secondary_component_ids=secondary_ids,
        valid_frame_ids=valid_frame_ids,
        context=f"component decision {case_id}",
    )
    for review_position, raw_review in enumerate(
        _list(entry.get("reviews"), f"component decision {case_id}.reviews")
    ):
        review_context = f"component decision {case_id}.reviews[{review_position}]"
        review = _mapping(raw_review, review_context)
        primary_rows = _primary_review_component_rows(review, review_context)
        primary_ids = [str(item["component_id"]) for item in primary_rows]
        if len(primary_ids) != len(secondary_ids) or set(primary_ids) != set(
            secondary_ids
        ):
            raise ValueError(
                f"{review_context} must cover every current secondary component "
                "exactly once."
            )
    final_rows = [
        _mapping(item, f"component decision {case_id}.final_component_decisions item")
        for item in _list(
            entry.get("final_component_decisions"),
            f"component decision {case_id}.final_component_decisions",
        )
    ]
    final_by_id = {str(item["component_id"]): item for item in final_rows}
    if set(final_by_id) != set(secondary_ids) or len(final_rows) != len(secondary_ids):
        raise ValueError(
            f"Component decision {case_id!r} must cover every current secondary "
            "component exactly once."
        )
    expected_retained = ["c001"] + [
        component_id
        for component_id in secondary_ids
        if final_by_id[component_id]["decision"] == "KEEP"
    ]
    if entry.get("final_retained_component_ids") != expected_retained:
        raise ValueError(
            f"Component decision {case_id!r} final_retained_component_ids does not "
            "equal c001 plus every final KEEP component."
        )
    decision = ComponentReviewDecision(
        source_mask_sha256=analysis.source_mask_sha256,
        retained_component_ids=tuple(expected_retained),
        reason=_text(entry.get("resolved_by"), f"component decision {case_id}.resolved_by"),
    )
    finalized = finalize_component_review(analysis, decision)
    consumed_case_ids.add(case_id)
    return (
        (
            finalized.final_bbox.x,
            finalized.final_bbox.y,
            finalized.final_bbox.width,
            finalized.final_bbox.height,
        ),
        finalized.final_mask,
        finalized.retained_component_ids,
        tuple(
            component_id
            for component_id in analysis.component_ids()
            if component_id not in finalized.retained_component_ids
        ),
        tuple(dict(item) for item in final_rows),
        str(entry["final_confidence"]),
        str(entry["resolved_by"]),
        bool(entry["author_attention"]),
    )


def _derive_observation(
    labels: np.ndarray,
    *,
    source_label: int,
    stream_id: str,
    selection: StreamSelection,
    frame_id: str,
    frame_index: int,
    source_filename: str,
    ledger: ComponentDecisionLedger,
    consumed_case_ids: set[str],
    secondary_area_ratio: float,
) -> Observation:
    mask = labels == source_label
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError(f"Cannot derive an observation for absent label {source_label}.")
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max())
    bottom = int(rows.max())
    visible_pixels = int(rows.size)
    raw_bbox = (left, top, right - left + 1, bottom - top + 1)
    analysis = analyze_component_mask(
        np.asarray(mask, dtype=np.bool_),
        secondary_area_ratio=secondary_area_ratio,
    )
    if analysis.raw_bbox.as_dict() != {
        "x": raw_bbox[0],
        "y": raw_bbox[1],
        "width": raw_bbox[2],
        "height": raw_bbox[3],
    }:
        raise AssertionError("Component analysis raw bbox differs from raw OCID evidence.")
    (
        final_bbox,
        retained_mask,
        retained_component_ids,
        removed_component_ids,
        final_component_decisions,
        final_confidence,
        resolved_by,
        author_attention,
    ) = _finalize_observation_components(
        analysis=analysis,
        ledger=ledger,
        consumed_case_ids=consumed_case_ids,
        selection=selection,
        stream_id=stream_id,
        frame_id=frame_id,
        frame_index=frame_index,
        source_filename=source_filename,
        source_label=source_label,
    )
    return Observation(
        physical_instance_id=_physical_instance_id(stream_id, source_label),
        source_label=source_label,
        visible_pixels=visible_pixels,
        visible_fraction=round(visible_pixels / labels.size, 12),
        raw_bbox=raw_bbox,
        bbox=final_bbox,
        centroid=(round(float(columns.mean()), 6), round(float(rows.mean()), 6)),
        connected_component_count=len(analysis.components),
        border_touch=bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ),
        source_mask_sha256=analysis.source_mask_sha256,
        component_review_required=analysis.review_required,
        automatic_retained_component_ids=analysis.automatic_retained_component_ids,
        retained_component_ids=retained_component_ids,
        removed_component_ids=removed_component_ids,
        final_component_decisions=final_component_decisions,
        final_confidence=final_confidence,
        resolved_by=resolved_by,
        author_attention=author_attention,
        retained_mask=retained_mask,
    )


def _compare_audit_observation(
    actual: Observation,
    recorded: dict[str, Any],
    context: str,
) -> None:
    expected = {
        "physical_instance_id": actual.physical_instance_id,
        "source_label": actual.source_label,
        "visible_pixels": actual.visible_pixels,
        "visible_fraction": actual.visible_fraction,
        "bbox": {
            "x": actual.raw_bbox[0],
            "y": actual.raw_bbox[1],
            "width": actual.raw_bbox[2],
            "height": actual.raw_bbox[3],
        },
        "centroid": {"x": actual.centroid[0], "y": actual.centroid[1]},
        "connected_component_count": actual.connected_component_count,
        "border_touch": actual.border_touch,
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(
                f"Structural-audit observation mismatch at {context}.{key}: "
                f"recorded={recorded.get(key)!r}, actual={value!r}."
            )


def _load_and_verify_frames(
    *,
    selection: StreamSelection,
    audit_sequence: dict[str, Any],
    pairs: Sequence[SourcePair],
    component_decisions: ComponentDecisionLedger,
    consumed_case_ids: set[str],
    secondary_area_ratio: float,
) -> tuple[int, list[FrameData]]:
    stream_id = _stream_id(selection.source_sequence)
    support_label = _integer(audit_sequence.get("support_label"), "audit support_label")
    if audit_sequence.get("object_label_minimum") != support_label + 1:
        raise ValueError(
            f"Structural audit does not use support_label + 1 for {selection.source_sequence}."
        )
    label_policy = _mapping(audit_sequence.get("label_policy"), "audit label_policy")
    if label_policy.get("object_label_rule") != "source_label_greater_than_support_label":
        raise ValueError("Selected audit sequence uses an unsupported object-label rule.")
    if label_policy.get("support_label_detection") != "unique_dominant_nonzero_label_in_first_mask":
        raise ValueError("Selected audit sequence uses an unsupported support-label detector.")
    if label_policy.get("expected_values_are_diagnostics_only") is not True:
        raise ValueError("Expected floor/table labels must remain diagnostic only.")

    if len(pairs) != selection.frames_per_stream:
        raise ValueError(
            f"Raw frame count mismatch for {selection.source_sequence}: "
            f"expected {selection.frames_per_stream}, found {len(pairs)}."
        )
    audit_frames = _list(audit_sequence.get("frames"), "audit sequence frames")
    if len(audit_frames) != len(pairs):
        raise ValueError(f"Structural-audit frame inventory mismatch for {selection.source_sequence}.")
    frame_size = _mapping(audit_sequence.get("frame_size"), "audit frame_size")
    expected_width = _integer(frame_size.get("width"), "audit frame_size.width", minimum=1)
    expected_height = _integer(frame_size.get("height"), "audit frame_size.height", minimum=1)

    frames: list[FrameData] = []
    all_physical_ids: set[str] = set()
    for index, (pair, raw_audit_frame) in enumerate(zip(pairs, audit_frames, strict=True), start=1):
        audit_frame = _mapping(raw_audit_frame, f"audit frame {index}")
        frame_id = f"frame_{index:04d}"
        if audit_frame.get("frame_id") != frame_id or audit_frame.get("frame_index") != index:
            raise ValueError(
                f"Structural-audit frame ID/order mismatch for {selection.source_sequence} "
                f"at position {index}."
            )
        if audit_frame.get("source_filename") != pair.filename:
            raise ValueError(
                f"Structural-audit filename/order mismatch for {selection.source_sequence} "
                f"at position {index}."
            )
        width, height = _rgb_properties(pair.rgb_path)
        labels = _load_label_array(pair.label_path)
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"RGB dimensions differ from the structural audit for "
                f"{selection.source_sequence}/{pair.filename}."
            )
        if labels.shape != (height, width):
            raise ValueError(
                f"RGB/label dimensions do not align for {selection.source_sequence}/"
                f"{pair.filename}: RGB={(width, height)}, label={labels.shape[::-1]}."
            )
        if audit_frame.get("width") != width or audit_frame.get("height") != height:
            raise ValueError(f"Audit frame dimensions mismatch at {selection.source_sequence}/{frame_id}.")
        if index == 1:
            detected = _detect_support_label(
                labels, f"{selection.source_sequence}/label/{pair.filename}"
            )
            if detected != support_label:
                raise ValueError(
                    f"Detected support label {detected} does not match audited support label "
                    f"{support_label} for {selection.source_sequence}."
                )
        _validate_support_label(
            labels,
            support_label,
            f"{selection.source_sequence}/label/{pair.filename}",
        )
        object_labels = sorted(
            int(value) for value in np.unique(labels) if int(value) > support_label
        )
        observations = tuple(
            _derive_observation(
                labels,
                source_label=label,
                stream_id=stream_id,
                selection=selection,
                frame_id=frame_id,
                frame_index=index,
                source_filename=pair.filename,
                ledger=component_decisions,
                consumed_case_ids=consumed_case_ids,
                secondary_area_ratio=secondary_area_ratio,
            )
            for label in object_labels
        )
        recorded_instances = [
            _mapping(item, f"audit frame {frame_id} physical_instances item")
            for item in _list(audit_frame.get("physical_instances"), "physical_instances")
        ]
        if len(recorded_instances) != len(observations):
            raise ValueError(f"Audit object count mismatch at {selection.source_sequence}/{frame_id}.")
        for observation, recorded in zip(observations, recorded_instances, strict=True):
            _compare_audit_observation(
                observation,
                recorded,
                f"{selection.source_sequence}/{frame_id}/{observation.physical_instance_id}",
            )
            all_physical_ids.add(observation.physical_instance_id)
        expected_ids = [item.physical_instance_id for item in observations]
        if audit_frame.get("physical_instance_ids") != expected_ids:
            raise ValueError(f"Audit physical-instance ordering mismatch at {frame_id}.")
        if audit_frame.get("object_count") != len(observations):
            raise ValueError(f"Audit object_count mismatch at {frame_id}.")
        frames.append(
            FrameData(
                frame_id=frame_id,
                frame_index=index,
                source_filename=pair.filename,
                width=width,
                height=height,
                rgb_path=pair.rgb_path,
                labels=labels,
                observations=observations,
            )
        )

    if audit_sequence.get("physical_instance_id_scope") != "sequence_local":
        raise ValueError("Gate B physical-instance proxies must be sequence-local.")
    if audit_sequence.get("paired_camera_identity_linkage") != "not_assessed":
        raise ValueError("Gate B must not claim paired-camera identity linkage.")
    recorded_all = sorted(
        _text(item, "audit physical_instance_ids item")
        for item in _list(audit_sequence.get("physical_instance_ids"), "physical_instance_ids")
    )
    if recorded_all != sorted(all_physical_ids):
        raise ValueError(f"Audit stream instance membership mismatch for {selection.source_sequence}.")
    return support_label, frames


def _analysis_manifest(
    *,
    spec: dict[str, Any],
    selection: StreamSelection,
    stream_id: str,
    frames: Sequence[FrameData],
) -> dict[str, object]:
    source = _mapping(spec.get("source"), "spec.source")
    return {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION,
        "stream_id": stream_id,
        "scene_description": (
            "Real OCID controlled scene with objects added incrementally to a fixed workspace."
        ),
        "ordering": "manifest",
        "frames": [
            {
                "frame_id": frame.frame_id,
                "index": frame.frame_index,
                "image_path": f"frames/{frame.frame_id}.png",
                "metadata": {
                    "ocid_source_filename": frame.source_filename,
                    "ocid_state_position": frame.frame_index,
                },
            }
            for frame in frames
        ],
        "notes": "Prepared OCID RGB input for the frozen Gate B benchmark.",
        "metadata": {
            "benchmark_id": spec["benchmark_id"],
            "source_dataset": "OCID",
            "source_sequence": selection.source_sequence,
            "source_url": source["source_url"],
            "scene_group_id": selection.scene_group_id,
            "role": selection.role,
            "adapter": "tools/prepare_ocid_gate_b_benchmark.py",
            "purpose": "gate_b_analysis_input",
            "frame_size": {"width": frames[0].width, "height": frames[0].height},
            "frame_format": "png_rgb",
            "is_synthetic": False,
        },
    }


def _annotation_payload(
    *,
    selection: StreamSelection,
    stream_id: str,
    support_label: int,
    frames: Sequence[FrameData],
    component_decisions_sha256: str,
) -> dict[str, object]:
    by_id: dict[str, int] = {}
    for frame in frames:
        for observation in frame.observations:
            by_id[observation.physical_instance_id] = observation.source_label
    visual_types = [
        {
            "visual_type_id": physical_id,
            "description": (
                f"Sequence-local OCID physical-instance proxy for source label {source_label}."
            ),
            "notes": (
                "Candidate-extraction grouping proxy only; this is not a ground-truth "
                "visual type and has no cross-camera identity claim."
            ),
            "metadata": {
                "identity_proxy": "sequence_local_physical_instance_id",
                "source_label": source_label,
                "proxy_scope": "candidate_extraction_only",
                "visual_type_claim": "not_assessed",
            },
        }
        for physical_id, source_label in sorted(by_id.items())
    ]
    instances: list[dict[str, object]] = []
    for frame in frames:
        for observation in frame.observations:
            instances.append(
                {
                    "instance_id": f"{observation.physical_instance_id}__{frame.frame_id}",
                    "frame_id": frame.frame_id,
                    "visual_type_id": observation.physical_instance_id,
                    "physical_instance_proxy_id": observation.physical_instance_id,
                    "source_label": observation.source_label,
                    "bbox": {
                        "x": observation.bbox[0],
                        "y": observation.bbox[1],
                        "width": observation.bbox[2],
                        "height": observation.bbox[3],
                    },
                    "characteristic_regions": [],
                    "uncertainty": "",
                    "notes": (
                        "Tight box from the reviewed retained-component union. The visible "
                        "observation remains included regardless of visibility or border contact."
                    ),
                }
            )
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "stream_id": stream_id,
        "manifest_ref": f"../../analysis_streams/{stream_id}/manifest.json",
        "annotation_scope": ANNOTATION_SCOPE,
        "frame_size": {"width": frames[0].width, "height": frames[0].height},
        "visual_types": visual_types,
        "expected_element_instances": instances,
        "frame_comparisons": [],
        "change_events": [],
        "supported_event_types": [],
        "allowed_event_types": [],
        "uncertainty": [],
        "notes": (
            "Candidate-extraction boxes only. Sequence-local physical IDs are evaluator "
            "grouping proxies, not visual-type ground truth."
        ),
        "metadata": {
            "source_dataset": "OCID",
            "source_sequence": selection.source_sequence,
            "scene_group_id": selection.scene_group_id,
            "role": selection.role,
            "audited_support_label": support_label,
            "support_label_detection": "unique_dominant_nonzero_label_in_first_mask",
            "object_label_rule": "source_label_greater_than_audited_support_label",
            "bbox_derivation": COMPONENT_BBOX_DERIVATION,
            "component_review_policy": COMPONENT_REVIEW_POLICY_ID,
            "component_connectivity": 4,
            "secondary_area_ratio": DEFAULT_SECONDARY_AREA_RATIO,
            "component_decisions_sha256": component_decisions_sha256,
            "identity_proxy": "sequence_local_physical_instance_id",
            "physical_instance_id_scope": "sequence_local",
            "paired_camera_identity_linkage": "not_assessed",
            "visual_type_claim": "not_assessed",
            "candidate_only": True,
            "ground_truth_role": "evaluation_only",
            "adapter": "tools/prepare_ocid_gate_b_benchmark.py",
        },
    }


def _color_for_id(identifier: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return tuple(48 + value % 176 for value in digest[:3])


def _draw_text_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = max(0, min(xy[0], max(0, image_size[0] - text_width - 4)))
    y = max(0, min(xy[1], max(0, image_size[1] - text_height - 4)))
    draw.rectangle((x, y, x + text_width + 4, y + text_height + 4), fill=(0, 0, 0))
    draw.text((x + 2, y + 2), text, fill=fill, font=font)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 1,
    dash_length: int = 5,
    gap_length: int = 3,
) -> None:
    horizontal = start[1] == end[1]
    if not horizontal and start[0] != end[0]:
        raise ValueError("Dashed overlay lines must be axis-aligned.")
    fixed = start[1] if horizontal else start[0]
    low = min(start[0], end[0]) if horizontal else min(start[1], end[1])
    high = max(start[0], end[0]) if horizontal else max(start[1], end[1])
    cursor = low
    while cursor <= high:
        segment_end = min(high, cursor + dash_length - 1)
        if horizontal:
            draw.line((cursor, fixed, segment_end, fixed), fill=fill, width=width)
        else:
            draw.line((fixed, cursor, fixed, segment_end), fill=fill, width=width)
        cursor += dash_length + gap_length


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
) -> None:
    x, y, width, height = bbox
    right = x + width - 1
    bottom = y + height - 1
    _draw_dashed_line(draw, (x, y), (right, y), fill=fill)
    _draw_dashed_line(draw, (x, bottom), (right, bottom), fill=fill)
    _draw_dashed_line(draw, (x, y), (x, bottom), fill=fill)
    _draw_dashed_line(draw, (right, y), (right, bottom), fill=fill)


def _render_overlay(frame: FrameData, output_path: Path) -> None:
    with Image.open(frame.rgb_path) as source:
        image = source.convert("RGBA")
    for observation in frame.observations:
        color = _color_for_id(observation.physical_instance_id)
        alpha = Image.fromarray(
            np.where(observation.retained_mask, 88, 0).astype(np.uint8),
            mode="L",
        )
        layer = Image.new("RGBA", image.size, color + (0,))
        layer.putalpha(alpha)
        image = Image.alpha_composite(image, layer)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_text_label(
        draw,
        (5, 5),
        f"{frame.frame_id} | {frame.source_filename}",
        fill=(255, 255, 255),
        image_size=image.size,
        font=font,
    )
    for observation in frame.observations:
        x, y, width, height = observation.bbox
        color = _color_for_id(observation.physical_instance_id)
        raw_changed = observation.raw_bbox != observation.bbox
        if raw_changed:
            raw_color = (0, 255, 255)
            _draw_dashed_rectangle(draw, observation.raw_bbox, fill=raw_color)
            raw_x, raw_y, _, _ = observation.raw_bbox
            raw_label_y = raw_y - 15 if raw_y >= 18 else raw_y + 3
            _draw_text_label(
                draw,
                (raw_x + 2, raw_label_y),
                f"RAW source={observation.source_label}",
                fill=raw_color,
                image_size=image.size,
                font=font,
            )
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=3)
        bbox_label = "FINAL" if raw_changed else "RAW=FINAL"
        label = (
            f"{bbox_label} source={observation.source_label} | "
            f"{observation.physical_instance_id}"
        )
        label_y = y + 15 if raw_changed and y < 18 else (y - 15 if y >= 18 else y + 3)
        _draw_text_label(
            draw,
            (x + 2, label_y),
            label,
            fill=color,
            image_size=image.size,
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG", compress_level=6, optimize=False)


def _render_contact_sheet(
    *,
    stream_id: str,
    frames: Sequence[FrameData],
    overlay_directory: Path,
    output_path: Path,
) -> None:
    tile_width = 320
    tile_height = max(1, round(tile_width * frames[0].height / frames[0].width))
    caption_height = 22
    padding = 6
    columns = min(5, max(1, math.ceil(math.sqrt(len(frames)))))
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (
            padding + columns * (tile_width + padding),
            padding + rows * (tile_height + caption_height + padding),
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, frame in enumerate(frames):
        column = position % columns
        row = position // columns
        x = padding + column * (tile_width + padding)
        y = padding + row * (tile_height + caption_height + padding)
        overlay_path = overlay_directory / f"{frame.frame_id}.png"
        with Image.open(overlay_path) as overlay:
            thumbnail = overlay.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x + 2, y + tile_height + 3),
            f"{frame.frame_id} | n={len(frame.observations)}",
            fill=(255, 255, 255),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=6, optimize=False)


def _semicolon(values: Iterable[str]) -> str:
    return ";".join(sorted(set(values)))


def _review_rows(
    *,
    selection: StreamSelection,
    stream_id: str,
    frames: Sequence[FrameData],
    severe_area_ratio: float,
) -> StreamReview:
    maxima: dict[str, int] = {}
    for frame in frames:
        for observation in frame.observations:
            maxima[observation.physical_instance_id] = max(
                maxima.get(observation.physical_instance_id, 0), observation.visible_pixels
            )

    frame_flag_details: dict[str, dict[str, set[str]]] = {
        frame.frame_id: {} for frame in frames
    }
    observation_flags: dict[tuple[str, str], set[str]] = {}
    outgoing_flags: dict[tuple[str, str], set[str]] = {}

    def add_frame_flag(frame_id: str, flag: str, physical_id: str) -> None:
        frame_flag_details[frame_id].setdefault(flag, set()).add(physical_id)

    for frame in frames:
        for observation in frame.observations:
            key = (frame.frame_id, observation.physical_instance_id)
            flags = observation_flags.setdefault(key, set())
            relative = observation.visible_pixels / maxima[observation.physical_instance_id]
            if relative < 0.10:
                flags.add("relative_visible_area_below_0_10")
                add_frame_flag(frame.frame_id, "relative_visible_area_below_0_10", observation.physical_instance_id)
            if observation.connected_component_count > 1:
                flags.add("connected_component_count_above_1")
                add_frame_flag(frame.frame_id, "connected_component_count_above_1", observation.physical_instance_id)
            if observation.border_touch:
                flags.add("border_touch")
                add_frame_flag(frame.frame_id, "border_touch", observation.physical_instance_id)

    seen_ids: set[str] = set()
    for position, frame in enumerate(frames):
        current = {item.physical_instance_id: item for item in frame.observations}
        if position == 0:
            seen_ids.update(current)
            continue
        previous_frame = frames[position - 1]
        previous = {item.physical_instance_id: item for item in previous_frame.observations}
        disappeared = sorted(set(previous) - set(current))
        returned = sorted((set(current) - set(previous)) & seen_ids)
        for physical_id in disappeared:
            add_frame_flag(frame.frame_id, "visible_to_absent", physical_id)
            outgoing_flags.setdefault((previous_frame.frame_id, physical_id), set()).add(
                "visible_to_absent"
            )
        for physical_id in returned:
            add_frame_flag(frame.frame_id, "returned_visible", physical_id)
            observation_flags.setdefault((frame.frame_id, physical_id), set()).add(
                "returned_visible"
            )
        for physical_id in sorted(set(previous) & set(current)):
            ratio = current[physical_id].visible_pixels / previous[physical_id].visible_pixels
            if ratio <= severe_area_ratio:
                add_frame_flag(
                    frame.frame_id, "severe_interframe_visibility_drop", physical_id
                )
                observation_flags.setdefault((frame.frame_id, physical_id), set()).add(
                    "severe_interframe_visibility_drop"
                )
        seen_ids.update(current)

    mandatory_by_position: dict[int, set[str]] = {
        0: {"mandatory_first"},
        (len(frames) - 1) // 2: {"mandatory_middle"},
        len(frames) - 1: {"mandatory_last"},
    }
    if len(frames) == 1:
        mandatory_by_position[0] = {
            "mandatory_first",
            "mandatory_middle",
            "mandatory_last",
        }
    elif (len(frames) - 1) // 2 == 0:
        mandatory_by_position[0].add("mandatory_middle")

    observation_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    queue_items: list[dict[str, object]] = []
    seen_ids.clear()
    previous_ids: set[str] = set()
    previous_by_id: dict[str, Observation] = {}
    for position, frame in enumerate(frames):
        current_by_id = {item.physical_instance_id: item for item in frame.observations}
        current_ids = set(current_by_id)
        added = sorted((current_ids - previous_ids) - seen_ids)
        returned = sorted((current_ids - previous_ids) & seen_ids)
        disappeared = sorted(previous_ids - current_ids)
        severe = sorted(
            physical_id
            for physical_id in (previous_ids & current_ids)
            if current_by_id[physical_id].visible_pixels
            / previous_by_id[physical_id].visible_pixels
            <= severe_area_ratio
        )
        details = {
            flag: sorted(ids)
            for flag, ids in sorted(frame_flag_details[frame.frame_id].items())
        }
        flags = sorted(details)
        mandatory = sorted(mandatory_by_position.get(position, set()))
        visible_ids = sorted(current_ids)
        frame_row: dict[str, object] = {
            "role": selection.role,
            "scene_group_id": selection.scene_group_id,
            "stream_id": stream_id,
            "source_sequence": selection.source_sequence,
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "source_filename": frame.source_filename,
            "visible_instance_count": len(visible_ids),
            "visible_instance_ids": _semicolon(visible_ids),
            "automatic_flags": _semicolon(flags),
            "automatic_flag_details": json.dumps(
                details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "mandatory_review_reasons": _semicolon(mandatory),
            "review_required": bool(flags or mandatory),
            "author_review_status": "pending",
            "author_review_decision": "",
            "author_review_notes": "",
            "author_reviewed_by": "",
        }
        frame_rows.append(frame_row)
        timeline_rows.append(
            {
                "role": selection.role,
                "scene_group_id": selection.scene_group_id,
                "stream_id": stream_id,
                "source_sequence": selection.source_sequence,
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "source_filename": frame.source_filename,
                "visible_instance_count": len(visible_ids),
                "visible_instance_ids": _semicolon(visible_ids),
                "first_visible_or_added_ids": _semicolon(added),
                "visible_to_absent_ids": _semicolon(disappeared),
                "returned_visible_ids": _semicolon(returned),
                "severe_interframe_visibility_drop_ids": _semicolon(severe),
                "count_delta": len(current_ids) - len(previous_ids) if position else len(current_ids),
                "automatic_flags": _semicolon(flags),
            }
        )
        if flags or mandatory:
            queue_items.append(
                {
                    "role": selection.role,
                    "scene_group_id": selection.scene_group_id,
                    "stream_id": stream_id,
                    "source_sequence": selection.source_sequence,
                    "frame_id": frame.frame_id,
                    "frame_index": frame.frame_index,
                    "source_filename": frame.source_filename,
                    "mandatory_reasons": mandatory,
                    "automatic_flags": flags,
                    "automatic_flag_details": details,
                    "author_review_status": "pending",
                }
            )

        for observation in frame.observations:
            key = (frame.frame_id, observation.physical_instance_id)
            direct = observation_flags.get(key, set())
            outgoing = outgoing_flags.get(key, set())
            all_flags = sorted(direct | outgoing)
            maximum = maxima[observation.physical_instance_id]
            observation_rows.append(
                {
                    "role": selection.role,
                    "scene_group_id": selection.scene_group_id,
                    "stream_id": stream_id,
                    "source_sequence": selection.source_sequence,
                    "frame_id": frame.frame_id,
                    "frame_index": frame.frame_index,
                    "source_filename": frame.source_filename,
                    "physical_instance_id": observation.physical_instance_id,
                    "source_label": observation.source_label,
                    "visible_pixels": observation.visible_pixels,
                    "frame_area_pixels": frame.width * frame.height,
                    "visible_fraction_of_frame": observation.visible_fraction,
                    "stream_instance_max_visible_pixels": maximum,
                    "relative_to_stream_instance_max_area": round(
                        observation.visible_pixels / maximum, 12
                    ),
                    "raw_bbox_x": observation.raw_bbox[0],
                    "raw_bbox_y": observation.raw_bbox[1],
                    "raw_bbox_width": observation.raw_bbox[2],
                    "raw_bbox_height": observation.raw_bbox[3],
                    "bbox_x": observation.bbox[0],
                    "bbox_y": observation.bbox[1],
                    "bbox_width": observation.bbox[2],
                    "bbox_height": observation.bbox[3],
                    "connected_component_count": observation.connected_component_count,
                    "border_touch": observation.border_touch,
                    "source_mask_sha256": observation.source_mask_sha256,
                    "component_review_required": observation.component_review_required,
                    "automatic_retained_component_ids": _semicolon(
                        observation.automatic_retained_component_ids
                    ),
                    "final_retained_component_ids": _semicolon(
                        observation.retained_component_ids
                    ),
                    "removed_component_ids": _semicolon(
                        observation.removed_component_ids
                    ),
                    "final_component_decisions": json.dumps(
                        observation.final_component_decisions,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "final_confidence": observation.final_confidence,
                    "resolved_by": observation.resolved_by,
                    "author_attention": observation.author_attention,
                    "automatic_flags": _semicolon(all_flags),
                    "outgoing_transition_flags": _semicolon(outgoing),
                    "diagnostic_only": bool(all_flags),
                    "retained_in_annotation": True,
                }
            )
        seen_ids.update(current_ids)
        previous_ids = current_ids
        previous_by_id = current_by_id

    return StreamReview(
        observation_rows=tuple(observation_rows),
        frame_rows=tuple(frame_rows),
        timeline_rows=tuple(timeline_rows),
        queue_items=tuple(queue_items),
    )


def _contains_firewall_marker(value: str) -> str | None:
    normalized = value.casefold().replace("-", "_").replace(" ", "_")
    return next((marker for marker in FIREWALL_MARKERS if marker in normalized), None)


def _scan_analysis_json(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise GroundTruthFirewallError(f"Non-string JSON key at {context}.")
            marker = _contains_firewall_marker(key)
            if marker:
                raise GroundTruthFirewallError(
                    f"Analysis-input key contains forbidden marker {marker!r}: {context}.{key}."
                )
            _scan_analysis_json(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_analysis_json(nested, f"{context}[{index}]")
    elif isinstance(value, str):
        marker = _contains_firewall_marker(value)
        if marker:
            raise GroundTruthFirewallError(
                f"Analysis-input value contains forbidden marker {marker!r}: {context}."
            )


def _validate_analysis_firewall(analysis_root: Path) -> None:
    """Recursively prove that analyze inputs contain RGB PNGs and safe manifests only."""

    if not analysis_root.is_dir():
        raise GroundTruthFirewallError(f"Analysis root does not exist: {analysis_root}.")
    stream_directories = sorted(path for path in analysis_root.iterdir() if path.is_dir())
    if not stream_directories:
        raise GroundTruthFirewallError("Analysis root contains no streams.")
    if any(path.is_file() for path in analysis_root.iterdir()):
        raise GroundTruthFirewallError("Analysis root may contain stream directories only.")

    for path in sorted(analysis_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise GroundTruthFirewallError(f"Analysis inputs must not contain symlinks: {path}.")
        relative = path.relative_to(analysis_root)
        for part in relative.parts:
            marker = _contains_firewall_marker(part)
            if marker:
                raise GroundTruthFirewallError(
                    f"Analysis-input path contains forbidden marker {marker!r}: {relative.as_posix()}."
                )

    for stream_directory in stream_directories:
        manifest_path = stream_directory / "manifest.json"
        frames_directory = stream_directory / "frames"
        if not manifest_path.is_file() or not frames_directory.is_dir():
            raise GroundTruthFirewallError(
                f"Analysis stream must contain manifest.json and frames/: {stream_directory}."
            )
        allowed_files = {manifest_path.resolve()}
        frame_files = sorted(frames_directory.glob("*.png"))
        if not frame_files:
            raise GroundTruthFirewallError(f"Analysis stream has no RGB frames: {stream_directory}.")
        allowed_files.update(path.resolve() for path in frame_files)
        actual_files = {path.resolve() for path in stream_directory.rglob("*") if path.is_file()}
        if actual_files != allowed_files:
            raise GroundTruthFirewallError(
                f"Analysis stream contains a file outside its manifest/RGB frame contract: "
                f"{stream_directory}."
            )
        manifest, _ = _load_json_object(manifest_path, "analysis manifest")
        if manifest.get("schema_version") != ANALYSIS_MANIFEST_SCHEMA_VERSION:
            raise GroundTruthFirewallError(f"Unsupported analysis manifest: {manifest_path}.")
        _scan_analysis_json(manifest, manifest_path.as_posix())
        frames = _list(manifest.get("frames"), "analysis manifest frames")
        expected_paths = []
        for position, raw_frame in enumerate(frames, start=1):
            frame = _mapping(raw_frame, f"analysis frame {position}")
            frame_id = _text(frame.get("frame_id"), "frame_id")
            image_path = _text(frame.get("image_path"), "image_path")
            expected_image_path = f"frames/{frame_id}.png"
            if image_path != expected_image_path:
                raise GroundTruthFirewallError(
                    "Analysis manifest image_path must be the exact in-stream RGB path "
                    f"{expected_image_path!r}, got {image_path!r}."
                )
            expected_paths.append(image_path)
        actual_paths = [
            path.relative_to(stream_directory).as_posix() for path in frame_files
        ]
        if expected_paths != actual_paths:
            raise GroundTruthFirewallError(
                f"Analysis manifest/frame inventory mismatch: {stream_directory}."
            )
        for frame_path in frame_files:
            _rgb_properties(frame_path)


def _event_support_payload(spec: dict[str, Any]) -> dict[str, object]:
    support = _mapping(spec.get("event_support"), "spec.event_support")
    rows = [
        {
            "signal": key,
            "support_status": support[key],
            "primary_candidate_bbox_metric": False,
        }
        for key in (
            "first_visible_or_added",
            "persisted_visible",
            "count_changed_visible",
            "visibility_reduced",
            "visibility_lost",
            "physical_removed",
            "returned_after_physical_removal",
            "object_motion",
        )
    ]
    return {
        "schema_version": "ocid-gate-b-event-support-matrix-1.0",
        "benchmark_id": spec["benchmark_id"],
        "annotation_scope": ANNOTATION_SCOPE,
        "primary_evaluation": "candidate_extraction_bbox_only",
        "rows": rows,
        "interpretation": support["interpretation"],
    }


def _heldout_access_payload(spec: dict[str, Any], heldout_stream_count: int) -> dict[str, object]:
    lock = _mapping(spec.get("heldout_lock"), "spec.heldout_lock")
    return {
        "schema_version": "ocid-gate-b-heldout-access-log-1.0",
        "benchmark_id": spec["benchmark_id"],
        "predictions_unlocked": False,
        "heldout_stream_count": heldout_stream_count,
        "entries": [
            {
                "operation_index": 1,
                "operation": "annotation_generation",
                "scope": "heldout_evaluation_annotations",
                "prediction_accessed": False,
                "status": "prepared_before_unlock",
            },
            {
                "operation_index": 2,
                "operation": "ground_truth_preview",
                "scope": "heldout_review_overlays_and_contact_sheets",
                "prediction_accessed": False,
                "status": "prepared_before_unlock",
            },
        ],
        "allowed_before_unlock": list(lock["allowed_before_unlock"]),
        "forbidden_before_unlock": list(lock["forbidden_before_unlock"]),
        "statement": (
            "This deterministic preparation log records annotation and preview preparation "
            "only. No held-out prediction, prediction overlay, metric, model selection, "
            "threshold selection, or postprocessing selection was performed."
        ),
    }


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _artifact_manifest(root: Path, benchmark_id: str) -> dict[str, object]:
    artifact_path = root / "artifact_manifest.json"
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != artifact_path.resolve()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records = [_artifact_record(path, root) for path in files]
    return {
        "schema_version": "ocid-gate-b-artifact-manifest-1.0",
        "benchmark_id": benchmark_id,
        "artifact_count": len(records),
        "artifacts": records,
        "self_excluded": "artifact_manifest.json",
    }


def build_gate_b_benchmark(
    *,
    ocid_root: Path,
    structural_audit: Path,
    spec_path: Path,
    component_decisions: Path,
    output_root: Path,
) -> BuildResult:
    """Verify inputs and atomically publish one deterministic Gate B benchmark."""

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Gate B output root already exists: {destination}.")
    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")
    audit_path = Path(structural_audit).resolve(strict=True)
    spec_file = Path(spec_path).resolve(strict=True)
    if not audit_path.is_file() or not spec_file.is_file():
        raise FileNotFoundError("Structural audit and spec must both be files.")

    spec, spec_bytes = _load_json_object(spec_file, "Gate B spec")
    selections, expected_totals = _validate_spec(spec)
    annotation_contract = _mapping(
        spec.get("annotation_contract"), "spec.annotation_contract"
    )
    secondary_area_ratio = float(annotation_contract["secondary_area_ratio"])
    expected_audit_sha = _digest(
        _mapping(spec.get("source"), "spec.source").get("structural_audit_sha256"),
        "spec.source.structural_audit_sha256",
    )
    actual_audit_sha = _sha256_path(audit_path)
    if actual_audit_sha != expected_audit_sha:
        raise ValueError(
            "Structural audit SHA-256 mismatch: "
            f"spec={expected_audit_sha}, actual={actual_audit_sha}."
        )
    audit, _ = _load_json_object(audit_path, "Gate A structural audit")
    audit_sequences, audit_groups = _audit_sequence_maps(audit)
    provenance = _mapping(audit.get("provenance"), "audit.provenance")
    recorded_source_fingerprint = _mapping(
        provenance.get("source_fingerprint"), "audit.provenance.source_fingerprint"
    )
    audit_source_sha = _digest(
        recorded_source_fingerprint.get("sha256"),
        "audit.provenance.source_fingerprint.sha256",
    )
    spec_source_sha = _digest(
        _mapping(spec.get("source"), "spec.source").get("source_fingerprint_sha256"),
        "spec.source.source_fingerprint_sha256",
    )
    if audit_source_sha != spec_source_sha:
        raise ValueError(
            "Spec source fingerprint does not match the structural audit: "
            f"spec={spec_source_sha}, audit={audit_source_sha}."
        )
    _validate_selected_membership(
        selections=selections,
        audit_sequences=audit_sequences,
        audit_groups=audit_groups,
    )
    pairs_by_source = _verify_complete_audit_source(
        root=root,
        audit=audit,
        sequences=audit_sequences,
    )
    decision_ledger = _load_component_decision_ledger(
        component_decisions,
        spec=spec,
    )

    severe_area_ratio_raw = _mapping(audit.get("parameters"), "audit.parameters").get(
        "severe_area_ratio"
    )
    if isinstance(severe_area_ratio_raw, bool) or not isinstance(
        severe_area_ratio_raw, (int, float)
    ):
        raise ValueError("audit.parameters.severe_area_ratio must be numeric.")
    severe_area_ratio = float(severe_area_ratio_raw)
    if not 0.0 < severe_area_ratio <= 1.0:
        raise ValueError("audit.parameters.severe_area_ratio must be in (0, 1].")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    observation_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    queue_items: list[dict[str, object]] = []
    manifest_streams: list[dict[str, object]] = []
    consumed_component_case_ids: set[str] = set()
    try:
        analysis_root = staging / "analysis_streams"
        analysis_root.mkdir()
        copied_decisions_path = staging / "review" / "component_review_decisions.json"
        copied_decisions_path.parent.mkdir(parents=True)
        copied_decisions_path.write_bytes(decision_ledger.raw_bytes)
        for selection in selections:
            stream_id = _stream_id(selection.source_sequence)
            audit_sequence = audit_sequences[selection.source_sequence]
            support_label, frames = _load_and_verify_frames(
                selection=selection,
                audit_sequence=audit_sequence,
                pairs=pairs_by_source[selection.source_sequence],
                component_decisions=decision_ledger,
                consumed_case_ids=consumed_component_case_ids,
                secondary_area_ratio=secondary_area_ratio,
            )

            stream_root = analysis_root / stream_id
            frames_root = stream_root / "frames"
            frames_root.mkdir(parents=True)
            for frame in frames:
                shutil.copyfile(frame.rgb_path, frames_root / f"{frame.frame_id}.png")
            _write_json(
                stream_root / "manifest.json",
                _analysis_manifest(
                    spec=spec,
                    selection=selection,
                    stream_id=stream_id,
                    frames=frames,
                ),
            )

            annotation_path = staging / "evaluation_annotations" / stream_id / "annotation.json"
            annotation_payload = _annotation_payload(
                selection=selection,
                stream_id=stream_id,
                support_label=support_label,
                frames=frames,
                component_decisions_sha256=decision_ledger.sha256,
            )
            _write_json(annotation_path, annotation_payload)

            overlay_root = staging / "review" / "overlays" / stream_id
            for frame in frames:
                _render_overlay(frame, overlay_root / f"{frame.frame_id}.png")
            contact_sheet_path = staging / "review" / "contact_sheets" / f"{stream_id}.png"
            _render_contact_sheet(
                stream_id=stream_id,
                frames=frames,
                overlay_directory=overlay_root,
                output_path=contact_sheet_path,
            )

            review = _review_rows(
                selection=selection,
                stream_id=stream_id,
                frames=frames,
                severe_area_ratio=severe_area_ratio,
            )
            observation_rows.extend(review.observation_rows)
            frame_rows.extend(review.frame_rows)
            queue_items.extend(review.queue_items)
            timeline_path = staging / "review" / "timelines" / f"{stream_id}.csv"
            _write_csv(timeline_path, TIMELINE_COLUMNS, review.timeline_rows)

            manifest_streams.append(
                {
                    "role": selection.role,
                    "scene_group_id": selection.scene_group_id,
                    "scene_group_key": selection.scene_group_key,
                    "selection_reason": selection.reason,
                    "stream_id": stream_id,
                    "source_sequence": selection.source_sequence,
                    "frame_count": len(frames),
                    "frame_size": {"width": frames[0].width, "height": frames[0].height},
                    "analysis_manifest": f"analysis_streams/{stream_id}/manifest.json",
                    "evaluation_annotation": (
                        f"evaluation_annotations/{stream_id}/annotation.json"
                    ),
                    "review_overlays": f"review/overlays/{stream_id}",
                    "review_contact_sheet": f"review/contact_sheets/{stream_id}.png",
                    "review_timeline": f"review/timelines/{stream_id}.csv",
                    "audited_support_label": support_label,
                    "physical_instance_id_scope": "sequence_local",
                    "visual_type_claim": "not_assessed",
                    "component_review_policy": COMPONENT_REVIEW_POLICY_ID,
                }
            )

        unconsumed_decisions = sorted(
            set(decision_ledger.decisions_by_case_id) - consumed_component_case_ids
        )
        if unconsumed_decisions:
            raise ValueError(
                "Component decisions contain extra cases not required by current analysis: "
                + ", ".join(unconsumed_decisions)
            )
        expected_component_reviews = _integer(
            _mapping(spec.get("review_contract"), "spec.review_contract").get(
                "component_review_expected_observations"
            ),
            "spec.review_contract.component_review_expected_observations",
        )
        if len(consumed_component_case_ids) != expected_component_reviews:
            raise ValueError(
                "Current component-review observation count does not match the frozen spec: "
                f"expected {expected_component_reviews}, found "
                f"{len(consumed_component_case_ids)}."
            )

        _write_csv(
            staging / "review" / "observation_ledger.csv",
            OBSERVATION_COLUMNS,
            observation_rows,
        )
        _write_csv(
            staging / "review" / "frame_review_ledger.csv",
            FRAME_REVIEW_COLUMNS,
            frame_rows,
        )
        _write_json(
            staging / "review" / "review_queue.json",
            {
                "schema_version": "ocid-gate-b-review-queue-1.0",
                "benchmark_id": spec["benchmark_id"],
                "author_review_status": "pending",
                "mandatory_sampling": {
                    "positions": ["first", "middle", "last"],
                    "middle_definition": "one_based_ceiling_half",
                },
                "automatic_flags": list(AUTOMATIC_FLAGS),
                "queue_count": len(queue_items),
                "frames": queue_items,
                "retention_policy": (
                    "Flags are diagnostics only; no frame or visible observation is excluded."
                ),
            },
        )
        _write_json(staging / "event_support_matrix.json", _event_support_payload(spec))
        _write_json(
            staging / "heldout_access_log.json",
            _heldout_access_payload(
                spec,
                sum(selection.role == "heldout" for selection in selections),
            ),
        )

        manifest_payload: dict[str, object] = {
            "schema_version": "ocid-gate-b-benchmark-manifest-1.0",
            "benchmark_id": spec["benchmark_id"],
            "status": "prepared",
            "purpose": spec["purpose"],
            "provenance": {
                "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                "structural_audit_sha256": actual_audit_sha,
                "source_fingerprint_sha256": audit_source_sha,
                "component_decisions_sha256": decision_ledger.sha256,
                "component_decisions_artifact": (
                    "review/component_review_decisions.json"
                ),
                "adapter": "tools/prepare_ocid_gate_b_benchmark.py",
            },
            "totals": {
                **expected_totals,
                "visible_instance_observations": len(observation_rows),
                "review_queue_frames": len(queue_items),
            },
            "ground_truth_boundary": {
                **_mapping(spec.get("ground_truth_boundary"), "ground_truth_boundary"),
                "firewall_validation": "passed_recursive_analysis_stream_scan",
            },
            "annotation_contract": _mapping(
                spec.get("annotation_contract"), "annotation_contract"
            ),
            "component_review": {
                "policy_id": COMPONENT_REVIEW_POLICY_ID,
                "review_status": "complete",
                "reviewed_observation_count": len(consumed_component_case_ids),
                "decisions_sha256": decision_ledger.sha256,
                "decisions_artifact": "review/component_review_decisions.json",
                "final_boxes_from_reviewed_retained_component_union": True,
            },
            "review_contract": {
                **_mapping(spec.get("review_contract"), "review_contract"),
                "relative_visible_area_threshold": 0.10,
                "severe_interframe_visibility_drop_ratio": severe_area_ratio,
                "flags_are_exclusions": False,
            },
            "heldout_lock": _mapping(spec.get("heldout_lock"), "heldout_lock"),
            "streams": manifest_streams,
        }
        _write_json(staging / "benchmark_manifest.json", manifest_payload)

        if len(manifest_streams) != expected_totals["benchmark_streams"]:
            raise AssertionError("Generated stream count differs from the validated spec.")
        if len(frame_rows) != expected_totals["benchmark_frames"]:
            raise AssertionError("Generated frame-review count differs from the validated spec.")
        _validate_analysis_firewall(analysis_root)
        _write_json(
            staging / "artifact_manifest.json",
            _artifact_manifest(staging, str(spec["benchmark_id"])),
        )
        if destination.exists():
            raise FileExistsError(
                f"Gate B output root appeared during generation: {destination}."
            )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return BuildResult(
        output_root=destination,
        scene_group_count=expected_totals["benchmark_scene_groups"],
        stream_count=expected_totals["benchmark_streams"],
        frame_count=expected_totals["benchmark_frames"],
        observation_count=len(observation_rows),
        review_queue_count=len(queue_items),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--component-decisions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_gate_b_benchmark(
        ocid_root=args.ocid_root,
        structural_audit=args.structural_audit,
        spec_path=args.spec,
        component_decisions=args.component_decisions,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_root": str(result.output_root),
                "scene_group_count": result.scene_group_count,
                "stream_count": result.stream_count,
                "frame_count": result.frame_count,
                "observation_count": result.observation_count,
                "review_queue_count": result.review_queue_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
