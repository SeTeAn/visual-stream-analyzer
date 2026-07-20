"""Development-only inventory and aggregation contract for OCID Gate C1.

This module deliberately knows nothing about providers or models.  A provider
may supply candidates only after :func:`load_development_inventory` has made
the frozen development inventory explicit.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stream_analysis.evaluation import (
    PredictedCandidate,
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
)
from stream_analysis.evaluation.annotation import load_annotation


IOU_GRID: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_SPEC = (
    REPOSITORY_ROOT / "data" / "ocid" / "benchmark" / "ocid_candidate_benchmark_v1.json"
)
DEFAULT_REVIEWED_ROOT = (
    REPOSITORY_ROOT / "data" / "ocid" / "derived" / "ocid_candidate_benchmark_v1_reviewed"
)


class GateC1ContractError(ValueError):
    """Raised when a Gate C1 input violates the frozen development contract."""


@dataclass(frozen=True, slots=True)
class DevelopmentStream:
    stream_id: str
    scene_group_id: str
    frame_count: int
    stream_directory: Path
    annotation_path: Path


def load_development_inventory(spec_path: Path, reviewed_root: Path) -> tuple[DevelopmentStream, ...]:
    """Validate the frozen benchmark and return its development streams only.

    Manifest/annotation paths are derived rather than accepted from callers so
    a held-out path cannot be substituted accidentally.
    """

    spec = _json_object(Path(spec_path), "benchmark spec")
    root = Path(reviewed_root).resolve()
    if not root.is_dir():
        raise GateC1ContractError(f"reviewed root does not exist: {root}")
    _validate_lock(spec)
    roles = _object(spec, "roles", "benchmark spec")
    development = _groups(roles, "development")
    heldout = _groups(roles, "heldout")
    expected = _object(spec, "expected_totals", "benchmark spec")

    development_members = _validate_groups(development, "development")
    heldout_members = _validate_groups(heldout, "heldout")
    if set(development_members).intersection(heldout_members):
        raise GateC1ContractError("a stream appears in both development and held-out roles")
    _validate_totals(expected, development, heldout)

    streams: list[DevelopmentStream] = []
    for stream_name, scene_group_id, frame_count in development_members:
        stream_id = _stream_id(stream_name)
        if stream_id in {_stream_id(item[0]) for item in heldout_members}:
            raise GateC1ContractError(f"development stream resolves to held-out id: {stream_id}")
        stream_directory = _under(root, root / "analysis_streams" / stream_id, "analysis stream")
        annotation_path = _under(root, root / "evaluation_annotations" / stream_id / "annotation.json", "annotation")
        manifest = _json_object(stream_directory / "manifest.json", f"manifest for {stream_id}")
        metadata = _object(manifest, "metadata", f"manifest for {stream_id}")
        if manifest.get("stream_id") != stream_id:
            raise GateC1ContractError(f"manifest stream_id mismatch for {stream_id}")
        if metadata.get("role") != "development":
            raise GateC1ContractError(f"manifest role mismatch for {stream_id}")
        if metadata.get("scene_group_id") != scene_group_id:
            raise GateC1ContractError(f"manifest scene group mismatch for {stream_id}")
        frames = _list(manifest, "frames", f"manifest for {stream_id}")
        if len(frames) != frame_count:
            raise GateC1ContractError(f"frame count mismatch for {stream_id}: expected {frame_count}, got {len(frames)}")
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("image_path"), str):
                raise GateC1ContractError(f"malformed frame entry in {stream_id}")
            _under(stream_directory, stream_directory / frame["image_path"], f"frame path in {stream_id}")
        if not annotation_path.is_file():
            raise GateC1ContractError(f"missing annotation for {stream_id}")
        streams.append(DevelopmentStream(stream_id, scene_group_id, frame_count, stream_directory, annotation_path))
    return tuple(sorted(streams, key=lambda item: item.stream_id))


def validate_development_input_pairs(
    stream_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    *,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> tuple[DevelopmentStream, ...]:
    """Require the exact canonical Gate C1 development stream/annotation pairs.

    Provider tools call this before loading a model or reading a supplied
    annotation. Subsets, duplicates, held-out paths, path/name lookalikes and
    cross-paired annotations are rejected.
    """

    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise GateC1ContractError("supply one annotation for every stream")
    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    expected = {
        (
            stream.stream_directory.resolve(strict=True),
            stream.annotation_path.resolve(strict=True),
        )
        for stream in inventory
    }
    supplied: list[tuple[Path, Path]] = []
    try:
        for stream_directory, annotation_path in zip(
            stream_directories, annotation_paths, strict=True
        ):
            supplied.append(
                (
                    Path(stream_directory).resolve(strict=True),
                    Path(annotation_path).resolve(strict=True),
                )
            )
    except OSError as error:
        raise GateC1ContractError(f"cannot resolve supplied development input: {error}") from error
    supplied_set = set(supplied)
    if len(supplied) != len(supplied_set):
        raise GateC1ContractError("development stream/annotation pairs must be unique")
    if supplied_set != expected:
        missing = sorted(stream.name for stream, _annotation in expected - supplied_set)
        extra = sorted(stream.name for stream, _annotation in supplied_set - expected)
        raise GateC1ContractError(
            "Gate C1 requires exactly the canonical development inputs; "
            f"missing_streams={missing}, rejected_streams={extra}"
        )
    return inventory


def evaluate_profiles(
    inventory: tuple[DevelopmentStream, ...],
    predictions_by_profile: Mapping[str, Mapping[str, tuple[PredictedCandidate, ...]]],
) -> dict[str, Any]:
    """Evaluate supplied development predictions at every fixed IoU threshold.

    Ground truth is loaded only here, after profile/stream keys have passed the
    development-only contract.  The result contains micro-pooled metrics and
    macro summaries over scene groups, preventing paired cameras from being
    presented as independent scenes.
    """

    streams = tuple(sorted(inventory, key=lambda stream: stream.stream_id))
    _validate_inventory_argument(streams)
    if not isinstance(predictions_by_profile, Mapping) or not predictions_by_profile:
        raise GateC1ContractError("predictions_by_profile must contain at least one profile")
    expected_ids = {stream.stream_id for stream in streams}
    annotations = {
        stream.stream_id: load_annotation(
            stream.annotation_path,
            manifest_path=stream.stream_directory / "manifest.json",
        )
        for stream in streams
    }
    profiles: dict[str, Any] = {}
    for profile_id in sorted(predictions_by_profile):
        if not isinstance(profile_id, str) or not profile_id:
            raise GateC1ContractError("profile IDs must be non-empty strings")
        supplied = predictions_by_profile[profile_id]
        if not isinstance(supplied, Mapping) or set(supplied) != expected_ids:
            raise GateC1ContractError(f"profile {profile_id!r} must supply exactly the development streams")
        per_stream: dict[str, Any] = {}
        for stream in streams:
            predicted = tuple(supplied[stream.stream_id])
            if not all(isinstance(item, PredictedCandidate) for item in predicted):
                raise GateC1ContractError(f"profile {profile_id!r} has invalid candidates for {stream.stream_id}")
            metrics = {
                _threshold_key(threshold): evaluate_candidate_predictions(annotations[stream.stream_id], predicted, iou_threshold=threshold)
                for threshold in IOU_GRID
            }
            per_stream[stream.stream_id] = {
                "scene_group_id": stream.scene_group_id,
                "frame_count": stream.frame_count,
                "metrics_by_iou": metrics,
            }
        profiles[profile_id] = {"per_stream": per_stream, "summary_by_iou": _summaries(streams, per_stream)}
    return {"schema_version": "ocid-gate-c1-development-results-1.0", "iou_grid": list(IOU_GRID), "profiles": profiles}


def write_canonical_json_atomic(path: Path, payload: Any) -> Path:
    """Atomically write canonical JSON; the sole file-writing helper here."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _summaries(streams: tuple[DevelopmentStream, ...], per_stream: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = {stream.scene_group_id: [] for stream in streams}
    for stream in streams:
        groups[stream.scene_group_id].append(stream.stream_id)
    for threshold in IOU_GRID:
        key = _threshold_key(threshold)
        all_metrics = [per_stream[stream.stream_id]["metrics_by_iou"][key] for stream in streams]
        scene_groups = {
            group_id: {
                "stream_ids": sorted(stream_ids),
                "metrics": aggregate_candidate_metrics(tuple(per_stream[stream_id]["metrics_by_iou"][key] for stream_id in sorted(stream_ids))),
            }
            for group_id, stream_ids in sorted(groups.items())
        }
        group_metrics = [item["metrics"] for item in scene_groups.values()]
        macro = {
            metric: _mean([_metric_for_macro(item, metric) for item in group_metrics])
            for metric in ("precision", "recall", "f1", "false_per_frame", "weighted_mean_tp_iou")
        }
        worst_id = min(
            scene_groups,
            key=lambda group_id: (_selection_f1(scene_groups[group_id]["metrics"]), group_id),
        )
        result[key] = {
            "pooled": aggregate_candidate_metrics(tuple(all_metrics)),
            "scene_groups": scene_groups,
            "scene_group_macro": macro,
            "worst_scene_group": {"scene_group_id": worst_id, "metrics": scene_groups[worst_id]["metrics"]},
        }
    return result


def _validate_lock(spec: Mapping[str, Any]) -> None:
    lock = _object(spec, "heldout_lock", "benchmark spec")
    if lock.get("predictions_unlocked") is not False:
        raise GateC1ContractError("held-out predictions must remain locked for Gate C1")
    forbidden = lock.get("forbidden_before_unlock")
    required = {"model_prediction", "metric_computation", "model_selection", "threshold_selection", "postprocessing_selection"}
    if not isinstance(forbidden, list) or not required.issubset(set(forbidden)):
        raise GateC1ContractError("held-out lock policy is malformed or incomplete")


def _validate_groups(groups: list[Any], role: str) -> list[tuple[str, str, int]]:
    values: list[tuple[str, str, int]] = []
    group_ids: set[str] = set()
    stream_names: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise GateC1ContractError(f"{role} group is not an object")
        group_id = group.get("scene_group_id")
        count = group.get("frames_per_stream")
        members = group.get("member_streams")
        if not isinstance(group_id, str) or not group_id or group_id in group_ids:
            raise GateC1ContractError(f"duplicate or missing {role} scene_group_id")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or not isinstance(members, list) or not members:
            raise GateC1ContractError(f"malformed {role} group {group_id!r}")
        group_ids.add(group_id)
        for member in members:
            if not isinstance(member, str) or not member or member in stream_names:
                raise GateC1ContractError(f"duplicate or missing {role} stream")
            stream_names.add(member)
            values.append((member, group_id, count))
    return values


def _validate_totals(expected: Mapping[str, Any], development: list[Any], heldout: list[Any]) -> None:
    checks = {
        "development_scene_groups": len(development), "heldout_scene_groups": len(heldout),
        "development_streams": sum(len(group["member_streams"]) for group in development),
        "heldout_streams": sum(len(group["member_streams"]) for group in heldout),
        "development_frames": sum(group["frames_per_stream"] * len(group["member_streams"]) for group in development),
        "heldout_frames": sum(group["frames_per_stream"] * len(group["member_streams"]) for group in heldout),
    }
    checks["benchmark_scene_groups"] = checks["development_scene_groups"] + checks["heldout_scene_groups"]
    checks["benchmark_streams"] = checks["development_streams"] + checks["heldout_streams"]
    checks["benchmark_frames"] = checks["development_frames"] + checks["heldout_frames"]
    for key, actual in checks.items():
        if expected.get(key) != actual:
            raise GateC1ContractError(f"expected_totals.{key} mismatch: expected {expected.get(key)!r}, got {actual}")


def _validate_inventory_argument(streams: tuple[DevelopmentStream, ...]) -> None:
    if any(not isinstance(stream, DevelopmentStream) for stream in streams):
        raise GateC1ContractError("inventory must contain DevelopmentStream values")
    if not streams or len({stream.stream_id for stream in streams}) != len(streams):
        raise GateC1ContractError("inventory must contain unique development streams")


def _stream_id(source_stream: str) -> str:
    return "ocid_" + source_stream.lower().replace("/", "_")


def _under(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise GateC1ContractError(f"{label} escapes reviewed root: {path}") from error
    return resolved


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateC1ContractError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise GateC1ContractError(f"{label} must be a JSON object")
    return value


def _object(value: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise GateC1ContractError(f"{label}.{key} must be an object")
    return result


def _groups(roles: Mapping[str, Any], role: str) -> list[Any]:
    value = roles.get(role)
    if not isinstance(value, list) or not value:
        raise GateC1ContractError(f"roles.{role} must be a non-empty list")
    return value


def _list(value: Mapping[str, Any], key: str, label: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise GateC1ContractError(f"{label}.{key} must be a list")
    return result


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return None if not defined else sum(defined) / len(defined)


def _metric_for_macro(metrics: Mapping[str, Any], field: str) -> float | None:
    value = metrics.get(field)
    if value is not None:
        return float(value)
    if field in {"precision", "f1"} and int(metrics.get("fn", 0)) > 0:
        return 0.0
    return None


def _selection_f1(metrics: Mapping[str, Any]) -> float:
    value = _metric_for_macro(metrics, "f1")
    return 0.0 if value is None else value
