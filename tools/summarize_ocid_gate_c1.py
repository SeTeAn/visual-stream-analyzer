"""Summarise completed OCID Gate C1 development reports.

The provider reports contain development-only, already computed candidate
metrics.  This tool does not load images, annotations, predictions, or model
assets.  It normalises their four deliberately different report layouts and
applies the pre-registered selection rule from EXP-OCID-008.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Support direct execution from ``tools`` and module execution.
    from tools.ocid_gate_c1_common import (
        IOU_GRID,
        DevelopmentStream,
        GateC1ContractError,
        load_development_inventory,
        write_canonical_json_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback.
    from ocid_gate_c1_common import (  # type: ignore[no-redef]
        IOU_GRID,
        DevelopmentStream,
        GateC1ContractError,
        load_development_inventory,
        write_canonical_json_atomic,
    )

from stream_analysis.evaluation import aggregate_candidate_metrics


SCHEMA_VERSION = "ocid-gate-c1-development-summary-1.0"
PRIMARY_TOLERANCE = 0.02
_IOU_KEYS = tuple(f"{value:.2f}" for value in IOU_GRID)
_MACRO_FIELDS = ("precision", "recall", "f1", "false_per_frame", "weighted_mean_tp_iou")


def summarize_reports(
    *,
    provider_ids: Sequence[str],
    run_ids: Sequence[str],
    report_paths: Sequence[Path],
    benchmark_spec: Path,
    reviewed_root: Path,
) -> dict[str, Any]:
    """Validate and compare completed reports over the frozen development set."""

    _validate_aligned_arguments(provider_ids, run_ids, report_paths)
    inventory = load_development_inventory(Path(benchmark_spec), Path(reviewed_root))
    expected_ids = {item.stream_id for item in inventory}
    runs: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for provider_id, run_id, report_path in zip(provider_ids, run_ids, report_paths, strict=True):
        canonical_report_path = Path(report_path).resolve(strict=True)
        report = _read_json_object(canonical_report_path, f"report for run {run_id!r}")
        if report.get("status") != "completed":
            raise GateC1ContractError(f"report for run {run_id!r} is not completed")
        profiles = _normalise_profiles(report, expected_ids, run_id)
        resources = _extract_resources(report, expected_ids)
        summaries = {
            profile_id: _summarise_profile(inventory, metrics_by_stream)
            for profile_id, metrics_by_stream in sorted(profiles.items())
        }
        for profile_id in summaries:
            candidate_id = _candidate_config_id(run_id, profile_id)
            if candidate_id in candidate_ids:
                raise GateC1ContractError(f"duplicate candidate config ID: {candidate_id}")
            candidate_ids.add(candidate_id)
        runs.append(
            {
                "provider_id": provider_id,
                "run_id": run_id,
                "report_path": canonical_report_path.as_posix(),
                "report_sha256": _sha256(canonical_report_path),
                "resources": resources,
                "profiles": {
                    profile_id: {
                        "candidate_config_id": _candidate_config_id(run_id, profile_id),
                        "summary_by_iou": summary,
                    }
                    for profile_id, summary in summaries.items()
                },
            }
        )

    provider_selections = _select_by_provider(runs)
    recommendation = _select_overall(provider_selections)
    repository_root = Path(__file__).resolve().parents[1]
    canonical_spec = Path(benchmark_spec).resolve(strict=True)
    canonical_reviewed_root = Path(reviewed_root).resolve(strict=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "development_only_existing_report_summarisation",
        "heldout_access": "none",
        "iou_grid": list(IOU_GRID),
        "inventory": {
            "stream_ids": [item.stream_id for item in inventory],
            "scene_group_ids": sorted({item.scene_group_id for item in inventory}),
            "stream_count": len(inventory),
            "scene_group_count": len({item.scene_group_id for item in inventory}),
            "frame_count": sum(item.frame_count for item in inventory),
        },
        "provenance": {
            "benchmark_spec": _file_provenance(canonical_spec),
            "reviewed_root": canonical_reviewed_root.as_posix(),
            "reviewed_artifact_manifest": _file_provenance(
                canonical_reviewed_root / "artifact_manifest.json"
            ),
            "implementation_files": [
                _file_provenance(repository_root / relative_path)
                for relative_path in (
                    "src/stream_analysis/evaluation/runner.py",
                    "tools/ocid_gate_c1_common.py",
                    "tools/evaluate_ocid_maskrcnn_extractor.py",
                    "tools/evaluate_ocid_maskrcnn_nms_postprocess.py",
                    "tools/evaluate_ocid_mobilesam_extractor.py",
                    "tools/evaluate_ocid_sam2_extractor.py",
                    "tools/evaluate_ocid_grounding_dino_extractor.py",
                    "tools/summarize_ocid_gate_c1.py",
                )
            ],
        },
        "selection_rule": {
            "primary": "maximum scene-group macro F1 at IoU 0.50",
            "tie_tolerance_absolute_f1": PRIMARY_TOLERANCE,
            "tie_break_order": [
                "higher worst-scene F1 at IoU 0.50",
                "higher scene-group macro recall at IoU 0.50",
                "higher scene-group macro F1 at IoU 0.70",
                "lower runtime seconds per frame",
                "lower peak CUDA allocated bytes",
                "lexicographically smaller candidate config ID",
            ],
            "resource_missing_policy": "missing resource fields are nullable and rank after reported values",
        },
        "runs": runs,
        "provider_selections": provider_selections,
        "recommendation": recommendation,
    }


def _validate_aligned_arguments(
    provider_ids: Sequence[str], run_ids: Sequence[str], report_paths: Sequence[Path]
) -> None:
    if not provider_ids or len(provider_ids) != len(run_ids) or len(run_ids) != len(report_paths):
        raise GateC1ContractError("--provider-id, --run-id, and --report must be non-empty aligned lists")
    if len(set(run_ids)) != len(run_ids):
        raise GateC1ContractError("run_id values must be unique")
    for provider_id, run_id in zip(provider_ids, run_ids, strict=True):
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise GateC1ContractError("provider_id values must be non-empty strings")
        if not isinstance(run_id, str) or not run_id.strip():
            raise GateC1ContractError("run_id values must be non-empty strings")


def _normalise_profiles(
    report: Mapping[str, Any], expected_ids: set[str], run_id: str
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    streams = report.get("streams")
    if not isinstance(streams, dict) or set(streams) != expected_ids:
        missing, extra = _key_difference(expected_ids, streams if isinstance(streams, dict) else {})
        raise GateC1ContractError(
            f"report for run {run_id!r} must contain exactly development streams; missing={missing}, extra={extra}"
        )
    per_stream: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for stream_id in sorted(expected_ids):
        payload = streams[stream_id]
        if not isinstance(payload, dict):
            raise GateC1ContractError(f"stream {stream_id!r} in run {run_id!r} must be an object")
        per_stream[stream_id] = _normalise_stream_profiles(payload, stream_id, run_id)
    profile_sets = {stream_id: set(profiles) for stream_id, profiles in per_stream.items()}
    first_stream = min(profile_sets)
    expected_profiles = profile_sets[first_stream]
    if not expected_profiles:
        raise GateC1ContractError(f"run {run_id!r} has no profiles")
    for stream_id, profile_ids in profile_sets.items():
        if profile_ids != expected_profiles:
            missing, extra = _key_difference(expected_profiles, profile_ids)
            raise GateC1ContractError(
                f"run {run_id!r} has inconsistent profiles in {stream_id!r}; missing={missing}, extra={extra}"
            )
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for profile_id in sorted(expected_profiles):
        if not isinstance(profile_id, str) or not profile_id:
            raise GateC1ContractError(f"run {run_id!r} has an invalid profile ID")
        result[profile_id] = {}
        for stream_id in sorted(expected_ids):
            metrics = per_stream[stream_id][profile_id]
            if set(metrics) != set(_IOU_KEYS):
                missing, extra = _key_difference(set(_IOU_KEYS), metrics)
                raise GateC1ContractError(
                    f"run {run_id!r}, profile {profile_id!r}, stream {stream_id!r} has incomplete IoU grid; "
                    f"missing={missing}, extra={extra}"
                )
            result[profile_id][stream_id] = {
                key: _validated_metrics(metrics[key], run_id=run_id, stream_id=stream_id, profile_id=profile_id, iou_key=key)
                for key in _IOU_KEYS
            }
    return result


def _normalise_stream_profiles(payload: Mapping[str, Any], stream_id: str, run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    if isinstance(payload.get("evaluations"), dict):
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for profile_id, profile_payload in payload["evaluations"].items():
            if not isinstance(profile_payload, dict):
                raise GateC1ContractError(f"invalid evaluation payload for {stream_id!r}")
            value = profile_payload.get("metrics_by_iou", profile_payload)
            if not isinstance(value, dict):
                raise GateC1ContractError(f"invalid IoU metrics for {stream_id!r}, profile {profile_id!r}")
            result[profile_id] = value
        return result
    if isinstance(payload.get("profiles"), dict):
        result = {}
        for profile_id, profile_payload in payload["profiles"].items():
            if not isinstance(profile_payload, dict) or not isinstance(profile_payload.get("iou_metrics"), dict):
                raise GateC1ContractError(f"invalid Grounding DINO profile for {stream_id!r}, profile {profile_id!r}")
            result[profile_id] = profile_payload["iou_metrics"]
        return result
    raise GateC1ContractError(
        f"run {run_id!r}, stream {stream_id!r} is neither Mask/Mobile/SAM2 nor Grounding DINO schema"
    )


def _validated_metrics(
    payload: Any, *, run_id: str, stream_id: str, profile_id: str, iou_key: str
) -> dict[str, Any]:
    label = f"run {run_id!r}, stream {stream_id!r}, profile {profile_id!r}, IoU {iou_key}"
    if not isinstance(payload, dict):
        raise GateC1ContractError(f"{label} metrics must be an object")
    for key in ("tp", "fp", "fn"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateC1ContractError(f"{label} has invalid {key}")
    for key in ("precision", "recall", "f1"):
        value = payload.get(key)
        if value is None:
            undefined_is_valid = (
                (key == "precision" and payload["tp"] + payload["fp"] == 0)
                or (key == "recall" and payload["tp"] + payload["fn"] == 0)
                or (key == "f1" and (payload.get("precision") is None or payload.get("recall") is None))
            )
            if not undefined_is_valid:
                raise GateC1ContractError(f"{label} has invalid {key}")
            continue
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise GateC1ContractError(f"{label} has invalid {key}")
    false_per_frame = payload.get("false_per_frame")
    if not _finite_number(false_per_frame) or float(false_per_frame) < 0.0:
        raise GateC1ContractError(f"{label} has invalid false_per_frame")
    weighted_iou = payload.get("weighted_mean_tp_iou")
    if weighted_iou is not None and (not _finite_number(weighted_iou) or not 0.0 <= float(weighted_iou) <= 1.0):
        raise GateC1ContractError(f"{label} has invalid weighted_mean_tp_iou")
    tp_iou = payload.get("tp_iou")
    if not isinstance(tp_iou, dict) or not _nonnegative_int(tp_iou.get("count")):
        raise GateC1ContractError(f"{label} has invalid tp_iou")
    if tp_iou.get("mean") is not None and (not _finite_number(tp_iou.get("mean")) or not 0.0 <= float(tp_iou["mean"]) <= 1.0):
        raise GateC1ContractError(f"{label} has invalid tp_iou.mean")
    if not isinstance(payload.get("per_frame"), list):
        raise GateC1ContractError(f"{label} has invalid per_frame")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict) or any(not _nonnegative_int(diagnostics.get(key)) for key in ("duplicate", "split", "merge", "fragment", "noise", "miss")):
        raise GateC1ContractError(f"{label} has invalid diagnostics")
    return payload


def _summarise_profile(
    inventory: Sequence[DevelopmentStream], metrics_by_stream: Mapping[str, Mapping[str, dict[str, Any]]]
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for stream in inventory:
        groups[stream.scene_group_id].append(stream.stream_id)
    result: dict[str, Any] = {}
    for iou_key in _IOU_KEYS:
        scene_groups = {
            group_id: {
                "stream_ids": sorted(stream_ids),
                "metrics": aggregate_candidate_metrics(tuple(metrics_by_stream[stream_id][iou_key] for stream_id in sorted(stream_ids))),
            }
            for group_id, stream_ids in sorted(groups.items())
        }
        group_metrics = [item["metrics"] for item in scene_groups.values()]
        macro = {
            field: _mean_defined([_metric_for_macro(item, field) for item in group_metrics])
            for field in _MACRO_FIELDS
        }
        worst_id = min(
            scene_groups,
            key=lambda group_id: (
                _selection_f1(scene_groups[group_id]["metrics"]),
                group_id,
            ),
        )
        result[iou_key] = {
            "pooled": aggregate_candidate_metrics(
                tuple(metrics_by_stream[stream.stream_id][iou_key] for stream in inventory)
            ),
            "scene_groups": scene_groups,
            "scene_group_macro": macro,
            "worst_scene_group": {
                "scene_group_id": worst_id,
                "metrics": scene_groups[worst_id]["metrics"],
                "selection_f1": _selection_f1(scene_groups[worst_id]["metrics"]),
            },
        }
    return result


def _extract_resources(report: Mapping[str, Any], expected_ids: set[str]) -> dict[str, Any]:
    source = "report_stream_runtime"
    runtime_streams = report.get("streams")
    if "source_report" in report:
        sourced = _load_local_mask_source(report.get("source_report"), expected_ids)
        if sourced is None:
            return _nullable_resources("source_report_unavailable_or_invalid")
        runtime_streams = sourced
        source = "mask_nms_source_report"
    if not isinstance(runtime_streams, dict) or set(runtime_streams) != expected_ids:
        return _nullable_resources("runtime_streams_unavailable")
    elapsed: list[float] = []
    frame_counts: list[int] = []
    peaks: list[int] = []
    for stream_id in sorted(expected_ids):
        payload = runtime_streams.get(stream_id)
        runtime = payload.get("runtime") if isinstance(payload, dict) else None
        if not isinstance(runtime, dict):
            return _nullable_resources(source)
        seconds = runtime.get("elapsed_seconds", runtime.get("total_seconds"))
        frame_count = runtime.get("frame_count")
        if not _finite_number(seconds) or not _nonnegative_int(frame_count) or frame_count == 0:
            return _nullable_resources(source)
        elapsed.append(float(seconds))
        frame_counts.append(int(frame_count))
        peak = runtime.get("peak_cuda_allocated_bytes", runtime.get("peak_gpu_memory_allocated_bytes"))
        if peak is None:
            peaks = []
        elif peaks is not None and _nonnegative_int(peak):
            peaks.append(int(peak))
        else:
            peaks = []
    return {
        "runtime_seconds_per_frame": sum(elapsed) / sum(frame_counts),
        "peak_cuda_allocated_bytes": None if len(peaks) != len(expected_ids) else max(peaks),
        "source": source,
    }


def _load_local_mask_source(value: Any, expected_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value or "://" in value:
        return None
    path = Path(value)
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        source = _read_json_object(path, "Mask R-CNN source report")
    except GateC1ContractError:
        return None
    streams = source.get("streams")
    return streams if isinstance(streams, dict) and set(streams) == expected_ids else None


def _nullable_resources(source: str) -> dict[str, Any]:
    return {"runtime_seconds_per_frame": None, "peak_cuda_allocated_bytes": None, "source": source}


def _select_by_provider(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates_by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for profile_id, profile in run["profiles"].items():
            candidates_by_provider[run["provider_id"]].append(_candidate_record(run, profile_id, profile))
    return [
        {"provider_id": provider_id, **_select_candidates(candidates)}
        for provider_id, candidates in sorted(candidates_by_provider.items())
    ]


def _select_overall(provider_selections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    winners = [selection["winner"] for selection in provider_selections]
    result = _select_candidates(winners)
    return {
        "status": "recommendation_from_pre_registered_development_rule",
        "recommended_provider_id": result["winner"]["provider_id"],
        "recommended_run_id": result["winner"]["run_id"],
        "recommended_profile_id": result["winner"]["profile_id"],
        **result,
    }


def _candidate_record(run: Mapping[str, Any], profile_id: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    summary = profile["summary_by_iou"]
    at_50 = summary["0.50"]
    at_70 = summary["0.70"]
    primary = at_50["scene_group_macro"]["f1"]
    worst = at_50["worst_scene_group"]["selection_f1"]
    recall = at_50["scene_group_macro"]["recall"]
    f1_70 = at_70["scene_group_macro"]["f1"]
    if not all(_finite_number(value) for value in (primary, worst, recall, f1_70)):
        raise GateC1ContractError(f"candidate {profile['candidate_config_id']} has invalid selection metrics")
    resources = run["resources"]
    return {
        "provider_id": run["provider_id"],
        "run_id": run["run_id"],
        "profile_id": profile_id,
        "candidate_config_id": profile["candidate_config_id"],
        "primary_scene_group_macro_f1_at_0_50": primary,
        "worst_scene_f1_at_0_50": worst,
        "scene_group_macro_recall_at_0_50": recall,
        "scene_group_macro_f1_at_0_70": f1_70,
        "runtime_seconds_per_frame": resources["runtime_seconds_per_frame"],
        "peak_cuda_allocated_bytes": resources["peak_cuda_allocated_bytes"],
    }


def _select_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise GateC1ContractError("selection requires at least one candidate")
    best_primary = max(float(item["primary_scene_group_macro_f1_at_0_50"]) for item in candidates)
    tie_group = [
        dict(item)
        for item in candidates
        if float(item["primary_scene_group_macro_f1_at_0_50"]) >= best_primary - PRIMARY_TOLERANCE - 1e-12
    ]
    ranked_tie_group = sorted(tie_group, key=_tie_break_key)
    tie_ids = [item["candidate_config_id"] for item in ranked_tie_group]
    audit: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda row: (-float(row["primary_scene_group_macro_f1_at_0_50"]), row["candidate_config_id"])):
        row = dict(item)
        row["within_primary_tolerance"] = row["candidate_config_id"] in tie_ids
        row["tie_group_rank"] = tie_ids.index(row["candidate_config_id"]) + 1 if row["within_primary_tolerance"] else None
        audit.append(row)
    return {
        "best_primary_scene_group_macro_f1_at_0_50": best_primary,
        "tie_group_candidate_config_ids": tie_ids,
        "candidate_audit": audit,
        "winner": dict(ranked_tie_group[0]),
    }


def _tie_break_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(candidate["worst_scene_f1_at_0_50"]),
        -float(candidate["scene_group_macro_recall_at_0_50"]),
        -float(candidate["scene_group_macro_f1_at_0_70"]),
        _nullable_low_first(candidate["runtime_seconds_per_frame"]),
        _nullable_low_first(candidate["peak_cuda_allocated_bytes"]),
        str(candidate["candidate_config_id"]),
    )


def _nullable_low_first(value: Any) -> tuple[int, float]:
    return (1, 0.0) if value is None else (0, float(value))


def _candidate_config_id(run_id: str, profile_id: str) -> str:
    if "/" in run_id or "/" in profile_id:
        raise GateC1ContractError("run_id and profile_id must not contain '/' for stable candidate IDs")
    return f"{run_id}/{profile_id}"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateC1ContractError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise GateC1ContractError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    canonical = Path(path).resolve(strict=True)
    return {
        "path": canonical.as_posix(),
        "sha256": _sha256(canonical),
        "size_bytes": canonical.stat().st_size,
    }


def _key_difference(expected: Any, actual: Any) -> tuple[list[str], list[str]]:
    actual_keys = set(actual) if isinstance(actual, (dict, set)) else set()
    return sorted(set(expected) - actual_keys), sorted(actual_keys - set(expected))


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean_defined(values: Sequence[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return None if not defined else sum(defined) / len(defined)


def _metric_for_macro(metrics: Mapping[str, Any], field: str) -> float | None:
    value = metrics.get(field)
    if value is not None:
        return float(value)
    if field in {"precision", "f1"} and int(metrics.get("fn", 0)) > 0:
        # No predicted positives while annotated objects exist is a failed
        # detection, not a scene that may be dropped from a macro average.
        return 0.0
    return None


def _selection_f1(metrics: Mapping[str, Any]) -> float:
    value = _metric_for_macro(metrics, "f1")
    if value is None:
        # Gate C1 scene groups contain annotated objects. Keep this defensive
        # fallback deterministic if a malformed empty-only fixture reaches us.
        return 0.0
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-id", action="append", required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--reviewed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = summarize_reports(
            provider_ids=args.provider_id,
            run_ids=args.run_id,
            report_paths=args.report,
            benchmark_spec=args.benchmark_spec,
            reviewed_root=args.reviewed_root,
        )
        write_canonical_json_atomic(args.output, payload)
    except GateC1ContractError as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
