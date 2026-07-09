"""Versioned, non-overwriting run-scoped output lifecycle."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from PIL import Image

from ..contracts import (
    CandidateExtractionResult, ErrorRecord, FrameMatchingResult,
    PairwiseScoreRecord, StreamAnalysisResult, WarningRecord,
)
from ..provenance import AnalyzeRunProvenance
from ..serialization import serialize_primary_result, to_json_compatible

RUN_MANIFEST_SCHEMA_ID = "stream_analysis.run_manifest.v1"
CANDIDATE_MANIFEST_SCHEMA_ID = "stream_analysis.candidate_manifest.v1"
RUNTIME_STATUS_SCHEMA_ID = "stream_analysis.runtime_status.v1"
PAIR_SCORES_SCHEMA_ID = "stream_analysis.diagnostic_pair_scores.v1"


class OutputCollisionError(FileExistsError):
    """Raised before writing when a run directory already exists."""


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    run_directory: Path
    files: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_directory", self.run_directory.resolve())
        object.__setattr__(self, "files", tuple(sorted(path.resolve() for path in self.files)))


def _json_text(value: Any) -> str:
    return json.dumps(
        to_json_compatible(value), ensure_ascii=False, allow_nan=False,
        indent=2, sort_keys=True,
    ) + "\n"


def build_candidate_manifest_payload(result: CandidateExtractionResult) -> dict[str, Any]:
    core = to_json_compatible(result)
    encoded = json.dumps(core, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema_id": CANDIDATE_MANIFEST_SCHEMA_ID,
        "candidate_snapshot_digest": {"algorithm": "sha256", "value": sha256(encoded).hexdigest()},
        "result": core,
    }


def _lineage_payload(record: PairwiseScoreRecord) -> dict[str, Any]:
    envelope = record.envelope
    return {
        "validity_status": envelope.validity_status.value,
        "warning_ids": list(envelope.warning_ids),
        "error_ids": list(envelope.error_ids),
        "provenance_refs": list(envelope.provenance_refs),
    }


def _pair_score_payload(score: PairwiseScoreRecord) -> dict[str, Any]:
    return {
        "pair_id": score.pair_id,
        "comparison_id": score.comparison_id,
        "frame_pair": {
            "from_frame_id": score.frame_pair.from_frame_id,
            "from_frame_index": score.frame_pair.from_frame_index,
            "to_frame_id": score.frame_pair.to_frame_id,
            "to_frame_index": score.frame_pair.to_frame_index,
        },
        "from_candidate_id": score.from_endpoint.candidate_id,
        "to_candidate_id": score.to_endpoint.candidate_id,
        "representation_record_ids": list(score.representation_record_ids),
        "raw_metric_name": score.raw_metric_name,
        "raw_metric_value": score.raw_metric_value,
        "metric_orientation": score.metric_orientation.value,
        "visual_score": score.visual_score,
        "real_cost": score.real_cost,
        "eligibility": score.eligibility.value,
        "eligibility_reason": score.eligibility_reason,
        "spatial": {
            "normalized_distance": score.spatial_evidence.normalized_distance,
        },
        "lineage": _lineage_payload(score),
    }


def build_pair_scores_payload(
    *,
    run_id: str,
    stream_id: str,
    analysis_config_digest: str,
    scoring_config_digest: str,
    representation_variant_id: str,
    scorer_id: str,
    scorer_version: str,
    matching_results: tuple[FrameMatchingResult, ...],
) -> dict[str, Any]:
    """Build compact deterministic diagnostics for representation ranking.

    The artifact intentionally contains score records only. It never serializes
    representation payloads, embedding vectors, handcrafted feature groups,
    masks, annotation, GT mapping or evaluator metrics.
    """

    for field_name, value in (
        ("run_id", run_id),
        ("stream_id", stream_id),
        ("analysis_config_digest", analysis_config_digest),
        ("scoring_config_digest", scoring_config_digest),
        ("representation_variant_id", representation_variant_id),
        ("scorer_id", scorer_id),
        ("scorer_version", scorer_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    results = tuple(sorted(
        matching_results,
        key=lambda item: (
            item.frame_pair.from_frame_index,
            item.frame_pair.to_frame_index,
            item.comparison_id,
        ),
    ))
    expected_identity = (representation_variant_id, scorer_id, scorer_version)
    for result in results:
        if result.envelope.stream_id != stream_id:
            raise ValueError(
                "Every matching result must belong to the artifact stream_id."
            )
        if (
            result.representation_variant_id,
            result.scorer_id,
            result.scorer_version,
        ) != expected_identity:
            raise ValueError(
                "Pair-score diagnostics cannot mix representation variants or "
                "scorer identities/versions."
            )
    if results:
        orientations = {
            score.metric_orientation.value
            for result in results
            for score in result.pairwise_scores
        }
    else:
        orientations = set()
    pairs = [
        _pair_score_payload(score)
        for result in results
        for score in sorted(
            result.pairwise_scores,
            key=lambda item: (
                item.frame_pair.from_frame_index,
                item.frame_pair.to_frame_index,
                item.from_endpoint.candidate_id,
                item.to_endpoint.candidate_id,
                item.pair_id,
            ),
        )
    ]
    return {
        "schema_id": PAIR_SCORES_SCHEMA_ID,
        "run_id": run_id,
        "stream_id": stream_id,
        "analysis_config_digest": analysis_config_digest,
        "scoring_config_digest": scoring_config_digest,
        "representation_variant_id": representation_variant_id,
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
        "score_semantics": {
            "visual_score": "bounded_higher_is_more_similar_not_probability",
            "raw_metric_orientations": sorted(orientations),
        },
        "pairs": pairs,
    }


def write_run_artifacts(
    *,
    output_root: Path,
    run_id: str,
    provenance: AnalyzeRunProvenance,
    candidate_result: CandidateExtractionResult,
    result: StreamAnalysisResult,
    warnings: tuple[WarningRecord, ...],
    errors: tuple[ErrorRecord, ...],
    report_text: str,
    overlays: Mapping[str, Image.Image],
    pair_scores_payload: Mapping[str, Any] | None = None,
) -> RunArtifacts:
    """Create one immutable run directory and all mandatory primary artifacts."""

    runs_root = Path(output_root).resolve() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_directory = runs_root / run_id
    if run_directory.exists():
        raise OutputCollisionError(f"Run directory already exists: {run_directory}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    relative_files: list[Path] = []

    def write_text(name: str, text: str) -> None:
        path = staging / name
        path.write_text(text, encoding="utf-8", newline="\n")
        relative_files.append(Path(name))

    try:
        write_text(
            "run_manifest.json",
            _json_text({
                "schema_id": RUN_MANIFEST_SCHEMA_ID,
                "provenance": provenance,
                "diagnostics": {"warnings": warnings, "errors": errors},
            }),
        )
        write_text("candidate_manifest.json", _json_text(build_candidate_manifest_payload(candidate_result)))
        write_text("stream_analysis.json", serialize_primary_result(result) + "\n")
        write_text("report.txt", report_text)
        write_text(
            "runtime_status.json",
            _json_text({
                "schema_id": RUNTIME_STATUS_SCHEMA_ID,
                "run_id": run_id,
                "run_status": result.run_status.value,
                "status_summary": result.status_summary,
            }),
        )
        overlay_dir = staging / "overlays"
        overlay_dir.mkdir()
        for frame_id in sorted(overlays):
            relative = Path("overlays") / f"{frame_id}.png"
            overlays[frame_id].save(staging / relative, format="PNG")
            relative_files.append(relative)
        if pair_scores_payload is not None:
            diagnostics_dir = staging / "diagnostics"
            diagnostics_dir.mkdir()
            pair_scores_relative = Path("diagnostics") / "pair_scores.json"
            (staging / pair_scores_relative).write_text(
                _json_text(pair_scores_payload),
                encoding="utf-8",
                newline="\n",
            )
            relative_files.append(pair_scores_relative)
        staging.rename(run_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RunArtifacts(
        run_directory=run_directory,
        files=tuple(run_directory / relative for relative in relative_files),
    )


def write_failed_run_artifacts(
    *, output_root: Path, run_id: str, started_at: str, finished_at: str,
    error_code: str, message: str, requested_provenance: Mapping[str, Any],
) -> RunArtifacts:
    """Persist a compact failed lifecycle when no primary result can be formed."""

    runs_root = Path(output_root).resolve() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_directory = runs_root / run_id
    if run_directory.exists():
        raise OutputCollisionError(f"Run directory already exists: {run_directory}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.failed.", dir=runs_root))
    manifest = {
        "schema_id": RUN_MANIFEST_SCHEMA_ID, "run_id": run_id,
        "run_status": "failed", "started_at": started_at, "finished_at": finished_at,
        "error": {"code": error_code, "message": message},
        "requested_provenance": requested_provenance,
    }
    status = {
        "schema_id": RUNTIME_STATUS_SCHEMA_ID, "run_id": run_id,
        "run_status": "failed", "error_count": 1,
        "error": {"code": error_code, "message": message},
    }
    run_manifest = staging / "run_manifest.json"
    runtime_status = staging / "runtime_status.json"
    report = staging / "report.txt"
    try:
        run_manifest.write_text(_json_text(manifest), encoding="utf-8", newline="\n")
        runtime_status.write_text(_json_text(status), encoding="utf-8", newline="\n")
        report.write_text(
            "Отчёт анализа потока изображений\n"
            f"Запуск: {run_id}\n"
            "Статус: ошибка\n"
            f"Ошибка: {error_code}: {message}\n",
            encoding="utf-8", newline="\n",
        )
        staging.rename(run_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RunArtifacts(
        run_directory=run_directory,
        files=tuple(run_directory / path.name for path in (run_manifest, runtime_status, report)),
    )


__all__ = [
    "CANDIDATE_MANIFEST_SCHEMA_ID", "PAIR_SCORES_SCHEMA_ID", "RUN_MANIFEST_SCHEMA_ID",
    "RUNTIME_STATUS_SCHEMA_ID", "OutputCollisionError", "RunArtifacts",
    "build_candidate_manifest_payload", "build_pair_scores_payload",
    "write_failed_run_artifacts", "write_run_artifacts",
]
