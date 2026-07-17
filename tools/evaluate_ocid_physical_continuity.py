"""Evaluate end-to-end neighboring physical continuity on OCID runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from stream_analysis.contracts import BBox
from stream_analysis.evaluation import (
    PredictedCandidate,
    evaluate_physical_instance_continuity,
    load_annotation,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return value


def _digest(path: Path) -> dict[str, str]:
    source = path.resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"algorithm": "sha256", "value": digest.hexdigest()}


def _predicted(candidate_manifest: dict[str, Any]) -> tuple[PredictedCandidate, ...]:
    result = candidate_manifest.get("result")
    rows = result.get("candidates") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise ValueError("candidate_manifest misses result.candidates.")
    values: list[PredictedCandidate] = []
    for row in rows:
        bbox = row["bbox"]
        envelope = row["envelope"]
        values.append(
            PredictedCandidate(
                candidate_id=row["candidate_id"],
                frame_id=row["frame_id"],
                frame_index=int(row["frame_index"]),
                bbox=BBox(bbox["x"], bbox["y"], bbox["width"], bbox["height"]),
                validity_status=envelope["validity_status"],
                warning_ids=tuple(envelope.get("warning_ids") or ()),
                error_ids=tuple(envelope.get("error_ids") or ()),
            )
        )
    return tuple(values)


def _bound_run_id(
    *,
    expected_stream_id: str,
    candidate_manifest: dict[str, Any],
    primary_result: dict[str, Any],
    predicted: tuple[PredictedCandidate, ...],
) -> str:
    candidate_result = candidate_manifest.get("result")
    if not isinstance(candidate_result, dict):
        raise ValueError("candidate_manifest misses result.")
    candidate_envelope = candidate_result.get("envelope")
    primary_envelope = primary_result.get("envelope")
    if not isinstance(candidate_envelope, dict) or not isinstance(primary_envelope, dict):
        raise ValueError("Candidate and primary results must contain envelopes.")
    candidate_stream_id = candidate_envelope.get("stream_id")
    primary_stream_id = primary_envelope.get("stream_id")
    if candidate_stream_id != expected_stream_id or primary_stream_id != expected_stream_id:
        raise ValueError(
            "Annotation, candidate manifest and primary result stream_id values must match."
        )

    predicted_ids = tuple(item.candidate_id for item in predicted)
    if len(set(predicted_ids)) != len(predicted_ids):
        raise ValueError("candidate_manifest contains duplicate candidate IDs.")
    primary_ids = primary_result.get("candidate_record_ids")
    if not isinstance(primary_ids, list) or not all(
        isinstance(item, str) and item for item in primary_ids
    ):
        raise ValueError("stream_analysis misses valid candidate_record_ids.")
    if len(set(primary_ids)) != len(primary_ids) or set(primary_ids) != set(predicted_ids):
        raise ValueError(
            "candidate_manifest candidates do not match stream_analysis candidate_record_ids."
        )

    run_id = primary_result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("stream_analysis misses run_id.")
    if primary_envelope.get("record_id") != run_id:
        raise ValueError("stream_analysis run_id does not match its envelope record_id.")
    return run_id


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _aggregate(reports: tuple[dict[str, Any], ...], policy: str) -> dict[str, Any]:
    tp = sum(int(item[policy]["tp"]) for item in reports)
    fp = sum(int(item[policy]["fp"]) for item in reports)
    fn = sum(int(item[policy]["fn"]) for item in reports)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        ),
    }


def evaluate_runs(
    *,
    stream_directories: tuple[Path, ...],
    run_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    output_path: Path,
) -> dict[str, Any]:
    if not stream_directories or not (
        len(stream_directories) == len(run_directories) == len(annotation_paths)
    ):
        raise ValueError("Supply one stream, run and annotation per evaluation item.")
    reports: dict[str, Any] = {}
    ordered: list[dict[str, Any]] = []
    for stream_directory, run_directory, annotation_path in zip(
        stream_directories,
        run_directories,
        annotation_paths,
        strict=True,
    ):
        stream_root = stream_directory.resolve(strict=True)
        run_root = run_directory.resolve(strict=True)
        resolved_annotation = annotation_path.resolve(strict=True)
        manifest_path = stream_root / "manifest.json"
        candidate_manifest_path = run_root / "candidate_manifest.json"
        stream_analysis_path = run_root / "stream_analysis.json"
        run_manifest_path = run_root / "run_manifest.json"
        annotation = load_annotation(
            resolved_annotation,
            manifest_path=manifest_path,
        )
        candidate_manifest = _load(candidate_manifest_path)
        primary_payload = _load(stream_analysis_path)
        primary_result = primary_payload.get("result")
        if not isinstance(primary_result, dict):
            raise ValueError("stream_analysis misses result.")
        predicted = _predicted(candidate_manifest)
        run_id = _bound_run_id(
            expected_stream_id=annotation.stream_id,
            candidate_manifest=candidate_manifest,
            primary_result=primary_result,
            predicted=predicted,
        )
        report = evaluate_physical_instance_continuity(
            annotation,
            predicted,
            primary_result,
        )
        if annotation.stream_id in reports:
            raise ValueError(f"Duplicate evaluated stream_id: {annotation.stream_id}.")
        bound_report = {
            **report,
            "evidence": {
                "parent_run_id": run_id,
                "input_digests": {
                    "annotation": _digest(resolved_annotation),
                    "stream_manifest": _digest(manifest_path),
                    "run_manifest": _digest(run_manifest_path),
                    "candidate_manifest": _digest(candidate_manifest_path),
                    "stream_analysis": _digest(stream_analysis_path),
                },
            },
        }
        reports[annotation.stream_id] = bound_report
        ordered.append(bound_report)
    values = tuple(ordered)
    payload = {
        "schema_version": "ocid-physical-continuity-evaluation-0.2",
        "status": "completed",
        "scope": "evaluation_only_end_to_end_physical_continuity",
        "ground_truth_boundary": "Physical-instance proxy IDs were read only after analyze completed.",
        "must_not_be_used_for": ["visual_type_grouping_claims", "event_quality_claims"],
        "streams": reports,
        "aggregate": {
            "stream_count": len(values),
            "frame_pair_count": sum(int(item["frame_pair_count"]) for item in values),
            "bbox_matched_candidate_count": sum(
                int(item["bbox_matched_candidate_count"]) for item in values
            ),
            "uncertain_selected_count": sum(
                int(item["uncertain_selected_count"]) for item in values
            ),
            "accepted_strict": _aggregate(values, "accepted_strict"),
            "selected_including_uncertain": _aggregate(
                values,
                "selected_including_uncertain",
            ),
        },
    }
    _atomic_write(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-directory", type=Path, action="append", required=True)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate_runs(
        stream_directories=tuple(args.stream_directory),
        run_directories=tuple(args.run_directory),
        annotation_paths=tuple(args.annotation),
        output_path=args.output,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
