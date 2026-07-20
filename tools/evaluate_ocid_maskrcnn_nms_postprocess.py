"""Re-evaluate saved Mask R-CNN predictions with class-agnostic NMS."""

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
    aggregate_candidate_metrics,
    evaluate_candidate_predictions,
    load_annotation,
)
try:  # Support both ``python -m tools...`` and direct script execution.
    from tools.ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke test.
    from ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        validate_development_input_pairs,
    )
try:  # Support both ``python -m tools...`` and direct script execution.
    from tools.evaluate_ocid_maskrcnn_extractor import (
        DEFAULT_MATCH_IOU_THRESHOLDS,
        DEFAULT_SCORE_THRESHOLDS,
        GEOMETRY_VARIANTS,
        RawPrediction,
        predictions_at_threshold,
        profile_definition,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke test.
    from evaluate_ocid_maskrcnn_extractor import (
        DEFAULT_MATCH_IOU_THRESHOLDS,
        DEFAULT_SCORE_THRESHOLDS,
        GEOMETRY_VARIANTS,
        RawPrediction,
        predictions_at_threshold,
        profile_definition,
    )


def _bbox(value: dict[str, Any] | None) -> BBox | None:
    if value is None:
        return None
    return BBox(value["x"], value["y"], value["width"], value["height"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def evaluate_saved_predictions(
    *,
    raw_report_path: Path,
    stream_directories: tuple[Path, ...],
    annotation_paths: tuple[Path, ...],
    thresholds: tuple[float, ...],
    nms_iou_thresholds: tuple[float, ...],
    output_path: Path,
    match_iou_thresholds: tuple[float, ...] = DEFAULT_MATCH_IOU_THRESHOLDS,
    include_no_nms: bool = True,
    benchmark_spec_path: Path = DEFAULT_BENCHMARK_SPEC,
    reviewed_root: Path = DEFAULT_REVIEWED_ROOT,
) -> dict[str, Any]:
    # This must precede reading the supplied raw report, manifests or
    # annotations so NMS postprocessing cannot accidentally evaluate held-out.
    inventory = validate_development_input_pairs(
        stream_directories,
        annotation_paths,
        benchmark_spec_path=benchmark_spec_path,
        reviewed_root=reviewed_root,
    )
    canonical_spec = Path(benchmark_spec_path).resolve(strict=True)
    canonical_root = Path(reviewed_root).resolve(strict=True)
    source = json.loads(raw_report_path.resolve(strict=True).read_text(encoding="utf-8"))
    reports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    profile_definitions: dict[str, dict[str, Any]] = {}
    streams: dict[str, Any] = {}
    for development_stream in inventory:
        stream_directory = development_stream.stream_directory
        annotation_path = development_stream.annotation_path
        manifest = json.loads(
            (stream_directory.resolve(strict=True) / "manifest.json").read_text(encoding="utf-8")
        )
        stream_id = manifest["stream_id"]
        raw_rows = source["streams"][stream_id]["raw_predictions"]
        raw = tuple(
            RawPrediction(
                frame_id=row["frame_id"],
                frame_index=int(row.get("frame_index", 0)),
                prediction_index=int(row["prediction_index"]),
                score=float(row["score"]),
                label=int(row["label"]),
                model_bbox=_bbox(row["model_bbox"]),
                mask_tight_bbox=_bbox(row["mask_tight_bbox"]),
            )
            for row in raw_rows
        )
        if any(item.model_bbox is None for item in raw):
            raise ValueError("Saved model_bbox rows must not be null.")
        annotation = load_annotation(
            annotation_path.resolve(strict=True),
            manifest_path=stream_directory.resolve(strict=True) / "manifest.json",
        )
        evaluations: dict[str, Any] = {}
        nms_profiles: tuple[float | None, ...] = (
            (None, *nms_iou_thresholds) if include_no_nms else nms_iou_thresholds
        )
        for geometry_variant in GEOMETRY_VARIANTS:
            for threshold in thresholds:
                for nms_iou in nms_profiles:
                    predictions = predictions_at_threshold(
                        raw,
                        threshold=threshold,
                        geometry_variant=geometry_variant,
                        class_agnostic_nms_iou=nms_iou,
                    )
                    definition = profile_definition(
                        geometry_variant=geometry_variant,
                        score_threshold=threshold,
                        nms_iou_threshold=nms_iou,
                    )
                    identifier = definition["profile_id"]
                    metrics_by_iou = {
                        f"{match_iou_threshold:.2f}": evaluate_candidate_predictions(
                            annotation,
                            predictions,
                            iou_threshold=match_iou_threshold,
                        )
                        for match_iou_threshold in match_iou_thresholds
                    }
                    evaluations[identifier] = {**definition, "metrics_by_iou": metrics_by_iou}
                    for iou_key, metrics in metrics_by_iou.items():
                        reports.setdefault(identifier, {}).setdefault(iou_key, []).append(metrics)
                    profile_definitions.setdefault(identifier, definition)
        streams[stream_id] = {"evaluations": evaluations}
    payload = {
        "schema_version": "ocid-maskrcnn-class-agnostic-nms-gate-0.3",
        "status": "completed",
        "scope": "evaluation_only_saved_prediction_postprocessing",
        "ground_truth_boundary": "Annotations were used only after loading saved model predictions.",
        "development_contract": {
            "benchmark_spec_path": canonical_spec.as_posix(),
            "benchmark_spec_sha256": _sha256(canonical_spec),
            "reviewed_root": canonical_root.as_posix(),
            "stream_ids": [stream.stream_id for stream in inventory],
        },
        "source_report": str(raw_report_path.resolve(strict=True)),
        "thresholds": thresholds,
        "match_iou_thresholds": match_iou_thresholds,
        "nms_iou_thresholds": nms_iou_thresholds,
        "include_no_nms": include_no_nms,
        "geometry_variants": GEOMETRY_VARIANTS,
        "profile_definitions": [profile_definitions[key] for key in sorted(profile_definitions)],
        "pooled_profiles": {
            identifier: {
                **profile_definitions[identifier],
                "metrics_by_iou": {
                    iou_key: aggregate_candidate_metrics(tuple(values))
                    for iou_key, values in sorted(metrics_by_iou.items())
                },
            }
            for identifier, metrics_by_iou in sorted(reports.items())
        },
        "profiles": {
            identifier: {
                **profile_definitions[identifier],
                "metrics_by_iou": {
                    iou_key: aggregate_candidate_metrics(tuple(values))
                    for iou_key, values in sorted(metrics_by_iou.items())
                },
            }
            for identifier, metrics_by_iou in sorted(reports.items())
        },
        "streams": streams,
    }
    _atomic_write(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--stream-directory", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, action="append")
    parser.add_argument("--nms-iou", type=float, action="append")
    parser.add_argument("--match-iou", type=float, action="append")
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument(
        "--exclude-no-nms",
        action="store_true",
        help="Do not repeat the no-NMS profiles from the raw prediction report.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate_saved_predictions(
        raw_report_path=args.raw_report,
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        thresholds=tuple(args.threshold or DEFAULT_SCORE_THRESHOLDS),
        nms_iou_thresholds=tuple(args.nms_iou or (0.30, 0.50, 0.70)),
        output_path=args.output,
        match_iou_thresholds=tuple(args.match_iou or DEFAULT_MATCH_IOU_THRESHOLDS),
        include_no_nms=not args.exclude_no_nms,
        benchmark_spec_path=args.benchmark_spec,
        reviewed_root=args.reviewed_root,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
