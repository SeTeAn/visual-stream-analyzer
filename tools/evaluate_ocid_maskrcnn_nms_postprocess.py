"""Re-evaluate saved Mask R-CNN predictions with class-agnostic NMS."""

from __future__ import annotations

import argparse
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
from tools.evaluate_ocid_maskrcnn_extractor import RawPrediction, predictions_at_threshold


def _bbox(value: dict[str, Any] | None) -> BBox | None:
    if value is None:
        return None
    return BBox(value["x"], value["y"], value["width"], value["height"])


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
) -> dict[str, Any]:
    if len(stream_directories) != len(annotation_paths) or not stream_directories:
        raise ValueError("Supply one annotation for every stream.")
    source = json.loads(raw_report_path.resolve(strict=True).read_text(encoding="utf-8"))
    reports: dict[str, list[dict[str, Any]]] = {}
    streams: dict[str, Any] = {}
    for stream_directory, annotation_path in zip(stream_directories, annotation_paths, strict=True):
        manifest = json.loads(
            (stream_directory.resolve(strict=True) / "manifest.json").read_text(encoding="utf-8")
        )
        stream_id = manifest["stream_id"]
        raw_rows = source["streams"][stream_id]["raw_predictions"]
        raw = tuple(
            RawPrediction(
                frame_id=row["frame_id"],
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
        for threshold in thresholds:
            for nms_iou in nms_iou_thresholds:
                profile_id = f"score_{threshold:.2f}_nms_{nms_iou:.2f}"
                metrics = evaluate_candidate_predictions(
                    annotation,
                    predictions_at_threshold(
                        raw,
                        threshold=threshold,
                        geometry_variant="model_bbox",
                        class_agnostic_nms_iou=nms_iou,
                    ),
                )
                evaluations[profile_id] = metrics
                reports.setdefault(profile_id, []).append(metrics)
        streams[stream_id] = {"evaluations": evaluations}
    payload = {
        "schema_version": "ocid-maskrcnn-class-agnostic-nms-gate-0.1",
        "status": "completed",
        "scope": "evaluation_only_saved_prediction_postprocessing",
        "ground_truth_boundary": "Annotations were used only after loading saved model predictions.",
        "source_report": str(raw_report_path.resolve(strict=True)),
        "profiles": {
            profile_id: aggregate_candidate_metrics(tuple(values))
            for profile_id, values in reports.items()
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate_saved_predictions(
        raw_report_path=args.raw_report,
        stream_directories=tuple(args.stream_directory),
        annotation_paths=tuple(args.annotation),
        thresholds=tuple(args.threshold or (0.05, 0.10, 0.25)),
        nms_iou_thresholds=tuple(args.nms_iou or (0.30, 0.50, 0.70)),
        output_path=args.output,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
