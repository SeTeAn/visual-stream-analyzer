"""Create a development-only qualitative review pack for a Gate C1 report.

The tool never runs a model.  It reconstructs a pre-registered Grounding DINO
profile from the report's raw predictions, then renders the two frames with
the largest ``fp + fn`` error from each paired-camera scene group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from stream_analysis.contracts import BBox
from stream_analysis.evaluation.annotation import load_annotation
try:  # Support module imports and direct execution from ``tools``.
    from tools.evaluate_ocid_grounding_dino_extractor import (
        FilterProfile,
        RawPrediction,
        predictions_for_profile,
    )
    from tools.ocid_gate_c1_common import (
        DevelopmentStream,
        GateC1ContractError,
        load_development_inventory,
        write_canonical_json_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI smoke tests.
    from evaluate_ocid_grounding_dino_extractor import (  # type: ignore[no-redef]
        FilterProfile,
        RawPrediction,
        predictions_for_profile,
    )
    from ocid_gate_c1_common import (  # type: ignore[no-redef]
        DevelopmentStream,
        GateC1ContractError,
        load_development_inventory,
        write_canonical_json_atomic,
    )


SCHEMA_VERSION = "ocid-gate-c1-grounding-dino-review-pack-1.0"
IOU_KEY = "0.50"
ERROR_FRAMES_PER_SCENE = 2
_PROFILE_RE = re.compile(r"^score_(0(?:\.\d+)?|1(?:\.0+)?)_nms_(none|0(?:\.\d+)?|1(?:\.0+)?)$")


def prepare_review_pack(
    *,
    report_path: Path,
    benchmark_spec_path: Path,
    reviewed_root: Path,
    profile_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate a completed development report and render its selected frames.

    The report is validated against the frozen inventory before this function
    opens an image or calls :func:`load_annotation`.
    """

    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    report_file = Path(report_path).resolve(strict=True)
    report = _json_object(report_file, "report")
    profile, selected_by_stream, frame_errors = _validate_report(
        report, inventory=inventory, profile_id=profile_id
    )
    selections = _select_error_frames(inventory, frame_errors)
    root = Path(output_dir).resolve(strict=False)
    rendered: list[dict[str, Any]] = []
    for selection in selections:
        stream = selection["stream"]
        frame_id = selection["frame_id"]
        annotation = load_annotation(
            stream.annotation_path, manifest_path=stream.stream_directory / "manifest.json"
        )
        image_path = _frame_path(stream, frame_id)
        predicted = selected_by_stream[stream.stream_id].get(frame_id, ())
        gt = annotation.instances_by_frame.get(frame_id, ())
        output_name = f"{stream.scene_group_id}__{stream.stream_id}__{frame_id}.png"
        output_path = root / output_name
        _render_frame(
            image_path=image_path,
            output_path=output_path,
            stream_id=stream.stream_id,
            frame_id=frame_id,
            errors=selection["errors"],
            gt_boxes=tuple(item.bbox for item in gt),
            predicted=predicted,
        )
        rendered.append(
            {
                "scene_group_id": stream.scene_group_id,
                "stream_id": stream.stream_id,
                "frame_id": frame_id,
                "errors": selection["errors"],
                "ground_truth_bboxes": [
                    {
                        "instance_id": item.instance_id,
                        "visual_type_id": item.visual_type_id,
                        "bbox": _bbox_dict(item.bbox),
                    }
                    for item in gt
                ],
                "predicted_bboxes": [_raw_row(item) for item in predicted],
                "output_file": output_name,
            }
        )
    contact_name = "contact_sheet.png"
    _render_contact_sheet(tuple(root / item["output_file"] for item in rendered), root / contact_name)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "development_only_qualitative_review",
        "input_report": {
            "path": report_file.as_posix(),
            "sha256": _sha256(report_file),
            "status": report["status"],
        },
        "profile": {
            "profile_id": profile_id,
            "score_threshold": profile.score_threshold,
            "class_agnostic_nms_iou": profile.class_agnostic_nms_iou,
            "iou_metrics_key": IOU_KEY,
        },
        "selection_rule": {
            "per_scene_group": ERROR_FRAMES_PER_SCENE,
            "sort_key": ["-(fp + fn)", "stream_id", "frame_id"],
        },
        "frames": rendered,
        "contact_sheet": contact_name,
    }
    write_canonical_json_atomic(root / "index.json", payload)
    return payload


def _validate_report(
    report: Mapping[str, Any], *, inventory: Sequence[DevelopmentStream], profile_id: str
) -> tuple[FilterProfile, dict[str, dict[str, tuple[RawPrediction, ...]]], dict[str, list[dict[str, Any]]]]:
    if report.get("status") != "completed":
        raise GateC1ContractError("review-pack report must have status='completed'")
    streams = report.get("streams")
    if not isinstance(streams, Mapping):
        raise GateC1ContractError("report.streams must be an object")
    expected_ids = {stream.stream_id for stream in inventory}
    if set(streams) != expected_ids:
        raise GateC1ContractError("report must contain exactly the frozen 10 development stream IDs")
    profile = _profile_from_report(report.get("profiles"), profile_id)
    selected_by_stream: dict[str, dict[str, tuple[RawPrediction, ...]]] = {}
    errors_by_stream: dict[str, list[dict[str, Any]]] = {}
    for stream in sorted(inventory, key=lambda item: item.stream_id):
        stream_report = streams[stream.stream_id]
        if not isinstance(stream_report, Mapping):
            raise GateC1ContractError(f"report stream {stream.stream_id!r} must be an object")
        profiles = stream_report.get("profiles")
        if not isinstance(profiles, Mapping) or profile_id not in profiles:
            raise GateC1ContractError(f"profile {profile_id!r} is absent from {stream.stream_id}")
        profile_metrics = profiles[profile_id]
        if not isinstance(profile_metrics, Mapping):
            raise GateC1ContractError(f"profile metrics are malformed for {stream.stream_id}")
        iou_metrics = profile_metrics.get("iou_metrics")
        if not isinstance(iou_metrics, Mapping) or not isinstance(iou_metrics.get(IOU_KEY), Mapping):
            raise GateC1ContractError(f"profile {profile_id!r} lacks {IOU_KEY} metrics for {stream.stream_id}")
        errors_by_stream[stream.stream_id] = _frame_errors(iou_metrics[IOU_KEY], stream.stream_id)
        raw = _raw_predictions(stream_report.get("raw_predictions"), stream.stream_id)
        selected = predictions_for_profile(raw, profile=profile)
        expected_count = profile_metrics.get("candidate_count")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count != len(selected):
            raise GateC1ContractError(f"profile reconstruction count mismatch for {stream.stream_id}")
        candidate_to_raw = {
            f"grounding-dino:{item.frame_id}:{item.prediction_index:03d}": item for item in raw
        }
        grouped: dict[str, list[RawPrediction]] = defaultdict(list)
        for candidate in selected:
            grouped[candidate.frame_id].append(candidate_to_raw[candidate.candidate_id])
        selected_by_stream[stream.stream_id] = {
            frame_id: tuple(sorted(rows, key=lambda item: item.prediction_index))
            for frame_id, rows in grouped.items()
        }
    return profile, selected_by_stream, errors_by_stream


def _profile_from_report(value: Any, profile_id: str) -> FilterProfile:
    if not isinstance(profile_id, str) or not profile_id:
        raise GateC1ContractError("profile_id must be a non-empty string")
    definitions = value if isinstance(value, list) else ()
    for item in definitions:
        if isinstance(item, Mapping) and item.get("profile_id") == profile_id:
            score = _finite_float(item.get("score_threshold"), "profile score_threshold")
            nms_value = item.get("class_agnostic_nms_iou")
            nms = None if nms_value is None else _finite_float(nms_value, "profile class_agnostic_nms_iou")
            profile = FilterProfile(score, nms)
            if profile.profile_id != profile_id:
                raise GateC1ContractError("profile definition is inconsistent with profile_id")
            return profile
    match = _PROFILE_RE.fullmatch(profile_id)
    if match is None:
        raise GateC1ContractError("profile_id is absent and cannot be parsed as an exact Grounding DINO profile")
    score = float(match.group(1))
    nms = None if match.group(2) == "none" else float(match.group(2))
    profile = FilterProfile(score, nms)
    if profile.profile_id != profile_id:
        raise GateC1ContractError("profile_id must use canonical two-decimal formatting")
    return profile


def _raw_predictions(value: Any, stream_id: str) -> tuple[RawPrediction, ...]:
    if not isinstance(value, list):
        raise GateC1ContractError(f"raw_predictions must be a list for {stream_id}")
    rows: list[RawPrediction] = []
    seen: set[tuple[str, int]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise GateC1ContractError(f"raw prediction is malformed for {stream_id}")
        frame_id = _text(item.get("frame_id"), "raw frame_id")
        frame_index = _nonnegative_int(item.get("frame_index"), "raw frame_index")
        prediction_index = _nonnegative_int(item.get("prediction_index"), "raw prediction_index")
        pair = (frame_id, prediction_index)
        if pair in seen:
            raise GateC1ContractError(f"duplicate raw prediction index in {stream_id}")
        seen.add(pair)
        bbox_value = item.get("bbox")
        if not isinstance(bbox_value, Mapping):
            raise GateC1ContractError(f"raw bbox is malformed for {stream_id}")
        try:
            bbox = BBox(
                _finite_float(bbox_value.get("x"), "raw bbox.x"),
                _finite_float(bbox_value.get("y"), "raw bbox.y"),
                _finite_float(bbox_value.get("width"), "raw bbox.width"),
                _finite_float(bbox_value.get("height"), "raw bbox.height"),
            )
        except ValueError as error:
            raise GateC1ContractError(f"invalid raw bbox in {stream_id}: {error}") from error
        phrase = item.get("phrase")
        if not isinstance(phrase, str):
            raise GateC1ContractError("raw phrase must be a string")
        rows.append(
            RawPrediction(
                frame_id=frame_id,
                frame_index=frame_index,
                prediction_index=prediction_index,
                score=_finite_float(item.get("score"), "raw score"),
                phrase=phrase,
                bbox=bbox,
            )
        )
    return tuple(rows)


def _frame_errors(metrics: Mapping[str, Any], stream_id: str) -> list[dict[str, Any]]:
    rows = metrics.get("per_frame")
    if not isinstance(rows, list) or not rows:
        raise GateC1ContractError(f"{IOU_KEY} per_frame metrics are missing for {stream_id}")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise GateC1ContractError(f"per_frame row is malformed for {stream_id}")
        frame_id = _text(row.get("frame_id"), "per_frame frame_id")
        if frame_id in seen:
            raise GateC1ContractError(f"duplicate per_frame frame_id in {stream_id}")
        seen.add(frame_id)
        errors = {name: _nonnegative_int(row.get(name), f"per_frame {name}") for name in ("tp", "fp", "fn")}
        result.append({"frame_id": frame_id, "errors": errors})
    return result


def _select_error_frames(
    inventory: Sequence[DevelopmentStream], errors_by_stream: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stream in inventory:
        if stream.stream_id not in errors_by_stream:
            raise GateC1ContractError(f"missing frame errors for {stream.stream_id}")
        for row in errors_by_stream[stream.stream_id]:
            grouped[stream.scene_group_id].append({"stream": stream, **row})
    selections: list[dict[str, Any]] = []
    for scene_group_id in sorted(grouped):
        rows = sorted(
            grouped[scene_group_id],
            key=lambda item: (
                -(item["errors"]["fp"] + item["errors"]["fn"]),
                item["stream"].stream_id,
                item["frame_id"],
            ),
        )
        if len(rows) < ERROR_FRAMES_PER_SCENE:
            raise GateC1ContractError(f"scene group {scene_group_id} has fewer than two frame rows")
        selections.extend(rows[:ERROR_FRAMES_PER_SCENE])
    return selections


def _frame_path(stream: DevelopmentStream, frame_id: str) -> Path:
    manifest = _json_object(stream.stream_directory / "manifest.json", f"manifest for {stream.stream_id}")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise GateC1ContractError(f"manifest frames are malformed for {stream.stream_id}")
    for row in frames:
        if isinstance(row, Mapping) and row.get("frame_id") == frame_id and isinstance(row.get("image_path"), str):
            path = (stream.stream_directory / row["image_path"]).resolve(strict=True)
            try:
                path.relative_to(stream.stream_directory.resolve())
            except ValueError as error:
                raise GateC1ContractError(f"frame path escapes stream directory for {stream.stream_id}") from error
            return path
    raise GateC1ContractError(f"selected frame {frame_id!r} is absent from {stream.stream_id}")


def _render_frame(
    *, image_path: Path, output_path: Path, stream_id: str, frame_id: str,
    errors: Mapping[str, int], gt_boxes: Sequence[BBox], predicted: Sequence[RawPrediction],
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for bbox in gt_boxes:
        draw.rectangle((bbox.left, bbox.top, bbox.right, bbox.bottom), outline=(0, 220, 70), width=3)
    for item in predicted:
        bbox = item.bbox
        draw.rectangle((bbox.left, bbox.top, bbox.right, bbox.bottom), outline=(235, 0, 190), width=3)
    banner_height = 38
    canvas = Image.new("RGB", (image.width, image.height + banner_height), (25, 25, 25))
    canvas.paste(image, (0, banner_height))
    banner = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text = (
        f"{stream_id} | {frame_id} | TP={errors['tp']} FP={errors['fp']} FN={errors['fn']} "
        "| GT: green, prediction: magenta"
    )
    banner.text((6, 12), text, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _render_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    if len(paths) != ERROR_FRAMES_PER_SCENE * 5:
        raise GateC1ContractError("review pack must contain exactly ten rendered frames")
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        sheet = Image.new("RGB", (width * 2, height * ((len(images) + 1) // 2)), (20, 20, 20))
        for index, image in enumerate(images):
            sheet.paste(image, ((index % 2) * width, (index // 2) * height))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, format="PNG")
    finally:
        for image in images:
            image.close()


def _raw_row(item: RawPrediction) -> dict[str, Any]:
    return {
        "candidate_id": f"grounding-dino:{item.frame_id}:{item.prediction_index:03d}",
        "frame_index": item.frame_index,
        "prediction_index": item.prediction_index,
        "score": item.score,
        "phrase": item.phrase,
        "bbox": _bbox_dict(item.bbox),
    }


def _bbox_dict(bbox: BBox) -> dict[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateC1ContractError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise GateC1ContractError(f"{label} must be a JSON object")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GateC1ContractError(f"{label} must be a finite number")
    return float(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateC1ContractError(f"{label} must be a non-negative integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateC1ContractError(f"{label} must be a non-empty string")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--reviewed-root", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = prepare_review_pack(
        report_path=args.report,
        benchmark_spec_path=args.benchmark_spec,
        reviewed_root=args.reviewed_root,
        profile_id=args.profile_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": "completed", "index": str(args.output_dir / "index.json"), "frames": len(payload["frames"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
