"""Evaluate the pre-registered OCID Gate C1.1 Grounding DINO hardening grid.

The tool derives the exact frozen development inventory, reuses the immutable
Gate C1 ``object.`` raw predictions, runs only the pre-registered additional
prompts, and opens annotations only after all RGB inference has completed.
Held-out streams are rejected by the shared benchmark contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from stream_analysis.contracts import BBox, ImageSize
from stream_analysis.evaluation import PredictedCandidate

try:  # Support module and direct-script execution.
    from tools import evaluate_ocid_grounding_dino_extractor as grounding
    from tools.ocid_gate_c1_common import (
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        DevelopmentStream,
        GateC1ContractError,
        evaluate_profiles,
        load_development_inventory,
        write_canonical_json_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback.
    import evaluate_ocid_grounding_dino_extractor as grounding  # type: ignore[no-redef]
    from ocid_gate_c1_common import (  # type: ignore[no-redef]
        DEFAULT_BENCHMARK_SPEC,
        DEFAULT_REVIEWED_ROOT,
        DevelopmentStream,
        GateC1ContractError,
        evaluate_profiles,
        load_development_inventory,
        write_canonical_json_atomic,
    )


SCHEMA_VERSION = "ocid-grounding-dino-hardening-gate-c1-1.v1"
SELECTED_CANDIDATE_SCHEMA = "ocid-grounding-dino-selected-candidates.v1"
FROZEN_SCORE_THRESHOLD = 0.15
FROZEN_NMS_IOU = 0.30
SURFACE_REDUCTION_MINIMUM = 0.25
MACRO_RECALL_MAX_DROP = 0.01
STREAM_RECALL_MAX_DROP = 0.03
SELECTION_F1_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class PromptProfile:
    prompt_id: str
    text: str
    selectable: bool
    source: str

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.prompt_id.replace("_", "").isalnum():
            raise ValueError("prompt_id must contain letters, digits or underscores.")
        if not self.text or self.text != self.text.casefold() or not self.text.endswith("."):
            raise ValueError("prompt text must be lowercase and end with a period.")
        if self.source not in {"gate_c1_raw_reuse", "new_inference"}:
            raise ValueError("unsupported prompt source.")


@dataclass(frozen=True, slots=True)
class GeometryProfile:
    geometry_id: str
    min_area_ratio: float | None = None
    min_span_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.geometry_id or not self.geometry_id.replace("_", "").isalnum():
            raise ValueError("geometry_id must contain letters, digits or underscores.")
        for value in (self.min_area_ratio, self.min_span_ratio):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError("geometry ratios must be in (0, 1].")
        if self.min_span_ratio is not None and self.min_area_ratio is None:
            raise ValueError("a span threshold requires an area threshold.")


PROMPTS: tuple[PromptProfile, ...] = (
    PromptProfile("p_object", "object.", True, "gate_c1_raw_reuse"),
    PromptProfile("p_item", "item.", True, "new_inference"),
    PromptProfile("p_foreground_object", "foreground object.", True, "new_inference"),
    PromptProfile(
        "p_background_labels_diag",
        "object. table. floor. wall.",
        False,
        "new_inference",
    ),
)

GEOMETRIES: tuple[GeometryProfile, ...] = (
    GeometryProfile("g_none"),
    GeometryProfile("g_frame_080", min_area_ratio=0.80),
    GeometryProfile(
        "g_surface_span_055_090",
        min_area_ratio=0.55,
        min_span_ratio=0.90,
    ),
)


def _profile_id(prompt_id: str, geometry_id: str) -> str:
    return f"{prompt_id}__{geometry_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateC1ContractError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise GateC1ContractError(f"{label} must be a JSON object")
    return payload


def _file_provenance(path: Path) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    return {
        "path": canonical.as_posix(),
        "sha256": _sha256(canonical),
        "size_bytes": canonical.stat().st_size,
    }


def _raw_prediction(payload: Mapping[str, Any]) -> grounding.RawPrediction:
    bbox = payload.get("bbox")
    if not isinstance(bbox, Mapping):
        raise GateC1ContractError("raw prediction bbox must be an object")
    try:
        return grounding.RawPrediction(
            frame_id=str(payload["frame_id"]),
            frame_index=int(payload["frame_index"]),
            prediction_index=int(payload["prediction_index"]),
            score=float(payload["score"]),
            phrase=str(payload.get("phrase", "")),
            bbox=BBox(
                x=float(bbox["x"]),
                y=float(bbox["y"]),
                width=float(bbox["width"]),
                height=float(bbox["height"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GateC1ContractError(f"malformed raw prediction: {error}") from error


def load_gate_c1_baseline_raw(
    report_path: Path,
    inventory: Sequence[DevelopmentStream],
) -> dict[str, tuple[grounding.RawPrediction, ...]]:
    """Load and strictly validate the immutable Gate C1 ``object.`` report."""

    report = _json_object(report_path, "Gate C1 Grounding DINO report")
    if report.get("schema_version") != grounding.SCHEMA_VERSION:
        raise GateC1ContractError("baseline report schema does not match Gate C1")
    if (
        report.get("status") != "completed"
        or report.get("scope") != "development_only_candidate_extraction_gate_c1"
    ):
        raise GateC1ContractError("baseline report must be the completed Gate C1 development report")
    if report.get("prompt") != "object.":
        raise GateC1ContractError("baseline report must use the frozen object. prompt")
    streams = report.get("streams")
    expected = {item.stream_id for item in inventory}
    if not isinstance(streams, Mapping) or set(streams) != expected:
        raise GateC1ContractError("baseline report must contain exactly the development streams")
    result: dict[str, tuple[grounding.RawPrediction, ...]] = {}
    for item in inventory:
        stream = streams[item.stream_id]
        if not isinstance(stream, Mapping):
            raise GateC1ContractError(f"malformed baseline stream {item.stream_id}")
        raw = stream.get("raw_predictions")
        if not isinstance(raw, list):
            raise GateC1ContractError(f"baseline stream {item.stream_id} lacks raw predictions")
        parsed = tuple(_raw_prediction(row) for row in raw if isinstance(row, Mapping))
        if len(parsed) != len(raw):
            raise GateC1ContractError(f"baseline stream {item.stream_id} has malformed raw rows")
        if any(row.frame_index < 0 or row.prediction_index < 0 for row in parsed):
            raise GateC1ContractError("baseline raw indices must be nonnegative")
        result[item.stream_id] = parsed
    return result


def _infer_stream_prompt(
    decoded: Any,
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    prompt: str,
) -> tuple[tuple[grounding.RawPrediction, ...], dict[str, Any]]:
    """Run one registered prompt without opening evaluation annotations."""

    from PIL import Image

    raw: list[grounding.RawPrediction] = []
    frames: list[dict[str, Any]] = []
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    stream_started = time.perf_counter()
    for frame_index, frame in enumerate(decoded.frames):
        image = Image.fromarray(grounding._frame_array(frame), mode="RGB")
        if device == "cuda":
            torch_module.cuda.synchronize()
        started = time.perf_counter()
        inputs = grounding._move_to_device(
            processor(images=image, text=prompt, return_tensors="pt"),
            device,
        )
        model_started = time.perf_counter()
        with torch_module.inference_mode():
            outputs = model(**inputs)
        if device == "cuda":
            torch_module.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        target_sizes = torch_module.tensor(
            [[frame.image_size.height, frame.image_size.width]]
        )
        processed = grounding._post_process(processor, outputs, inputs, target_sizes)
        if not isinstance(processed, (list, tuple)) or len(processed) != 1:
            raise ValueError("Grounding DINO processor must return one frame result")
        rows = grounding._normalise_processor_predictions(
            processed[0],
            frame_id=frame.frame_id,
            frame_index=frame_index,
            image_size=frame.image_size,
        )
        total_seconds = time.perf_counter() - started
        raw.extend(rows)
        frames.append(
            {
                "frame_id": frame.frame_id,
                "raw_prediction_count": len(rows),
                "model_forward_seconds": model_seconds,
                "total_inference_seconds": total_seconds,
            }
        )
        del image, inputs, outputs, processed
    elapsed = time.perf_counter() - stream_started
    latencies = [float(item["total_inference_seconds"]) for item in frames]
    return tuple(raw), {
        "frame_count": len(frames),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_frame": None if not frames else elapsed / len(frames),
        "latency_seconds": {
            "p50": grounding._percentile(latencies, 50),
            "p95": grounding._percentile(latencies, 95),
        },
        "process_rss_bytes": grounding._rss_bytes(),
        "peak_gpu_memory_allocated_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch_module.cuda.max_memory_reserved()) if device == "cuda" else None
        ),
        "frames": frames,
    }


def _frame_sizes(decoded: Any) -> dict[str, ImageSize]:
    return {frame.frame_id: frame.image_size for frame in decoded.frames}


def geometry_rejects(
    bbox: BBox,
    image_size: ImageSize,
    geometry: GeometryProfile,
) -> bool:
    frame_area = image_size.width * image_size.height
    if frame_area <= 0 or geometry.min_area_ratio is None:
        return False
    area_ratio = bbox.area / frame_area
    if area_ratio < geometry.min_area_ratio:
        return False
    if geometry.min_span_ratio is None:
        return True
    return (
        bbox.width / image_size.width >= geometry.min_span_ratio
        or bbox.height / image_size.height >= geometry.min_span_ratio
    )


def _surface_like(bbox: BBox, image_size: ImageSize) -> bool:
    return geometry_rejects(bbox, image_size, GEOMETRIES[1]) or geometry_rejects(
        bbox, image_size, GEOMETRIES[2]
    )


def candidates_for_profile(
    raw: Sequence[grounding.RawPrediction],
    *,
    prompt: PromptProfile,
    geometry: GeometryProfile,
    image_sizes: Mapping[str, ImageSize],
) -> tuple[tuple[PredictedCandidate, ...], tuple[dict[str, Any], ...]]:
    """Apply frozen score/NMS and one registered geometry policy."""

    selected = tuple(row for row in raw if row.score >= FROZEN_SCORE_THRESHOLD)
    selected = grounding._class_agnostic_nms(selected, FROZEN_NMS_IOU)
    candidates: list[PredictedCandidate] = []
    records: list[dict[str, Any]] = []
    for row in selected:
        image_size = image_sizes.get(row.frame_id)
        if image_size is None:
            raise GateC1ContractError(f"missing frame size for {row.frame_id}")
        rejected = geometry_rejects(row.bbox, image_size, geometry)
        source_key = f"{prompt.prompt_id}:{row.frame_id}:{row.prediction_index:03d}"
        candidate_id = (
            f"grounding-dino-hardening:{prompt.prompt_id}:{geometry.geometry_id}:"
            f"{row.frame_id}:{row.prediction_index:03d}"
        )
        area_ratio = row.bbox.area / (image_size.width * image_size.height)
        record = {
            "source_prediction_key": source_key,
            "candidate_id": candidate_id,
            "frame_id": row.frame_id,
            "frame_index": row.frame_index,
            "prediction_index": row.prediction_index,
            "score": row.score,
            "phrase": row.phrase,
            "bbox": {
                "x": row.bbox.x,
                "y": row.bbox.y,
                "width": row.bbox.width,
                "height": row.bbox.height,
            },
            "bbox_area_ratio": area_ratio,
            "bbox_width_ratio": row.bbox.width / image_size.width,
            "bbox_height_ratio": row.bbox.height / image_size.height,
            "surface_like": _surface_like(row.bbox, image_size),
            "geometry_rejected": rejected,
        }
        records.append(record)
        if rejected:
            continue
        candidates.append(
            PredictedCandidate(
                candidate_id=candidate_id,
                frame_id=row.frame_id,
                frame_index=row.frame_index,
                bbox=row.bbox,
                validity_status="valid",
                warning_ids=(),
                error_ids=(),
            )
        )
    return tuple(candidates), tuple(records)


def _candidate_record_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        candidate_id = record.get("candidate_id")
        if record.get("geometry_rejected") is False and isinstance(candidate_id, str):
            if candidate_id in result:
                raise GateC1ContractError("candidate IDs must be unique within a stream")
            result[candidate_id] = record
    return result


def _profile_diagnostics(
    development_result: Mapping[str, Any],
    records_by_profile: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    profiles = development_result.get("profiles")
    if not isinstance(profiles, Mapping):
        raise GateC1ContractError("development result lacks profiles")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, Mapping):
            raise GateC1ContractError(f"malformed profile {profile_id}")
        surface_fp_ids: list[str] = []
        empty_frame_fp = 0
        per_stream: dict[str, Any] = {}
        for stream_id, stream in profile["per_stream"].items():
            index = _candidate_record_index(records_by_profile[profile_id][stream_id])
            metrics = stream["metrics_by_iou"]["0.50"]
            false_ids = tuple(metrics["false_positive_candidate_ids"])
            stream_surface = [
                candidate_id
                for candidate_id in false_ids
                if bool(index.get(candidate_id, {}).get("surface_like"))
            ]
            surface_fp_ids.extend(stream_surface)
            stream_empty_fp = sum(
                int(row["fp"])
                for row in metrics["per_frame"]
                if int(row["gt_count"]) == 0
            )
            empty_frame_fp += stream_empty_fp
            all_records = records_by_profile[profile_id][stream_id]
            per_stream[stream_id] = {
                "candidate_count": sum(
                    1 for record in all_records if record["geometry_rejected"] is False
                ),
                "geometry_rejected_count": sum(
                    1 for record in all_records if record["geometry_rejected"] is True
                ),
                "surface_like_fp_count": len(stream_surface),
                "empty_frame_fp_count": stream_empty_fp,
                "recall_at_0_50": metrics["recall"],
            }
        all_records = [
            record
            for stream_records in records_by_profile[profile_id].values()
            for record in stream_records
        ]
        large_phrases: dict[str, int] = {}
        for record in all_records:
            if not record["surface_like"]:
                continue
            phrase = str(record["phrase"]).strip().casefold() or "<empty>"
            large_phrases[phrase] = large_phrases.get(phrase, 0) + 1
        output[profile_id] = {
            "candidate_count": sum(
                1 for record in all_records if record["geometry_rejected"] is False
            ),
            "geometry_rejected_count": sum(
                1 for record in all_records if record["geometry_rejected"] is True
            ),
            "surface_like_candidate_count": sum(
                1
                for record in all_records
                if record["geometry_rejected"] is False and record["surface_like"]
            ),
            "surface_like_fp_count": len(surface_fp_ids),
            "surface_like_false_positive_candidate_ids": sorted(surface_fp_ids),
            "empty_frame_fp_count": empty_frame_fp,
            "large_candidate_phrase_counts_before_geometry": dict(
                sorted(large_phrases.items())
            ),
            "per_stream": per_stream,
        }
    return output


def _selection_values(
    development_result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    profile = development_result["profiles"][profile_id]
    at_50 = profile["summary_by_iou"]["0.50"]
    at_70 = profile["summary_by_iou"]["0.70"]
    return {
        "profile_id": profile_id,
        "macro_f1_at_0_50": at_50["scene_group_macro"]["f1"],
        "macro_recall_at_0_50": at_50["scene_group_macro"]["recall"],
        "worst_scene_f1_at_0_50": at_50["worst_scene_group"]["metrics"]["f1"],
        "macro_f1_at_0_70": at_70["scene_group_macro"]["f1"],
        "surface_like_fp_count": diagnostics[profile_id]["surface_like_fp_count"],
        "empty_frame_fp_count": diagnostics[profile_id]["empty_frame_fp_count"],
        "per_stream_recall_at_0_50": {
            stream_id: stream["metrics_by_iou"]["0.50"]["recall"]
            for stream_id, stream in profile["per_stream"].items()
        },
    }


def _split_profile_id(profile_id: str) -> tuple[str, str]:
    parts = profile_id.split("__", 1)
    if len(parts) != 2:
        raise GateC1ContractError(f"malformed profile ID: {profile_id}")
    return parts[0], parts[1]


def _simplicity_key(profile_id: str) -> tuple[int, int, str]:
    prompt_id, geometry_id = _split_profile_id(profile_id)
    prompt_order = {"p_object": 0, "p_item": 1, "p_foreground_object": 2}
    geometry_order = {
        "g_none": 0,
        "g_frame_080": 1,
        "g_surface_span_055_090": 2,
    }
    return (
        prompt_order.get(prompt_id, 99),
        geometry_order.get(geometry_id, 99),
        profile_id,
    )


def select_profile(
    development_result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered safety gates and deterministic ranking."""

    baseline_id = _profile_id("p_object", "g_none")
    baseline = _selection_values(development_result, diagnostics, baseline_id)
    baseline_surface = int(baseline["surface_like_fp_count"])
    selectable_prompts = {item.prompt_id for item in PROMPTS if item.selectable}
    audit: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for prompt_id in sorted(selectable_prompts):
        for geometry in GEOMETRIES:
            profile_id = _profile_id(prompt_id, geometry.geometry_id)
            values = _selection_values(development_result, diagnostics, profile_id)
            recall_drop = float(baseline["macro_recall_at_0_50"]) - float(
                values["macro_recall_at_0_50"]
            )
            worst_drop = float(baseline["worst_scene_f1_at_0_50"]) - float(
                values["worst_scene_f1_at_0_50"]
            )
            stream_drops = {
                stream_id: float(baseline["per_stream_recall_at_0_50"][stream_id])
                - float(recall)
                for stream_id, recall in values["per_stream_recall_at_0_50"].items()
            }
            reduction = (
                0.0
                if baseline_surface == 0
                else (baseline_surface - int(values["surface_like_fp_count"]))
                / baseline_surface
            )
            reasons: list[str] = []
            if profile_id == baseline_id:
                reasons.append("frozen_baseline_not_challenger")
            if recall_drop > MACRO_RECALL_MAX_DROP + 1e-12:
                reasons.append("macro_recall_drop")
            if any(drop > STREAM_RECALL_MAX_DROP + 1e-12 for drop in stream_drops.values()):
                reasons.append("per_stream_recall_drop")
            if worst_drop > 1e-12:
                reasons.append("worst_scene_f1_drop")
            if reduction + 1e-12 < SURFACE_REDUCTION_MINIMUM:
                reasons.append("insufficient_surface_fp_reduction")
            row = {
                **values,
                "surface_like_fp_reduction_fraction": reduction,
                "macro_recall_drop_vs_baseline": recall_drop,
                "worst_scene_f1_drop_vs_baseline": worst_drop,
                "max_stream_recall_drop_vs_baseline": max(stream_drops.values()),
                "eligible": not reasons,
                "ineligibility_reasons": reasons,
            }
            audit.append(row)
            if not reasons:
                eligible.append(row)

    if not eligible:
        return {
            "status": "fallback_to_frozen_baseline",
            "selected_profile_id": baseline_id,
            "baseline": baseline,
            "eligible_profile_ids": [],
            "audit": audit,
        }

    best_reduction = max(float(row["surface_like_fp_reduction_fraction"]) for row in eligible)
    reduction_group = [
        row
        for row in eligible
        if math.isclose(
            float(row["surface_like_fp_reduction_fraction"]),
            best_reduction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    best_f1 = max(float(row["macro_f1_at_0_50"]) for row in reduction_group)
    f1_group = [
        row
        for row in reduction_group
        if float(row["macro_f1_at_0_50"]) >= best_f1 - SELECTION_F1_TOLERANCE - 1e-12
    ]
    selected = sorted(
        f1_group,
        key=lambda row: (
            -float(row["worst_scene_f1_at_0_50"]),
            -float(row["macro_recall_at_0_50"]),
            -float(row["macro_f1_at_0_70"]),
            _simplicity_key(str(row["profile_id"])),
        ),
    )[0]
    return {
        "status": "selected_by_preregistered_rule",
        "selected_profile_id": selected["profile_id"],
        "baseline": baseline,
        "selected": selected,
        "eligible_profile_ids": sorted(row["profile_id"] for row in eligible),
        "surface_reduction_tie_group": sorted(
            row["profile_id"] for row in reduction_group
        ),
        "f1_tolerance_group": sorted(row["profile_id"] for row in f1_group),
        "audit": audit,
    }


def _raw_payload(row: grounding.RawPrediction) -> dict[str, Any]:
    return {
        "frame_id": row.frame_id,
        "frame_index": row.frame_index,
        "prediction_index": row.prediction_index,
        "score": row.score,
        "phrase": row.phrase,
        "bbox": {
            "x": row.bbox.x,
            "y": row.bbox.y,
            "width": row.bbox.width,
            "height": row.bbox.height,
        },
    }


def _selected_candidate_manifest(
    *,
    selected_profile_id: str,
    records_by_profile: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    inventory: Sequence[DevelopmentStream],
    frame_sizes: Mapping[str, Mapping[str, ImageSize]],
) -> dict[str, Any]:
    prompt_id, geometry_id = _split_profile_id(selected_profile_id)
    streams: dict[str, Any] = {}
    for item in inventory:
        records = [
            dict(record)
            for record in records_by_profile[selected_profile_id][item.stream_id]
            if record["geometry_rejected"] is False
        ]
        streams[item.stream_id] = {
            "scene_group_id": item.scene_group_id,
            "stream_directory": item.stream_directory.as_posix(),
            "annotation_path": item.annotation_path.as_posix(),
            "frame_sizes": {
                frame_id: {"width": size.width, "height": size.height}
                for frame_id, size in sorted(frame_sizes[item.stream_id].items())
            },
            "candidates": records,
        }
    return {
        "schema_version": SELECTED_CANDIDATE_SCHEMA,
        "scope": "development_only_selected_candidates",
        "heldout_access": "none",
        "selected_profile_id": selected_profile_id,
        "prompt_id": prompt_id,
        "geometry_id": geometry_id,
        "score_threshold": FROZEN_SCORE_THRESHOLD,
        "nms_iou": FROZEN_NMS_IOU,
        "streams": streams,
    }


def run_benchmark(
    *,
    model_directory: Path,
    expected_model_sha256: str,
    baseline_report_path: Path,
    benchmark_spec_path: Path,
    reviewed_root: Path,
    device: str,
    output_path: Path,
    selected_candidates_output_path: Path,
) -> dict[str, Any]:
    """Run all new RGB prompts first, then open development annotations."""

    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    baseline_raw = load_gate_c1_baseline_raw(baseline_report_path, inventory)
    model_root = model_directory.resolve(strict=True)
    checkpoint = model_root / "model.safetensors"
    if not checkpoint.is_file() or _sha256(checkpoint) != expected_model_sha256.casefold():
        raise GateC1ContractError("Grounding DINO model.safetensors SHA-256 mismatch")

    decoded_by_stream: dict[str, Any] = {}
    sizes_by_stream: dict[str, dict[str, ImageSize]] = {}
    expected_frames: dict[str, set[str]] = {}
    for item in inventory:
        decoded = grounding._load_stream(item.stream_directory)
        decoded_by_stream[item.stream_id] = decoded
        sizes_by_stream[item.stream_id] = _frame_sizes(decoded)
        expected_frames[item.stream_id] = set(sizes_by_stream[item.stream_id])
        if len(expected_frames[item.stream_id]) != item.frame_count:
            raise GateC1ContractError(f"decoded frame count mismatch for {item.stream_id}")
        if {row.frame_id for row in baseline_raw[item.stream_id]} - expected_frames[item.stream_id]:
            raise GateC1ContractError(f"baseline raw frame mismatch for {item.stream_id}")

    import torch
    import transformers

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    processor, model = grounding._load_model_processor(
        model_root, torch_module=torch, device=device
    )
    raw_by_prompt: dict[str, dict[str, tuple[grounding.RawPrediction, ...]]] = {
        "p_object": baseline_raw
    }
    runtime_by_prompt: dict[str, Any] = {
        "p_object": {
            "source": "reused_gate_c1_raw_report",
            "report": _file_provenance(baseline_report_path),
        }
    }
    started = time.perf_counter()
    for prompt in PROMPTS:
        if prompt.source != "new_inference":
            continue
        raw_by_stream: dict[str, tuple[grounding.RawPrediction, ...]] = {}
        runtime_by_stream: dict[str, Any] = {}
        for item in inventory:
            raw, runtime = _infer_stream_prompt(
                decoded_by_stream[item.stream_id],
                processor=processor,
                model=model,
                torch_module=torch,
                device=device,
                prompt=prompt.text,
            )
            raw_by_stream[item.stream_id] = raw
            runtime_by_stream[item.stream_id] = runtime
        raw_by_prompt[prompt.prompt_id] = raw_by_stream
        runtime_by_prompt[prompt.prompt_id] = {
            "source": "new_rgb_inference",
            "aggregate": grounding._aggregate_runtime(tuple(runtime_by_stream.values())),
            "per_stream": runtime_by_stream,
        }

    # No annotation has been opened above this boundary.
    predictions_by_profile: dict[str, dict[str, tuple[PredictedCandidate, ...]]] = {}
    records_by_profile: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for prompt in PROMPTS:
        geometries = GEOMETRIES if prompt.selectable else (GEOMETRIES[0],)
        for geometry in geometries:
            profile_id = _profile_id(prompt.prompt_id, geometry.geometry_id)
            predictions_by_profile[profile_id] = {}
            records_by_profile[profile_id] = {}
            for item in inventory:
                candidates, records = candidates_for_profile(
                    raw_by_prompt[prompt.prompt_id][item.stream_id],
                    prompt=prompt,
                    geometry=geometry,
                    image_sizes=sizes_by_stream[item.stream_id],
                )
                predictions_by_profile[profile_id][item.stream_id] = candidates
                records_by_profile[profile_id][item.stream_id] = records

    development_result = evaluate_profiles(tuple(inventory), predictions_by_profile)
    diagnostics = _profile_diagnostics(development_result, records_by_profile)
    selection = select_profile(development_result, diagnostics)
    selected_manifest = _selected_candidate_manifest(
        selected_profile_id=selection["selected_profile_id"],
        records_by_profile=records_by_profile,
        inventory=inventory,
        frame_sizes=sizes_by_stream,
    )
    write_canonical_json_atomic(selected_candidates_output_path, selected_manifest)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": "development_only",
        "heldout_access": "none",
        "ground_truth_boundary": "all_prompt_rgb_inference_completed_before_annotation_evaluation",
        "preregistration": "ai-docs/research/ocid-gate-c1-1-extractor-hardening.md",
        "benchmark": _file_provenance(benchmark_spec_path),
        "reviewed_root": reviewed_root.resolve(strict=True).as_posix(),
        "model": {
            "family": grounding.MODEL_FAMILY,
            "directory": model_root.as_posix(),
            "model_safetensors_sha256": _sha256(checkpoint),
            "hf_revision": grounding.HF_REVISION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": device,
            "dtype": grounding._model_dtype(model),
            "local_files_only": True,
        },
        "frozen_postprocess": {
            "raw_score_floor": grounding.RAW_POSTPROCESS_FLOOR,
            "score_threshold": FROZEN_SCORE_THRESHOLD,
            "class_agnostic_nms_iou": FROZEN_NMS_IOU,
        },
        "prompt_profiles": [
            {
                "prompt_id": item.prompt_id,
                "text": item.text,
                "selectable": item.selectable,
                "source": item.source,
            }
            for item in PROMPTS
        ],
        "geometry_profiles": [
            {
                "geometry_id": item.geometry_id,
                "min_area_ratio": item.min_area_ratio,
                "min_span_ratio": item.min_span_ratio,
            }
            for item in GEOMETRIES
        ],
        "raw_predictions": {
            prompt_id: {
                stream_id: [_raw_payload(row) for row in rows]
                for stream_id, rows in sorted(streams.items())
            }
            for prompt_id, streams in sorted(raw_by_prompt.items())
        },
        "runtime_by_prompt": runtime_by_prompt,
        "development_result": development_result,
        "surface_diagnostics": diagnostics,
        "selection": selection,
        "selected_candidates": {
            "path": selected_candidates_output_path.resolve().as_posix(),
            "schema_version": SELECTED_CANDIDATE_SCHEMA,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_canonical_json_atomic(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-candidates-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(
            model_directory=args.model_directory,
            expected_model_sha256=args.expected_model_sha256,
            baseline_report_path=args.baseline_report,
            benchmark_spec_path=args.benchmark_spec,
            reviewed_root=args.reviewed_root,
            device=args.device,
            output_path=args.output,
            selected_candidates_output_path=args.selected_candidates_output,
        )
    except (GateC1ContractError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
