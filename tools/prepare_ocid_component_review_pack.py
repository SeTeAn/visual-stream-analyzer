"""Build an evaluation-only OCID Gate B fragmented-component review pack.

This tool deliberately depends on ``prepare_ocid_gate_b_benchmark.py`` for
the frozen spec, structural-audit, source-provenance, and frame validation
boundary.  Component evidence is produced exclusively by
``stream_analysis.evaluation.component_review.analyze_component_mask``.
It never reads or writes model predictions or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))

from stream_analysis.evaluation.component_review import (  # noqa: E402
    POLICY_ID,
    ComponentAnalysis,
    analyze_component_mask,
    component_union_mask,
)


GATE_B_BUILDER_PATH = Path(__file__).with_name("prepare_ocid_gate_b_benchmark.py")
GATE_B_DEPENDENCY_ID = "tools/prepare_ocid_gate_b_benchmark.py"


def _load_gate_b_dependency():
    """Load the sibling builder without requiring ``tools`` to be a package."""

    module_name = "prepare_ocid_gate_b_benchmark"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(module_name, GATE_B_BUILDER_PATH)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load Gate B dependency from {GATE_B_BUILDER_PATH}.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


GATE_B = _load_gate_b_dependency()

PACK_SCHEMA_VERSION = "ocid-component-review-pack-2.0"
DECISIONS_SCHEMA_VERSION = "ocid-component-review-decisions-1.0"
STREAM_INDEX_SCHEMA_VERSION = "ocid-component-review-stream-index-1.0"
SCOPE = "evaluation_only_annotation_review"
ROLE_ORDER = ("development", "heldout")
BLIND_EVIDENCE_DISCLOSURES = {
    "schema_version": "ocid-blind-evidence-disclosures-1.0",
    "disclosed": [
        "case_identity",
        "source_identity",
        "source_mask_sha256",
        "temporal_rgb",
        "temporal_source_label_masks",
        "raw_bbox",
        "component_ids",
        "component_colored_masks",
        "component_markers",
        "component_bboxes",
    ],
    "withheld": [
        "exact_component_areas",
        "component_area_ratios",
        "largest_component_area",
        "automatic_proposal_threshold",
        "automatic_component_decisions",
        "automatic_proposal_bbox",
    ],
}


@dataclass(frozen=True, slots=True)
class ReviewPackResult:
    output_root: Path
    case_count: int
    stream_count: int
    development_case_count: int
    heldout_case_count: int


@dataclass(frozen=True, slots=True)
class ReviewCase:
    case_id: str
    role: str
    stream_id: str
    source_sequence: str
    frame_id: str
    frame_index: int
    source_filename: str
    source_label: int
    source_mask_sha256: str
    analysis: ComponentAnalysis
    frame: Any
    previous_frame: Any | None
    next_frame: Any | None


@dataclass(frozen=True, slots=True)
class RawObservation:
    physical_instance_id: str
    source_label: int
    connected_component_count: int


@dataclass(frozen=True, slots=True)
class RawReviewFrame:
    frame_id: str
    frame_index: int
    source_filename: str
    width: int
    height: int
    rgb_path: Path
    labels: np.ndarray
    observations: tuple[RawObservation, ...]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _bbox_from_component_ids(
    analysis: ComponentAnalysis, component_ids: Sequence[str]
) -> dict[str, int]:
    retained = set(component_ids)
    chosen = [item.bbox for item in analysis.components if item.component_id in retained]
    if not chosen:
        raise ValueError("Automatic component proposal cannot be empty.")
    left = min(item.x for item in chosen)
    top = min(item.y for item in chosen)
    right = max(item.right for item in chosen)
    bottom = max(item.bottom for item in chosen)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _case_id(stream_id: str, frame_id: str, source_label: int) -> str:
    return f"{stream_id}__{frame_id}__label_{source_label:03d}"


def _load_validated_inputs(
    *, ocid_root: Path, structural_audit: Path, spec_path: Path
) -> tuple[
    Path,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    list[Any],
    dict[str, dict[str, Any]],
    dict[str, list[Any]],
]:
    """Apply the exact Gate B provenance and selected-frame boundary."""

    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")
    audit_path = Path(structural_audit).resolve(strict=True)
    spec_file = Path(spec_path).resolve(strict=True)
    if not audit_path.is_file() or not spec_file.is_file():
        raise FileNotFoundError("Structural audit and spec must both be files.")

    spec, spec_bytes = GATE_B._load_json_object(spec_file, "Gate B spec")
    # A review pack precedes final decisions.  The final policy fields remain
    # mandatory, while the not-yet-existing decisions digest is intentionally
    # absent and no placeholder digest is fabricated.
    selections, _ = GATE_B._validate_spec(
        spec, require_component_decisions=False
    )
    expected_audit_sha = GATE_B._digest(
        GATE_B._mapping(spec.get("source"), "spec.source").get(
            "structural_audit_sha256"
        ),
        "spec.source.structural_audit_sha256",
    )
    audit_bytes = audit_path.read_bytes()
    actual_audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    if actual_audit_sha != expected_audit_sha:
        raise ValueError(
            "Structural audit SHA-256 mismatch: "
            f"spec={expected_audit_sha}, actual={actual_audit_sha}."
        )

    audit, loaded_audit_bytes = GATE_B._load_json_object(
        audit_path, "Gate A structural audit"
    )
    if loaded_audit_bytes != audit_bytes:
        raise ValueError("Structural audit changed while it was being validated.")
    audit_sequences, audit_groups = GATE_B._audit_sequence_maps(audit)
    provenance = GATE_B._mapping(audit.get("provenance"), "audit.provenance")
    recorded_source = GATE_B._mapping(
        provenance.get("source_fingerprint"),
        "audit.provenance.source_fingerprint",
    )
    audit_source_sha = GATE_B._digest(
        recorded_source.get("sha256"),
        "audit.provenance.source_fingerprint.sha256",
    )
    spec_source_sha = GATE_B._digest(
        GATE_B._mapping(spec.get("source"), "spec.source").get(
            "source_fingerprint_sha256"
        ),
        "spec.source.source_fingerprint_sha256",
    )
    if audit_source_sha != spec_source_sha:
        raise ValueError(
            "Spec source fingerprint does not match the structural audit: "
            f"spec={spec_source_sha}, audit={audit_source_sha}."
        )
    GATE_B._validate_selected_membership(
        selections=selections,
        audit_sequences=audit_sequences,
        audit_groups=audit_groups,
    )
    pairs_by_source = GATE_B._verify_complete_audit_source(
        root=root,
        audit=audit,
        sequences=audit_sequences,
    )
    return (
        root,
        spec,
        spec_bytes,
        audit,
        audit_bytes,
        selections,
        audit_sequences,
        pairs_by_source,
    )


def _collect_cases(
    *,
    selections: Sequence[Any],
    audit_sequences: dict[str, dict[str, Any]],
    pairs_by_source: dict[str, list[Any]],
) -> tuple[list[ReviewCase], dict[str, list[ReviewCase]]]:
    cases: list[ReviewCase] = []
    by_stream: dict[str, list[ReviewCase]] = {}
    for selection in selections:
        stream_id = GATE_B._stream_id(selection.source_sequence)
        _, frames = _load_and_verify_review_frames(
            selection=selection,
            audit_sequence=audit_sequences[selection.source_sequence],
            pairs=pairs_by_source[selection.source_sequence],
        )
        stream_cases: list[ReviewCase] = []
        for position, frame in enumerate(frames):
            previous_frame = frames[position - 1] if position > 0 else None
            next_frame = frames[position + 1] if position + 1 < len(frames) else None
            for observation in frame.observations:
                if observation.connected_component_count <= 1:
                    continue
                mask = np.asarray(
                    frame.labels == observation.source_label, dtype=np.bool_
                )
                analysis = analyze_component_mask(mask)
                # Frozen candidate rule: fragmentation alone is insufficient; the
                # raw tight box must differ from the largest-component box.
                if not analysis.review_required:
                    continue
                if analysis.raw_bbox == analysis.largest_bbox:
                    raise AssertionError("Component API returned an inconsistent review flag.")
                case = ReviewCase(
                    case_id=_case_id(
                        stream_id, frame.frame_id, observation.source_label
                    ),
                    role=selection.role,
                    stream_id=stream_id,
                    source_sequence=selection.source_sequence,
                    frame_id=frame.frame_id,
                    frame_index=frame.frame_index,
                    source_filename=frame.source_filename,
                    source_label=observation.source_label,
                    source_mask_sha256=analysis.source_mask_sha256,
                    analysis=analysis,
                    frame=frame,
                    previous_frame=previous_frame,
                    next_frame=next_frame,
                )
                cases.append(case)
                stream_cases.append(case)
        by_stream[stream_id] = stream_cases
    expected_order = sorted(
        cases,
        key=lambda item: (
            ROLE_ORDER.index(item.role),
            item.stream_id,
            item.frame_index,
            item.source_label,
        ),
    )
    if cases != expected_order:
        # Selection order is frozen by the spec, while stream IDs need not be
        # lexicographic.  Canonicalize the published pack explicitly.
        cases = expected_order
        by_stream = {
            stream_id: [case for case in cases if case.stream_id == stream_id]
            for stream_id in sorted(by_stream)
        }
    return cases, by_stream


def _load_and_verify_review_frames(
    *, selection: Any, audit_sequence: dict[str, Any], pairs: Sequence[Any]
) -> tuple[int, list[RawReviewFrame]]:
    """Validate selected raw frames without requiring not-yet-created decisions."""

    stream_id = GATE_B._stream_id(selection.source_sequence)
    support_label = GATE_B._integer(
        audit_sequence.get("support_label"), "audit support_label"
    )
    if audit_sequence.get("object_label_minimum") != support_label + 1:
        raise ValueError(
            f"Structural audit does not use support_label + 1 for "
            f"{selection.source_sequence}."
        )
    label_policy = GATE_B._mapping(
        audit_sequence.get("label_policy"), "audit label_policy"
    )
    if (
        label_policy.get("object_label_rule")
        != "source_label_greater_than_support_label"
    ):
        raise ValueError("Selected audit sequence uses an unsupported object-label rule.")
    if (
        label_policy.get("support_label_detection")
        != "unique_dominant_nonzero_label_in_first_mask"
    ):
        raise ValueError("Selected audit sequence uses an unsupported support-label detector.")
    if label_policy.get("expected_values_are_diagnostics_only") is not True:
        raise ValueError("Expected floor/table labels must remain diagnostic only.")
    if len(pairs) != selection.frames_per_stream:
        raise ValueError(
            f"Raw frame count mismatch for {selection.source_sequence}: expected "
            f"{selection.frames_per_stream}, found {len(pairs)}."
        )
    audit_frames = GATE_B._list(
        audit_sequence.get("frames"), "audit sequence frames"
    )
    if len(audit_frames) != len(pairs):
        raise ValueError(
            f"Structural-audit frame inventory mismatch for {selection.source_sequence}."
        )
    frame_size = GATE_B._mapping(
        audit_sequence.get("frame_size"), "audit frame_size"
    )
    expected_width = GATE_B._integer(
        frame_size.get("width"), "audit frame_size.width", minimum=1
    )
    expected_height = GATE_B._integer(
        frame_size.get("height"), "audit frame_size.height", minimum=1
    )

    frames: list[RawReviewFrame] = []
    all_physical_ids: set[str] = set()
    for index, (pair, raw_audit_frame) in enumerate(
        zip(pairs, audit_frames, strict=True), start=1
    ):
        audit_frame = GATE_B._mapping(raw_audit_frame, f"audit frame {index}")
        frame_id = f"frame_{index:04d}"
        if (
            audit_frame.get("frame_id") != frame_id
            or audit_frame.get("frame_index") != index
        ):
            raise ValueError(
                f"Structural-audit frame ID/order mismatch for "
                f"{selection.source_sequence} at position {index}."
            )
        if audit_frame.get("source_filename") != pair.filename:
            raise ValueError(
                f"Structural-audit filename/order mismatch for "
                f"{selection.source_sequence} at position {index}."
            )
        width, height = GATE_B._rgb_properties(pair.rgb_path)
        labels = GATE_B._load_label_array(pair.label_path)
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"RGB dimensions differ from the structural audit for "
                f"{selection.source_sequence}/{pair.filename}."
            )
        if labels.shape != (height, width):
            raise ValueError(
                f"RGB/label dimensions do not align for "
                f"{selection.source_sequence}/{pair.filename}."
            )
        if audit_frame.get("width") != width or audit_frame.get("height") != height:
            raise ValueError(
                f"Audit frame dimensions mismatch at "
                f"{selection.source_sequence}/{frame_id}."
            )
        if index == 1:
            detected = GATE_B._detect_support_label(
                labels, f"{selection.source_sequence}/label/{pair.filename}"
            )
            if detected != support_label:
                raise ValueError(
                    f"Detected support label {detected} does not match audited "
                    f"support label {support_label} for {selection.source_sequence}."
                )
        GATE_B._validate_support_label(
            labels,
            support_label,
            f"{selection.source_sequence}/label/{pair.filename}",
        )
        source_labels = sorted(
            int(value) for value in np.unique(labels) if int(value) > support_label
        )
        recorded_instances = [
            GATE_B._mapping(item, f"audit frame {frame_id} physical_instances item")
            for item in GATE_B._list(
                audit_frame.get("physical_instances"), "physical_instances"
            )
        ]
        if len(recorded_instances) != len(source_labels):
            raise ValueError(
                f"Audit object count mismatch at {selection.source_sequence}/{frame_id}."
            )
        observations: list[RawObservation] = []
        for source_label, recorded in zip(
            source_labels, recorded_instances, strict=True
        ):
            mask = np.asarray(labels == source_label, dtype=np.bool_)
            rows, columns = np.nonzero(mask)
            left = int(columns.min())
            top = int(rows.min())
            right = int(columns.max()) + 1
            bottom = int(rows.max()) + 1
            raw_bbox = {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            }
            component_count = GATE_B._connected_component_count(mask)
            physical_id = f"{stream_id}__label_{source_label:03d}"
            expected = {
                "physical_instance_id": physical_id,
                "source_label": source_label,
                "visible_pixels": int(rows.size),
                "visible_fraction": round(int(rows.size) / labels.size, 12),
                "bbox": raw_bbox,
                "centroid": {
                    "x": round(float(columns.mean()), 6),
                    "y": round(float(rows.mean()), 6),
                },
                "connected_component_count": component_count,
                "border_touch": bool(
                    np.any(mask[0, :])
                    or np.any(mask[-1, :])
                    or np.any(mask[:, 0])
                    or np.any(mask[:, -1])
                ),
            }
            for key, value in expected.items():
                if recorded.get(key) != value:
                    raise ValueError(
                        f"Structural-audit observation mismatch at "
                        f"{selection.source_sequence}/{frame_id}/{physical_id}.{key}: "
                        f"recorded={recorded.get(key)!r}, actual={value!r}."
                    )
            observations.append(
                RawObservation(physical_id, source_label, component_count)
            )
            all_physical_ids.add(physical_id)
        expected_ids = [item.physical_instance_id for item in observations]
        if audit_frame.get("physical_instance_ids") != expected_ids:
            raise ValueError(f"Audit physical-instance ordering mismatch at {frame_id}.")
        if audit_frame.get("object_count") != len(observations):
            raise ValueError(f"Audit object_count mismatch at {frame_id}.")
        frames.append(
            RawReviewFrame(
                frame_id=frame_id,
                frame_index=index,
                source_filename=pair.filename,
                width=width,
                height=height,
                rgb_path=pair.rgb_path,
                labels=labels,
                observations=tuple(observations),
            )
        )
    if audit_sequence.get("physical_instance_id_scope") != "sequence_local":
        raise ValueError("Gate B physical-instance proxies must be sequence-local.")
    if audit_sequence.get("paired_camera_identity_linkage") != "not_assessed":
        raise ValueError("Gate B must not claim paired-camera identity linkage.")
    recorded_all = sorted(
        GATE_B._text(item, "audit physical_instance_ids item")
        for item in GATE_B._list(
            audit_sequence.get("physical_instance_ids"), "physical_instance_ids"
        )
    )
    if recorded_all != sorted(all_physical_ids):
        raise ValueError(
            f"Audit stream instance membership mismatch for {selection.source_sequence}."
        )
    return support_label, frames


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: dict[str, int],
    *,
    scale_x: float,
    scale_y: float,
    offset: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x = offset[0] + round(bbox["x"] * scale_x)
    y = offset[1] + round(bbox["y"] * scale_y)
    right = offset[0] + round((bbox["x"] + bbox["width"]) * scale_x) - 1
    bottom = offset[1] + round((bbox["y"] + bbox["height"]) * scale_y) - 1
    draw.rectangle((x, y, max(x, right), max(y, bottom)), outline=color, width=width)


def _context_panel(
    frame: Any | None,
    *,
    source_label: int,
    title: str,
    size: tuple[int, int],
) -> Image.Image:
    panel = Image.new("RGB", size, (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    font = _font()
    if frame is None:
        draw.text((12, 12), f"{title}: stream boundary", fill=(255, 190, 96), font=font)
        return panel
    with Image.open(frame.rgb_path) as source:
        rgb = source.convert("RGB")
    available_height = size[1] - 28
    scale = min(size[0] / rgb.width, available_height / rgb.height)
    rendered_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    resized = rgb.resize(rendered_size, Image.Resampling.LANCZOS)
    x = (size[0] - rendered_size[0]) // 2
    y = 26 + (available_height - rendered_size[1]) // 2
    panel.paste(resized, (x, y))
    mask = np.asarray(frame.labels == source_label, dtype=np.uint8) * 120
    if bool(mask.any()):
        alpha = Image.fromarray(mask, mode="L").resize(rendered_size, Image.Resampling.NEAREST)
        tint = Image.new("RGB", rendered_size, (40, 220, 255))
        panel.paste(tint, (x, y), alpha)
        state = "visible"
    else:
        state = "label absent"
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(90, 90, 90), width=1)
    draw.text(
        (8, 7),
        f"{title}: {frame.frame_id} | label {source_label} {state}",
        fill=(255, 255, 255),
        font=font,
    )
    return panel


def _render_case(case: ReviewCase, output_path: Path, *, blind: bool) -> None:
    analysis = case.analysis
    component_count = len(analysis.components)
    table_rows = max(10, component_count)
    canvas_width = 1680
    context_height = 300
    detail_top = 350
    canvas_height = max(940, detail_top + 110 + table_rows * 18)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text(
        (20, 14),
        f"{case.case_id} | role={case.role} | scope={SCOPE}",
        fill=(255, 255, 255),
        font=font,
    )
    if blind:
        legend = "BLIND PRIMARY REVIEW | cyan=raw bbox | colors identify components only"
    else:
        legend = (
            "ADJUDICATION ONLY | cyan=raw bbox | green=automatic retain/proposal "
            "| magenta=proposed discard"
        )
    draw.text((20, 31), legend, fill=(210, 210, 210), font=font)

    panel_width = 540
    panel_size = (panel_width, context_height)
    for index, (frame, title) in enumerate(
        (
            (case.previous_frame, "PREV"),
            (case.frame, "CURRENT"),
            (case.next_frame, "NEXT"),
        )
    ):
        panel = _context_panel(
            frame, source_label=case.source_label, title=title, size=panel_size
        )
        canvas.paste(panel, (20 + index * (panel_width + 10), 52))

    with Image.open(case.frame.rgb_path) as source:
        detail_source = source.convert("RGB")
    detail_area = (20, detail_top, 1040, canvas_height - 20)
    available_width = detail_area[2] - detail_area[0]
    available_height = detail_area[3] - detail_area[1]
    scale = min(
        available_width / detail_source.width,
        available_height / detail_source.height,
    )
    detail_size = (
        max(1, round(detail_source.width * scale)),
        max(1, round(detail_source.height * scale)),
    )
    detail = detail_source.resize(detail_size, Image.Resampling.LANCZOS)
    offset = (
        detail_area[0] + (available_width - detail_size[0]) // 2,
        detail_area[1] + (available_height - detail_size[1]) // 2,
    )
    canvas.paste(detail, offset)
    scale_x = detail_size[0] / detail_source.width
    scale_y = detail_size[1] / detail_source.height
    retained = set(analysis.automatic_retained_component_ids) if not blind else set()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for component in analysis.components:
        number = int(component.component_id[1:])
        component_mask = analysis.label_map == number
        if blind:
            digest = hashlib.sha256(component.component_id.encode("ascii")).digest()
            color = tuple(70 + value % 170 for value in digest[:3])
        else:
            color = (
                (30, 220, 80)
                if component.component_id in retained
                else (255, 40, 190)
            )
        alpha = Image.fromarray(
            np.where(component_mask, 100, 0).astype(np.uint8), mode="L"
        ).resize(detail_size, Image.Resampling.NEAREST)
        layer = Image.new("RGBA", detail_size, color + (0,))
        layer.putalpha(alpha)
        overlay.alpha_composite(layer, offset)
        cx = offset[0] + round(component.centroid[0] * scale_x)
        cy = offset[1] + round(component.centroid[1] * scale_y)
        # Marker geometry must not encode the automatic area-threshold proposal.
        # In particular, blind reviewers must see the same marker size on both
        # sides of the 5% proposal boundary.
        radius = 7
        overlay_draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=color + (255,),
            width=3,
        )
        overlay_draw.line((cx - radius - 3, cy, cx + radius + 3, cy), fill=color + (255,), width=2)
        overlay_draw.line((cx, cy - radius - 3, cx, cy + radius + 3), fill=color + (255,), width=2)
        overlay_draw.text((cx + radius + 2, cy - 7), component.component_id, fill=color + (255,), font=font)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    _draw_box(
        draw,
        analysis.raw_bbox.as_dict(),
        scale_x=scale_x,
        scale_y=scale_y,
        offset=offset,
        color=(30, 220, 255),
        width=4,
    )
    proposal_bbox = None
    if not blind:
        proposal_bbox = _bbox_from_component_ids(
            analysis, analysis.automatic_retained_component_ids
        )
        _draw_box(
            draw,
            proposal_bbox,
            scale_x=scale_x,
            scale_y=scale_y,
            offset=offset,
            color=(30, 255, 90),
            width=3,
        )

    table_x = 1070
    draw.rectangle((table_x, detail_top, canvas_width - 20, canvas_height - 20), fill=(12, 12, 12))
    if blind:
        lines = [
            f"source_sequence: {case.source_sequence}",
            f"source: {case.source_filename} | label={case.source_label}",
            f"mask_sha256: {analysis.source_mask_sha256}",
            f"raw_bbox: {analysis.raw_bbox.as_dict()}",
            "",
            "ID    bbox(x,y,w,h)",
        ]
    else:
        lines = [
            f"source_sequence: {case.source_sequence}",
            f"source: {case.source_filename} | label={case.source_label}",
            f"mask_sha256: {analysis.source_mask_sha256}",
            f"raw_bbox: {analysis.raw_bbox.as_dict()}",
            f"largest_bbox: {analysis.largest_bbox.as_dict()}",
            f"proposal_bbox: {proposal_bbox}",
            f"threshold: secondary/largest >= {analysis.secondary_area_ratio:.3f}",
            "",
            "ID    area    ratio      bbox(x,y,w,h)      proposal",
        ]
    y = detail_top + 12
    for line in lines:
        draw.text((table_x + 12, y), line, fill=(235, 235, 235), font=font)
        y += 18
    largest_area = analysis.components[0].area
    for component in analysis.components:
        ratio = component.area / largest_area
        bbox = component.bbox
        keep = component.component_id in retained
        if blind:
            digest = hashlib.sha256(component.component_id.encode("ascii")).digest()
            color = tuple(70 + value % 170 for value in digest[:3])
            line = (
                f"{component.component_id:<5} "
                f"({bbox.x},{bbox.y},{bbox.width},{bbox.height})"
            )
        else:
            color = (70, 255, 120) if keep else (255, 80, 205)
            line = (
                f"{component.component_id:<5} {component.area:<7} {ratio:>7.4f}    "
                f"({bbox.x},{bbox.y},{bbox.width},{bbox.height})    "
                f"{'RETAIN' if keep else 'DISCARD'}"
            )
        draw.text((table_x + 12, y), line, fill=color, font=font)
        y += 18
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=6, optimize=False)


def _render_contact_sheet(
    *,
    stream_id: str,
    cases: Sequence[ReviewCase],
    staging: Path,
    output_path: Path,
    blind: bool,
) -> None:
    tile_width = 320
    tile_height = 180
    caption_height = 34
    padding = 8
    columns = min(4, max(1, math.ceil(math.sqrt(len(cases)))))
    rows = max(1, math.ceil(len(cases) / columns))
    sheet = Image.new(
        "RGB",
        (
            padding + columns * (tile_width + padding),
            30 + rows * (tile_height + caption_height + padding),
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font()
    mode = "blind primary review" if blind else "adjudication"
    draw.text(
        (padding, 8),
        f"{stream_id} | {mode} | review cases={len(cases)}",
        fill="white",
        font=font,
    )
    for position, case in enumerate(cases):
        column = position % columns
        row = position // columns
        x = padding + column * (tile_width + padding)
        y = 30 + row * (tile_height + caption_height + padding)
        case_directory = "cases_blind" if blind else "cases_adjudication"
        image_path = staging / case_directory / stream_id / f"{case.case_id}.png"
        with Image.open(image_path) as image:
            thumbnail = image.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
        sheet.paste(thumbnail, (x, y))
        draw.text(
            (x + 2, y + tile_height + 3),
            f"{case.frame_id} | label={case.source_label}\n{case.case_id}",
            fill=(255, 255, 255),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=6, optimize=False)


def _decision_entry(case: ReviewCase) -> dict[str, object]:
    evidence = case.analysis.as_dict()
    return {
        "case_id": case.case_id,
        "role": case.role,
        "stream_id": case.stream_id,
        "source_sequence": case.source_sequence,
        "frame_id": case.frame_id,
        "frame_index": case.frame_index,
        "source_filename": case.source_filename,
        "source_label": case.source_label,
        "source_mask_sha256": case.source_mask_sha256,
        "analysis": evidence,
        "automatic_retained_component_ids": list(
            case.analysis.automatic_retained_component_ids
        ),
        "review_status": "pending",
        "reviews": [],
        "adjudication": None,
        "final_component_decisions": None,
        "final_retained_component_ids": None,
        "final_confidence": None,
        "resolved_by": None,
        "notes": "",
        "author_attention": False,
    }


def _artifact_records(root: Path) -> list[dict[str, object]]:
    manifest_path = root / "component_review_manifest.json"
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        for path in files
    ]


def _validate_output_boundary(staging: Path) -> None:
    for path in staging.rglob("*"):
        relative_parts = [part.casefold() for part in path.relative_to(staging).parts]
        if "analysis_streams" in relative_parts:
            raise ValueError("Review-pack artifacts must not be placed in analysis_streams.")
        if path.is_file() and "prediction" in path.name.casefold():
            raise ValueError("Review-pack artifacts must not contain prediction outputs.")


def build_component_review_pack(
    *,
    ocid_root: Path,
    structural_audit: Path,
    spec_path: Path,
    output_root: Path,
) -> ReviewPackResult:
    """Validate provenance and atomically publish a deterministic review pack."""

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Component review output root already exists: {destination}.")
    if "analysis_streams" in {part.casefold() for part in destination.parts}:
        raise ValueError("Component review output cannot be placed in analysis_streams.")

    (
        _,
        spec,
        spec_bytes,
        audit,
        audit_bytes,
        selections,
        audit_sequences,
        pairs_by_source,
    ) = _load_validated_inputs(
        ocid_root=ocid_root,
        structural_audit=structural_audit,
        spec_path=spec_path,
    )
    cases, by_stream = _collect_cases(
        selections=selections,
        audit_sequences=audit_sequences,
        pairs_by_source=pairs_by_source,
    )
    expected_case_count = GATE_B._integer(
        GATE_B._mapping(
            spec.get("review_contract"), "spec.review_contract"
        ).get("component_review_expected_observations"),
        "spec.review_contract.component_review_expected_observations",
    )
    if len(cases) != expected_case_count:
        raise ValueError(
            "Review candidate count differs from the frozen spec: "
            f"expected={expected_case_count}, actual={len(cases)}."
        )
    source_fingerprint = GATE_B._mapping(
        GATE_B._mapping(audit.get("provenance"), "audit.provenance").get(
            "source_fingerprint"
        ),
        "audit.provenance.source_fingerprint",
    )
    source_sha = GATE_B._digest(
        source_fingerprint.get("sha256"),
        "audit.provenance.source_fingerprint.sha256",
    )
    benchmark_id = str(spec["benchmark_id"])

    # No directory, including the staging directory, is created until all spec,
    # audit, complete-source, selected-frame, and component analyses succeed.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        decisions = [_decision_entry(case) for case in cases]
        decision_payload: dict[str, object] = {
            "schema_version": DECISIONS_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "policy_id": POLICY_ID,
            "source_fingerprint_sha256": source_sha,
            "review_status": "pending",
            "decisions": decisions,
        }
        _write_json(staging / "decision_template.json", decision_payload)

        for case in cases:
            _render_case(
                case,
                staging
                / "cases_blind"
                / case.stream_id
                / f"{case.case_id}.png",
                blind=True,
            )
            _render_case(
                case,
                staging
                / "cases_adjudication"
                / case.stream_id
                / f"{case.case_id}.png",
                blind=False,
            )

        stream_index_paths: dict[str, str] = {}
        stream_contact_paths: dict[str, str | None] = {}
        for selection in selections:
            stream_id = GATE_B._stream_id(selection.source_sequence)
            stream_cases = by_stream.get(stream_id, [])
            index_path = staging / "streams" / stream_id / "index.json"
            contact_path: Path | None = None
            adjudication_contact_path: Path | None = None
            if stream_cases:
                contact_path = (
                    staging / "streams" / stream_id / "contact_sheet_blind.png"
                )
                _render_contact_sheet(
                    stream_id=stream_id,
                    cases=stream_cases,
                    staging=staging,
                    output_path=contact_path,
                    blind=True,
                )
                adjudication_contact_path = (
                    staging
                    / "streams"
                    / stream_id
                    / "contact_sheet_adjudication.png"
                )
                _render_contact_sheet(
                    stream_id=stream_id,
                    cases=stream_cases,
                    staging=staging,
                    output_path=adjudication_contact_path,
                    blind=False,
                )
            _write_json(
                index_path,
                {
                    "schema_version": STREAM_INDEX_SCHEMA_VERSION,
                    "scope": SCOPE,
                    "benchmark_id": benchmark_id,
                    "role": selection.role,
                    "stream_id": stream_id,
                    "source_sequence": selection.source_sequence,
                    "case_count": len(stream_cases),
                    "blind_contact_sheet": (
                        contact_path.relative_to(staging).as_posix()
                        if contact_path is not None
                        else None
                    ),
                    "adjudication_contact_sheet": (
                        adjudication_contact_path.relative_to(staging).as_posix()
                        if adjudication_contact_path is not None
                        else None
                    ),
                    "cases": [
                        {
                            "case_id": case.case_id,
                            "frame_id": case.frame_id,
                            "source_label": case.source_label,
                            "blind_visual_path": (
                                Path("cases_blind")
                                / stream_id
                                / f"{case.case_id}.png"
                            ).as_posix(),
                            "adjudication_visual_path": (
                                Path("cases_adjudication")
                                / stream_id
                                / f"{case.case_id}.png"
                            ).as_posix(),
                        }
                        for case in stream_cases
                    ],
                },
            )
            stream_index_paths[stream_id] = index_path.relative_to(staging).as_posix()
            stream_contact_paths[stream_id] = (
                contact_path.relative_to(staging).as_posix()
                if contact_path is not None
                else None
            )

        _validate_output_boundary(staging)
        role_totals = {
            role: sum(case.role == role for case in cases) for role in ROLE_ORDER
        }
        stream_totals = [
            {
                "role": selection.role,
                "stream_id": GATE_B._stream_id(selection.source_sequence),
                "source_sequence": selection.source_sequence,
                "case_count": len(
                    by_stream.get(GATE_B._stream_id(selection.source_sequence), [])
                ),
                "index": stream_index_paths[
                    GATE_B._stream_id(selection.source_sequence)
                ],
                "contact_sheet": stream_contact_paths[
                    GATE_B._stream_id(selection.source_sequence)
                ],
            }
            for selection in selections
        ]
        stream_totals.sort(
            key=lambda item: (
                ROLE_ORDER.index(str(item["role"])), str(item["stream_id"])
            )
        )
        artifacts = _artifact_records(staging)
        artifact_by_path = {str(item["path"]): item for item in artifacts}
        decision_artifact = artifact_by_path["decision_template.json"]
        case_manifest = [
            {
                "case_id": case.case_id,
                "role": case.role,
                "stream_id": case.stream_id,
                "frame_id": case.frame_id,
                "frame_index": case.frame_index,
                "source_label": case.source_label,
                "source_mask_sha256": case.source_mask_sha256,
                "blind_visual_path": (
                    Path("cases_blind") / case.stream_id / f"{case.case_id}.png"
                ).as_posix(),
                "adjudication_visual_path": (
                    Path("cases_adjudication")
                    / case.stream_id
                    / f"{case.case_id}.png"
                ).as_posix(),
            }
            for case in cases
        ]
        manifest_payload: dict[str, object] = {
            "schema_version": PACK_SCHEMA_VERSION,
            "scope": SCOPE,
            "status": "pending_review",
            "benchmark_id": benchmark_id,
            "policy": {
                "policy_id": POLICY_ID,
                "review_candidate_rule": "raw_bbox != largest_bbox",
                "automatic_proposal": (
                    "retain c001 and secondary components with area at least "
                    "5 percent of c001; proposal is not final truth"
                ),
                "observations_excluded": False,
                "semantic_claim": "not_assessed",
                "visual_type_claim": "not_assessed",
            },
            "provenance": {
                "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                "structural_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "source_fingerprint_sha256": source_sha,
                "gate_b_validation_dependency": GATE_B_DEPENDENCY_ID,
                "component_evidence_dependency": (
                    "src/stream_analysis/evaluation/component_review.py"
                ),
            },
            "ground_truth_boundary": {
                "scope": SCOPE,
                "uses_ground_truth": True,
                "allowed_consumers": ["annotation", "evaluation"],
                "forbidden_consumer": "analyze",
                "model_predictions_accessed": False,
                "metrics_computed": False,
                "analysis_streams_artifacts_created": False,
            },
            "review_protocol": {
                "primary_review_count": 2,
                "primary_review_is_blind": True,
                "blind_evidence_disclosures": BLIND_EVIDENCE_DISCLOSURES,
                "primary_reviewer_instruction": (
                    "Open only cases_blind and contact_sheet_blind artifacts. "
                    "Do not inspect automatic_retained_component_ids, "
                    "cases_adjudication, or adjudication contact sheets until both "
                    "independent component-level review records are complete."
                ),
                "component_decision_values": [
                    "KEEP",
                    "DROP",
                    "UNRESOLVED",
                    "INTEGRITY_ALERT",
                ],
                "confidence_values": ["HIGH", "MEDIUM", "LOW"],
                "largest_component_rule": "c001 is always retained",
                "secondary_component_rule": (
                    "Every component after c001 must receive an explicit decision."
                ),
                "future_review_record_shape": {
                    "reviewer": "string",
                    "blind_to_automatic_proposal": True,
                    "components": [
                        {
                            "component_id": "cNNN",
                            "decision": (
                                "KEEP|DROP|UNRESOLVED|INTEGRITY_ALERT"
                            ),
                            "rationale_codes": [],
                            "confidence": "HIGH|MEDIUM|LOW",
                            "evidence_frame_ids": [],
                            "notes": "",
                        }
                    ],
                    "case_notes": "",
                },
            },
            "totals": {
                "cases": len(cases),
                "streams": len(selections),
                "by_role": role_totals,
                "by_stream": stream_totals,
            },
            "decision_template": {
                "path": "decision_template.json",
                "sha256": decision_artifact["sha256"],
                "size_bytes": decision_artifact["size_bytes"],
                "review_status": "pending",
            },
            "cases": case_manifest,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_hash_algorithm": "sha256",
            "artifact_inventory_scope": "recursive_all_files_except_manifest_itself",
            "self_excluded": "component_review_manifest.json",
        }
        _write_json(staging / "component_review_manifest.json", manifest_payload)
        _validate_output_boundary(staging)
        if destination.exists():
            raise FileExistsError(
                f"Component review output root appeared during generation: {destination}."
            )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ReviewPackResult(
        output_root=destination,
        case_count=len(cases),
        stream_count=len(selections),
        development_case_count=sum(case.role == "development" for case in cases),
        heldout_case_count=sum(case.role == "heldout" for case in cases),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_component_review_pack(
        ocid_root=args.ocid_root,
        structural_audit=args.structural_audit,
        spec_path=args.spec,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "pending_review",
                "scope": SCOPE,
                "output_root": str(result.output_root),
                "case_count": result.case_count,
                "stream_count": result.stream_count,
                "development_case_count": result.development_case_count,
                "heldout_case_count": result.heldout_case_count,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
