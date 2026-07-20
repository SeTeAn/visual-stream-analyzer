"""Build a reproducible, ground-truth-only structural audit of OCID.

The output is evaluation evidence.  It must never be consumed by the
production ``analyze`` path.  Numeric OCID labels are treated as physical
instance identifiers local to one sequence; this tool does not infer visual
types, select dataset splits, or run models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


AUDIT_SCHEMA_ID = "ocid_structural_audit"
AUDIT_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_ID = "ocid_structural_audit_artifact_manifest"
MANIFEST_SCHEMA_VERSION = "1.0.0"
SCOPE = "evaluation_only_dataset_audit"
DEFAULT_SEVERE_AREA_RATIO = 0.50
DEFAULT_OCCLUDER_COVERAGE = 0.50
SOURCE_FINGERPRINT_METHOD = (
    "sha256 over lexicographically sorted UTF-8 records: "
    "relative_posix_path\\0size_bytes\\0sha256(content)\\n"
)
COMPONENT_STRUCTURE = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)

INVENTORY_COLUMNS = (
    "sequence_id",
    "scene_group_id",
    "source_sequence",
    "paired_scene_key",
    "family",
    "surface",
    "camera",
    "scene_variant",
    "frame_count",
    "width",
    "height",
    "support_label",
    "object_label_minimum",
    "expected_normal_support_label",
    "expected_normal_object_label_minimum",
    "support_label_matches_expected",
    "unexpected_label_offset",
    "physical_instance_count",
    "initial_object_count",
    "final_object_count",
    "maximum_object_count",
    "added_occurrence_count",
    "removed_occurrence_count",
    "returned_occurrence_count",
    "visibility_drop_count",
    "severe_visibility_drop_candidate_count",
    "occlusion_candidate_count",
    "border_exit_candidate_count",
    "multi_component_observation_count",
    "border_touch_observation_count",
    "source_file_count",
    "source_size_bytes",
    "source_sha256",
)

SUSPICIOUS_COLUMNS = (
    "sequence_id",
    "source_sequence",
    "paired_scene_key",
    "from_frame_id",
    "to_frame_id",
    "from_source_filename",
    "to_source_filename",
    "physical_instance_id",
    "signal_type",
    "proxy_only",
    "previous_visible_pixels",
    "current_visible_pixels",
    "visible_area_ratio",
    "lost_visible_pixels",
    "lost_area_reassigned_to_objects_pixels",
    "lost_area_reassigned_to_objects",
    "removed_object_occluder_pixels",
    "removed_object_occluder_coverage",
    "previous_border_touch",
)


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SequenceDescriptor:
    path: Path
    source_sequence: str
    sequence_id: str
    scene_group_id: str
    paired_scene_key: str
    family: str
    surface: str
    camera: str
    scene_variant: str


@dataclass(frozen=True, slots=True)
class AuditResult:
    output_directory: Path
    sequence_count: int
    scene_group_count: int
    frame_count: int
    source_sha256: str


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_ratio(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be greater than 0 and at most 1.")
    return result


def _validate_expected(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when provided.")
    return int(value)


def _stable_identifier(source_sequence: str) -> str:
    safe = source_sequence.casefold().replace("-", "_").replace("/", "_")
    return f"ocid_{safe}"


def _scene_group_identifier(paired_scene_key: str) -> str:
    digest = hashlib.sha256(paired_scene_key.encode("utf-8")).hexdigest()[:16]
    return f"ocid_scene_{digest}"


def _one_named_part(parts: Sequence[str], choices: set[str], name: str) -> tuple[int, str]:
    matches = [(index, value.casefold()) for index, value in enumerate(parts) if value.casefold() in choices]
    if len(matches) != 1:
        expected = "/".join(sorted(choices))
        raise ValueError(
            f"OCID sequence path must contain exactly one {name} ({expected}): "
            f"{'/'.join(parts)}."
        )
    return matches[0]


def _describe_sequence(root: Path, path: Path) -> SequenceDescriptor:
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ValueError(f"OCID sequence resolves outside the dataset root: {path}.")
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 4 or not parts[-1].casefold().startswith("seq"):
        raise ValueError(f"Directory does not look like an OCID sequence: {relative.as_posix()}.")
    surface_index, surface = _one_named_part(parts, {"floor", "table"}, "surface")
    camera_index, camera = _one_named_part(parts, {"bottom", "top"}, "camera")
    if surface_index >= camera_index:
        raise ValueError(
            f"OCID surface must precede camera in sequence path: {relative.as_posix()}."
        )
    paired_parts = [part for index, part in enumerate(parts) if index != camera_index]
    paired_scene_key = "/".join(paired_parts)
    scene_variant = "/".join(parts[camera_index + 1 : -1])
    source_sequence = relative.as_posix()
    return SequenceDescriptor(
        path=resolved,
        source_sequence=source_sequence,
        sequence_id=_stable_identifier(source_sequence),
        scene_group_id=_scene_group_identifier(paired_scene_key),
        paired_scene_key=paired_scene_key,
        family=parts[0],
        surface=surface,
        camera=camera,
        scene_variant=scene_variant,
    )


def discover_sequences(ocid_root: Path) -> list[SequenceDescriptor]:
    """Discover OCID sequence parents containing exactly ``rgb`` and ``label`` peers."""

    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")

    discovered: list[SequenceDescriptor] = []
    for current, directory_names, _ in os.walk(root, followlinks=False):
        directory_names.sort(key=lambda value: (value.casefold(), value))
        names = set(directory_names)
        has_rgb = "rgb" in names
        has_label = "label" in names
        if has_rgb != has_label:
            missing = "label" if has_rgb else "rgb"
            raise ValueError(f"OCID sequence directory is missing {missing}/: {current}.")
        if has_rgb and has_label:
            parent = Path(current)
            discovered.append(_describe_sequence(root, parent))
            directory_names[:] = [name for name in directory_names if name not in {"rgb", "label"}]

    if not discovered:
        raise ValueError(f"No OCID sequence parents containing rgb/ and label/ were found: {root}.")
    discovered.sort(key=lambda item: item.source_sequence)
    identifiers = [item.sequence_id for item in discovered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Deterministic OCID sequence identifiers are not unique.")
    scene_group_keys: dict[str, str] = {}
    for item in discovered:
        if item.scene_group_id == item.sequence_id:
            raise ValueError("A scene_group_id must be distinct from every sequence_id.")
        prior = scene_group_keys.setdefault(item.scene_group_id, item.paired_scene_key)
        if prior != item.paired_scene_key:
            raise ValueError(
                "Deterministic scene_group_id collision between paired scene keys: "
                f"{prior!r} and {item.paired_scene_key!r}."
            )
    return discovered


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
    folded_names = [path.name.casefold() for path in files]
    if len(folded_names) != len(set(folded_names)):
        raise ValueError(f"OCID {description} filenames are not case-insensitively unique: {directory}.")
    return files


def _paired_files(sequence: SequenceDescriptor) -> list[tuple[Path, Path]]:
    rgb_files = _png_files(sequence.path / "rgb", "RGB")
    label_files = _png_files(sequence.path / "label", "label")
    rgb_names = [path.name for path in rgb_files]
    label_names = [path.name for path in label_files]
    if rgb_names != label_names:
        rgb_only = sorted(set(rgb_names) - set(label_names))
        label_only = sorted(set(label_names) - set(rgb_names))
        raise ValueError(
            "RGB and label filenames do not align for "
            f"{sequence.source_sequence}; RGB-only={rgb_only}, label-only={label_only}."
        )
    return list(zip(rgb_files, label_files, strict=True))


def _sha256_file(path: Path, root: Path) -> SourceDescriptor:
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
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _source_fingerprint(descriptors: Iterable[SourceDescriptor]) -> dict[str, object]:
    entries = sorted(descriptors, key=lambda item: item.relative_path)
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


def _rgb_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except OSError as error:
        raise ValueError(f"Cannot decode OCID RGB image {path}: {error}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"OCID RGB image has invalid dimensions: {path}.")
    return int(width), int(height)


def _label_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            labels = np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f"Cannot decode OCID label mask {path}: {error}") from error
    if labels.ndim != 2:
        raise ValueError(f"OCID label mask must be a 2D integer array, got {labels.shape}: {path}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"OCID label mask must use an integer dtype, got {labels.dtype}: {path}.")
    if np.any(labels < 0):
        raise ValueError(f"OCID label mask must not contain negative labels: {path}.")
    return labels


def _physical_instance_id(sequence_id: str, label: int) -> str:
    return f"{sequence_id}__label_{label:03d}"


def _label_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def _detect_support_label(labels: np.ndarray, source: str) -> int:
    counts = _label_counts(labels)
    nonzero = {label: count for label, count in counts.items() if label != 0}
    if not nonzero:
        raise ValueError(f"Cannot detect a support label from an all-zero first mask: {source}.")
    maximum = max(nonzero.values())
    candidates = sorted(label for label, count in nonzero.items() if count == maximum)
    if len(candidates) != 1:
        raise ValueError(
            f"First mask has no unique dominant nonzero support label in {source}: "
            f"candidates={candidates}."
        )
    return candidates[0]


def _validate_support_label(
    labels: np.ndarray,
    *,
    support_label: int,
    source: str,
) -> None:
    counts = _label_counts(labels)
    support_count = counts.get(support_label, 0)
    if support_count == 0:
        raise ValueError(f"Detected support label {support_label} is absent from {source}.")
    competing = {
        label: count
        for label, count in counts.items()
        if label > support_label and count >= support_count
    }
    if competing:
        raise ValueError(
            f"Detected support label {support_label} is not dominant over every "
            f"object-label candidate in {source}: competing={competing}."
        )


def _object_labels(labels: np.ndarray, minimum: int) -> list[int]:
    return sorted(int(value) for value in np.unique(labels) if int(value) >= minimum)


def _round_fraction(value: float) -> float:
    return round(float(value), 12)


def _frame_payload(
    labels: np.ndarray,
    *,
    sequence_id: str,
    frame_index: int,
    filename: str,
    minimum_object_label: int,
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    height, width = labels.shape
    instances: list[dict[str, object]] = []
    by_label: dict[int, dict[str, object]] = {}
    for label in _object_labels(labels, minimum_object_label):
        mask = labels == label
        rows, columns = np.nonzero(mask)
        visible_pixels = int(rows.size)
        left = int(columns.min())
        top = int(rows.min())
        right = int(columns.max())
        bottom = int(rows.max())
        _, component_count = ndimage.label(mask, structure=COMPONENT_STRUCTURE)
        border_touch = bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        )
        instance: dict[str, object] = {
            "physical_instance_id": _physical_instance_id(sequence_id, label),
            "source_label": label,
            "visible_pixels": visible_pixels,
            "visible_fraction": _round_fraction(visible_pixels / labels.size),
            "bbox": {
                "x": left,
                "y": top,
                "width": right - left + 1,
                "height": bottom - top + 1,
            },
            "centroid": {
                "x": round(float(columns.mean()), 6),
                "y": round(float(rows.mean()), 6),
            },
            "connected_component_count": int(component_count),
            "border_touch": border_touch,
        }
        instances.append(instance)
        by_label[label] = instance
    frame_id = f"frame_{frame_index:04d}"
    return (
        {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "source_filename": filename,
            "width": width,
            "height": height,
            "object_count": len(instances),
            "physical_instance_ids": [item["physical_instance_id"] for item in instances],
            "physical_instances": instances,
        },
        by_label,
    )


def _mask_iou(previous: np.ndarray, current: np.ndarray) -> float:
    union = int(np.count_nonzero(previous | current))
    if union == 0:
        return 0.0
    return _round_fraction(np.count_nonzero(previous & current) / union)


def _suspicious_row(
    *,
    sequence: SequenceDescriptor,
    previous_frame: dict[str, object],
    current_frame: dict[str, object],
    physical_instance_id: str,
    signal_type: str,
    previous_visible_pixels: int | str = "",
    current_visible_pixels: int | str = "",
    visible_area_ratio: float | str = "",
    lost_visible_pixels: int | str = "",
    reassigned_pixels: int | str = "",
    reassigned_fraction: float | str = "",
    occluder_pixels: int | str = "",
    occluder_coverage: float | str = "",
    previous_border_touch: bool | str = "",
) -> dict[str, object]:
    return {
        "sequence_id": sequence.sequence_id,
        "source_sequence": sequence.source_sequence,
        "paired_scene_key": sequence.paired_scene_key,
        "from_frame_id": previous_frame["frame_id"],
        "to_frame_id": current_frame["frame_id"],
        "from_source_filename": previous_frame["source_filename"],
        "to_source_filename": current_frame["source_filename"],
        "physical_instance_id": physical_instance_id,
        "signal_type": signal_type,
        "proxy_only": True,
        "previous_visible_pixels": previous_visible_pixels,
        "current_visible_pixels": current_visible_pixels,
        "visible_area_ratio": visible_area_ratio,
        "lost_visible_pixels": lost_visible_pixels,
        "lost_area_reassigned_to_objects_pixels": reassigned_pixels,
        "lost_area_reassigned_to_objects": reassigned_fraction,
        "removed_object_occluder_pixels": occluder_pixels,
        "removed_object_occluder_coverage": occluder_coverage,
        "previous_border_touch": previous_border_touch,
    }


def _transition_payload(
    *,
    sequence: SequenceDescriptor,
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    previous_frame: dict[str, object],
    current_frame: dict[str, object],
    previous_instances: dict[int, dict[str, object]],
    current_instances: dict[int, dict[str, object]],
    seen_labels: set[int],
    minimum_object_label: int,
    severe_area_ratio: float,
    occluder_coverage: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    previous_ids = set(previous_instances)
    current_ids = set(current_instances)
    newly_visible = current_ids - previous_ids
    returned = newly_visible & seen_labels
    added = newly_visible - seen_labels
    removed = previous_ids - current_ids
    retained = previous_ids & current_ids

    visibility_drops: list[dict[str, object]] = []
    removed_candidates: list[dict[str, object]] = []
    suspicious_rows: list[dict[str, object]] = []

    for label in sorted(retained):
        previous_visible = int(previous_instances[label]["visible_pixels"])
        current_visible = int(current_instances[label]["visible_pixels"])
        if current_visible >= previous_visible:
            continue
        previous_mask = previous_labels == label
        current_mask = current_labels == label
        lost_mask = previous_mask & ~current_mask
        lost_pixels = int(np.count_nonzero(lost_mask))
        reassigned_mask = (
            lost_mask
            & (current_labels >= minimum_object_label)
            & (current_labels != label)
        )
        reassigned_pixels = int(np.count_nonzero(reassigned_mask))
        reassigned_fraction = _round_fraction(reassigned_pixels / lost_pixels) if lost_pixels else 0.0
        area_ratio = _round_fraction(current_visible / previous_visible)
        severe = area_ratio <= severe_area_ratio
        occlusion = lost_pixels > 0 and reassigned_fraction >= occluder_coverage
        candidate_labels: list[str] = []
        if severe:
            candidate_labels.append("severe_visibility_drop_candidate")
        if occlusion:
            candidate_labels.append("occlusion_candidate")
        physical_id = _physical_instance_id(sequence.sequence_id, label)
        detail = {
            "physical_instance_id": physical_id,
            "previous_visible_pixels": previous_visible,
            "current_visible_pixels": current_visible,
            "visible_area_ratio": area_ratio,
            "lost_visible_pixels": lost_pixels,
            "lost_area_reassigned_to_objects_pixels": reassigned_pixels,
            "lost_area_reassigned_to_objects": reassigned_fraction,
            "same_id_mask_iou": _mask_iou(previous_mask, current_mask),
            "candidate_labels": candidate_labels,
        }
        visibility_drops.append(detail)
        for signal_type in candidate_labels:
            suspicious_rows.append(
                _suspicious_row(
                    sequence=sequence,
                    previous_frame=previous_frame,
                    current_frame=current_frame,
                    physical_instance_id=physical_id,
                    signal_type=signal_type,
                    previous_visible_pixels=previous_visible,
                    current_visible_pixels=current_visible,
                    visible_area_ratio=area_ratio,
                    lost_visible_pixels=lost_pixels,
                    reassigned_pixels=reassigned_pixels,
                    reassigned_fraction=reassigned_fraction,
                    previous_border_touch=bool(previous_instances[label]["border_touch"]),
                )
            )

    for label in sorted(removed):
        previous_mask = previous_labels == label
        previous_visible = int(previous_instances[label]["visible_pixels"])
        occluder_mask = (
            previous_mask
            & (current_labels >= minimum_object_label)
            & (current_labels != label)
        )
        occluder_pixels = int(np.count_nonzero(occluder_mask))
        coverage = _round_fraction(occluder_pixels / previous_visible)
        previous_border_touch = bool(previous_instances[label]["border_touch"])
        candidate_labels = ["annotation_absence_candidate"]
        if coverage >= occluder_coverage:
            candidate_labels.append("occlusion_candidate")
        if previous_border_touch:
            candidate_labels.append("border_exit_candidate")
        physical_id = _physical_instance_id(sequence.sequence_id, label)
        removed_candidates.append(
            {
                "physical_instance_id": physical_id,
                "previous_visible_pixels": previous_visible,
                "previous_border_touch": previous_border_touch,
                "removed_object_occluder_pixels": occluder_pixels,
                "removed_object_occluder_coverage": coverage,
                "candidate_labels": candidate_labels,
                "interpretation": (
                    "Annotation-visibility proxy only; it does not prove physical removal "
                    "or disappearance."
                ),
            }
        )
        for signal_type in candidate_labels:
            suspicious_rows.append(
                _suspicious_row(
                    sequence=sequence,
                    previous_frame=previous_frame,
                    current_frame=current_frame,
                    physical_instance_id=physical_id,
                    signal_type=signal_type,
                    previous_visible_pixels=previous_visible,
                    current_visible_pixels=0,
                    visible_area_ratio=0.0,
                    lost_visible_pixels=previous_visible,
                    reassigned_pixels=occluder_pixels,
                    reassigned_fraction=coverage,
                    occluder_pixels=occluder_pixels,
                    occluder_coverage=coverage,
                    previous_border_touch=previous_border_touch,
                )
            )

    for label in sorted(returned):
        physical_id = _physical_instance_id(sequence.sequence_id, label)
        suspicious_rows.append(
            _suspicious_row(
                sequence=sequence,
                previous_frame=previous_frame,
                current_frame=current_frame,
                physical_instance_id=physical_id,
                signal_type="annotation_return_candidate",
                current_visible_pixels=int(current_instances[label]["visible_pixels"]),
            )
        )

    transition = {
        "from_frame_id": previous_frame["frame_id"],
        "to_frame_id": current_frame["frame_id"],
        "from_source_filename": previous_frame["source_filename"],
        "to_source_filename": current_frame["source_filename"],
        "added_ids": [
            _physical_instance_id(sequence.sequence_id, label) for label in sorted(added)
        ],
        "removed_ids": [
            _physical_instance_id(sequence.sequence_id, label) for label in sorted(removed)
        ],
        "returned_ids": [
            _physical_instance_id(sequence.sequence_id, label) for label in sorted(returned)
        ],
        "retained_ids": [
            _physical_instance_id(sequence.sequence_id, label) for label in sorted(retained)
        ],
        "count_delta": len(current_ids) - len(previous_ids),
        "visibility_drops": visibility_drops,
        "removed_observation_candidates": removed_candidates,
        "interpretation": (
            "All transition labels describe changes in observable instance masks, not "
            "verified physical scene events."
        ),
    }
    return transition, suspicious_rows


def _audit_sequence(
    *,
    root: Path,
    sequence: SequenceDescriptor,
    severe_area_ratio: float,
    occluder_coverage: float,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[SourceDescriptor]]:
    pairs = _paired_files(sequence)
    expected_normal_support_label = 2 if sequence.surface == "table" else 1
    expected_normal_object_label_minimum = expected_normal_support_label + 1
    support_label: int | None = None
    minimum_object_label: int | None = None
    frames: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    suspicious_rows: list[dict[str, object]] = []
    source_descriptors: list[SourceDescriptor] = []
    all_labels: set[int] = set()
    seen_labels: set[int] = set()
    expected_size: tuple[int, int] | None = None
    previous_labels: np.ndarray | None = None
    previous_frame: dict[str, object] | None = None
    previous_instances: dict[int, dict[str, object]] | None = None

    for frame_index, (rgb_path, label_path) in enumerate(pairs, start=1):
        source_descriptors.append(_sha256_file(rgb_path, root))
        source_descriptors.append(_sha256_file(label_path, root))
        rgb_size = _rgb_size(rgb_path)
        labels = _label_array(label_path)
        label_size = (int(labels.shape[1]), int(labels.shape[0]))
        if label_size != rgb_size:
            raise ValueError(
                f"RGB and label dimensions do not align for {sequence.source_sequence}/"
                f"{rgb_path.name}: RGB={rgb_size}, label={label_size}."
            )
        if expected_size is None:
            expected_size = rgb_size
        elif rgb_size != expected_size:
            raise ValueError(
                f"Frame dimensions are inconsistent in {sequence.source_sequence}: "
                f"expected {expected_size}, got {rgb_size} for {rgb_path.name}."
            )

        source_reference = f"{sequence.source_sequence}/label/{label_path.name}"
        if support_label is None:
            support_label = _detect_support_label(labels, source_reference)
            minimum_object_label = support_label + 1
        if minimum_object_label is None:
            raise AssertionError("Support-label initialization failed.")
        _validate_support_label(
            labels,
            support_label=support_label,
            source=source_reference,
        )

        frame, current_instances = _frame_payload(
            labels,
            sequence_id=sequence.sequence_id,
            frame_index=frame_index,
            filename=rgb_path.name,
            minimum_object_label=minimum_object_label,
        )
        current_labels = set(current_instances)
        all_labels.update(current_labels)
        frames.append(frame)

        if previous_labels is not None and previous_frame is not None and previous_instances is not None:
            transition, rows = _transition_payload(
                sequence=sequence,
                previous_labels=previous_labels,
                current_labels=labels,
                previous_frame=previous_frame,
                current_frame=frame,
                previous_instances=previous_instances,
                current_instances=current_instances,
                seen_labels=seen_labels,
                minimum_object_label=minimum_object_label,
                severe_area_ratio=severe_area_ratio,
                occluder_coverage=occluder_coverage,
            )
            transitions.append(transition)
            suspicious_rows.extend(rows)

        seen_labels.update(current_labels)
        previous_labels = labels
        previous_frame = frame
        previous_instances = current_instances

    if expected_size is None or support_label is None or minimum_object_label is None:
        raise AssertionError("A discovered OCID sequence unexpectedly contained no frames.")

    support_label_matches_expected = support_label == expected_normal_support_label

    object_counts = [int(frame["object_count"]) for frame in frames]
    added_count = sum(len(item["added_ids"]) for item in transitions)
    removed_count = sum(len(item["removed_ids"]) for item in transitions)
    returned_count = sum(len(item["returned_ids"]) for item in transitions)
    visibility_drop_count = sum(len(item["visibility_drops"]) for item in transitions)
    signal_types = [str(row["signal_type"]) for row in suspicious_rows]
    multi_component_count = sum(
        int(instance["connected_component_count"]) > 1
        for frame in frames
        for instance in frame["physical_instances"]
    )
    border_touch_count = sum(
        bool(instance["border_touch"])
        for frame in frames
        for instance in frame["physical_instances"]
    )
    sequence_fingerprint = _source_fingerprint(source_descriptors)
    strict_add_one = bool(transitions) and all(
        len(transition["added_ids"]) == 1
        and not transition["removed_ids"]
        and not transition["returned_ids"]
        and int(transition["count_delta"]) == 1
        for transition in transitions
    )
    signals = {
        "initial_object_count": object_counts[0],
        "final_object_count": object_counts[-1],
        "maximum_object_count": max(object_counts),
        "added_occurrence_count": added_count,
        "removed_occurrence_count": removed_count,
        "returned_occurrence_count": returned_count,
        "visibility_drop_count": visibility_drop_count,
        "severe_visibility_drop_candidate_count": signal_types.count(
            "severe_visibility_drop_candidate"
        ),
        "occlusion_candidate_count": signal_types.count("occlusion_candidate"),
        "border_exit_candidate_count": signal_types.count("border_exit_candidate"),
        "multi_component_observation_count": multi_component_count,
        "border_touch_observation_count": border_touch_count,
        "physical_observation_count": sum(object_counts),
        "strict_add_one": strict_add_one,
        "observable_removal": removed_count > 0,
        "observable_return": returned_count > 0,
    }
    sequence_payload: dict[str, object] = {
        "sequence_id": sequence.sequence_id,
        "scene_group_id": sequence.scene_group_id,
        "source_sequence": sequence.source_sequence,
        "paired_scene_key": sequence.paired_scene_key,
        "family": sequence.family,
        "surface": sequence.surface,
        "camera": sequence.camera,
        "scene_variant": sequence.scene_variant,
        "support_label": support_label,
        "object_label_minimum": minimum_object_label,
        "expected_normal_support_label": expected_normal_support_label,
        "expected_normal_object_label_minimum": expected_normal_object_label_minimum,
        "support_label_matches_expected": support_label_matches_expected,
        "unexpected_label_offset": not support_label_matches_expected,
        "label_policy": {
            "support_label_detection": "unique_dominant_nonzero_label_in_first_mask",
            "support_label_validation": (
                "present_and_larger_than_every_label_above_support_in_every_mask"
            ),
            "object_label_rule": "source_label_greater_than_support_label",
            "expected_values_are_diagnostics_only": True,
        },
        "frame_count": len(frames),
        "frame_size": {"width": expected_size[0], "height": expected_size[1]},
        "physical_instance_id_scope": "sequence_local",
        "paired_camera_identity_linkage": "not_assessed",
        "physical_instance_ids": [
            _physical_instance_id(sequence.sequence_id, label) for label in sorted(all_labels)
        ],
        "visual_type_status": "not_assessed_structurally",
        "source_fingerprint": sequence_fingerprint,
        "structural_signals": signals,
        "frames": frames,
        "transitions": transitions,
    }
    inventory_row = {
        "sequence_id": sequence.sequence_id,
        "scene_group_id": sequence.scene_group_id,
        "source_sequence": sequence.source_sequence,
        "paired_scene_key": sequence.paired_scene_key,
        "family": sequence.family,
        "surface": sequence.surface,
        "camera": sequence.camera,
        "scene_variant": sequence.scene_variant,
        "frame_count": len(frames),
        "width": expected_size[0],
        "height": expected_size[1],
        "support_label": support_label,
        "object_label_minimum": minimum_object_label,
        "expected_normal_support_label": expected_normal_support_label,
        "expected_normal_object_label_minimum": expected_normal_object_label_minimum,
        "support_label_matches_expected": support_label_matches_expected,
        "unexpected_label_offset": not support_label_matches_expected,
        "physical_instance_count": len(all_labels),
        **signals,
        "source_file_count": sequence_fingerprint["entry_count"],
        "source_size_bytes": sequence_fingerprint["total_size_bytes"],
        "source_sha256": sequence_fingerprint["sha256"],
    }
    return sequence_payload, inventory_row, suspicious_rows, source_descriptors


def _group_counts(sequences: Sequence[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for sequence in sequences:
        value = str(sequence[key])
        current = counts.setdefault(value, {"sequences": 0, "frames": 0})
        current["sequences"] += 1
        current["frames"] += int(sequence["frame_count"])
    return dict(sorted(counts.items()))


def _build_scene_groups(sequences: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    keys: dict[str, str] = {}
    for sequence in sequences:
        scene_group_id = str(sequence["scene_group_id"])
        scene_group_key = str(sequence["paired_scene_key"])
        prior_key = keys.setdefault(scene_group_id, scene_group_key)
        if prior_key != scene_group_key:
            raise ValueError(
                f"scene_group_id {scene_group_id!r} maps to multiple keys: "
                f"{prior_key!r} and {scene_group_key!r}."
            )
        buckets.setdefault(scene_group_id, []).append(sequence)

    groups: list[dict[str, object]] = []
    for scene_group_id, members_unsorted in sorted(
        buckets.items(), key=lambda item: (keys[item[0]], item[0])
    ):
        members = sorted(members_unsorted, key=lambda item: str(item["sequence_id"]))
        member_sequence_ids = [str(member["sequence_id"]) for member in members]
        cameras = sorted(str(member["camera"]) for member in members)
        frame_counts = [int(member["frame_count"]) for member in members]
        groups.append(
            {
                "scene_group_id": scene_group_id,
                "scene_group_key": keys[scene_group_id],
                "member_sequence_ids": member_sequence_ids,
                "cameras": cameras,
                "frame_counts": [
                    {
                        "sequence_id": str(member["sequence_id"]),
                        "frame_count": int(member["frame_count"]),
                    }
                    for member in members
                ],
                "frame_count_total": sum(frame_counts),
                "complete_top_bottom_pair": len(members) == 2
                and cameras == ["bottom", "top"],
                "matching_frame_count": len(set(frame_counts)) == 1,
            }
        )
    return groups


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _artifact_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": path.name, "size_bytes": size, "sha256": digest.hexdigest()}


def audit_ocid_sequences(
    *,
    ocid_root: Path,
    output_directory: Path,
    expected_sequences: int | None = None,
    expected_frames: int | None = None,
    expected_scene_groups: int | None = None,
    severe_area_ratio: float = DEFAULT_SEVERE_AREA_RATIO,
    occluder_coverage: float = DEFAULT_OCCLUDER_COVERAGE,
) -> AuditResult:
    """Audit all discovered OCID sequences and atomically publish four artifacts."""

    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")
    destination = Path(output_directory).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Audit output directory already exists: {destination}.")
    expected_sequences = _validate_expected(expected_sequences, "expected_sequences")
    expected_frames = _validate_expected(expected_frames, "expected_frames")
    expected_scene_groups = _validate_expected(expected_scene_groups, "expected_scene_groups")
    severe_area_ratio = _validate_ratio(severe_area_ratio, "severe_area_ratio")
    occluder_coverage = _validate_ratio(occluder_coverage, "occluder_coverage")

    sequence_descriptors = discover_sequences(root)
    if expected_sequences is not None and len(sequence_descriptors) != expected_sequences:
        raise ValueError(
            f"Expected {expected_sequences} OCID sequences, discovered {len(sequence_descriptors)}."
        )
    discovered_scene_group_count = len(
        {sequence.scene_group_id for sequence in sequence_descriptors}
    )
    if (
        expected_scene_groups is not None
        and discovered_scene_group_count != expected_scene_groups
    ):
        raise ValueError(
            f"Expected {expected_scene_groups} OCID scene groups, discovered "
            f"{discovered_scene_group_count}."
        )

    sequence_payloads: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    suspicious_rows: list[dict[str, object]] = []
    source_descriptors: list[SourceDescriptor] = []
    for sequence in sequence_descriptors:
        payload, inventory, suspicious, descriptors = _audit_sequence(
            root=root,
            sequence=sequence,
            severe_area_ratio=severe_area_ratio,
            occluder_coverage=occluder_coverage,
        )
        sequence_payloads.append(payload)
        inventory_rows.append(inventory)
        suspicious_rows.extend(suspicious)
        source_descriptors.extend(descriptors)

    frame_count = sum(int(sequence["frame_count"]) for sequence in sequence_payloads)
    if expected_frames is not None and frame_count != expected_frames:
        raise ValueError(f"Expected {expected_frames} OCID frames, audited {frame_count}.")

    source_fingerprint = _source_fingerprint(source_descriptors)
    scene_groups = _build_scene_groups(sequence_payloads)
    complete_scene_group_count = sum(
        bool(group["complete_top_bottom_pair"]) for group in scene_groups
    )
    observable_removal_sequence_ids = [
        str(sequence["sequence_id"])
        for sequence in sequence_payloads
        if bool(sequence["structural_signals"]["observable_removal"])
    ]
    observable_return_sequence_ids = [
        str(sequence["sequence_id"])
        for sequence in sequence_payloads
        if bool(sequence["structural_signals"]["observable_return"])
    ]
    summary: dict[str, object] = {
        "sequence_count": len(sequence_payloads),
        "scene_group_count": len(scene_groups),
        "complete_top_bottom_scene_group_count": complete_scene_group_count,
        "incomplete_scene_group_count": len(scene_groups) - complete_scene_group_count,
        "frame_count": frame_count,
        "stream_local_physical_instance_count": sum(
            len(sequence["physical_instance_ids"]) for sequence in sequence_payloads
        ),
        "physical_observation_count": sum(
            int(sequence["structural_signals"]["physical_observation_count"])
            for sequence in sequence_payloads
        ),
        "strict_add_one_sequence_count": sum(
            bool(sequence["structural_signals"]["strict_add_one"])
            for sequence in sequence_payloads
        ),
        "observable_removal_sequence_ids": observable_removal_sequence_ids,
        "observable_return_sequence_ids": observable_return_sequence_ids,
        "source_file_count": source_fingerprint["entry_count"],
        "dataset_counts": {
            "OCID": {"sequences": len(sequence_payloads), "frames": frame_count}
        },
        "family_counts": _group_counts(sequence_payloads, "family"),
        "surface_counts": _group_counts(sequence_payloads, "surface"),
        "camera_counts": _group_counts(sequence_payloads, "camera"),
        "unexpected_support_offset_sequence_count": sum(
            bool(sequence["unexpected_label_offset"]) for sequence in sequence_payloads
        ),
        "unexpected_support_offset_sequence_ids": [
            sequence["sequence_id"]
            for sequence in sequence_payloads
            if bool(sequence["unexpected_label_offset"])
        ],
        "suspicious_signal_count": len(suspicious_rows),
    }
    audit_payload: dict[str, object] = {
        "schema_id": AUDIT_SCHEMA_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "scope": SCOPE,
        "dataset": "OCID",
        "ground_truth_boundary": {
            "uses_ground_truth": True,
            "allowed_consumers": ["dataset_audit", "evaluation"],
            "forbidden_consumer": "analyze",
            "statement": (
                "Ground-truth instance masks are used only to describe dataset structure. "
                "No audit field may enter production analysis input."
            ),
        },
        "signal_semantics": {
            "proxy_only": True,
            "physical_instance_id_scope": "sequence_local",
            "paired_camera_identity_linkage": "not_assessed",
            "statement": (
                "Removed, returned, occlusion, and border-exit labels describe observable "
                "mask evidence or candidates; they do not prove physical scene events."
            ),
        },
        "parameters": {
            "support_label_detection": "unique_dominant_nonzero_label_in_first_mask",
            "support_label_validation": (
                "present_and_larger_than_every_label_above_support_in_every_mask"
            ),
            "object_label_rule": "source_label_greater_than_support_label",
            "scene_group_id_derivation": (
                "ocid_scene_ + first_16_hex_sha256(utf8(camera_removed_paired_scene_key))"
            ),
            "expected_normal_support_label_by_surface_diagnostic": {"table": 2, "floor": 1},
            "expected_normal_object_label_minimum_by_surface_diagnostic": {
                "table": 3,
                "floor": 2,
            },
            "expected_normal_values_role": "diagnostic_only",
            "expected_normal_values_used_for_object_parsing": False,
            "severe_area_ratio": severe_area_ratio,
            "occluder_coverage_threshold": occluder_coverage,
            "connected_component_connectivity": 4,
            "source_fingerprint_method": SOURCE_FINGERPRINT_METHOD,
        },
        "provenance": {
            "adapter": "tools/audit_ocid_sequences.py",
            "source_fingerprint": source_fingerprint,
        },
        "summary": summary,
        "scene_groups": scene_groups,
        "sequences": sequence_payloads,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        structural_path = staging / "structural_audit.json"
        inventory_path = staging / "sequence_inventory.csv"
        suspicious_path = staging / "suspicious_transitions.csv"
        _write_json(structural_path, audit_payload)
        _write_csv(inventory_path, INVENTORY_COLUMNS, inventory_rows)
        _write_csv(suspicious_path, SUSPICIOUS_COLUMNS, suspicious_rows)
        artifact_records = [
            _artifact_record(structural_path),
            _artifact_record(inventory_path),
            _artifact_record(suspicious_path),
        ]
        manifest_payload: dict[str, object] = {
            "schema_id": MANIFEST_SCHEMA_ID,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "scope": SCOPE,
            "source_fingerprint": source_fingerprint,
            "artifacts": artifact_records,
        }
        _write_json(staging / "artifact_manifest.json", manifest_payload)
        if destination.exists():
            raise FileExistsError(f"Audit output directory appeared during the run: {destination}.")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AuditResult(
        output_directory=destination,
        sequence_count=len(sequence_payloads),
        scene_group_count=len(scene_groups),
        frame_count=frame_count,
        source_sha256=str(source_fingerprint["sha256"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-scene-groups", type=int)
    parser.add_argument(
        "--severe-area-ratio",
        type=float,
        default=DEFAULT_SEVERE_AREA_RATIO,
        help="Current/previous visible-area ratio at or below which a drop is severe.",
    )
    parser.add_argument(
        "--occluder-coverage",
        type=float,
        default=DEFAULT_OCCLUDER_COVERAGE,
        help="Lost/removed area fraction occupied by other object labels for an occlusion candidate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_ocid_sequences(
        ocid_root=args.ocid_root,
        output_directory=args.output_directory,
        expected_sequences=args.expected_sequences,
        expected_frames=args.expected_frames,
        expected_scene_groups=args.expected_scene_groups,
        severe_area_ratio=args.severe_area_ratio,
        occluder_coverage=args.occluder_coverage,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "scope": SCOPE,
                "output_directory": str(result.output_directory),
                "sequence_count": result.sequence_count,
                "scene_group_count": result.scene_group_count,
                "frame_count": result.frame_count,
                "source_sha256": result.source_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
