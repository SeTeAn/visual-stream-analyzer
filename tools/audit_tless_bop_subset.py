"""Audit a T-LESS BOP'19 ZIP as evaluation-only dataset evidence.

The archive is inspected directly and is never extracted.  ``obj_id`` is
reported as a T-LESS model identifier and therefore only as a candidate
strict visual-type label.  A BOP list index (``gt_id``) is exposed as a
candidate physical-instance identifier only when its model identity and
model-origin position in world coordinates are stable across multiple views.

This tool uses ground truth.  Its outputs must not enter the production
``analyze`` path, and the selected BOP views must not be described as natural
video or as appearance/disappearance events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import stat
import statistics
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


AUDIT_SCHEMA_ID = "tless_bop_subset_structural_audit"
AUDIT_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_ID = "tless_bop_subset_audit_artifact_manifest"
MANIFEST_SCHEMA_VERSION = "1.0.0"
SCOPE = "evaluation_only_dataset_audit"
DATASET_ROOT = "test_primesense"
DEFAULT_WORLD_POSITION_TOLERANCE_MM = 1.0
MAX_JSON_MEMBER_BYTES = 64 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
IMAGE_KEY_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")
SCENE_GT_PATTERN = re.compile(
    rf"^{re.escape(DATASET_ROOT)}/(?P<scene_id>[0-9]{{6}})/scene_gt\.json$"
)

SCENE_INVENTORY_COLUMNS = (
    "scene_id",
    "source_scene_directory",
    "image_count",
    "object_count",
    "minimum_object_count",
    "maximum_object_count",
    "obj_id_multiplicities",
    "obj_id_order_stable_across_images",
    "obj_id_multiplicities_stable_across_images",
    "repeated_visual_type_count",
    "repeated_visual_type_obj_ids",
    "maximum_copies_of_one_visual_type",
    "visibility_fraction_minimum",
    "visibility_fraction_median",
    "rgb_expected_count",
    "rgb_member_count",
    "rgb_coverage",
    "mask_expected_count",
    "mask_member_count",
    "mask_coverage",
    "mask_visib_expected_count",
    "mask_visib_member_count",
    "mask_visib_coverage",
    "validated_physical_instance_count",
    "all_gt_ids_validated_as_physical_instances",
    "maximum_world_position_delta_mm",
    "strict_visual_type_obj_ids",
    "strict_visual_type_scene_candidate",
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    output_directory: Path
    scene_count: int
    image_count: int
    strict_visual_type_scene_count: int
    archive_sha256: str


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object contains a duplicate key."""


def _validate_expected_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when provided.")
    return value


def _validate_tolerance(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("world_position_tolerance_mm must be a finite non-negative number.")
    return result


def _validate_expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError("expected_archive_sha256 must contain exactly 64 hexadecimal characters.")
    return value.casefold()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    raw = info.filename
    if not raw or "\x00" in raw:
        raise ValueError("ZIP contains an empty member name or a NUL byte.")
    if "\\" in raw:
        raise ValueError(f"ZIP member paths must use forward slashes only: {raw!r}.")
    if raw.startswith("/"):
        raise ValueError(f"ZIP member path must be relative: {raw!r}.")

    logical = raw[:-1] if raw.endswith("/") else raw
    parts = logical.split("/")
    if not logical or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"ZIP member path is unsafe or non-canonical: {raw!r}.")
    if any(":" in part for part in parts):
        raise ValueError(f"ZIP member path contains a drive or alternate-stream marker: {raw!r}.")

    mode = (info.external_attr >> 16) & 0xFFFF
    member_type = stat.S_IFMT(mode)
    if member_type == stat.S_IFLNK:
        raise ValueError(f"ZIP symbolic links are not accepted: {raw!r}.")
    if member_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"ZIP contains an unsupported filesystem member: {raw!r}.")
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted ZIP members are not accepted: {raw!r}.")
    return "/".join(parts)


def _index_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_name: dict[str, zipfile.ZipInfo] = {}
    by_casefolded_name: dict[str, str] = {}

    for info in archive.infolist():
        logical_name = _safe_member_name(info)
        if logical_name in by_name:
            raise ValueError(f"ZIP contains a duplicate member path: {logical_name!r}.")
        folded = logical_name.casefold()
        if folded in by_casefolded_name:
            previous = by_casefolded_name[folded]
            raise ValueError(
                "ZIP contains case-insensitively duplicate member paths: "
                f"{previous!r} and {logical_name!r}."
            )
        by_name[logical_name] = info
        by_casefolded_name[folded] = logical_name

    for logical_name in by_name:
        parts = logical_name.split("/")
        for end in range(1, len(parts)):
            parent = "/".join(parts[:end])
            parent_info = by_name.get(parent)
            if parent_info is not None and not parent_info.is_dir():
                raise ValueError(
                    f"ZIP file member {parent!r} is also used as a parent directory."
                )
    return by_name


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    member_name: str,
) -> dict[str, object]:
    if info.is_dir():
        raise ValueError(f"Required BOP JSON member is a directory: {member_name!r}.")
    if info.file_size > MAX_JSON_MEMBER_BYTES:
        raise ValueError(
            f"BOP JSON member exceeds the {MAX_JSON_MEMBER_BYTES}-byte safety limit: "
            f"{member_name!r}."
        )
    try:
        with archive.open(info, "r") as stream:
            raw = stream.read(MAX_JSON_MEMBER_BYTES + 1)
        if len(raw) > MAX_JSON_MEMBER_BYTES:
            raise ValueError(f"BOP JSON member is too large: {member_name!r}.")
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        raise ValueError(f"Malformed JSON in {member_name!r}: {error}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"BOP JSON member must contain an object at the top level: {member_name!r}.")
    return payload


def _required_member(
    members: Mapping[str, zipfile.ZipInfo],
    member_name: str,
) -> zipfile.ZipInfo:
    try:
        info = members[member_name]
    except KeyError as error:
        raise ValueError(f"Required BOP member is missing: {member_name!r}.") from error
    if info.is_dir():
        raise ValueError(f"Required BOP member is a directory: {member_name!r}.")
    return info


def _discover_scene_ids(members: Mapping[str, zipfile.ZipInfo]) -> list[str]:
    scene_ids: list[str] = []
    for member_name, info in members.items():
        if info.is_dir() or not member_name.endswith("/scene_gt.json"):
            continue
        match = SCENE_GT_PATTERN.fullmatch(member_name)
        if match is None:
            raise ValueError(f"Malformed T-LESS scene_gt.json path: {member_name!r}.")
        scene_id = match.group("scene_id")
        if int(scene_id) <= 0:
            raise ValueError(f"T-LESS scene identifier must be positive: {scene_id!r}.")
        scene_ids.append(scene_id)
    if not scene_ids:
        raise ValueError(
            f"No {DATASET_ROOT}/<six-digit-scene>/scene_gt.json records were found."
        )
    return sorted(scene_ids, key=int)


def _image_mapping(payload: dict[str, object], context: str) -> dict[int, object]:
    result: dict[int, object] = {}
    for raw_key, value in payload.items():
        if not IMAGE_KEY_PATTERN.fullmatch(raw_key):
            raise ValueError(f"{context} contains a non-canonical image key: {raw_key!r}.")
        image_id = int(raw_key)
        if image_id > 999_999:
            raise ValueError(f"{context} image id cannot be represented by BOP filenames: {image_id}.")
        if image_id in result:
            raise ValueError(f"{context} contains duplicate numeric image id {image_id}.")
        result[image_id] = value
    if not result:
        raise ValueError(f"{context} contains no image records.")
    return dict(sorted(result.items()))


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array.")
    return value


def _integer(value: object, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}.")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite.")
    return result


def _numeric_vector(value: object, length: int, context: str) -> list[float]:
    items = _sequence(value, context)
    if len(items) != length:
        raise ValueError(f"{context} must contain exactly {length} numbers.")
    return [_number(item, f"{context}[{index}]") for index, item in enumerate(items)]


def _integer_vector(value: object, length: int, context: str) -> list[int]:
    items = _sequence(value, context)
    if len(items) != length:
        raise ValueError(f"{context} must contain exactly {length} integers.")
    return [_integer(item, f"{context}[{index}]") for index, item in enumerate(items)]


def _world_translation_mm(
    cam_r_w2c: Sequence[float],
    cam_t_w2c: Sequence[float],
    cam_t_m2c: Sequence[float],
) -> list[float]:
    camera_delta = [
        cam_t_m2c[index] - cam_t_w2c[index]
        for index in range(3)
    ]
    # BOP uses X_c = R_w2c * X_w + t_w2c.  For the model origin,
    # X_w = R_w2c^T * (t_m2c - t_w2c).
    world = [
        sum(cam_r_w2c[row * 3 + column] * camera_delta[row] for row in range(3))
        for column in range(3)
    ]
    if not all(math.isfinite(value) for value in world):
        raise ValueError("Derived model-to-world translation is not finite.")
    return world


def _validate_camera_record(value: object, context: str) -> dict[str, list[float]]:
    record = _mapping(value, context)
    for required in ("cam_K", "cam_R_w2c", "cam_t_w2c", "depth_scale"):
        if required not in record:
            raise ValueError(f"{context} is missing required field {required!r}.")
    camera = {
        "cam_K": _numeric_vector(record["cam_K"], 9, f"{context}.cam_K"),
        "cam_R_w2c": _numeric_vector(
            record["cam_R_w2c"], 9, f"{context}.cam_R_w2c"
        ),
        "cam_t_w2c": _numeric_vector(
            record["cam_t_w2c"], 3, f"{context}.cam_t_w2c"
        ),
    }
    depth_scale = _number(record["depth_scale"], f"{context}.depth_scale")
    if depth_scale <= 0.0:
        raise ValueError(f"{context}.depth_scale must be positive.")
    return camera


def _validate_gt_record(value: object, context: str) -> dict[str, object]:
    record = _mapping(value, context)
    for required in ("obj_id", "cam_R_m2c", "cam_t_m2c"):
        if required not in record:
            raise ValueError(f"{context} is missing required field {required!r}.")
    return {
        "obj_id": _integer(record["obj_id"], f"{context}.obj_id", minimum=1),
        "cam_R_m2c": _numeric_vector(
            record["cam_R_m2c"], 9, f"{context}.cam_R_m2c"
        ),
        "cam_t_m2c": _numeric_vector(
            record["cam_t_m2c"], 3, f"{context}.cam_t_m2c"
        ),
    }


def _validate_gt_info_record(value: object, context: str) -> float:
    record = _mapping(value, context)
    required_fields = (
        "bbox_obj",
        "bbox_visib",
        "px_count_all",
        "px_count_valid",
        "px_count_visib",
        "visib_fract",
    )
    for required in required_fields:
        if required not in record:
            raise ValueError(f"{context} is missing required field {required!r}.")
    _integer_vector(record["bbox_obj"], 4, f"{context}.bbox_obj")
    _integer_vector(record["bbox_visib"], 4, f"{context}.bbox_visib")
    count_all = _integer(record["px_count_all"], f"{context}.px_count_all", minimum=0)
    count_valid = _integer(
        record["px_count_valid"], f"{context}.px_count_valid", minimum=0
    )
    count_visible = _integer(
        record["px_count_visib"], f"{context}.px_count_visib", minimum=0
    )
    if count_valid > count_all or count_visible > count_all:
        raise ValueError(
            f"{context} pixel counts must independently satisfy valid <= all "
            "and visible <= all."
        )
    visibility = _number(record["visib_fract"], f"{context}.visib_fract")
    if not 0.0 <= visibility <= 1.0:
        raise ValueError(f"{context}.visib_fract must be between 0 and 1.")
    return visibility


def _member_files_under(
    members: Mapping[str, zipfile.ZipInfo],
    prefix: str,
) -> set[str]:
    return {
        member_name
        for member_name, info in members.items()
        if not info.is_dir() and member_name.startswith(prefix)
    }


def _short_paths(paths: set[str], limit: int = 5) -> list[str]:
    ordered = sorted(paths)
    if len(ordered) <= limit:
        return ordered
    return [*ordered[:limit], f"... and {len(ordered) - limit} more"]


def _validate_member_coverage(
    members: Mapping[str, zipfile.ZipInfo],
    *,
    prefix: str,
    expected: set[str],
    description: str,
) -> dict[str, object]:
    actual = _member_files_under(members, prefix)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Incomplete or malformed {description} member coverage; "
            f"missing={_short_paths(missing)}, unexpected={_short_paths(unexpected)}."
        )
    expected_count = len(expected)
    return {
        "expected_count": expected_count,
        "member_count": len(actual),
        "coverage": 1.0 if expected_count else 1.0,
    }


def _maximum_pairwise_distance(points: Sequence[Sequence[float]]) -> float | None:
    if len(points) < 2:
        return None
    maximum = 0.0
    for left_index, left in enumerate(points[:-1]):
        for right in points[left_index + 1 :]:
            maximum = max(maximum, math.dist(left, right))
    return maximum


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _physical_instance_candidates(
    *,
    scene_id: str,
    image_ids: Sequence[int],
    objects_by_image: Mapping[int, list[dict[str, object]]],
    tolerance_mm: float,
) -> list[dict[str, object]]:
    maximum_object_count = max(len(objects_by_image[image_id]) for image_id in image_ids)
    candidates: list[dict[str, object]] = []
    for gt_id in range(maximum_object_count):
        observations: list[tuple[int, dict[str, object]]] = []
        for image_id in image_ids:
            objects = objects_by_image[image_id]
            if gt_id < len(objects):
                observations.append((image_id, objects[gt_id]))

        reference_obj_id = (
            int(observations[0][1]["obj_id"]) if observations else None
        )
        present_in_all_images = len(observations) == len(image_ids)
        obj_id_stable = present_in_all_images and all(
            int(record["obj_id"]) == reference_obj_id for _, record in observations
        )
        points = [record["world_translation_mm"] for _, record in observations]
        maximum_delta = _maximum_pairwise_distance(points)
        sufficient_view_count = len(image_ids) >= 2 and len(observations) >= 2
        world_position_stable = (
            present_in_all_images
            and sufficient_view_count
            and maximum_delta is not None
            and maximum_delta <= tolerance_mm
        )
        eligible = bool(obj_id_stable and world_position_stable)
        unstable_image_ids = [
            image_id
            for image_id, record in observations
            if int(record["obj_id"]) != reference_obj_id
        ]
        missing_image_ids = [
            image_id
            for image_id in image_ids
            if gt_id >= len(objects_by_image[image_id])
        ]
        candidates.append(
            {
                "gt_id": gt_id,
                "reference_obj_id": reference_obj_id,
                "observation_count": len(observations),
                "present_in_all_images": present_in_all_images,
                "sufficient_view_count": sufficient_view_count,
                "obj_id_stable_across_images": obj_id_stable,
                "world_position_stable_within_tolerance": world_position_stable,
                "maximum_pairwise_world_position_delta_mm": _rounded(maximum_delta),
                "world_position_tolerance_mm": tolerance_mm,
                "missing_image_ids": missing_image_ids,
                "different_obj_id_image_ids": unstable_image_ids,
                "eligible_as_physical_instance_id": eligible,
                "physical_instance_id": (
                    f"tless_test_primesense_scene_{scene_id}__gt_{gt_id:06d}"
                    if eligible
                    else None
                ),
            }
        )
    return candidates


def _audit_scene(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    scene_id: str,
    tolerance_mm: float,
) -> tuple[dict[str, object], dict[str, object]]:
    base = f"{DATASET_ROOT}/{scene_id}"
    gt_name = f"{base}/scene_gt.json"
    info_name = f"{base}/scene_gt_info.json"
    camera_name = f"{base}/scene_camera.json"
    gt_payload = _read_json_member(archive, _required_member(members, gt_name), gt_name)
    info_payload = _read_json_member(
        archive, _required_member(members, info_name), info_name
    )
    camera_payload = _read_json_member(
        archive, _required_member(members, camera_name), camera_name
    )
    gt_by_image = _image_mapping(gt_payload, gt_name)
    info_by_image = _image_mapping(info_payload, info_name)
    camera_by_image = _image_mapping(camera_payload, camera_name)
    key_sets = {
        "scene_gt": set(gt_by_image),
        "scene_gt_info": set(info_by_image),
        "scene_camera": set(camera_by_image),
    }
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError(
            f"Scene {scene_id} JSON image keys do not align: "
            + ", ".join(f"{name}={sorted(keys)}" for name, keys in key_sets.items())
        )

    image_ids = list(gt_by_image)
    objects_by_image: dict[int, list[dict[str, object]]] = {}
    visibility_fractions: list[float] = []
    expected_rgb: set[str] = set()
    expected_masks: set[str] = set()
    expected_visible_masks: set[str] = set()

    for image_id in image_ids:
        image_context = f"scene {scene_id}, image {image_id}"
        camera = _validate_camera_record(camera_by_image[image_id], f"{image_context}.camera")
        gt_records = _sequence(gt_by_image[image_id], f"{image_context}.scene_gt")
        info_records = _sequence(info_by_image[image_id], f"{image_context}.scene_gt_info")
        if not gt_records:
            raise ValueError(f"{image_context}.scene_gt must contain at least one object.")
        if len(gt_records) != len(info_records):
            raise ValueError(
                f"{image_context} has {len(gt_records)} scene_gt records but "
                f"{len(info_records)} scene_gt_info records."
            )

        parsed_objects: list[dict[str, object]] = []
        for gt_id, (raw_gt, raw_info) in enumerate(zip(gt_records, info_records, strict=True)):
            gt_context = f"{image_context}.gt[{gt_id}]"
            gt_record = _validate_gt_record(raw_gt, gt_context)
            visibility_fractions.append(
                _validate_gt_info_record(raw_info, f"{image_context}.gt_info[{gt_id}]")
            )
            parsed_objects.append(
                {
                    "obj_id": gt_record["obj_id"],
                    "world_translation_mm": _world_translation_mm(
                        camera["cam_R_w2c"],
                        camera["cam_t_w2c"],
                        gt_record["cam_t_m2c"],
                    ),
                }
            )
            expected_masks.add(f"{base}/mask/{image_id:06d}_{gt_id:06d}.png")
            expected_visible_masks.add(
                f"{base}/mask_visib/{image_id:06d}_{gt_id:06d}.png"
            )
        objects_by_image[image_id] = parsed_objects
        expected_rgb.add(f"{base}/rgb/{image_id:06d}.png")

    coverage = {
        "rgb": _validate_member_coverage(
            members,
            prefix=f"{base}/rgb/",
            expected=expected_rgb,
            description=f"scene {scene_id} RGB",
        ),
        "mask": _validate_member_coverage(
            members,
            prefix=f"{base}/mask/",
            expected=expected_masks,
            description=f"scene {scene_id} mask",
        ),
        "mask_visib": _validate_member_coverage(
            members,
            prefix=f"{base}/mask_visib/",
            expected=expected_visible_masks,
            description=f"scene {scene_id} visible-mask",
        ),
    }

    orders = [
        [int(record["obj_id"]) for record in objects_by_image[image_id]]
        for image_id in image_ids
    ]
    reference_order = orders[0]
    obj_id_order_stable = all(order == reference_order for order in orders)
    different_order_image_ids = [
        image_id
        for image_id, order in zip(image_ids, orders, strict=True)
        if order != reference_order
    ]
    multiplicities = [Counter(order) for order in orders]
    reference_multiplicities = multiplicities[0]
    multiplicities_stable = all(
        counts == reference_multiplicities for counts in multiplicities
    )
    repeated_obj_ids = sorted(
        {
            obj_id
            for counts in multiplicities
            for obj_id, count in counts.items()
            if count >= 2
        }
    )
    maximum_copies = max(
        count for counts in multiplicities for count in counts.values()
    )
    candidates = _physical_instance_candidates(
        scene_id=scene_id,
        image_ids=image_ids,
        objects_by_image=objects_by_image,
        tolerance_mm=tolerance_mm,
    )
    eligible_candidates = [
        candidate for candidate in candidates if candidate["eligible_as_physical_instance_id"]
    ]
    eligible_by_obj_id = Counter(
        int(candidate["reference_obj_id"])
        for candidate in eligible_candidates
        if candidate["reference_obj_id"] is not None
    )
    strict_visual_type_obj_ids = sorted(
        obj_id for obj_id in repeated_obj_ids if eligible_by_obj_id[obj_id] >= 2
    )
    all_candidates_validated = bool(candidates) and all(
        candidate["eligible_as_physical_instance_id"] for candidate in candidates
    )
    candidate_deltas = [
        float(candidate["maximum_pairwise_world_position_delta_mm"])
        for candidate in candidates
        if candidate["maximum_pairwise_world_position_delta_mm"] is not None
    ]
    maximum_world_delta = max(candidate_deltas, default=None)
    object_counts = [len(order) for order in orders]
    obj_id_multiplicities = [
        {"obj_id": obj_id, "count": count}
        for obj_id, count in sorted(reference_multiplicities.items())
    ]

    scene_payload: dict[str, object] = {
        "scene_id": scene_id,
        "source_scene_directory": base,
        "image_ids": image_ids,
        "image_count": len(image_ids),
        "object_count": len(reference_order),
        "minimum_object_count": min(object_counts),
        "maximum_object_count": max(object_counts),
        "reference_obj_id_order": reference_order,
        "obj_id_multiplicities": obj_id_multiplicities,
        "obj_id_order_stable_across_images": obj_id_order_stable,
        "obj_id_multiplicities_stable_across_images": multiplicities_stable,
        "different_obj_id_order_image_ids": different_order_image_ids,
        "repeated_visual_type_obj_ids": repeated_obj_ids,
        "repeated_visual_type_count": len(repeated_obj_ids),
        "maximum_copies_of_one_visual_type": maximum_copies,
        "visibility_fraction": {
            "observation_count": len(visibility_fractions),
            "minimum": _rounded(min(visibility_fractions)),
            "median": _rounded(statistics.median(visibility_fractions)),
        },
        "member_coverage": coverage,
        "physical_instance_candidate_validation": {
            "candidate_source": "zero_based_scene_gt_list_index",
            "candidate_scope": "scene_local",
            "requires_multiple_views": True,
            "world_position_formula": "R_w2c^T * (t_m2c - t_w2c)",
            "world_position_tolerance_mm": tolerance_mm,
            "validated_candidate_count": len(eligible_candidates),
            "all_gt_ids_validated_as_physical_instances": all_candidates_validated,
            "maximum_world_position_delta_mm": _rounded(maximum_world_delta),
            "candidates": candidates,
        },
        "strict_visual_type_obj_ids": strict_visual_type_obj_ids,
        "strict_visual_type_scene_candidate": bool(strict_visual_type_obj_ids),
    }
    inventory_row: dict[str, object] = {
        "scene_id": scene_id,
        "source_scene_directory": base,
        "image_count": len(image_ids),
        "object_count": len(reference_order),
        "minimum_object_count": min(object_counts),
        "maximum_object_count": max(object_counts),
        "obj_id_multiplicities": obj_id_multiplicities,
        "obj_id_order_stable_across_images": obj_id_order_stable,
        "obj_id_multiplicities_stable_across_images": multiplicities_stable,
        "repeated_visual_type_count": len(repeated_obj_ids),
        "repeated_visual_type_obj_ids": repeated_obj_ids,
        "maximum_copies_of_one_visual_type": maximum_copies,
        "visibility_fraction_minimum": _rounded(min(visibility_fractions)),
        "visibility_fraction_median": _rounded(statistics.median(visibility_fractions)),
        "rgb_expected_count": coverage["rgb"]["expected_count"],
        "rgb_member_count": coverage["rgb"]["member_count"],
        "rgb_coverage": coverage["rgb"]["coverage"],
        "mask_expected_count": coverage["mask"]["expected_count"],
        "mask_member_count": coverage["mask"]["member_count"],
        "mask_coverage": coverage["mask"]["coverage"],
        "mask_visib_expected_count": coverage["mask_visib"]["expected_count"],
        "mask_visib_member_count": coverage["mask_visib"]["member_count"],
        "mask_visib_coverage": coverage["mask_visib"]["coverage"],
        "validated_physical_instance_count": len(eligible_candidates),
        "all_gt_ids_validated_as_physical_instances": all_candidates_validated,
        "maximum_world_position_delta_mm": _rounded(maximum_world_delta),
        "strict_visual_type_obj_ids": strict_visual_type_obj_ids,
        "strict_visual_type_scene_candidate": bool(strict_visual_type_obj_ids),
    }
    return scene_payload, inventory_row


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row[column]) for column in columns})


def _artifact_record(path: Path) -> dict[str, object]:
    size, sha256 = _sha256_file(path)
    return {"path": path.name, "size_bytes": size, "sha256": sha256}


def audit_tless_bop_subset(
    *,
    archive: Path,
    output_directory: Path,
    expected_scenes: int | None = None,
    expected_images: int | None = None,
    expected_archive_sha256: str | None = None,
    world_position_tolerance_mm: float = DEFAULT_WORLD_POSITION_TOLERANCE_MM,
) -> AuditResult:
    """Validate a T-LESS BOP archive and publish deterministic audit artifacts."""

    expected_scenes = _validate_expected_count(expected_scenes, "expected_scenes")
    expected_images = _validate_expected_count(expected_images, "expected_images")
    expected_archive_sha256 = _validate_expected_sha256(expected_archive_sha256)
    tolerance_mm = _validate_tolerance(world_position_tolerance_mm)

    source = Path(archive).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"T-LESS archive is not a file: {source}.")
    destination = Path(output_directory).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Audit output directory already exists: {destination}.")

    archive_size, archive_sha256 = _sha256_file(source)
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "T-LESS archive SHA-256 does not match the expected fingerprint: "
            f"expected={expected_archive_sha256}, actual={archive_sha256}."
        )

    try:
        archive_stream = zipfile.ZipFile(source, "r")
    except zipfile.BadZipFile as error:
        raise ValueError(f"T-LESS source is not a valid ZIP archive: {source}.") from error

    with archive_stream:
        members = _index_members(archive_stream)
        bad_crc_member = archive_stream.testzip()
        if bad_crc_member is not None:
            raise ValueError(f"ZIP CRC validation failed for member {bad_crc_member!r}.")
        scene_ids = _discover_scene_ids(members)
        if expected_scenes is not None and len(scene_ids) != expected_scenes:
            raise ValueError(
                f"Expected {expected_scenes} T-LESS scenes, found {len(scene_ids)}."
            )

        scene_payloads: list[dict[str, object]] = []
        inventory_rows: list[dict[str, object]] = []
        for scene_id in scene_ids:
            scene_payload, inventory_row = _audit_scene(
                archive_stream, members, scene_id, tolerance_mm
            )
            scene_payloads.append(scene_payload)
            inventory_rows.append(inventory_row)

        total_images = sum(int(scene["image_count"]) for scene in scene_payloads)
        if expected_images is not None and total_images != expected_images:
            raise ValueError(
                f"Expected {expected_images} total T-LESS images, found {total_images}."
            )
        file_members = [info for info in members.values() if not info.is_dir()]
        directory_members = [info for info in members.values() if info.is_dir()]

    strict_scenes = [
        scene for scene in scene_payloads if scene["strict_visual_type_scene_candidate"]
    ]
    unstable_order_scenes = [
        scene["scene_id"]
        for scene in scene_payloads
        if not scene["obj_id_order_stable_across_images"]
    ]
    incomplete_identity_scenes = [
        scene["scene_id"]
        for scene in scene_payloads
        if not scene["physical_instance_candidate_validation"]
        ["all_gt_ids_validated_as_physical_instances"]
    ]
    source_fingerprint: dict[str, object] = {
        "algorithm": "sha256",
        "archive_filename": source.name,
        "size_bytes": archive_size,
        "sha256": archive_sha256,
        "zip_member_count": len(members),
        "zip_file_member_count": len(file_members),
        "zip_directory_member_count": len(directory_members),
        "zip_uncompressed_file_size_bytes": sum(info.file_size for info in file_members),
        "zip_compressed_file_size_bytes": sum(info.compress_size for info in file_members),
        "crc_validation": "passed_all_members",
    }
    summary: dict[str, object] = {
        "scene_count": len(scene_payloads),
        "image_count": total_images,
        "object_observation_count": sum(
            int(scene["visibility_fraction"]["observation_count"])
            for scene in scene_payloads
        ),
        "scene_ids": [scene["scene_id"] for scene in scene_payloads],
        "scene_ids_with_repeated_obj_ids": [
            scene["scene_id"]
            for scene in scene_payloads
            if scene["repeated_visual_type_obj_ids"]
        ],
        "strict_visual_type_scene_candidate_count": len(strict_scenes),
        "strict_visual_type_scene_candidates": [
            {
                "scene_id": scene["scene_id"],
                "strict_visual_type_obj_ids": scene["strict_visual_type_obj_ids"],
                "maximum_copies_of_one_visual_type": scene[
                    "maximum_copies_of_one_visual_type"
                ],
            }
            for scene in strict_scenes
        ],
        "scene_ids_with_unstable_obj_id_order": unstable_order_scenes,
        "scene_ids_with_unvalidated_gt_ids": incomplete_identity_scenes,
    }
    audit_payload: dict[str, object] = {
        "schema_id": AUDIT_SCHEMA_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "scope": SCOPE,
        "dataset": "T-LESS test_primesense BOP'19 subset",
        "ground_truth_boundary": {
            "uses_ground_truth": True,
            "allowed_consumers": ["dataset_audit", "evaluation"],
            "forbidden_consumer": "analyze",
            "statement": (
                "BOP poses, object identities, visibility metadata, and masks are used "
                "only to validate dataset structure and evaluation suitability."
            ),
        },
        "semantics": {
            "obj_id": (
                "T-LESS model identifier; treated only as a candidate strict "
                "visual_type_id for evaluation design."
            ),
            "gt_id": (
                "Zero-based scene_gt list index; promoted to a scene-local candidate "
                "physical_instance_id only after multi-view obj_id and world-position validation."
            ),
            "natural_video": False,
            "events_assessed": False,
            "statement": (
                "Images are selected views of static BOP scenes, not an ordered natural "
                "video. This audit makes no appearance, disappearance, or tracking-event claim."
            ),
        },
        "parameters": {
            "dataset_root": DATASET_ROOT,
            "world_position_formula": "R_w2c^T * (t_m2c - t_w2c)",
            "world_position_tolerance_mm": tolerance_mm,
            "physical_instance_candidate_requires_multiple_views": True,
            "expected_scenes": expected_scenes,
            "expected_total_images": expected_images,
            "expected_archive_sha256": expected_archive_sha256,
            "member_coverage_policy": "exact_rgb_mask_and_mask_visib_sets_per_scene",
        },
        "source_archive_fingerprint": source_fingerprint,
        "summary": summary,
        "scenes": scene_payloads,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        structural_path = staging / "tless_structural_audit.json"
        inventory_path = staging / "scene_inventory.csv"
        _write_json(structural_path, audit_payload)
        _write_csv(inventory_path, SCENE_INVENTORY_COLUMNS, inventory_rows)
        artifacts = [_artifact_record(structural_path), _artifact_record(inventory_path)]
        manifest_payload: dict[str, object] = {
            "schema_id": MANIFEST_SCHEMA_ID,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "scope": SCOPE,
            "source_archive_fingerprint": source_fingerprint,
            "artifacts": artifacts,
        }
        _write_json(staging / "artifact_manifest.json", manifest_payload)
        if destination.exists():
            raise FileExistsError(
                f"Audit output directory appeared during the run: {destination}."
            )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AuditResult(
        output_directory=destination,
        scene_count=len(scene_payloads),
        image_count=total_images,
        strict_visual_type_scene_count=len(strict_scenes),
        archive_sha256=archive_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Expected total image-record count across all audited scenes.",
    )
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument(
        "--world-position-tolerance-mm",
        type=float,
        default=DEFAULT_WORLD_POSITION_TOLERANCE_MM,
        help=(
            "Maximum pairwise drift of a gt_id model origin in world coordinates "
            "for physical-instance eligibility."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_tless_bop_subset(
        archive=args.archive,
        output_directory=args.output_directory,
        expected_scenes=args.expected_scenes,
        expected_images=args.expected_images,
        expected_archive_sha256=args.expected_archive_sha256,
        world_position_tolerance_mm=args.world_position_tolerance_mm,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "scope": SCOPE,
                "output_directory": str(result.output_directory),
                "scene_count": result.scene_count,
                "image_count": result.image_count,
                "strict_visual_type_scene_count": result.strict_visual_type_scene_count,
                "archive_sha256": result.archive_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
