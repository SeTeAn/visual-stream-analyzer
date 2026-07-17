"""Prepare candidate-only OCID annotations for the saved-run evaluator.

The tool converts OCID instance-label masks into per-frame ground-truth
bounding boxes.  It writes a separate annotation file and never modifies the
RGB stream manifest consumed by ``analyze``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image


ANNOTATION_SCHEMA_VERSION = "stream-pilot-annotation-0.1"
MANIFEST_SCHEMA_VERSION = "stream-input-0.1"


@dataclass(frozen=True, slots=True)
class PreparedCandidateAnnotation:
    stream_id: str
    source_sequence: str
    annotation_path: Path
    frame_count: int
    instance_count: int
    physical_instance_count: int


def _object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}.")
    return value


def _relative_sequence(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest metadata.source_sequence must be a non-empty path.")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ValueError("manifest source_sequence must remain inside the OCID root.")
    if "table" not in posix.parts and "floor" not in posix.parts:
        raise ValueError("manifest source_sequence must identify an OCID table or floor scene.")
    return posix


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace.")
    return value


def _source_filename(frame: dict[str, object]) -> str:
    metadata = frame.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Every stream frame must contain metadata.")
    value = _text(metadata.get("ocid_source_filename"), "ocid_source_filename")
    if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
        raise ValueError("ocid_source_filename must be a filename, not a path.")
    if not value.casefold().endswith(".png"):
        raise ValueError("ocid_source_filename must reference a PNG label filename.")
    return value


def _label_array(path: Path, *, width: int, height: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            labels = np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f"Cannot decode OCID label mask {path}: {error}") from error
    if labels.ndim != 2 or labels.shape != (height, width):
        raise ValueError(
            f"OCID label mask must have shape ({height}, {width}), got {labels.shape}: {path}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"OCID label mask must use an integer dtype: {path}.")
    if np.any(labels < 0):
        raise ValueError(f"OCID label mask must not contain negative labels: {path}.")
    return labels


def _bbox(labels: np.ndarray, label: int) -> dict[str, int]:
    rows, columns = np.nonzero(labels == label)
    if rows.size == 0 or columns.size == 0:
        raise ValueError(f"Cannot derive a bbox for absent label {label}.")
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max())
    bottom = int(rows.max())
    return {
        "x": left,
        "y": top,
        "width": right - left + 1,
        "height": bottom - top + 1,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Candidate annotation already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_candidate_annotation(
    *,
    ocid_root: Path,
    stream_directory: Path,
    output_path: Path,
) -> PreparedCandidateAnnotation:
    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")
    stream_root = Path(stream_directory).resolve(strict=True)
    if not stream_root.is_dir():
        raise NotADirectoryError(f"Stream directory is not a directory: {stream_root}.")

    manifest_path = stream_root / "manifest.json"
    manifest = _object(manifest_path, "stream manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Stream manifest uses an unsupported schema_version.")
    stream_id = _text(manifest.get("stream_id"), "stream_id")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Stream manifest must contain metadata.")
    sequence = _relative_sequence(metadata.get("source_sequence"))
    frame_size = metadata.get("frame_size")
    if not isinstance(frame_size, dict):
        raise ValueError("Stream manifest metadata must contain frame_size.")
    width = _positive_int(frame_size.get("width"), "frame_size.width")
    height = _positive_int(frame_size.get("height"), "frame_size.height")

    source_sequence = root.joinpath(*sequence.parts).resolve(strict=False)
    if not _inside(source_sequence, root):
        raise ValueError("Resolved source_sequence is outside the OCID root.")
    label_root = (source_sequence / "label").resolve(strict=False)
    if not label_root.is_dir() or not _inside(label_root, root):
        raise FileNotFoundError(f"OCID label directory does not exist: {label_root}.")

    minimum_object_label = 3 if "table" in sequence.parts else 2
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Stream manifest frames must be a non-empty list.")

    instances: list[dict[str, object]] = []
    physical_labels: set[int] = set()
    frame_ids: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Stream manifest frame entries must be objects.")
        frame_id = _text(frame.get("frame_id"), "frame_id")
        if frame_id in frame_ids:
            raise ValueError(f"Duplicate frame_id in stream manifest: {frame_id}.")
        frame_ids.add(frame_id)
        source_filename = _source_filename(frame)
        label_path = (label_root / source_filename).resolve(strict=False)
        if not _inside(label_path, label_root) or not label_path.is_file():
            raise FileNotFoundError(f"OCID label mask does not exist: {label_path}.")
        labels = _label_array(label_path, width=width, height=height)
        object_labels = sorted(
            int(value) for value in np.unique(labels) if int(value) >= minimum_object_label
        )
        for label in object_labels:
            physical_labels.add(label)
            instances.append(
                {
                    "instance_id": f"ocid_label_{label:03d}_{frame_id}",
                    "frame_id": frame_id,
                    "visual_type_id": f"ocid_physical_instance_{label:03d}",
                    "bbox": _bbox(labels, label),
                    "characteristic_regions": [],
                    "uncertainty": "",
                    "notes": (
                        "Bounding box derived from an OCID instance mask for candidate-only "
                        "evaluation; visual_type_id is a physical-instance proxy."
                    ),
                }
            )

    visual_types = [
        {
            "visual_type_id": f"ocid_physical_instance_{label:03d}",
            "description": f"OCID physical-instance proxy for numeric label {label}.",
            "notes": (
                "This identifier supports candidate evaluation only and is not a "
                "ground-truth visual type."
            ),
        }
        for label in sorted(physical_labels)
    ]
    payload: dict[str, object] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "stream_id": stream_id,
        "manifest_ref": "manifest.json",
        "annotation_scope": "candidate_extraction_bbox_only",
        "frame_size": {"width": width, "height": height},
        "visual_types": visual_types,
        "expected_element_instances": instances,
        "frame_comparisons": [],
        "change_events": [],
        "supported_event_types": [],
        "allowed_event_types": [],
        "uncertainty": [],
        "notes": (
            "Candidate-only evaluation annotation generated from OCID instance masks. "
            "Grouping and event metrics are outside this annotation scope."
        ),
        "metadata": {
            "source_dataset": "OCID",
            "source_sequence": sequence.as_posix(),
            "object_label_minimum": minimum_object_label,
            "bbox_derivation": "tight_axis_aligned_bbox_from_instance_mask",
            "ground_truth_role": "evaluation_only",
            "adapter": "tools/prepare_ocid_candidate_annotations.py",
        },
    }
    destination = Path(output_path).resolve(strict=False)
    _atomic_write_json(destination, payload)
    return PreparedCandidateAnnotation(
        stream_id=stream_id,
        source_sequence=sequence.as_posix(),
        annotation_path=destination,
        frame_count=len(frames),
        instance_count=len(instances),
        physical_instance_count=len(physical_labels),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--stream-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_candidate_annotation(
        ocid_root=args.ocid_root,
        stream_directory=args.stream_directory,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "stream_id": prepared.stream_id,
                "source_sequence": prepared.source_sequence,
                "annotation_path": str(prepared.annotation_path),
                "frame_count": prepared.frame_count,
                "instance_count": prepared.instance_count,
                "physical_instance_count": prepared.physical_instance_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
