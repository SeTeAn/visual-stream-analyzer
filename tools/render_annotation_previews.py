"""Render visual previews for probe stream annotation JSON files.

The tool is intentionally independent from the production analysis pipeline:
it only needs Pillow and does not import ML, OpenCV, DINOv2, or evaluation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_STREAMS = (
    "probe_01_stationery",
    "probe_02_fruits_vegetables_berries",
    "probe_03_tableware",
    "probe_04_technical_tools",
)
EXPECTED_MANIFEST_SCHEMA = "stream-input-0.1"
EXPECTED_ANNOTATION_SCHEMA = "stream-pilot-annotation-0.1"
PREVIEW_DIR_NAME = "annotation_preview"


@dataclass(frozen=True)
class FrameEntry:
    frame_id: str
    index: int
    image_path: Path


@dataclass(frozen=True)
class AnnotationInstance:
    instance_id: str
    frame_id: str
    visual_type_id: str
    bbox: tuple[int, int, int, int]


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON at line {error.lineno}, column {error.colno}.") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return data


def require_mapping(data: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object.")
    return value


def require_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def require_int(data: dict[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be an integer.")
    return value


def relative_path(raw_path: str, context: str) -> Path:
    posix = PurePosixPath(raw_path.replace("\\", "/"))
    windows = PureWindowsPath(raw_path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ValueError(f"{context} must be a relative path inside the stream directory.")
    return Path(*posix.parts)


def load_manifest(stream_root: Path) -> tuple[str, list[FrameEntry]]:
    manifest_path = stream_root / "manifest.json"
    manifest = load_json(manifest_path)
    schema_version = require_text(manifest, "schema_version", "manifest")
    if schema_version != EXPECTED_MANIFEST_SCHEMA:
        raise ValueError(
            f"{manifest_path}: unsupported schema_version {schema_version!r}; "
            f"expected {EXPECTED_MANIFEST_SCHEMA!r}."
        )

    stream_id = require_text(manifest, "stream_id", "manifest")
    frames_raw = manifest.get("frames")
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError(f"{manifest_path}: manifest.frames must be a non-empty array.")

    frames: list[FrameEntry] = []
    seen_frame_ids: set[str] = set()
    for position, raw_frame in enumerate(frames_raw):
        context = f"manifest.frames[{position}]"
        if not isinstance(raw_frame, dict):
            raise ValueError(f"{context} must be an object.")
        frame_id = require_text(raw_frame, "frame_id", context)
        index = require_int(raw_frame, "index", context)
        image_path_raw = require_text(raw_frame, "image_path", context)
        if frame_id in seen_frame_ids:
            raise ValueError(f"{context}.frame_id duplicates {frame_id!r}.")
        seen_frame_ids.add(frame_id)
        frames.append(
            FrameEntry(
                frame_id=frame_id,
                index=index,
                image_path=stream_root / relative_path(image_path_raw, f"{context}.image_path"),
            )
        )

    frames.sort(key=lambda frame: frame.index)
    return stream_id, frames


def load_annotations(stream_root: Path, expected_stream_id: str) -> tuple[set[str], list[AnnotationInstance]]:
    annotation_path = stream_root / "annotation.json"
    annotation = load_json(annotation_path)
    schema_version = require_text(annotation, "schema_version", "annotation")
    if schema_version != EXPECTED_ANNOTATION_SCHEMA:
        raise ValueError(
            f"{annotation_path}: unsupported schema_version {schema_version!r}; "
            f"expected {EXPECTED_ANNOTATION_SCHEMA!r}."
        )
    stream_id = require_text(annotation, "stream_id", "annotation")
    if stream_id != expected_stream_id:
        raise ValueError(
            f"{annotation_path}: stream_id {stream_id!r} does not match manifest stream_id "
            f"{expected_stream_id!r}."
        )

    visual_types_raw = annotation.get("visual_types")
    if not isinstance(visual_types_raw, list) or not visual_types_raw:
        raise ValueError(f"{annotation_path}: annotation.visual_types must be a non-empty array.")
    visual_type_ids: set[str] = set()
    for position, raw_visual_type in enumerate(visual_types_raw):
        context = f"annotation.visual_types[{position}]"
        if not isinstance(raw_visual_type, dict):
            raise ValueError(f"{context} must be an object.")
        visual_type_ids.add(require_text(raw_visual_type, "visual_type_id", context))

    instances_raw = annotation.get("expected_element_instances")
    if not isinstance(instances_raw, list):
        raise ValueError(f"{annotation_path}: annotation.expected_element_instances must be an array.")

    instances: list[AnnotationInstance] = []
    for position, raw_instance in enumerate(instances_raw):
        context = f"annotation.expected_element_instances[{position}]"
        if not isinstance(raw_instance, dict):
            raise ValueError(f"{context} must be an object.")
        bbox_raw = require_mapping(raw_instance, "bbox", context)
        x = require_int(bbox_raw, "x", f"{context}.bbox")
        y = require_int(bbox_raw, "y", f"{context}.bbox")
        width = require_int(bbox_raw, "width", f"{context}.bbox")
        height = require_int(bbox_raw, "height", f"{context}.bbox")
        if width <= 0 or height <= 0:
            raise ValueError(f"{context}.bbox width and height must be positive.")
        instances.append(
            AnnotationInstance(
                instance_id=require_text(raw_instance, "instance_id", context),
                frame_id=require_text(raw_instance, "frame_id", context),
                visual_type_id=require_text(raw_instance, "visual_type_id", context),
                bbox=(x, y, width, height),
            )
        )
    return visual_type_ids, instances


def color_for_type(visual_type_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(visual_type_id.encode("utf-8")).digest()
    return tuple(48 + value % 176 for value in digest[:3])


def draw_text_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    text_bbox = draw.textbbox(xy, text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = max(0, min(xy[0], image_size[0] - text_width - 4))
    y = max(0, min(xy[1], image_size[1] - text_height - 4))
    background = (0, 0, 0)
    draw.rectangle((x, y, x + text_width + 4, y + text_height + 4), fill=background)
    draw.text((x + 2, y + 2), text, fill=fill, font=font)


def render_frame(
    frame: FrameEntry,
    instances: list[AnnotationInstance],
    output_path: Path,
    *,
    font: ImageFont.ImageFont,
) -> list[str]:
    warnings: list[str] = []
    if not frame.image_path.is_file():
        raise ValueError(f"{frame.frame_id}: image file not found: {frame.image_path}")

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    draw_text_label(
        draw,
        (6, 6),
        f"{frame.frame_id} | index={frame.index}",
        fill=(255, 255, 255),
        image_size=image.size,
        font=font,
    )

    for instance in sorted(instances, key=lambda item: (item.visual_type_id, item.instance_id)):
        x, y, bbox_width, bbox_height = instance.bbox
        right = x + bbox_width - 1
        bottom = y + bbox_height - 1
        if x < 0 or y < 0 or right >= width or bottom >= height:
            warnings.append(
                f"{frame.frame_id}: bbox for {instance.instance_id} extends outside image "
                f"{width}x{height}: x={x}, y={y}, width={bbox_width}, height={bbox_height}."
            )
        clipped_left = max(0, min(width - 1, x))
        clipped_top = max(0, min(height - 1, y))
        clipped_right = max(clipped_left, min(width - 1, right))
        clipped_bottom = max(clipped_top, min(height - 1, bottom))
        color = color_for_type(instance.visual_type_id)
        draw.rectangle((clipped_left, clipped_top, clipped_right, clipped_bottom), outline=color, width=3)
        label = f"{instance.visual_type_id} | {instance.instance_id}"
        label_y = clipped_top - 16 if clipped_top >= 20 else clipped_top + 4
        draw_text_label(
            draw,
            (clipped_left + 2, label_y),
            label,
            fill=color,
            image_size=image.size,
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return warnings


def render_stream(stream_root: Path) -> tuple[int, list[str]]:
    stream_id, frames = load_manifest(stream_root)
    visual_type_ids, instances = load_annotations(stream_root, stream_id)
    frame_ids = {frame.frame_id for frame in frames}
    warnings: list[str] = []

    instances_by_frame: dict[str, list[AnnotationInstance]] = defaultdict(list)
    for instance in instances:
        if instance.frame_id not in frame_ids:
            raise ValueError(
                f"{stream_id}: instance {instance.instance_id!r} references unknown frame_id "
                f"{instance.frame_id!r}."
            )
        if instance.visual_type_id not in visual_type_ids:
            raise ValueError(
                f"{stream_id}: instance {instance.instance_id!r} references unknown visual_type_id "
                f"{instance.visual_type_id!r}."
            )
        instances_by_frame[instance.frame_id].append(instance)

    preview_dir = stream_root / PREVIEW_DIR_NAME
    font = ImageFont.load_default()
    for frame in frames:
        output_path = preview_dir / f"{frame.frame_id}_annotation.png"
        warnings.extend(render_frame(frame, instances_by_frame[frame.frame_id], output_path, font=font))
    return len(frames), warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render annotation_preview images for probe streams."
    )
    parser.add_argument(
        "--streams-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "streams",
        help="Directory containing probe stream folders.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        help="Stream folder names to render.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    streams_root = args.streams_root.resolve()
    total = 0
    all_warnings: list[str] = []
    for stream_name in args.streams:
        stream_root = streams_root / stream_name
        if not stream_root.is_dir():
            raise ValueError(f"Stream directory does not exist: {stream_root}")
        count, warnings = render_stream(stream_root)
        total += count
        all_warnings.extend(warnings)
        print(f"{stream_name}: rendered {count} preview images into {stream_root / PREVIEW_DIR_NAME}")

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"- {warning}")
    print(f"\nDone: rendered {total} preview images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
