"""Writer for the compact public product-result bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..contracts import ProductFrame, ProductMask, ProductResult


class ProductOutputCollisionError(FileExistsError):
    """Raised when publication would overwrite an existing output bundle."""


@dataclass(frozen=True, slots=True)
class ProductOutput:
    """Published product bundle and its deterministic portable references."""

    output_root: Path
    result_path: Path
    mask_paths: Mapping[tuple[str, str], Path]
    overlay_paths: Mapping[str, Path]


def _publish_directory_no_replace(staging_root: Path, output_root: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    if os.name == "nt":
        try:
            os.rename(staging_root, output_root)
        except FileExistsError as error:
            raise ProductOutputCollisionError("Product output already exists.") from error
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Atomic no-replace output publication is supported on Windows and Linux."
        )

    import ctypes
    import errno

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace output publication is unavailable.")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_current_working_directory = -100
    rename_no_replace = 1
    result = renameat2(
        at_current_working_directory,
        os.fsencode(staging_root),
        at_current_working_directory,
        os.fsencode(output_root),
        rename_no_replace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ProductOutputCollisionError("Product output already exists.")
    raise OSError(error_number, os.strerror(error_number))


def _type_color(type_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(type_id.encode("utf-8")).digest()
    return tuple(64 + component % 160 for component in digest[:3])


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=14)
    except TypeError:  # Pillow versions before sized bitmap fonts.
        return ImageFont.load_default()


def _mask_filename(object_id: str) -> str:
    return f"{object_id}.png"


def _read_saved_mask(path: Path, expected: ProductMask, frame: ProductFrame) -> np.ndarray:
    with Image.open(path) as image:
        saved = np.asarray(image.convert("L"))
    if saved.shape != (frame.image_size.height, frame.image_size.width):
        raise ValueError(f"Saved mask {path.name} has incorrect dimensions.")
    if not np.all((saved == 0) | (saved == 255)):
        raise ValueError(f"Saved mask {path.name} is not binary.")
    binary = saved == 255
    if not np.array_equal(binary, expected.mask):
        raise ValueError(f"Saved mask {path.name} does not match its source mask.")
    return binary


def render_product_overlay(
    source: Image.Image,
    frame: ProductFrame,
    masks: tuple[np.ndarray, ...] | None = None,
) -> Image.Image:
    """Render the canonical type-only overlay from a public product frame."""

    if source.size != (frame.image_size.width, frame.image_size.height):
        raise ValueError(f"Source image dimensions do not match frame {frame.frame_id!r}.")
    saved_masks = (
        tuple(np.asarray(mask, dtype=np.bool_) for mask in masks)
        if masks is not None
        else tuple(product.mask for product in frame.masks)
    )
    if len(saved_masks) != len(frame.masks):
        raise ValueError("masks must contain exactly one mask per product object.")
    base = source.convert("RGBA")
    drawing = ImageDraw.Draw(base)
    font = _label_font()
    for product, binary_mask in zip(frame.masks, saved_masks, strict=True):
        color = _type_color(product.type_id)
        tint = Image.new("RGBA", base.size, (*color, 0))
        tint.putalpha(Image.fromarray(np.where(binary_mask, 96, 0).astype(np.uint8), mode="L"))
        base = Image.alpha_composite(base, tint)
        drawing = ImageDraw.Draw(base)
        rows, columns = np.nonzero(binary_mask)
        left, right = int(columns.min()), int(columns.max())
        top, bottom = int(rows.min()), int(rows.max())
        drawing.rectangle((left, top, right, bottom), outline=color, width=2)
        label_box = drawing.textbbox((0, 0), product.type_id, font=font, stroke_width=1)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_left = max(0, min(left, base.width - label_width - 4))
        label_top = max(0, top - label_height - 4)
        drawing.rectangle(
            (label_left, label_top, label_left + label_width + 4, label_top + label_height + 3),
            fill=tuple(max(0, component - 48) for component in color),
        )
        drawing.text(
            (label_left + 2, label_top + 1),
            product.type_id,
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return base.convert("RGB")


def _validate_sources(result: ProductResult, frame_images: Mapping[str, Image.Image]) -> None:
    if not isinstance(result, ProductResult):
        raise TypeError("result must be ProductResult.")
    if not isinstance(frame_images, Mapping):
        raise TypeError("frame_images must be a mapping from frame_id to PIL images.")
    expected_ids = {frame.frame_id for frame in result.frames}
    if set(frame_images) != expected_ids:
        raise ValueError("frame_images keys must exactly resolve ProductResult frame IDs.")
    for frame in result.frames:
        image = frame_images[frame.frame_id]
        if not isinstance(image, Image.Image):
            raise TypeError("frame_images values must be PIL Image instances.")
        if image.size != (frame.image_size.width, frame.image_size.height):
            raise ValueError(f"Source image dimensions do not match frame {frame.frame_id!r}.")


def _write_bundle(
    staging_root: Path,
    result: ProductResult,
    frame_images: Mapping[str, Image.Image],
) -> ProductOutput:
    masks_root = staging_root / "masks"
    overlays_root = staging_root / "overlays"
    masks_root.mkdir()
    overlays_root.mkdir()
    payload_frames: list[dict[str, object]] = []
    mask_paths: dict[tuple[str, str], Path] = {}
    overlay_paths: dict[str, Path] = {}

    for frame in result.frames:
        frame_mask_root = masks_root / frame.frame_id
        frame_mask_root.mkdir()
        objects: list[dict[str, object]] = []
        saved_masks: list[np.ndarray] = []
        for product in frame.masks:
            assert product.object_id is not None
            relative_mask_path = f"masks/{frame.frame_id}/{_mask_filename(product.object_id)}"
            mask_path = staging_root / Path(relative_mask_path)
            encoded = np.where(product.mask, 255, 0).astype(np.uint8)
            Image.fromarray(encoded, mode="L").save(mask_path, format="PNG", optimize=False)
            saved_mask = _read_saved_mask(mask_path, product, frame)
            saved_masks.append(saved_mask)
            mask_paths[(frame.frame_id, product.object_id)] = mask_path
            rows, columns = np.nonzero(saved_mask)
            objects.append(
                {
                    "bbox": {
                        "height": int(rows.max() - rows.min() + 1),
                        "width": int(columns.max() - columns.min() + 1),
                        "x": int(columns.min()),
                        "y": int(rows.min()),
                    },
                    "mask": relative_mask_path,
                    "object_id": product.object_id,
                    "type_id": product.type_id,
                }
            )

        relative_overlay_path = f"overlays/{frame.frame_id}.png"
        overlay_path = staging_root / Path(relative_overlay_path)
        render_product_overlay(frame_images[frame.frame_id], frame, tuple(saved_masks)).save(
            overlay_path, format="PNG", optimize=False
        )
        if not overlay_path.is_file():
            raise RuntimeError(f"Overlay for frame {frame.frame_id!r} was not written.")
        overlay_paths[frame.frame_id] = overlay_path
        payload_frames.append(
            {
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "image_size": {"height": frame.image_size.height, "width": frame.image_size.width},
                "objects": objects,
                "overlay": relative_overlay_path,
                "source_image": frame.source_image,
            }
        )

    payload = {
        "events": [
            {
                "event_id": event.event_id,
                "from_frame_id": event.from_frame_id,
                "from_objects": [
                    {"frame_id": item.frame_id, "object_id": item.object_id}
                    for item in event.from_objects
                ],
                "kind": event.kind.value,
                "to_frame_id": event.to_frame_id,
                "to_objects": [
                    {"frame_id": item.frame_id, "object_id": item.object_id}
                    for item in event.to_objects
                ],
                "type_id": event.type_id,
            }
            for event in result.events
        ],
        "frames": payload_frames,
        "matches": [
            {
                "from": {
                    "frame_id": match.from_object.frame_id,
                    "object_id": match.from_object.object_id,
                },
                "status": match.status,
                "to": {
                    "frame_id": match.to_object.frame_id,
                    "object_id": match.to_object.object_id,
                },
            }
            for match in result.matches
        ],
        "pipeline": {
            "models": list(result.pipeline.models),
            "profile_id": result.pipeline.profile_id,
            "profile_sha256": result.pipeline.profile_sha256,
        },
        "schema_version": result.schema_version,
        "status": result.status.value,
        "stream": {
            "frame_count": result.stream.frame_count,
            "input_schema_version": result.stream.input_schema_version,
            "manifest_sha256": result.stream.manifest_sha256,
            "stream_id": result.stream.stream_id,
        },
        "summary": {
            "event_count": len(result.events),
            "frame_count": len(result.frames),
            "match_count": len(result.matches),
            "object_count": sum(len(frame.masks) for frame in result.frames),
            "visual_type_count": len(result.visual_types),
        },
        "visual_types": [
            {
                "objects": [
                    {"frame_id": item.frame_id, "object_id": item.object_id}
                    for item in visual_type.objects
                ],
                "type_id": visual_type.type_id,
            }
            for visual_type in result.visual_types
        ],
    }
    result_path = staging_root / "result.json"
    result_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not result_path.is_file():
        raise RuntimeError("result.json was not written.")
    for frame_payload in payload_frames:
        references = [frame_payload["overlay"]] + [
            item["mask"] for item in frame_payload["objects"]  # type: ignore[index]
        ]
        for reference in references:
            if not (staging_root / Path(reference)).is_file():
                raise RuntimeError(f"Product output reference does not resolve: {reference}")
    return ProductOutput(staging_root, result_path, mask_paths, overlay_paths)


def write_product_output(
    output_root: str | Path,
    result: ProductResult,
    frame_images: Mapping[str, Image.Image],
) -> ProductOutput:
    """Publish ``result.json``, cleaned masks, and type-only overlays safely.

    ``output_root`` must not already exist.  Every JSON path is relative to that
    root and uses POSIX separators, so a published bundle remains portable. The
    complete staging directory is published in one atomic no-replace operation.
    """

    _validate_sources(result, frame_images)
    root = Path(output_root)
    if root.exists():
        raise ProductOutputCollisionError("Product output already exists.")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=parent))
    try:
        staged = _write_bundle(staging_root, result, frame_images)
        _publish_directory_no_replace(staging_root, root)
        remap = lambda path: root / path.relative_to(staging_root)
        return ProductOutput(
            output_root=root,
            result_path=remap(staged.result_path),
            mask_paths={key: remap(path) for key, path in staged.mask_paths.items()},
            overlay_paths={key: remap(path) for key, path in staged.overlay_paths.items()},
        )
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


__all__ = [
    "ProductOutput",
    "ProductOutputCollisionError",
    "render_product_overlay",
    "write_product_output",
]
