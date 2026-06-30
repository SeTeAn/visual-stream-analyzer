"""Primary per-frame overlays rendered from candidates and type assignments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from PIL import Image, ImageDraw, ImageFont

from ..contracts import CandidateRecord, ChangeEvent, GroupingResult
from ..input import DecodedStream


def _type_color(type_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(type_id.encode("utf-8")).digest()
    return tuple(64 + value % 160 for value in digest[:3])


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=14)
    except TypeError:  # Pillow versions before sized bitmap fonts.
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_type_label(
    draw: ImageDraw.ImageDraw,
    *,
    image_width: int,
    left: int,
    top: int,
    type_id: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    text_width, text_height = _text_size(draw, type_id, font)
    x = max(0, min(left, max(0, image_width - text_width - 4)))
    y = max(0, top - text_height - 4)
    background = (
        max(0, color[0] - 48),
        max(0, color[1] - 48),
        max(0, color[2] - 48),
    )
    draw.rectangle((x, y, x + text_width + 4, y + text_height + 3), fill=background)
    draw.text(
        (x + 2, y + 1),
        type_id,
        fill=(255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def render_primary_overlays(
    decoded_stream: DecodedStream,
    candidates: tuple[CandidateRecord, ...],
    grouping: GroupingResult,
    events: tuple[ChangeEvent, ...] = (),
) -> Mapping[str, Image.Image]:
    """Return detached RGB overlays with bbox rectangles and type labels only.

    ``events`` is accepted for API compatibility with orchestration, but primary
    overlays intentionally do not render transition summaries over image content.
    Event facts belong to ``report.txt`` and machine-readable JSON artifacts.
    """

    assignment_by_candidate = {
        assignment.candidate_id: assignment for assignment in grouping.assignments
    }
    by_frame: dict[str, list[CandidateRecord]] = {}
    for candidate in candidates:
        by_frame.setdefault(candidate.frame_id, []).append(candidate)
    rendered: dict[str, Image.Image] = {}
    font = _label_font()
    for frame in decoded_stream.frames:
        image = frame.to_pillow_image()
        draw = ImageDraw.Draw(image)
        for candidate in sorted(by_frame.get(frame.frame_id, ()), key=lambda item: item.candidate_id):
            assignment = assignment_by_candidate.get(candidate.candidate_id)
            type_id = assignment.primary_type_id if assignment is not None else "unassigned"
            color = _type_color(type_id)
            left = max(0, min(frame.image_size.width - 1, int(candidate.bbox.left)))
            top = max(0, min(frame.image_size.height - 1, int(candidate.bbox.top)))
            right = max(left, min(frame.image_size.width - 1, int(candidate.bbox.right) - 1))
            bottom = max(top, min(frame.image_size.height - 1, int(candidate.bbox.bottom) - 1))
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            _draw_type_label(
                draw,
                image_width=frame.image_size.width,
                left=left,
                top=top,
                type_id=type_id,
                color=color,
                font=font,
            )
        rendered[frame.frame_id] = image
    return rendered


__all__ = ["render_primary_overlays"]
