from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


STREAM_ID = "heldout_01_stationery"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
AA = 4
BBOX_PADDING = 2
MIN_BBOX_GAP = 20.0
POSITION_NORMALIZER = FRAME_WIDTH
POSITION_MIN_NORMALIZED = 0.1
POSITION_MAX_PIXELS = 80.0
BORDER_RATIO = 0.08
ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "frames"
PREVIEW_DIR = ROOT / "annotation_preview"

BACKGROUND_TOP = "#183B3B"
BACKGROUND_BOTTOM = "#234B47"


@dataclass(frozen=True)
class TypeSpec:
    kind: str
    object_name: str
    geometric_class: str
    visual_type_id: str
    description: str
    similarity_group: str


@dataclass(frozen=True)
class Placement:
    kind: str
    slot: int
    center: tuple[int, int]
    rotation: float = 0.0
    scale: float = 1.0
    body_color: str = "#FFFFFF"
    difficulty_tags: str = ""


TYPE_SPECS = [
    TypeSpec(
        "sticky_note",
        "стикер",
        "square",
        "square_subclass_01",
        "Квадратный стикер с загнутым углом, тонкой рамкой и двумя линиями.",
        "",
    ),
    TypeSpec(
        "notebook",
        "блокнот",
        "rectangle",
        "rectangle_subclass_01",
        "Блокнот с корешком, креплениями, рамкой обложки и линиями страниц.",
        "rectangle_stationery",
    ),
    TypeSpec(
        "eraser",
        "ластик",
        "rectangle",
        "rectangle_subclass_02",
        "Короткий ластик со скругленными концами и бумажной полосой.",
        "rectangle_stationery",
    ),
    TypeSpec(
        "pencil",
        "карандаш",
        "rectangle",
        "rectangle_subclass_03",
        "Тонкий карандаш с деревянным кончиком, грифелем и ластиком.",
        "elongated_stationery",
    ),
    TypeSpec(
        "marker",
        "маркер",
        "rectangle",
        "rectangle_subclass_04",
        "Широкий маркер с тупым колпачком, кольцом и плоским торцом.",
        "elongated_stationery",
    ),
    TypeSpec(
        "tape_roll",
        "рулон скотча",
        "circle",
        "circle_subclass_01",
        "Круглый рулон скотча с крупным центральным отверстием и двойным ободом.",
        "round_stationery",
    ),
    TypeSpec(
        "sharpener",
        "точилка",
        "circle",
        "circle_subclass_02",
        "Компактная круглая точилка с малым отверстием, лезвием и крепежным винтом.",
        "round_stationery",
    ),
    TypeSpec(
        "set_square",
        "чертежный угольник",
        "triangle",
        "triangle_subclass_01",
        "Треугольный чертежный угольник с внутренним вырезом и делениями.",
        "",
    ),
    TypeSpec(
        "scissors",
        "ножницы",
        "undefined",
        "undefined_subclass_01",
        "Ножницы с двумя кольцами, общим шарниром и двумя лезвиями.",
        "undefined_stationery",
    ),
    TypeSpec(
        "binder_clip",
        "канцелярский зажим",
        "undefined",
        "undefined_subclass_02",
        "Канцелярский зажим с трапециевидным корпусом и двумя проволочными ручками.",
        "undefined_stationery",
    ),
]

TYPE_BY_KIND = {spec.kind: spec for spec in TYPE_SPECS}
TYPE_ORDER = {spec.visual_type_id: index for index, spec in enumerate(TYPE_SPECS)}


def pl(
    kind: str,
    center: tuple[int, int],
    *,
    slot: int = 1,
    rotation: float = 0.0,
    scale: float = 1.0,
    body_color: str,
    difficulty_tags: str = "",
) -> Placement:
    return Placement(kind, slot, center, rotation, scale, body_color, difficulty_tags)


N = lambda: pl("notebook", (420, 83), body_color="#D96A62", difficulty_tags="similar_subclass")
T = lambda: pl("tape_roll", (550, 78), body_color="#B49BE3", difficulty_tags="similar_subclass,border_near")
C0 = lambda: pl("scissors", (435, 400), rotation=8, body_color="#E76F69")
C1 = lambda: pl("scissors", (510, 400), rotation=8, body_color="#E76F69", difficulty_tags="position_change")
S1 = lambda: pl("sticky_note", (50, 70), body_color="#F1C75B", difficulty_tags="border_near")
S2 = lambda: pl("sticky_note", (160, 95), slot=2, body_color="#75D6B2", difficulty_tags="count_change")
S3 = lambda: pl("sticky_note", (275, 70), slot=3, body_color="#F08F78", difficulty_tags="count_change")
P0 = lambda: pl("pencil", (145, 335), body_color="#E9A641", difficulty_tags="similar_subclass")
P1 = lambda: pl("pencil", (220, 335), body_color="#E9A641", difficulty_tags="position_change,similar_subclass")
P_ROT = lambda: pl("pencil", (220, 335), rotation=45, body_color="#E9A641", difficulty_tags="rotation,similar_subclass")
P_GREEN = lambda: pl("pencil", (220, 335), rotation=45, body_color="#78B58D", difficulty_tags="rotation,color_variation,similar_subclass")
H = lambda: pl("sharpener", (195, 180), body_color="#E4B84F", difficulty_tags="similar_subclass,reappearance")
M0 = lambda color="#57BFC1": pl("marker", (360, 300), body_color=color, difficulty_tags="similar_subclass")
M1 = lambda: pl("marker", (435, 300), body_color="#B879A8", difficulty_tags="position_change,color_variation,similar_subclass")
E0 = lambda: pl("eraser", (82, 190), body_color="#70C8C2", difficulty_tags="similar_subclass")
E1 = lambda: pl("eraser", (82, 190), scale=1.3, body_color="#70C8C2", difficulty_tags="scale,similar_subclass")
R = lambda: pl("set_square", (315, 198), rotation=-8, body_color="#6FD0A7")
B0 = lambda: pl("binder_clip", (475, 185), rotation=0, body_color="#E6A64C")
B1 = lambda: pl("binder_clip", (475, 185), rotation=35, body_color="#E6A64C", difficulty_tags="rotation")


SCENES: list[list[Placement]] = [
    [N(), P0(), T(), C0()],
    [N(), P0(), T(), C0(), S1()],
    [N(), P1(), T(), C0(), S1()],
    [N(), P1(), T(), C0(), S1(), S2()],
    [N(), P1(), C0(), S1(), S2()],
    [N(), P1(), C0(), S1(), S2(), H()],
    [N(), P_ROT(), C0(), S1(), S2(), H()],
    [N(), P_ROT(), C0(), S1(), S2(), H(), M0()],
    [N(), P_ROT(), C0(), S1(), S2(), H(), M0("#B879A8")],
    [N(), P_ROT(), C0(), S1(), S2(), H(), M0("#B879A8"), E0()],
    [N(), P_ROT(), C0(), S1(), S2(), H(), M0("#B879A8"), E1()],
    [N(), P_ROT(), C0(), S1(), S2(), H(), M0("#B879A8"), E1(), R()],
    [N(), P_ROT(), C1(), S1(), S2(), H(), M0("#B879A8"), E1(), R()],
    [N(), P_ROT(), C1(), S1(), S2(), H(), M0("#B879A8"), E1(), R(), B0()],
    [N(), P_ROT(), C1(), S1(), S2(), M0("#B879A8"), E1(), R(), B0()],
    [P_ROT(), C1(), S1(), S2(), M0("#B879A8"), E1(), R(), B0()],
    [P_GREEN(), C1(), S1(), S2(), H(), M0("#B879A8"), E1(), R(), B0()],
    [P_GREEN(), C1(), S1(), S2(), S3(), H(), M1(), E1(), R(), B0()],
    [N(), P_GREEN(), C1(), S1(), S2(), S3(), H(), M1(), E1(), R(), B0()],
    [N(), P_GREEN(), T(), C1(), S1(), S2(), S3(), H(), M1(), E1(), R(), B1()],
]

FRAME_NOTES = [
    "Начальная сцена: блокнот, карандаш, рулон скотча и ножницы.",
    "Появляется первый стикер около левой границы.",
    "Карандаш намеренно перемещается на 75 px.",
    "Добавляется второй стикер того же visual type.",
    "Исчезает рулон скотча.",
    "Появляется точилка.",
    "Карандаш поворачивается вокруг неизменного центра.",
    "Появляется маркер.",
    "Меняется цвет тела маркера без смены visual type.",
    "Появляется ластик.",
    "Ластик увеличивается вокруг неизменного центра.",
    "Появляется чертежный угольник.",
    "Ножницы намеренно перемещаются на 75 px.",
    "Появляется канцелярский зажим.",
    "Точилка исчезает.",
    "Точилка отсутствует второй кадр; блокнот исчезает.",
    "Точилка возвращается; меняется цвет карандаша.",
    "Добавляется третий стикер; маркер перемещается на 75 px.",
    "Возвращается блокнот.",
    "Возвращается рулон скотча; зажим поворачивается вокруг центра.",
]

POSITION_CHANGES = {
    3: {"rectangle_subclass_03"},
    13: {"undefined_subclass_01"},
    18: {"rectangle_subclass_04"},
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def shade(hex_color: str, factor: float) -> str:
    values = rgb(hex_color)
    adjusted = tuple(max(0, min(255, round(channel * factor))) for channel in values)
    return "#" + "".join(f"{channel:02X}" for channel in adjusted)


def scaled(value: float) -> int:
    return int(round(value * AA))


def scaled_box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(scaled(value) for value in values)


def scaled_points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(scaled(x), scaled(y)) for x, y in values]


def new_asset(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width * AA, height * AA), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def draw_sticky_note(color: str) -> Image.Image:
    image, draw = new_asset(58, 58)
    outline = shade(color, 0.55)
    draw.polygon(
        scaled_points([(3, 3), (43, 3), (55, 15), (55, 55), (3, 55)]),
        fill=color,
        outline=outline,
        width=scaled(2),
    )
    draw.polygon(scaled_points([(43, 3), (43, 15), (55, 15)]), fill=shade(color, 1.18), outline=outline)
    draw.line(scaled_points([(12, 31), (45, 31)]), fill=shade(color, 0.72), width=scaled(2))
    draw.line(scaled_points([(12, 41), (38, 41)]), fill=shade(color, 0.72), width=scaled(2))
    return image


def draw_notebook(color: str) -> Image.Image:
    image, draw = new_asset(112, 78)
    outline = shade(color, 0.45)
    draw.rounded_rectangle(scaled_box((3, 3, 109, 75)), radius=scaled(7), fill=color, outline=outline, width=scaled(3))
    draw.rounded_rectangle(scaled_box((6, 6, 22, 72)), radius=scaled(4), fill=shade(color, 0.55))
    for y in (16, 29, 42, 55, 68):
        draw.rounded_rectangle(scaled_box((14, y - 2, 26, y + 2)), radius=scaled(1), fill="#EAD8C4")
    draw.rounded_rectangle(scaled_box((42, 20, 94, 58)), radius=scaled(4), fill="#F0E7D6", outline=outline, width=scaled(2))
    draw.line(scaled_points([(51, 34), (84, 34)]), fill=shade(color, 0.75), width=scaled(2))
    draw.line(scaled_points([(51, 45), (78, 45)]), fill=shade(color, 0.75), width=scaled(2))
    return image


def draw_eraser(color: str) -> Image.Image:
    image, draw = new_asset(68, 38)
    outline = shade(color, 0.45)
    draw.rounded_rectangle(scaled_box((2, 3, 66, 35)), radius=scaled(9), fill=color, outline=outline, width=scaled(2))
    draw.rectangle(scaled_box((24, 4, 47, 34)), fill="#EEE4D1", outline="#9D8D78", width=scaled(2))
    draw.line(scaled_points([(29, 14), (42, 14)]), fill="#B6A58E", width=scaled(2))
    draw.line(scaled_points([(29, 23), (42, 23)]), fill="#B6A58E", width=scaled(2))
    return image


def draw_pencil(color: str) -> Image.Image:
    image, draw = new_asset(130, 28)
    outline = shade(color, 0.42)
    draw.polygon(
        scaled_points([(3, 14), (20, 3), (111, 3), (111, 25), (20, 25)]),
        fill=color,
        outline=outline,
        width=scaled(2),
    )
    draw.polygon(scaled_points([(3, 14), (20, 3), (20, 25)]), fill="#E8D0A7", outline=outline, width=scaled(2))
    draw.polygon(scaled_points([(3, 14), (10, 10), (10, 18)]), fill="#263238")
    draw.rectangle(scaled_box((111, 3, 127, 25)), fill="#E78791", outline=outline, width=scaled(2))
    draw.line(scaled_points([(24, 9), (106, 9)]), fill=shade(color, 1.22), width=scaled(2))
    return image


def draw_marker(color: str) -> Image.Image:
    image, draw = new_asset(112, 36)
    outline = shade(color, 0.42)
    draw.rounded_rectangle(scaled_box((3, 4, 109, 32)), radius=scaled(7), fill=color, outline=outline, width=scaled(2))
    draw.rounded_rectangle(scaled_box((3, 4, 28, 32)), radius=scaled(7), fill=shade(color, 0.62), outline=outline, width=scaled(2))
    draw.line(scaled_points([(31, 5), (31, 31)]), fill="#E9D9B9", width=scaled(3))
    draw.rectangle(scaled_box((91, 8, 106, 28)), fill=shade(color, 1.14), outline=outline, width=scaled(1))
    draw.line(scaled_points([(43, 12), (79, 12)]), fill=shade(color, 1.22), width=scaled(2))
    return image


def draw_tape_roll(color: str) -> Image.Image:
    image, draw = new_asset(64, 64)
    outline = shade(color, 0.48)
    draw.ellipse(scaled_box((2, 2, 62, 62)), fill=color, outline=outline, width=scaled(3))
    draw.ellipse(scaled_box((18, 18, 46, 46)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((18, 18, 46, 46)), outline=outline, width=scaled(3))
    draw.arc(scaled_box((8, 8, 56, 56)), 205, 310, fill=shade(color, 1.22), width=scaled(3))
    return image


def draw_sharpener(color: str) -> Image.Image:
    image, draw = new_asset(60, 56)
    outline = shade(color, 0.45)
    draw.ellipse(scaled_box((3, 2, 57, 54)), fill=color, outline=outline, width=scaled(3))
    draw.ellipse(scaled_box((10, 18, 25, 33)), fill="#28363A", outline=outline, width=scaled(2))
    draw.polygon(scaled_points([(27, 11), (50, 18), (39, 43), (21, 35)]), fill="#D8E0E3", outline="#5E6A70", width=scaled(2))
    draw.ellipse(scaled_box((34, 23, 42, 31)), fill="#E2B74B", outline="#5E6A70", width=scaled(1))
    draw.line(scaled_points([(27, 34), (43, 17)]), fill="#839198", width=scaled(2))
    return image


def draw_set_square(color: str) -> Image.Image:
    image, draw = new_asset(86, 76)
    outline = shade(color, 0.43)
    outer = [(3, 73), (3, 3), (83, 73)]
    inner = [(22, 58), (22, 27), (59, 58)]
    draw.polygon(scaled_points(outer), fill=color, outline=outline, width=scaled(3))
    draw.polygon(scaled_points(inner), fill=(0, 0, 0, 0))
    draw.line(scaled_points(inner + [inner[0]]), fill=outline, width=scaled(2))
    for x in (12, 22, 32, 42, 52, 62):
        draw.line(scaled_points([(x, 69), (x, 65)]), fill=outline, width=scaled(1))
    return image


def draw_scissors(color: str) -> Image.Image:
    image, draw = new_asset(112, 76)
    outline = shade(color, 0.42)
    blade_outline = "#526169"
    pivot = (53, 38)
    draw.polygon(scaled_points([pivot, (108, 5), (110, 13), (60, 42)]), fill="#D5DDE0", outline=blade_outline, width=scaled(2))
    draw.polygon(scaled_points([pivot, (108, 63), (105, 72), (58, 44)]), fill="#D5DDE0", outline=blade_outline, width=scaled(2))
    draw.line(scaled_points([(51, 37), (35, 22)]), fill=color, width=scaled(10))
    draw.line(scaled_points([(51, 41), (35, 56)]), fill=color, width=scaled(10))
    draw.ellipse(scaled_box((3, 3, 41, 39)), fill=color, outline=outline, width=scaled(3))
    draw.ellipse(scaled_box((3, 37, 41, 73)), fill=color, outline=outline, width=scaled(3))
    draw.ellipse(scaled_box((13, 12, 32, 30)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((13, 46, 32, 64)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((13, 12, 32, 30)), outline=outline, width=scaled(2))
    draw.ellipse(scaled_box((13, 46, 32, 64)), outline=outline, width=scaled(2))
    draw.ellipse(scaled_box((47, 32, 59, 44)), fill="#E7B94D", outline=blade_outline, width=scaled(2))
    return image


def draw_binder_clip(color: str) -> Image.Image:
    image, draw = new_asset(66, 68)
    outline = shade(color, 0.4)
    draw.polygon(scaled_points([(13, 30), (53, 30), (60, 63), (6, 63)]), fill=color, outline=outline, width=scaled(3))
    draw.line(scaled_points([(16, 32), (12, 10), (27, 3), (29, 31)]), fill="#D7DEE0", width=scaled(3))
    draw.line(scaled_points([(50, 32), (54, 10), (39, 3), (37, 31)]), fill="#D7DEE0", width=scaled(3))
    draw.line(scaled_points([(17, 43), (49, 43)]), fill=shade(color, 1.18), width=scaled(2))
    return image


DRAWERS: dict[str, Callable[[str], Image.Image]] = {
    "sticky_note": draw_sticky_note,
    "notebook": draw_notebook,
    "eraser": draw_eraser,
    "pencil": draw_pencil,
    "marker": draw_marker,
    "tape_roll": draw_tape_roll,
    "sharpener": draw_sharpener,
    "set_square": draw_set_square,
    "scissors": draw_scissors,
    "binder_clip": draw_binder_clip,
}


def make_background() -> Image.Image:
    width = FRAME_WIDTH * AA
    height = FRAME_HEIGHT * AA
    top = rgb(BACKGROUND_TOP)
    bottom = rgb(BACKGROUND_BOTTOM)
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(round(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3))
        draw.line((0, y, width, y), fill=color)
    return image.convert("RGBA")


def transform_asset(placement: Placement) -> Image.Image:
    asset = DRAWERS[placement.kind](placement.body_color)
    if placement.scale != 1.0:
        asset = asset.resize(
            (max(1, round(asset.width * placement.scale)), max(1, round(asset.height * placement.scale))),
            Image.Resampling.LANCZOS,
        )
    if placement.rotation:
        asset = asset.rotate(placement.rotation, resample=Image.Resampling.BICUBIC, expand=True)
    alpha_bbox = asset.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError(f"Rendered asset {placement.kind!r} is empty")
    return asset.crop(alpha_bbox)


def padded_bbox(actual: tuple[int, int, int, int]) -> dict[str, int]:
    left = max(0, actual[0] - BBOX_PADDING)
    top = max(0, actual[1] - BBOX_PADDING)
    right = min(FRAME_WIDTH, actual[2] + BBOX_PADDING)
    bottom = min(FRAME_HEIGHT, actual[3] + BBOX_PADDING)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def bbox_distance(a: dict[str, int], b: dict[str, int]) -> float:
    a_right = a["x"] + a["width"]
    a_bottom = a["y"] + a["height"]
    b_right = b["x"] + b["width"]
    b_bottom = b["y"] + b["height"]
    dx = max(a["x"] - b_right, b["x"] - a_right, 0)
    dy = max(a["y"] - b_bottom, b["y"] - a_bottom, 0)
    return math.hypot(dx, dy)


def render_frame(frame_index: int, placements: list[Placement]) -> list[dict[str, object]]:
    background = make_background()
    background_output = background.convert("RGB").resize(
        (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS
    )
    frame = background.copy()
    instances: list[dict[str, object]] = []
    bboxes: list[dict[str, int]] = []

    for placement in placements:
        spec = TYPE_BY_KIND[placement.kind]
        asset = transform_asset(placement)
        center_x = placement.center[0] * AA
        center_y = placement.center[1] * AA
        left = round(center_x - asset.width / 2)
        top = round(center_y - asset.height / 2)
        right = left + asset.width
        bottom = top + asset.height
        if left < 0 or top < 0 or right > frame.width or bottom > frame.height:
            raise ValueError(f"{placement.kind} leaves frame bounds in frame {frame_index}")

        object_frame = background.copy()
        object_frame.alpha_composite(asset, (left, top))
        object_output = object_frame.convert("RGB").resize(
            (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS
        )
        actual_bbox = ImageChops.difference(object_output, background_output).convert("L").getbbox()
        if actual_bbox is None:
            raise ValueError(f"{placement.kind} produced no foreground pixels in frame {frame_index}")
        bbox = padded_bbox(actual_bbox)
        for previous in bboxes:
            distance = bbox_distance(previous, bbox)
            if distance < MIN_BBOX_GAP:
                raise ValueError(
                    f"bbox gap {distance:.2f} px is below {MIN_BBOX_GAP:.0f} px "
                    f"in frame {frame_index}: {previous} vs {bbox}"
                )
        bboxes.append(bbox)
        frame.alpha_composite(asset, (left, top))

        frame_id = f"frame_{frame_index:03d}"
        tags = placement.difficulty_tags or "standard"
        instances.append(
            {
                "instance_id": f"inst_{placement.kind}_{placement.slot:02d}_f{frame_index:03d}",
                "frame_id": frame_id,
                "visual_type_id": spec.visual_type_id,
                "bbox": bbox,
                "characteristic_regions": [],
                "uncertainty": "",
                "notes": (
                    f"object={spec.kind}; geometric_class={spec.geometric_class}; slot={placement.slot}; "
                    f"body_color={placement.body_color}; rotation_deg={placement.rotation:g}; "
                    f"scale={placement.scale:g}; anchor_center=({placement.center[0]},{placement.center[1]}); "
                    f"difficulty_tags={tags}."
                ),
            }
        )

    output = frame.convert("RGB").resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
    difference = ImageChops.difference(output, background_output)
    for instance in instances:
        bbox = instance["bbox"]
        search_left = max(0, bbox["x"] - 8)
        search_top = max(0, bbox["y"] - 8)
        search_right = min(FRAME_WIDTH, bbox["x"] + bbox["width"] + 8)
        search_bottom = min(FRAME_HEIGHT, bbox["y"] + bbox["height"] + 8)
        local_bbox = difference.crop((search_left, search_top, search_right, search_bottom)).getbbox()
        if local_bbox is None:
            raise ValueError(f"Final PNG foreground is missing for {instance['instance_id']}")
        foreground_left = search_left + local_bbox[0]
        foreground_top = search_top + local_bbox[1]
        foreground_right = search_left + local_bbox[2] - 1
        foreground_bottom = search_top + local_bbox[3] - 1
        bbox_left = max(0, foreground_left - BBOX_PADDING)
        bbox_top = max(0, foreground_top - BBOX_PADDING)
        bbox_right = min(FRAME_WIDTH - 1, foreground_right + BBOX_PADDING)
        bbox_bottom = min(FRAME_HEIGHT - 1, foreground_bottom + BBOX_PADDING)
        instance["bbox"] = {
            "x": bbox_left,
            "y": bbox_top,
            "width": bbox_right - bbox_left + 1,
            "height": bbox_bottom - bbox_top + 1,
        }

    output.save(FRAMES_DIR / f"frame_{frame_index:03d}.png", format="PNG", optimize=True)
    return instances


def counts_by_type(placements: list[Placement]) -> dict[str, int]:
    counts = {spec.visual_type_id: 0 for spec in TYPE_SPECS}
    for placement in placements:
        counts[TYPE_BY_KIND[placement.kind].visual_type_id] += 1
    return counts


def bbox_center(instance: dict[str, object]) -> tuple[float, float]:
    bbox = instance["bbox"]
    return (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)


def average_bbox_center(instances: list[dict[str, object]], visual_type_id: str) -> tuple[float, float]:
    centers = [bbox_center(instance) for instance in instances if instance["visual_type_id"] == visual_type_id]
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def build_events(
    instances_by_frame: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    def add_event(
        event_type: str,
        visual_type_id: str,
        from_frame_id: str,
        to_frame_id: str,
        evidence: str,
    ) -> str:
        event_id = f"event_{len(events) + 1:03d}"
        events.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "visual_type_id": visual_type_id,
                "from_frame_id": from_frame_id,
                "to_frame_id": to_frame_id,
                "evidence": evidence,
                "uncertainty": "",
                "notes": "Событие размечено на уровне локального visual type.",
            }
        )
        return event_id

    for to_index in range(2, len(SCENES) + 1):
        previous_scene = SCENES[to_index - 2]
        current_scene = SCENES[to_index - 1]
        previous_counts = counts_by_type(previous_scene)
        current_counts = counts_by_type(current_scene)
        from_frame_id = f"frame_{to_index - 1:03d}"
        to_frame_id = f"frame_{to_index:03d}"
        event_ids: list[str] = []

        for visual_type_id in sorted(TYPE_ORDER, key=TYPE_ORDER.get):
            old_count = previous_counts[visual_type_id]
            new_count = current_counts[visual_type_id]
            if old_count == 0 and new_count > 0:
                event_ids.append(add_event("appeared", visual_type_id, from_frame_id, to_frame_id, f"Количество изменилось с 0 до {new_count}."))
                continue
            if old_count > 0 and new_count == 0:
                event_ids.append(add_event("disappeared", visual_type_id, from_frame_id, to_frame_id, f"Количество изменилось с {old_count} до 0."))
                continue
            if old_count == 0:
                continue

            event_ids.append(add_event("persisted", visual_type_id, from_frame_id, to_frame_id, f"Тип присутствует в обоих кадрах: {old_count} -> {new_count}."))
            if old_count != new_count:
                event_ids.append(add_event("count_changed", visual_type_id, from_frame_id, to_frame_id, f"Количество изменилось с {old_count} до {new_count}."))

            if visual_type_id in POSITION_CHANGES.get(to_index, set()):
                old_center = average_bbox_center(instances_by_frame[from_frame_id], visual_type_id)
                new_center = average_bbox_center(instances_by_frame[to_frame_id], visual_type_id)
                shift = math.dist(old_center, new_center)
                normalized = shift / POSITION_NORMALIZER
                event_ids.append(
                    add_event(
                        "position_changed",
                        visual_type_id,
                        from_frame_id,
                        to_frame_id,
                        f"Средний центр bbox смещен на {shift:.1f} px; normalized_by_width={normalized:.4f}.",
                    )
                )

        comparisons.append(
            {
                "from_frame_id": from_frame_id,
                "to_frame_id": to_frame_id,
                "expected_change_event_ids": event_ids,
                "uncertainty": "",
                "notes": "Полная разметка всех положительных событий между соседними кадрами; отсутствие события является отрицательной меткой.",
            }
        )

    return comparisons, events


def preview_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_preview(frame_index: int, instances: list[dict[str, object]]) -> None:
    frame_path = FRAMES_DIR / f"frame_{frame_index:03d}.png"
    with Image.open(frame_path) as opened:
        frame = opened.convert("RGB")
    preview = Image.new("RGB", (940, FRAME_HEIGHT), "#F5F6F7")
    preview.paste(frame, (0, 0))
    draw = ImageDraw.Draw(preview)
    colors = ["#FFDB58", "#FF7F73", "#74C69D", "#6FA8FF", "#D6A5FF", "#FFB45E", "#78D5D7", "#9BE36C", "#F18BB8", "#AAB7C4"]
    font = preview_font(13)
    small = preview_font(12)

    draw.rectangle((640, 0, 939, 479), fill="#F5F6F7", outline="#C7CDD2")
    draw.text((654, 14), f"{STREAM_ID} / frame_{frame_index:03d}", fill="#1F2A30", font=font)
    draw.text((654, 38), "Local visual types", fill="#46545C", font=small)
    legend_y = 62
    for type_index, spec in enumerate(TYPE_SPECS, start=1):
        color = colors[type_index - 1]
        draw.rectangle((654, legend_y + 2, 666, legend_y + 14), fill=color, outline="#27343A")
        draw.text((674, legend_y), f"T{type_index:02d} {spec.visual_type_id}", fill="#1F2A30", font=small)
        legend_y += 25

    for instance in instances:
        type_index = TYPE_ORDER[instance["visual_type_id"]]
        color = colors[type_index]
        bbox = instance["bbox"]
        left = bbox["x"]
        top = bbox["y"]
        right = left + bbox["width"] - 1
        bottom = top + bbox["height"] - 1
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        label = f"T{type_index + 1:02d}"
        label_top = max(0, top - 15)
        draw.rectangle((left, label_top, left + 25, label_top + 14), fill=color)
        draw.text((left + 2, label_top), label, fill="#142025", font=small)

    preview.save(PREVIEW_DIR / f"frame_{frame_index:03d}_annotation.png", format="PNG", optimize=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_outputs(manifest: dict[str, object], annotation: dict[str, object]) -> dict[str, object]:
    errors: list[str] = []
    frames = manifest["frames"]
    frame_ids = [frame["frame_id"] for frame in frames]
    expected_frame_ids = [f"frame_{index:03d}" for index in range(1, 21)]
    if frame_ids != expected_frame_ids:
        errors.append("Manifest must contain frame_001 through frame_020 in order")
    if [frame["index"] for frame in frames] != list(range(1, 21)):
        errors.append("Manifest indexes must be 1 through 20")
    if manifest["stream_id"] != annotation["stream_id"]:
        errors.append("stream_id differs between manifest and annotation")
    if manifest["metadata"]["frame_size"] != {"width": 640, "height": 480}:
        errors.append("metadata.frame_size has an unexpected value")
    if manifest["metadata"]["frame_format"] != "png_rgb":
        errors.append("metadata.frame_format has an unexpected value")

    visual_type_ids = [item["visual_type_id"] for item in annotation["visual_types"]]
    if len(visual_type_ids) != 10 or len(set(visual_type_ids)) != 10:
        errors.append("Exactly 10 unique visual_type_id values are required")
    known_types = set(visual_type_ids)

    instances = annotation["expected_element_instances"]
    instance_ids = [instance["instance_id"] for instance in instances]
    if len(instance_ids) != len(set(instance_ids)):
        errors.append("instance_id values are not unique")
    instances_by_frame: dict[str, list[dict[str, object]]] = {frame_id: [] for frame_id in frame_ids}
    represented_types: set[str] = set()
    min_gap = float("inf")
    max_foreground_ratio = 0.0
    max_bbox_ratio = 0.0

    for instance in instances:
        frame_id = instance["frame_id"]
        if frame_id not in instances_by_frame:
            errors.append(f"Unknown frame_id in {instance['instance_id']}")
            continue
        if instance["visual_type_id"] not in known_types:
            errors.append(f"Unknown visual_type_id in {instance['instance_id']}")
        represented_types.add(instance["visual_type_id"])
        bbox = instance["bbox"]
        if bbox["width"] <= 0 or bbox["height"] <= 0:
            errors.append(f"Non-positive bbox in {instance['instance_id']}")
        if bbox["x"] < 0 or bbox["y"] < 0 or bbox["x"] + bbox["width"] > FRAME_WIDTH or bbox["y"] + bbox["height"] > FRAME_HEIGHT:
            errors.append(f"Out-of-frame bbox in {instance['instance_id']}")
        if bbox["width"] * bbox["height"] / (FRAME_WIDTH * FRAME_HEIGHT) > 0.15:
            errors.append(f"Single bbox exceeds 15% in {instance['instance_id']}")
        instances_by_frame[frame_id].append(instance)
    if represented_types != known_types:
        errors.append("Not all visual types are represented")

    background = make_background().convert("RGB").resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
    border_x = round(FRAME_WIDTH * BORDER_RATIO)
    border_y = round(FRAME_HEIGHT * BORDER_RATIO)
    border_mask = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT), 255)
    ImageDraw.Draw(border_mask).rectangle((border_x, border_y, FRAME_WIDTH - border_x - 1, FRAME_HEIGHT - border_y - 1), fill=0)
    border_area = sum(border_mask.histogram()[1:])

    for frame in frames:
        frame_id = frame["frame_id"]
        path = ROOT / frame["image_path"]
        if not path.is_file():
            errors.append(f"Missing frame file: {path}")
            continue
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.mode != "RGB" or opened.size != (FRAME_WIDTH, FRAME_HEIGHT):
                errors.append(f"Unexpected image properties for {frame_id}")
            image = opened.convert("RGB")

        difference = ImageChops.difference(image, background).convert("L")
        foreground = difference.point(lambda value: 255 if value else 0)
        foreground_pixels = sum(foreground.histogram()[1:])
        max_foreground_ratio = max(max_foreground_ratio, foreground_pixels / (FRAME_WIDTH * FRAME_HEIGHT))
        if foreground_pixels / (FRAME_WIDTH * FRAME_HEIGHT) > 0.30:
            errors.append(f"Foreground exceeds 30% in {frame_id}")

        frame_instances = instances_by_frame[frame_id]
        bbox_area = sum(item["bbox"]["width"] * item["bbox"]["height"] for item in frame_instances)
        max_bbox_ratio = max(max_bbox_ratio, bbox_area / (FRAME_WIDTH * FRAME_HEIGHT))
        if bbox_area / (FRAME_WIDTH * FRAME_HEIGHT) > 0.40:
            errors.append(f"Bbox area exceeds 40% in {frame_id}")

        union_mask = Image.new("L", image.size, 0)
        union_draw = ImageDraw.Draw(union_mask)
        for index, instance in enumerate(frame_instances):
            bbox = instance["bbox"]
            crop_box = (bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"])
            if difference.crop(crop_box).getbbox() is None:
                errors.append(f"bbox contains no object pixels: {instance['instance_id']}")
            union_draw.rectangle((bbox["x"], bbox["y"], bbox["x"] + bbox["width"] - 1, bbox["y"] + bbox["height"] - 1), fill=255)
            for other in frame_instances[index + 1 :]:
                distance = bbox_distance(bbox, other["bbox"])
                min_gap = min(min_gap, distance)
                if distance < MIN_BBOX_GAP:
                    errors.append(f"bbox gap below 20 px in {frame_id}: {distance:.2f}")
        outside = ImageChops.multiply(foreground, ImageOps.invert(union_mask))
        if outside.getbbox() is not None:
            errors.append(f"Foreground pixels exist outside annotated bboxes in {frame_id}")
        border_foreground = ImageChops.multiply(foreground, border_mask)
        border_pixels = sum(border_foreground.histogram()[1:])
        if border_pixels / border_area >= 0.10:
            errors.append(f"Border foreground coverage reaches 10% in {frame_id}")

    comparisons = annotation["frame_comparisons"]
    if len(comparisons) != 19:
        errors.append("Exactly 19 neighboring frame comparisons are required")
    event_ids = [event["event_id"] for event in annotation["change_events"]]
    if len(event_ids) != len(set(event_ids)):
        errors.append("event_id values are not unique")
    event_by_id = {event["event_id"]: event for event in annotation["change_events"]}
    referenced_ids: list[str] = []
    expected_pairs = [(f"frame_{index:03d}", f"frame_{index + 1:03d}") for index in range(1, 20)]
    for comparison, expected_pair in zip(comparisons, expected_pairs):
        pair = (comparison["from_frame_id"], comparison["to_frame_id"])
        if pair != expected_pair:
            errors.append(f"Unexpected comparison pair: {pair}")
        for event_id in comparison["expected_change_event_ids"]:
            referenced_ids.append(event_id)
            event = event_by_id.get(event_id)
            if event is None or (event["from_frame_id"], event["to_frame_id"]) != pair:
                errors.append(f"Invalid event reference: {event_id}")
    if sorted(referenced_ids) != sorted(event_ids):
        errors.append("Events and frame comparison references differ")

    allowed = set(annotation["allowed_event_types"])
    for event in annotation["change_events"]:
        if event["event_type"] not in allowed or event["visual_type_id"] not in known_types:
            errors.append(f"Invalid event: {event['event_id']}")

    for to_index in range(2, 21):
        old_id = f"frame_{to_index - 1:03d}"
        new_id = f"frame_{to_index:03d}"
        old_counts = counts_by_type(SCENES[to_index - 2])
        new_counts = counts_by_type(SCENES[to_index - 1])
        explicit_positions = POSITION_CHANGES.get(to_index, set())
        for visual_type_id in known_types:
            if old_counts[visual_type_id] == new_counts[visual_type_id] and old_counts[visual_type_id] > 0:
                old_center = average_bbox_center(instances_by_frame[old_id], visual_type_id)
                new_center = average_bbox_center(instances_by_frame[new_id], visual_type_id)
                shift = math.dist(old_center, new_center)
                if visual_type_id in explicit_positions:
                    if shift / POSITION_NORMALIZER <= POSITION_MIN_NORMALIZED or shift > POSITION_MAX_PIXELS:
                        errors.append(f"Invalid intended position shift {shift:.2f} for {visual_type_id} in {new_id}")
                elif shift > 2.0:
                    errors.append(f"Unintended center shift {shift:.2f} for {visual_type_id} in {new_id}")

    preview_paths = sorted(PREVIEW_DIR.glob("frame_*_annotation.png"))
    if len(preview_paths) != 20:
        errors.append("Exactly 20 annotation previews are required")

    if errors:
        raise ValueError("Generated stream validation failed:\n- " + "\n- ".join(errors))
    return {
        "frames": len(frames),
        "instances": len(instances),
        "events": len(annotation["change_events"]),
        "min_bbox_gap_px": round(min_gap, 2),
        "max_foreground_ratio": round(max_foreground_ratio, 4),
        "max_bbox_ratio": round(max_bbox_ratio, 4),
    }


def generate() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in list(FRAMES_DIR.glob("*.png")) + list(PREVIEW_DIR.glob("*.png")):
        old_file.unlink()

    expected_instances: list[dict[str, object]] = []
    instances_by_frame: dict[str, list[dict[str, object]]] = {}
    for frame_index, scene in enumerate(SCENES, start=1):
        instances = render_frame(frame_index, scene)
        expected_instances.extend(instances)
        instances_by_frame[f"frame_{frame_index:03d}"] = instances

    comparisons, events = build_events(instances_by_frame)
    manifest = {
        "schema_version": "stream-input-0.1",
        "stream_id": STREAM_ID,
        "scene_description": "Independent controlled held-out stationery stream with ten local visual types, irregular placement, reappearance, count changes, motion, rotation, scale, and color variations.",
        "ordering": "manifest",
        "frames": [
            {"frame_id": f"frame_{index:03d}", "index": index, "image_path": f"frames/frame_{index:03d}.png", "notes": FRAME_NOTES[index - 1]}
            for index in range(1, 21)
        ],
        "notes": "Независимый held-out поток для финальной оценки frozen pipeline; не используется для настройки алгоритма.",
        "metadata": {
            "source": "deterministic_pillow_generator",
            "generator": "generate_stream.py",
            "purpose": "final_evaluation_heldout",
            "frame_size": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
            "frame_format": "png_rgb",
            "background": {"kind": "fixed_vertical_gradient", "top": BACKGROUND_TOP, "bottom": BACKGROUND_BOTTOM},
            "is_final_dataset": True,
        },
    }
    annotation = {
        "schema_version": "stream-pilot-annotation-0.1",
        "stream_id": STREAM_ID,
        "manifest_ref": "manifest.json",
        "annotation_scope": "final_heldout",
        "visual_types": [
            {
                "visual_type_id": spec.visual_type_id,
                "description": spec.description,
                "notes": (
                    f"Локальный ID потока {STREAM_ID}; geometric_class={spec.geometric_class}; "
                    f"object={spec.kind}; similarity_group={spec.similarity_group or 'none'}."
                ),
            }
            for spec in TYPE_SPECS
        ],
        "expected_element_instances": expected_instances,
        "frame_comparisons": comparisons,
        "change_events": events,
        "allowed_event_types": ["persisted", "appeared", "disappeared", "count_changed", "position_changed"],
        "uncertainty": [],
        "notes": "Полная детерминированная held-out разметка. visual_type_id локальны внутри stream_id. Цвет, масштаб и поворот записаны в notes и не создают отдельного события. Все положительные события соседних кадров перечислены; отсутствие события является отрицательной меткой.",
    }

    write_json(ROOT / "manifest.json", manifest)
    write_json(ROOT / "annotation.json", annotation)
    for frame_index in range(1, 21):
        render_preview(frame_index, instances_by_frame[f"frame_{frame_index:03d}"])

    parsed_manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    parsed_annotation = json.loads((ROOT / "annotation.json").read_text(encoding="utf-8"))
    summary = validate_outputs(parsed_manifest, parsed_annotation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    generate()
