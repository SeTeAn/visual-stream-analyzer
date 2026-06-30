from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageOps


STREAM_ID = "probe_01_stationery"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
AA = 4
BBOX_PADDING = 2
ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "frames"

BACKGROUND_TOP = "#1E3035"
BACKGROUND_BOTTOM = "#29464A"


@dataclass(frozen=True)
class TypeSpec:
    kind: str
    object_name: str
    geometric_class: str
    visual_type_id: str
    description: str


@dataclass(frozen=True)
class Placement:
    kind: str
    slot: int
    center: tuple[int, int]
    rotation: float = 0.0
    scale: float = 1.0


TYPE_SPECS = [
    TypeSpec(
        "sticky_note",
        "стикер",
        "square",
        "square_subclass_01",
        "Квадратный стикер с загнутым верхним углом.",
    ),
    TypeSpec(
        "notebook",
        "блокнот",
        "rectangle",
        "rectangle_subclass_01",
        "Прямоугольный блокнот с корешком, этикеткой и линиями страниц.",
    ),
    TypeSpec(
        "eraser",
        "ластик",
        "rectangle",
        "rectangle_subclass_02",
        "Прямоугольный ластик со светлой бумажной полоской.",
    ),
    TypeSpec(
        "pencil",
        "карандаш",
        "rectangle",
        "rectangle_subclass_03",
        "Вытянутый карандаш с деревянным кончиком и ластиком.",
    ),
    TypeSpec(
        "tape_roll",
        "рулон скотча",
        "circle",
        "circle_subclass_01",
        "Круглый рулон скотча с центральным отверстием.",
    ),
    TypeSpec(
        "set_square",
        "чертежный угольник",
        "triangle",
        "triangle_subclass_01",
        "Треугольный чертежный угольник с внутренним вырезом.",
    ),
    TypeSpec(
        "scissors",
        "ножницы",
        "undefined",
        "undefined_subclass_01",
        "Ножницы с двумя лезвиями, кольцевыми ручками и общим шарниром.",
    ),
]

TYPE_BY_KIND = {spec.kind: spec for spec in TYPE_SPECS}
TYPE_ORDER = {spec.visual_type_id: index for index, spec in enumerate(TYPE_SPECS)}


SCENES: list[list[Placement]] = [
    [
        Placement("notebook", 1, (315, 105)),
        Placement("pencil", 1, (165, 390), rotation=-10),
        Placement("tape_roll", 1, (540, 100)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("notebook", 1, (315, 105)),
        Placement("pencil", 1, (165, 390), rotation=35),
        Placement("tape_roll", 1, (540, 215)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("notebook", 1, (315, 105)),
        Placement("pencil", 1, (165, 390), rotation=35),
        Placement("tape_roll", 1, (540, 215)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("sticky_note", 2, (170, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("pencil", 1, (165, 390), rotation=35),
        Placement("tape_roll", 1, (540, 215)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("sticky_note", 2, (170, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("eraser", 1, (105, 235)),
        Placement("pencil", 1, (165, 390), rotation=35),
        Placement("tape_roll", 1, (540, 215)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("sticky_note", 2, (170, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("eraser", 1, (105, 235), scale=1.7),
        Placement("pencil", 1, (165, 390), rotation=110),
        Placement("tape_roll", 1, (540, 215)),
        Placement("scissors", 1, (500, 380), rotation=15),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("sticky_note", 2, (170, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("eraser", 1, (105, 235), scale=1.4),
        Placement("pencil", 1, (165, 390), rotation=110),
        Placement("tape_roll", 1, (540, 215)),
        Placement("set_square", 1, (500, 380)),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("eraser", 1, (105, 235), scale=1.25),
        Placement("pencil", 1, (165, 390), rotation=110),
        Placement("tape_roll", 1, (540, 215)),
        Placement("set_square", 1, (485, 365), rotation=120),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("notebook", 1, (315, 215)),
        Placement("eraser", 1, (520, 210), scale=1.25),
        Placement("pencil", 1, (165, 390), rotation=110),
        Placement("set_square", 1, (485, 365), rotation=120),
    ],
    [
        Placement("sticky_note", 1, (65, 100)),
        Placement("notebook", 1, (290, 145), rotation=12, scale=1.15),
        Placement("eraser", 1, (520, 210), scale=1.25),
        Placement("pencil", 1, (165, 390), rotation=260),
        Placement("set_square", 1, (485, 365), rotation=120),
    ],
]

FRAME_NOTES = [
    "Начальное состояние: блокнот, карандаш, рулон скотча и ножницы.",
    "Рулон скотча перемещен; карандаш повернут без смены подкласса.",
    "Появился первый стикер.",
    "Добавлен второй стикер; блокнот перемещен.",
    "Появился ластик обычного размера.",
    "Ластик значительно увеличен; карандаш повернут без перемещения.",
    "Ножницы исчезли; появился чертежный угольник.",
    "Один стикер удален; угольник перемещен и повернут.",
    "Рулон скотча исчез; ластик перемещен и уменьшен.",
    "Блокнот перемещен, повернут и немного увеличен; карандаш повернут.",
]

POSITION_CHANGES = {
    2: {"circle_subclass_01"},
    4: {"rectangle_subclass_01"},
    8: {"triangle_subclass_01"},
    9: {"rectangle_subclass_02"},
    10: {"rectangle_subclass_01"},
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def scaled(value: float) -> int:
    return int(round(value * AA))


def scaled_box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(scaled(value) for value in values)


def scaled_points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(scaled(x), scaled(y)) for x, y in values]


def new_asset(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width * AA, height * AA), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def draw_sticky_note() -> Image.Image:
    image, draw = new_asset(80, 80)
    outline = "#8B6B18"
    draw.polygon(
        scaled_points([(4, 4), (61, 4), (76, 19), (76, 76), (4, 76)]),
        fill="#F2C94C",
        outline=outline,
        width=scaled(3),
    )
    draw.polygon(
        scaled_points([(61, 4), (61, 19), (76, 19)]),
        fill="#FFE58A",
        outline=outline,
    )
    draw.line(scaled_points([(16, 45), (63, 45)]), fill="#B9912A", width=scaled(2))
    return image


def draw_notebook() -> Image.Image:
    image, draw = new_asset(150, 108)
    draw.rounded_rectangle(
        scaled_box((3, 3, 147, 105)),
        radius=scaled(8),
        fill="#D95D55",
        outline="#6E3034",
        width=scaled(3),
    )
    draw.rounded_rectangle(scaled_box((7, 7, 27, 101)), radius=scaled(5), fill="#74333A")
    for y in (18, 36, 54, 72, 90):
        draw.rounded_rectangle(scaled_box((18, y - 3, 32, y + 3)), radius=scaled(2), fill="#E9D6C2")
    draw.rounded_rectangle(
        scaled_box((55, 31, 124, 77)),
        radius=scaled(5),
        fill="#F1E7D3",
        outline="#8F4C48",
        width=scaled(2),
    )
    draw.line(scaled_points([(67, 48), (112, 48)]), fill="#B57268", width=scaled(2))
    draw.line(scaled_points([(67, 60), (105, 60)]), fill="#B57268", width=scaled(2))
    return image


def draw_eraser() -> Image.Image:
    image, draw = new_asset(90, 50)
    draw.rounded_rectangle(
        scaled_box((3, 3, 87, 47)),
        radius=scaled(10),
        fill="#5BC0BE",
        outline="#275D61",
        width=scaled(3),
    )
    draw.rectangle(scaled_box((31, 4, 61, 46)), fill="#F1E7D3", outline="#A89A82", width=scaled(2))
    draw.line(scaled_points([(36, 17), (56, 17)]), fill="#C6B89F", width=scaled(2))
    draw.line(scaled_points([(36, 32), (56, 32)]), fill="#C6B89F", width=scaled(2))
    return image


def draw_pencil() -> Image.Image:
    image, draw = new_asset(170, 34)
    outline = "#6E4A1F"
    draw.polygon(
        scaled_points([(5, 17), (27, 4), (146, 4), (146, 30), (27, 30)]),
        fill="#E8A33B",
        outline=outline,
        width=scaled(2),
    )
    draw.polygon(
        scaled_points([(5, 17), (27, 4), (27, 30)]),
        fill="#E8D0A9",
        outline=outline,
        width=scaled(2),
    )
    draw.polygon(scaled_points([(5, 17), (13, 12), (13, 22)]), fill="#2C3438")
    draw.rectangle(scaled_box((146, 4, 166, 30)), fill="#E27B8C", outline=outline, width=scaled(2))
    draw.line(scaled_points([(32, 12), (140, 12)]), fill="#F3C96E", width=scaled(2))
    return image


def draw_tape_roll() -> Image.Image:
    image, draw = new_asset(86, 86)
    draw.ellipse(
        scaled_box((3, 3, 83, 83)),
        fill="#A58ADE",
        outline="#56457E",
        width=scaled(3),
    )
    draw.ellipse(scaled_box((25, 25, 61, 61)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((25, 25, 61, 61)), outline="#56457E", width=scaled(3))
    draw.arc(scaled_box((12, 12, 74, 74)), start=205, end=310, fill="#D8C9F3", width=scaled(3))
    return image


def draw_set_square() -> Image.Image:
    image, draw = new_asset(125, 105)
    outer = [(5, 100), (5, 5), (120, 100)]
    inner = [(30, 82), (30, 35), (87, 82)]
    draw.polygon(scaled_points(outer), fill="#65CFA4", outline="#246B5A", width=scaled(3))
    draw.polygon(scaled_points(inner), fill=(0, 0, 0, 0))
    draw.line(scaled_points(inner + [inner[0]]), fill="#246B5A", width=scaled(3))
    return image


def draw_scissors() -> Image.Image:
    image, draw = new_asset(155, 105)
    blade = "#CBD5DC"
    blade_outline = "#59656C"
    handle = "#D95D55"
    handle_outline = "#703239"
    pivot = (73, 52)

    draw.polygon(
        scaled_points([pivot, (148, 8), (151, 18), (83, 57)]),
        fill=blade,
        outline=blade_outline,
        width=scaled(2),
    )
    draw.polygon(
        scaled_points([pivot, (149, 88), (144, 98), (80, 59)]),
        fill=blade,
        outline=blade_outline,
        width=scaled(2),
    )
    draw.line(scaled_points([(70, 51), (48, 30)]), fill=handle, width=scaled(13))
    draw.line(scaled_points([(70, 56), (48, 77)]), fill=handle, width=scaled(13))
    draw.ellipse(scaled_box((7, 5, 57, 53)), fill=handle, outline=handle_outline, width=scaled(3))
    draw.ellipse(scaled_box((7, 53, 57, 101)), fill=handle, outline=handle_outline, width=scaled(3))
    draw.ellipse(scaled_box((20, 17, 45, 41)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((20, 65, 45, 89)), fill=(0, 0, 0, 0))
    draw.ellipse(scaled_box((20, 17, 45, 41)), outline=handle_outline, width=scaled(2))
    draw.ellipse(scaled_box((20, 65, 45, 89)), outline=handle_outline, width=scaled(2))
    draw.ellipse(scaled_box((65, 44, 81, 60)), fill="#E7B84A", outline=blade_outline, width=scaled(2))
    return image


DRAWERS: dict[str, Callable[[], Image.Image]] = {
    "sticky_note": draw_sticky_note,
    "notebook": draw_notebook,
    "eraser": draw_eraser,
    "pencil": draw_pencil,
    "tape_roll": draw_tape_roll,
    "set_square": draw_set_square,
    "scissors": draw_scissors,
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


def transform_asset(kind: str, scale_factor: float, rotation: float) -> Image.Image:
    asset = DRAWERS[kind]()
    if scale_factor != 1.0:
        asset = asset.resize(
            (
                max(1, round(asset.width * scale_factor)),
                max(1, round(asset.height * scale_factor)),
            ),
            Image.Resampling.LANCZOS,
        )
    if rotation:
        asset = asset.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    alpha_bbox = asset.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError(f"Rendered asset {kind!r} is empty")
    return asset.crop(alpha_bbox)


def boxes_too_close(a: dict[str, int], b: dict[str, int], gap: int = 12) -> bool:
    return not (
        a["x"] + a["width"] + gap <= b["x"] - gap
        or b["x"] + b["width"] + gap <= a["x"] - gap
        or a["y"] + a["height"] + gap <= b["y"] - gap
        or b["y"] + b["height"] + gap <= a["y"] - gap
    )


def render_frame(frame_index: int, placements: list[Placement]) -> list[dict[str, object]]:
    base_frame = make_background()
    base_output = base_frame.convert("RGB").resize(
        (FRAME_WIDTH, FRAME_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    frame = base_frame.copy()
    instances: list[dict[str, object]] = []
    bboxes: list[dict[str, int]] = []

    for placement in placements:
        spec = TYPE_BY_KIND[placement.kind]
        asset = transform_asset(placement.kind, placement.scale, placement.rotation)
        center_x = placement.center[0] * AA
        center_y = placement.center[1] * AA
        left = round(center_x - asset.width / 2)
        top = round(center_y - asset.height / 2)
        right = left + asset.width
        bottom = top + asset.height
        if left < 0 or top < 0 or right > frame.width or bottom > frame.height:
            raise ValueError(f"{placement.kind} leaves frame bounds in frame {frame_index}")

        object_frame = base_frame.copy()
        object_frame.alpha_composite(asset, (left, top))
        object_output = object_frame.convert("RGB").resize(
            (FRAME_WIDTH, FRAME_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        rendered_bbox = ImageChops.difference(object_output, base_output).convert("L").getbbox()
        if rendered_bbox is None:
            raise ValueError(f"{placement.kind} produced no rendered pixels in frame {frame_index}")
        bbox_left = max(0, rendered_bbox[0] - BBOX_PADDING)
        bbox_top = max(0, rendered_bbox[1] - BBOX_PADDING)
        bbox_right = min(FRAME_WIDTH, rendered_bbox[2] + BBOX_PADDING)
        bbox_bottom = min(FRAME_HEIGHT, rendered_bbox[3] + BBOX_PADDING)
        bbox = {
            "x": bbox_left,
            "y": bbox_top,
            "width": bbox_right - bbox_left,
            "height": bbox_bottom - bbox_top,
        }
        for previous in bboxes:
            if boxes_too_close(previous, bbox):
                raise ValueError(
                    f"Objects are closer than 24 px in frame {frame_index}: "
                    f"{previous} and {bbox}"
                )
        bboxes.append(bbox)
        frame.alpha_composite(asset, (left, top))

        frame_id = f"frame_{frame_index:03d}"
        instances.append(
            {
                "instance_id": f"inst_{placement.kind}_{placement.slot:02d}_f{frame_index:03d}",
                "frame_id": frame_id,
                "visual_type_id": spec.visual_type_id,
                "bbox": bbox,
                "characteristic_regions": [],
                "uncertainty": "",
                "notes": (
                    f"Объект: {spec.object_name}; geometric_class={spec.geometric_class}; "
                    f"slot={placement.slot}; rotation_deg={placement.rotation:g}; "
                    f"scale={placement.scale:g}."
                ),
            }
        )

    output = frame.convert("RGB").resize(
        (FRAME_WIDTH, FRAME_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    output.save(FRAMES_DIR / f"frame_{frame_index:03d}.png", format="PNG", optimize=True)
    return instances


def counts_by_type(scene: list[Placement]) -> dict[str, int]:
    counts = {spec.visual_type_id: 0 for spec in TYPE_SPECS}
    for placement in scene:
        counts[TYPE_BY_KIND[placement.kind].visual_type_id] += 1
    return counts


def average_center(scene: list[Placement], visual_type_id: str) -> tuple[float, float]:
    centers = [
        placement.center
        for placement in scene
        if TYPE_BY_KIND[placement.kind].visual_type_id == visual_type_id
    ]
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def build_events() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
                "notes": "Событие размечено на уровне визуального подкласса.",
            }
        )
        return event_id

    for to_index in range(2, len(SCENES) + 1):
        previous = SCENES[to_index - 2]
        current = SCENES[to_index - 1]
        previous_counts = counts_by_type(previous)
        current_counts = counts_by_type(current)
        from_frame_id = f"frame_{to_index - 1:03d}"
        to_frame_id = f"frame_{to_index:03d}"
        comparison_event_ids: list[str] = []

        for visual_type_id in sorted(TYPE_ORDER, key=TYPE_ORDER.get):
            old_count = previous_counts[visual_type_id]
            new_count = current_counts[visual_type_id]
            if old_count == 0 and new_count > 0:
                comparison_event_ids.append(
                    add_event(
                        "appeared",
                        visual_type_id,
                        from_frame_id,
                        to_frame_id,
                        f"Подкласс {visual_type_id} отсутствует в {from_frame_id} и присутствует в {to_frame_id}.",
                    )
                )
            elif old_count > 0 and new_count == 0:
                comparison_event_ids.append(
                    add_event(
                        "disappeared",
                        visual_type_id,
                        from_frame_id,
                        to_frame_id,
                        f"Подкласс {visual_type_id} присутствует в {from_frame_id} и отсутствует в {to_frame_id}.",
                    )
                )
            elif old_count > 0 and new_count > 0:
                comparison_event_ids.append(
                    add_event(
                        "persisted",
                        visual_type_id,
                        from_frame_id,
                        to_frame_id,
                        f"Подкласс {visual_type_id} присутствует в обоих соседних кадрах.",
                    )
                )
                if old_count != new_count:
                    comparison_event_ids.append(
                        add_event(
                            "count_changed",
                            visual_type_id,
                            from_frame_id,
                            to_frame_id,
                            f"Количество экземпляров изменилось с {old_count} до {new_count}.",
                        )
                    )
                if visual_type_id in POSITION_CHANGES.get(to_index, set()):
                    old_center = average_center(previous, visual_type_id)
                    new_center = average_center(current, visual_type_id)
                    shift = math.dist(old_center, new_center)
                    if shift <= 0:
                        raise ValueError(
                            f"Zero-distance position change for {visual_type_id}: "
                            f"{from_frame_id} -> {to_frame_id}"
                        )
                    comparison_event_ids.append(
                        add_event(
                            "position_changed",
                            visual_type_id,
                            from_frame_id,
                            to_frame_id,
                            f"Средний центр подкласса смещен примерно на {shift:.1f} px.",
                        )
                    )

        comparisons.append(
            {
                "from_frame_id": from_frame_id,
                "to_frame_id": to_frame_id,
                "expected_change_event_ids": comparison_event_ids,
                "uncertainty": "",
                "notes": "Ожидаемые события между соседними кадрами согласованного probe-сценария.",
            }
        )

    return comparisons, events


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_outputs(manifest: dict[str, object], annotation: dict[str, object]) -> None:
    errors: list[str] = []
    frames = manifest["frames"]
    frame_ids = [frame["frame_id"] for frame in frames]
    expected_frame_ids = [f"frame_{index:03d}" for index in range(1, len(SCENES) + 1)]
    if frame_ids != expected_frame_ids:
        errors.append(f"Unexpected manifest frame order: {frame_ids}")
    if manifest["stream_id"] != annotation["stream_id"]:
        errors.append("stream_id differs between manifest and annotation")

    visual_type_ids = [item["visual_type_id"] for item in annotation["visual_types"]]
    if len(visual_type_ids) != len(set(visual_type_ids)):
        errors.append("visual_type_id values are not unique inside the stream")
    known_types = set(visual_type_ids)

    instances = annotation["expected_element_instances"]
    instance_ids = [instance["instance_id"] for instance in instances]
    if len(instance_ids) != len(set(instance_ids)):
        errors.append("instance_id values are not unique")

    instances_by_frame: dict[str, list[dict[str, object]]] = {frame_id: [] for frame_id in frame_ids}
    for instance in instances:
        frame_id = instance["frame_id"]
        if frame_id not in instances_by_frame:
            errors.append(f"Unknown instance frame_id: {frame_id}")
            continue
        if instance["visual_type_id"] not in known_types:
            errors.append(f"Unknown visual_type_id in {instance['instance_id']}")
        bbox = instance["bbox"]
        if bbox["width"] <= 0 or bbox["height"] <= 0:
            errors.append(f"Non-positive bbox in {instance['instance_id']}")
        if (
            bbox["x"] < 0
            or bbox["y"] < 0
            or bbox["x"] + bbox["width"] > FRAME_WIDTH
            or bbox["y"] + bbox["height"] > FRAME_HEIGHT
        ):
            errors.append(f"Out-of-frame bbox in {instance['instance_id']}: {bbox}")
        instances_by_frame[frame_id].append(instance)

    background = make_background().convert("RGB").resize(
        (FRAME_WIDTH, FRAME_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    for frame in frames:
        frame_id = frame["frame_id"]
        image_path = ROOT / frame["image_path"]
        if not image_path.is_file():
            errors.append(f"Missing frame file: {image_path}")
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if opened.format != "PNG":
                errors.append(f"Unexpected format for {frame_id}: {opened.format}")
            if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                errors.append(f"Unexpected image size for {frame_id}: {image.size}")

        difference = ImageChops.difference(image, background).convert("L")
        union_mask = Image.new("L", image.size, 0)
        union_draw = ImageDraw.Draw(union_mask)
        for instance in instances_by_frame[frame_id]:
            bbox = instance["bbox"]
            crop = difference.crop(
                (
                    bbox["x"],
                    bbox["y"],
                    bbox["x"] + bbox["width"],
                    bbox["y"] + bbox["height"],
                )
            )
            if crop.getbbox() is None:
                errors.append(f"bbox contains no rendered object pixels: {instance['instance_id']}")
            union_draw.rectangle(
                (
                    bbox["x"],
                    bbox["y"],
                    bbox["x"] + bbox["width"] - 1,
                    bbox["y"] + bbox["height"] - 1,
                ),
                fill=255,
            )
        outside = ImageChops.multiply(difference, ImageOps.invert(union_mask))
        if outside.getbbox() is not None:
            errors.append(f"Rendered pixels exist outside annotated bboxes in {frame_id}")

    comparisons = annotation["frame_comparisons"]
    if len(comparisons) != len(frame_ids) - 1:
        errors.append("Unexpected number of frame comparisons")
    event_ids = [event["event_id"] for event in annotation["change_events"]]
    if len(event_ids) != len(set(event_ids)):
        errors.append("event_id values are not unique")
    event_by_id = {event["event_id"]: event for event in annotation["change_events"]}
    referenced_event_ids: list[str] = []
    frame_position = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    for comparison in comparisons:
        from_frame_id = comparison["from_frame_id"]
        to_frame_id = comparison["to_frame_id"]
        if frame_position.get(to_frame_id) != frame_position.get(from_frame_id, -2) + 1:
            errors.append(f"Non-neighboring comparison: {from_frame_id} -> {to_frame_id}")
        for event_id in comparison["expected_change_event_ids"]:
            referenced_event_ids.append(event_id)
            event = event_by_id.get(event_id)
            if event is None:
                errors.append(f"Unknown referenced event_id: {event_id}")
            elif (
                event["from_frame_id"] != from_frame_id
                or event["to_frame_id"] != to_frame_id
            ):
                errors.append(f"Event {event_id} points to a different frame pair")
    if sorted(referenced_event_ids) != sorted(event_ids):
        errors.append("Events and frame comparison references differ")

    allowed_event_types = set(annotation["allowed_event_types"])
    for event in annotation["change_events"]:
        if event["event_type"] not in allowed_event_types:
            errors.append(f"Unsupported event type: {event['event_type']}")
        if event["visual_type_id"] not in known_types:
            errors.append(f"Unknown event visual_type_id: {event['visual_type_id']}")

    if errors:
        raise ValueError("Generated stream validation failed:\n- " + "\n- ".join(errors))


def generate() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()

    expected_instances: list[dict[str, object]] = []
    for frame_index, scene in enumerate(SCENES, start=1):
        expected_instances.extend(render_frame(frame_index, scene))

    comparisons, events = build_events()
    manifest = {
        "schema_version": "stream-input-0.1",
        "stream_id": STREAM_ID,
        "scene_description": (
            "Controlled flat stationery stream with seven local visual subclasses, "
            "motion, rotation, scale changes, appearance, disappearance, and count changes."
        ),
        "ordering": "manifest",
        "frames": [
            {
                "frame_id": f"frame_{index:03d}",
                "index": index,
                "image_path": f"frames/frame_{index:03d}.png",
                "notes": FRAME_NOTES[index - 1],
            }
            for index in range(1, len(SCENES) + 1)
        ],
        "notes": "Пробный development-поток проекта; не является финальным evaluation-набором.",
        "metadata": {
            "source": "deterministic_pillow_generator",
            "generator": "generate_stream.py",
            "purpose": "stream_analysis_probe_development",
            "frame_size": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
            "frame_format": "png_rgb",
            "background": {
                "kind": "fixed_vertical_gradient",
                "top": BACKGROUND_TOP,
                "bottom": BACKGROUND_BOTTOM,
            },
            "is_final_dataset": False,
        },
    }

    annotation = {
        "schema_version": "stream-pilot-annotation-0.1",
        "stream_id": STREAM_ID,
        "manifest_ref": "manifest.json",
        "annotation_scope": "pilot_development",
        "visual_types": [
            {
                "visual_type_id": spec.visual_type_id,
                "description": spec.description,
                "notes": (
                    f"Локальный ID потока {STREAM_ID}; geometric_class={spec.geometric_class}; "
                    f"предмет={spec.object_name}."
                ),
            }
            for spec in TYPE_SPECS
        ],
        "expected_element_instances": expected_instances,
        "frame_comparisons": comparisons,
        "change_events": events,
        "allowed_event_types": [
            "persisted",
            "appeared",
            "disappeared",
            "count_changed",
            "position_changed",
        ],
        "uncertainty": [],
        "notes": (
            "Детерминированная pilot-разметка probe_01_stationery. "
            "visual_type_id локальны внутри stream_id; изменение только поворота или масштаба "
            "не создает отдельного события."
        ),
    }

    write_json(ROOT / "manifest.json", manifest)
    write_json(ROOT / "annotation.json", annotation)
    parsed_manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    parsed_annotation = json.loads((ROOT / "annotation.json").read_text(encoding="utf-8"))
    validate_outputs(parsed_manifest, parsed_annotation)
    print(
        f"Generated {len(SCENES)} frames, {len(expected_instances)} instances, "
        f"and {len(events)} events in {ROOT}; validation passed"
    )


if __name__ == "__main__":
    generate()
