from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


STREAM_ID = "heldout_05_game_pieces"
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

BACKGROUND_TOP = "#292F4A"
BACKGROUND_BOTTOM = "#373D59"


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
    TypeSpec("checker", "круглая шашка", "circle", "circle_subclass_01", "Плоская круглая шашка с двойным концентрическим ободом.", "round_game"),
    TypeSpec("round_token", "круглый жетон", "circle", "circle_subclass_02", "Круглый жетон с толстым ободом и крупной внутренней звездой.", "round_game"),
    TypeSpec("square_tile", "квадратная игровая плитка", "square", "square_subclass_01", "Квадратная плитка со сплошным полем и двойной рамкой.", "square_game"),
    TypeSpec("emblem_tile", "квадратная плитка с эмблемой", "square", "square_subclass_02", "Квадратная плитка с ромбовидной эмблемой и угловыми метками.", "square_game"),
    TypeSpec("domino", "кость домино", "rectangle", "rectangle_subclass_01", "Домино с центральным разделителем и крупными круглыми точками.", "rectangle_game"),
    TypeSpec("card", "игровая карта", "rectangle", "rectangle_subclass_02", "Игровая карта с тонкой рамкой и одним крупным центральным знаком.", "rectangle_game"),
    TypeSpec("triangle_token", "треугольный жетон", "triangle", "triangle_subclass_01", "Треугольный жетон с центральным круглым отверстием.", ""),
    TypeSpec("pawn", "пешка", "undefined", "undefined_subclass_01", "Пешка с круглой головой, узкой шеей и широким основанием.", "figure_game"),
    TypeSpec("meeple", "фишка-человечек", "undefined", "undefined_subclass_02", "Фишка с головой, разведенными руками и двумя ногами.", "figure_game"),
    TypeSpec("hex_token", "шестиугольный жетон", "undefined", "undefined_subclass_03", "Шестиугольный жетон с тремя крупными внутренними точками.", ""),
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


D0 = lambda: pl("checker", (50, 72), body_color="#E2B85C", difficulty_tags="similar_subclass,border_near")
D1 = lambda: pl("checker", (125, 72), body_color="#E2B85C", difficulty_tags="position_change,similar_subclass")
D2 = lambda: pl("checker", (220, 95), slot=2, body_color="#C99745", difficulty_tags="count_change,similar_subclass")
D3 = lambda: pl("checker", (315, 70), slot=3, body_color="#E0C06F", difficulty_tags="count_change,similar_subclass")
TOKEN = lambda color="#D97873": pl("round_token", (410, 85), body_color=color, difficulty_tags="similar_subclass,reappearance")
Q1 = lambda: pl("square_tile", (500, 70), body_color="#75B8D1", difficulty_tags="similar_subclass")
Q2 = lambda: pl("square_tile", (590, 95), slot=2, body_color="#90C6D8", difficulty_tags="count_change,similar_subclass")
EMBLEM0 = lambda: pl("emblem_tile", (90, 205), body_color="#88C17A", difficulty_tags="similar_subclass")
EMBLEM1 = lambda: pl("emblem_tile", (90, 205), rotation=25, body_color="#D18A66", difficulty_tags="rotation,color_variation,similar_subclass")
DOMINO = lambda: pl("domino", (220, 195), body_color="#E8E3D7", difficulty_tags="similar_subclass,reappearance")
CARD0 = lambda rotation=0: pl("card", (370, 225), rotation=rotation, body_color="#B897D3", difficulty_tags="similar_subclass" + (",rotation" if rotation else ""))
TRIANGLE = lambda: pl("triangle_token", (520, 205), rotation=-8, body_color="#E48C76")
PAWN = lambda scale=1.0: pl("pawn", (100, 380), scale=scale, body_color="#72BFA8", difficulty_tags="figure_game" + (",scale" if scale != 1 else ""))
MEEPLE0 = lambda: pl("meeple", (300, 365), body_color="#D5A45A", difficulty_tags="figure_game")
MEEPLE1 = lambda: pl("meeple", (375, 365), body_color="#D5A45A", difficulty_tags="position_change,figure_game")
HEX0 = lambda center=(525, 320): pl("hex_token", center, body_color="#8E9DD6", difficulty_tags="position_change")


SCENES: list[list[Placement]] = [
    [D0(), Q1(), DOMINO(), PAWN()],
    [D0(), TOKEN(), Q1(), DOMINO(), PAWN()],
    [D1(), TOKEN(), Q1(), DOMINO(), PAWN()],
    [D1(), TOKEN(), Q1(), Q2(), DOMINO(), PAWN()],
    [D1(), TOKEN(), Q1(), Q2(), PAWN()],
    [D1(), TOKEN(), Q1(), Q2(), EMBLEM0(), PAWN()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), PAWN()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(), PAWN()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), PAWN()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3)],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE0()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1()],
    [D1(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), Q1(), Q2(), EMBLEM0(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), Q1(), Q2(), EMBLEM0(), DOMINO(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM0(), DOMINO(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), TOKEN("#6FAEC0"), Q1(), Q2(), EMBLEM1(), DOMINO(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0()],
    [D1(), D2(), D3(), TOKEN("#6FAEC0"), Q1(), EMBLEM1(), DOMINO(), CARD0(30), TRIANGLE(), PAWN(1.3), MEEPLE1(), HEX0((525, 395))],
]

FRAME_NOTES = [
    "Начальная сцена: шашка, квадратная плитка, домино и пешка.", "Появляется круглый жетон.",
    "Шашка перемещается на 75 px.", "Добавляется вторая квадратная плитка.", "Исчезает домино.",
    "Появляется плитка с эмблемой.", "Круглый жетон меняет цвет.", "Появляется игровая карта.",
    "Карта поворачивается вокруг центра.", "Появляется треугольный жетон.", "Пешка увеличивается вокруг центра.",
    "Появляется фишка-человечек.", "Фишка-человечек перемещается на 75 px.", "Появляется шестиугольный жетон.",
    "Число шашек увеличивается до трех.", "Исчезает круглый жетон.", "Возвращается домино.",
    "Возвращается круглый жетон.", "Плитка с эмблемой меняет цвет и поворачивается вокруг центра.",
    "Число обычных плиток уменьшается до одной; шестиугольный жетон перемещается на 75 px.",
]

POSITION_CHANGES = {
    3: {"circle_subclass_01"},
    13: {"undefined_subclass_02"},
    20: {"undefined_subclass_03"},
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


def draw_checker(color: str) -> Image.Image:
    image,draw=new_asset(62,62);outline=shade(color,0.42);draw.ellipse(scaled_box((2,2,60,60)),fill=color,outline=outline,width=scaled(3));draw.ellipse(scaled_box((10,10,52,52)),outline=shade(color,1.2),width=scaled(3));draw.ellipse(scaled_box((19,19,43,43)),outline=outline,width=scaled(2));return image


def draw_round_token(color: str) -> Image.Image:
    image,draw=new_asset(64,64);outline=shade(color,0.42);draw.ellipse(scaled_box((2,2,62,62)),fill=color,outline=outline,width=scaled(3));draw.ellipse(scaled_box((9,9,55,55)),outline=shade(color,1.2),width=scaled(2))
    pts=[]
    for i in range(10):
        a=-math.pi/2+i*math.pi/5;r=18 if i%2==0 else 8;pts.append((32+r*math.cos(a),32+r*math.sin(a)))
    draw.polygon(scaled_points(pts),fill="#F1D47A",outline=outline,width=scaled(2));return image


def draw_square_tile(color: str) -> Image.Image:
    image,draw=new_asset(64,64);outline=shade(color,0.42);draw.rounded_rectangle(scaled_box((2,2,62,62)),radius=scaled(7),fill=color,outline=outline,width=scaled(3));draw.rounded_rectangle(scaled_box((10,10,54,54)),radius=scaled(4),outline=shade(color,1.2),width=scaled(3));return image


def draw_emblem_tile(color: str) -> Image.Image:
    image,draw=new_asset(66,66);outline=shade(color,0.42);draw.rounded_rectangle(scaled_box((2,2,64,64)),radius=scaled(7),fill=color,outline=outline,width=scaled(3));draw.polygon(scaled_points([(33,13),(53,33),(33,53),(13,33)]),fill=shade(color,1.2),outline=outline,width=scaled(2))
    for x,y in ((11,11),(55,11),(11,55),(55,55)):
        draw.ellipse(scaled_box((x-3,y-3,x+3,y+3)),fill="#F0D47A")
    return image


def draw_domino(color: str) -> Image.Image:
    image,draw=new_asset(108,54);outline="#5C6468";draw.rounded_rectangle(scaled_box((2,2,106,52)),radius=scaled(7),fill=color,outline=outline,width=scaled(3));draw.line(scaled_points([(54,5),(54,49)]),fill=outline,width=scaled(2))
    for x,y in ((20,16),(38,36),(70,14),(86,27),(70,40)):
        draw.ellipse(scaled_box((x-4,y-4,x+4,y+4)),fill="#4B5256")
    return image


def draw_card(color: str) -> Image.Image:
    image,draw=new_asset(72,96);outline=shade(color,0.42);draw.rounded_rectangle(scaled_box((2,2,70,94)),radius=scaled(7),fill=color,outline=outline,width=scaled(3));draw.rounded_rectangle(scaled_box((9,9,63,87)),radius=scaled(5),outline=shade(color,1.2),width=scaled(2));draw.polygon(scaled_points([(36,28),(51,48),(36,68),(21,48)]),fill="#F0D47A",outline=outline,width=scaled(2));return image


def draw_triangle_token(color: str) -> Image.Image:
    image,draw=new_asset(78,72);outline=shade(color,0.42);draw.polygon(scaled_points([(39,3),(75,68),(3,68)]),fill=color,outline=outline,width=scaled(3));draw.ellipse(scaled_box((28,35,50,57)),fill=(0,0,0,0));draw.ellipse(scaled_box((28,35,50,57)),outline=outline,width=scaled(3));return image


def draw_pawn(color: str) -> Image.Image:
    image,draw=new_asset(64,82);outline=shade(color,0.42);draw.ellipse(scaled_box((20,3,44,27)),fill=color,outline=outline,width=scaled(3));draw.polygon(scaled_points([(25,25),(39,25),(45,57),(54,69),(54,78),(10,78),(10,69),(19,57)]),fill=color,outline=outline,width=scaled(3));draw.line(scaled_points([(15,67),(49,67)]),fill=shade(color,1.2),width=scaled(2));return image


def draw_meeple(color: str) -> Image.Image:
    image,draw=new_asset(78,82);outline=shade(color,0.42);draw.ellipse(scaled_box((28,3,50,25)),fill=color,outline=outline,width=scaled(3));draw.polygon(scaled_points([(29,24),(49,24),(75,42),(67,54),(52,45),(57,77),(42,77),(39,57),(36,77),(21,77),(26,45),(11,54),(3,42)]),fill=color,outline=outline,width=scaled(3));return image


def draw_hex_token(color: str) -> Image.Image:
    image,draw=new_asset(76,70);outline=shade(color,0.42);pts=[(20,3),(56,3),(73,35),(56,67),(20,67),(3,35)];draw.polygon(scaled_points(pts),fill=color,outline=outline,width=scaled(3))
    for x,y in ((26,27),(50,27),(38,47)):
        draw.ellipse(scaled_box((x-5,y-5,x+5,y+5)),fill="#F1D47A",outline=outline,width=scaled(1))
    return image


DRAWERS = {
    "checker": draw_checker,
    "round_token": draw_round_token,
    "square_tile": draw_square_tile,
    "emblem_tile": draw_emblem_tile,
    "domino": draw_domino,
    "card": draw_card,
    "triangle_token": draw_triangle_token,
    "pawn": draw_pawn,
    "meeple": draw_meeple,
    "hex_token": draw_hex_token,
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
        "scene_description": "Independent controlled held-out game-pieces stream with ten local visual types, irregular placement, structurally distinct similar tokens, reappearance, count changes, motion, rotation, scale, and color variations.",
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
