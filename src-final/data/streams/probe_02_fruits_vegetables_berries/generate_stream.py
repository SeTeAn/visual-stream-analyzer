from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageOps


STREAM_ID = "probe_02_fruits_vegetables_berries"
WIDTH, HEIGHT, AA, BBOX_PADDING = 640, 480, 4, 2
ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "frames"
BACKGROUND_TOP = "#25324A"
BACKGROUND_BOTTOM = "#31445C"


@dataclass(frozen=True)
class TypeSpec:
    kind: str
    name: str
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
    tone: str = "default"


TYPE_SPECS = [
    TypeSpec("apple", "яблоко", "circle", "circle_subclass_01", "Округлое яблоко с углублением, плодоножкой и листом."),
    TypeSpec("orange", "апельсин", "circle", "circle_subclass_02", "Круглый апельсин с точечной фактурой кожуры."),
    TypeSpec("lemon", "лимон", "oval", "oval_subclass_01", "Вытянутый целый лимон с заостренными концами."),
    TypeSpec("cucumber", "огурец", "oval", "oval_subclass_02", "Сильно вытянутый огурец с простыми отметками на кожуре."),
    TypeSpec("strawberry", "клубника", "triangle", "triangle_subclass_01", "Клубника с сужением книзу, чашелистиком и семенами."),
    TypeSpec("carrot", "морковь", "triangle", "triangle_subclass_02", "Вытянутая морковь с ботвой и продольными линиями."),
    TypeSpec("berry_cluster", "связанная гроздь ягод", "undefined", "undefined_subclass_01", "Гроздь ягод, соединенных общей ветвью и размечаемых одним объектом."),
]
TYPE_BY_KIND = {item.kind: item for item in TYPE_SPECS}
TYPE_ORDER = {item.visual_type_id: index for index, item in enumerate(TYPE_SPECS)}


SCENES = [
    [Placement("apple", 1, (85, 90), tone="dark_green"), Placement("lemon", 1, (285, 90)), Placement("strawberry", 1, (90, 365), tone="red"), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="green_red"), Placement("lemon", 1, (285, 90), rotation=25), Placement("strawberry", 1, (90, 365), tone="red"), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="green_red"), Placement("lemon", 1, (285, 90), rotation=25), Placement("orange", 1, (510, 235), tone="dark_orange"), Placement("strawberry", 1, (90, 365), tone="red"), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="light_green"), Placement("lemon", 1, (285, 90), rotation=25), Placement("orange", 1, (510, 235), tone="dark_orange"), Placement("cucumber", 1, (300, 220), tone="dark_green"), Placement("strawberry", 1, (90, 365), tone="red"), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="light_green"), Placement("lemon", 1, (285, 90), rotation=25), Placement("orange", 1, (510, 235), tone="dark_orange"), Placement("cucumber", 1, (300, 220), tone="dark_green"), Placement("strawberry", 1, (150, 350), tone="red"), Placement("carrot", 1, (300, 385)), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="light_green"), Placement("lemon", 1, (285, 90), rotation=25, scale=1.6), Placement("orange", 1, (510, 235), tone="dark_orange"), Placement("cucumber", 1, (300, 220), tone="light_green"), Placement("strawberry", 1, (150, 350), tone="blue"), Placement("carrot", 1, (300, 385), rotation=18), Placement("berry_cluster", 1, (510, 90))],
    [Placement("apple", 1, (85, 90), tone="light_green"), Placement("lemon", 1, (285, 90), rotation=25, scale=1.6), Placement("orange", 1, (510, 235), tone="light_orange"), Placement("cucumber", 1, (300, 220), tone="light_green"), Placement("strawberry", 1, (150, 350), tone="blue"), Placement("carrot", 1, (300, 385), rotation=18)],
    [Placement("apple", 1, (85, 90), tone="light_green"), Placement("lemon", 1, (285, 90), rotation=25, scale=1.6), Placement("orange", 1, (510, 235), scale=0.8, tone="light_orange"), Placement("cucumber", 1, (300, 220), tone="light_green"), Placement("strawberry", 1, (85, 365), rotation=28, tone="red"), Placement("carrot", 1, (300, 385), rotation=18)],
    [Placement("lemon", 1, (285, 90), rotation=25, scale=1.6), Placement("orange", 1, (510, 235), scale=0.8, tone="light_orange"), Placement("cucumber", 1, (300, 220), tone="light_green"), Placement("strawberry", 1, (85, 365), rotation=28, tone="red"), Placement("carrot", 1, (300, 385), rotation=18), Placement("berry_cluster", 1, (510, 90))],
    [Placement("lemon", 1, (285, 90), rotation=25), Placement("orange", 1, (510, 235), scale=0.8, tone="light_orange"), Placement("cucumber", 1, (300, 220), tone="light_green"), Placement("strawberry", 1, (85, 365), rotation=28, tone="red"), Placement("carrot", 1, (500, 390), rotation=18), Placement("berry_cluster", 1, (510, 90))],
]

FRAME_NOTES = [
    "Начальная сцена: темно-зеленое яблоко, лимон, красная клубника и гроздь ягод.",
    "Яблоко стало зелено-красным; лимон повернут без смены подклассов.",
    "Появился апельсин темно-оранжевого оттенка.",
    "Появился огурец; яблоко стало светло-зеленым.",
    "Появилась морковь; клубника перемещена.",
    "Лимон значительно увеличен; огурец стал светлее; клубника стала синей; морковь повернута.",
    "Гроздь исчезла; апельсин стал светло-оранжевым.",
    "Апельсин уменьшен; клубника снова красная, перемещена и повернута.",
    "Яблоко исчезло; гроздь вернулась после двух кадров отсутствия.",
    "Морковь перемещена; лимон возвращен к обычному размеру.",
]

POSITION_CHANGES = {5: {"triangle_subclass_01"}, 8: {"triangle_subclass_01"}, 10: {"triangle_subclass_02"}}


def s(value: float) -> int:
    return round(value * AA)


def box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(s(value) for value in values)


def points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(s(x), s(y)) for x, y in values]


def asset(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width * AA, height * AA), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def draw_apple(tone: str) -> Image.Image:
    image, draw = asset(92, 96)
    colors = {"dark_green": "#356B3D", "light_green": "#86C96B", "green_red": "#5AA04F"}
    body_box = box((8, 17, 84, 91))
    draw.ellipse(body_box, fill=colors.get(tone, "#86C96B"), outline="#294B31", width=s(3))
    if tone == "green_red":
        draw.pieslice(body_box, start=-90, end=90, fill="#C9584F")
        draw.ellipse(body_box, outline="#294B31", width=s(3))
    draw.line(points([(47, 20), (51, 5)]), fill="#65432B", width=s(5))
    draw.ellipse(box((50, 6, 78, 25)), fill="#679B4B", outline="#315A37", width=s(2))
    return image


def draw_orange(tone: str) -> Image.Image:
    image, draw = asset(86, 86)
    fill = "#C96724" if tone == "dark_orange" else "#F2A34A"
    draw.ellipse(box((4, 4, 82, 82)), fill=fill, outline="#87451F", width=s(3))
    for x, y in [(24, 24), (45, 18), (64, 31), (28, 52), (51, 45), (64, 61), (42, 70)]:
        draw.ellipse(box((x - 2, y - 2, x + 2, y + 2)), fill="#A95525")
    draw.ellipse(box((38, 5, 49, 12)), fill="#527443")
    return image


def draw_lemon(_: str) -> Image.Image:
    image, draw = asset(124, 68)
    draw.polygon(points([(4, 34), (18, 13), (55, 5), (101, 10), (120, 34), (103, 57), (52, 63), (17, 55)]), fill="#E9CC4D", outline="#8C7625", width=s(3))
    draw.arc(box((30, 12, 103, 57)), 205, 325, fill="#F7E58C", width=s(3))
    return image


def draw_cucumber(tone: str) -> Image.Image:
    image, draw = asset(134, 58)
    fill = "#3E8A52" if tone == "dark_green" else "#72B96A"
    draw.rounded_rectangle(box((3, 5, 131, 53)), radius=s(22), fill=fill, outline="#275A39", width=s(3))
    for x, y in [(25, 20), (45, 37), (66, 18), (88, 36), (110, 20)]:
        draw.line(points([(x - 3, y), (x + 3, y)]), fill="#B5D987", width=s(2))
    return image


def draw_strawberry(tone: str) -> Image.Image:
    image, draw = asset(84, 106)
    fill = "#4F75C9" if tone == "blue" else "#D84F55"
    draw.polygon(points([(9, 27), (42, 15), (75, 27), (68, 69), (42, 100), (16, 69)]), fill=fill, outline="#73333A", width=s(3))
    draw.polygon(points([(12, 28), (27, 7), (42, 22), (55, 5), (73, 28), (42, 22)]), fill="#5A9E4B", outline="#315F35", width=s(2))
    for x, y in [(25, 39), (48, 38), (62, 51), (34, 58), (49, 73), (29, 77)]:
        draw.ellipse(box((x - 2, y - 3, x + 2, y + 3)), fill="#F4D27A")
    return image


def draw_carrot(_: str) -> Image.Image:
    image, draw = asset(82, 138)
    draw.polygon(points([(16, 35), (66, 35), (43, 132)]), fill="#E88B32", outline="#8A4D23", width=s(3))
    draw.line(points([(28, 50), (48, 54)]), fill="#F2B160", width=s(2))
    draw.line(points([(25, 72), (43, 76)]), fill="#F2B160", width=s(2))
    draw.line(points([(39, 35), (22, 4)]), fill="#4F8C48", width=s(7))
    draw.line(points([(42, 35), (43, 2)]), fill="#5E9D50", width=s(7))
    draw.line(points([(45, 35), (64, 6)]), fill="#4F8C48", width=s(7))
    return image


def draw_berry_cluster(_: str) -> Image.Image:
    image, draw = asset(126, 108)
    draw.line(points([(63, 8), (62, 35), (35, 47), (88, 48)]), fill="#668B49", width=s(6))
    berries = [(38, 44), (62, 39), (86, 46), (49, 65), (75, 65), (61, 86), (93, 70)]
    for index, (x, y) in enumerate(berries):
        fill = "#7652A6" if index % 2 else "#8B5AB3"
        draw.ellipse(box((x - 15, y - 15, x + 15, y + 15)), fill=fill, outline="#4C356C", width=s(2))
    draw.ellipse(box((65, 5, 100, 23)), fill="#5F984D", outline="#35613A", width=s(2))
    return image


DRAWERS: dict[str, Callable[[str], Image.Image]] = {
    "apple": draw_apple, "orange": draw_orange, "lemon": draw_lemon,
    "cucumber": draw_cucumber, "strawberry": draw_strawberry,
    "carrot": draw_carrot, "berry_cluster": draw_berry_cluster,
}


def background() -> Image.Image:
    top = tuple(bytes.fromhex(BACKGROUND_TOP[1:])); bottom = tuple(bytes.fromhex(BACKGROUND_BOTTOM[1:]))
    image = Image.new("RGB", (WIDTH * AA, HEIGHT * AA)); draw = ImageDraw.Draw(image)
    for y in range(HEIGHT * AA):
        ratio = y / (HEIGHT * AA - 1)
        color = tuple(round(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line((0, y, WIDTH * AA, y), fill=color)
    return image.convert("RGBA")


def transformed(item: Placement) -> Image.Image:
    image = DRAWERS[item.kind](item.tone)
    if item.scale != 1:
        image = image.resize((round(image.width * item.scale), round(image.height * item.scale)), Image.Resampling.LANCZOS)
    if item.rotation:
        image = image.rotate(item.rotation, resample=Image.Resampling.BICUBIC, expand=True)
    return image.crop(image.getchannel("A").getbbox())


def too_close(a: dict[str, int], b: dict[str, int], gap: int = 12) -> bool:
    return not (a["x"] + a["width"] + gap <= b["x"] - gap or b["x"] + b["width"] + gap <= a["x"] - gap or a["y"] + a["height"] + gap <= b["y"] - gap or b["y"] + b["height"] + gap <= a["y"] - gap)


def render_frame(index: int, scene: list[Placement]) -> list[dict[str, object]]:
    base_frame = background()
    base_output = base_frame.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    frame = base_frame.copy(); instances = []; bboxes = []
    for item in scene:
        image = transformed(item); left = round(item.center[0] * AA - image.width / 2); top = round(item.center[1] * AA - image.height / 2)
        if left < 0 or top < 0 or left + image.width > frame.width or top + image.height > frame.height:
            raise ValueError(f"{item.kind} leaves frame {index}")
        object_frame = base_frame.copy(); object_frame.alpha_composite(image, (left, top))
        object_output = object_frame.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        rendered_bbox = ImageChops.difference(object_output, base_output).convert("L").getbbox()
        if rendered_bbox is None: raise ValueError(f"{item.kind} produced no rendered pixels in frame {index}")
        x = max(0, rendered_bbox[0] - BBOX_PADDING); y = max(0, rendered_bbox[1] - BBOX_PADDING)
        right = min(WIDTH, rendered_bbox[2] + BBOX_PADDING); bottom = min(HEIGHT, rendered_bbox[3] + BBOX_PADDING)
        bbox = {"x": x, "y": y, "width": right - x, "height": bottom - y}
        if any(too_close(old, bbox) for old in bboxes): raise ValueError(f"Objects are too close in frame {index}: {item.kind} {bbox}")
        bboxes.append(bbox); frame.alpha_composite(image, (left, top)); spec = TYPE_BY_KIND[item.kind]
        instances.append({"instance_id": f"inst_{item.kind}_{item.slot:02d}_f{index:03d}", "frame_id": f"frame_{index:03d}", "visual_type_id": spec.visual_type_id, "bbox": bbox, "characteristic_regions": [], "uncertainty": "", "notes": f"Объект: {spec.name}; geometric_class={spec.geometric_class}; slot={item.slot}; rotation_deg={item.rotation:g}; scale={item.scale:g}; tone={item.tone}."})
    frame.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(FRAMES_DIR / f"frame_{index:03d}.png", optimize=True)
    return instances


def counts(scene: list[Placement]) -> Counter[str]:
    return Counter(TYPE_BY_KIND[item.kind].visual_type_id for item in scene)


def average_center(scene: list[Placement], type_id: str) -> tuple[float, float]:
    values = [item.center for item in scene if TYPE_BY_KIND[item.kind].visual_type_id == type_id]
    return sum(x for x, _ in values) / len(values), sum(y for _, y in values) / len(values)


def build_events() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events = []; comparisons = []
    for to_index in range(2, 11):
        previous, current = SCENES[to_index - 2], SCENES[to_index - 1]
        old, new = counts(previous), counts(current); ids = []
        for type_id in sorted(TYPE_ORDER, key=TYPE_ORDER.get):
            event_types = []
            if old[type_id] == 0 < new[type_id]: event_types = ["appeared"]
            elif old[type_id] > 0 == new[type_id]: event_types = ["disappeared"]
            elif old[type_id] and new[type_id]:
                event_types = ["persisted"]
                if old[type_id] != new[type_id]: event_types.append("count_changed")
                if type_id in POSITION_CHANGES.get(to_index, set()): event_types.append("position_changed")
            for event_type in event_types:
                event_id = f"event_{len(events) + 1:03d}"; ids.append(event_id)
                if event_type == "count_changed": evidence = f"Количество экземпляров изменилось с {old[type_id]} до {new[type_id]}."
                elif event_type == "position_changed": evidence = f"Средний центр подкласса смещен примерно на {math.dist(average_center(previous, type_id), average_center(current, type_id)):.1f} px."
                else: evidence = f"Событие {event_type} установлено по присутствию подкласса в соседних кадрах."
                events.append({"event_id": event_id, "event_type": event_type, "visual_type_id": type_id, "from_frame_id": f"frame_{to_index-1:03d}", "to_frame_id": f"frame_{to_index:03d}", "evidence": evidence, "uncertainty": "", "notes": "Событие размечено на уровне визуального подкласса; смена оттенка не является отдельным событием."})
        comparisons.append({"from_frame_id": f"frame_{to_index-1:03d}", "to_frame_id": f"frame_{to_index:03d}", "expected_change_event_ids": ids, "uncertainty": "", "notes": "Согласованный переход probe-сценария."})
    return comparisons, events


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(manifest: dict[str, object], annotation: dict[str, object]) -> None:
    errors = []; frame_ids = [f["frame_id"] for f in manifest["frames"]]
    if frame_ids != [f"frame_{i:03d}" for i in range(1, 11)]: errors.append("Invalid frame order")
    if manifest["stream_id"] != annotation["stream_id"]: errors.append("stream_id mismatch")
    known = {item["visual_type_id"] for item in annotation["visual_types"]}; instances = annotation["expected_element_instances"]
    if len(known) != 7 or len({item["instance_id"] for item in instances}) != len(instances): errors.append("Non-unique IDs")
    by_frame = {frame_id: [] for frame_id in frame_ids}
    for item in instances:
        if item["frame_id"] not in by_frame or item["visual_type_id"] not in known: errors.append(f"Invalid instance reference: {item['instance_id']}")
        else: by_frame[item["frame_id"]].append(item)
        b = item["bbox"]
        if b["x"] < 0 or b["y"] < 0 or b["width"] <= 0 or b["height"] <= 0 or b["x"] + b["width"] > WIDTH or b["y"] + b["height"] > HEIGHT: errors.append(f"Invalid bbox: {item['instance_id']}")
    base = background().convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    for frame in manifest["frames"]:
        path = ROOT / frame["image_path"]
        if not path.is_file(): errors.append(f"Missing {path}"); continue
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if opened.format != "PNG" or image.size != (WIDTH, HEIGHT): errors.append(f"Invalid image: {path}")
        difference = ImageChops.difference(image, base).convert("L"); mask = Image.new("L", image.size); draw = ImageDraw.Draw(mask)
        for item in by_frame[frame["frame_id"]]:
            b = item["bbox"]; crop = difference.crop((b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]))
            if crop.getbbox() is None: errors.append(f"Empty bbox: {item['instance_id']}")
            draw.rectangle((b["x"], b["y"], b["x"] + b["width"] - 1, b["y"] + b["height"] - 1), fill=255)
        if ImageChops.multiply(difference, ImageOps.invert(mask)).getbbox(): errors.append(f"Pixels outside bbox: {frame['frame_id']}")
    event_ids = {item["event_id"] for item in annotation["change_events"]}; referenced = [eid for comp in annotation["frame_comparisons"] for eid in comp["expected_change_event_ids"]]
    if len(annotation["frame_comparisons"]) != 9 or set(referenced) != event_ids or len(referenced) != len(event_ids): errors.append("Invalid event references")
    if errors: raise ValueError("Validation failed:\n- " + "\n- ".join(errors))


def generate() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for path in FRAMES_DIR.glob("frame_*.png"): path.unlink()
    instances = []
    for index, scene in enumerate(SCENES, 1): instances.extend(render_frame(index, scene))
    comparisons, events = build_events()
    manifest = {"schema_version": "stream-input-0.1", "stream_id": STREAM_ID, "scene_description": "Controlled flat stream of whole fruits, vegetables, and berries with stable subclasses across color, rotation, and scale changes.", "ordering": "manifest", "frames": [{"frame_id": f"frame_{i:03d}", "index": i, "image_path": f"frames/frame_{i:03d}.png", "notes": FRAME_NOTES[i-1]} for i in range(1, 11)], "notes": "Пробный development-поток проекта; не финальный evaluation-набор.", "metadata": {"source": "deterministic_pillow_generator", "generator": "generate_stream.py", "purpose": "stream_analysis_probe_development", "frame_size": {"width": WIDTH, "height": HEIGHT}, "frame_format": "png_rgb", "background": {"kind": "fixed_vertical_gradient", "top": BACKGROUND_TOP, "bottom": BACKGROUND_BOTTOM}, "is_final_dataset": False}}
    annotation = {"schema_version": "stream-pilot-annotation-0.1", "stream_id": STREAM_ID, "manifest_ref": "manifest.json", "annotation_scope": "pilot_development", "visual_types": [{"visual_type_id": item.visual_type_id, "description": item.description, "notes": f"Локальный ID потока {STREAM_ID}; geometric_class={item.geometric_class}; предмет={item.name}."} for item in TYPE_SPECS], "expected_element_instances": instances, "frame_comparisons": comparisons, "change_events": events, "allowed_event_types": ["persisted", "appeared", "disappeared", "count_changed", "position_changed"], "uncertainty": [], "notes": "Цвет, поворот и масштаб не меняют visual_type_id и сами по себе не создают отдельное событие."}
    write_json(ROOT / "manifest.json", manifest); write_json(ROOT / "annotation.json", annotation)
    validate(manifest, annotation)
    print(f"Generated 10 frames, {len(instances)} instances, and {len(events)} events in {ROOT}; validation passed")


if __name__ == "__main__":
    generate()
