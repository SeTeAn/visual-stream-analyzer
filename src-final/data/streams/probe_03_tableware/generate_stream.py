from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageOps


STREAM_ID = "probe_03_tableware"
WIDTH, HEIGHT, AA, BBOX_PADDING = 640, 480, 4, 2
ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "frames"
BACKGROUND_TOP = "#704A63"
BACKGROUND_BOTTOM = "#5B3D54"


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
    TypeSpec("round_plate", "круглая тарелка", "circle", "circle_subclass_01", "Круглая тарелка в виде сверху с тонким ободом и центральной зоной."),
    TypeSpec("bowl", "миска", "circle", "circle_subclass_02", "Миска в виде сверху с толстым ободом и глубокой внутренней областью."),
    TypeSpec("oval_platter", "овальное блюдо", "oval", "oval_subclass_01", "Овальное блюдо в виде сверху с внутренним кантом."),
    TypeSpec("square_plate", "квадратная тарелка", "square", "square_subclass_01", "Квадратная тарелка в виде сверху с декоративной рамкой."),
    TypeSpec("tray", "прямоугольный поднос", "rectangle", "rectangle_subclass_01", "Прямоугольный поднос в виде сверху с двумя ручками."),
    TypeSpec("triangle_plate", "треугольная сервировочная тарелка", "triangle", "triangle_subclass_01", "Треугольная тарелка в виде сверху с внутренним кантом."),
    TypeSpec("cup", "чашка", "undefined", "undefined_subclass_01", "Чашка в виде сбоку с корпусом, верхним краем и боковой ручкой."),
]
TYPE_BY_KIND = {item.kind: item for item in TYPE_SPECS}
TYPE_ORDER = {item.visual_type_id: index for index, item in enumerate(TYPE_SPECS)}


SCENES = [
    [Placement("round_plate", 1, (90, 100), tone="ivory"), Placement("oval_platter", 1, (300, 100)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("oval_platter", 1, (300, 100)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("oval_platter", 1, (300, 100)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("oval_platter", 1, (300, 100)), Placement("square_plate", 1, (300, 250)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("bowl", 1, (510, 250)), Placement("oval_platter", 1, (300, 100)), Placement("square_plate", 1, (300, 250)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("round_plate", 3, (90, 365), tone="coral"), Placement("bowl", 1, (510, 250)), Placement("oval_platter", 1, (300, 100), scale=1.6), Placement("square_plate", 1, (300, 250)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("bowl", 1, (510, 250)), Placement("oval_platter", 1, (300, 100), scale=1.6), Placement("square_plate", 1, (300, 250)), Placement("tray", 1, (300, 390)), Placement("cup", 1, (520, 100))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("round_plate", 2, (90, 100), tone="teal"), Placement("bowl", 1, (510, 250)), Placement("oval_platter", 1, (300, 100), scale=1.6), Placement("square_plate", 1, (300, 250)), Placement("tray", 1, (300, 390)), Placement("triangle_plate", 1, (510, 390))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("bowl", 1, (510, 250)), Placement("oval_platter", 1, (300, 100), scale=1.6), Placement("square_plate", 1, (300, 250), scale=1.15), Placement("tray", 1, (510, 105)), Placement("triangle_plate", 1, (510, 390))],
    [Placement("round_plate", 1, (90, 230), tone="ivory"), Placement("bowl", 1, (300, 390)), Placement("oval_platter", 1, (300, 100), scale=1.6), Placement("square_plate", 1, (300, 250), scale=1.15), Placement("tray", 1, (510, 105)), Placement("triangle_plate", 1, (510, 390)), Placement("cup", 1, (90, 390))],
]

FRAME_NOTES = [
    "Начальная сцена: круглая тарелка, овальное блюдо и чашка.",
    "Круглая тарелка перемещена.",
    "Добавлена вторая круглая тарелка другого цвета, но того же подкласса.",
    "Появилась квадратная тарелка.",
    "Появилась миска.",
    "Добавлена третья круглая тарелка; овальное блюдо значительно увеличено.",
    "Одна круглая тарелка удалена; появился прямоугольный поднос.",
    "Чашка исчезла; появилась треугольная сервировочная тарелка.",
    "Оставлена одна круглая тарелка; поднос перемещен; квадратная тарелка немного увеличена.",
    "Чашка вернулась после двух кадров отсутствия; миска перемещена.",
]

POSITION_CHANGES = {2: {"circle_subclass_01"}, 9: {"rectangle_subclass_01"}, 10: {"circle_subclass_02"}}


def s(value: float) -> int: return round(value * AA)
def box(v: tuple[float, float, float, float]) -> tuple[int, int, int, int]: return tuple(s(x) for x in v)
def points(v: list[tuple[float, float]]) -> list[tuple[int, int]]: return [(s(x), s(y)) for x, y in v]
def asset(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (w * AA, h * AA), (0, 0, 0, 0)); return image, ImageDraw.Draw(image)


def draw_round_plate(tone: str) -> Image.Image:
    image, draw = asset(92, 92)
    colors = {"ivory": ("#ECE3D1", "#A96658"), "teal": ("#63AAA1", "#285E5C"), "coral": ("#D66C65", "#7D3D42")}; fill, line = colors.get(tone, colors["ivory"])
    draw.ellipse(box((3, 3, 89, 89)), fill=fill, outline=line, width=s(3)); draw.ellipse(box((17, 17, 75, 75)), outline=line, width=s(3)); draw.ellipse(box((39, 39, 53, 53)), fill=line)
    return image


def draw_bowl(_: str) -> Image.Image:
    image, draw = asset(90, 90)
    draw.ellipse(box((3, 3, 87, 87)), fill="#5B91BD", outline="#294E70", width=s(3)); draw.ellipse(box((14, 14, 76, 76)), fill="#376D99", outline="#B8D6E8", width=s(5)); draw.ellipse(box((28, 28, 62, 62)), fill="#2E5D84")
    return image


def draw_oval_platter(_: str) -> Image.Image:
    image, draw = asset(134, 82)
    draw.ellipse(box((3, 3, 131, 79)), fill="#E7D9BC", outline="#8A6850", width=s(3)); draw.ellipse(box((15, 15, 119, 67)), outline="#B78A63", width=s(3)); draw.line(points([(42, 41), (92, 41)]), fill="#C49B70", width=s(2))
    return image


def draw_square_plate(_: str) -> Image.Image:
    image, draw = asset(94, 94)
    draw.rounded_rectangle(box((3, 3, 91, 91)), radius=s(9), fill="#57A8A0", outline="#285F5B", width=s(3)); draw.rounded_rectangle(box((16, 16, 78, 78)), radius=s(6), outline="#BDE1D9", width=s(3)); draw.rectangle(box((37, 37, 57, 57)), fill="#D8B44D")
    return image


def draw_tray(_: str) -> Image.Image:
    image, draw = asset(154, 82)
    draw.rounded_rectangle(box((10, 5, 144, 77)), radius=s(10), fill="#C55E63", outline="#733841", width=s(3)); draw.rounded_rectangle(box((23, 17, 131, 65)), radius=s(7), outline="#E7B1A5", width=s(3)); draw.rounded_rectangle(box((1, 28, 18, 54)), radius=s(7), fill="#733841"); draw.rounded_rectangle(box((136, 28, 153, 54)), radius=s(7), fill="#733841")
    return image


def draw_triangle_plate(_: str) -> Image.Image:
    image, draw = asset(116, 104)
    outer = [(58, 3), (112, 99), (4, 99)]; inner = [(58, 21), (94, 86), (22, 86)]
    draw.polygon(points(outer), fill="#D2A942", outline="#755D27"); draw.line(points(outer + [outer[0]]), fill="#755D27", width=s(3)); draw.line(points(inner + [inner[0]]), fill="#F0D988", width=s(3)); draw.ellipse(box((53, 69, 63, 79)), fill="#755D27")
    return image


def draw_cup(_: str) -> Image.Image:
    image, draw = asset(126, 92)
    draw.polygon(points([(12, 18), (85, 18), (78, 82), (20, 82)]), fill="#E8DFCC", outline="#795B50", width=s(3)); draw.ellipse(box((12, 9, 85, 29)), fill="#B8D8D0", outline="#795B50", width=s(3)); draw.ellipse(box((80, 29, 121, 72)), fill=(0,0,0,0), outline="#E8DFCC", width=s(12)); draw.ellipse(box((83, 31, 118, 69)), outline="#795B50", width=s(3)); draw.line(points([(31, 51), (66, 51)]), fill="#C55E63", width=s(3))
    return image


DRAWERS: dict[str, Callable[[str], Image.Image]] = {"round_plate": draw_round_plate, "bowl": draw_bowl, "oval_platter": draw_oval_platter, "square_plate": draw_square_plate, "tray": draw_tray, "triangle_plate": draw_triangle_plate, "cup": draw_cup}


def background() -> Image.Image:
    top = tuple(bytes.fromhex(BACKGROUND_TOP[1:])); bottom = tuple(bytes.fromhex(BACKGROUND_BOTTOM[1:])); image = Image.new("RGB", (WIDTH*AA, HEIGHT*AA)); draw = ImageDraw.Draw(image)
    for y in range(HEIGHT*AA):
        q=y/(HEIGHT*AA-1); draw.line((0,y,WIDTH*AA,y), fill=tuple(round(top[i]*(1-q)+bottom[i]*q) for i in range(3)))
    return image.convert("RGBA")


def transformed(item: Placement) -> Image.Image:
    image = DRAWERS[item.kind](item.tone)
    if item.scale != 1: image = image.resize((round(image.width*item.scale), round(image.height*item.scale)), Image.Resampling.LANCZOS)
    if item.rotation: image = image.rotate(item.rotation, resample=Image.Resampling.BICUBIC, expand=True)
    return image.crop(image.getchannel("A").getbbox())


def too_close(a: dict[str,int], b: dict[str,int], gap: int=12) -> bool:
    return not (a["x"]+a["width"]+gap <= b["x"]-gap or b["x"]+b["width"]+gap <= a["x"]-gap or a["y"]+a["height"]+gap <= b["y"]-gap or b["y"]+b["height"]+gap <= a["y"]-gap)


def render_frame(index: int, scene: list[Placement]) -> list[dict[str,object]]:
    base_frame=background(); base_output=base_frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS); frame=base_frame.copy(); instances=[]; bboxes=[]
    for item in scene:
        image=transformed(item); left=round(item.center[0]*AA-image.width/2); top=round(item.center[1]*AA-image.height/2)
        if left<0 or top<0 or left+image.width>frame.width or top+image.height>frame.height: raise ValueError(f"{item.kind} leaves frame {index}")
        object_frame=base_frame.copy(); object_frame.alpha_composite(image,(left,top)); object_output=object_frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS); rendered_bbox=ImageChops.difference(object_output,base_output).convert("L").getbbox()
        if rendered_bbox is None: raise ValueError(f"{item.kind} produced no rendered pixels in frame {index}")
        x=max(0,rendered_bbox[0]-BBOX_PADDING); y=max(0,rendered_bbox[1]-BBOX_PADDING); right=min(WIDTH,rendered_bbox[2]+BBOX_PADDING); bottom=min(HEIGHT,rendered_bbox[3]+BBOX_PADDING); bbox={"x":x,"y":y,"width":right-x,"height":bottom-y}
        if any(too_close(old,bbox) for old in bboxes): raise ValueError(f"Objects are too close in frame {index}: {item.kind} {bbox}")
        bboxes.append(bbox); frame.alpha_composite(image,(left,top)); spec=TYPE_BY_KIND[item.kind]
        instances.append({"instance_id":f"inst_{item.kind}_{item.slot:02d}_f{index:03d}","frame_id":f"frame_{index:03d}","visual_type_id":spec.visual_type_id,"bbox":bbox,"characteristic_regions":[],"uncertainty":"","notes":f"Объект: {spec.name}; geometric_class={spec.geometric_class}; slot={item.slot}; rotation_deg={item.rotation:g}; scale={item.scale:g}; tone={item.tone}."})
    frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS).save(FRAMES_DIR/f"frame_{index:03d}.png",optimize=True); return instances


def counts(scene: list[Placement]) -> Counter[str]: return Counter(TYPE_BY_KIND[item.kind].visual_type_id for item in scene)
def average_center(scene: list[Placement], type_id: str) -> tuple[float,float]:
    values=[item.center for item in scene if TYPE_BY_KIND[item.kind].visual_type_id==type_id]; return sum(x for x,_ in values)/len(values),sum(y for _,y in values)/len(values)


def build_events() -> tuple[list[dict[str,object]],list[dict[str,object]]]:
    events=[]; comparisons=[]
    for to_index in range(2,11):
        previous,current=SCENES[to_index-2],SCENES[to_index-1]; old,new=counts(previous),counts(current); ids=[]
        for type_id in sorted(TYPE_ORDER,key=TYPE_ORDER.get):
            kinds=[]
            if old[type_id]==0<new[type_id]: kinds=["appeared"]
            elif old[type_id]>0==new[type_id]: kinds=["disappeared"]
            elif old[type_id] and new[type_id]:
                kinds=["persisted"]
                if old[type_id]!=new[type_id]: kinds.append("count_changed")
                if type_id in POSITION_CHANGES.get(to_index,set()): kinds.append("position_changed")
            for kind in kinds:
                event_id=f"event_{len(events)+1:03d}"; ids.append(event_id)
                evidence=f"Количество экземпляров изменилось с {old[type_id]} до {new[type_id]}." if kind=="count_changed" else (f"Средний центр подкласса смещен примерно на {math.dist(average_center(previous,type_id),average_center(current,type_id)):.1f} px." if kind=="position_changed" else f"Событие {kind} установлено по присутствию подкласса в соседних кадрах.")
                events.append({"event_id":event_id,"event_type":kind,"visual_type_id":type_id,"from_frame_id":f"frame_{to_index-1:03d}","to_frame_id":f"frame_{to_index:03d}","evidence":evidence,"uncertainty":"","notes":"Событие размечено на уровне визуального подкласса."})
        comparisons.append({"from_frame_id":f"frame_{to_index-1:03d}","to_frame_id":f"frame_{to_index:03d}","expected_change_event_ids":ids,"uncertainty":"","notes":"Согласованный переход probe-сценария."})
    return comparisons,events


def write_json(path: Path,value: dict[str,object]) -> None: path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def validate(manifest: dict[str,object],annotation: dict[str,object]) -> None:
    errors=[]; frame_ids=[x["frame_id"] for x in manifest["frames"]]
    if frame_ids != [f"frame_{i:03d}" for i in range(1,11)]: errors.append("Invalid frame order")
    if manifest["stream_id"]!=annotation["stream_id"]: errors.append("stream_id mismatch")
    known={x["visual_type_id"] for x in annotation["visual_types"]}; instances=annotation["expected_element_instances"]; by_frame={fid:[] for fid in frame_ids}
    if len(known)!=7 or len({x["instance_id"] for x in instances})!=len(instances): errors.append("Non-unique IDs")
    for item in instances:
        if item["frame_id"] not in by_frame or item["visual_type_id"] not in known: errors.append(f"Invalid instance: {item['instance_id']}")
        else: by_frame[item["frame_id"]].append(item)
        b=item["bbox"]
        if b["x"]<0 or b["y"]<0 or b["width"]<=0 or b["height"]<=0 or b["x"]+b["width"]>WIDTH or b["y"]+b["height"]>HEIGHT: errors.append(f"Invalid bbox: {item['instance_id']}")
    base=background().convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
    for frame in manifest["frames"]:
        path=ROOT/frame["image_path"]
        if not path.is_file(): errors.append(f"Missing {path}"); continue
        with Image.open(path) as opened:
            image=opened.convert("RGB")
            if opened.format!="PNG" or image.size!=(WIDTH,HEIGHT): errors.append(f"Invalid image: {path}")
        difference=ImageChops.difference(image,base).convert("L"); mask=Image.new("L",image.size); draw=ImageDraw.Draw(mask)
        for item in by_frame[frame["frame_id"]]:
            b=item["bbox"]
            if difference.crop((b["x"],b["y"],b["x"]+b["width"],b["y"]+b["height"])).getbbox() is None: errors.append(f"Empty bbox: {item['instance_id']}")
            draw.rectangle((b["x"],b["y"],b["x"]+b["width"]-1,b["y"]+b["height"]-1),fill=255)
        if ImageChops.multiply(difference,ImageOps.invert(mask)).getbbox(): errors.append(f"Pixels outside bbox: {frame['frame_id']}")
    event_ids={x["event_id"] for x in annotation["change_events"]}; refs=[eid for c in annotation["frame_comparisons"] for eid in c["expected_change_event_ids"]]
    if len(annotation["frame_comparisons"])!=9 or set(refs)!=event_ids or len(refs)!=len(event_ids): errors.append("Invalid event references")
    if any(x["event_type"]=="position_changed" and " 0.0 px." in x["evidence"] for x in annotation["change_events"]): errors.append("Zero-distance position event")
    if errors: raise ValueError("Validation failed:\n- "+"\n- ".join(errors))


def generate() -> None:
    FRAMES_DIR.mkdir(parents=True,exist_ok=True)
    for path in FRAMES_DIR.glob("frame_*.png"): path.unlink()
    instances=[]
    for index,scene in enumerate(SCENES,1): instances.extend(render_frame(index,scene))
    comparisons,events=build_events()
    manifest={"schema_version":"stream-input-0.1","stream_id":STREAM_ID,"scene_description":"Controlled flat tableware stream with fixed views and changing counts of a recurring plate subclass.","ordering":"manifest","frames":[{"frame_id":f"frame_{i:03d}","index":i,"image_path":f"frames/frame_{i:03d}.png","notes":FRAME_NOTES[i-1]} for i in range(1,11)],"notes":"Пробный development-поток проекта; не финальный evaluation-набор.","metadata":{"source":"deterministic_pillow_generator","generator":"generate_stream.py","purpose":"stream_analysis_probe_development","frame_size":{"width":WIDTH,"height":HEIGHT},"frame_format":"png_rgb","background":{"kind":"fixed_vertical_gradient","top":BACKGROUND_TOP,"bottom":BACKGROUND_BOTTOM},"is_final_dataset":False}}
    annotation={"schema_version":"stream-pilot-annotation-0.1","stream_id":STREAM_ID,"manifest_ref":"manifest.json","annotation_scope":"pilot_development","visual_types":[{"visual_type_id":x.visual_type_id,"description":x.description,"notes":f"Локальный ID потока {STREAM_ID}; geometric_class={x.geometric_class}; предмет={x.name}."} for x in TYPE_SPECS],"expected_element_instances":instances,"frame_comparisons":comparisons,"change_events":events,"allowed_event_types":["persisted","appeared","disappeared","count_changed","position_changed"],"uncertainty":[],"notes":"Выбранный вид каждого подкласса постоянен; цвет и масштаб сами по себе не создают отдельное событие."}
    write_json(ROOT/"manifest.json",manifest); write_json(ROOT/"annotation.json",annotation); validate(manifest,annotation)
    print(f"Generated 10 frames, {len(instances)} instances, and {len(events)} events in {ROOT}; validation passed")


if __name__=="__main__": generate()
