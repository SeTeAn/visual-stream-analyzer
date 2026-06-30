from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageDraw, ImageOps


STREAM_ID = "probe_04_technical_tools"
WIDTH, HEIGHT, AA, BBOX_PADDING = 640, 480, 4, 2
ROOT = Path(__file__).resolve().parent
FRAMES_DIR = ROOT / "frames"
BACKGROUND_TOP = "#91B9BD"
BACKGROUND_BOTTOM = "#7FA9AE"


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


TYPE_SPECS = [
    TypeSpec("washer", "шайба", "circle", "circle_subclass_01", "Круглая шайба с центральным отверстием."),
    TypeSpec("gear", "шестеренка", "circle", "circle_subclass_02", "Круглая шестеренка с зубцами и центральным отверстием."),
    TypeSpec("square_plate", "квадратная монтажная пластина", "square", "square_subclass_01", "Квадратная пластина с четырьмя крепежными отверстиями."),
    TypeSpec("rect_plate", "прямоугольная монтажная пластина", "rectangle", "rectangle_subclass_01", "Вытянутая пластина с двумя крепежными отверстиями."),
    TypeSpec("triangle_bracket", "треугольный кронштейн", "triangle", "triangle_subclass_01", "Треугольный кронштейн с внутренним вырезом и отверстиями."),
    TypeSpec("allen_key", "Г-образный шестигранный ключ", "undefined", "undefined_subclass_01", "Тонкий Г-образный шестигранный ключ."),
    TypeSpec("wrench", "двусторонний гаечный ключ", "undefined", "undefined_subclass_02", "Гаечный ключ с вытянутой рукояткой и открытыми рабочими концами."),
]
TYPE_BY_KIND = {item.kind: item for item in TYPE_SPECS}
TYPE_ORDER = {item.visual_type_id: index for index, item in enumerate(TYPE_SPECS)}


SCENES = [
    [Placement("washer", 1, (90, 100)), Placement("square_plate", 1, (300, 100)), Placement("rect_plate", 1, (300, 250)), Placement("wrench", 1, (500, 410))],
    [Placement("washer", 1, (90, 240)), Placement("square_plate", 1, (300, 100)), Placement("rect_plate", 1, (300, 250)), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("gear", 1, (510, 100)), Placement("square_plate", 1, (300, 100)), Placement("rect_plate", 1, (300, 250)), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("washer", 2, (90, 100)), Placement("gear", 1, (510, 100)), Placement("square_plate", 1, (300, 100)), Placement("rect_plate", 1, (300, 250)), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("washer", 2, (90, 100)), Placement("gear", 1, (510, 100)), Placement("square_plate", 1, (300, 100), scale=1.15), Placement("rect_plate", 1, (300, 250)), Placement("triangle_bracket", 1, (510, 270)), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("washer", 2, (90, 100)), Placement("gear", 1, (510, 100)), Placement("square_plate", 1, (300, 100), scale=1.15), Placement("rect_plate", 1, (300, 390)), Placement("triangle_bracket", 1, (510, 270)), Placement("allen_key", 1, (90, 390)), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("washer", 2, (90, 100)), Placement("gear", 1, (510, 100), rotation=18, scale=1.6), Placement("square_plate", 1, (300, 100), scale=1.15), Placement("rect_plate", 1, (300, 390)), Placement("triangle_bracket", 1, (510, 270)), Placement("allen_key", 1, (90, 390))],
    [Placement("washer", 1, (90, 240)), Placement("gear", 1, (510, 100), rotation=18, scale=1.6), Placement("square_plate", 1, (300, 100), scale=1.15), Placement("rect_plate", 1, (300, 390)), Placement("triangle_bracket", 1, (510, 270)), Placement("allen_key", 1, (300, 250), rotation=30)],
    [Placement("washer", 1, (90, 240)), Placement("gear", 1, (510, 100), rotation=18, scale=1.6), Placement("rect_plate", 1, (300, 390)), Placement("triangle_bracket", 1, (510, 270)), Placement("allen_key", 1, (300, 250), rotation=30), Placement("wrench", 1, (500, 410), rotation=25)],
    [Placement("washer", 1, (90, 240)), Placement("gear", 1, (510, 100), rotation=18, scale=1.6), Placement("rect_plate", 1, (300, 390), scale=0.65), Placement("triangle_bracket", 1, (90, 100)), Placement("allen_key", 1, (300, 250), rotation=30), Placement("wrench", 1, (500, 410), rotation=25)],
]

FRAME_NOTES = [
    "Начальная сцена: шайба, квадратная и прямоугольная пластины, гаечный ключ.",
    "Шайба перемещена; гаечный ключ повернут.",
    "Появилась шестеренка.",
    "Добавлена вторая шайба того же подкласса.",
    "Появился треугольный кронштейн; квадратная пластина немного увеличена.",
    "Появился шестигранный ключ; прямоугольная пластина перемещена.",
    "Гаечный ключ исчез; шестеренка значительно увеличена и повернута.",
    "Одна шайба удалена; шестигранный ключ перемещен и повернут.",
    "Гаечный ключ вернулся после двух кадров отсутствия; квадратная пластина исчезла.",
    "Кронштейн перемещен; прямоугольная пластина значительно уменьшена.",
]

POSITION_CHANGES = {2: {"circle_subclass_01"}, 6: {"rectangle_subclass_01"}, 8: {"undefined_subclass_01"}, 10: {"triangle_subclass_01"}}


def s(value: float) -> int: return round(value * AA)
def box(v: tuple[float,float,float,float]) -> tuple[int,int,int,int]: return tuple(s(x) for x in v)
def points(v: list[tuple[float,float]]) -> list[tuple[int,int]]: return [(s(x),s(y)) for x,y in v]
def asset(w: int,h: int) -> tuple[Image.Image,ImageDraw.ImageDraw]:
    image=Image.new("RGBA",(w*AA,h*AA),(0,0,0,0)); return image,ImageDraw.Draw(image)


def draw_washer() -> Image.Image:
    image,draw=asset(82,82); draw.ellipse(box((3,3,79,79)),fill="#718A99",outline="#344B57",width=s(3)); draw.ellipse(box((25,25,57,57)),fill=(0,0,0,0),outline="#344B57",width=s(3)); draw.arc(box((11,11,71,71)),195,305,fill="#C3D1D6",width=s(3)); return image


def draw_gear() -> Image.Image:
    image,draw=asset(106,106); cx=cy=53; pts=[]
    for i in range(32):
        angle=math.radians(i*360/32-90); radius=49 if i%2==0 else 40; pts.append((cx+radius*math.cos(angle),cy+radius*math.sin(angle)))
    draw.polygon(points(pts),fill="#B86B47",outline="#653B31"); draw.ellipse(box((31,31,75,75)),fill="#D38A62",outline="#653B31",width=s(3)); draw.ellipse(box((43,43,63,63)),fill=(0,0,0,0),outline="#653B31",width=s(3)); return image


def draw_square_plate() -> Image.Image:
    image,draw=asset(102,102); draw.rounded_rectangle(box((3,3,99,99)),radius=s(7),fill="#367E78",outline="#1F4D4B",width=s(3)); draw.rectangle(box((17,17,85,85)),outline="#76B9AE",width=s(3))
    for x,y in [(18,18),(84,18),(18,84),(84,84)]: draw.ellipse(box((x-6,y-6,x+6,y+6)),fill=(0,0,0,0),outline="#1F4D4B",width=s(2))
    return image


def draw_rect_plate() -> Image.Image:
    image,draw=asset(154,78); draw.rounded_rectangle(box((3,3,151,75)),radius=s(8),fill="#8F4E5C",outline="#512E38",width=s(3)); draw.rounded_rectangle(box((18,16,136,62)),radius=s(5),outline="#C9828E",width=s(3))
    for x in (31,123): draw.ellipse(box((x-8,31,x+8,47)),fill=(0,0,0,0),outline="#512E38",width=s(2))
    return image


def draw_triangle_bracket() -> Image.Image:
    image,draw=asset(118,108); outer=[(5,103),(5,5),(113,103)]; inner=[(29,84),(29,39),(80,84)]
    draw.polygon(points(outer),fill="#D2A642",outline="#70591F"); draw.line(points(outer+[outer[0]]),fill="#70591F",width=s(3)); draw.polygon(points(inner),fill=(0,0,0,0)); draw.line(points(inner+[inner[0]]),fill="#70591F",width=s(3))
    for x,y in [(17,88),(17,20),(94,91)]: draw.ellipse(box((x-5,y-5,x+5,y+5)),fill="#F2D982",outline="#70591F",width=s(2))
    return image


def draw_allen_key() -> Image.Image:
    image,draw=asset(126,94); draw.line(points([(20,12),(20,72),(108,72)]),fill="#4E4776",width=s(17),joint="curve"); draw.line(points([(20,12),(20,72),(108,72)]),fill="#877FAD",width=s(7),joint="curve"); draw.ellipse(box((11,3,29,21)),fill="#4E4776"); draw.ellipse(box((99,63,117,81)),fill="#4E4776"); return image


def draw_wrench() -> Image.Image:
    image,draw=asset(170,60); shape=[(3,8),(25,17),(43,20),(127,20),(145,7),(166,13),(153,30),(166,47),(145,53),(127,40),(43,40),(25,51),(3,45),(17,30)]
    draw.polygon(points(shape),fill="#3F4A50",outline="#222B30"); draw.line(points([(48,29),(122,29)]),fill="#82939B",width=s(5)); draw.polygon(points([(3,8),(25,17),(17,30)]),fill=(0,0,0,0)); draw.polygon(points([(166,13),(153,30),(166,47)]),fill=(0,0,0,0)); return image


DRAWERS: dict[str,Callable[[],Image.Image]]={"washer":draw_washer,"gear":draw_gear,"square_plate":draw_square_plate,"rect_plate":draw_rect_plate,"triangle_bracket":draw_triangle_bracket,"allen_key":draw_allen_key,"wrench":draw_wrench}


def background() -> Image.Image:
    top=tuple(bytes.fromhex(BACKGROUND_TOP[1:])); bottom=tuple(bytes.fromhex(BACKGROUND_BOTTOM[1:])); image=Image.new("RGB",(WIDTH*AA,HEIGHT*AA)); draw=ImageDraw.Draw(image)
    for y in range(HEIGHT*AA):
        q=y/(HEIGHT*AA-1); draw.line((0,y,WIDTH*AA,y),fill=tuple(round(top[i]*(1-q)+bottom[i]*q) for i in range(3)))
    return image.convert("RGBA")


def transformed(item: Placement) -> Image.Image:
    image=DRAWERS[item.kind]()
    if item.scale!=1: image=image.resize((round(image.width*item.scale),round(image.height*item.scale)),Image.Resampling.LANCZOS)
    if item.rotation: image=image.rotate(item.rotation,resample=Image.Resampling.BICUBIC,expand=True)
    return image.crop(image.getchannel("A").getbbox())


def too_close(a: dict[str,int],b: dict[str,int],gap: int=12) -> bool:
    return not (a["x"]+a["width"]+gap<=b["x"]-gap or b["x"]+b["width"]+gap<=a["x"]-gap or a["y"]+a["height"]+gap<=b["y"]-gap or b["y"]+b["height"]+gap<=a["y"]-gap)


def render_frame(index: int,scene: list[Placement]) -> list[dict[str,object]]:
    base_frame=background(); base_output=base_frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS); frame=base_frame.copy(); instances=[]; bboxes=[]
    for item in scene:
        image=transformed(item); left=round(item.center[0]*AA-image.width/2); top=round(item.center[1]*AA-image.height/2)
        if left<0 or top<0 or left+image.width>frame.width or top+image.height>frame.height: raise ValueError(f"{item.kind} leaves frame {index}")
        object_frame=base_frame.copy(); object_frame.alpha_composite(image,(left,top)); object_output=object_frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS); rendered_bbox=ImageChops.difference(object_output,base_output).convert("L").getbbox()
        if rendered_bbox is None: raise ValueError(f"{item.kind} produced no rendered pixels in frame {index}")
        x=max(0,rendered_bbox[0]-BBOX_PADDING); y=max(0,rendered_bbox[1]-BBOX_PADDING); right=min(WIDTH,rendered_bbox[2]+BBOX_PADDING); bottom=min(HEIGHT,rendered_bbox[3]+BBOX_PADDING); bbox={"x":x,"y":y,"width":right-x,"height":bottom-y}
        if any(too_close(old,bbox) for old in bboxes): raise ValueError(f"Objects are too close in frame {index}: {item.kind} {bbox}")
        bboxes.append(bbox); frame.alpha_composite(image,(left,top)); spec=TYPE_BY_KIND[item.kind]
        instances.append({"instance_id":f"inst_{item.kind}_{item.slot:02d}_f{index:03d}","frame_id":f"frame_{index:03d}","visual_type_id":spec.visual_type_id,"bbox":bbox,"characteristic_regions":[],"uncertainty":"","notes":f"Объект: {spec.name}; geometric_class={spec.geometric_class}; slot={item.slot}; rotation_deg={item.rotation:g}; scale={item.scale:g}."})
    frame.convert("RGB").resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS).save(FRAMES_DIR/f"frame_{index:03d}.png",optimize=True); return instances


def counts(scene: list[Placement]) -> Counter[str]: return Counter(TYPE_BY_KIND[item.kind].visual_type_id for item in scene)
def average_center(scene: list[Placement],type_id: str) -> tuple[float,float]:
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
    manifest={"schema_version":"stream-input-0.1","stream_id":STREAM_ID,"scene_description":"Controlled flat technical-tools stream with holes, teeth, cutouts, count changes, rotation, and scale changes.","ordering":"manifest","frames":[{"frame_id":f"frame_{i:03d}","index":i,"image_path":f"frames/frame_{i:03d}.png","notes":FRAME_NOTES[i-1]} for i in range(1,11)],"notes":"Пробный development-поток проекта; не финальный evaluation-набор.","metadata":{"source":"deterministic_pillow_generator","generator":"generate_stream.py","purpose":"stream_analysis_probe_development","frame_size":{"width":WIDTH,"height":HEIGHT},"frame_format":"png_rgb","background":{"kind":"fixed_vertical_gradient","top":BACKGROUND_TOP,"bottom":BACKGROUND_BOTTOM},"is_final_dataset":False}}
    annotation={"schema_version":"stream-pilot-annotation-0.1","stream_id":STREAM_ID,"manifest_ref":"manifest.json","annotation_scope":"pilot_development","visual_types":[{"visual_type_id":x.visual_type_id,"description":x.description,"notes":f"Локальный ID потока {STREAM_ID}; geometric_class={x.geometric_class}; предмет={x.name}."} for x in TYPE_SPECS],"expected_element_instances":instances,"frame_comparisons":comparisons,"change_events":events,"allowed_event_types":["persisted","appeared","disappeared","count_changed","position_changed"],"uncertainty":[],"notes":"Отверстия, зубцы и вырезы являются частями объектов; поворот и масштаб сами по себе не создают отдельное событие."}
    write_json(ROOT/"manifest.json",manifest); write_json(ROOT/"annotation.json",annotation); validate(manifest,annotation)
    print(f"Generated 10 frames, {len(instances)} instances, and {len(events)} events in {ROOT}; validation passed")


if __name__=="__main__": generate()
