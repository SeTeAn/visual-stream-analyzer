# Probe 02: Fruits, Vegetables, And Berries

Status: probe development stream for the project version. This is not a final
evaluation set.

## Purpose

This stream contains ten frames with whole stylized produce items. It checks
visual subtype preservation under hue, rotation, and scale changes, as well as
subtype appearance, disappearance, and noticeable motion.

## Objects And Visual Subtypes

`visual_type_id` values are local to `probe_02_fruits_vegetables_berries`.

| Object | Geometric class | `visual_type_id` |
| --- | --- | --- |
| Apple | `circle` | `circle_subclass_01` |
| Orange | `circle` | `circle_subclass_02` |
| Lemon | `oval` | `oval_subclass_01` |
| Cucumber | `oval` | `oval_subclass_02` |
| Strawberry | `triangle` | `triangle_subclass_01` |
| Carrot | `triangle` | `triangle_subclass_02` |
| Linked berry cluster | `undefined` | `undefined_subclass_01` |

Stems, leaves, seeds, peel marks, and the shared berry-cluster branch are part
of their corresponding objects and are not annotated separately. All produce is
shown whole, without cut surfaces.

## Scenario

The apple is shown as dark green, green-red, and light green. The orange changes
from dark orange to light orange, the strawberry appears in red and blue, and
the cucumber changes green shade. These changes do not create a new subtype or
a separate change event.

The stream also includes orange, cucumber, and carrot appearances; berry-cluster
and apple disappearances; and a berry-cluster return after two missing frames
with the same `undefined_subclass_01`. The lemon and orange change size, while
the strawberry and carrot move noticeably.

## Technical Parameters

- 10 RGB PNG frames at `640 x 480`;
- fixed dark blue-indigo vertical gradient;
- flat style without shadows;
- objects do not touch or overlap;
- bounding boxes are computed from rendered pixels with a `2 px` margin;
- `characteristic_regions` are not used.

## Reproducibility

Frames and JSON files are produced by a deterministic Pillow generator with no
external assets:

```powershell
.\.venv\Scripts\python.exe data\streams\probe_02_fruits_vegetables_berries\generate_stream.py
```

## Limitations

The stream does not contain real photographs, cut produce, overlaps, shape
changes, or physical identity annotations for individual instances.
