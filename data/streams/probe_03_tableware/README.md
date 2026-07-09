# Probe 03: Tableware

Status: probe development stream for the project version. This is not a final
evaluation set.

## Purpose

This stream contains ten frames of stylized tableware. It fixes a stable
appearance for each object type and includes several separated instances of one
subtype whose count changes over time.

## Objects And Visual Subtypes

| Object | View | Geometric class | `visual_type_id` |
| --- | --- | --- | --- |
| Round plate | Top | `circle` | `circle_subclass_01` |
| Bowl | Top | `circle` | `circle_subclass_02` |
| Oval serving dish | Top | `oval` | `oval_subclass_01` |
| Square plate | Top | `square` | `square_subclass_01` |
| Rectangular tray | Top | `rectangle` | `rectangle_subclass_01` |
| Triangular plate | Top | `triangle` | `triangle_subclass_01` |
| Cup | Side | `undefined` | `undefined_subclass_01` |

Rims, borders, handles, and decorative details are internal object parts and are
not annotated separately. `visual_type_id` values are local to the stream.

## Scenario

The number of round plates follows `1 -> 2 -> 3 -> 2 -> 1`. Plates with
different colors remain instances of `circle_subclass_01`. The square plate,
bowl, tray, and triangular plate appear over the stream. The cup disappears for
two frames and returns with the same subtype. The oval serving dish grows
substantially, the square plate grows slightly, and the tray and bowl move.

## Technical Parameters

- 10 RGB PNG frames at `640 x 480`;
- fixed dark plum vertical gradient;
- cup is always shown from the side, all other items are top-down;
- flat style without shadows, touching, or overlaps;
- bounding boxes are computed from rendered pixels with a `2 px` margin;
- `characteristic_regions` are not used.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\probe_03_tableware\generate_stream.py
```

The generator uses Pillow and does not require external assets.

## Limitations

The stream does not annotate physical identity for individual plates and does
not contain real photographs, perspective transforms, or overlaps.
