# Probe 04: Technical Tools And Fasteners

Status: probe development stream for the project version. This is not a final
evaluation set.

## Purpose

This stream contains ten frames of flat, stylized technical objects with holes,
teeth, cutouts, and varied silhouettes. The generic wrench wording is split into
an L-shaped hex key and a double-ended wrench.

## Objects And Visual Subtypes

| Object | Geometric class | `visual_type_id` |
| --- | --- | --- |
| Washer | `circle` | `circle_subclass_01` |
| Gear | `circle` | `circle_subclass_02` |
| Square mounting plate | `square` | `square_subclass_01` |
| Rectangular mounting plate | `rectangle` | `rectangle_subclass_01` |
| Triangular bracket | `triangle` | `triangle_subclass_01` |
| L-shaped hex key | `undefined` | `undefined_subclass_01` |
| Double-ended wrench | `undefined` | `undefined_subclass_02` |

Holes, teeth, and cutouts are object parts and are not annotated separately.
`visual_type_id` values are local to the stream.

## Scenario

The gear, bracket, and hex key appear. Washer count changes as `1 -> 2 -> 1`.
The double-ended wrench disappears for two frames and returns with the same
`undefined_subclass_02`; the square plate disappears. The gear grows
substantially, the square plate grows slightly, and the rectangular plate
shrinks substantially. Several subtypes move or rotate.

## Technical Parameters

- 10 RGB PNG frames at `640 x 480`;
- fixed light muted blue-green vertical gradient;
- flat style without shadows, highlights, touching, or overlaps;
- bounding boxes are computed from rendered pixels with a `2 px` margin;
- `characteristic_regions` are not used.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\probe_04_technical_tools\generate_stream.py
```

The generator uses Pillow and does not require external assets.

## Limitations

The stream does not model 3D perspective, real metal, dense overlaps, or
physical identity of individual instances.
