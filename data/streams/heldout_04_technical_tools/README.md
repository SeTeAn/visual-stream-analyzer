# Held-out 04: Technical Tools And Fasteners

Status: independent held-out stream for final evaluation of the frozen
pipeline. This stream is not used for algorithm tuning.

## Purpose

This stream contains 20 controlled frames and 10 local visual types. The main
stress case is distinguishing round technical parts by outline, holes, and
internal structure, together with several elongated hand tools.

## Visual Types

| Object | Class | `visual_type_id` |
| --- | --- | --- |
| Washer | `circle` | `circle_subclass_01` |
| Gear | `circle` | `circle_subclass_02` |
| Bearing | `circle` | `circle_subclass_03` |
| Square plate | `square` | `square_subclass_01` |
| Rectangular plate | `rectangle` | `rectangle_subclass_01` |
| Rail | `rectangle` | `rectangle_subclass_02` |
| Bracket | `triangle` | `triangle_subclass_01` |
| Hex key | `undefined` | `undefined_subclass_01` |
| Wrench | `undefined` | `undefined_subclass_02` |
| Screwdriver | `undefined` | `undefined_subclass_03` |

## Scenario

- F01-F05: bearing appears, washer is added, wrench moves, gear disappears;
- F06-F10: hex key appears and rotates, rail and bracket appear, plate changes color;
- F11-F15: bearing grows, second bearing, square plate, and screwdriver appear; screwdriver moves, third washer is added;
- F16-F18: hex key disappears, gear and wrench return, bearing count decreases;
- F19-F20: rail changes color and rotation, washer count decreases, square plate moves.

## Technical Rules

- 20 RGB PNG frames at `640 x 480`, format `png_rgb`;
- fixed dark burgundy-graphite background with a light gradient;
- bounding boxes are computed from rendered RGB pixels with a `2 px` margin;
- minimum bounding-box gap is `20 px`; touching and overlaps are forbidden;
- annotation is complete for all 19 neighboring-frame comparisons;
- color, size, and rotation remain in `notes` and do not create a new visual type.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\heldout_04_technical_tools\generate_stream.py
```

The generator does not run the pipeline or evaluator.
