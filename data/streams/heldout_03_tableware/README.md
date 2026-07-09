# Held-out 03: Tableware

Status: independent held-out stream for final evaluation of the frozen
pipeline. This stream is not used for algorithm tuning.

## Purpose

This stream contains 20 controlled frames and 10 local visual types for
tableware and serving items. It checks similar round types, plate and spoon
count changes, disappearance and reappearance, position, color, size, and
rotation.

## Visual Types

| Object | Class | `visual_type_id` |
| --- | --- | --- |
| Round plate | `circle` | `circle_subclass_01` |
| Bowl | `circle` | `circle_subclass_02` |
| Saucer | `circle` | `circle_subclass_03` |
| Oval serving dish | `oval` | `oval_subclass_01` |
| Square plate | `square` | `square_subclass_01` |
| Tray | `rectangle` | `rectangle_subclass_01` |
| Folded napkin | `rectangle` | `rectangle_subclass_02` |
| Triangular plate | `triangle` | `triangle_subclass_01` |
| Cup | `undefined` | `undefined_subclass_01` |
| Spoon | `undefined` | `undefined_subclass_02` |

The napkin is compact, does not sit under other objects, and is annotated as a
separate object.

## Scenario

- F01-F05: saucer appears, plate moves, spoon is added, bowl disappears;
- F06-F10: oval dish, square plate, and tray appear; cup changes color, saucer grows;
- F11-F14: tray moves, napkin appears, then rotates and shrinks, triangular plate appears;
- F15-F18: cup disappears and returns, a second plate is added, bowl returns;
- F19-F20: saucer disappears and returns, spoon count decreases.

## Technical Rules

- 20 RGB PNG frames at `640 x 480`, format `png_rgb`;
- fixed dark blue background with a light gradient;
- bounding boxes are computed from rendered RGB pixels with a `2 px` margin;
- minimum bounding-box gap is `20 px`; touching and overlaps are forbidden;
- annotation is complete for all 19 neighboring-frame comparisons;
- variations are stored in `notes` and do not extend the schema.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\heldout_03_tableware\generate_stream.py
```

The generator creates frames, manifest, annotation, and previews, but does not
run the pipeline or evaluator.
