# Held-out 02: Fruits, Vegetables, And Berries

Status: independent held-out stream for final evaluation of the frozen
pipeline. This stream is not used for algorithm tuning.

## Purpose

This stream contains 20 controlled frames and 10 local visual types. It includes
similar round and elongated subtypes, multiple instances, disappearance and
reappearance, and changes in count, position, color, size, and rotation.

## Structure

The directory contains `generate_stream.py`, `manifest.json`, `annotation.json`,
20 files in `frames/`, and 20 files in `annotation_preview/`.

## Visual Types

| Object | Geometric class | `visual_type_id` |
| --- | --- | --- |
| Apple | `circle` | `circle_subclass_01` |
| Orange | `circle` | `circle_subclass_02` |
| Plum | `circle` | `circle_subclass_03` |
| Lemon | `oval` | `oval_subclass_01` |
| Cucumber | `oval` | `oval_subclass_02` |
| Eggplant | `oval` | `oval_subclass_03` |
| Strawberry | `triangle` | `triangle_subclass_01` |
| Carrot | `triangle` | `triangle_subclass_02` |
| Banana | `undefined` | `undefined_subclass_01` |
| Berry cluster | `undefined` | `undefined_subclass_02` |

The cluster with a shared stem is treated as one compound object. Color, size,
and rotation do not create a new visual type.

## Scenario

- F01-F05: strawberry and cucumber appear, apple changes hue, apple count increases;
- F06-F09: orange lightens, lemon moves, plum and a dark-red strawberry variant appear;
- F10-F14: banana disappears, eggplant and carrot appear, eggplant rotates, orange count increases;
- F15-F18: cucumber disappears, lemon grows, cucumber returns, berry cluster appears;
- F19-F20: apples change plausible hues, banana returns, strawberry moves.

## Technical Rules

- 20 RGB PNG frames at `640 x 480`, format `png_rgb`;
- fixed dark plum background with a light gradient;
- bounding boxes are computed from rendered RGB pixels with a `2 px` margin;
- minimum bounding-box gap is `20 px`; touching and overlaps are forbidden;
- intentional moves exceed the normalized `0.1` threshold;
- variation properties are stored in `notes` without extending the annotation schema;
- annotation is complete for all 19 neighboring-frame comparisons.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\heldout_02_fruits_vegetables_berries\generate_stream.py
```

The generator does not run the pipeline or evaluator. Physical identity between
frames is not assumed.
