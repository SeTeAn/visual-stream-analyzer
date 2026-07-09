# Held-out 01: Stationery

Status: independent held-out stream for final evaluation of the frozen project
pipeline. This stream is not used for algorithm tuning.

## Purpose

This stream contains 20 frames of a controlled scene with ten local visual
types. It checks appearance, disappearance, count changes, noticeable motion,
and type return after absence. Color, size, and rotation changes test visual
type identity stability, but they are not separate change events.

## Structure

```text
heldout_01_stationery/
  README.md
  generate_stream.py
  manifest.json
  annotation.json
  frames/
    frame_001.png
    ...
    frame_020.png
  annotation_preview/
    frame_001_annotation.png
    ...
    frame_020_annotation.png
```

## Objects And Visual Types

`visual_type_id` values are local to `heldout_01_stationery`. A full type
reference is the pair `(stream_id, visual_type_id)`.

| Object | Geometric class | `visual_type_id` |
| --- | --- | --- |
| Sticky note | `square` | `square_subclass_01` |
| Notebook | `rectangle` | `rectangle_subclass_01` |
| Eraser | `rectangle` | `rectangle_subclass_02` |
| Pencil | `rectangle` | `rectangle_subclass_03` |
| Marker | `rectangle` | `rectangle_subclass_04` |
| Tape roll | `circle` | `circle_subclass_01` |
| Pencil sharpener | `circle` | `circle_subclass_02` |
| Drafting triangle | `triangle` | `triangle_subclass_01` |
| Scissors | `undefined` | `undefined_subclass_01` |
| Binder clip | `undefined` | `undefined_subclass_02` |

Similar types differ by stable outline, proportions, holes, or internal
details. Color, size, and rotation alone do not create a new type.

## Scenario

| Frame | Main change |
| --- | --- |
| `frame_001` | Initial scene: notebook, pencil, tape roll, and scissors. |
| `frame_002` | First sticky note appears near the left edge. |
| `frame_003` | Pencil moves by 75 px. |
| `frame_004` | Second sticky note is added. |
| `frame_005` | Tape roll disappears. |
| `frame_006` | Pencil sharpener appears. |
| `frame_007` | Pencil rotates around its previous center. |
| `frame_008` | Marker appears. |
| `frame_009` | Marker body color changes. |
| `frame_010` | Eraser appears. |
| `frame_011` | Eraser scales up around its previous center. |
| `frame_012` | Drafting triangle appears. |
| `frame_013` | Scissors move by 75 px. |
| `frame_014` | Binder clip appears. |
| `frame_015` | Pencil sharpener disappears. |
| `frame_016` | Pencil sharpener stays absent; notebook disappears. |
| `frame_017` | Pencil sharpener returns; pencil color changes. |
| `frame_018` | Third sticky note is added; marker moves by 75 px. |
| `frame_019` | Notebook returns. |
| `frame_020` | Tape roll returns; binder clip rotates around its previous center. |

## Event Rules

- type present in both frames: `persisted`;
- transition `N -> M` where both values are non-zero: `persisted` and
  `count_changed`;
- transition `0 -> N`: `appeared`;
- transition `N -> 0`: `disappeared`;
- intentional motion: `persisted` and `position_changed`;
- color-only, size-only, or rotation-only change: `persisted` only.

Annotation is complete for all 19 neighboring-frame comparisons. The absence of
a positive event means a negative label for that event kind.

## Technical Parameters

- 20 frames at `640 x 480`;
- `png_rgb` format;
- fixed dark petrol background with a light vertical gradient;
- flat stylized objects without shadows or complex textures;
- irregular coordinates without a fixed grid;
- objects do not overlap and have at least `20 px` bounding-box gap;
- bounding boxes are computed from rendered RGB pixels and include an external
  `2 px` margin;
- size and rotation changes preserve the anchor center;
- intentional moves are 75 px and exceed the normalized `0.1` frame-width
  threshold;
- additional variation properties are stored in `notes`; the annotation schema
  is not extended.

## Reproducibility

The stream does not use external assets. Frames, JSON files, and previews are
created by one deterministic Pillow generator:

```powershell
.\.venv\Scripts\python.exe data\streams\heldout_01_stationery\generate_stream.py
```

The generator runs structural, geometric, and semantic checks. It does not run
candidate extraction, ML representation, the pipeline, or the evaluator.

## Limitations

- the stream belongs to a controlled domain only;
- real photographs, touching, and overlaps are absent;
- physical identity of instances between frames is not annotated;
- the stream is one of five included held-out dataset parts.
