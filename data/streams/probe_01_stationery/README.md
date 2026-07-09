# Probe 01: Stationery

Status: probe development stream for the project version. This is not a final
evaluation set.

## Purpose

This stream contains ten frames of one controlled scene with flat, stylized
stationery objects. It provides an agreed data example for pipeline development
and checks.

The stream exercises the agreed data scope only: visual subtype preservation
under rotation and scale changes, type appearance and disappearance, instance
count changes, and noticeable motion between neighboring frames.

## Structure

```text
probe_01_stationery/
  README.md
  manifest.json
  annotation.json
  generate_stream.py
  frames/
    frame_001.png
    ...
    frame_010.png
```

## Objects And Visual Subtypes

`visual_type_id` values are local to `probe_01_stationery`.

| Object | Geometric class | `visual_type_id` |
| --- | --- | --- |
| Sticky note | `square` | `square_subclass_01` |
| Notebook | `rectangle` | `rectangle_subclass_01` |
| Eraser | `rectangle` | `rectangle_subclass_02` |
| Pencil | `rectangle` | `rectangle_subclass_03` |
| Tape roll | `circle` | `circle_subclass_01` |
| Drafting triangle | `triangle` | `triangle_subclass_01` |
| Scissors | `undefined` | `undefined_subclass_01` |

Internal details, including holes, labels, lines, ring handles, and hinges, are
part of their corresponding objects.

## Scenario

| Frame | Main change |
| --- | --- |
| `frame_001` | Initial state: notebook, pencil, tape roll, and scissors. |
| `frame_002` | Tape roll moves, pencil rotates. |
| `frame_003` | First sticky note appears. |
| `frame_004` | Second sticky note is added, notebook moves. |
| `frame_005` | Regular-size eraser appears. |
| `frame_006` | Eraser becomes much larger, pencil rotates without moving. |
| `frame_007` | Scissors disappear, drafting triangle appears. |
| `frame_008` | One sticky note is removed, drafting triangle moves and rotates. |
| `frame_009` | Tape roll disappears, eraser moves and shrinks. |
| `frame_010` | Notebook moves, rotates, and grows slightly; pencil rotates. |

Rotation-only or scale-only changes do not create separate change events.

## Technical Parameters

- `stream_id`: `probe_01_stationery`;
- 10 frames at `640 x 480`;
- RGB PNG without transparency;
- fixed dark teal-blue background with a soft vertical gradient;
- flat style without shadows;
- objects do not touch or overlap;
- bounding boxes cover the rendered object, outline, and a `2 px` antialiasing
  margin;
- `characteristic_regions` are not used.

## Reproducibility

Frames, `manifest.json`, and `annotation.json` are produced by one deterministic
generator with no external assets:

```powershell
.\.venv\Scripts\python.exe data\streams\probe_01_stationery\generate_stream.py
```

The generator uses Pillow, draws at a larger resolution for antialiasing, then
saves the final frames and measured bounding boxes.

## Limitations

- the stream contains only controlled flat images;
- the background and style are fixed within the stream;
- dense overlaps are absent;
- physical identity of individual instances is not annotated;
- the stream is not intended for final system-quality conclusions.
