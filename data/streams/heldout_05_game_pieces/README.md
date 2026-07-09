# Held-out 05: Game Pieces

Status: independent held-out stream for final evaluation of the frozen
pipeline. This stream is not used for algorithm tuning.

## Purpose

This stream contains 20 controlled frames and 10 local visual types for board
game pieces. Similar types differ by stable outline, border, emblems, holes, or
internal dots; small text is not used.

## Visual Types

| Object | Class | `visual_type_id` |
| --- | --- | --- |
| Round checker | `circle` | `circle_subclass_01` |
| Round token | `circle` | `circle_subclass_02` |
| Square tile | `square` | `square_subclass_01` |
| Emblem tile | `square` | `square_subclass_02` |
| Domino | `rectangle` | `rectangle_subclass_01` |
| Playing card | `rectangle` | `rectangle_subclass_02` |
| Triangular token | `triangle` | `triangle_subclass_01` |
| Pawn | `undefined` | `undefined_subclass_01` |
| Meeple | `undefined` | `undefined_subclass_02` |
| Hex token | `undefined` | `undefined_subclass_03` |

## Scenario

- F01-F05: round token appears, checker moves, square tile is added, domino disappears;
- F06-F10: emblem tile, card, and triangular token appear; token changes color, card rotates;
- F11-F15: pawn grows, meeple appears and moves, hex token appears, checker count increases to three;
- F16-F18: round token disappears, domino and round token return;
- F19-F20: emblem tile changes color and rotation, ordinary tile count decreases, hex token moves.

## Technical Rules

- 20 RGB PNG frames at `640 x 480`, format `png_rgb`;
- fixed dark indigo background with a light gradient;
- bounding boxes are computed from rendered RGB pixels with a `2 px` margin;
- minimum bounding-box gap is `20 px`; touching and overlaps are forbidden;
- annotation is complete for all 19 neighboring-frame comparisons;
- variations are stored in `notes` without extending the schema.

## Reproducibility

```powershell
.\.venv\Scripts\python.exe data\streams\heldout_05_game_pieces\generate_stream.py
```

The generator does not run the pipeline or evaluator.
