# Selected visual-result provenance

The `current-results` directory contains selected input/result pairs from the current Visual Stream Analyzer pipeline. Every input is an unchanged RGB frame from the Object Clutter Indoor Dataset (OCID). Each result is an adaptation that adds predicted masks, contours, bounding boxes, and `type_...` visual-group labels.

## Files

| Files | OCID source sequence | Frame |
|---|---|---:|
| `current-results/ocid_arid20_floor_top_seq12-frame_0015-input.png`, `current-results/ocid_arid20_floor_top_seq12-frame_0015-result.png` | `ARID20/floor/top/seq12` | 15 |
| `current-results/ocid_arid20_table_bottom_seq01-frame_0013-input.png`, `current-results/ocid_arid20_table_bottom_seq01-frame_0013-result.png` | `ARID20/table/bottom/seq01` | 13 |
| `current-results/ocid_ycb10_table_top_mixed_seq21-frame_0009-input.png`, `current-results/ocid_ycb10_table_top_mixed_seq21-frame_0009-result.png` | `YCB10/table/top/mixed/seq21` | 9 |

## Processing

The result images were produced by the configured Grounding DINO, SAM2, and DINOv2 pipeline. The displayed masks use the current candidate selection, temporal support, and aggregate-mask resolution logic. A `type_...` label is a predicted visual group, not a semantic class; confidence values and internal candidate identifiers are not displayed.

## Attribution

- Dataset: Object Clutter Indoor Dataset (OCID)
- Creators listed by the official record: Jean-Baptiste Nicolas Weibel and Markus Suchi
- Publisher: TU Wien
- Source: <https://researchdata.tuwien.at/records/pcbjd-4wa12>
- DOI: `10.48436/pcbjd-4wa12`
- License: Creative Commons Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>
- Adaptation: selected input frames were renamed; result images add Visual Stream Analyzer overlays. The input pixels were not otherwise edited.
