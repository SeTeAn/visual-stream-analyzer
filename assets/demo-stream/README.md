# Complete OCID analysis result

This directory contains the complete compact Visual Stream Analyzer output for one 11-frame OCID sequence. The corresponding unchanged RGB inputs are stored in [`data/streams/ocid_arid10_table_top_fruits_seq10/frames`](../../data/streams/ocid_arid10_table_top_fruits_seq10/frames).

## Sequence

- OCID source sequence: `ARID10/table/top/fruits/seq10`
- Ordered frames: 11
- Input resolution: 640 x 480 pixels
- Input modifications: none; the RGB files were only renamed to stable sequential names
- Structured result: `ocid_arid10_table_top_fruits_seq10/result.json`
- Final binary masks: `ocid_arid10_table_top_fruits_seq10/masks`
- Result images: `ocid_arid10_table_top_fruits_seq10/overlays`

The result images add predicted masks, contours, bounding boxes, and `type_...` labels. A `type_...` label represents a predicted visual group rather than a semantic class. Frame-local `Pnn` identifiers remain in `result.json` and mask filenames; confidence values are not displayed.

## Processing

The result was produced from the first frame to the last by the public `python -m stream_analysis analyze` command with the current Grounding DINO, SAM2, and DINOv2 profile.

## Attribution

- Dataset: Object Clutter Indoor Dataset (OCID)
- Creators listed by the official record: Jean-Baptiste Nicolas Weibel and Markus Suchi
- Publisher: TU Wien
- Source: <https://researchdata.tuwien.at/records/pcbjd-4wa12>
- DOI: `10.48436/pcbjd-4wa12`
- License: Creative Commons Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>
- Adaptation: result images add Visual Stream Analyzer overlays to the source RGB frames.
