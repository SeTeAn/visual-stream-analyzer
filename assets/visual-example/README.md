# Visual example provenance

The input images in this directory are selected RGB frames from the Object Clutter Indoor Dataset (OCID). The result images are adaptations produced by Visual Stream Analyzer by adding masks, contours, and object-group labels.

## Files

| Files | OCID source sequence | Frame | Processing |
|---|---|---|---|
| `full-pipeline-input.png`, `full-pipeline-result.png` | `ARID20/table/bottom/seq02` | 11 | Grounding DINO candidate extraction, SAM2 mask refinement, and DINOv2-based matching/grouping |
| `mask-refinement-input.png`, `mask-refinement-result.png` | `ARID10/table/top/fruits/seq10` | 11 | Grounding DINO candidate extraction and SAM2 mask refinement |

## Attribution

- Dataset: Object Clutter Indoor Dataset (OCID)
- Creators listed by the official record: Jean-Baptiste Nicolas Weibel and Markus Suchi
- Publisher: TU Wien
- Source: <https://researchdata.tuwien.at/records/pcbjd-4wa12>
- DOI: `10.48436/pcbjd-4wa12`
- License: Creative Commons Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>
- Adaptation: selected frames were renamed; result images add Visual Stream Analyzer overlays. The source pixels were not otherwise edited.
