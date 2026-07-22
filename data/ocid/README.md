# Object Clutter Indoor Dataset (OCID)

This directory contains configuration and preparation utilities for using OCID with Visual Stream Analyzer. The full dataset and generated derivatives are local assets and are not committed to Git. One compact attributed RGB example stream is included under `data/streams/ocid_arid10_table_top_fruits_seq10`.

## Source and License

- Dataset: Object Clutter Indoor Dataset (OCID)
- Official record: <https://researchdata.tuwien.at/records/pcbjd-4wa12>
- DOI: `10.48436/pcbjd-4wa12`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

When redistributing selected OCID images, retain the dataset attribution and license notice. The repository does not claim ownership of OCID materials.

## Local Layout

```text
data/ocid/
  README.md
  benchmark/
    ocid_candidate_benchmark_v1.json
    ocid_pipeline_protocol_v1.json
  raw/
    OCID-dataset/
      ARID10/
      ARID20/
      YCB10/
  derived/
    ocid_source_label_inventory_v1.json
```

`raw/` and `derived/` are ignored by Git. RGB frames are analysis inputs; instance labels are kept separate and are opened only by evaluation utilities.

## Prepare Compact RGB Streams

After downloading and extracting OCID from the official source, prepare the configured local streams:

```powershell
$env:PYTHONPATH = "src"

.\.venv\Scripts\python.exe -B -m tools.prepare_ocid_streams `
  --ocid-root data\ocid\raw\OCID-dataset `
  --output-root data\ocid\derived\streams
```

Each generated analysis stream contains copied RGB frames and an annotation-free manifest.

Create the local source-label inventory used by evaluation checks:

```powershell
.\.venv\Scripts\python.exe -B -m tools.build_ocid_label_inventory
```

## Prepare Candidate Annotations

Candidate annotations can be generated separately from an OCID instance-label sequence:

```powershell
.\.venv\Scripts\python.exe -B -m tools.prepare_ocid_candidate_annotations `
  --ocid-root data\ocid\raw\OCID-dataset `
  --stream-directory data\ocid\derived\streams\ocid_arid10_table_top_box_seq05 `
  --output .local_outputs\ocid_annotations\ocid_arid10_table_top_box_seq05\annotation.json
```

The generated annotation file stays outside the RGB analysis stream.

## Configured Real-Image Pipeline

`ocid_pipeline_protocol_v1.json` records the local asset paths and configuration for the Grounding DINO + SAM2 + DINOv2 pipeline. Run its read-only setup check with:

```powershell
.\.venv\Scripts\python.exe -B -m tools.run_ocid_pipeline check `
  --output .local_outputs\ocid_pipeline\check.json
```

Model weights, full OCID files, and generated OCID inventories are intentionally excluded from the repository.
