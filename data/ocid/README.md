# OCID local dataset

This directory is the repository-local entry point for the Object Clutter
Indoor Dataset (OCID). The extracted dataset is intentionally not committed to
Git.

## Provenance

- Source: <https://researchdata.tuwien.at/records/pcbjd-4wa12>
- DOI: `10.48436/pcbjd-4wa12`
- Archive: `OCID-dataset.tar.gz`
- Archive size: `6,822,716,319` bytes
- MD5: `aa0eff01f75104c40e4243e6f892a757`
- License: CC BY 4.0

## Local layout

```text
data/ocid/
  README.md
  raw/
    OCID-dataset/
      ARID10/
      ARID20/
      YCB10/
```

For the initial dataset gate, `raw/` contains RGB frames, instance-label masks,
and the small dataset metadata files. Depth maps and PCD point clouds remain in
the verified source archive because the current Visual Stream Analyzer pipeline
is 2D and does not consume them.

The selective extraction was verified to contain 178 sequences, 2,390 RGB
frames, and 2,390 label masks. Every RGB filename has a matching label filename
in the same sequence.

To reproduce the selective extraction from a verified archive:

```powershell
New-Item -ItemType Directory -Force data\ocid\raw | Out-Null
tar -xzf <path-to-OCID-dataset.tar.gz> -C data\ocid\raw `
  --exclude '*/depth/*' `
  --exclude '*/pcd/*'
```

## Prepare analysis streams

Generate the three local RGB-only streams selected by the dataset gate:

```powershell
python tools\prepare_ocid_streams.py `
  --ocid-root data\ocid\raw\OCID-dataset `
  --output-root data\ocid\derived\streams
```

The generated `derived/` directory is ignored by Git. Each stream contains only
copied RGB frames and a `stream-input-0.1` manifest. OCID label masks remain in
`raw/` and are not referenced by analyze input.

## Prepare candidate-only evaluation annotations

After an analysis run has been saved, convert the corresponding OCID instance
masks into a separate bbox annotation for the existing evaluator:

```powershell
python tools\prepare_ocid_candidate_annotations.py `
  --ocid-root data\ocid\raw\OCID-dataset `
  --stream-directory data\ocid\derived\streams\ocid_arid10_table_top_box_seq05 `
  --output .local_outputs\ocid_annotations\ocid_arid10_table_top_box_seq05\annotation.json
```

The generated file deliberately stays outside the analysis stream. It provides
tight bounding boxes derived from instance masks and supports candidate-level
evaluation only. Its physical-instance proxy IDs are not visual-type ground
truth, so grouping and event metrics from it must not be used as OCID claims.

## Experimental first-frame background

The candidate extractor also accepts the experimental configuration fragment:

```json
{
  "candidate_extraction": {
    "background_strategy": "first_frame"
  }
}
```

This mode uses the first RGB frame as a fixed background reference. It does not
read labels, and it is not the default strategy. It is suitable only when the
first frame is known to show the empty, stable workspace; connected foreground
objects can still merge into one candidate.
