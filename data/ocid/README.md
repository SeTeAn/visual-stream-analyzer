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
  benchmark/
    ocid_candidate_benchmark_v1.json
    ocid_candidate_component_decisions_v1.json
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

## Review and build the frozen Gate B benchmark

The source-controlled benchmark specification freezes the approved grouped
split: 5 development scene groups (10 camera streams, 148 frames) and 3
held-out scene groups (6 camera streams, 82 frames). The annotation policy
uses reviewed 4-connected mask components; raw mask-union boxes must not be
used for benchmark metrics.

The original local `ocid_component_review_v1` pack is invalid debugging
evidence and is rejected by the coordinator. Build the hardened v2
evaluation-only blind/adjudication review pack:

```powershell
python tools\prepare_ocid_component_review_pack.py `
  --ocid-root data\ocid\raw\OCID-dataset `
  --structural-audit .local_outputs\gate_a\ocid_structural_audit_v1\structural_audit.json `
  --spec data\ocid\benchmark\ocid_candidate_benchmark_v1.json `
  --output-root .local_outputs\gate_b\ocid_component_review_v2
```

Prepare two blind assignment passes without exposing the automatic proposal:

```powershell
python tools\coordinate_ocid_component_reviews.py prepare `
  --review-pack .local_outputs\gate_b\ocid_component_review_v2 `
  --output-root .local_outputs\gate_b\ocid_component_review_assignments_v2
```

After two distinct reviewers complete all eight shard files, merge agreements
and prepare the adjudication queue:

```powershell
python tools\coordinate_ocid_component_reviews.py merge-primary `
  --review-pack .local_outputs\gate_b\ocid_component_review_v2 `
  --assignments-root .local_outputs\gate_b\ocid_component_review_completed_v2 `
  --output-root .local_outputs\gate_b\ocid_component_review_primary_merge_v2
```

After every disagreement is independently adjudicated and any remaining
author decisions are recorded in a completed adjudication file, create the
final decision ledger and author-preview queue:

```powershell
python tools\coordinate_ocid_component_reviews.py finalize `
  --review-pack .local_outputs\gate_b\ocid_component_review_v2 `
  --primary-merge .local_outputs\gate_b\ocid_component_review_primary_merge_v2\primary_merge.json `
  --adjudication .local_outputs\gate_b\ocid_component_adjudication_author_resolved_v1\adjudication.json `
  --output-root .local_outputs\gate_b\ocid_component_review_final_author_v1
```

The frozen ledger is stored as
`data/ocid/benchmark/ocid_candidate_component_decisions_v1.json`; its SHA-256
is pinned in the source-controlled spec. Build all RGB analysis inputs and
separate evaluation/review artifacts from that exact ledger:

```powershell
python tools\prepare_ocid_gate_b_benchmark.py `
  --ocid-root data\ocid\raw\OCID-dataset `
  --structural-audit .local_outputs\gate_a\ocid_structural_audit_v1\structural_audit.json `
  --spec data\ocid\benchmark\ocid_candidate_benchmark_v1.json `
  --component-decisions data\ocid\benchmark\ocid_candidate_component_decisions_v1.json `
  --output-root data\ocid\derived\ocid_candidate_benchmark_v1_reviewed
```

Every stage is atomic and local-only. Final `analysis_streams/` contains copied
RGB frames plus annotation-free `stream-input-0.1` manifests. Instance masks,
candidate annotations, overlays, contact sheets and review ledgers stay outside
that directory. The final builder validates this boundary recursively and
refuses missing, stale, incomplete or unhashed component decisions.

Held-out annotations may be prepared and visually checked at Gate B, but the
frozen policy keeps held-out model predictions, prediction overlays, metrics
and parameter selection locked until the author explicitly opens the final
run.

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
