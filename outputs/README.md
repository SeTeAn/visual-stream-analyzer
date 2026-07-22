# Analysis outputs

The command-line interface writes analysis runs under an output root selected by the user:

```text
<output-root>/
  runs/
    <run_id>/
      run_manifest.json
      candidate_manifest.json
      stream_analysis.json
      report.txt
      runtime_status.json
      overlays/
      diagnostics/
```

`stream_analysis.json` contains the structured matching, grouping, and event records. `overlays/` contains visualizations, and `diagnostics/` contains optional technical details requested through the diagnostic level.

Evaluation is a separate operation and writes its own artifacts:

```text
<output-root>/
  evaluations/
    <evaluation_id>/
      evaluation_manifest.json
      evaluation_report.json
      error_ledger.json
      summary.txt
```

Run IDs are immutable: the tools refuse to overwrite an existing completed run. For normal local work, use `.local_outputs`, which is ignored by Git. The source-controlled `outputs` directory is reserved for examples deliberately selected for publication.
