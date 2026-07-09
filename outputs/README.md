# Run outputs

Primary analyze runs are created under:

```text
outputs/
  runs/
    <run_id>/
      run_manifest.json
      candidate_manifest.json
      stream_analysis.json
      report.txt
      runtime_status.json
      overlays/
        <frame_id>.png
      diagnostics/
        pair_scores.json   # only when diagnostic_level=standard
```

Schema identifiers:

- `stream_analysis.run_manifest.v1`;
- `stream_analysis.candidate_manifest.v1`;
- `stream_analysis.primary_stream_result.v1`;
- `stream_analysis.runtime_status.v1`;
- `stream_analysis.diagnostic_pair_scores.v1` for optional pair-score
  diagnostics.

`candidate_manifest.json` is the frozen compact candidate snapshot and its
SHA-256. Bitmap masks remain stage-local. `stream_analysis.json` contains
compact representation summaries and structured matching/grouping/event facts,
but no embedding vectors, full handcrafted features, score matrices,
annotations, GT mappings or evaluation metrics.

`run_manifest.json` contains run provenance and the complete ordered typed
warning/error ledger. Every warning or error ID referenced by primary records
can therefore be resolved without loading stage-local working payloads.

Run directories are immutable: a repeated `run_id` is rejected. Files are
first written to a temporary sibling directory and published by rename, so a
write failure does not leave a partial successful run. Fatal runs contain only
the failure manifest, runtime status and text report.

Diagnostic artifacts may be added only by an explicit diagnostic level and
must not change semantic decisions. With `diagnostic_level=standard`,
analyze writes `diagnostics/pair_scores.json`. This artifact stores compact
pairwise visual score records: run/stream IDs, representation variant, scorer
ID/version, score semantics, compared frame pair, candidate endpoint IDs, raw
metric value/name when available, bounded `visual_score`, validity and
warning/error/provenance lineage. It does not store embeddings, handcrafted
working vectors, bitmap masks, annotation, GT mappings or evaluation metrics.
No retention or automatic deletion policy is implemented at this stage.

Evaluation runs are created separately under:

```text
outputs/
  evaluations/
    <evaluation_id>/
      evaluation_manifest.json
      evaluation_report.json
      error_ledger.json
      summary.txt
```

Schema identifiers:

- `stream_analysis.evaluation_manifest.v1`;
- `stream_analysis.evaluation_report.v1`;
- `stream_analysis.evaluation_error_ledger.v1`.

`evaluate` reads a saved run plus `annotation.json`; it never updates the run
directory or probe data. If `diagnostics/pair_scores.json` exists, evaluator
uses it to compute representation AP/AUROC/Recall@k/MRR without rerunning
analyze and without loading embeddings. If it is absent, representation
ranking metrics are explicitly marked `not_supported_by_saved_artifacts`.
Before scoring, evaluator requires the typed artifact reference and verifies
the schema, run/stream IDs, analysis and scorer config digests, visual-score
semantics, row types, uniqueness and finite `[0,1]` scores. Malformed,
unreferenced or foreign-run diagnostics are rejected rather than silently
included in metrics.
The evaluator also reconstructs every neighboring-frame Cartesian matrix from
the candidate manifest, checks candidate/frame and representation-summary
references, and rejects missing or extra rows. Retrieval is reported as
`forward`, `reverse` and their unweighted directional `aggregate`; endpoint
validity comes from compact representation summaries, so an invalid query has
zero AP, RR and Recall@k in its direction.
Alongside strict metrics, `valid_only` reports AP/AUROC, bidirectional
retrieval after removing invalid endpoints, and candidate, pair and directional
query coverage. A valid score row must contain a finite `[0,1]` visual score;
an invalid row must keep `visual_score` null.
