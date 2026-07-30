# Analysis output

The public command writes one compact, self-contained result directory:

```text
<output>/
  result.json
  masks/
    <frame_id>/
      P00.png
      P01.png
  overlays/
    <frame_id>.png
```

The destination must not already exist. Files are first produced in a temporary sibling directory and then published as one atomic directory operation that refuses to replace existing data. If analysis or publication fails, no partial result directory is left behind.

## `result.json`

The JSON document contains:

- stream identity, input-manifest hash, and ordered frame count;
- the fixed runtime profile and the Grounding DINO, SAM2, and DINOv2 model families;
- every frame and its frame-local objects;
- predicted visual groups (`type_...`);
- accepted or uncertain matches between neighboring frames;
- structured scene-change events;
- aggregate counts for frames, objects, visual groups, matches, and events.

An object reference is always the pair `(<frame_id>, <Pnn>)`. The same `P00` filename in two different frame directories does not by itself identify the same object. Cross-frame relations are represented explicitly in `matches` and `visual_types`.

## Masks

Every mask is a full-frame, single-channel binary PNG:

- `0` represents background;
- `255` represents the predicted object region;
- the image dimensions equal the corresponding RGB frame;
- the mask is the final cleaned SAM2 result after aggregate-mask resolution.

There is exactly one JSON object record for every mask file and no extra mask files.

## Overlays

Each frame has exactly one RGB overlay generated from the published masks. An overlay displays the predicted `type_...` visual-group label. It does not display detector confidence or internal candidate identifiers.

All paths stored in `result.json` are relative POSIX paths. The public result does not contain model responses, embeddings, evaluation manifests, receipts, or absolute local paths.

A complete checked-in example is available at [`assets/demo-stream/ocid_arid10_table_top_fruits_seq10`](../assets/demo-stream/ocid_arid10_table_top_fruits_seq10).
