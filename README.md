# Visual Stream Analyzer

Visual Stream Analyzer is a Python project for analyzing ordered image streams. It detects candidate visual regions, builds visual representations, matches objects between neighboring frames, groups recurring visual entities, and reports change events such as appearance, disappearance, persistence, count changes, and position changes.

The project is organized as a command-line pipeline with separate validation, analysis, and evaluation stages. The main analysis pipeline does not read ground-truth annotations; annotations are used only by the evaluation command for saved run artifacts.

## What Is Included

- Source code for the stream analysis package: `src/stream_analysis`
- Unit and integration tests: `tests`
- Data streams with frames, manifests, and annotations: `data/streams`
- A compact example run: `outputs/runs/demonstration_final_h04_dino_bbox`
- A compact evaluation matrix: `outputs/evaluations/demonstration_final_matrix`
- Base Python dependencies: `requirements.txt`
- Optional DINOv2/PyTorch dependencies: `requirements-ml.txt`

Large experiment archives and model weights are intentionally not included.

## Visual Example

The included example stream contains generated image sequences with recurring objects. The analysis output overlays detected recurring visual types and frame-to-frame change events.

| Input frame | Analysis overlay |
|---|---|
| ![Input frame](assets/visual-example/input-frame.png) | ![Analysis overlay](assets/visual-example/analysis-overlay.png) |

## Pipeline

1. Load an ordered frame stream and validate its manifest.
2. Extract candidate regions from each frame.
3. Build visual representations using either handcrafted features or DINOv2-based features.
4. Compare neighboring frames and select candidate matches.
5. Group recurring visual entities across the stream.
6. Detect temporal change events.
7. Write JSON artifacts, overlay images, runtime status, and a text report.
8. Evaluate saved artifacts against annotations when evaluation data is available.

## Technologies

- Python
- NumPy
- OpenCV
- Pillow
- SciPy
- PyTorch and torchvision for the optional DINOv2 path
- DINOv2-style visual representations with local model source and checkpoint paths
- `unittest` test suite

## Repository Layout

```text
src/stream_analysis/        # CLI, pipeline, matching, grouping, reporting, evaluation
data/streams/               # image streams, manifests, annotations
outputs/
  runs/                     # curated example analysis run
  evaluations/              # curated example evaluation summary
tests/                      # unit and integration tests
tools/                      # helper scripts
```

## Setup

Create a virtual environment and install the base dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `PYTHONPATH` before running the package from the repository root:

```powershell
$env:PYTHONPATH = "src"
```

## Run A Handcrafted Example

The handcrafted path does not require external model weights:

```powershell
.\.venv\Scripts\python.exe -B -m stream_analysis validate `
  data\streams\probe_01_stationery `
  --config outputs\evaluations\demonstration_final_matrix\config_handcrafted_bbox_v1.json `
  --representation-family handcrafted `
  --variant handcrafted_bbox_v1

.\.venv\Scripts\python.exe -B -m stream_analysis analyze `
  data\streams\probe_01_stationery `
  --config outputs\evaluations\demonstration_final_matrix\config_handcrafted_bbox_v1.json `
  --representation-family handcrafted `
  --variant handcrafted_bbox_v1 `
  --device cpu `
  --diagnostic-level standard `
  --output-root .local_outputs `
  --run-id smoke_handcrafted_bbox
```

Evaluate the saved run:

```powershell
.\.venv\Scripts\python.exe -B -m stream_analysis evaluate `
  data\streams\probe_01_stationery `
  --run-directory .local_outputs\runs\smoke_handcrafted_bbox `
  --output-root .local_outputs `
  --evaluation-id smoke_handcrafted_bbox_eval
```

## Optional DINOv2 Path

The DINOv2 path requires optional PyTorch dependencies and local DINOv2 assets. Model code and weights are not included in this repository.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

When running DINOv2 analysis, pass explicit local paths:

```powershell
--dinov2-source <path-to-dinov2-source> --dinov2-checkpoint <path-to-checkpoint>
```

## Learned Candidate Extraction

The real-image path can replace the controlled-background extractor with the
typed `torchvision_maskrcnn` candidate family in the analysis JSON config. Its
parameters must pin the Torchvision version, checkpoint SHA-256 and checkpoint
size; the model weights are local assets and are not downloaded by `analyze`.

Pass the verified checkpoint explicitly together with the DINOv2 assets:

```powershell
--candidate-checkpoint <path-to-maskrcnn-checkpoint> `
--dinov2-source <path-to-dinov2-source> `
--dinov2-checkpoint <path-to-dinov2-checkpoint>
```

The implemented working configuration uses Mask R-CNN only as a
COCO closed-vocabulary provider. It must not be described as a general
open-world extractor. Candidate-only OCID annotations are evaluation inputs and
must remain outside the RGB stream passed to `analyze`.

## Example Results

The curated evaluation matrix contains 20 completed analyze/evaluate entries across five held-out streams and four representation variants. The full numeric table is stored in:

```text
outputs/evaluations/demonstration_final_matrix/variant_macro_summary.csv
```

Selected values from that table:

| Variant | Representation | Candidate F1 | Matching F1 | Grouping B-cubed F1 | Event pooled F1 |
|---|---:|---:|---:|---:|---:|
| `handcrafted_bbox_v1` | handcrafted | 1.000 | 0.886 | 0.862 | 0.822 |
| `handcrafted_mask_v1` | handcrafted | 1.000 | 0.862 | 0.892 | 0.838 |
| `bbox_rgb_letterbox_v1` | DINOv2 | 1.000 | 0.785 | 1.000 | 0.927 |
| `mask_neutral_letterbox_v1` | DINOv2 | 1.000 | 0.794 | 0.989 | 0.920 |

These results describe the included evaluation artifacts. They should not be read as production performance or as a general benchmark claim.

## Tests

Run the test suite from the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

Some DINOv2 integration tests require local model assets. The base handcrafted pipeline can be exercised without model weights.
