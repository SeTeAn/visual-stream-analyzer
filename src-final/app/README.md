# Application Package

`app/` contains the Python package used by the command-line stream analysis pipeline.

The public entry point is:

```powershell
python -m stream_analysis
```

Main commands:

- `validate` checks stream input, configuration, manifest consistency, and image decoding.
- `analyze` runs the annotation-free analysis pipeline and writes run artifacts.
- `evaluate` scores a saved run against `annotation.json` and writes evaluation artifacts.

The package keeps analysis and evaluation separate: `analyze` does not read ground-truth annotations, and `evaluate` does not modify saved run artifacts.
