# Source Package

This directory contains the working source tree for Visual Stream Analyzer.

The package is intentionally split into independent stages:

- input validation and frame loading
- candidate extraction
- visual representation building
- neighboring-frame matching
- cross-frame grouping
- event detection
- artifact reporting
- saved-run evaluation

The main analysis command does not read `annotation.json`. Evaluation is performed separately against saved run artifacts.

Use the repository-level `README.md` for setup and example commands.
