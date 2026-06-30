"""Immutable lifecycle for evaluation outputs."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATION_REPORT_SCHEMA_ID = "stream_analysis.evaluation_report.v1"
EVALUATION_ERROR_LEDGER_SCHEMA_ID = "stream_analysis.evaluation_error_ledger.v1"
EVALUATION_MANIFEST_SCHEMA_ID = "stream_analysis.evaluation_manifest.v1"


class EvaluationCollisionError(FileExistsError):
    """Raised before writing when an evaluation directory already exists."""


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_PORTABLE_PATH_FORBIDDEN = frozenset('<>:"/\\|?*')


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    evaluation_directory: Path
    files: tuple[Path, ...]


def _validate_portable_evaluation_id(evaluation_id: str) -> None:
    """Require one unambiguous path component on Windows, Linux and macOS."""

    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ValueError("evaluation_id must be a non-empty string.")
    if any(character in _PORTABLE_PATH_FORBIDDEN for character in evaluation_id):
        raise ValueError("evaluation_id contains a character forbidden in portable path segments.")
    if evaluation_id.endswith((".", " ")):
        raise ValueError("evaluation_id must not end with a dot or space.")
    if evaluation_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("evaluation_id must not use a Windows reserved device name.")
    if len(evaluation_id.encode("utf-8")) > 255:
        raise ValueError("evaluation_id is too long for a portable path segment.")


def write_evaluation_artifacts(
    *,
    output_root: Path,
    evaluation_id: str,
    report: dict[str, Any],
    error_ledger: list[dict[str, Any]],
    summary_text: str,
    manifest: dict[str, Any],
) -> EvaluationArtifacts:
    _validate_portable_evaluation_id(evaluation_id)
    root = Path(output_root).resolve() / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    target = root / evaluation_id
    if target.exists():
        raise EvaluationCollisionError(f"Evaluation directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{evaluation_id}.", dir=root))
    relative_files: list[Path] = []

    def write_json(name: str, value: Any) -> None:
        path = staging / name
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        relative_files.append(Path(name))

    def write_text(name: str, value: str) -> None:
        path = staging / name
        path.write_text(value, encoding="utf-8", newline="\n")
        relative_files.append(Path(name))

    try:
        write_json("evaluation_manifest.json", {"schema_id": EVALUATION_MANIFEST_SCHEMA_ID, **manifest})
        write_json("evaluation_report.json", {"schema_id": EVALUATION_REPORT_SCHEMA_ID, **report})
        write_json("error_ledger.json", {"schema_id": EVALUATION_ERROR_LEDGER_SCHEMA_ID, "errors": error_ledger})
        write_text("summary.txt", summary_text)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return EvaluationArtifacts(
        evaluation_directory=target,
        files=tuple(target / item for item in sorted(relative_files)),
    )
