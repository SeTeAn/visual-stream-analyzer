"""Build a local inventory of OCID source labels used for mask evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from tools.ocid_pipeline_contract import (
    DEFAULT_PROTOCOL,
    REPOSITORY_ROOT,
    OcidPipelineError,
    load_pipeline_protocol,
    sha256_file,
    write_json_exclusive,
)


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "data" / "ocid" / "derived" / "ocid_source_label_inventory_v1.json"
)


def build_inventory(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_pipeline_protocol(protocol_path)
    benchmark = _mapping(protocol.payload, "benchmark", "protocol")
    source_root = protocol.repository_root / "data" / "ocid" / "raw" / "OCID-dataset"
    source_root = source_root.resolve(strict=True)
    streams: list[dict[str, Any]] = []
    for item in (*protocol.development, *protocol.heldout):
        manifest_path = item.stream_directory / "manifest.json"
        manifest = _json_object(manifest_path, f"analysis manifest {item.stream_id}")
        metadata = _mapping(manifest, "metadata", "analysis manifest")
        frames = manifest.get("frames")
        if (
            manifest.get("stream_id") != item.stream_id
            or metadata.get("source_sequence") != item.source_sequence
            or metadata.get("role") != item.role
            or not isinstance(frames, list)
            or len(frames) != item.frame_count
        ):
            raise OcidPipelineError(f"analysis manifest differs for {item.stream_id}")
        labels: list[dict[str, Any]] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise OcidPipelineError(f"malformed frame metadata for {item.stream_id}")
            frame_metadata = _mapping(frame, "metadata", "frame")
            frame_id = _text(frame, "frame_id", "frame")
            source_filename = _text(
                frame_metadata,
                "ocid_source_filename",
                "frame metadata",
            )
            _safe_filename(source_filename)
            relative = PurePosixPath(item.source_sequence) / "label" / source_filename
            label_path = (source_root / Path(*relative.parts)).resolve(strict=True)
            try:
                label_path.relative_to(source_root)
            except ValueError as error:
                raise OcidPipelineError("source label path escapes the OCID root") from error
            if not label_path.is_file():
                raise OcidPipelineError(f"source label is not a file: {label_path}")
            labels.append(
                {
                    "frame_id": frame_id,
                    "frame_index": int(frame["index"]),
                    "source_filename": source_filename,
                    "path": relative.as_posix(),
                    "size_bytes": label_path.stat().st_size,
                    "sha256": sha256_file(label_path),
                }
            )
        streams.append(
            {
                "stream_id": item.stream_id,
                "role": item.role,
                "source_sequence": item.source_sequence,
                "frame_count": item.frame_count,
                "labels": labels,
            }
        )
    payload = {
        "schema_version": "ocid-pipeline-source-label-inventory-1.0",
        "benchmark_id": benchmark["benchmark_id"],
        "purpose": "catalog_evaluation_only_source_labels_for_mask_metrics",
        "source_root": "data/ocid/raw/OCID-dataset",
        "benchmark_spec_sha256": _mapping(benchmark, "spec", "benchmark")["sha256"],
        "totals": {
            "development_streams": len(protocol.development),
            "development_label_files": sum(item.frame_count for item in protocol.development),
            "heldout_streams": len(protocol.heldout),
            "heldout_label_files": sum(item.frame_count for item in protocol.heldout),
            "label_files": sum(item.frame_count for item in (*protocol.development, *protocol.heldout)),
        },
        "streams": sorted(streams, key=lambda row: str(row["stream_id"])),
    }
    write_json_exclusive(output_path, payload)
    return payload


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcidPipelineError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise OcidPipelineError(f"{label} must be a JSON object")
    return payload


def _mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise OcidPipelineError(f"{label}.{key} must be an object")
    return result


def _text(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise OcidPipelineError(f"{label}.{key} must be a non-empty string")
    return result


def _safe_filename(value: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if len(path.parts) != 1 or path.name != value or value in {".", ".."}:
        raise OcidPipelineError("OCID source filename must be one safe path segment")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_inventory(protocol_path=args.protocol, output_path=args.output)
    print(
        json.dumps(
            {
                "output": args.output.resolve(strict=True).as_posix(),
                "label_files": payload["totals"]["label_files"],
                "status": "completed",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
