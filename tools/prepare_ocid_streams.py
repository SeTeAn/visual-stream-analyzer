"""Prepare selected OCID RGB sequences for the Visual Stream Analyzer CLI.

The generated stream directories contain RGB frames and ``manifest.json`` only.
OCID label masks are checked for one-to-one filename alignment but are never
copied or referenced by the analyze-input manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


MANIFEST_SCHEMA_VERSION = "stream-input-0.1"
SOURCE_URL = "https://researchdata.tuwien.at/records/pcbjd-4wa12"
DEFAULT_SEQUENCES = (
    "ARID10/table/top/box/seq05",
    "YCB10/table/top/mixed/seq21",
    "ARID20/table/top/seq01",
)


@dataclass(frozen=True, slots=True)
class PreparedStream:
    stream_id: str
    source_sequence: str
    stream_directory: Path
    frame_count: int


def _relative_sequence(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sequence must be a non-empty relative OCID path.")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise ValueError(f"sequence must remain inside the OCID root: {value!r}.")
    if len(posix.parts) < 4 or not posix.parts[-1].startswith("seq"):
        raise ValueError(f"sequence does not look like an OCID sequence path: {value!r}.")
    return posix


def _stream_id(sequence: PurePosixPath) -> str:
    return "ocid_" + "_".join(part.casefold().replace("-", "_") for part in sequence.parts)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _source_files(ocid_root: Path, sequence: PurePosixPath) -> tuple[Path, list[Path]]:
    source = ocid_root.joinpath(*sequence.parts).resolve(strict=False)
    if not _inside(source, ocid_root):
        raise ValueError(f"Resolved sequence is outside OCID root: {sequence.as_posix()}.")
    if not source.is_dir():
        raise FileNotFoundError(f"OCID sequence directory does not exist: {source}.")
    rgb_files = sorted((source / "rgb").glob("*.png"))
    label_files = sorted((source / "label").glob("*.png"))
    if not rgb_files:
        raise ValueError(f"OCID sequence contains no RGB PNG files: {source}.")
    if [path.name for path in rgb_files] != [path.name for path in label_files]:
        raise ValueError(f"RGB and label filenames do not align: {source}.")
    return source, rgb_files


def _manifest(sequence: PurePosixPath, stream_id: str, rgb_files: list[Path]) -> dict[str, object]:
    frames = []
    for index, source_path in enumerate(rgb_files, start=1):
        frame_id = f"frame_{index:03d}"
        frames.append(
            {
                "frame_id": frame_id,
                "index": index,
                "image_path": f"frames/{frame_id}.png",
                "metadata": {
                    "ocid_source_filename": source_path.name,
                    "ocid_state_position": index,
                },
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stream_id": stream_id,
        "scene_description": (
            "Real OCID controlled scene with objects added incrementally to a fixed workspace."
        ),
        "ordering": "manifest",
        "frames": frames,
        "notes": "Generated from OCID RGB frames for an annotation-free external baseline.",
        "metadata": {
            "source_dataset": "OCID",
            "source_sequence": sequence.as_posix(),
            "source_url": SOURCE_URL,
            "adapter": "tools/prepare_ocid_streams.py",
            "purpose": "external_real_baseline",
            "frame_size": {"width": 640, "height": 480},
            "frame_format": "png_rgb",
            "is_synthetic": False,
        },
    }


def prepare_sequence(*, ocid_root: Path, output_root: Path, sequence_name: str) -> PreparedStream:
    root = Path(ocid_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"OCID root is not a directory: {root}.")
    output = Path(output_root).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    sequence = _relative_sequence(sequence_name)
    _, rgb_files = _source_files(root, sequence)
    stream_id = _stream_id(sequence)
    destination = output / stream_id
    if destination.exists():
        raise FileExistsError(f"Prepared stream already exists: {destination}.")

    staging = Path(tempfile.mkdtemp(prefix=f".{stream_id}-", dir=output))
    try:
        frames_directory = staging / "frames"
        frames_directory.mkdir()
        for index, source_path in enumerate(rgb_files, start=1):
            shutil.copy2(source_path, frames_directory / f"frame_{index:03d}.png")
        payload = _manifest(sequence, stream_id, rgb_files)
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PreparedStream(
        stream_id=stream_id,
        source_sequence=sequence.as_posix(),
        stream_directory=destination,
        frame_count=len(rgb_files),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="Relative OCID sequence path. Repeat to prepare multiple sequences.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sequences = tuple(args.sequences) if args.sequences else DEFAULT_SEQUENCES
    prepared = [
        prepare_sequence(
            ocid_root=args.ocid_root,
            output_root=args.output_root,
            sequence_name=sequence,
        )
        for sequence in sequences
    ]
    print(
        json.dumps(
            {
                "status": "prepared",
                "streams": [
                    {
                        "stream_id": item.stream_id,
                        "source_sequence": item.source_sequence,
                        "stream_directory": str(item.stream_directory),
                        "frame_count": item.frame_count,
                    }
                    for item in prepared
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
