"""Public command-line interface for the current Visual Stream Analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ProductEvent,
    ProductEventKind,
    ProductFrame,
    ProductMask,
    ProductMatch,
    ProductObjectRef,
    ProductPipeline,
    ProductResult,
    ProductStatus,
    ProductStream,
    ProducerProvenance,
)
from .input import ManifestLoadRequest, StreamInputError, decode_stream, load_manifest
from .reporting import ProductOutputCollisionError, write_product_output
from .runtime.config import (
    RuntimeConfigurationError,
    load_runtime_profile,
    require_cuda,
    require_portable_component,
    resolve_runtime_assets,
)
from .runtime.engine import RuntimeOutcome, RuntimePipelineError, execute_pipeline
from .runtime.workspace import RuntimeWorkspace


EXIT_OK = 0
EXIT_USAGE_OR_CONFIG = 2
EXIT_INPUT_ERROR = 3
EXIT_OUTPUT_COLLISION = 4
EXIT_ANALYSIS_FAILED = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stream_analysis",
        description=(
            "Analyse an ordered image stream with the fixed Visual Stream Analyzer pipeline."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyse one manifest-defined image stream.")
    analyze.add_argument("stream_directory", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--models-root", type=Path)
    return parser


def _mask_sort_key(type_id: str, mask: np.ndarray) -> tuple[str, int, int, int, bytes]:
    rows, columns = np.nonzero(mask)
    return (
        type_id,
        int(rows.min()),
        int(columns.min()),
        int(mask.sum()),
        hashlib.sha256(mask.tobytes(order="C")).digest(),
    )


def _product_result(outcome: RuntimeOutcome) -> ProductResult:
    if len(outcome.streams) != 1:
        raise RuntimePipelineError("the public command expects exactly one stream outcome")
    stream_outcome = outcome.streams[0]
    decoded = stream_outcome.detector.decoded
    analysis = stream_outcome.analysis
    mask_by_id = {record.candidate_id: record for record in stream_outcome.masks.records}
    type_by_candidate = {
        assignment.candidate_id: assignment.primary_type_id
        for assignment in analysis.grouping.assignments
    }
    candidate_by_frame: dict[str, list[str]] = {
        frame.frame_id: [] for frame in decoded.frames
    }
    for candidate in analysis.snapshot.result.candidates:
        candidate_by_frame[candidate.frame_id].append(candidate.candidate_id)

    frames: list[ProductFrame] = []
    object_ref_by_candidate: dict[str, ProductObjectRef] = {}
    for frame in decoded.frames:
        entries: list[tuple[str, str, np.ndarray]] = []
        for candidate_id in sorted(candidate_by_frame[frame.frame_id]):
            observation = mask_by_id.get(candidate_id)
            type_id = type_by_candidate.get(candidate_id)
            if observation is None or type_id is None or observation.result.cleaned_mask is None:
                raise RuntimePipelineError(
                    f"final candidate lacks a cleaned mask or visual type: {candidate_id}"
                )
            entries.append(
                (
                    candidate_id,
                    type_id,
                    np.ascontiguousarray(observation.result.cleaned_mask, dtype=np.bool_),
                )
            )
        ordered_entries = sorted(
            entries,
            key=lambda item: _mask_sort_key(item[1], item[2]),
        )
        product_frame = ProductFrame(
            frame_id=frame.frame_id,
            frame_index=frame.record.index,
            source_image=frame.record.image_path,
            image_size=frame.image_size,
            masks=tuple(
                ProductMask(type_id=type_id, mask=mask)
                for _candidate_id, type_id, mask in ordered_entries
            ),
        )
        for (candidate_id, _type_id, _mask), product_mask in zip(
            ordered_entries, product_frame.masks, strict=True
        ):
            assert product_mask.object_id is not None
            object_ref_by_candidate[candidate_id] = ProductObjectRef(
                frame_id=frame.frame_id,
                object_id=product_mask.object_id,
            )
        frames.append(product_frame)

    matches: list[ProductMatch] = []
    for comparison in analysis.matching_results:
        for match in comparison.selected_matches:
            matches.append(
                ProductMatch(
                    from_object=object_ref_by_candidate[match.from_endpoint.candidate_id],
                    to_object=object_ref_by_candidate[match.to_endpoint.candidate_id],
                    status=match.status.value,
                )
            )

    events: list[ProductEvent] = []
    ordered_events = sorted(
        analysis.event_batch.events,
        key=lambda item: (
            item.frame_pair.from_frame_index,
            item.frame_pair.to_frame_index,
            item.predicted_type_id,
            item.kind.value,
        ),
    )
    for index, event in enumerate(ordered_events, start=1):
        try:
            from_objects = tuple(
                object_ref_by_candidate[candidate_id]
                for candidate_id in event.evidence.from_member_ids
            )
            to_objects = tuple(
                object_ref_by_candidate[candidate_id]
                for candidate_id in event.evidence.to_member_ids
            )
        except KeyError as error:
            raise RuntimePipelineError(
                f"event references an unknown final candidate: {error.args[0]}"
            ) from error
        events.append(
            ProductEvent(
                event_id=f"event_{index:03d}",
                kind=ProductEventKind(event.kind.value),
                type_id=event.predicted_type_id,
                from_frame_id=event.frame_pair.from_frame_id,
                to_frame_id=event.frame_pair.to_frame_id,
                from_objects=from_objects,
                to_objects=to_objects,
            )
        )

    has_warnings = bool(
        analysis.representation_batch.warnings or analysis.event_batch.warnings
    )
    return ProductResult(
        stream=ProductStream(
            stream_id=decoded.stream.stream_id,
            input_schema_version="stream-input-0.1",
            manifest_sha256=decoded.manifest.manifest_digest.value,
            frame_count=len(decoded.frames),
        ),
        pipeline=ProductPipeline(
            profile_id=outcome.profile_id,
            profile_sha256=outcome.profile_sha256,
        ),
        frames=tuple(frames),
        status=(
            ProductStatus.COMPLETED_WITH_WARNINGS
            if has_warnings
            else ProductStatus.COMPLETED
        ),
        matches=tuple(matches),
        events=tuple(events),
    )


def _producer(profile_sha256: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0.0",
        config_version="1.0.0",
        config_digest=f"sha256:{profile_sha256}",
    )


def _run_analyze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve(strict=False)
    if output.exists():
        raise ProductOutputCollisionError("Product output already exists.")
    profile = load_runtime_profile()
    manifest = load_manifest(
        ManifestLoadRequest(
            stream_root=Path(args.stream_directory),
            producer=_producer(profile.sha256),
        )
    )
    require_portable_component(manifest.stream_id, "stream_id")
    for frame in manifest.frames:
        require_portable_component(frame.frame_id, f"frame_id {frame.frame_id!r}")
    assets = resolve_runtime_assets(profile, args.models_root, verify=True)
    require_cuda()
    decoded = decode_stream(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.runtime-",
        dir=output.parent,
    ) as workspace_directory:
        outcome = execute_pipeline(
            (decoded,),
            profile=profile,
            assets=assets,
            workspace=RuntimeWorkspace(Path(workspace_directory)),
        )
        result = _product_result(outcome)
        write_product_output(
            output,
            result,
            {frame.frame_id: frame.to_pillow_image() for frame in decoded.frames},
        )
    return {
        "status": result.status.value,
        "stream_id": result.stream.stream_id,
        "frame_count": result.stream.frame_count,
        "object_count": sum(len(frame.masks) for frame in result.frames),
        "visual_type_count": len(result.visual_types),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "analyze":
            parser.error("analyze is the only public command")
        payload = _run_analyze(args)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK
    except ProductOutputCollisionError as error:
        print(str(error), file=sys.stderr)
        return EXIT_OUTPUT_COLLISION
    except StreamInputError as error:
        print(str(error), file=sys.stderr)
        return EXIT_INPUT_ERROR
    except RuntimeConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG
    except RuntimePipelineError as error:
        print(str(error), file=sys.stderr)
        return EXIT_ANALYSIS_FAILED
    except OSError:
        print("A required file or directory could not be accessed.", file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG
    except RuntimeError as error:
        print(f"Analysis failed ({type(error).__name__}).", file=sys.stderr)
        return EXIT_ANALYSIS_FAILED


__all__ = [
    "EXIT_ANALYSIS_FAILED",
    "EXIT_INPUT_ERROR",
    "EXIT_OK",
    "EXIT_OUTPUT_COLLISION",
    "EXIT_USAGE_OR_CONFIG",
    "main",
]
