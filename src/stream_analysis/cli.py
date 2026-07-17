"""Command-line entry point for validation and primary analyze."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .contracts import ProducerProvenance
from .config import EvaluationDataRole, semantic_config_digest
from .evaluation import (
    EvaluationCollisionError,
    EvaluationError,
    EvaluationRequest,
    evaluate_saved_run,
)
from .input import ManifestLoadRequest, StreamInputError, load_decoded_stream
from .integration_config import load_analyze_configuration
from .orchestration import (
    AnalyzeRequest, AnalyzeRunError, CandidateExtractorAssetPaths,
    DinoV2AssetPaths, run_analysis,
)
from .reporting import OutputCollisionError
from .representations import DinoV2RepresentationConfig
from .candidates import MaskRCNNCandidateExtractionConfig

EXIT_OK = 0
EXIT_USAGE_OR_CONFIG = 2
EXIT_INPUT_ERROR = 3
EXIT_OUTPUT_COLLISION = 4
EXIT_ANALYZE_FAILED = 5
EXIT_PARTIAL = 6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m stream_analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate config, manifest and image decoding.")
    analyze = commands.add_parser("analyze", help="Run the primary annotation-free pipeline.")
    evaluate = commands.add_parser("evaluate", help="Evaluate saved analyze artifacts against annotation.json.")
    for command in (validate, analyze):
        command.add_argument("stream_directory", type=Path)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--representation-family", choices=("handcrafted", "dinov2"), required=True)
        command.add_argument("--variant", required=True)
        command.add_argument("--device", choices=("cpu", "cuda", "auto"))
        command.add_argument(
            "--diagnostic-level",
            choices=("none", "standard"),
            help=(
                "Override config diagnostic policy. 'standard' writes "
                "diagnostics/pair_scores.json for saved-run representation evaluation."
            ),
        )
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--dinov2-source", type=Path)
    analyze.add_argument("--dinov2-checkpoint", type=Path)
    analyze.add_argument("--candidate-checkpoint", type=Path)
    evaluate.add_argument("stream_directory", type=Path)
    evaluate.add_argument("--run-directory", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--evaluation-id", required=True)
    evaluate.add_argument("--annotation", type=Path)
    evaluate.add_argument(
        "--data-role",
        choices=tuple(item.value for item in EvaluationDataRole),
        default=EvaluationDataRole.PROBE_DEVELOPMENT.value,
    )
    return parser


def _load_and_check(args: argparse.Namespace):
    loaded = load_analyze_configuration(args.config)
    representation = loaded.pipeline.representation
    actual_family = "dinov2" if isinstance(representation, DinoV2RepresentationConfig) else "handcrafted"
    if args.representation_family != actual_family or args.variant != representation.variant:
        raise ValueError("CLI representation family/variant must match the typed configuration.")
    if args.device is not None:
        candidate_config = loaded.pipeline.candidate_extraction
        learned_candidate = isinstance(candidate_config, MaskRCNNCandidateExtractionConfig)
        learned_representation = isinstance(representation, DinoV2RepresentationConfig)
        if not learned_candidate and not learned_representation and args.device != "cpu":
            raise ValueError("The selected pipeline supports only cpu.")
        loaded = replace(
            loaded,
            pipeline=replace(
                loaded.pipeline,
                candidate_extraction=(
                    replace(candidate_config, device_policy=args.device)
                    if learned_candidate
                    else candidate_config
                ),
                representation=(
                    replace(representation, device_policy=args.device)
                    if learned_representation
                    else representation
                ),
            ),
            environment=replace(loaded.environment, requested_device=args.device),
        )
    if args.diagnostic_level is not None:
        loaded = replace(
            loaded,
            pipeline=replace(loaded.pipeline, diagnostic_level=args.diagnostic_level),
        )
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "evaluate":
            result = evaluate_saved_run(EvaluationRequest(
                stream_directory=args.stream_directory,
                run_directory=args.run_directory,
                output_root=args.output_root,
                evaluation_id=args.evaluation_id,
                annotation_path=args.annotation,
                data_role=EvaluationDataRole(args.data_role),
            ))
            print(json.dumps({
                "evaluation_id": args.evaluation_id,
                "status": "completed",
                "evaluation_directory": str(result.artifacts.evaluation_directory),
            }, sort_keys=True))
            return EXIT_OK
        loaded = _load_and_check(args)
        if args.command == "validate":
            stage = loaded.pipeline.analysis_config.stream_input
            decoded = load_decoded_stream(ManifestLoadRequest(
                stream_root=args.stream_directory,
                producer=ProducerProvenance(
                    producer_stage="stream_input", producer_version="1.0.0",
                    config_version=stage.config_version,
                    config_digest=semantic_config_digest(stage),
                ),
            ))
            print(json.dumps({"status": "valid", "stream_id": decoded.stream.stream_id, "frame_count": len(decoded.frames)}, sort_keys=True))
            return EXIT_OK

        representation = loaded.pipeline.representation
        candidate_config = loaded.pipeline.candidate_extraction
        candidate_assets = None
        if isinstance(candidate_config, MaskRCNNCandidateExtractionConfig):
            if args.candidate_checkpoint is None:
                raise ValueError("Mask R-CNN candidate extraction requires --candidate-checkpoint.")
            candidate_assets = CandidateExtractorAssetPaths(
                checkpoint_path=args.candidate_checkpoint,
            )
        elif args.candidate_checkpoint is not None:
            raise ValueError("--candidate-checkpoint is invalid for controlled-background extraction.")
        assets = None
        if isinstance(representation, DinoV2RepresentationConfig):
            if args.dinov2_source is None or args.dinov2_checkpoint is None:
                raise ValueError("DINOv2 analyze requires --dinov2-source and --dinov2-checkpoint.")
            assets = DinoV2AssetPaths(
                source_directory=args.dinov2_source,
                checkpoint_path=args.dinov2_checkpoint,
            )
        elif args.dinov2_source is not None or args.dinov2_checkpoint is not None:
            raise ValueError("DINOv2 asset paths are not valid for handcrafted analyze.")
        outcome = run_analysis(AnalyzeRequest(
            stream_directory=args.stream_directory, output_root=args.output_root,
            run_id=args.run_id, config=loaded.pipeline,
            source_provenance=loaded.source,
            environment_provenance=loaded.environment,
            candidate_extractor_assets=candidate_assets,
            dinov2_assets=assets,
        ))
        print(json.dumps({"run_id": outcome.result.run_id, "status": outcome.result.run_status.value,
                          "run_directory": str(outcome.artifacts.run_directory)}, sort_keys=True))
        return EXIT_PARTIAL if outcome.result.run_status.value == "partial" else EXIT_OK
    except OutputCollisionError as error:
        print(str(error), file=sys.stderr)
        return EXIT_OUTPUT_COLLISION
    except EvaluationCollisionError as error:
        print(str(error), file=sys.stderr)
        return EXIT_OUTPUT_COLLISION
    except EvaluationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_INPUT_ERROR
    except AnalyzeRunError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR if error.code == "INPUT_FAILED" else EXIT_ANALYZE_FAILED
    except StreamInputError as error:
        print(str(error), file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG


__all__ = [
    "EXIT_ANALYZE_FAILED", "EXIT_INPUT_ERROR", "EXIT_OK", "EXIT_OUTPUT_COLLISION",
    "EXIT_PARTIAL", "EXIT_USAGE_OR_CONFIG", "main",
]
