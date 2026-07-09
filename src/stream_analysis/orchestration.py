"""F14 orchestration for the primary, annotation-free analyze lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from .candidates import CandidateExtractionConfig, CandidateExtractionSnapshot, extract_candidates
from .config import AnalysisRunConfig, StageConfig, semantic_config_digest
from .contracts import (
    ArtifactReference, DinoEmbeddingPayload, EmissionStatus, FramePair,
    ChangeEvent, HandcraftedPayload, ProducerProvenance, RecordEnvelope, RepresentationRecord,
    RepresentationSummary, RunStatus, RuntimeMetadata, StageContext,
    StreamAnalysisResult, ValidityStatus, VersionedMetadata,
)
from .input import ManifestLoadRequest, load_decoded_stream
from .matching import (
    EventConfig, GroupingConfig, MatchingConfig, PairScoringConfig,
    build_change_events, group_recurring_visual_types, match_neighboring_frames,
)
from .provenance import (
    AnalyzeRunProvenance, DataProvenance, EnvironmentProvenance, Fingerprint,
    ModelProvenance, RunRole, RunTimestamps, SourceProvenance,
    StageConfigProvenance,
)
from .reporting import (
    CANDIDATE_MANIFEST_SCHEMA_ID, PAIR_SCORES_SCHEMA_ID,
    RUN_MANIFEST_SCHEMA_ID, RUNTIME_STATUS_SCHEMA_ID, RunArtifacts,
    build_candidate_manifest_payload, build_pair_scores_payload,
    build_text_report, render_primary_overlays, write_failed_run_artifacts,
    write_run_artifacts,
)
from .representations import (
    DINO_MODEL_NAME, DINO_MODEL_VERSION, DinoV2RepresentationConfig,
    HandcraftedRepresentationConfig, LocalDinoV2Provider,
    build_dinov2_representations, build_handcrafted_representations,
)
from .representations.scoring import (
    DINO_CANONICAL_SCORER_ID, DINO_CANONICAL_SCORER_VERSION,
    HANDCRAFTED_CANONICAL_SCORER_ID, HANDCRAFTED_CANONICAL_SCORER_VERSION,
    DinoV2CosineScorer, HandcraftedCanonicalScorer,
)
from .serialization import PRIMARY_STREAM_RESULT_SCHEMA_ID

PIPELINE_VERSION = "1.0.0"
REPORTING_CONFIG_SCHEMA_ID = "stream_analysis.reporting_config.v1"
RUNTIME_POLICY_SCHEMA_ID = "stream_analysis.runtime_policy.v1"
STREAM_INPUT_CONFIG_SCHEMA_ID = "stream_analysis.stream_input_config.v1"

RepresentationConfig = HandcraftedRepresentationConfig | DinoV2RepresentationConfig

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_PORTABLE_PATH_FORBIDDEN = frozenset('<>:"/\\|?*')


def _validate_portable_run_segment(run_id: str) -> None:
    """Require one unambiguous path component on Windows, Linux and macOS."""

    if any(character in _PORTABLE_PATH_FORBIDDEN for character in run_id):
        raise ValueError("run_id contains a character forbidden in portable path segments.")
    if run_id.endswith((".", " ")):
        raise ValueError("run_id must not end with a dot or space.")
    if run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_id must not use a Windows reserved device name.")
    if len(run_id.encode("utf-8")) > 255:
        raise ValueError("run_id is too long for a portable path segment.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzePipelineConfig:
    """Typed aggregate over the already-versioned F03-F12 configurations."""

    candidate_extraction: CandidateExtractionConfig
    representation: RepresentationConfig
    scoring: PairScoringConfig
    matching: MatchingConfig
    grouping: GroupingConfig
    events: EventConfig
    config_version: str = "1.0.0"
    diagnostic_level: Literal["none", "standard"] = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_extraction, CandidateExtractionConfig):
            raise TypeError("candidate_extraction must be CandidateExtractionConfig.")
        if not isinstance(self.representation, (HandcraftedRepresentationConfig, DinoV2RepresentationConfig)):
            raise TypeError("representation must be a supported representation config.")
        for name, expected in (
            ("scoring", PairScoringConfig), ("matching", MatchingConfig),
            ("grouping", GroupingConfig), ("events", EventConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}.")
        if self.diagnostic_level not in {"none", "standard"}:
            raise ValueError("diagnostic_level must be none or standard.")
        if self.grouping.representation_variant_id != self.representation.variant:
            raise ValueError("Grouping representation variant must match the selected representation.")
        if (self.grouping.scorer_id, self.grouping.scorer_version) != (
            self.scoring.scorer_id, self.scoring.scorer_version
        ):
            raise ValueError("Scoring and grouping scorer identity/version must match.")
        expected_scorer = (
            (DINO_CANONICAL_SCORER_ID, DINO_CANONICAL_SCORER_VERSION)
            if isinstance(self.representation, DinoV2RepresentationConfig)
            else (HANDCRAFTED_CANONICAL_SCORER_ID, HANDCRAFTED_CANONICAL_SCORER_VERSION)
        )
        if (self.scoring.scorer_id, self.scoring.scorer_version) != expected_scorer:
            raise ValueError("Selected representation requires its canonical scorer ID/version.")

    @property
    def analysis_config(self) -> AnalysisRunConfig:
        reporting = StageConfig(
            stage_id="reporting", schema_id=REPORTING_CONFIG_SCHEMA_ID,
            config_version="1.0.0",
            semantic_parameters={"report_policy": "structured_facts_v1", "overlay_policy": "candidate_type_bbox_v1"},
            diagnostic_parameters={"diagnostic_level": self.diagnostic_level},
        )
        runtime = StageConfig(
            stage_id="runtime_policy", schema_id=RUNTIME_POLICY_SCHEMA_ID,
            config_version="1.0.0",
            runtime_parameters={"diagnostic_level": self.diagnostic_level},
        )
        stream_input = StageConfig(
            stage_id="stream_input", schema_id=STREAM_INPUT_CONFIG_SCHEMA_ID,
            config_version="1.0.0",
            semantic_parameters={"schema_version": "stream-input-0.1", "ordering": "manifest"},
        )
        return AnalysisRunConfig(
            config_version=self.config_version,
            stream_input=stream_input,
            candidate_extraction=self.candidate_extraction.to_stage_config(),
            representation=self.representation.to_stage_config(),
            scorer=self.scoring.to_stage_config(), matching=self.matching.to_stage_config(),
            grouping=self.grouping.to_stage_config(), events=self.events.to_stage_config(),
            reporting=reporting, runtime_policy=runtime,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DinoV2AssetPaths:
    source_directory: Path
    checkpoint_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzeRequest:
    stream_directory: Path
    output_root: Path
    run_id: str
    config: AnalyzePipelineConfig
    source_provenance: SourceProvenance
    environment_provenance: EnvironmentProvenance
    dinov2_assets: DinoV2AssetPaths | None = None

    def __post_init__(self) -> None:
        _validate_portable_run_segment(self.run_id)
        RecordEnvelope(
            record_id=self.run_id, schema_version="analyze-request-1.0",
            stream_id="request", producer=ProducerProvenance(
                producer_stage="orchestration", producer_version=PIPELINE_VERSION,
                config_version=self.config.config_version,
                config_digest=semantic_config_digest(self.config.analysis_config),
            ), context=StageContext(),
        )
        if not isinstance(self.source_provenance, SourceProvenance):
            raise TypeError("source_provenance must be SourceProvenance.")
        if not isinstance(self.environment_provenance, EnvironmentProvenance):
            raise TypeError("environment_provenance must be EnvironmentProvenance.")
        if isinstance(self.config.representation, DinoV2RepresentationConfig) and self.dinov2_assets is None:
            raise ValueError("DINOv2 analyze requires explicit local source and checkpoint paths.")
        requested = self.environment_provenance.requested_device
        if isinstance(self.config.representation, DinoV2RepresentationConfig):
            if requested != self.config.representation.device_policy:
                raise ValueError("Environment requested_device must match DINOv2 device_policy.")
        elif requested != "cpu":
            raise ValueError("Handcrafted analyze requires requested_device='cpu'.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzeOutcome:
    result: StreamAnalysisResult
    provenance: AnalyzeRunProvenance
    candidate_snapshot: CandidateExtractionSnapshot
    artifacts: RunArtifacts


class AnalyzeRunError(RuntimeError):
    """Fatal analyze failure with persisted run-scoped diagnostics."""

    def __init__(self, code: str, message: str, artifacts: RunArtifacts) -> None:
        super().__init__(message)
        self.code = code
        self.artifacts = artifacts


def _fingerprint_stage(config: StageConfig) -> StageConfigProvenance:
    digest = semantic_config_digest(config)
    algorithm, value = digest.split(":", 1)
    return StageConfigProvenance(
        stage_id=config.stage_id, schema_id=config.schema_id,
        config_version=config.config_version,
        semantic_digest=Fingerprint(algorithm=algorithm, value=value),
        runtime_parameters=config.runtime_parameters,
        diagnostic_parameters=config.diagnostic_parameters,
    )


def _representation_summary(record: RepresentationRecord) -> RepresentationSummary:
    dimension = record.payload.embedding_dimension if isinstance(record.payload, DinoEmbeddingPayload) else None
    groups = tuple(group.group_name for group in record.payload.feature_groups) if isinstance(record.payload, HandcraftedPayload) else ()
    return RepresentationSummary(
        representation_record_id=record.envelope.record_id,
        candidate_id=record.candidate_id, frame_id=record.frame_id,
        family=record.family, representation_type=record.representation_type,
        representation_version=record.representation_version,
        input_variant=record.input_variant,
        semantic_config_digest=record.semantic_config_digest,
        validity_status=record.envelope.validity_status,
        embedding_dimension=dimension, feature_group_names=groups,
        warning_ids=record.envelope.warning_ids,
    )


def _artifact_references(run_id: str, *, include_pair_scores: bool = False) -> tuple[ArtifactReference, ...]:
    specs = [
        ("run_manifest", "run_manifest", "run_manifest.json", "application/json"),
        ("candidate_manifest", "candidate_manifest", "candidate_manifest.json", "application/json"),
        ("stream_analysis", "stream_analysis", "stream_analysis.json", "application/json"),
        ("text_report", "text_report", "report.txt", "text/plain"),
        ("runtime_status", "runtime_status", "runtime_status.json", "application/json"),
        ("primary_overlays", "primary_overlays", "overlays/", "image/png"),
    ]
    if include_pair_scores:
        specs.append(
            (
                "diagnostic_pair_scores",
                "diagnostic_pair_scores",
                "diagnostics/pair_scores.json",
                "application/json",
            )
        )
    return tuple(
        ArtifactReference(
            artifact_id=f"artifact:{run_id}:{suffix}", artifact_kind=kind,
            reference=reference, producer_record_id=run_id,
            metadata={
                "mimetype": mime,
                **({"artifact_format": PAIR_SCORES_SCHEMA_ID} if suffix == "diagnostic_pair_scores" else {}),
            },
        )
        for suffix, kind, reference, mime in specs
    )


def _compact_event(event: ChangeEvent) -> ChangeEvent:
    """Project stage-local diagnostics onto the approved compact primary schema."""

    source = event.evidence.details
    details: dict[str, object] = {"predicate": event.kind.value}
    if "from_centroid_norm" in source:
        details["normalized_from_center"] = source["from_centroid_norm"]
    if "to_centroid_norm" in source:
        details["normalized_to_center"] = source["to_centroid_norm"]
    if "position_threshold_norm" in source:
        details["position_threshold"] = source["position_threshold_norm"]
    flags = source.get("severe_quality_flags", ())
    if flags:
        details["reason_codes"] = tuple(flags)
    return replace(event, evidence=replace(event.evidence, details=details))


def _build_model_provenance(config: RepresentationConfig) -> tuple[ModelProvenance, ...]:
    if not isinstance(config, DinoV2RepresentationConfig):
        return ()
    return (ModelProvenance(
        model_id=DINO_MODEL_NAME, model_version=DINO_MODEL_VERSION,
        source_fingerprint=Fingerprint(algorithm="sha256", value=config.expected_source_tree_fingerprint),
        checkpoint_fingerprint=Fingerprint(algorithm="sha256", value=config.expected_checkpoint_sha256),
        metadata={"checkpoint_size_bytes": config.expected_checkpoint_size_bytes},
    ),)


def _run_analysis(request: AnalyzeRequest, started: datetime, started_clock: float) -> AnalyzeOutcome:
    """Execute F03-F12 once and persist the mandatory primary run artifacts."""

    analysis_config = request.config.analysis_config
    run_digest = semantic_config_digest(analysis_config)
    input_producer = ProducerProvenance(
        producer_stage="stream_input", producer_version="1.0.0",
        config_version=analysis_config.stream_input.config_version,
        config_digest=semantic_config_digest(analysis_config.stream_input),
    )
    decoded = load_decoded_stream(ManifestLoadRequest(
        stream_root=request.stream_directory, producer=input_producer,
    ))
    snapshot = extract_candidates(decoded, request.config.candidate_extraction)

    if isinstance(request.config.representation, HandcraftedRepresentationConfig):
        representation_batch = build_handcrafted_representations(decoded, snapshot, request.config.representation)
        scorer = HandcraftedCanonicalScorer(request.config.representation)
        resolved_device = "cpu"
    else:
        assets = request.dinov2_assets
        assert assets is not None
        provider = LocalDinoV2Provider(
            source_dir=assets.source_directory,
            checkpoint_path=assets.checkpoint_path,
            expected_checkpoint_sha256=request.config.representation.expected_checkpoint_sha256,
            expected_source_tree_fingerprint=request.config.representation.expected_source_tree_fingerprint,
            expected_checkpoint_size_bytes=request.config.representation.expected_checkpoint_size_bytes,
            device_policy=request.config.representation.device_policy,
            batch_size=request.config.representation.batch_size,
        )
        representation_batch = build_dinov2_representations(decoded, snapshot, provider, request.config.representation)
        scorer = DinoV2CosineScorer()
        resolved_device = representation_batch.resolved_device or "unresolved"

    candidates = snapshot.result.candidates
    representations = representation_batch.records
    matching_batches = []
    for left, right in zip(decoded.frames, decoded.frames[1:]):
        pair = FramePair(
            from_frame_id=left.frame_id, from_frame_index=left.record.index,
            from_frame_size=left.image_size, to_frame_id=right.frame_id,
            to_frame_index=right.record.index, to_frame_size=right.image_size,
        )
        matching_batches.append(match_neighboring_frames(
            stream_id=decoded.stream.stream_id, frame_pair=pair,
            from_candidates=tuple(item for item in candidates if item.frame_id == left.frame_id),
            to_candidates=tuple(item for item in candidates if item.frame_id == right.frame_id),
            representations=representations,
            representation_variant_id=request.config.representation.variant,
            scorer=scorer, scoring_config=request.config.scoring,
            matching_config=request.config.matching,
        ))
    matching_results = tuple(batch.result for batch in matching_batches)
    grouping = group_recurring_visual_types(
        stream_id=decoded.stream.stream_id, candidates=candidates,
        representations=representations, matching_results=matching_results,
        representation_variant_id=request.config.representation.variant,
        scorer=scorer, config=request.config.grouping,
    )
    event_batch = build_change_events(
        stream_id=decoded.stream.stream_id, candidates=candidates,
        grouping=grouping, matching_results=matching_results,
        config=request.config.events,
    )

    warnings = list(snapshot.result.warnings) + list(representation_batch.warnings)
    errors = list(snapshot.result.errors) + list(representation_batch.errors)
    for batch in matching_batches:
        warnings.extend(batch.warnings)
        errors.extend(batch.errors)
    warnings.extend(event_batch.warnings)
    errors.extend(event_batch.errors)
    invalid_records = sum(record.envelope.validity_status is ValidityStatus.INVALID for record in representations)
    withheld = sum(item.emission_status is EmissionStatus.WITHHELD for item in event_batch.comparisons)
    if errors or invalid_records or withheld:
        run_status = RunStatus.PARTIAL
    elif warnings:
        run_status = RunStatus.COMPLETED_WITH_WARNINGS
    else:
        run_status = RunStatus.COMPLETED

    candidate_payload = build_candidate_manifest_payload(snapshot.result)
    candidate_digest = candidate_payload["candidate_snapshot_digest"]["value"]
    data = DataProvenance(
        manifest_digest=decoded.data_provenance.manifest_digest,
        frame_content_digests=decoded.data_provenance.frame_content_digests,
        candidate_snapshot_digest=Fingerprint(algorithm="sha256", value=candidate_digest),
    )
    elapsed_ms = (perf_counter() - started_clock) * 1000.0
    runtime = RuntimeMetadata(runtime_id="primary_analyze_runtime_v1", details={
        "requested_device": request.environment_provenance.requested_device,
        "resolved_device": resolved_device, "dtype": request.environment_provenance.dtype,
        "deterministic": request.environment_provenance.determinism_enabled,
        "duration_ms": elapsed_ms, "frame_count": len(decoded.frames),
        "candidate_count": len(candidates), "event_count": len(event_batch.events),
    })
    producer = ProducerProvenance(
        producer_stage="orchestration", producer_version=PIPELINE_VERSION,
        config_version=analysis_config.config_version, config_digest=run_digest,
    )
    primary_events = tuple(_compact_event(event) for event in event_batch.events)
    include_pair_scores = request.config.diagnostic_level == "standard"
    result = StreamAnalysisResult(
        envelope=RecordEnvelope(
            record_id=request.run_id, schema_version="stream-analysis-result-1.0",
            stream_id=decoded.stream.stream_id, producer=producer, context=StageContext(),
            validity_status=ValidityStatus.INVALID if run_status is RunStatus.PARTIAL else ValidityStatus.VALID,
            warning_ids=tuple(item.record_id for item in warnings),
            error_ids=tuple(item.record_id for item in errors),
        ),
        run_id=request.run_id, pipeline_version=PIPELINE_VERSION, run_status=run_status,
        frame_ids=tuple(frame.frame_id for frame in decoded.frames),
        data_provenance=VersionedMetadata(identifier="stream_data", version="1.0", details={
            "manifest_digest": data.manifest_digest.value,
            "candidate_count": len(candidates), "frame_count": len(decoded.frames),
        }),
        model_provenance=tuple(
            VersionedMetadata(identifier=model.model_id, version=model.model_version, details={
                "model_name": model.model_id, "model_version": model.model_version,
                "checkpoint_digest": model.checkpoint_fingerprint.value,
            }) for model in _build_model_provenance(request.config.representation)
        ),
        runtime_summary=runtime,
        candidate_extraction_result_id=snapshot.result.envelope.record_id,
        candidate_record_ids=tuple(item.candidate_id for item in candidates),
        representation_summaries=tuple(_representation_summary(item) for item in representations),
        frame_matching_result_ids=tuple(item.envelope.record_id for item in matching_results),
        grouping_result_id=grouping.envelope.record_id,
        recurring_type_ids=tuple(item.type_id for item in grouping.recurring_types),
        type_assignment_ids=tuple(item.assignment_id for item in grouping.assignments),
        frame_comparisons=event_batch.comparisons, change_events=primary_events,
        artifacts=_artifact_references(request.run_id, include_pair_scores=include_pair_scores),
        status_summary={"status": run_status.value, "warning_count": len(warnings),
                        "error_count": len(errors), "frame_count": len(decoded.frames),
                        "candidate_count": len(candidates), "event_count": len(event_batch.events)},
    )
    finished = datetime.now(timezone.utc)
    resolved_environment = replace(
        request.environment_provenance, resolved_device=resolved_device
    )
    output_schema_versions = {
        "run_manifest": RUN_MANIFEST_SCHEMA_ID,
        "candidate_manifest": CANDIDATE_MANIFEST_SCHEMA_ID,
        "stream_analysis": PRIMARY_STREAM_RESULT_SCHEMA_ID,
        "runtime_status": RUNTIME_STATUS_SCHEMA_ID,
    }
    if include_pair_scores:
        output_schema_versions["diagnostic_pair_scores"] = PAIR_SCORES_SCHEMA_ID
    provenance = AnalyzeRunProvenance(
        run_id=request.run_id, role=RunRole.PRIMARY_ANALYZE,
        timestamps=RunTimestamps(started_at=started.isoformat(), finished_at=finished.isoformat()),
        source=request.source_provenance, data=data,
        stage_configs=tuple(_fingerprint_stage(item) for item in analysis_config.stage_configs),
        models=_build_model_provenance(request.config.representation),
        environment=resolved_environment,
        output_schema_versions=output_schema_versions,
    )
    overlays = render_primary_overlays(decoded, candidates, grouping, primary_events)
    pair_scores_payload = None
    if include_pair_scores:
        pair_scores_payload = build_pair_scores_payload(
            run_id=request.run_id,
            stream_id=decoded.stream.stream_id,
            analysis_config_digest=run_digest,
            scoring_config_digest=semantic_config_digest(analysis_config.scorer),
            representation_variant_id=request.config.representation.variant,
            scorer_id=request.config.scoring.scorer_id,
            scorer_version=request.config.scoring.scorer_version,
            matching_results=matching_results,
        )
    artifacts = write_run_artifacts(
        output_root=request.output_root, run_id=request.run_id,
        provenance=provenance, candidate_result=snapshot.result, result=result,
        warnings=tuple(warnings), errors=tuple(errors),
        report_text=build_text_report(result), overlays=overlays,
        pair_scores_payload=pair_scores_payload,
    )
    return AnalyzeOutcome(result=result, provenance=provenance,
                          candidate_snapshot=snapshot, artifacts=artifacts)


def run_analysis(request: AnalyzeRequest) -> AnalyzeOutcome:
    """Execute one run and preserve a failed lifecycle for fatal stage errors."""

    started = datetime.now(timezone.utc)
    started_clock = perf_counter()
    target = Path(request.output_root).resolve() / "runs" / request.run_id
    if target.exists():
        from .reporting import OutputCollisionError
        raise OutputCollisionError(f"Run directory already exists: {target}")
    try:
        return _run_analysis(request, started, started_clock)
    except AnalyzeRunError:
        raise
    except Exception as error:
        from .input import StreamInputError
        from .reporting import OutputCollisionError
        if isinstance(error, OutputCollisionError):
            raise
        code = "INPUT_FAILED" if isinstance(error, StreamInputError) else "ANALYZE_FAILED"
        message = str(error).strip() or type(error).__name__
        finished = datetime.now(timezone.utc)
        artifacts = write_failed_run_artifacts(
            output_root=request.output_root, run_id=request.run_id,
            started_at=started.isoformat(), finished_at=finished.isoformat(),
            error_code=code, message=message,
            requested_provenance={
                "pipeline_version": PIPELINE_VERSION,
                "analysis_config_digest": semantic_config_digest(request.config.analysis_config),
                "source": request.source_provenance,
                "environment": request.environment_provenance,
            },
        )
        raise AnalyzeRunError(code, message, artifacts) from error


__all__ = [
    "PIPELINE_VERSION", "AnalyzeOutcome", "AnalyzePipelineConfig", "AnalyzeRequest",
    "AnalyzeRunError",
    "DinoV2AssetPaths", "run_analysis",
]
