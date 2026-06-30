"""Strict JSON adapter for the typed F13-F14 integration configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from .candidates import (
    BackgroundModelConfig, CandidateExtractionConfig, HysteresisMaskConfig,
    MorphologyCleanupConfig,
)
from .matching import EventConfig, GroupingConfig, MatchingConfig, PairScoringConfig
from .orchestration import AnalyzePipelineConfig
from .provenance import EnvironmentProvenance, Fingerprint, SourceProvenance
from .representations import DinoV2RepresentationConfig, HandcraftedRepresentationConfig

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LoadedAnalyzeConfiguration:
    pipeline: AnalyzePipelineConfig
    source: SourceProvenance
    environment: EnvironmentProvenance


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings.")
    return dict(value)


def _construct(cls: type[T], values: Any, path: str, *, tuple_fields: tuple[str, ...] = ()) -> T:
    mapping = _object(values, path)
    allowed = {item.name for item in fields(cls)}
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(sorted(unknown))}.")
    for name in tuple_fields:
        if name in mapping:
            if not isinstance(mapping[name], list):
                raise TypeError(f"{path}.{name} must be a JSON array.")
            mapping[name] = tuple(mapping[name])
    return cls(**mapping)


def load_analyze_configuration(path: Path) -> LoadedAnalyzeConfiguration:
    """Load a JSON file without applying integration-owned numeric presets."""

    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    root = _object(document, "config")
    allowed = {
        "config_version", "diagnostic_level", "candidate_extraction",
        "representation", "scoring", "matching", "grouping", "events",
        "source_provenance", "environment_provenance",
    }
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"config contains unknown fields: {', '.join(sorted(unknown))}.")
    required = allowed - {"config_version", "diagnostic_level"}
    missing = required - set(root)
    if missing:
        raise ValueError(f"config is missing fields: {', '.join(sorted(missing))}.")

    candidate_values = _object(root["candidate_extraction"], "candidate_extraction")
    if "background" in candidate_values:
        candidate_values["background"] = _construct(BackgroundModelConfig, candidate_values["background"], "candidate_extraction.background")
    if "hysteresis" in candidate_values:
        candidate_values["hysteresis"] = _construct(HysteresisMaskConfig, candidate_values["hysteresis"], "candidate_extraction.hysteresis")
    if "morphology" in candidate_values:
        candidate_values["morphology"] = _construct(MorphologyCleanupConfig, candidate_values["morphology"], "candidate_extraction.morphology")
    candidate = _construct(CandidateExtractionConfig, candidate_values, "candidate_extraction")

    representation_object = _object(root["representation"], "representation")
    if set(representation_object) != {"family", "variant", "parameters"}:
        raise ValueError("representation requires exactly family, variant and parameters.")
    parameters = _object(representation_object["parameters"], "representation.parameters")
    if "variant" in parameters:
        raise ValueError("representation variant must be declared only once.")
    parameters["variant"] = representation_object["variant"]
    family = representation_object["family"]
    if family == "handcrafted":
        representation = _construct(
            HandcraftedRepresentationConfig, parameters, "representation.parameters",
            tuple_fields=("shape_component_weights", "structure_component_weights", "mask_group_weights", "bbox_group_weights"),
        )
    elif family == "dinov2":
        representation = _construct(
            DinoV2RepresentationConfig, parameters, "representation.parameters",
            tuple_fields=("neutral_rgb", "imagenet_mean", "imagenet_std"),
        )
    else:
        raise ValueError("representation.family must be handcrafted or dinov2.")

    scoring = _construct(PairScoringConfig, root["scoring"], "scoring")
    matching = _construct(MatchingConfig, root["matching"], "matching", tuple_fields=("severe_quality_flags",))
    grouping = _construct(GroupingConfig, root["grouping"], "grouping", tuple_fields=("severe_quality_flags",))
    events = _construct(EventConfig, root["events"], "events", tuple_fields=("severe_quality_flags",))
    pipeline = AnalyzePipelineConfig(
        candidate_extraction=candidate, representation=representation,
        scoring=scoring, matching=matching, grouping=grouping, events=events,
        config_version=root.get("config_version", "1.0.0"),
        diagnostic_level=root.get("diagnostic_level", "none"),
    )

    source_values = _object(root["source_provenance"], "source_provenance")
    if set(source_values) != {"revision", "tree_fingerprint"}:
        raise ValueError("source_provenance requires revision and tree_fingerprint.")
    fingerprint = _construct(Fingerprint, source_values["tree_fingerprint"], "source_provenance.tree_fingerprint")
    source = SourceProvenance(revision=source_values["revision"], tree_fingerprint=fingerprint)
    environment = _construct(EnvironmentProvenance, root["environment_provenance"], "environment_provenance")
    return LoadedAnalyzeConfiguration(pipeline=pipeline, source=source, environment=environment)


__all__ = ["LoadedAnalyzeConfiguration", "load_analyze_configuration"]
