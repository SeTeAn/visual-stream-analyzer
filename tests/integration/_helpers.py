from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from stream_analysis import (
    CandidateExtractionConfig, EnvironmentProvenance, EventConfig, Fingerprint,
    GroupingConfig, HandcraftedRepresentationConfig, MatchingConfig,
    PairScoringConfig, SourceProvenance,
    HANDCRAFTED_BBOX_VARIANT, HANDCRAFTED_CANONICAL_SCORER_ID,
    HANDCRAFTED_CANONICAL_SCORER_VERSION,
)
from stream_analysis.orchestration import AnalyzePipelineConfig


def pipeline_config(*, diagnostic_level: str = "none") -> AnalyzePipelineConfig:
    scorer_id = HANDCRAFTED_CANONICAL_SCORER_ID
    scorer_version = HANDCRAFTED_CANONICAL_SCORER_VERSION
    return AnalyzePipelineConfig(
        candidate_extraction=CandidateExtractionConfig(),
        representation=HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT),
        scoring=PairScoringConfig(
            scorer_id=scorer_id, scorer_version=scorer_version,
            visual_gate=0.8, spatial_gate=None,
        ),
        matching=MatchingConfig(
            unmatched_pair_cost=0.7, local_margin_gate=0.01,
            global_margin_gate=0.01,
            severe_quality_flags=("POSSIBLE_SPLITTING", "POSSIBLE_MERGING"),
        ),
        grouping=GroupingConfig(
            medoid_gate=0.8, support_quantile=0.25, quantile_gate=0.8,
            support_pair_gate=0.8, support_ratio_gate=0.75, visual_gate=0.8,
            medoid_weight=0.4, quantile_weight=0.4, support_ratio_weight=0.2,
            second_best_margin=0.01,
            representation_variant_id=HANDCRAFTED_BBOX_VARIANT,
            scorer_id=scorer_id, scorer_version=scorer_version,
        ),
        events=EventConfig(
            position_threshold_norm=0.1,
            severe_quality_flags=("POSSIBLE_SPLITTING", "POSSIBLE_MERGING"),
        ),
        diagnostic_level=diagnostic_level,
    )


def source() -> SourceProvenance:
    return SourceProvenance(
        revision="test-tree",
        tree_fingerprint=Fingerprint(algorithm="sha256", value="0" * 64),
    )


def environment() -> EnvironmentProvenance:
    return EnvironmentProvenance(
        dependency_versions={}, environment_metadata={}, requested_device="cpu",
        resolved_device="cpu", dtype="float64", determinism_enabled=True,
    )


def create_tiny_stream(root: Path) -> Path:
    frames = root / "frames"
    frames.mkdir(parents=True)
    for index in (1, 2):
        image = Image.new("RGB", (32, 24), "white")
        left = 6 if index == 1 else 14
        for x in range(left, left + 10):
            for y in range(7, 17):
                image.putpixel((x, y), (220, 20, 20))
        image.save(frames / f"frame_{index:03d}.png")
    manifest = {
        "schema_version": "stream-input-0.1", "stream_id": "tiny_stream",
        "ordering": "manifest",
        "frames": [
            {"frame_id": f"frame_{index:03d}", "index": index,
             "image_path": f"frames/frame_{index:03d}.png"}
            for index in (1, 2)
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def config_document(*, diagnostic_level: str = "none") -> dict:
    config = pipeline_config(diagnostic_level=diagnostic_level)
    return {
        "config_version": config.config_version,
        "diagnostic_level": config.diagnostic_level,
        "candidate_extraction": {},
        "representation": {
            "family": "handcrafted", "variant": HANDCRAFTED_BBOX_VARIANT,
            "parameters": {},
        },
        "scoring": {
            "scorer_id": config.scoring.scorer_id,
            "scorer_version": config.scoring.scorer_version,
            "visual_gate": config.scoring.visual_gate,
            "spatial_gate": config.scoring.spatial_gate,
        },
        "matching": {
            "unmatched_pair_cost": config.matching.unmatched_pair_cost,
            "local_margin_gate": config.matching.local_margin_gate,
            "global_margin_gate": config.matching.global_margin_gate,
            "severe_quality_flags": list(config.matching.severe_quality_flags),
        },
        "grouping": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in {
                "medoid_gate": config.grouping.medoid_gate,
                "support_quantile": config.grouping.support_quantile,
                "quantile_gate": config.grouping.quantile_gate,
                "support_pair_gate": config.grouping.support_pair_gate,
                "support_ratio_gate": config.grouping.support_ratio_gate,
                "visual_gate": config.grouping.visual_gate,
                "medoid_weight": config.grouping.medoid_weight,
                "quantile_weight": config.grouping.quantile_weight,
                "support_ratio_weight": config.grouping.support_ratio_weight,
                "second_best_margin": config.grouping.second_best_margin,
                "representation_variant_id": config.grouping.representation_variant_id,
                "scorer_id": config.grouping.scorer_id,
                "scorer_version": config.grouping.scorer_version,
                "severe_quality_flags": config.grouping.severe_quality_flags,
            }.items()
        },
        "events": {
            "position_threshold_norm": config.events.position_threshold_norm,
            "severe_quality_flags": list(config.events.severe_quality_flags),
        },
        "source_provenance": {
            "revision": "test-tree",
            "tree_fingerprint": {"algorithm": "sha256", "value": "0" * 64},
        },
        "environment_provenance": {
            "dependency_versions": {}, "environment_metadata": {},
            "requested_device": "cpu", "resolved_device": "cpu",
            "dtype": "float64", "determinism_enabled": True,
        },
    }
