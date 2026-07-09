from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stream_analysis import (
    CandidateExtractionConfig, DinoV2RepresentationConfig,
    EnvironmentProvenance, EventConfig, GroupingConfig, MatchingConfig,
    PairScoringConfig, RunStatus,
    DINO_BBOX_VARIANT, DINO_CANONICAL_SCORER_ID,
    DINO_CANONICAL_SCORER_VERSION, DINO_EMBEDDING_DIMENSION,
    DinoV2ModelSpec, DinoV2ProviderOutput,
)
from stream_analysis.orchestration import (
    AnalyzePipelineConfig, AnalyzeRequest, DinoV2AssetPaths, run_analysis,
)

try:
    from ._helpers import create_tiny_stream, source
except ImportError:
    from _helpers import create_tiny_stream, source


class _FakeProvider:
    def __init__(self, **kwargs):
        self.model_spec = DinoV2ModelSpec(
            expected_checkpoint_sha256=kwargs["expected_checkpoint_sha256"],
            expected_source_tree_fingerprint=kwargs["expected_source_tree_fingerprint"],
            expected_checkpoint_size_bytes=kwargs["expected_checkpoint_size_bytes"],
        )
        self.requested_device = kwargs["device_policy"]
        self.batch_size = kwargs["batch_size"]
        self.resolved_device = "cpu"

    def embed_batch(self, normalized_batch):
        count = len(normalized_batch)
        vectors = np.zeros((count, DINO_EMBEDDING_DIMENSION), dtype=np.float32)
        vectors[:, 0] = 1.0
        return DinoV2ProviderOutput(
            embeddings=vectors,
            runtime_details={"requested_device": self.requested_device,
                             "resolved_device": "cpu", "batch_size": self.batch_size,
                             "dtype": "float32", "provider": "test_provider"},
        )

    def provider_metadata(self):
        return {"provider": "test_provider", "requested_device": self.requested_device,
                "resolved_device": "cpu", "batch_size": self.batch_size}

    def model_metadata(self):
        return {"model_name": "dinov2_vits14", "expected_checkpoint_sha256": "a" * 64,
                "expected_source_tree_fingerprint": "b" * 64,
                "embedding_dimension": DINO_EMBEDDING_DIMENSION}


class DinoV2OrchestrationTest(unittest.TestCase):
    def test_dinov2_uses_the_same_lifecycle_and_compact_output(self) -> None:
        representation = DinoV2RepresentationConfig(
            expected_checkpoint_sha256="a" * 64,
            expected_source_tree_fingerprint="b" * 64,
            expected_checkpoint_size_bytes=123,
            variant=DINO_BBOX_VARIANT, device_policy="cpu", batch_size=2,
        )
        config = AnalyzePipelineConfig(
            candidate_extraction=CandidateExtractionConfig(), representation=representation,
            scoring=PairScoringConfig(
                scorer_id=DINO_CANONICAL_SCORER_ID,
                scorer_version=DINO_CANONICAL_SCORER_VERSION,
                visual_gate=0.8, spatial_gate=None,
            ),
            matching=MatchingConfig(
                unmatched_pair_cost=0.7, local_margin_gate=0.01,
                global_margin_gate=0.01, severe_quality_flags=(),
            ),
            grouping=GroupingConfig(
                medoid_gate=0.8, support_quantile=0.25, quantile_gate=0.8,
                support_pair_gate=0.8, support_ratio_gate=0.75, visual_gate=0.8,
                medoid_weight=0.4, quantile_weight=0.4, support_ratio_weight=0.2,
                second_best_margin=0.01, representation_variant_id=DINO_BBOX_VARIANT,
                scorer_id=DINO_CANONICAL_SCORER_ID,
                scorer_version=DINO_CANONICAL_SCORER_VERSION,
            ),
            events=EventConfig(position_threshold_norm=0.1, severe_quality_flags=()),
        )
        environment = EnvironmentProvenance(
            dependency_versions={}, environment_metadata={}, requested_device="cpu",
            resolved_device="unresolved", dtype="float32", determinism_enabled=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            request = AnalyzeRequest(
                stream_directory=stream, output_root=root / "outputs", run_id="dino_run",
                config=config, source_provenance=source(), environment_provenance=environment,
                dinov2_assets=DinoV2AssetPaths(
                    source_directory=root / "model_source",
                    checkpoint_path=root / "model.pth",
                ),
            )
            with patch("stream_analysis.orchestration.LocalDinoV2Provider", _FakeProvider):
                outcome = run_analysis(request)
            self.assertIn(outcome.result.run_status, {
                RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS, RunStatus.PARTIAL,
            })
            self.assertEqual(outcome.provenance.environment.resolved_device, "cpu")
            payload = json.loads((outcome.artifacts.run_directory / "stream_analysis.json").read_text(encoding="utf-8"))
            encoded = json.dumps(payload)
            self.assertNotIn('"embedding"', encoded)
            self.assertEqual(payload["result"]["model_provenance"][0]["identifier"], "dinov2_vits14")


if __name__ == "__main__":
    unittest.main()
