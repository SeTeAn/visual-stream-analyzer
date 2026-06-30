import unittest
from pathlib import Path

import numpy as np

from stream_analysis import (
    CandidateExtractionConfig,
    DinoV2RepresentationConfig,
    ManifestLoadRequest,
    ProducerProvenance,
    StageConfig,
    ValidityStatus,
    build_dinov2_representations,
    extract_candidates,
    load_decoded_stream,
    semantic_config_digest,
)
from stream_analysis.representations import (
    DINO_BBOX_VARIANT,
    DINO_EMBEDDING_DIMENSION,
    DINO_MASK_NEUTRAL_VARIANT,
    DinoV2ModelSpec,
    DinoV2ProviderOutput,
)

try:
    from .test_handcrafted import make_fixture
except ImportError:
    from test_handcrafted import make_fixture


CHECKPOINT_HASH = "a" * 64
SOURCE_HASH = "b" * 64


class FakeDinoProvider:
    def __init__(self, *, device="cpu", batch_size=4, auto_fallback=False) -> None:
        self.model_spec = DinoV2ModelSpec(
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_source_tree_fingerprint=SOURCE_HASH,
            expected_checkpoint_size_bytes=123,
        )
        self.requested_device = "auto" if auto_fallback else device
        self.batch_size = batch_size
        self._resolved_device = "cpu" if auto_fallback else device
        self.calls = []

    @property
    def resolved_device(self):
        return self._resolved_device

    def embed_batch(self, normalized_batch):
        batch = np.asarray(normalized_batch, dtype=np.float32)
        self.calls.append(batch.copy())
        vectors = np.zeros((len(batch), DINO_EMBEDDING_DIMENSION), dtype=np.float32)
        means = batch.mean(axis=(2, 3))
        vectors[:, :3] = means
        vectors[:, 3] = 1.0
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return DinoV2ProviderOutput(
            embeddings=vectors,
            runtime_details={
                "requested_device": self.requested_device,
                "resolved_device": self._resolved_device,
                "batch_size": self.batch_size,
                "dtype": "float32",
                "provider": "fake_no_assets",
            },
            warning_code="DEVICE_FALLBACK_CPU" if self.requested_device == "auto" else None,
        )

    def provider_metadata(self):
        return {
            "provider": "fake_no_assets",
            "requested_device": self.requested_device,
            "resolved_device": self._resolved_device,
            "batch_size": self.batch_size,
        }

    def model_metadata(self):
        return {
            "model_name": "fake_dinov2_vits14",
            "expected_checkpoint_sha256": CHECKPOINT_HASH,
            "expected_source_tree_fingerprint": SOURCE_HASH,
            "embedding_dimension": DINO_EMBEDDING_DIMENSION,
        }


def config_for(provider, *, variant=DINO_BBOX_VARIANT):
    return DinoV2RepresentationConfig(
        expected_checkpoint_sha256=provider.model_spec.expected_checkpoint_sha256,
        expected_source_tree_fingerprint=provider.model_spec.expected_source_tree_fingerprint,
        expected_checkpoint_size_bytes=provider.model_spec.expected_checkpoint_size_bytes,
        variant=variant,
        device_policy=provider.requested_device,
        batch_size=provider.batch_size,
    )


def input_producer() -> ProducerProvenance:
    config = StageConfig(
        stage_id="stream_input",
        schema_id="stream_analysis.stream_input_config.v1",
        config_version="1.0",
    )
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version=config.config_version,
        config_digest=semantic_config_digest(config),
    )


class DinoV2IntegrationTest(unittest.TestCase):
    def test_public_package_exports(self) -> None:
        import stream_analysis
        import stream_analysis.representations as representations

        expected = (
            "DinoV2RepresentationConfig",
            "DinoV2RepresentationBatch",
            "LocalDinoV2Provider",
            "build_dinov2_representations",
            "preprocess_dinov2_candidate",
            "source_tree_fingerprint",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, stream_analysis.__all__)
                self.assertIn(name, representations.__all__)
                self.assertTrue(hasattr(stream_analysis, name))
                self.assertTrue(hasattr(representations, name))

    def test_semantic_digest_excludes_device_and_batch_but_includes_variant(self) -> None:
        cpu = DinoV2RepresentationConfig(
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_source_tree_fingerprint=SOURCE_HASH,
            device_policy="cpu",
            batch_size=1,
        )
        cuda = DinoV2RepresentationConfig(
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_source_tree_fingerprint=SOURCE_HASH,
            device_policy="cuda",
            batch_size=99,
        )
        mask = DinoV2RepresentationConfig(
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_source_tree_fingerprint=SOURCE_HASH,
            variant=DINO_MASK_NEUTRAL_VARIANT,
        )
        self.assertEqual(cpu.config_digest, cuda.config_digest)
        self.assertNotEqual(cpu.config_digest, mask.config_digest)
        self.assertEqual(cpu.config_digest, DinoV2RepresentationConfig(
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_source_tree_fingerprint=SOURCE_HASH,
            device_policy="cpu",
            batch_size=1,
        ).config_digest)

    def test_fake_provider_builds_contract_with_batch_order_and_lineage(self) -> None:
        mask = np.zeros((12, 16), dtype=bool)
        mask[2:10, 3:13] = True
        decoded, snapshot = make_fixture(
            mask,
            candidate_warning_ids=("warn_candidate_quality",),
            mask_warning_ids=("warn_mask_quality",),
        )
        provider = FakeDinoProvider(batch_size=1)
        config = config_for(provider, variant=DINO_MASK_NEUTRAL_VARIANT)

        batch = build_dinov2_representations(decoded, snapshot, provider, config)
        record = batch.records[0]

        self.assertIs(record.envelope.validity_status, ValidityStatus.VALID)
        self.assertEqual(record.input_variant, DINO_MASK_NEUTRAL_VARIANT)
        self.assertEqual(record.payload.embedding_dimension, DINO_EMBEDDING_DIMENSION)
        self.assertTrue(record.payload.l2_normalized)
        self.assertAlmostEqual(np.linalg.norm(record.payload.embedding), 1.0, places=6)
        self.assertEqual(record.payload.embedding, tuple(float(value) for value in record.payload.embedding))
        self.assertIn("warn_candidate_quality", record.envelope.warning_ids)
        self.assertIn("warn_mask_quality", record.envelope.warning_ids)
        self.assertEqual(record.preprocessing_metadata.details["mask_digest"], snapshot.result.candidates[0].mask.mask_digest)
        self.assertEqual(record.runtime_metadata.details["batch_index"], 0)
        self.assertEqual(record.runtime_metadata.details["batch_position"], 0)
        self.assertEqual(record.provider_metadata.details["provider"], "fake_no_assets")

    def test_invalid_required_mask_produces_invalid_record_without_provider_call(self) -> None:
        decoded, snapshot = make_fixture(
            np.ones((8, 8), dtype=bool),
            mask_validity_status=ValidityStatus.INVALID,
            mask_error_ids=("err_mask_invalid",),
        )
        provider = FakeDinoProvider(batch_size=1)
        config = config_for(provider, variant=DINO_MASK_NEUTRAL_VARIANT)

        batch = build_dinov2_representations(decoded, snapshot, provider, config)

        self.assertEqual(provider.calls, [])
        self.assertEqual(len(batch.records), 1)
        self.assertIs(batch.records[0].envelope.validity_status, ValidityStatus.INVALID)
        self.assertIsNone(batch.records[0].payload)
        self.assertEqual(batch.errors[0].code, "MASK_REQUIRED")
        self.assertEqual(batch.records[0].input_variant, DINO_MASK_NEUTRAL_VARIANT)

    def test_auto_cpu_fallback_is_an_explicit_record_warning(self) -> None:
        decoded, snapshot = make_fixture(np.ones((8, 8), dtype=bool))
        provider = FakeDinoProvider(batch_size=1, auto_fallback=True)
        batch = build_dinov2_representations(
            decoded,
            snapshot,
            provider,
            config_for(provider),
        )

        self.assertEqual(batch.resolved_device, "cpu")
        self.assertEqual([warning.code for warning in batch.warnings], ["DEVICE_FALLBACK_CPU"])
        self.assertIn(batch.warnings[0].record_id, batch.records[0].envelope.warning_ids)

    def test_read_only_smoke_all_probes_for_both_variants(self) -> None:
        producer = input_producer()
        extraction_config = CandidateExtractionConfig()
        roots = (
            Path("src-final/data/streams/probe_01_stationery"),
            Path("src-final/data/streams/probe_02_fruits_vegetables_berries"),
            Path("src-final/data/streams/probe_03_tableware"),
            Path("src-final/data/streams/probe_04_technical_tools"),
        )
        for root in roots:
            before = tuple(
                sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
            )
            decoded = load_decoded_stream(ManifestLoadRequest(stream_root=root, producer=producer))
            snapshot = extract_candidates(decoded, extraction_config)
            for variant in (DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT):
                with self.subTest(stream=root.name, variant=variant):
                    provider = FakeDinoProvider(batch_size=17)
                    batch = build_dinov2_representations(
                        decoded,
                        snapshot,
                        provider,
                        config_for(provider, variant=variant),
                    )
                    self.assertEqual(
                        tuple(record.candidate_id for record in batch.records),
                        tuple(candidate.candidate_id for candidate in snapshot.result.candidates),
                    )
                    self.assertEqual(len(batch.records), len(snapshot.result.candidates))
                    self.assertTrue(all(record.input_variant == variant for record in batch.records))
                    self.assertTrue(all(record.envelope.validity_status is ValidityStatus.VALID for record in batch.records))
                    self.assertEqual(batch.errors, ())
            after = tuple(
                sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
            )
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
