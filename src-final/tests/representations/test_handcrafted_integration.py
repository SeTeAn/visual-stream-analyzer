import unittest
from pathlib import Path

from stream_analysis import (
    CandidateExtractionConfig,
    HandcraftedRepresentationConfig,
    ManifestLoadRequest,
    ProducerProvenance,
    StageConfig,
    build_handcrafted_representations,
    extract_candidates,
    load_decoded_stream,
    semantic_config_digest,
)
from stream_analysis.representations import (
    HANDCRAFTED_BBOX_VARIANT,
    HANDCRAFTED_MASK_VARIANT,
)


def _input_producer() -> ProducerProvenance:
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


class HandcraftedIntegrationTest(unittest.TestCase):
    def test_public_package_exports(self) -> None:
        import stream_analysis
        import stream_analysis.representations as representations

        expected = (
            "HandcraftedDistanceResult",
            "HandcraftedRepresentationBatch",
            "HandcraftedRepresentationConfig",
            "build_handcrafted_representations",
            "handcrafted_distance",
            "srgb_uint8_to_normalized_cielab",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, stream_analysis.__all__)
                self.assertIn(name, representations.__all__)
                self.assertTrue(hasattr(stream_analysis, name))
                self.assertTrue(hasattr(representations, name))

    def test_read_only_smoke_all_probes_for_both_variants(self) -> None:
        producer = _input_producer()
        extraction_config = CandidateExtractionConfig()
        roots = (
            Path("src-final/data/streams/probe_01_stationery"),
            Path("src-final/data/streams/probe_02_fruits_vegetables_berries"),
            Path("src-final/data/streams/probe_03_tableware"),
            Path("src-final/data/streams/probe_04_technical_tools"),
        )
        for root in roots:
            before = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()))
            with self.subTest(stream=root.name):
                decoded = load_decoded_stream(
                    ManifestLoadRequest(stream_root=root, producer=producer)
                )
                snapshot = extract_candidates(decoded, extraction_config)
                batches = tuple(
                    build_handcrafted_representations(
                        decoded,
                        snapshot,
                        HandcraftedRepresentationConfig(variant=variant),
                    )
                    for variant in (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT)
                )
                for batch, variant in zip(
                    batches,
                    (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT),
                    strict=True,
                ):
                    self.assertEqual(batch.variant, variant)
                    self.assertEqual(len(batch.records), len(snapshot.result.candidates))
                    self.assertTrue(all(record.input_variant == variant for record in batch.records))
                    self.assertTrue(all(record.frame_id in {frame.frame_id for frame in decoded.frames} for record in batch.records))
                after = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()))
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
