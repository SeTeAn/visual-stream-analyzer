from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_analysis import BBox, ManifestLoadRequest, ProducerProvenance, load_decoded_stream
from stream_analysis.candidates import (
    MaskRCNNCandidateExtractionConfig,
    MaskRCNNFrameOutput,
    MaskRCNNPrediction,
    extract_maskrcnn_candidates,
)
from tests.integration._helpers import create_tiny_stream


def _config(*, device: str = "cpu") -> MaskRCNNCandidateExtractionConfig:
    return MaskRCNNCandidateExtractionConfig(
        expected_checkpoint_sha256="a" * 64,
        expected_checkpoint_size_bytes=123,
        expected_torchvision_version="0.26.0+cu128",
        device_policy=device,
    )


class _FakeProvider:
    def __init__(self, config: MaskRCNNCandidateExtractionConfig) -> None:
        self.config = config
        self.resolved_device = "cpu"
        self.calls = 0

    def predict(self, rgb: np.ndarray) -> MaskRCNNFrameOutput:
        self.calls += 1
        mask = np.ones((5, 6), dtype=np.bool_)
        return MaskRCNNFrameOutput(
            predictions=(
                MaskRCNNPrediction(BBox(20, 4, 4, 4), 0.8, 2, None),
                MaskRCNNPrediction(BBox(2, 3, 6, 5), 0.9, 1, mask),
            ),
            runtime_details={"inference_seconds": 0.01, "resolved_device": "cpu"},
        )

    def close(self) -> None:
        return None


class MaskRCNNCandidateExtractionTest(unittest.TestCase):
    def test_device_policy_is_runtime_only_but_score_threshold_is_semantic(self) -> None:
        cpu = _config(device="cpu")
        cuda = _config(device="cuda")
        stricter = MaskRCNNCandidateExtractionConfig(
            expected_checkpoint_sha256="a" * 64,
            expected_checkpoint_size_bytes=123,
            expected_torchvision_version="0.26.0+cu128",
            score_threshold=0.5,
            device_policy="cpu",
        )

        self.assertEqual(cpu.config_digest, cuda.config_digest)
        self.assertNotEqual(cpu.config_digest, stricter.config_digest)

    def test_fake_provider_builds_stable_mask_and_bbox_only_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stream_root = create_tiny_stream(Path(temporary) / "stream")
            decoded = load_decoded_stream(
                ManifestLoadRequest(
                    stream_root=stream_root,
                    producer=ProducerProvenance(
                        producer_stage="stream_input",
                        producer_version="1.0",
                        config_version="1.0",
                        config_digest="sha256:test-input",
                    ),
                )
            )
            config = _config()
            provider = _FakeProvider(config)

            snapshot = extract_maskrcnn_candidates(decoded, provider, config)

        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(snapshot.result.candidates), 4)
        first, second = snapshot.result.candidates[:2]
        self.assertEqual(first.candidate_id, "cand_frame_001_001")
        self.assertEqual(first.bbox, BBox(2, 3, 6, 5))
        self.assertIsNotNone(first.mask)
        self.assertEqual(snapshot.mask_for_candidate(first.candidate_id).mask.shape, (5, 6))
        self.assertEqual(second.bbox, BBox(20, 4, 4, 4))
        self.assertIsNone(second.mask)
        with self.assertRaises(KeyError):
            snapshot.mask_for_candidate(second.candidate_id)
        self.assertEqual(first.geometry.values["model_label_id"], 1)
        self.assertNotIn("label", first.candidate_id)

    def test_class_agnostic_nms_removes_overlapping_cross_label_prediction(self) -> None:
        config = _config()

        class DuplicateProvider(_FakeProvider):
            def predict(self, rgb: np.ndarray) -> MaskRCNNFrameOutput:
                return MaskRCNNFrameOutput(
                    predictions=(
                        MaskRCNNPrediction(BBox(2, 3, 10, 10), 0.9, 1),
                        MaskRCNNPrediction(BBox(3, 4, 10, 10), 0.8, 47),
                    ),
                    runtime_details={},
                )

        with tempfile.TemporaryDirectory() as temporary:
            decoded = load_decoded_stream(
                ManifestLoadRequest(
                    stream_root=create_tiny_stream(Path(temporary) / "stream"),
                    producer=ProducerProvenance(
                        producer_stage="stream_input",
                        producer_version="1.0",
                        config_version="1.0",
                        config_digest="sha256:test-input",
                    ),
                )
            )
            snapshot = extract_maskrcnn_candidates(decoded, DuplicateProvider(config), config)

        self.assertEqual(len(snapshot.result.candidates), 2)
        self.assertTrue(all(item.candidate_confidence == 0.9 for item in snapshot.result.candidates))


if __name__ == "__main__":
    unittest.main()
