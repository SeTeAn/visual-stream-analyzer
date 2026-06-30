from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from PIL import Image

import stream_analysis
from stream_analysis import DataProvenance, ProducerProvenance
from stream_analysis.input import (
    ManifestLoadRequest,
    load_decoded_stream,
)


PROBE_NAMES = (
    "probe_01_stationery",
    "probe_02_fruits_vegetables_berries",
    "probe_03_tableware",
    "probe_04_technical_tools",
)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:f03-integration",
    )


def _probe_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "streams"


class InputIntegrationTest(unittest.TestCase):
    def test_all_four_probes_load_with_stable_rgb_frames_and_digests(self) -> None:
        for probe_name in PROBE_NAMES:
            with self.subTest(probe=probe_name):
                request = ManifestLoadRequest(
                    stream_root=_probe_root() / probe_name,
                    producer=_producer(),
                )
                first = load_decoded_stream(request)
                second = load_decoded_stream(request)

                self.assertEqual(first.manifest.stream_id, probe_name)
                self.assertEqual(len(first.frames), 10)
                self.assertEqual(
                    tuple(frame.record.index for frame in first.frames),
                    tuple(range(1, 11)),
                )
                self.assertEqual(
                    tuple(frame.frame_id for frame in first.frames),
                    tuple(f"frame_{index:03d}" for index in range(1, 11)),
                )
                self.assertEqual(first.manifest.manifest_digest, second.manifest.manifest_digest)
                self.assertEqual(
                    first.data_provenance.frame_content_digests,
                    second.data_provenance.frame_content_digests,
                )
                for frame in first.frames:
                    self.assertEqual((frame.image_size.width, frame.image_size.height), (640, 480))
                    self.assertEqual(frame.image_format, "PNG")
                    with frame.to_pillow_image() as image:
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, (640, 480))
                    expected = hashlib.sha256(frame.resolved_path.read_bytes()).hexdigest()
                    self.assertEqual(frame.raw_digest.value, expected)

    def test_invalid_annotation_file_is_not_read_by_analyze_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            with Image.new("RGB", (12, 10), (1, 2, 3)) as image:
                image.save(frames / "frame_001.png", format="PNG")
            manifest = {
                "schema_version": "stream-input-0.1",
                "stream_id": "stream_001",
                "ordering": "manifest",
                "frames": [
                    {
                        "frame_id": "frame_001",
                        "index": 1,
                        "image_path": "frames/frame_001.png",
                    }
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "annotation.json").write_bytes(b"{invalid annotation")

            decoded = load_decoded_stream(
                ManifestLoadRequest(stream_root=root, producer=_producer())
            )
            self.assertEqual(len(decoded.frames), 1)
            self.assertEqual(decoded.frames[0].image_size.width, 12)

    def test_data_provenance_has_no_annotation_and_no_candidate_snapshot(self) -> None:
        decoded = load_decoded_stream(
            ManifestLoadRequest(
                stream_root=_probe_root() / PROBE_NAMES[0],
                producer=_producer(),
            )
        )
        names = {item.name for item in fields(DataProvenance)}
        self.assertNotIn("annotation_digest", names)
        self.assertIsNone(decoded.data_provenance.candidate_snapshot_digest)
        self.assertEqual(decoded.data_provenance.manifest_digest, decoded.manifest.manifest_digest)
        self.assertEqual(
            set(decoded.data_provenance.frame_content_digests),
            {frame.frame_id for frame in decoded.frames},
        )

    def test_public_package_exports(self) -> None:
        names = (
            "STREAM_INPUT_SCHEMA_VERSION",
            "SUPPORTED_IMAGE_EXTENSIONS",
            "ManifestLoadRequest",
            "ManifestLoadResult",
            "DecodedFrame",
            "DecodedStream",
            "InputIssue",
            "InputIssueCode",
            "StreamInputError",
            "load_manifest",
            "decode_stream",
            "load_decoded_stream",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, stream_analysis.__all__)
                self.assertTrue(hasattr(stream_analysis, name))

    def test_request_requires_explicit_stream_input_producer(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream_input"):
            ManifestLoadRequest(
                stream_root=_probe_root() / PROBE_NAMES[0],
                producer=ProducerProvenance(
                    producer_stage="pipeline",
                    producer_version="1.0",
                    config_version="pipeline_v1",
                    config_digest="sha256:not-stream-input",
                ),
            )


if __name__ == "__main__":
    unittest.main()
