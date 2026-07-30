from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stream_analysis import ProducerProvenance
from stream_analysis.input import (
    InputIssueCode,
    ManifestLoadRequest,
    StreamInputError,
    decode_stream,
    load_decoded_stream,
    load_manifest,
)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:frame-tests",
    )


def _write_manifest(root: Path, paths: list[str]) -> None:
    data = {
        "schema_version": "stream-input-0.1",
        "stream_id": "stream_001",
        "ordering": "manifest",
        "frames": [
            {"frame_id": f"frame_{index:03d}", "index": index, "image_path": path}
            for index, path in enumerate(paths, start=1)
        ],
    }
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def _save(path: Path, *, mode: str = "RGB", image_format: str = "PNG") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "RGB": (11, 22, 33),
        "RGBA": (11, 22, 33, 44),
        "L": 77,
    }
    with Image.new(mode, (9, 7), color=colors[mode]) as image:
        image.save(path, format=image_format)
    return path.read_bytes()


class FrameDecodingTest(unittest.TestCase):
    def _request(self, root: Path) -> ManifestLoadRequest:
        return ManifestLoadRequest(stream_root=root, producer=_producer())

    def test_raw_digest_rgb_bytes_and_frame_records_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = _save(root / "frames" / "frame_001.PNG")
            _write_manifest(root, ["frames/frame_001.PNG"])
            decoded = load_decoded_stream(self._request(root))
            frame = decoded.frames[0]
            self.assertEqual(frame.image_format, "PNG")
            self.assertEqual(frame.image_size.width, 9)
            self.assertEqual(frame.image_size.height, 7)
            self.assertEqual(frame.raw_digest.value, hashlib.sha256(raw).hexdigest())
            self.assertEqual(frame.record.envelope.producer.config_digest, "sha256:frame-tests")
            self.assertEqual(decoded.stream.frames, (frame.record,))
            self.assertEqual(decoded.data_provenance.frame_content_digests["frame_001"], frame.raw_digest)

    def test_grayscale_and_rgba_are_fully_converted_to_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _save(root / "frames" / "gray.png", mode="L")
            _save(root / "frames" / "rgba.png", mode="RGBA")
            _write_manifest(root, ["frames/gray.png", "frames/rgba.png"])
            decoded = load_decoded_stream(self._request(root))
            for frame in decoded.frames:
                with frame.to_pillow_image() as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (9, 7))
                self.assertEqual(len(frame.rgb_bytes), 9 * 7 * 3)

    def test_jpg_jpeg_are_supported_without_hidden_exif_transformation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            exif = Image.Exif()
            exif[274] = 6
            with Image.new("RGB", (9, 7), (12, 34, 56)) as image:
                image.save(frames / "oriented.jpg", format="JPEG", exif=exif)
                image.save(frames / "plain.jpeg", format="JPEG")
            _write_manifest(root, ["frames/oriented.jpg", "frames/plain.jpeg"])
            decoded = load_decoded_stream(self._request(root))
            self.assertEqual(tuple(frame.image_format for frame in decoded.frames), ("JPEG", "JPEG"))
            self.assertEqual(
                tuple((frame.image_size.width, frame.image_size.height) for frame in decoded.frames),
                ((9, 7), (9, 7)),
            )

    def test_decoded_bytes_are_immutable_and_pillow_views_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _save(root / "frames" / "frame.png")
            _write_manifest(root, ["frames/frame.png"])
            frame = load_decoded_stream(self._request(root)).frames[0]
            with self.assertRaises(TypeError):
                frame.rgb_bytes[0] = 0  # type: ignore[index]
            with frame.to_pillow_image() as first, frame.to_pillow_image() as second:
                original = second.getpixel((0, 0))
                first.putpixel((0, 0), (255, 0, 0))
                self.assertEqual(second.getpixel((0, 0)), original)

    def test_extension_and_actual_format_must_match(self) -> None:
        cases = (
            ("frame.png", "JPEG"),
            ("frame.jpg", "PNG"),
            ("frame.jpeg", "PNG"),
        )
        for name, image_format in cases:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(name=name):
                root = Path(temporary)
                _save(root / "frames" / name, image_format=image_format)
                _write_manifest(root, [f"frames/{name}"])
                with self.assertRaises(StreamInputError) as raised:
                    load_decoded_stream(self._request(root))
                self.assertIn(InputIssueCode.IMAGE_FORMAT_MISMATCH, {i.code for i in raised.exception.issues})

    def test_corrupted_and_truncated_png_and_jpeg_are_rejected(self) -> None:
        cases: list[tuple[str, bytes]] = [("corrupt.png", b"not-an-image")]
        with tempfile.TemporaryDirectory() as source_temporary:
            source = Path(source_temporary)
            png = _save(source / "source.png", image_format="PNG")
            jpeg = _save(source / "source.jpg", image_format="JPEG")
            cases.extend(
                [
                    ("truncated.png", png[:-10]),
                    ("truncated.jpg", jpeg[:-2]),
                ]
            )
            for name, raw in cases:
                with tempfile.TemporaryDirectory() as temporary, self.subTest(name=name):
                    root = Path(temporary)
                    target = root / "frames" / name
                    target.parent.mkdir(parents=True)
                    target.write_bytes(raw)
                    _write_manifest(root, [f"frames/{name}"])
                    with self.assertRaises(StreamInputError) as raised:
                        load_decoded_stream(self._request(root))
                    self.assertIn(InputIssueCode.IMAGE_DECODE_ERROR, {i.code for i in raised.exception.issues})

    def test_file_handles_are_closed_after_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "frames" / "frame.png"
            _save(image_path)
            _write_manifest(root, ["frames/frame.png"])
            decoded = load_decoded_stream(self._request(root))
            moved_image = root / "frames" / "moved.png"
            moved_manifest = root / "moved_manifest.json"
            image_path.rename(moved_image)
            (root / "manifest.json").rename(moved_manifest)
            with decoded.frames[0].to_pillow_image() as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (9, 7))

    def test_decode_errors_are_aggregated_in_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.png", "b.jpg"):
                target = root / "frames" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"broken")
            _write_manifest(root, ["frames/a.png", "frames/b.jpg"])
            manifest = load_manifest(self._request(root))
            with self.assertRaises(StreamInputError) as raised:
                decode_stream(manifest)
            self.assertEqual(
                tuple(issue.frame_id for issue in raised.exception.issues),
                ("frame_001", "frame_002"),
            )
            self.assertTrue(
                all(issue.code is InputIssueCode.IMAGE_DECODE_ERROR for issue in raised.exception.issues)
            )


if __name__ == "__main__":
    unittest.main()
