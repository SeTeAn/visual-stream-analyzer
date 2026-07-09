from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stream_analysis import ProducerProvenance
from stream_analysis.input import (
    STREAM_INPUT_SCHEMA_VERSION,
    InputIssueCode,
    ManifestLoadRequest,
    StreamInputError,
    load_manifest,
)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:stream-input-test",
    )


def _valid_manifest(image_path: str = "frames/frame_001.png") -> dict:
    return {
        "schema_version": STREAM_INPUT_SCHEMA_VERSION,
        "stream_id": "stream_001",
        "ordering": "manifest",
        "scene_description": "Controlled test stream.",
        "frames": [
            {
                "frame_id": "frame_001",
                "index": 1,
                "image_path": image_path,
                "notes": "Input-only note.",
                "metadata": {"source": "unit_test"},
            }
        ],
        "notes": "No annotation dependency.",
        "metadata": {"purpose": "smoke", "is_final_dataset": False},
    }


def _write_manifest(root: Path, data: object, *, bom: bool = False) -> bytes:
    encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
    raw = (b"\xef\xbb\xbf" + encoded) if bom else encoded
    (root / "manifest.json").write_bytes(raw)
    return raw


def _write_png(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (10, 20, 30) if mode == "RGB" else 10
    with Image.new(mode, (8, 6), color=color) as image:
        image.save(path, format="PNG")


class ManifestTest(unittest.TestCase):
    def _load(self, root: Path):
        return load_manifest(ManifestLoadRequest(stream_root=root, producer=_producer()))

    def test_valid_manifest_supports_bom_optional_fields_and_deterministic_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_png(root / "frames" / "frame_001.png")
            _write_png(root / "frames" / "frame_002.png")
            data = _valid_manifest()
            data["frames"] = [
                {"frame_id": "frame_002", "index": 2, "image_path": "frames/frame_002.png"},
                data["frames"][0],
            ]
            raw = _write_manifest(root, data, bom=True)
            result = self._load(root)
            self.assertEqual(tuple(frame.frame_id for frame in result.frames), ("frame_001", "frame_002"))
            self.assertEqual(result.scene_description, "Controlled test stream.")
            self.assertEqual(result.metadata["purpose"], "smoke")
            self.assertEqual(
                result.manifest_digest.value,
                hashlib.sha256(raw).hexdigest(),
            )

    def test_invalid_json_utf8_and_root_type_are_structured(self) -> None:
        cases = (
            (b"{invalid", InputIssueCode.MANIFEST_INVALID_JSON),
            (b"\xff\xfe", InputIssueCode.MANIFEST_INVALID_UTF8),
            (b"[]", InputIssueCode.MANIFEST_ROOT_NOT_OBJECT),
        )
        for raw, expected in cases:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(code=expected):
                root = Path(temporary)
                (root / "manifest.json").write_bytes(raw)
                with self.assertRaises(StreamInputError) as raised:
                    self._load(root)
                self.assertEqual(raised.exception.issues[0].code, expected)
                self.assertIsNotNone(raised.exception.issues[0].cause_type if expected is not InputIssueCode.MANIFEST_ROOT_NOT_OBJECT else "root")

    def test_missing_and_wrong_type_fields_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_manifest(root, {})
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            paths = {issue.field_path for issue in raised.exception.issues}
            self.assertTrue({"schema_version", "stream_id", "ordering", "frames"}.issubset(paths))

            _write_manifest(
                root,
                {
                    "schema_version": 1,
                    "stream_id": [],
                    "ordering": 1,
                    "frames": {},
                },
            )
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            self.assertTrue(
                all(issue.code is InputIssueCode.INVALID_FIELD_TYPE for issue in raised.exception.issues)
            )

    def test_frame_missing_and_wrong_type_fields_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _valid_manifest()
            data["frames"] = [{}, {"frame_id": 1, "index": "1", "image_path": []}]
            _write_manifest(root, data)
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            paths = {issue.field_path for issue in raised.exception.issues}
            self.assertTrue(
                {
                    "frames[0].frame_id",
                    "frames[0].index",
                    "frames[0].image_path",
                    "frames[1].frame_id",
                    "frames[1].index",
                    "frames[1].image_path",
                }.issubset(paths)
            )

    def test_stream_root_manifest_type_and_manifest_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            missing = base / "missing"
            with self.assertRaises(StreamInputError) as raised:
                self._load(missing)
            self.assertEqual(raised.exception.issues[0].code, InputIssueCode.STREAM_ROOT_NOT_FOUND)

            file_root = base / "not-a-directory"
            file_root.write_bytes(b"file")
            with self.assertRaises(StreamInputError) as raised:
                self._load(file_root)
            self.assertEqual(
                raised.exception.issues[0].code,
                InputIssueCode.STREAM_ROOT_NOT_DIRECTORY,
            )

            stream_root = base / "stream"
            stream_root.mkdir()
            (stream_root / "manifest.json").mkdir()
            with self.assertRaises(StreamInputError) as raised:
                self._load(stream_root)
            self.assertEqual(raised.exception.issues[0].code, InputIssueCode.MANIFEST_NOT_FILE)

            with self.assertRaisesRegex(ValueError, "within stream_root"):
                ManifestLoadRequest(
                    stream_root=stream_root,
                    manifest_name="../manifest.json",
                    producer=_producer(),
                )

    def test_nested_annotation_and_gt_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_png(root / "frames" / "frame_001.png")
            data = _valid_manifest()
            data["metadata"] = {
                "nested": [{"annotationDigest": "forbidden", "gtBoxes": True}]
            }
            _write_manifest(root, data)
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            issues = [
                issue
                for issue in raised.exception.issues
                if issue.code is InputIssueCode.FORBIDDEN_LABEL_FIELD
            ]
            self.assertEqual(len(issues), 2)

    def test_schema_ordering_empty_and_unknown_fields_are_rejected(self) -> None:
        mutations = (
            ("schema_version", "stream-input-9.9", InputIssueCode.UNSUPPORTED_SCHEMA_VERSION),
            ("ordering", "filename", InputIssueCode.INVALID_ORDERING),
            ("frames", [], InputIssueCode.EMPTY_FRAMES),
            ("annotation_path", "annotation.json", InputIssueCode.UNKNOWN_FIELD),
        )
        for key, value, code in mutations:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(key=key):
                root = Path(temporary)
                _write_png(root / "frames" / "frame_001.png")
                data = _valid_manifest()
                data[key] = value
                _write_manifest(root, data)
                with self.assertRaises(StreamInputError) as raised:
                    self._load(root)
                self.assertIn(code, {issue.code for issue in raised.exception.issues})

    def test_duplicate_ids_indices_and_bool_index_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.png", "b.png"):
                _write_png(root / "frames" / name)
            data = _valid_manifest("frames/a.png")
            data["frames"].append(
                {"frame_id": "frame_001", "index": 1, "image_path": "frames/b.png"}
            )
            _write_manifest(root, data)
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            codes = {issue.code for issue in raised.exception.issues}
            self.assertIn(InputIssueCode.DUPLICATE_FRAME_ID, codes)
            self.assertIn(InputIssueCode.DUPLICATE_FRAME_INDEX, codes)

            data = _valid_manifest("frames/a.png")
            data["frames"][0]["index"] = True
            _write_manifest(root, data)
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            self.assertIn(InputIssueCode.INVALID_FRAME_INDEX, {i.code for i in raised.exception.issues})

    def test_absolute_drive_and_parent_paths_are_rejected_syntactically(self) -> None:
        paths = ("/outside.png", "C:\\outside.png", "frames/../outside.png")
        for image_path in paths:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(image_path=image_path):
                root = Path(temporary)
                _write_manifest(root, _valid_manifest(image_path))
                with self.assertRaises(StreamInputError) as raised:
                    self._load(root)
                self.assertIn(InputIssueCode.INVALID_IMAGE_PATH, {i.code for i in raised.exception.issues})

    def test_missing_directory_and_unsupported_extension_are_rejected(self) -> None:
        cases = (
            ("frames/missing.png", None, InputIssueCode.IMAGE_NOT_FOUND),
            ("frames/directory.png", "directory", InputIssueCode.IMAGE_NOT_FILE),
            ("frames/frame.gif", "file", InputIssueCode.UNSUPPORTED_IMAGE_EXTENSION),
        )
        for image_path, kind, expected in cases:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(code=expected):
                root = Path(temporary)
                target = root.joinpath(*Path(image_path).parts)
                if kind == "directory":
                    target.mkdir(parents=True)
                elif kind == "file":
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"not-used")
                _write_manifest(root, _valid_manifest(image_path))
                with self.assertRaises(StreamInputError) as raised:
                    self._load(root)
                self.assertIn(expected, {issue.code for issue in raised.exception.issues})

    def test_symlink_inside_root_is_allowed_and_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "stream"
            root.mkdir()
            inside = root / "frames" / "inside.png"
            outside = base / "outside.png"
            _write_png(inside)
            _write_png(outside)
            internal_link = root / "frames" / "internal.png"
            escape_link = root / "frames" / "escape.png"
            try:
                internal_link.symlink_to(inside)
                escape_link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"Symlink creation is unavailable: {type(error).__name__}")

            _write_manifest(root, _valid_manifest("frames/internal.png"))
            self.assertEqual(self._load(root).frames[0].resolved_path, inside.resolve())
            _write_manifest(root, _valid_manifest("frames/escape.png"))
            with self.assertRaises(StreamInputError) as raised:
                self._load(root)
            self.assertIn(InputIssueCode.PATH_OUTSIDE_STREAM_ROOT, {i.code for i in raised.exception.issues})

    def test_error_order_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _valid_manifest("../outside.png")
            data["frames"].append(
                {"frame_id": "bad/id", "index": True, "image_path": "/absolute.png"}
            )
            _write_manifest(root, data)
            sequences = []
            for _ in range(3):
                with self.assertRaises(StreamInputError) as raised:
                    self._load(root)
                sequences.append(
                    tuple((issue.code, issue.field_path, issue.frame_id) for issue in raised.exception.issues)
                )
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[1], sequences[2])
            self.assertGreater(len(sequences[0]), 1)


if __name__ == "__main__":
    unittest.main()
