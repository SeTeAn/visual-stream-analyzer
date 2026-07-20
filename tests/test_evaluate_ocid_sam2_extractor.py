from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stream_analysis.contracts import BBox, ImageSize
from tools.evaluate_ocid_sam2_extractor import (
    DEFAULT_PROFILES,
    FilterProfile,
    GeneratorConfig,
    QualitySetting,
    _parser,
    predictions_for_profile,
    raw_predictions_from_output,
    run_benchmark,
)


class Sam2ExtractorGateTest(unittest.TestCase):
    def test_profile_and_generator_validation(self) -> None:
        with self.assertRaises(ValueError):
            QualitySetting("", 0.8, 0.9)
        with self.assertRaises(ValueError):
            QualitySetting("bad", 1.1, 0.9)
        with self.assertRaises(ValueError):
            FilterProfile("bad", "quality", 0.5, 0.5)
        with self.assertRaises(ValueError):
            GeneratorConfig(points_per_crop=0)
        with self.assertRaises(ValueError):
            GeneratorConfig(crops_nms_threshold=1.1)

    def test_mask_normalization_clips_box_and_filters_by_actual_area(self) -> None:
        size = ImageSize(width=10, height=8)
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:4, 3:6] = True
        rows = raw_predictions_from_output(
            {
                "masks": [mask],
                "scores": [np.array(0.91)],
                "bounding_boxes": [[-2.1, 1.2, 12.3, 4.0]],
            },
            quality_id="quality",
            frame_id="frame_001",
            frame_index=4,
            image_size=size,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].bbox, BBox(0, 1, 10, 3))
        self.assertEqual(rows[0].mask_area, 6)
        self.assertEqual(rows[0].model_bbox, BBox(0, 1, 10, 3))

        kept = predictions_for_profile(
            rows,
            profile=FilterProfile("gate", "quality", 0.07, 0.08),
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].candidate_id, "sam2:gate:frame_001:000")
        dropped = predictions_for_profile(
            rows,
            profile=FilterProfile("gate", "quality", 0.08, 0.20),
        )
        self.assertEqual(dropped, ())
        other_quality = predictions_for_profile(
            rows,
            profile=FilterProfile("other", "other_quality", 0.01),
        )
        self.assertEqual(other_quality, ())

    def test_hash_mismatch_rejects_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_directory = root / "model"
            model_directory.mkdir()
            (model_directory / "model.safetensors").write_bytes(b"weights")
            stream = root / "stream"
            stream.mkdir()
            (stream / "manifest.json").write_text(
                json.dumps({"metadata": {"role": "development"}}), encoding="utf-8"
            )
            with (
                patch("tools.evaluate_ocid_sam2_extractor.validate_development_input_pairs", return_value=()),
                patch("tools.evaluate_ocid_sam2_extractor._load_mask_pipeline") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    run_benchmark(
                        stream_directories=(stream,),
                        annotation_paths=(root / "annotation.json",),
                        model_directory=model_directory,
                        expected_model_sha256="0" * 64,
                        revision="fixed",
                        quality_settings=(QualitySetting("quality", 0.8, 0.9),),
                        profiles=(FilterProfile("gate", "quality", 0.01),),
                        generator_config=GeneratorConfig(),
                        device="cpu",
                        output_path=root / "result.json",
                    )
            loader.assert_not_called()

    def test_exact_development_guard_rejects_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = root / "stream"
            stream.mkdir()
            model_directory = root / "model"
            model_directory.mkdir()
            weights = model_directory / "model.safetensors"
            weights.write_bytes(b"weights")
            with (
                patch(
                    "tools.evaluate_ocid_sam2_extractor.validate_development_input_pairs",
                    side_effect=ValueError("canonical development inputs required"),
                ) as guard,
                patch("tools.evaluate_ocid_sam2_extractor._load_mask_pipeline") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "canonical development"):
                    run_benchmark(
                        stream_directories=(stream,),
                        annotation_paths=(root / "annotation.json",),
                        model_directory=model_directory,
                        expected_model_sha256=hashlib.sha256(b"weights").hexdigest(),
                        revision="fixed",
                        quality_settings=(QualitySetting("quality", 0.8, 0.9),),
                        profiles=(FilterProfile("gate", "quality", 0.01),),
                        generator_config=GeneratorConfig(),
                        device="cpu",
                        output_path=root / "result.json",
                    )
            guard.assert_called_once()
            loader.assert_not_called()

    def test_iou_grid_schema_and_annotation_boundary_with_fakes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_directory = root / "model"
            model_directory.mkdir()
            weights = model_directory / "model.safetensors"
            weights.write_bytes(b"weights")
            model_hash = hashlib.sha256(b"weights").hexdigest()
            stream_directory = root / "stream"
            stream_directory.mkdir()
            (stream_directory / "manifest.json").write_text(
                json.dumps({"metadata": {"role": "development"}}), encoding="utf-8"
            )
            annotation_path = root / "annotation.json"
            annotation_path.write_text("{}", encoding="utf-8")
            events: list[str] = []
            frame = SimpleNamespace(
                frame_id="frame_001",
                image_size=ImageSize(width=4, height=3),
                rgb_bytes=bytes([0, 0, 0] * 12),
            )
            decoded = SimpleNamespace(
                stream=SimpleNamespace(stream_id="stream-a"), frames=(frame,)
            )

            generator_calls: list[dict[str, object]] = []

            def fake_generator(_image, **kwargs):
                self.assertEqual(type(_image).__name__, "Image")
                events.append("inference")
                generator_calls.append(kwargs)
                mask = np.zeros((3, 4), dtype=bool)
                mask[0:2, 0:2] = True
                return {
                    "masks": [mask],
                    "scores": [0.95],
                    "bounding_boxes": [[0, 0, 2, 2]],
                }

            def fake_annotation(*_args, **_kwargs):
                events.append("annotation")
                return SimpleNamespace(stream_id="stream-a")

            calls: list[float] = []

            def fake_evaluate(_annotation, candidates, *, iou_threshold):
                calls.append(iou_threshold)
                return {"candidate_count": len(candidates), "iou": iou_threshold}

            with (
                patch("tools.evaluate_ocid_sam2_extractor._load_mask_pipeline", return_value=(
                    fake_generator,
                    SimpleNamespace(__version__="torch-test"),
                    SimpleNamespace(__version__="transformers-test"),
                )),
                patch("tools.evaluate_ocid_sam2_extractor._load_stream", return_value=decoded),
                patch(
                    "tools.evaluate_ocid_sam2_extractor.validate_development_input_pairs",
                    return_value=(SimpleNamespace(stream_id="stream-a"),),
                ),
                patch("tools.evaluate_ocid_sam2_extractor.load_annotation", side_effect=fake_annotation),
                patch("tools.evaluate_ocid_sam2_extractor.evaluate_candidate_predictions", side_effect=fake_evaluate),
                patch("tools.evaluate_ocid_sam2_extractor.aggregate_candidate_metrics", side_effect=lambda values: {"count": len(values)}),
                patch("tools.evaluate_ocid_sam2_extractor._process_rss_bytes", return_value=123),
            ):
                payload = run_benchmark(
                    stream_directories=(stream_directory,),
                    annotation_paths=(annotation_path,),
                    model_directory=model_directory,
                    expected_model_sha256=model_hash,
                    revision="de431c4043854a71d8101e17995dfe596bf101a5",
                    quality_settings=(
                        QualitySetting("permissive", 0.75, 0.85),
                        QualitySetting("balanced", 0.85, 0.90),
                        QualitySetting("official", 0.88, 0.95),
                    ),
                    profiles=DEFAULT_PROFILES,
                    generator_config=GeneratorConfig(points_per_crop=2, points_per_batch=2),
                    iou_thresholds=(0.50, 0.70, 0.90),
                    device="cpu",
                    output_path=root / "result.json",
                )

            self.assertEqual(calls, [0.50, 0.70, 0.90] * 9)
            self.assertEqual(len(generator_calls), 3)
            self.assertEqual(
                [call["pred_iou_thresh"] for call in generator_calls], [0.75, 0.85, 0.88]
            )
            self.assertEqual(events, ["inference", "inference", "inference", "annotation"])
            self.assertEqual(payload["iou_thresholds"], [0.50, 0.70, 0.90])
            self.assertEqual(len(payload["quality_settings"]), 3)
            self.assertEqual(len(payload["profiles"]), 9)
            self.assertEqual(payload["streams"]["stream-a"]["runtime"]["quality_frame_calls"], 3)
            self.assertEqual(payload["model"]["dtype"], "float32")
            self.assertTrue(payload["model"]["local_files_only"])
            self.assertEqual(payload["gate_c1_development_contract"]["stream_ids"], ["stream-a"])
            output = json.loads((root / "result.json").read_text(encoding="utf-8"))
            rows = output["streams"]["stream-a"]["raw_predictions"]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["quality_id"] for row in rows}, {"permissive", "balanced", "official"})

    def test_parser_does_not_load_model(self) -> None:
        parser = _parser()
        args = parser.parse_args(
            [
                "--stream-directory", "stream",
                "--annotation", "annotation.json",
                "--model-directory", "model",
                "--expected-model-sha256", "a" * 64,
                "--revision", "fixed",
                "--benchmark-spec", "benchmark.json",
                "--reviewed-root", "reviewed",
                "--output", "result.json",
            ]
        )
        self.assertEqual(args.points_per_crop, 16)
        self.assertEqual(args.benchmark_spec, Path("benchmark.json"))


if __name__ == "__main__":
    unittest.main()
