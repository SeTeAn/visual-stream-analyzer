from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from stream_analysis.contracts import BBox, ImageSize
from tools import evaluate_ocid_grounding_dino_extractor as subject


class GroundingDinoExtractorTest(unittest.TestCase):
    def test_prompt_is_fixed_to_registered_generic_value(self) -> None:
        self.assertEqual(subject._validate_prompt("object."), "object.")
        with self.assertRaisesRegex(ValueError, "configured"):
            subject._validate_prompt("marker.")

    def test_clipped_integral_bbox_rejects_invalid_geometry(self) -> None:
        size = ImageSize(width=20, height=10)
        self.assertEqual(subject._integral_box([-1.2, 1.1, 20.7, 9.01], size), BBox(0, 1, 20, 9))
        self.assertIsNone(subject._integral_box([3, 2, 3, 7], size))
        self.assertIsNone(subject._integral_box([0, 0, float("nan"), 1], size))

    def test_threshold_and_class_agnostic_nms_use_stable_ids(self) -> None:
        raw = (
            subject.RawPrediction("f", 0, 0, 0.91, "object", BBox(0, 0, 10, 10)),
            subject.RawPrediction("f", 0, 1, 0.80, "object", BBox(1, 1, 10, 10)),
            subject.RawPrediction("f", 0, 2, 0.20, "object", BBox(30, 0, 5, 5)),
        )
        none = subject.predictions_for_profile(raw, profile=subject.FilterProfile(0.15, None))
        nms = subject.predictions_for_profile(raw, profile=subject.FilterProfile(0.15, 0.5))
        high = subject.predictions_for_profile(raw, profile=subject.FilterProfile(0.85, None))
        self.assertEqual(tuple(item.candidate_id for item in none), ("grounding-dino:f:000", "grounding-dino:f:001", "grounding-dino:f:002"))
        self.assertEqual(tuple(item.candidate_id for item in nms), ("grounding-dino:f:000", "grounding-dino:f:002"))
        self.assertEqual(tuple(item.candidate_id for item in high), ("grounding-dino:f:000",))

    def test_processor_predictions_are_sorted_and_clipped(self) -> None:
        rows = subject._normalise_processor_predictions(
            {"boxes": [[5.2, 1.1, 10.0, 5.0], [-5.0, 0.0, 4.1, 3.1]], "scores": [0.5, 0.8], "labels": ["second", "first"]},
            frame_id="frame_001",
            frame_index=2,
            image_size=ImageSize(width=8, height=6),
        )
        self.assertEqual(tuple(item.prediction_index for item in rows), (0, 1))
        self.assertEqual(rows[0].phrase, "first")
        self.assertEqual(rows[0].bbox, BBox(0, 0, 5, 4))

    def test_processor_prefers_transformers_text_labels_over_legacy_labels(self) -> None:
        rows = subject._normalise_processor_predictions(
            {
                "boxes": [[1.0, 1.0, 5.0, 5.0]],
                "scores": [0.9],
                "text_labels": ["processor phrase"],
                "labels": ["legacy phrase"],
            },
            frame_id="frame_001",
            frame_index=0,
            image_size=ImageSize(width=8, height=6),
        )
        self.assertEqual(rows[0].phrase, "processor phrase")

    def test_profile_schema_evaluates_every_iou_level_without_model(self) -> None:
        raw = (subject.RawPrediction("f", 0, 0, 0.9, "object", BBox(0, 0, 10, 10)),)
        calls: list[float] = []
        def fake_evaluate(_annotation, _predictions, *, iou_threshold):
            calls.append(iou_threshold)
            return {"threshold": iou_threshold}

        with mock.patch.object(subject, "evaluate_candidate_predictions", side_effect=fake_evaluate) as evaluator:
            result = subject._profile_metrics(object(), raw)
        self.assertEqual(len(result), 20)
        self.assertEqual(set(result["score_0.15_nms_none"]["iou_metrics"]), {"0.50", "0.60", "0.70", "0.80", "0.90"})
        self.assertEqual(evaluator.call_count, 100)
        self.assertEqual(subject.IOU_LEVELS, (0.50, 0.60, 0.70, 0.80, 0.90))
        self.assertEqual(calls.count(0.50), 20)
        self.assertEqual(calls.count(0.90), 20)

    def test_model_loader_uses_explicit_local_directory_only(self) -> None:
        processor = mock.Mock()
        model = mock.Mock()
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=processor))
        transformers.AutoModelForZeroShotObjectDetection = types.SimpleNamespace(
            from_pretrained=mock.Mock(return_value=model)
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            loaded_processor, loaded_model = subject._load_model_processor(
                Path("C:/models/grounding-dino"), torch_module=mock.Mock(), device="cpu"
            )
        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        expected = {
            "local_files_only": True,
            "revision": subject.HF_REVISION,
        }
        transformers.AutoProcessor.from_pretrained.assert_called_once_with(
            "C:\\models\\grounding-dino", **expected
        )
        transformers.AutoModelForZeroShotObjectDetection.from_pretrained.assert_called_once_with(
            "C:\\models\\grounding-dino", **expected
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()

    def test_model_dtype_is_taken_from_loaded_parameters(self) -> None:
        parameter = types.SimpleNamespace(dtype="torch.float32")
        model = types.SimpleNamespace(parameters=lambda: iter((parameter,)))
        self.assertEqual(subject._model_dtype(model), "float32")

    def test_development_contract_failure_happens_before_model_hash_or_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"local-only")
            with (
                mock.patch.object(
                    subject,
                    "validate_development_input_pairs",
                    side_effect=ValueError("exact canonical development inputs required"),
                ) as validator,
                mock.patch.object(subject, "_sha256") as digest,
                mock.patch.object(subject, "_load_model_processor") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "exact canonical"):
                    subject.run_benchmark(
                        stream_directories=(root,),
                        annotation_paths=(root / "annotation.json",),
                        model_directory=root,
                        expected_model_sha256="0" * 64,
                        device="cpu",
                        output_path=root / "output.json",
                    )
            validator.assert_called_once()
            digest.assert_not_called()
            loader.assert_not_called()

    def test_exact_development_contract_is_called_with_spec_and_reviewed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = root / "ocid_arid10_table_bottom_box_seq05"
            stream.mkdir()
            annotation = root / "annotation.json"
            annotation.write_text("{}", encoding="utf-8")
            spec = root / "benchmark.json"
            spec.write_text("{}", encoding="utf-8")
            reviewed_root = root / "reviewed"
            reviewed_root.mkdir()
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"local-only")
            inventory = (
                types.SimpleNamespace(
                    stream_id="ocid_arid10_table_bottom_box_seq05",
                    stream_directory=stream,
                    annotation_path=annotation,
                ),
            )
            with (
                mock.patch.object(subject, "validate_development_input_pairs", return_value=inventory) as validator,
                mock.patch.object(subject, "_sha256", return_value="a" * 64),
                mock.patch.object(subject, "_load_model_processor", side_effect=RuntimeError("stop after guard")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after guard"):
                    subject.run_benchmark(
                        stream_directories=(stream,),
                        annotation_paths=(annotation,),
                        model_directory=root,
                        expected_model_sha256="a" * 64,
                        device="cpu",
                        output_path=root / "output.json",
                        benchmark_spec_path=spec,
                        reviewed_root=reviewed_root,
                    )
            validator.assert_called_once_with(
                (stream,),
                (annotation,),
                benchmark_spec_path=spec,
                reviewed_root=reviewed_root,
            )

    def test_aggregate_runtime_contains_pooled_latency_and_memory_peaks(self) -> None:
        result = subject._aggregate_runtime(
            (
                {
                    "elapsed_seconds": 1.0,
                    "process_rss_bytes": 100,
                    "peak_gpu_memory_allocated_bytes": 20,
                    "peak_gpu_memory_reserved_bytes": 30,
                    "frames": [{"total_inference_seconds": 0.2}, {"total_inference_seconds": 0.4}],
                },
                {
                    "elapsed_seconds": 0.5,
                    "process_rss_bytes": 120,
                    "peak_gpu_memory_allocated_bytes": 18,
                    "peak_gpu_memory_reserved_bytes": 35,
                    "frames": [{"total_inference_seconds": 0.1}],
                },
            )
        )
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["max_process_rss_bytes"], 120)
        self.assertEqual(result["max_peak_gpu_memory_allocated_bytes"], 20)
        self.assertEqual(result["max_peak_gpu_memory_reserved_bytes"], 35)
        self.assertEqual(result["latency_seconds"]["p50"], 0.2)

    def test_import_and_help_do_not_load_model(self) -> None:
        with mock.patch.object(subject, "run_benchmark") as runner:
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stdout(io.StringIO()):
                subject.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        runner.assert_not_called()
        self.assertIs(importlib.import_module("tools.evaluate_ocid_grounding_dino_extractor"), subject)


if __name__ == "__main__":
    unittest.main()
