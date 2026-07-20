from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from stream_analysis.contracts import BBox
from tools.evaluate_ocid_mobilesam_extractor import (
    FilterProfile,
    RawPrediction,
    DEFAULT_IOU_THRESHOLDS,
    predictions_for_profile,
    run_benchmark,
)


class MobileSAMExtractorGateTest(unittest.TestCase):
    def test_profile_filters_quality_and_area_without_changing_prediction_ids(self) -> None:
        profile = FilterProfile("gate", 0.8, 0.9, 0.01, 0.5)
        raw = (
            RawPrediction("frame_001", 3, 2, BBox(1, 2, 10, 10), 100, 10_000, 0.8, 0.9),
            RawPrediction("frame_001", 3, 4, BBox(2, 3, 5, 5), 25, 10_000, 0.9, 0.95),
            RawPrediction("frame_001", 3, 7, BBox(0, 0, 80, 80), 6_400, 10_000, 0.9, 0.95),
        )

        candidates = predictions_for_profile(raw, profile=profile)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, "mobilesam:frame_001:002")
        self.assertEqual(candidates[0].frame_index, 3)
        self.assertEqual(candidates[0].bbox, BBox(1, 2, 10, 10))

    def test_profile_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            FilterProfile("bad", 1.1, 0.9, 0.01, 0.5)
        with self.assertRaises(ValueError):
            FilterProfile("bad", 0.8, 0.9, 0.5, 0.5)

    def test_default_profiles_keep_the_nine_quality_area_combinations(self) -> None:
        from tools.evaluate_ocid_mobilesam_extractor import DEFAULT_PROFILES

        self.assertEqual(len(DEFAULT_PROFILES), 9)
        self.assertEqual(DEFAULT_IOU_THRESHOLDS, (0.50, 0.60, 0.70, 0.80, 0.90))

    def test_run_benchmark_nests_profiles_and_iou_grid_and_preserves_raw_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "mobile_sam.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "report.json"
            first_stream = root / "z_stream"
            second_stream = root / "a_stream"
            first_stream.mkdir()
            second_stream.mkdir()
            for stream in (first_stream, second_stream):
                (stream / "manifest.json").write_text("{}", encoding="utf-8")
            annotations = (root / "z.json", root / "a.json")
            for annotation in annotations:
                annotation.write_text("{}", encoding="utf-8")

            raw = (
                RawPrediction("frame_001", 0, 0, BBox(1, 2, 3, 4), 100, 10_000, 0.9, 0.95),
            )
            runtimes = iter(
                (
                    {
                        "frame_count": 1,
                        "elapsed_seconds": 0.4,
                        "total_seconds": 0.4,
                        "mean_seconds_per_frame": 0.4,
                        "frame_latency_seconds": {"count": 1, "mean_seconds": 0.4, "p50_seconds": 0.4, "p95_seconds": 0.4},
                        "peak_gpu_memory_allocated_bytes": None,
                        "peak_gpu_memory_reserved_bytes": None,
                        "process_rss": {"available": False, "peak_bytes": None, "final_bytes": None},
                        "frames": [{"frame_id": "frame_001", "inference_seconds": 0.4}],
                    },
                    {
                        "frame_count": 1,
                        "elapsed_seconds": 0.2,
                        "total_seconds": 0.2,
                        "mean_seconds_per_frame": 0.2,
                        "frame_latency_seconds": {"count": 1, "mean_seconds": 0.2, "p50_seconds": 0.2, "p95_seconds": 0.2},
                        "peak_gpu_memory_allocated_bytes": None,
                        "peak_gpu_memory_reserved_bytes": None,
                        "process_rss": {"available": False, "peak_bytes": None, "final_bytes": None},
                        "frames": [{"frame_id": "frame_001", "inference_seconds": 0.2}],
                    },
                )
            )
            loaded_streams = {
                first_stream: SimpleNamespace(stream=SimpleNamespace(stream_id="z")),
                second_stream: SimpleNamespace(stream=SimpleNamespace(stream_id="a")),
            }

            with (
                patch(
                    "tools.evaluate_ocid_mobilesam_extractor.validate_development_input_pairs",
                    return_value=(
                        SimpleNamespace(stream_id="a"),
                        SimpleNamespace(stream_id="z"),
                    ),
                ),
                patch("tools.evaluate_ocid_mobilesam_extractor._load_generator", return_value=object()),
                patch("tools.evaluate_ocid_mobilesam_extractor._load_stream", side_effect=lambda path: loaded_streams[path]),
                patch("tools.evaluate_ocid_mobilesam_extractor._infer_stream", side_effect=lambda *args: (raw, next(runtimes))),
                patch("tools.evaluate_ocid_mobilesam_extractor.load_annotation", side_effect=lambda path, manifest_path: SimpleNamespace(stream_id="a" if path.name == "a.json" else "z")),
                patch("tools.evaluate_ocid_mobilesam_extractor.evaluate_candidate_predictions", side_effect=lambda annotation, predictions, iou_threshold: {"iou": iou_threshold, "count": len(predictions)}),
                patch("tools.evaluate_ocid_mobilesam_extractor.aggregate_candidate_metrics", side_effect=lambda metrics: {"count": len(metrics)}),
                patch("tools.evaluate_ocid_mobilesam_extractor._module_provenance", return_value={"module": "mobile_sam", "module_version": None, "module_file": None, "module_file_sha256": None}),
                patch.dict("sys.modules", {"torch": SimpleNamespace(__version__="fake", cuda=SimpleNamespace(is_available=lambda: False)), "torchvision": SimpleNamespace(__version__="fake")}),
            ):
                payload = run_benchmark(
                    stream_directories=(first_stream, second_stream),
                    annotation_paths=annotations,
                    checkpoint_path=checkpoint,
                    expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
                    profiles=(FilterProfile("profile", 0.8, 0.9, 0.001),),
                    iou_thresholds=(0.50, 0.90),
                    points_per_side=8,
                    points_per_batch=64,
                    device="cpu",
                    output_path=output,
                )

            self.assertEqual(list(payload["streams"]), ["a", "z"])
            self.assertEqual(set(payload["streams"]["a"]["evaluations"]["profile"]), {"0.50", "0.90"})
            self.assertEqual(payload["aggregate"]["profile"]["0.50"], {"count": 2})
            self.assertEqual(payload["streams"]["a"]["raw_predictions"][0]["bbox"], {"x": 1, "y": 2, "width": 3, "height": 4})
            self.assertAlmostEqual(payload["resource_summary"]["frame_latency_seconds"]["p50_seconds"], 0.3)
            self.assertEqual(payload["development_contract"]["stream_ids"], ["a", "z"])
            self.assertIn("benchmark_spec_sha256", payload["development_contract"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], "ocid-learned-extractor-gate-0.2")

    def test_development_guard_runs_before_checkpoint_or_generator_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch(
                    "tools.evaluate_ocid_mobilesam_extractor.validate_development_input_pairs",
                    side_effect=ValueError("canonical development inputs required"),
                ) as guard,
                patch("tools.evaluate_ocid_mobilesam_extractor._load_generator") as loader,
                patch("tools.evaluate_ocid_mobilesam_extractor._sha256") as sha256,
            ):
                with self.assertRaisesRegex(ValueError, "canonical development"):
                    run_benchmark(
                        stream_directories=(root / "heldout_stream",),
                        annotation_paths=(root / "annotation.json",),
                        checkpoint_path=root / "missing-checkpoint.pt",
                        expected_checkpoint_sha256="0" * 64,
                        profiles=(FilterProfile("profile", 0.8, 0.9, 0.001),),
                        points_per_side=8,
                        points_per_batch=64,
                        device="cpu",
                        output_path=root / "report.json",
                    )
            guard.assert_called_once()
            loader.assert_not_called()
            sha256.assert_not_called()


if __name__ == "__main__":
    unittest.main()
