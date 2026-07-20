from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stream_analysis.contracts import BBox, ImageSize
from tools import evaluate_ocid_grounding_dino_extractor as grounding
from tools import evaluate_ocid_grounding_dino_hardening as subject
from tools.ocid_gate_c1_common import DevelopmentStream, GateC1ContractError


def _inventory(root: Path) -> tuple[DevelopmentStream, ...]:
    return (
        DevelopmentStream(
            stream_id="stream_a",
            scene_group_id="scene_a",
            frame_count=1,
            stream_directory=root / "stream_a",
            annotation_path=root / "annotation.json",
        ),
    )


def _selection_fixture(*, challenger_surface: int = 6, challenger_recall: float = 0.81):
    profiles = {}
    diagnostics = {}
    for prompt in (item.prompt_id for item in subject.PROMPTS if item.selectable):
        for geometry in subject.GEOMETRIES:
            profile_id = subject._profile_id(prompt, geometry.geometry_id)
            is_baseline = profile_id == "p_object__g_none"
            is_challenger = profile_id == "p_item__g_frame_080"
            recall = 0.82 if is_baseline else (challenger_recall if is_challenger else 0.75)
            f1 = 0.82 if is_baseline else (0.84 if is_challenger else 0.70)
            worst = 0.71 if is_baseline else (0.72 if is_challenger else 0.60)
            profiles[profile_id] = {
                "summary_by_iou": {
                    "0.50": {
                        "scene_group_macro": {"f1": f1, "recall": recall},
                        "worst_scene_group": {"metrics": {"f1": worst}},
                    },
                    "0.70": {"scene_group_macro": {"f1": f1 - 0.1}},
                },
                "per_stream": {
                    "stream_a": {"metrics_by_iou": {"0.50": {"recall": recall}}}
                },
            }
            diagnostics[profile_id] = {
                "surface_like_fp_count": (
                    10 if is_baseline else (challenger_surface if is_challenger else 9)
                ),
                "empty_frame_fp_count": 0,
            }
    return {"profiles": profiles}, diagnostics


class GroundingDinoHardeningTest(unittest.TestCase):
    def test_grid_is_exactly_preregistered(self) -> None:
        self.assertEqual(
            [(item.prompt_id, item.text, item.selectable) for item in subject.PROMPTS],
            [
                ("p_object", "object.", True),
                ("p_item", "item.", True),
                ("p_foreground_object", "foreground object.", True),
                ("p_background_labels_diag", "object. table. floor. wall.", False),
            ],
        )
        self.assertEqual(len([p for p in subject.PROMPTS if p.selectable]) * len(subject.GEOMETRIES), 9)
        self.assertEqual(subject.FROZEN_SCORE_THRESHOLD, 0.15)
        self.assertEqual(subject.FROZEN_NMS_IOU, 0.30)

    def test_geometry_boundaries_are_inclusive(self) -> None:
        size = ImageSize(width=100, height=100)
        frame_filter = subject.GEOMETRIES[1]
        surface_filter = subject.GEOMETRIES[2]
        self.assertTrue(subject.geometry_rejects(BBox(0, 0, 80, 100), size, frame_filter))
        self.assertFalse(subject.geometry_rejects(BBox(0, 0, 79, 100), size, frame_filter))
        self.assertTrue(subject.geometry_rejects(BBox(0, 0, 90, 62), size, surface_filter))
        self.assertFalse(subject.geometry_rejects(BBox(0, 0, 89, 62), size, surface_filter))
        self.assertFalse(subject.geometry_rejects(BBox(0, 0, 90, 60), size, surface_filter))

    def test_threshold_then_nms_then_geometry_preserve_stable_ids(self) -> None:
        raw = (
            grounding.RawPrediction("f", 0, 0, 0.90, "object", BBox(0, 0, 95, 70)),
            grounding.RawPrediction("f", 0, 1, 0.80, "object", BBox(1, 1, 95, 70)),
            grounding.RawPrediction("f", 0, 2, 0.14, "object", BBox(50, 50, 5, 5)),
            grounding.RawPrediction("f", 0, 3, 0.70, "object", BBox(70, 70, 5, 5)),
        )
        candidates, records = subject.candidates_for_profile(
            raw,
            prompt=subject.PROMPTS[0],
            geometry=subject.GEOMETRIES[2],
            image_sizes={"f": ImageSize(width=100, height=100)},
        )
        self.assertEqual([row["prediction_index"] for row in records], [0, 3])
        self.assertTrue(records[0]["geometry_rejected"])
        self.assertFalse(records[1]["geometry_rejected"])
        self.assertEqual(
            [item.candidate_id for item in candidates],
            ["grounding-dino-hardening:p_object:g_surface_span_055_090:f:003"],
        )

    def test_loads_only_exact_completed_gate_c1_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "schema_version": grounding.SCHEMA_VERSION,
                "status": "completed",
                "scope": "development_only_candidate_extraction_gate_c1",
                "prompt": "object.",
                "streams": {
                    "stream_a": {
                        "raw_predictions": [
                            {
                                "frame_id": "frame_0000",
                                "frame_index": 0,
                                "prediction_index": 0,
                                "score": 0.9,
                                "phrase": "object",
                                "bbox": {"x": 0, "y": 0, "width": 2, "height": 2},
                            }
                        ]
                    }
                },
            }
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded = subject.load_gate_c1_baseline_raw(path, _inventory(root))
            self.assertEqual(loaded["stream_a"][0].bbox, BBox(0, 0, 2, 2))
            report["scope"] = "development_only"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(GateC1ContractError, "completed Gate C1"):
                subject.load_gate_c1_baseline_raw(path, _inventory(root))

    def test_selection_uses_eligible_challenger_or_frozen_fallback(self) -> None:
        result, diagnostics = _selection_fixture(challenger_surface=6, challenger_recall=0.81)
        selected = subject.select_profile(result, diagnostics)
        self.assertEqual(selected["status"], "selected_by_preregistered_rule")
        self.assertEqual(selected["selected_profile_id"], "p_item__g_frame_080")

        result, diagnostics = _selection_fixture(challenger_surface=8, challenger_recall=0.81)
        fallback = subject.select_profile(result, diagnostics)
        self.assertEqual(fallback["status"], "fallback_to_frozen_baseline")
        self.assertEqual(fallback["selected_profile_id"], "p_object__g_none")

    def test_surface_diagnostics_scope_repeated_candidate_ids_by_stream(self) -> None:
        profile_id = "p_object__g_none"
        development_result = {
            "profiles": {
                profile_id: {
                    "per_stream": {
                        "stream_a": {
                            "metrics_by_iou": {
                                "0.50": {
                                    "false_positive_candidate_ids": ["candidate:frame_0001:000"],
                                    "per_frame": [{"gt_count": 0, "fp": 1}],
                                    "recall": 1.0,
                                }
                            }
                        },
                        "stream_b": {
                            "metrics_by_iou": {
                                "0.50": {
                                    "false_positive_candidate_ids": ["candidate:frame_0001:000"],
                                    "per_frame": [{"gt_count": 0, "fp": 1}],
                                    "recall": 1.0,
                                }
                            }
                        },
                    }
                }
            }
        }
        records = {
            profile_id: {
                "stream_a": (
                    {
                        "candidate_id": "candidate:frame_0001:000",
                        "geometry_rejected": False,
                        "surface_like": True,
                        "phrase": "table",
                    },
                ),
                "stream_b": (
                    {
                        "candidate_id": "candidate:frame_0001:000",
                        "geometry_rejected": False,
                        "surface_like": False,
                        "phrase": "object",
                    },
                ),
            }
        }
        result = subject._profile_diagnostics(development_result, records)[profile_id]
        self.assertEqual(result["surface_like_fp_count"], 1)
        self.assertEqual(result["per_stream"]["stream_a"]["surface_like_fp_count"], 1)
        self.assertEqual(result["per_stream"]["stream_b"]["surface_like_fp_count"], 0)
        self.assertEqual(result["empty_frame_fp_count"], 2)

    def test_contract_guard_precedes_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    subject,
                    "load_development_inventory",
                    side_effect=GateC1ContractError("held-out remains locked"),
                ),
                mock.patch.object(subject, "load_gate_c1_baseline_raw") as baseline,
                mock.patch.object(subject, "_sha256") as digest,
            ):
                with self.assertRaisesRegex(GateC1ContractError, "held-out"):
                    subject.run_benchmark(
                        model_directory=root,
                        expected_model_sha256="0" * 64,
                        baseline_report_path=root / "baseline.json",
                        benchmark_spec_path=root / "spec.json",
                        reviewed_root=root / "reviewed",
                        device="cpu",
                        output_path=root / "result.json",
                        selected_candidates_output_path=root / "candidates.json",
                    )
            baseline.assert_not_called()
            digest.assert_not_called()

    def test_help_does_not_run_benchmark(self) -> None:
        with mock.patch.object(subject, "run_benchmark") as runner:
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stdout(io.StringIO()):
                subject.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
