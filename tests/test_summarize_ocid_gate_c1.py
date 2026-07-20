from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ocid_gate_c1_common import IOU_GRID, load_development_inventory  # noqa: E402
from summarize_ocid_gate_c1 import GateC1ContractError, summarize_reports  # noqa: E402


SPEC = ROOT / "data" / "ocid" / "benchmark" / "ocid_candidate_benchmark_v1.json"
REVIEWED = ROOT / "data" / "ocid" / "derived" / "ocid_candidate_benchmark_v1_reviewed"
IOU_KEYS = tuple(f"{value:.2f}" for value in IOU_GRID)


class SummarizeGateC1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = load_development_inventory(SPEC, REVIEWED)

    def test_normalizes_all_four_real_report_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mask_source = directory / "mask_raw.json"
            self._write(mask_source, self._mask_raw_source())
            reports = [
                ("maskrcnn", "mask", self._write_report(directory, "mask.json", self._report("mask", {"p": self._perfect}, source_report=mask_source))),
                ("mobilesam", "mobile", self._write_report(directory, "mobile.json", self._report("mobile", {"p": self._perfect}))),
                ("sam2", "sam2", self._write_report(directory, "sam2.json", self._report("sam2", {"p": self._perfect}))),
                ("grounding_dino", "dino", self._write_report(directory, "dino.json", self._report("dino", {"p": self._perfect}))),
            ]
            result = self._summarize(reports)
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(result["inventory"]["scene_group_count"], 5)
            self.assertEqual(len(result["runs"][0]["report_sha256"]), 64)
            self.assertEqual(
                result["provenance"]["benchmark_spec"]["sha256"],
                "df26bd5bccc54b4674c1068627ffb431f26892309acdf1d114ef38e8b52d5b27",
            )
            self.assertEqual(len(result["provenance"]["implementation_files"]), 8)
            self.assertEqual(result["inventory"]["stream_count"], 10)
            for run in result["runs"]:
                at_50 = run["profiles"]["p"]["summary_by_iou"]["0.50"]
                self.assertEqual(at_50["pooled"]["f1"], 1.0)
                self.assertEqual(len(at_50["scene_groups"]), 5)
                self.assertTrue(all(len(value["stream_ids"]) == 2 for value in at_50["scene_groups"].values()))
            mask_run = next(item for item in result["runs"] if item["run_id"] == "mask")
            self.assertEqual(mask_run["resources"]["source"], "mask_nms_source_report")
            self.assertIsNotNone(mask_run["resources"]["runtime_seconds_per_frame"])

    def test_exact_stream_firewall_and_incomplete_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            incomplete = self._report("mobile", {"p": self._perfect})
            incomplete["status"] = "running"
            path = self._write_report(directory, "incomplete.json", incomplete)
            with self.assertRaisesRegex(GateC1ContractError, "not completed"):
                self._summarize([("mobile", "run", path)])

            missing = self._report("mobile", {"p": self._perfect})
            missing["streams"].pop(self.inventory[0].stream_id)
            path = self._write_report(directory, "missing.json", missing)
            with self.assertRaisesRegex(GateC1ContractError, "exactly development streams"):
                self._summarize([("mobile", "run", path)])

            extra = self._report("mobile", {"p": self._perfect})
            extra["streams"]["ocid_heldout_not_allowed"] = extra["streams"][self.inventory[0].stream_id]
            path = self._write_report(directory, "extra.json", extra)
            with self.assertRaisesRegex(GateC1ContractError, "extra="):
                self._summarize([("mobile", "run", path)])

    def test_rejects_inconsistent_profiles_iou_grid_invalid_metrics_and_duplicate_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inconsistent = self._report("mobile", {"left": self._perfect, "right": self._perfect})
            inconsistent["streams"][self.inventory[0].stream_id]["evaluations"].pop("right")
            path = self._write_report(directory, "profiles.json", inconsistent)
            with self.assertRaisesRegex(GateC1ContractError, "inconsistent profiles"):
                self._summarize([("mobile", "profiles", path)])

            incomplete_iou = self._report("sam2", {"p": self._perfect})
            incomplete_iou["streams"][self.inventory[0].stream_id]["evaluations"]["p"].pop("0.90")
            path = self._write_report(directory, "iou.json", incomplete_iou)
            with self.assertRaisesRegex(GateC1ContractError, "incomplete IoU grid"):
                self._summarize([("sam2", "iou", path)])

            invalid = self._report("dino", {"p": self._perfect})
            invalid["streams"][self.inventory[0].stream_id]["profiles"]["p"]["iou_metrics"]["0.50"]["f1"] = float("nan")
            path = self._write_report(directory, "invalid.json", invalid)
            with self.assertRaisesRegex(GateC1ContractError, "invalid f1"):
                self._summarize([("dino", "invalid", path)])

            valid = self._write_report(directory, "valid.json", self._report("mobile", {"p": self._perfect}))
            with self.assertRaisesRegex(GateC1ContractError, "run_id values must be unique"):
                self._summarize([("mobile", "duplicate", valid), ("sam2", "duplicate", valid)])

    def test_scene_pair_pooling_macro_and_worst_scene(self) -> None:
        groups = sorted({item.scene_group_id for item in self.inventory})
        weak_group = groups[0]

        def mixed(stream, _iou):
            return self._metric(48, 0, 4) if stream.scene_group_id == weak_group else self._metric(50, 0, 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_report(Path(temporary), "mixed.json", self._report("mobile", {"mixed": mixed}))
            result = self._summarize([("mobile", "mixed", path)])
        at_50 = result["runs"][0]["profiles"]["mixed"]["summary_by_iou"]["0.50"]
        self.assertAlmostEqual(at_50["scene_group_macro"]["f1"], 0.992)
        self.assertEqual(at_50["worst_scene_group"]["scene_group_id"], weak_group)
        self.assertAlmostEqual(at_50["worst_scene_group"]["metrics"]["f1"], 0.96)
        self.assertEqual(at_50["scene_groups"][weak_group]["metrics"]["tp"], 96)

    def test_within_0_02_tie_uses_worst_scene_then_reports_audit(self) -> None:
        groups = sorted({item.scene_group_id for item in self.inventory})

        def higher_primary_but_weakest(stream, _iou):
            return self._metric(48, 0, 4) if stream.scene_group_id == groups[0] else self._metric(50, 0, 0)

        def lower_primary_but_robust(_stream, _iou):
            return self._metric(49, 2, 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_report(
                Path(temporary),
                "tie.json",
                self._report("mobile", {"primary": higher_primary_but_weakest, "robust": lower_primary_but_robust}),
            )
            result = self._summarize([("mobile", "tie", path)])
        selection = result["provider_selections"][0]
        self.assertEqual(selection["winner"]["candidate_config_id"], "tie/robust")
        self.assertEqual(set(selection["tie_group_candidate_config_ids"]), {"tie/primary", "tie/robust"})
        audit = {item["candidate_config_id"]: item for item in selection["candidate_audit"]}
        self.assertTrue(audit["tie/primary"]["within_primary_tolerance"])
        self.assertEqual(audit["tie/robust"]["tie_group_rank"], 1)

    def test_deterministic_candidate_id_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reports = [
                ("same", "z_run", self._write_report(directory, "z.json", self._report("sam2", {"p": self._perfect}))),
                ("same", "a_run", self._write_report(directory, "a.json", self._report("sam2", {"p": self._perfect}))),
            ]
            result = self._summarize(reports)
        selection = result["provider_selections"][0]
        self.assertEqual(selection["winner"]["candidate_config_id"], "a_run/p")
        self.assertEqual(result["recommendation"]["recommended_run_id"], "a_run")

    def test_no_predictions_with_ground_truth_counts_as_zero_for_selection(self) -> None:
        def no_predictions(_stream, _iou):
            metric = self._metric(0, 0, 50)
            metric["precision"] = None
            metric["f1"] = None
            return metric

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_report(
                Path(temporary),
                "none.json",
                self._report("sam2", {"none": no_predictions}),
            )
            result = self._summarize([("sam2", "none", path)])

        summary = result["runs"][0]["profiles"]["none"]["summary_by_iou"]["0.50"]
        self.assertEqual(summary["scene_group_macro"]["precision"], 0.0)
        self.assertEqual(summary["scene_group_macro"]["f1"], 0.0)
        self.assertIsNone(summary["worst_scene_group"]["metrics"]["f1"])
        self.assertEqual(summary["worst_scene_group"]["selection_f1"], 0.0)
        self.assertEqual(
            result["recommendation"]["winner"]["primary_scene_group_macro_f1_at_0_50"],
            0.0,
        )

    def _summarize(self, values):
        return summarize_reports(
            provider_ids=[item[0] for item in values],
            run_ids=[item[1] for item in values],
            report_paths=[item[2] for item in values],
            benchmark_spec=SPEC,
            reviewed_root=REVIEWED,
        )

    def _report(self, schema, profiles, source_report=None):
        streams = {}
        for stream in self.inventory:
            by_profile = {
                profile_id: {iou: metric_fn(stream, iou) for iou in IOU_KEYS}
                for profile_id, metric_fn in profiles.items()
            }
            runtime = {
                "frame_count": stream.frame_count,
                "elapsed_seconds": float(stream.frame_count),
                "peak_cuda_allocated_bytes": 100,
            }
            if schema == "mask":
                streams[stream.stream_id] = {
                    "evaluations": {
                        profile_id: {"metrics_by_iou": values} for profile_id, values in by_profile.items()
                    }
                }
            elif schema in {"mobile", "sam2"}:
                streams[stream.stream_id] = {"runtime": runtime, "evaluations": by_profile}
            elif schema == "dino":
                streams[stream.stream_id] = {
                    "runtime": runtime,
                    "profiles": {
                        profile_id: {"iou_metrics": values} for profile_id, values in by_profile.items()
                    },
                }
            else:  # pragma: no cover - fixture programmer error.
                raise AssertionError(schema)
        report = {"status": "completed", "streams": streams}
        if source_report is not None:
            report["source_report"] = str(source_report.resolve())
        return report

    def _mask_raw_source(self):
        return {
            "status": "completed",
            "streams": {
                stream.stream_id: {
                    "runtime": {
                        "frame_count": stream.frame_count,
                        "elapsed_seconds": float(stream.frame_count),
                        "peak_cuda_allocated_bytes": 100,
                    }
                }
                for stream in self.inventory
            },
        }

    @staticmethod
    def _metric(tp, fp, fn):
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_per_frame": float(fp),
            "weighted_mean_tp_iou": 0.9 if tp else None,
            "tp_iou": {"count": tp, "mean": 0.9 if tp else None},
            "diagnostics": {"duplicate": 0, "split": 0, "merge": 0, "fragment": 0, "noise": fp, "miss": fn},
            "per_frame": [{}],
        }

    def _perfect(self, _stream, _iou):
        return self._metric(50, 0, 0)

    @staticmethod
    def _write(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_report(self, directory, name, payload):
        return self._write(directory / name, payload)


if __name__ == "__main__":
    unittest.main()
