from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from stream_analysis import BBox, ImageSize
from stream_analysis.candidates import CandidateMaskRecord, candidate_mask_digest
from tools import evaluate_ocid_gate_c2_1_visual_types as subject
from tools import evaluate_ocid_gate_c2_masked_dinov2 as gate_c2


def _unit(x: float, y: float) -> np.ndarray:
    value = np.asarray((x, y), dtype=np.float32)
    return value / np.linalg.norm(value)


def _prototype(label: int, *vectors: np.ndarray) -> subject.Prototype:
    return subject._prototype("stream", label, [(f"c{index}", value) for index, value in enumerate(vectors)])


def _hypotheses(scope: str = "development_only") -> dict:
    primary = [
        ("vt_markers", (5, 19)), ("vt_pliers", (4, 8)),
        ("vt_potatoes", (14, 17)), ("vt_stone", (3, 10)),
    ]
    streams = [
        "ocid_arid20_table_bottom_seq01", "ocid_ycb10_table_bottom_mixed_seq21",
        "ocid_arid10_table_bottom_fruits_seq10", "ocid_arid10_table_top_fruits_seq10",
    ]
    values = []
    for (identifier, labels), stream in zip(primary, streams, strict=True):
        values.append({"hypothesis_id": identifier, "role": "primary_positive", "source_labels": list(labels), "stream_ids": [stream, stream.replace("_bottom_", "_top_")]})
    values.extend([
        {"hypothesis_id": "vt_challenge", "role": "challenge_positive", "source_labels": [3, 9], "stream_ids": ["ocid_arid20_floor_bottom_seq12", "ocid_arid20_floor_top_seq12"]},
        {"hypothesis_id": "vt_boxes", "role": "uncertain_diagnostic", "source_labels": [5, 6], "stream_ids": ["ocid_arid10_table_bottom_box_seq05", "ocid_arid10_table_top_box_seq05"]},
        {"hypothesis_id": "vt_lego", "role": "uncertain_diagnostic", "source_labels": [11, 12], "stream_ids": ["ocid_ycb10_table_bottom_mixed_seq21", "ocid_ycb10_table_top_mixed_seq21"]},
    ])
    return {
        "schema_version": subject.HYPOTHESIS_SCHEMA_VERSION, "benchmark_id": "bench", "status": subject.STATUS,
        "scope": scope, "heldout_access": "none", "embedding_source": {
            "variant": subject.VARIANT, "prototype_policy": "l2_normalized_mean_v1",
            "secondary_prototype_policy": "observed_medoid_v1", "similarity_policy": "shifted_cosine_v1",
            "frozen_visual_gate": .8,
        },
        "evaluation_policy": {
            "primary_case_count": 8, "primary_directed_query_count": 16,
            "support_rule": {"required_available_primary_cases": 8, "minimum_passed_primary_cases": 7,
                             "minimum_primary_hypotheses_passing_both_views": 3, "required_hard_negative_checks_passed": 2},
        },
        "hypotheses": values,
        "hard_negative_checks": [
            {"stream_id": "ocid_arid10_table_bottom_fruits_seq10", "positive_source_labels": [3, 10], "negative_source_label": 6},
            {"stream_id": "ocid_arid10_table_top_fruits_seq10", "positive_source_labels": [3, 10], "negative_source_label": 6},
        ],
    }


def _benchmark() -> dict:
    names = [
        "ARID20/table/bottom/seq01", "ARID20/table/top/seq01", "YCB10/table/bottom/mixed/seq21", "YCB10/table/top/mixed/seq21",
        "ARID10/table/bottom/fruits/seq10", "ARID10/table/top/fruits/seq10", "ARID20/floor/bottom/seq12", "ARID20/floor/top/seq12",
        "ARID10/table/bottom/box/seq05", "ARID10/table/top/box/seq05",
    ]
    return {"benchmark_id": "bench", "roles": {"development": [{"member_streams": names}]}}


def _candidate(identifier: str, frame_id: str = "frame_001") -> gate_c2.CandidateInput:
    source = type("Source", (), {"candidate_id": identifier, "frame_id": frame_id, "frame_index": 0, "score": .9, "bbox": BBox(0, 0, 4, 4)})()
    mask = np.ones((4, 4), dtype=np.bool_)
    record = CandidateMaskRecord("sha256:mask", candidate_mask_digest(mask), identifier, frame_id, source.bbox, mask)
    candidate = gate_c2._candidate_from_selected(source, stream_id="stream", size=ImageSize(4, 4), frame_record_index=0, mask=record)
    return gate_c2.CandidateInput("stream", candidate, record)


class GateC21VisualTypesTest(unittest.TestCase):
    def test_mean_primary_and_medoid_tie_break_are_deterministic(self) -> None:
        first = _unit(1, 0); second = _unit(0, 1)
        result = subject._prototype("stream", 3, [("z", first), ("a", second)])
        self.assertEqual(result.candidate_ids, ("a", "z"))
        self.assertEqual(result.medoid_candidate_id, "a")
        self.assertTrue(np.allclose(result.mean, _unit(1, 1)))
        self.assertTrue(np.allclose(result.medoid, second))

    def test_case_reports_directed_ranks_margins_and_frozen_gate(self) -> None:
        gallery = {
            3: _prototype(3, _unit(1, 0), _unit(1, .01)),
            10: _prototype(10, _unit(1, .04), _unit(1, .02)),
            6: _prototype(6, _unit(-1, 0), _unit(-1, .01)),
        }
        hypothesis = {"hypothesis_id": "pair", "role": "primary_positive", "source_labels": [3, 10]}
        result = subject._case_report(hypothesis=hypothesis, stream_id="stream", prototypes=gallery, primary_minimum=2, diagnostic_minimum=1, gallery_minimum=2)
        self.assertTrue(result["case_pass"])
        self.assertTrue(result["mean_pair_passes_frozen_gate"])
        self.assertEqual(result["directed_queries"]["003_to_010"]["counterpart_rank"], 1)
        self.assertGreater(result["directed_queries"]["010_to_003"]["counterpart_minus_hardest_other_margin"], 0)
        self.assertIn("secondary_medoid_pair_score", result)

    def test_uncertain_single_observation_is_available_but_marked_weak(self) -> None:
        gallery = {3: _prototype(3, _unit(1, 0)), 9: _prototype(9, _unit(1, .1)), 7: _prototype(7, _unit(-1, 0), _unit(-1, .1))}
        hypothesis = {"hypothesis_id": "uncertain", "role": "uncertain_diagnostic", "source_labels": [3, 9]}
        result = subject._case_report(hypothesis=hypothesis, stream_id="stream", prototypes=gallery, primary_minimum=2, diagnostic_minimum=1, gallery_minimum=2)
        self.assertTrue(result["available"])
        self.assertTrue(result["weak_evidence"])
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["directed_queries"]["003_to_009"]["counterpart_rank"], 1)

    def test_apple_hard_negative_requires_positive_to_beat_both_negative_scores(self) -> None:
        prototypes = {3: _prototype(3, _unit(1, 0), _unit(1, .01)), 10: _prototype(10, _unit(1, .04), _unit(1, .05)), 6: _prototype(6, _unit(-1, 0), _unit(-1, .01))}
        result = subject._hard_negative_report(check={"check_id": "apple", "stream_id": "stream", "positive_source_labels": [3, 10], "negative_source_label": 6}, prototypes=prototypes, gallery_minimum=2)
        self.assertTrue(result["available"])
        self.assertTrue(result["check_pass"])
        self.assertGreater(result["positive_minus_max_negative_margin"], 0)

    def test_support_uses_only_primary_and_hard_negative_structured_rules(self) -> None:
        policy = _hypotheses()["evaluation_policy"]
        primary = [{"hypothesis_id": f"h{index // 2}", "available": True, "case_pass": index != 7} for index in range(8)]
        hard = [{"check_pass": True}, {"check_pass": True}]
        result = subject._support(primary, hard, policy)
        self.assertTrue(result["supported"])
        primary[-2]["case_pass"] = False
        self.assertFalse(subject._support(primary, hard, policy)["supported"])

    def test_hypothesis_validation_rejects_non_development_before_annotations(self) -> None:
        with self.assertRaisesRegex(ValueError, "development-only"):
            subject._validate_hypotheses(_hypotheses(scope="heldout"), _benchmark())

    def test_gate_c2_manifest_rejects_wrong_scope_before_embedding_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"schema_version": gate_c2.INFERENCE_SCHEMA_VERSION, "embedding_schema_version": gate_c2.EMBEDDING_SCHEMA_VERSION, "status": "rgb_predicted_mask_inference_completed_before_ground_truth", "scope": "heldout", "heldout_access": "none", "variants": list(gate_c2.SUPPORTED_VARIANTS)}), encoding="utf-8")
            hypotheses = _hypotheses()
            hypotheses["embedding_source"]["gate_c2_inference_manifest_sha256"] = gate_c2._sha256(manifest_path)
            with self.assertRaisesRegex(ValueError, "development-only"):
                subject._validate_gate_c2_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), hypotheses, manifest_path)

    def test_false_positive_is_excluded_from_prototype_observations(self) -> None:
        first, second = _candidate("first"), _candidate("second")
        annotation = type("Annotation", (), {"raw": {"expected_element_instances": [{"source_label": 3, "visual_type_id": "physical-3"}]}})()
        stream = type("Stream", (), {"stream_id": "stream"})()
        with mock.patch.object(gate_c2, "_assignment_identity", return_value=({"first": "physical-3", "second": "fp:stream:second"}, {}, {})):
            prototypes, _items = subject._prototypes_for_stream(stream=stream, inputs=(first, second), embeddings={subject.VARIANT: np.stack([_unit(1, 0), _unit(0, 1)])}, positions=(0, 1), annotation=annotation)
        self.assertEqual(set(prototypes), {3})
        self.assertEqual(prototypes[3].candidate_ids, ("first",))

    def test_runner_rejects_bad_preregistration_before_annotation_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hypotheses = _hypotheses(scope="heldout")
            hypotheses_path = root / "hypotheses.json"; hypotheses_path.write_text(json.dumps(hypotheses), encoding="utf-8")
            benchmark_path = root / "benchmark.json"; benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")
            with mock.patch.object(subject, "load_annotation") as annotations:
                with self.assertRaisesRegex(ValueError, "development-only"):
                    subject.run_gate_c2_1(hypotheses_spec_path=hypotheses_path, gate_c2_inference_manifest_path=root / "missing.json", benchmark_spec_path=benchmark_path, reviewed_root=root / "reviewed", artifact_root=root / "artifacts", output_path=root / "report.json")
            annotations.assert_not_called()

    def test_contact_sheet_has_pixels_from_synthetic_rgb_and_predicted_masks(self) -> None:
        first = _candidate("first"); second = _candidate("second")
        case = {"hypothesis_id": "case", "stream_id": "stream", "source_labels": [3, 9], "available": True}
        frame = type("Frame", (), {"frame_id": "frame_001", "image_size": ImageSize(4, 4), "rgb_bytes": bytes([120, 10, 10] * 16)})()
        decoded = type("Decoded", (), {"frames": (frame,)})()
        stream = type("Stream", (), {"stream_directory": Path("unused")})()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate_c2, "_load_stream", return_value=decoded):
            result = subject._write_contact_sheet(cases=[case], items_by_stream={"stream": {3: [first], 9: [second]}}, inventory={"stream": stream}, artifact_root=Path(temporary))
            output = Path(temporary) / result["relative_path"]
            self.assertTrue(output.is_file())
            self.assertEqual(result["panel_count"], 1)
            self.assertEqual(result["panels"][0]["evaluation_assigned_source_labels"], [3, 9])
            self.assertNotIn("predicted_source_labels", result["panels"][0])
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertGreater(image.getbbox()[2], 0)

    def test_separation_reports_signed_gap_without_selecting_threshold(self) -> None:
        primary = [{
            "available": True,
            "primary_mean_pair_score": 0.8,
            "directed_queries": {
                "003_to_010": {"hardest_other_score": 0.9},
                "010_to_003": {"hardest_other_score": 0.7},
            },
        }]
        result = subject._separation(primary)
        self.assertAlmostEqual(result["minimum_positive_minus_maximum_competitor_gap"], -0.1)
        self.assertFalse(result["strict_interval_exists"])

    def test_help_never_runs_evaluation(self) -> None:
        with mock.patch.object(subject, "run_gate_c2_1") as runner:
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stdout(io.StringIO()):
                subject.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
