from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GATE_B_TEST_PATH = REPOSITORY_ROOT / "tests" / "test_prepare_ocid_gate_b_benchmark.py"
REVIEW_TOOL_PATH = (
    REPOSITORY_ROOT / "tools" / "prepare_ocid_component_review_pack.py"
)


def _load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


# Reuse the already audited synthetic OCID/audit/spec fixture rather than
# weakening provenance fields for this review-pack test suite.
GATE_B_FIXTURE_MODULE = _load_module(
    "gate_b_fixture_for_component_review", GATE_B_TEST_PATH
)
TOOL = _load_module("prepare_ocid_component_review_pack", REVIEW_TOOL_PATH)


class PrepareOcidComponentReviewPackTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_type = GATE_B_FIXTURE_MODULE.PrepareOcidGateBBenchmarkTest
        self.fixture = fixture_type(methodName="test_builds_complete_atomic_rgb_only_benchmark")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.output_root = self.root / "component_review_pack"
        self.fixture.spec["source"].pop("component_decisions_sha256", None)
        self.fixture.spec["annotation_contract"].update(
            {
                "bbox_derivation": (
                    "tight_axis_aligned_bbox_from_reviewed_instance_mask_components"
                ),
                "component_review_policy": "largest_plus_reviewed_components_v1",
                "component_connectivity": 4,
                "secondary_area_ratio": 0.05,
                "review_candidate_rule": (
                    "raw_bbox_differs_from_largest_component_bbox"
                ),
            }
        )
        self.fixture.spec["review_contract"][
            "component_review_expected_observations"
        ] = 2
        self.fixture._write_json(self.fixture.spec_path, self.fixture.spec)

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _build(self, *, output_root: Path | None = None, spec_path: Path | None = None):
        return TOOL.build_component_review_pack(
            ocid_root=self.fixture.ocid_root,
            structural_audit=self.fixture.audit_path,
            spec_path=spec_path or self.fixture.spec_path,
            output_root=output_root or self.output_root,
        )

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_builds_expected_cases_and_pending_component_decision_schema(self) -> None:
        result = self._build()

        # Each of the two camera streams has one frame/label whose two corner
        # components expand the raw bbox beyond the largest-component bbox.
        self.assertEqual(result.case_count, 2)
        self.assertEqual(result.stream_count, 2)
        self.assertEqual(result.development_case_count, 2)
        self.assertEqual(result.heldout_case_count, 0)

        manifest = self._json(self.output_root / "component_review_manifest.json")
        self.assertEqual(manifest["schema_version"], "ocid-component-review-pack-2.0")

        template = self._json(self.output_root / "decision_template.json")
        self.assertEqual(
            template["schema_version"], "ocid-component-review-decisions-1.0"
        )
        self.assertEqual(template["benchmark_id"], "synthetic_ocid_gate_b")
        self.assertEqual(template["policy_id"], "largest_plus_reviewed_components_v1")
        self.assertEqual(template["review_status"], "pending")
        self.assertEqual(len(template["decisions"]), 2)

        expected_ids = {
            (
                f"{self.fixture._stream_id(source)}__frame_0001__label_258"
            )
            for source in self.fixture.SOURCE_SEQUENCES
        }
        decisions = template["decisions"]
        self.assertEqual({item["case_id"] for item in decisions}, expected_ids)
        for entry in decisions:
            self.assertEqual(entry["review_status"], "pending")
            self.assertEqual(entry["reviews"], [])
            self.assertIsNone(entry["adjudication"])
            self.assertIsNone(entry["final_component_decisions"])
            self.assertIsNone(entry["final_retained_component_ids"])
            self.assertIsNone(entry["final_confidence"])
            self.assertIsNone(entry["resolved_by"])
            self.assertEqual(entry["notes"], "")
            self.assertFalse(entry["author_attention"])
            self.assertNotIn("retained_component_ids", entry)
            self.assertNotIn("decision_code", entry)
            self.assertEqual(entry["source_mask_sha256"], entry["analysis"]["source_mask_sha256"])
            self.assertTrue(entry["analysis"]["review_required"])
            self.assertEqual(entry["analysis"]["connectivity"], 4)
            self.assertEqual(len(entry["analysis"]["components"]), 2)
            self.assertEqual(
                entry["automatic_retained_component_ids"],
                entry["analysis"]["automatic_retained_component_ids"],
            )

    def test_accepts_policy_complete_spec_without_decisions_hash(self) -> None:
        spec = self._json(self.fixture.spec_path)
        self.assertNotIn("component_decisions_sha256", spec["source"])

        result = self._build()

        self.assertEqual(result.case_count, 2)
        manifest = self._json(self.output_root / "component_review_manifest.json")
        self.assertEqual(
            manifest["policy"]["policy_id"],
            "largest_plus_reviewed_components_v1",
        )

    def test_creates_blind_and_adjudication_temporal_context_artifacts(self) -> None:
        self._build()
        manifest = self._json(self.output_root / "component_review_manifest.json")

        self.assertEqual(manifest["scope"], "evaluation_only_annotation_review")
        self.assertTrue(manifest["review_protocol"]["primary_review_is_blind"])
        self.assertEqual(manifest["review_protocol"]["primary_review_count"], 2)
        disclosures = manifest["review_protocol"]["blind_evidence_disclosures"]
        self.assertEqual(
            disclosures["schema_version"],
            "ocid-blind-evidence-disclosures-1.0",
        )
        self.assertIn("component_bboxes", disclosures["disclosed"])
        self.assertIn("exact_component_areas", disclosures["withheld"])
        self.assertIn("component_area_ratios", disclosures["withheld"])
        self.assertIn("automatic_proposal_threshold", disclosures["withheld"])
        self.assertIn("automatic_component_decisions", disclosures["withheld"])
        self.assertIn("automatic_proposal_bbox", disclosures["withheld"])
        self.assertIn(
            "Do not inspect automatic_retained_component_ids",
            manifest["review_protocol"]["primary_reviewer_instruction"],
        )
        self.assertFalse(
            manifest["ground_truth_boundary"]["model_predictions_accessed"]
        )
        self.assertFalse(manifest["ground_truth_boundary"]["metrics_computed"])

        for case in manifest["cases"]:
            blind = self.output_root / case["blind_visual_path"]
            adjudication = self.output_root / case["adjudication_visual_path"]
            self.assertTrue(blind.is_file())
            self.assertTrue(adjudication.is_file())
            with Image.open(blind) as image:
                self.assertGreaterEqual(image.width, 1600)
                self.assertGreaterEqual(image.height, 900)
                self.assertEqual(image.format, "PNG")
            with Image.open(adjudication) as image:
                self.assertEqual(image.format, "PNG")

        streams = manifest["totals"]["by_stream"]
        self.assertEqual(len(streams), 2)
        for stream in streams:
            index = self._json(self.output_root / stream["index"])
            self.assertEqual(index["case_count"], 1)
            self.assertTrue((self.output_root / index["blind_contact_sheet"]).is_file())
            self.assertTrue(
                (self.output_root / index["adjudication_contact_sheet"]).is_file()
            )

        output_parts = {
            part.casefold()
            for path in self.output_root.rglob("*")
            for part in path.relative_to(self.output_root).parts
        }
        self.assertNotIn("analysis_streams", output_parts)
        self.assertFalse(any("prediction" in part for part in output_parts))

    def test_blind_renderer_discloses_no_area_ratio_or_proposal_evidence(self) -> None:
        (
            _,
            _,
            _,
            _,
            _,
            selections,
            audit_sequences,
            pairs_by_source,
        ) = TOOL._load_validated_inputs(
            ocid_root=self.fixture.ocid_root,
            structural_audit=self.fixture.audit_path,
            spec_path=self.fixture.spec_path,
        )
        cases, _ = TOOL._collect_cases(
            selections=selections,
            audit_sequences=audit_sequences,
            pairs_by_source=pairs_by_source,
        )
        captured: list[str] = []
        original_text = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            captured.append(str(text))
            return original_text(draw, xy, text, *args, **kwargs)

        output = self.root / "blind_probe.png"
        with mock.patch.object(ImageDraw.ImageDraw, "text", new=capture):
            TOOL._render_case(cases[0], output, blind=True)

        visible_text = "\n".join(captured).casefold()
        for forbidden in ("area", "ratio", "auto", "keep", "retain", "drop", "proposal"):
            self.assertNotIn(forbidden, visible_text)
        self.assertIsNone(
            re.search(r"(?m)^c\d{3}\s+\d+\s+\(", visible_text),
            "Blind component rows must not contain a numeric area column.",
        )
        with Image.open(output) as image:
            metadata = json.dumps(image.info, sort_keys=True).casefold()
        for forbidden in ("area", "ratio", "auto", "keep", "retain", "drop", "proposal"):
            self.assertNotIn(forbidden, metadata)

    def test_adjudication_renderer_keeps_full_proposal_evidence(self) -> None:
        (
            _,
            _,
            _,
            _,
            _,
            selections,
            audit_sequences,
            pairs_by_source,
        ) = TOOL._load_validated_inputs(
            ocid_root=self.fixture.ocid_root,
            structural_audit=self.fixture.audit_path,
            spec_path=self.fixture.spec_path,
        )
        cases, _ = TOOL._collect_cases(
            selections=selections,
            audit_sequences=audit_sequences,
            pairs_by_source=pairs_by_source,
        )
        captured: list[str] = []
        original_text = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            captured.append(str(text))
            return original_text(draw, xy, text, *args, **kwargs)

        output = self.root / "adjudication_probe.png"
        with mock.patch.object(ImageDraw.ImageDraw, "text", new=capture):
            TOOL._render_case(cases[0], output, blind=False)

        visible_text = "\n".join(captured).casefold()
        for required in ("area", "ratio", "proposal", "threshold", "retain"):
            self.assertIn(required, visible_text)

    def test_four_point_eight_eight_percent_case_hides_proposal_metadata(self) -> None:
        mask = np.zeros((96, 96), dtype=bool)
        mask[4:50, 4:51] = True  # 2,162 pixels.
        mask[50, 4:36] = True  # Largest component total: 2,194.
        mask[70:80, 70:80] = True
        mask[80, 70:77] = True  # Secondary component total: 107 (4.88%).
        analysis = TOOL.analyze_component_mask(mask, secondary_area_ratio=0.05)
        self.assertEqual([item.area for item in analysis.components], [2194, 107])
        self.assertEqual(analysis.automatic_retained_component_ids, ("c001",))

        rgb_path = self.root / "four_point_eight_eight.png"
        Image.new("RGB", (96, 96), (100, 100, 100)).save(rgb_path)
        frame = TOOL.RawReviewFrame(
            frame_id="frame_0001",
            frame_index=1,
            source_filename="RGB_0001.png",
            width=96,
            height=96,
            rgb_path=rgb_path,
            labels=np.where(mask, 2, 0).astype(np.uint16),
            observations=(),
        )
        case = TOOL.ReviewCase(
            case_id="regression_4_88",
            role="development",
            stream_id="regression_stream",
            source_sequence="regression/stream",
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            source_filename=frame.source_filename,
            source_label=2,
            source_mask_sha256=analysis.source_mask_sha256,
            analysis=analysis,
            frame=frame,
            previous_frame=None,
            next_frame=None,
        )
        captured: list[str] = []
        original_text = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            captured.append(str(text))
            return original_text(draw, xy, text, *args, **kwargs)

        output = self.root / "blind_4_88.png"
        with mock.patch.object(ImageDraw.ImageDraw, "text", new=capture):
            TOOL._render_case(case, output, blind=True)

        payload = "\n".join(captured).casefold()
        for hidden in ("2194", "107", "0.0488", "0.05", "area", "ratio", "proposal"):
            self.assertNotIn(hidden, payload)
        with Image.open(output) as image:
            metadata = json.dumps(image.info, sort_keys=True).casefold()
        for hidden in ("2194", "107", "0.0488", "0.05", "area", "ratio", "proposal"):
            self.assertNotIn(hidden, metadata)

    def test_manifest_hashes_every_artifact_and_links_decision_template(self) -> None:
        self._build()
        manifest = self._json(self.output_root / "component_review_manifest.json")
        records = manifest["artifacts"]
        recorded_paths = [item["path"] for item in records]
        self.assertEqual(recorded_paths, sorted(recorded_paths))
        self.assertNotIn("component_review_manifest.json", recorded_paths)
        actual_paths = sorted(
            path.relative_to(self.output_root).as_posix()
            for path in self.output_root.rglob("*")
            if path.is_file() and path.name != "component_review_manifest.json"
        )
        self.assertEqual(recorded_paths, actual_paths)
        for record in records:
            path = self.output_root / record["path"]
            data = path.read_bytes()
            self.assertEqual(record["size_bytes"], len(data))
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
        template_record = next(
            item for item in records if item["path"] == "decision_template.json"
        )
        self.assertEqual(
            manifest["decision_template"]["sha256"], template_record["sha256"]
        )
        self.assertEqual(
            manifest["decision_template"]["size_bytes"],
            template_record["size_bytes"],
        )

    def test_is_deterministic_and_refuses_overwrite_without_cleanup(self) -> None:
        first = self.root / "pack_first"
        second = self.root / "pack_second"
        self._build(output_root=first)
        self._build(output_root=second)

        def digests(root: Path) -> dict[str, str]:
            return {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }

        before = digests(first)
        self.assertEqual(before, digests(second))
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._build(output_root=first)
        self.assertEqual(before, digests(first))
        self.assertEqual(list(self.root.glob(".pack_first-*")), [])

    def test_rejects_source_spec_mismatch_before_any_output(self) -> None:
        inconsistent = copy.deepcopy(self.fixture.spec)
        inconsistent["source"]["source_fingerprint_sha256"] = "0" * 64
        bad_spec = self.root / "bad_source_spec.json"
        self.fixture._write_json(bad_spec, inconsistent)
        output = self.root / "must_not_exist"

        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            self._build(output_root=output, spec_path=bad_spec)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".must_not_exist-*")), [])


if __name__ == "__main__":
    unittest.main()
