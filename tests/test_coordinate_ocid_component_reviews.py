from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "coordinate_ocid_component_reviews.py"
SPEC = importlib.util.spec_from_file_location("coordinate_ocid_component_reviews", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COORDINATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COORDINATOR)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoordinateOcidComponentReviewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pack = self.root / "pack"
        self.assignments = self.root / "assignments"
        self.merge = self.root / "merge"
        self.final = self.root / "final"
        self.stream_counts = {
            "stream_big": 4,
            "stream_medium": 3,
            "stream_small": 2,
            "stream_one_a": 1,
            "stream_one_b": 1,
        }
        self._make_pack()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_pack(self) -> None:
        self.pack.mkdir(parents=True)
        decisions: list[dict[str, object]] = []
        manifest_cases: list[dict[str, object]] = []
        index = 0
        for stream_id, count in self.stream_counts.items():
            for stream_position in range(count):
                index += 1
                case_id = f"{stream_id}__frame_{stream_position + 1:04d}__label_002"
                frame_id = f"frame_{stream_position + 1:04d}"
                mask_sha = hashlib.sha256(case_id.encode()).hexdigest()
                # The first case has a distant secondary component so finalization
                # exercises IoU/edge-change author-preview rules.
                distant = index == 1
                secondary_x = 80 if distant else 12
                raw_width = secondary_x + 5
                c001 = {
                    "id": "c001",
                    "area": 100,
                    "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "binary_mask_sha256": hashlib.sha256(f"{case_id}-c001".encode()).hexdigest(),
                    "border_touch": True,
                    "centroid": {"x": 4.5, "y": 4.5},
                }
                c002 = {
                    "id": "c002",
                    "area": 4 if distant else 10,
                    "bbox": {"x": secondary_x, "y": 0, "width": 5, "height": 5},
                    "binary_mask_sha256": hashlib.sha256(f"{case_id}-c002".encode()).hexdigest(),
                    "border_touch": False,
                    "centroid": {"x": secondary_x + 2.0, "y": 2.0},
                }
                entry = {
                    "case_id": case_id,
                    "role": "development" if index <= 7 else "heldout",
                    "stream_id": stream_id,
                    "source_sequence": stream_id.replace("_", "/"),
                    "frame_id": frame_id,
                    "frame_index": stream_position + 1,
                    "source_filename": f"result_{stream_position + 1:04d}.png",
                    "source_label": 2,
                    "source_mask_sha256": mask_sha,
                    "analysis": {
                        "policy_id": COORDINATOR.POLICY_ID,
                        "connectivity": 4,
                        "secondary_area_ratio": 0.05,
                        "binary_mask_digest_encoding": "test",
                        "source_mask_sha256": mask_sha,
                        "mask_shape": [100, 100],
                        "components": [c001, c002],
                        "largest_bbox": copy.deepcopy(c001["bbox"]),
                        "raw_bbox": {"x": 0, "y": 0, "width": raw_width, "height": 10},
                        "automatic_retained_component_ids": [
                            "c001",
                            *([] if distant else ["c002"]),
                        ],
                        "review_required": True,
                    },
                    "automatic_retained_component_ids": [
                        "c001",
                        *([] if distant else ["c002"]),
                    ],
                    "review_status": "pending",
                    "reviews": [],
                    "adjudication": None,
                    "final_component_decisions": None,
                    "final_retained_component_ids": None,
                    "final_confidence": None,
                    "resolved_by": None,
                    "notes": "",
                    "author_attention": False,
                }
                decisions.append(entry)
                blind = f"cases_blind/{stream_id}/{case_id}.png"
                adjudication = f"cases_adjudication/{stream_id}/{case_id}.png"
                for relative, content in ((blind, b"blind"), (adjudication, b"adjudication")):
                    path = self.pack / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if index == 1 and relative == blind:
                        image = Image.new("RGB", (1680, 940), (30, 30, 30))
                        draw = ImageDraw.Draw(image)
                        draw.text((1082, 370), "ID area ratio proposal", fill="white")
                        metadata = PngImagePlugin.PngInfo()
                        metadata.add_text(
                            "legacy_leak", "area ratio automatic proposal"
                        )
                        image.save(path, pnginfo=metadata)
                    else:
                        path.write_bytes(content + case_id.encode())
                manifest_cases.append(
                    {
                        "case_id": case_id,
                        "role": entry["role"],
                        "stream_id": stream_id,
                        "frame_id": frame_id,
                        "frame_index": stream_position + 1,
                        "source_label": 2,
                        "source_mask_sha256": mask_sha,
                        "blind_visual_path": blind,
                        "adjudication_visual_path": adjudication,
                    }
                )
        template = {
            "schema_version": COORDINATOR.DECISIONS_SCHEMA,
            "benchmark_id": "synthetic_ocid",
            "policy_id": COORDINATOR.POLICY_ID,
            "source_fingerprint_sha256": "a" * 64,
            "review_status": "pending",
            "decisions": decisions,
        }
        template_path = self.pack / "decision_template.json"
        _write_json(template_path, template)
        artifacts = []
        for path in sorted(item for item in self.pack.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": path.relative_to(self.pack).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
        manifest = {
            "schema_version": COORDINATOR.PACK_SCHEMA,
            "scope": COORDINATOR.PACK_SCOPE,
            "status": COORDINATOR.PACK_STATUS,
            "benchmark_id": "synthetic_ocid",
            "policy": {
                "policy_id": COORDINATOR.POLICY_ID,
                "automatic_proposal": "hidden during blind review",
            },
            "provenance": {"source_fingerprint_sha256": "a" * 64},
            "ground_truth_boundary": {
                "scope": COORDINATOR.PACK_SCOPE,
                "model_predictions_accessed": False,
                "metrics_computed": False,
                "forbidden_consumer": "analyze",
            },
            "review_protocol": {
                "primary_review_count": 2,
                "primary_review_is_blind": True,
                "component_decision_values": [
                    "KEEP",
                    "DROP",
                    "UNRESOLVED",
                    "INTEGRITY_ALERT",
                ],
            },
            "totals": {
                "cases": len(decisions),
                "streams": len(self.stream_counts),
                "by_role": {"development": 7, "heldout": 4},
                "by_stream": [],
            },
            "decision_template": {
                "path": "decision_template.json",
                "sha256": _sha(template_path),
                "size_bytes": template_path.stat().st_size,
                "review_status": "pending",
            },
            "cases": manifest_cases,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        _write_json(self.pack / "component_review_manifest.json", manifest)

    def _prepare(self) -> None:
        COORDINATOR.prepare_assignments(
            review_pack=self.pack, output_root=self.assignments
        )

    def _complete_assignments(
        self,
        *,
        pass_a_decision: str = "KEEP",
        pass_b_decision: str = "KEEP",
        target_case: str | None = None,
    ) -> None:
        for pass_id, default_decision, reviewer in (
            ("pass_a", pass_a_decision, "reviewer-a"),
            ("pass_b", pass_b_decision, "reviewer-b"),
        ):
            for shard_id in COORDINATOR.SHARD_IDS:
                path = self.assignments / pass_id / f"{shard_id}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["reviewer_id"] = reviewer
                payload["review_status"] = "complete"
                for case in payload["cases"]:
                    decision = (
                        default_decision
                        if target_case is None or case["case_id"] == target_case
                        else "KEEP"
                    )
                    code = "K_TEMPORAL" if decision == "KEEP" else "D_OTHER_REGION"
                    case["case_integrity_decision"] = {
                        "decision": "OK",
                        "rationale_codes": [],
                        "confidence": "HIGH",
                        "evidence_frame_ids": [case["frame_id"]],
                        "notes": "c001 is a coherent physical object",
                    }
                    case["component_decisions"] = [
                        {
                            "component_id": component["component_id"],
                            "decision": decision,
                            "rationale_codes": [code],
                            "confidence": "HIGH",
                            "evidence_frame_ids": [case["frame_id"]],
                            "notes": "synthetic review",
                        }
                        for component in case["secondary_components"]
                    ]
                _write_json(path, payload)

    def _merge(self) -> None:
        COORDINATOR.merge_primary_reviews(
            review_pack=self.pack,
            assignments_root=self.assignments,
            output_root=self.merge,
        )

    def _complete_adjudication(self, *, decision: str = "DROP") -> Path:
        path = self.merge / "adjudication_template.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["review_status"] = "complete"
        for case in payload["cases"]:
            case["reviewer_id"] = "reviewer-c"
            case["review_status"] = "complete"
            case["case_integrity_resolution"] = {
                "decision": "OK",
                "rationale_codes": [],
                "confidence": "HIGH",
                "evidence_frame_ids": [case["frame_id"]],
                "notes": "c001 integrity resolved",
            }
            case["component_decisions"] = [
                {
                    "component_id": component["component_id"],
                    "decision": decision,
                    "rationale_codes": [
                        "K_OCCLUSION" if decision == "KEEP" else "D_OTHER_REGION"
                    ],
                    "confidence": "HIGH",
                    "evidence_frame_ids": [case["frame_id"]],
                    "notes": "adjudicated",
                }
                for component in case["disputed_components"]
            ]
            case["case_notes"] = "resolved"
        completed = self.root / "completed_adjudication.json"
        _write_json(completed, payload)
        return completed

    def test_prepare_validates_hashes(self) -> None:
        blind = next((self.pack / "cases_blind").rglob("*.png"))
        blind.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self._prepare()
        self.assertFalse(self.assignments.exists())

    def test_prepare_rejects_invalidated_v1_pack_schema(self) -> None:
        manifest_path = self.pack / "component_review_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "ocid-component-review-pack-1.0"
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "Unsupported component review"):
            self._prepare()
        self.assertFalse(self.assignments.exists())

    def test_prepare_rejects_unregistered_pack_artifact(self) -> None:
        (self.pack / "prediction_overlay.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory is not exact"):
            self._prepare()
        self.assertFalse(self.assignments.exists())

    def test_prepare_balances_without_stream_split_and_keeps_passes_identical_blind(self) -> None:
        self._prepare()
        pass_cases: dict[str, dict[str, list[str]]] = {}
        stream_shards: dict[str, set[str]] = {}
        for pass_id in COORDINATOR.PASSES:
            pass_cases[pass_id] = {}
            for shard_id in COORDINATOR.SHARD_IDS:
                path = self.assignments / pass_id / f"{shard_id}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                serialized = json.dumps(payload)
                self.assertNotIn("automatic_retained", serialized)
                self.assertNotIn("adjudication", serialized)
                self.assertNotIn('"area"', serialized)
                self.assertNotIn('"ratio"', serialized)
                self.assertTrue(payload["blind_to_automatic_proposal"])
                ids = [case["case_id"] for case in payload["cases"]]
                pass_cases[pass_id][shard_id] = ids
                for case in payload["cases"]:
                    stream_shards.setdefault(case["stream_id"], set()).add(shard_id)
                    self.assertIsNone(case["component_decisions"])
                    self.assertTrue(
                        case["blind_visual_path"].startswith(f"{pass_id}/evidence/")
                    )
                    self.assertTrue(
                        (self.assignments / case["blind_visual_path"]).is_file()
                    )
                    self.assertIn(case["frame_id"], case["allowed_evidence_frame_ids"])
        self.assertEqual(pass_cases["pass_a"], pass_cases["pass_b"])
        self.assertTrue(all(len(shards) == 1 for shards in stream_shards.values()))
        counts = [len(pass_cases["pass_a"][shard]) for shard in COORDINATOR.SHARD_IDS]
        self.assertLessEqual(max(counts) - min(counts), 2)
        inventory = {
            path.relative_to(self.assignments).as_posix()
            for path in self.assignments.rglob("*")
            if path.is_file()
        }
        self.assertFalse(any("adjudication" in path for path in inventory))
        self.assertFalse(any("decision_template" in path for path in inventory))
        self.assertFalse(any("automatic" in path for path in inventory))
        first_case = sorted(
            json.loads((self.pack / "decision_template.json").read_text())["decisions"],
            key=lambda row: row["case_id"],
        )[0]
        sanitized = next(
            self.assignments.rglob(f"{first_case['case_id']}.png")
        ).read_bytes()
        self.assertNotIn(b"legacy_leak", sanitized)
        self.assertNotIn(b"automatic proposal", sanitized)

    def test_prepare_refuses_overwrite(self) -> None:
        self._prepare()
        with self.assertRaises(FileExistsError):
            self._prepare()

    def test_merge_rejects_missing_case(self) -> None:
        self._prepare()
        self._complete_assignments()
        path = self.assignments / "pass_a" / "shard_01.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cases"].pop()
        payload["case_count"] -= 1
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self._merge()

    def test_merge_rejects_invalid_rubric(self) -> None:
        self._prepare()
        self._complete_assignments()
        path = self.assignments / "pass_a" / "shard_01.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cases"][0]["component_decisions"][0]["rationale_codes"] = ["K_COLOR"]
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "rubric rationale"):
            self._merge()

    def test_merge_rejects_same_primary_reviewer(self) -> None:
        self._prepare()
        self._complete_assignments()
        for shard_id in COORDINATOR.SHARD_IDS:
            path = self.assignments / "pass_b" / f"{shard_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["reviewer_id"] = "reviewer-a"
            _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "reviewers must be distinct"):
            self._merge()

    def test_merge_rejects_noncanonical_case_variant_reviewer_id(self) -> None:
        self._prepare()
        self._complete_assignments()
        path = self.assignments / "pass_b" / "shard_01.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reviewer_id"] = "Reviewer-A"
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "canonical lowercase reviewer ID"):
            self._merge()

    def test_merge_rejects_dotted_reviewer_id_consistently_with_final_builder(self) -> None:
        self._prepare()
        self._complete_assignments()
        path = self.assignments / "pass_b" / "shard_01.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reviewer_id"] = "reviewer.b"
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "canonical lowercase reviewer ID"):
            self._merge()

    def test_merge_rejects_evidence_frame_outside_allowed_stream_context(self) -> None:
        self._prepare()
        self._complete_assignments()
        path = self.assignments / "pass_a" / "shard_01.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cases"][0]["component_decisions"][0]["evidence_frame_ids"] = [
            "frame_9999"
        ]
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "allowed stream context"):
            self._merge()

    def test_merge_disagreement_creates_adjudication_queue_and_preserves_reviews(self) -> None:
        self._prepare()
        target = sorted(
            json.loads((self.pack / "decision_template.json").read_text())["decisions"],
            key=lambda row: row["case_id"],
        )[0]["case_id"]
        self._complete_assignments(
            pass_a_decision="KEEP", pass_b_decision="DROP", target_case=target
        )
        self._merge()
        merge = json.loads((self.merge / "primary_merge.json").read_text())
        queue = json.loads((self.merge / "adjudication_template.json").read_text())
        self.assertEqual(merge["disputed_case_count"], 1)
        self.assertEqual(queue["case_count"], 1)
        self.assertFalse(queue["blind_to_automatic_proposal"])
        self.assertEqual(len(queue["cases"][0]["primary_reviews"]), 2)
        self.assertIn("automatic_retained_component_ids", queue["cases"][0])
        self.assertEqual(queue["cases"][0]["case_id"], target)
        self.assertIn("case_integrity_resolution", queue["cases"][0])

    def test_finalize_recomputes_all_primary_disputes_and_rejects_tampered_merge(self) -> None:
        self._prepare()
        self._complete_assignments(pass_a_decision="KEEP", pass_b_decision="DROP")
        self._merge()
        merge_path = self.merge / "primary_merge.json"
        payload = json.loads(merge_path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            first = copy.deepcopy(case["reviews"][0]["component_decisions"][0])
            case["agreed_final_component_decisions"] = [first]
            case["disputed_component_ids"] = []
        payload["disputed_case_count"] = 0
        payload["disputed_component_count"] = 0
        _write_json(merge_path, payload)
        empty_adjudication = json.loads(
            (self.merge / "adjudication_template.json").read_text(encoding="utf-8")
        )
        empty_adjudication["review_status"] = "complete"
        empty_adjudication["case_count"] = 0
        empty_adjudication["cases"] = []
        adjudication_path = self.root / "tampered_adjudication.json"
        _write_json(adjudication_path, empty_adjudication)
        with self.assertRaisesRegex(ValueError, "derived field was tampered"):
            COORDINATOR.finalize_decisions(
                review_pack=self.pack,
                primary_merge=merge_path,
                adjudication=adjudication_path,
                output_root=self.final,
            )

    def test_successful_finalization_builds_exact_complete_ledger(self) -> None:
        self._prepare()
        self._complete_assignments()
        self._merge()
        adjudication = self._complete_adjudication()
        COORDINATOR.finalize_decisions(
            review_pack=self.pack,
            primary_merge=self.merge / "primary_merge.json",
            adjudication=adjudication,
            output_root=self.final,
        )
        ledger = json.loads((self.final / "component_decisions.json").read_text())
        self.assertEqual(ledger["schema_version"], COORDINATOR.DECISIONS_SCHEMA)
        self.assertEqual(ledger["review_status"], "complete")
        self.assertEqual(len(ledger["decisions"]), sum(self.stream_counts.values()))
        for case in ledger["decisions"]:
            self.assertEqual(len(case["reviews"]), 2)
            self.assertIsNone(case["adjudication"])
            self.assertEqual(case["final_retained_component_ids"], ["c001", "c002"])
            self.assertEqual(case["resolved_by"], "primary_agreement")
        preview = json.loads((self.final / "author_preview_queue.json").read_text())
        first_preview = next(
            row
            for row in preview["cases"]
            if row["case_id"] == ledger["decisions"][0]["case_id"]
        )
        self.assertIn(
            "AUTOMATIC_PROPOSAL_OVERRIDDEN", first_preview["reasons"]
        )
        summary = json.loads((self.final / "summary.json").read_text())
        self.assertEqual(summary["component_decisions_sha256"], _sha(self.final / "component_decisions.json"))

    def test_finalize_surfaces_unresolved_component_without_complete_ledger(self) -> None:
        self._prepare()
        target = json.loads((self.pack / "decision_template.json").read_text())["decisions"][0]["case_id"]
        self._complete_assignments(
            pass_a_decision="KEEP", pass_b_decision="DROP", target_case=target
        )
        self._merge()
        path = self._complete_adjudication()
        payload = json.loads(path.read_text())
        payload["cases"][0]["component_decisions"][0].update(
            {
                "decision": "UNRESOLVED",
                "confidence": "LOW",
                "rationale_codes": ["A_OCCLUDED"],
            }
        )
        payload["review_status"] = "author_decision_required"
        payload["cases"][0]["review_status"] = "author_decision_required"
        _write_json(path, payload)
        COORDINATOR.finalize_decisions(
            review_pack=self.pack,
            primary_merge=self.merge / "primary_merge.json",
            adjudication=path,
            output_root=self.final,
        )
        self.assertFalse((self.final / "component_decisions.json").exists())
        queue = json.loads((self.final / "author_decision_queue.json").read_text())
        self.assertEqual(queue["status"], "author_decision_required")
        self.assertEqual(queue["cases"][0]["unresolved_component_ids"], ["c002"])

    def test_integrity_alert_requires_author_then_resolved_integrity_can_finalize(self) -> None:
        self._prepare()
        self._complete_assignments()
        target = None
        for shard_id in COORDINATOR.SHARD_IDS:
            path = self.assignments / "pass_a" / f"{shard_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["cases"]:
                target = payload["cases"][0]["case_id"]
                payload["cases"][0]["case_integrity_decision"] = {
                    "decision": "INTEGRITY_ALERT",
                    "rationale_codes": ["I_C001_INVALID"],
                    "confidence": "LOW",
                    "evidence_frame_ids": [payload["cases"][0]["frame_id"]],
                    "notes": "c001 may cover the wrong physical region",
                }
                _write_json(path, payload)
                break
        self.assertIsNotNone(target)
        self._merge()
        path = self.merge / "adjudication_template.json"
        adjudication = json.loads(path.read_text(encoding="utf-8"))
        adjudication["review_status"] = "author_decision_required"
        case = next(item for item in adjudication["cases"] if item["case_id"] == target)
        case["reviewer_id"] = "reviewer-c"
        case["review_status"] = "author_decision_required"
        case["case_integrity_resolution"] = {
            "decision": "INTEGRITY_ALERT",
            "rationale_codes": ["I_C001_INVALID"],
            "confidence": "LOW",
            "evidence_frame_ids": [case["frame_id"]],
            "notes": "author must inspect c001",
        }
        case["component_decisions"] = []
        case["case_notes"] = "integrity remains unresolved"
        unresolved_path = self.root / "integrity_unresolved.json"
        _write_json(unresolved_path, adjudication)
        COORDINATOR.finalize_decisions(
            review_pack=self.pack,
            primary_merge=self.merge / "primary_merge.json",
            adjudication=unresolved_path,
            output_root=self.final,
        )
        self.assertFalse((self.final / "component_decisions.json").exists())
        queue = json.loads((self.final / "author_decision_queue.json").read_text())
        self.assertTrue(queue["cases"][0]["case_integrity_requires_author"])

        adjudication["review_status"] = "complete"
        case["review_status"] = "complete"
        case["case_integrity_resolution"] = {
            "decision": "OK",
            "rationale_codes": [],
            "confidence": "HIGH",
            "evidence_frame_ids": [case["frame_id"]],
            "notes": "author confirmed c001 integrity",
        }
        case["case_notes"] = "resolved by author inspection"
        resolved_path = self.root / "integrity_resolved.json"
        _write_json(resolved_path, adjudication)
        resolved_output = self.root / "final_resolved"
        COORDINATOR.finalize_decisions(
            review_pack=self.pack,
            primary_merge=self.merge / "primary_merge.json",
            adjudication=resolved_path,
            output_root=resolved_output,
        )
        ledger = json.loads((resolved_output / "component_decisions.json").read_text())
        resolved_case = next(item for item in ledger["decisions"] if item["case_id"] == target)
        self.assertEqual(resolved_case["case_integrity_resolution"]["decision"], "OK")
        self.assertTrue(resolved_case["author_attention"])

    def test_finalize_rejects_primary_reviewer_as_adjudicator_even_with_exception(self) -> None:
        self._prepare()
        target = json.loads((self.pack / "decision_template.json").read_text())["decisions"][0]["case_id"]
        self._complete_assignments(
            pass_a_decision="KEEP", pass_b_decision="DROP", target_case=target
        )
        self._merge()
        path = self._complete_adjudication()
        payload = json.loads(path.read_text())
        payload["cases"][0]["reviewer_id"] = "reviewer-a"
        payload["cases"][0]["reviewer_identity_exception"] = (
            "No exception may bypass independent adjudication."
        )
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "exceptions are not permitted"):
            COORDINATOR.finalize_decisions(
                review_pack=self.pack,
                primary_merge=self.merge / "primary_merge.json",
                adjudication=path,
                output_root=self.final,
            )

    def test_author_preview_flags_adjudication_iou_override_and_edge_move(self) -> None:
        self._prepare()
        first_case = json.loads((self.pack / "decision_template.json").read_text())["decisions"][0]["case_id"]
        self._complete_assignments(
            pass_a_decision="KEEP", pass_b_decision="DROP", target_case=first_case
        )
        self._merge()
        completed = self._complete_adjudication(decision="DROP")
        COORDINATOR.finalize_decisions(
            review_pack=self.pack,
            primary_merge=self.merge / "primary_merge.json",
            adjudication=completed,
            output_root=self.final,
        )
        queue = json.loads((self.final / "author_preview_queue.json").read_text())
        flagged = next(row for row in queue["cases"] if row["case_id"] == first_case)
        self.assertIn("ADJUDICATED", flagged["reasons"])
        self.assertIn("RAW_FINAL_BBOX_IOU_LT_0_75", flagged["reasons"])
        self.assertIn("BBOX_EDGE_MOVE_GT_10_PERCENT_FRAME", flagged["reasons"])
        # The automatic proposal for the distant 4%-area component was DROP,
        # so adjudicated DROP itself is not an automatic override.
        self.assertNotIn("AUTOMATIC_PROPOSAL_OVERRIDDEN", flagged["reasons"])


if __name__ == "__main__":
    unittest.main()
