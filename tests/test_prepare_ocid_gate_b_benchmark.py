from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from stream_analysis.evaluation import load_annotation
from stream_analysis.evaluation.component_review import analyze_component_mask


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "prepare_ocid_gate_b_benchmark.py"
FROZEN_SPEC_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "ocid"
    / "benchmark"
    / "ocid_candidate_benchmark_v1.json"
)
SPEC = importlib.util.spec_from_file_location("prepare_ocid_gate_b_benchmark", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load tool module from {TOOL_PATH}.")
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


SOURCE_FINGERPRINT_METHOD = (
    "sha256 over lexicographically sorted UTF-8 records: "
    r"relative_posix_path\0size_bytes\0sha256(content)\n"
)


class PrepareOcidGateBBenchmarkTest(unittest.TestCase):
    WIDTH = 40
    HEIGHT = 40
    SUPPORT_LABEL = 256
    SCENE_GROUP_ID = "ocid_scene_0123456789abcdef"
    SCENE_GROUP_KEY = "ARID10/floor/box/seq01"
    SOURCE_SEQUENCES = (
        "ARID10/floor/bottom/box/seq01",
        "ARID10/floor/top/box/seq01",
    )
    FILENAMES = ("state_01.png", "state_02.png", "state_03.png")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ocid_root = self.root / "OCID-dataset"
        self.ocid_root.mkdir()
        self.audit_path = self.root / "structural_audit.json"
        self.spec_path = self.root / "synthetic_spec.json"
        self.component_decisions_path = self.root / "component_decisions.json"
        self.output_root = self.root / "gate_b_benchmark"
        self._write_synthetic_ocid()
        self.audit = self._make_structural_audit()
        self._write_json(self.audit_path, self.audit)
        self.spec = self._make_spec()
        self.component_decisions = self._make_component_decisions()
        self._write_json(self.component_decisions_path, self.component_decisions)
        self.spec["source"]["component_decisions_sha256"] = hashlib.sha256(
            self.component_decisions_path.read_bytes()
        ).hexdigest()
        self._write_json(self.spec_path, self.spec)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _stream_id(source_sequence: str) -> str:
        return "ocid_" + "_".join(source_sequence.casefold().split("/"))

    @staticmethod
    def _integrity_record(
        frame_id: str, *, decision: str = "OK"
    ) -> dict[str, object]:
        if decision == "OK":
            return {
                "decision": "OK",
                "rationale_codes": [],
                "confidence": "HIGH",
                "evidence_frame_ids": [frame_id],
                "notes": "Synthetic fixture confirms one valid c001 target.",
            }
        if decision == "INTEGRITY_ALERT":
            return {
                "decision": "INTEGRITY_ALERT",
                "rationale_codes": ["I_C001_INVALID"],
                "confidence": "LOW",
                "evidence_frame_ids": [frame_id],
                "notes": "Synthetic fixture requires an integrity adjudication.",
            }
        raise ValueError(f"Unsupported synthetic integrity decision: {decision}.")

    def _masks(self) -> dict[str, np.ndarray]:
        first = np.full(
            (self.HEIGHT, self.WIDTH), self.SUPPORT_LABEL, dtype=np.uint16
        )
        first[2:7, 2:6] = 257
        first[0, 0] = 258
        first[self.HEIGHT - 1, self.WIDTH - 1] = 258
        first[20:35, 15] = 259
        first[20:35, 29] = 259
        first[20, 15:30] = 259
        first[34, 15:30] = 259
        first[27, 22] = 259

        second = np.full(
            (self.HEIGHT, self.WIDTH), self.SUPPORT_LABEL, dtype=np.uint16
        )
        second[4, 4] = 257

        third = np.full(
            (self.HEIGHT, self.WIDTH), self.SUPPORT_LABEL, dtype=np.uint16
        )
        third[2:7, 2:6] = 257
        third[self.HEIGHT - 1, 0] = 258
        return {
            "state_01.png": first,
            "state_02.png": second,
            "state_03.png": third,
        }

    def _write_synthetic_ocid(self) -> None:
        masks = self._masks()
        creation_order = ("state_03.png", "state_01.png", "state_02.png")
        for camera_index, source_sequence in enumerate(self.SOURCE_SEQUENCES):
            sequence = self.ocid_root.joinpath(*source_sequence.split("/"))
            (sequence / "rgb").mkdir(parents=True)
            (sequence / "label").mkdir()
            for frame_index, filename in enumerate(creation_order, start=1):
                color = (
                    20 + camera_index * 40,
                    30 + frame_index * 20,
                    40 + camera_index + frame_index,
                )
                Image.new("RGB", (self.WIDTH, self.HEIGHT), color).save(
                    sequence / "rgb" / filename
                )
                Image.fromarray(masks[filename]).save(sequence / "label" / filename)

    def _descriptor(self, path: Path) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "relative_path": path.relative_to(self.ocid_root).as_posix(),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    @staticmethod
    def _fingerprint(descriptors: list[dict[str, object]]) -> dict[str, object]:
        digest = hashlib.sha256()
        total_size = 0
        ordered = sorted(descriptors, key=lambda item: str(item["relative_path"]))
        for item in ordered:
            size = int(item["size_bytes"])
            record = (
                f"{item['relative_path']}\0{size}\0{item['sha256']}\n"
            ).encode("utf-8")
            digest.update(record)
            total_size += size
        return {
            "algorithm": "sha256",
            "method": SOURCE_FINGERPRINT_METHOD,
            "scope": "paired_rgb_and_label_png_files",
            "entry_count": len(ordered),
            "total_size_bytes": total_size,
            "sha256": digest.hexdigest(),
        }

    @staticmethod
    def _component_count(mask: np.ndarray) -> int:
        visited = np.zeros(mask.shape, dtype=bool)
        count = 0
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not bool(mask[y, x]) or bool(visited[y, x]):
                    continue
                count += 1
                visited[y, x] = True
                pending = [(x, y)]
                while pending:
                    current_x, current_y = pending.pop()
                    for next_x, next_y in (
                        (current_x - 1, current_y),
                        (current_x + 1, current_y),
                        (current_x, current_y - 1),
                        (current_x, current_y + 1),
                    ):
                        if not (0 <= next_x < width and 0 <= next_y < height):
                            continue
                        if bool(mask[next_y, next_x]) and not bool(
                            visited[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            pending.append((next_x, next_y))
        return count

    def _physical_instance(
        self,
        labels: np.ndarray,
        *,
        stream_id: str,
        source_label: int,
    ) -> dict[str, object]:
        mask = labels == source_label
        rows, columns = np.nonzero(mask)
        left = int(columns.min())
        top = int(rows.min())
        right = int(columns.max())
        bottom = int(rows.max())
        physical_id = f"{stream_id}__label_{source_label:03d}"
        return {
            "physical_instance_id": physical_id,
            "source_label": source_label,
            "visible_pixels": int(rows.size),
            "visible_fraction": round(int(rows.size) / labels.size, 12),
            "bbox": {
                "x": left,
                "y": top,
                "width": right - left + 1,
                "height": bottom - top + 1,
            },
            "centroid": {
                "x": round(float(columns.mean()), 6),
                "y": round(float(rows.mean()), 6),
            },
            "connected_component_count": self._component_count(mask),
            "border_touch": bool(
                np.any(mask[0, :])
                or np.any(mask[-1, :])
                or np.any(mask[:, 0])
                or np.any(mask[:, -1])
            ),
        }

    def _audit_sequence(
        self, source_sequence: str
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        sequence = self.ocid_root.joinpath(*source_sequence.split("/"))
        stream_id = self._stream_id(source_sequence)
        descriptors: list[dict[str, object]] = []
        frames: list[dict[str, object]] = []
        all_physical_ids: set[str] = set()
        for frame_index, filename in enumerate(self.FILENAMES, start=1):
            rgb_path = sequence / "rgb" / filename
            label_path = sequence / "label" / filename
            descriptors.extend(
                [self._descriptor(rgb_path), self._descriptor(label_path)]
            )
            with Image.open(label_path) as image:
                labels = np.asarray(image).copy()
            source_labels = sorted(
                int(value)
                for value in np.unique(labels)
                if int(value) > self.SUPPORT_LABEL
            )
            instances = [
                self._physical_instance(
                    labels, stream_id=stream_id, source_label=source_label
                )
                for source_label in source_labels
            ]
            physical_ids = [str(item["physical_instance_id"]) for item in instances]
            all_physical_ids.update(physical_ids)
            frames.append(
                {
                    "frame_id": f"frame_{frame_index:04d}",
                    "frame_index": frame_index,
                    "source_filename": filename,
                    "width": self.WIDTH,
                    "height": self.HEIGHT,
                    "object_count": len(instances),
                    "physical_instance_ids": physical_ids,
                    "physical_instances": instances,
                }
            )
        camera = "bottom" if "/bottom/" in source_sequence else "top"
        return (
            {
                "sequence_id": stream_id,
                "scene_group_id": self.SCENE_GROUP_ID,
                "source_sequence": source_sequence,
                "paired_scene_key": self.SCENE_GROUP_KEY,
                "camera": camera,
                "support_label": self.SUPPORT_LABEL,
                "object_label_minimum": self.SUPPORT_LABEL + 1,
                "label_policy": {
                    "support_label_detection": (
                        "unique_dominant_nonzero_label_in_first_mask"
                    ),
                    "object_label_rule": (
                        "source_label_greater_than_support_label"
                    ),
                    "expected_values_are_diagnostics_only": True,
                },
                "frame_count": len(frames),
                "frame_size": {"width": self.WIDTH, "height": self.HEIGHT},
                "physical_instance_id_scope": "sequence_local",
                "paired_camera_identity_linkage": "not_assessed",
                "physical_instance_ids": sorted(all_physical_ids),
                "source_fingerprint": self._fingerprint(descriptors),
                "frames": frames,
            },
            descriptors,
        )

    def _make_structural_audit(self) -> dict[str, object]:
        sequences: list[dict[str, object]] = []
        all_descriptors: list[dict[str, object]] = []
        for source_sequence in self.SOURCE_SEQUENCES:
            sequence, descriptors = self._audit_sequence(source_sequence)
            sequences.append(sequence)
            all_descriptors.extend(descriptors)
        source_fingerprint = self._fingerprint(all_descriptors)
        self.source_fingerprint_sha256 = str(source_fingerprint["sha256"])
        return {
            "schema_id": "ocid_structural_audit",
            "schema_version": "1.0.0",
            "status": "complete",
            "scope": "evaluation_only_dataset_audit",
            "ground_truth_boundary": {
                "uses_ground_truth": True,
                "allowed_consumers": ["dataset_audit", "evaluation"],
                "forbidden_consumer": "analyze",
            },
            "parameters": {
                "support_label_detection": (
                    "unique_dominant_nonzero_label_in_first_mask"
                ),
                "object_label_rule": "source_label_greater_than_support_label",
                "expected_normal_values_used_for_object_parsing": False,
                "severe_area_ratio": 0.5,
            },
            "provenance": {"source_fingerprint": source_fingerprint},
            "summary": {"sequence_count": len(sequences)},
            "scene_groups": [
                {
                    "scene_group_id": self.SCENE_GROUP_ID,
                    "scene_group_key": self.SCENE_GROUP_KEY,
                    "member_sequence_ids": [
                        self._stream_id(item) for item in self.SOURCE_SEQUENCES
                    ],
                    "complete_top_bottom_pair": True,
                    "matching_frame_count": True,
                }
            ],
            "sequences": sequences,
        }

    def _make_spec(self) -> dict[str, object]:
        payload = json.loads(FROZEN_SPEC_PATH.read_text(encoding="utf-8"))
        payload["benchmark_id"] = "synthetic_ocid_gate_b"
        payload["purpose"] = "Deterministic synthetic Gate B builder verification."
        payload["source"]["source_fingerprint_sha256"] = (
            self.source_fingerprint_sha256
        )
        payload["source"]["structural_audit_path"] = self.audit_path.name
        payload["source"]["structural_audit_sha256"] = hashlib.sha256(
            self.audit_path.read_bytes()
        ).hexdigest()
        payload["roles"] = {
            "development": [
                {
                    "scene_group_id": self.SCENE_GROUP_ID,
                    "scene_group_key": self.SCENE_GROUP_KEY,
                    "frames_per_stream": len(self.FILENAMES),
                    "member_streams": list(self.SOURCE_SEQUENCES),
                    "reason": "Tiny paired synthetic regression fixture.",
                }
            ],
            "heldout": [],
        }
        payload["expected_totals"] = {
            "development_scene_groups": 1,
            "development_streams": 2,
            "development_frames": 6,
            "heldout_scene_groups": 0,
            "heldout_streams": 0,
            "heldout_frames": 0,
            "benchmark_scene_groups": 1,
            "benchmark_streams": 2,
            "benchmark_frames": 6,
        }
        payload["review_contract"]["frame_coverage"] = "all_6_frames"
        payload["annotation_contract"].update(
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
        payload["review_contract"]["component_review_expected_observations"] = 2
        payload["source"]["component_decisions_status"] = "frozen_complete"
        payload["source"]["component_decisions_sha256"] = "0" * 64
        return payload

    def _make_component_decisions(self) -> dict[str, object]:
        decisions: list[dict[str, object]] = []
        masks = self._masks()
        for source_sequence in self.SOURCE_SEQUENCES:
            stream_id = self._stream_id(source_sequence)
            for frame_index, filename in enumerate(self.FILENAMES, start=1):
                labels = masks[filename]
                frame_id = f"frame_{frame_index:04d}"
                for source_label in sorted(
                    int(value)
                    for value in np.unique(labels)
                    if int(value) > self.SUPPORT_LABEL
                ):
                    analysis = analyze_component_mask(
                        np.asarray(labels == source_label, dtype=np.bool_)
                    )
                    if not analysis.review_required:
                        continue
                    final_rows = [
                        {
                            "component_id": component.component_id,
                            "decision": "DROP",
                            "rationale_codes": ["D_OTHER_REGION"],
                            "confidence": "HIGH",
                            "evidence_frame_ids": [frame_id],
                            "notes": "Synthetic disconnected-noise regression fixture.",
                        }
                        for component in analysis.components[1:]
                    ]
                    decisions.append(
                        {
                            "case_id": (
                                f"{stream_id}__{frame_id}__label_{source_label:03d}"
                            ),
                            "role": "development",
                            "stream_id": stream_id,
                            "source_sequence": source_sequence,
                            "frame_id": frame_id,
                            "frame_index": frame_index,
                            "source_filename": filename,
                            "source_label": source_label,
                            "source_mask_sha256": analysis.source_mask_sha256,
                            "analysis": analysis.as_dict(),
                            "automatic_retained_component_ids": list(
                                analysis.automatic_retained_component_ids
                            ),
                            "review_status": "complete",
                            "reviews": [
                                {
                                    "reviewer_id": "blind_reviewer_a",
                                    "blind_to_automatic_proposal": True,
                                    "case_integrity_decision": self._integrity_record(
                                        frame_id
                                    ),
                                    "component_decisions": copy.deepcopy(final_rows),
                                },
                                {
                                    "reviewer_id": "blind_reviewer_b",
                                    "blind_to_automatic_proposal": True,
                                    "case_integrity_decision": self._integrity_record(
                                        frame_id
                                    ),
                                    "component_decisions": copy.deepcopy(final_rows),
                                },
                            ],
                            "adjudication": None,
                            "case_integrity_resolution": self._integrity_record(
                                frame_id
                            ),
                            "final_component_decisions": final_rows,
                            "final_retained_component_ids": ["c001"],
                            "final_confidence": "HIGH",
                            "resolved_by": "blind_consensus",
                            "author_attention": False,
                        }
                    )
        return {
            "schema_version": "ocid-component-review-decisions-1.0",
            "benchmark_id": self.spec["benchmark_id"],
            "policy_id": "largest_plus_reviewed_components_v1",
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "review_status": "complete",
            "decisions": decisions,
        }

    def _build(
        self,
        *,
        output_root: Path | None = None,
        structural_audit: Path | None = None,
        spec_path: Path | None = None,
        component_decisions: Path | None = None,
    ):
        return TOOL.build_gate_b_benchmark(
            ocid_root=self.ocid_root,
            structural_audit=structural_audit or self.audit_path,
            spec_path=spec_path or self.spec_path,
            component_decisions=component_decisions or self.component_decisions_path,
            output_root=output_root or self.output_root,
        )

    def _decision_variant(
        self,
        name: str,
        payload: dict[str, object],
        *,
        expected_count: int | None = None,
        update_hash: bool = True,
    ) -> tuple[Path, Path]:
        decisions_path = self.root / f"{name}_decisions.json"
        self._write_json(decisions_path, payload)
        spec = copy.deepcopy(self.spec)
        if expected_count is not None:
            spec["review_contract"][
                "component_review_expected_observations"
            ] = expected_count
        if update_hash:
            spec["source"]["component_decisions_sha256"] = hashlib.sha256(
                decisions_path.read_bytes()
            ).hexdigest()
        spec_path = self.root / f"{name}_spec.json"
        self._write_json(spec_path, spec)
        return decisions_path, spec_path

    def _extra_nonreview_decision(self) -> dict[str, object]:
        source_sequence = self.SOURCE_SEQUENCES[0]
        stream_id = self._stream_id(source_sequence)
        frame_id = "frame_0001"
        source_label = 257
        analysis = analyze_component_mask(
            np.asarray(self._masks()["state_01.png"] == source_label, dtype=np.bool_)
        )
        self.assertFalse(analysis.review_required)
        return {
            "case_id": f"{stream_id}__{frame_id}__label_{source_label:03d}",
            "role": "development",
            "stream_id": stream_id,
            "source_sequence": source_sequence,
            "frame_id": frame_id,
            "frame_index": 1,
            "source_filename": "state_01.png",
            "source_label": source_label,
            "source_mask_sha256": analysis.source_mask_sha256,
            "analysis": analysis.as_dict(),
            "automatic_retained_component_ids": list(
                analysis.automatic_retained_component_ids
            ),
            "review_status": "complete",
            "reviews": [
                {
                    "reviewer_id": "blind_reviewer_a",
                    "blind_to_automatic_proposal": True,
                    "case_integrity_decision": self._integrity_record(frame_id),
                    "component_decisions": [],
                },
                {
                    "reviewer_id": "blind_reviewer_b",
                    "blind_to_automatic_proposal": True,
                    "case_integrity_decision": self._integrity_record(frame_id),
                    "component_decisions": [],
                },
            ],
            "adjudication": None,
            "case_integrity_resolution": self._integrity_record(frame_id),
            "final_component_decisions": [],
            "final_retained_component_ids": ["c001"],
            "final_confidence": "HIGH",
            "resolved_by": "blind_consensus",
            "author_attention": False,
        }

    @staticmethod
    def _csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_builds_complete_atomic_rgb_only_benchmark(self) -> None:
        self.assertFalse(self.output_root.exists())

        result = self._build()

        self.assertEqual(result.output_root, self.output_root.resolve())
        self.assertEqual(result.scene_group_count, 1)
        self.assertEqual(result.stream_count, 2)
        self.assertEqual(result.frame_count, 6)
        self.assertEqual(result.observation_count, 12)
        self.assertEqual(result.review_queue_count, 6)
        self.assertEqual(list(self.root.glob(".gate_b_benchmark-*")), [])

        required = {
            "artifact_manifest.json",
            "benchmark_manifest.json",
            "event_support_matrix.json",
            "heldout_access_log.json",
            "review/frame_review_ledger.csv",
            "review/observation_ledger.csv",
            "review/review_queue.json",
            "review/component_review_decisions.json",
        }
        existing = {
            path.relative_to(self.output_root).as_posix()
            for path in self.output_root.rglob("*")
            if path.is_file()
        }
        self.assertTrue(required.issubset(existing))

        benchmark = json.loads(
            (self.output_root / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(benchmark["status"], "prepared")
        self.assertEqual(benchmark["totals"]["benchmark_scene_groups"], 1)
        self.assertEqual(benchmark["totals"]["benchmark_streams"], 2)
        self.assertEqual(benchmark["totals"]["benchmark_frames"], 6)
        self.assertEqual(benchmark["totals"]["visible_instance_observations"], 12)
        self.assertEqual(benchmark["totals"]["review_queue_frames"], 6)
        self.assertEqual(
            benchmark["ground_truth_boundary"]["firewall_validation"],
            "passed_recursive_analysis_stream_scan",
        )
        self.assertEqual(
            benchmark["provenance"]["source_fingerprint_sha256"],
            self.source_fingerprint_sha256,
        )
        decision_sha = hashlib.sha256(self.component_decisions_path.read_bytes()).hexdigest()
        self.assertEqual(benchmark["provenance"]["component_decisions_sha256"], decision_sha)
        self.assertEqual(
            benchmark["component_review"],
            {
                "policy_id": "largest_plus_reviewed_components_v1",
                "review_status": "complete",
                "reviewed_observation_count": 2,
                "decisions_sha256": decision_sha,
                "decisions_artifact": "review/component_review_decisions.json",
                "final_boxes_from_reviewed_retained_component_union": True,
            },
        )
        self.assertEqual(
            (self.output_root / "review" / "component_review_decisions.json").read_bytes(),
            self.component_decisions_path.read_bytes(),
        )

        TOOL._validate_analysis_firewall(self.output_root / "analysis_streams")
        loaded_observations = 0
        for stream in benchmark["streams"]:
            stream_id = stream["stream_id"]
            source_sequence = stream["source_sequence"]
            analysis = self.output_root / "analysis_streams" / stream_id
            manifest_path = analysis / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual({item.name for item in analysis.iterdir()}, {"frames", "manifest.json"})
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertFalse(any("label" in part.casefold() for path in analysis.rglob("*") for part in path.relative_to(analysis).parts))
            for frame in manifest["frames"]:
                prepared = analysis / frame["image_path"]
                source = self.ocid_root.joinpath(*source_sequence.split("/")) / "rgb" / frame["metadata"]["ocid_source_filename"]
                self.assertEqual(prepared.read_bytes(), source.read_bytes())
                with Image.open(prepared) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "RGB")

            annotation_path = (
                self.output_root
                / "evaluation_annotations"
                / stream_id
                / "annotation.json"
            )
            annotation = load_annotation(annotation_path, manifest_path=manifest_path)
            self.assertEqual(
                annotation.annotation_scope,
                "ocid_candidate_extraction_bbox_from_instance_masks",
            )
            self.assertEqual(annotation.supported_event_types, ())
            self.assertEqual(annotation.frame_ids, ("frame_0001", "frame_0002", "frame_0003"))
            self.assertEqual(len(annotation.instances), 6)
            self.assertEqual(len(annotation.visual_type_ids), 3)
            self.assertTrue(annotation.raw["metadata"]["candidate_only"])
            self.assertEqual(
                annotation.raw["metadata"]["audited_support_label"],
                self.SUPPORT_LABEL,
            )
            self.assertEqual(
                {item["source_label"] for item in annotation.raw["expected_element_instances"]},
                {257, 258, 259},
            )
            self.assertEqual(
                annotation.raw["metadata"]["bbox_derivation"],
                "tight_axis_aligned_bbox_from_reviewed_instance_mask_components",
            )
            self.assertEqual(
                annotation.raw["metadata"]["component_decisions_sha256"],
                decision_sha,
            )
            cleaned = next(
                item
                for item in annotation.raw["expected_element_instances"]
                if item["frame_id"] == "frame_0001" and item["source_label"] == 258
            )
            self.assertEqual(cleaned["bbox"], {"x": 0, "y": 0, "width": 1, "height": 1})
            loaded_observations += len(annotation.instances)

            overlays = self.output_root / "review" / "overlays" / stream_id
            self.assertEqual(len(list(overlays.glob("*.png"))), 3)
            with Image.open(overlays / "frame_0001.png") as overlay, Image.open(
                self.ocid_root.joinpath(*source_sequence.split("/"))
                / "rgb"
                / "state_01.png"
            ) as source_rgb:
                self.assertEqual(
                    overlay.convert("RGB").getpixel((self.WIDTH - 1, self.HEIGHT - 1)),
                    source_rgb.getpixel((self.WIDTH - 1, self.HEIGHT - 1)),
                )
                self.assertNotEqual(overlay.convert("RGB").getpixel((0, 0)), source_rgb.getpixel((0, 0)))
                self.assertNotEqual(
                    overlay.convert("RGB").getpixel((22, 27)),
                    source_rgb.getpixel((22, 27)),
                )
            contact_sheet = self.output_root / "review" / "contact_sheets" / f"{stream_id}.png"
            with Image.open(contact_sheet) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
            timeline = self.output_root / "review" / "timelines" / f"{stream_id}.csv"
            self.assertEqual(len(self._csv_rows(timeline)), 3)
        self.assertEqual(loaded_observations, 12)

        observation_rows = self._csv_rows(
            self.output_root / "review" / "observation_ledger.csv"
        )
        frame_rows = self._csv_rows(
            self.output_root / "review" / "frame_review_ledger.csv"
        )
        self.assertEqual(len(observation_rows), 12)
        self.assertEqual(len(frame_rows), 6)
        self.assertTrue(all(row["retained_in_annotation"] == "true" for row in observation_rows))
        cleaned_rows = [
            row
            for row in observation_rows
            if row["frame_id"] == "frame_0001" and row["source_label"] == "258"
        ]
        self.assertEqual(len(cleaned_rows), 2)
        self.assertTrue(all(row["raw_bbox_width"] == "40" for row in cleaned_rows))
        self.assertTrue(all(row["bbox_width"] == "1" for row in cleaned_rows))
        self.assertTrue(all(row["component_review_required"] == "true" for row in cleaned_rows))
        self.assertTrue(all(row["final_retained_component_ids"] == "c001" for row in cleaned_rows))
        self.assertTrue(all(row["removed_component_ids"] == "c002" for row in cleaned_rows))
        self.assertTrue(all(row["resolved_by"] == "blind_consensus" for row in cleaned_rows))
        internal_fragment_rows = [
            row
            for row in observation_rows
            if row["frame_id"] == "frame_0001" and row["source_label"] == "259"
        ]
        self.assertEqual(len(internal_fragment_rows), 2)
        self.assertTrue(
            all(row["component_review_required"] == "false" for row in internal_fragment_rows)
        )
        self.assertTrue(
            all(row["final_retained_component_ids"] == "c001;c002" for row in internal_fragment_rows)
        )
        self.assertTrue(all(row["removed_component_ids"] == "" for row in internal_fragment_rows))
        self.assertTrue(
            all(
                row["resolved_by"] == "raw_union_bbox_unchanged_no_manual_review"
                for row in internal_fragment_rows
            )
        )
        flag_union = {
            flag
            for row in frame_rows
            for flag in row["automatic_flags"].split(";")
            if flag
        }
        self.assertEqual(flag_union, set(TOOL.AUTOMATIC_FLAGS))
        low_visibility = [
            row
            for row in observation_rows
            if row["frame_id"] == "frame_0002" and row["source_label"] == "257"
        ]
        self.assertEqual(len(low_visibility), 2)
        self.assertTrue(
            all(float(row["relative_to_stream_instance_max_area"]) == 0.05 for row in low_visibility)
        )
        fragmented = [
            row
            for row in observation_rows
            if row["frame_id"] == "frame_0001" and row["source_label"] == "258"
        ]
        self.assertEqual(len(fragmented), 2)
        self.assertTrue(all(row["connected_component_count"] == "2" for row in fragmented))
        self.assertTrue(all(row["border_touch"] == "true" for row in fragmented))

        review_queue = json.loads(
            (self.output_root / "review" / "review_queue.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(review_queue["queue_count"], 6)
        self.assertEqual(review_queue["automatic_flags"], list(TOOL.AUTOMATIC_FLAGS))
        self.assertTrue(
            all(item["author_review_status"] == "pending" for item in review_queue["frames"])
        )

        heldout_log = json.loads(
            (self.output_root / "heldout_access_log.json").read_text(encoding="utf-8")
        )
        self.assertFalse(heldout_log["predictions_unlocked"])
        self.assertEqual(heldout_log["heldout_stream_count"], 0)
        self.assertTrue(
            all(not item["prediction_accessed"] for item in heldout_log["entries"])
        )

        artifact_manifest = json.loads(
            (self.output_root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact_manifest["artifact_count"], len(existing) - 1)
        self.assertNotIn(
            "artifact_manifest.json",
            {item["path"] for item in artifact_manifest["artifacts"]},
        )
        for artifact in artifact_manifest["artifacts"]:
            path = self.output_root / artifact["path"]
            data = path.read_bytes()
            self.assertEqual(artifact["size_bytes"], len(data))
            self.assertEqual(artifact["sha256"], hashlib.sha256(data).hexdigest())

        leaked = (
            self.output_root
            / "analysis_streams"
            / benchmark["streams"][0]["stream_id"]
            / "frames"
            / "label_preview.png"
        )
        Image.fromarray(self._masks()["state_01.png"]).save(leaked)
        with self.assertRaisesRegex(TOOL.GroundTruthFirewallError, "forbidden marker"):
            TOOL._validate_analysis_firewall(self.output_root / "analysis_streams")

    def test_audit_metadata_uses_literal_fingerprint_delimiter_description(self) -> None:
        recorded = self.audit["provenance"]["source_fingerprint"]["method"]

        self.assertEqual(recorded, TOOL.SOURCE_FINGERPRINT_METHOD)
        self.assertIn(r"\0", recorded)
        self.assertIn(r"\n", recorded)
        self.assertNotIn("\0", recorded)
        self.assertNotIn("\n", recorded)

        digest_record = "path/to/frame.png\0" + "123\0" + "a" * 64 + "\n"
        encoded = digest_record.encode("utf-8")
        self.assertIn(b"\x00", encoded)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertNotIn(b"\\0", encoded)
        self.assertNotIn(b"\\n", encoded)

    def test_rejects_audit_and_spec_source_hash_mismatches_without_output(self) -> None:
        tampered_audit = self.root / "tampered_structural_audit.json"
        tampered_audit.write_bytes(self.audit_path.read_bytes() + b" ")
        hash_output = self.root / "hash_mismatch_output"
        with self.assertRaisesRegex(ValueError, "Structural audit SHA-256 mismatch"):
            self._build(
                output_root=hash_output,
                structural_audit=tampered_audit,
            )
        self.assertFalse(hash_output.exists())

        source_mismatch = copy.deepcopy(self.spec)
        source_mismatch["source"]["source_fingerprint_sha256"] = "0" * 64
        source_spec_path = self.root / "source_mismatch_spec.json"
        self._write_json(source_spec_path, source_mismatch)
        source_output = self.root / "source_mismatch_output"
        with self.assertRaisesRegex(
            ValueError, "Spec source fingerprint does not match the structural audit"
        ):
            self._build(output_root=source_output, spec_path=source_spec_path)
        self.assertFalse(source_output.exists())

    def test_refuses_to_overwrite_published_benchmark(self) -> None:
        self._build()
        original = {
            path.relative_to(self.output_root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.output_root.rglob("*")
            if path.is_file()
        }

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._build()

        current = {
            path.relative_to(self.output_root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.output_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(current, original)
        self.assertEqual(list(self.root.glob(".gate_b_benchmark-*")), [])

    def test_rejects_spec_that_allows_analyze_to_consume_ground_truth(self) -> None:
        inconsistent = copy.deepcopy(self.spec)
        inconsistent["ground_truth_boundary"]["allowed_consumers"].append("analyze")
        inconsistent_path = self.root / "inconsistent_boundary_spec.json"
        self._write_json(inconsistent_path, inconsistent)
        output = self.root / "inconsistent_boundary_output"

        try:
            self._build(output_root=output, spec_path=inconsistent_path)
        except ValueError as error:
            self.assertRegex(str(error), "allowed_consumers|analyze")
        else:
            published = json.loads(
                (output / "benchmark_manifest.json").read_text(encoding="utf-8")
            )
            self.fail(
                "Builder published an internally contradictory ground-truth boundary: "
                f"allowed_consumers={published['ground_truth_boundary']['allowed_consumers']!r}, "
                f"forbidden_consumer={published['ground_truth_boundary']['forbidden_consumer']!r}."
            )

    def test_rejects_contradictory_heldout_operation_policy(self) -> None:
        inconsistent = copy.deepcopy(self.spec)
        inconsistent["heldout_lock"]["allowed_before_unlock"].append(
            "model_prediction"
        )
        inconsistent_path = self.root / "contradictory_heldout_spec.json"
        self._write_json(inconsistent_path, inconsistent)
        output = self.root / "contradictory_heldout_output"

        with self.assertRaisesRegex(ValueError, "allowed_before_unlock|disjoint"):
            self._build(output_root=output, spec_path=inconsistent_path)
        self.assertFalse(output.exists())

    def test_rejects_heldout_model_selection_role(self) -> None:
        inconsistent = copy.deepcopy(self.spec)
        inconsistent["split_policy"]["model_selection_role"] = "heldout"
        inconsistent_path = self.root / "heldout_model_selection_spec.json"
        self._write_json(inconsistent_path, inconsistent)
        output = self.root / "heldout_model_selection_output"

        with self.assertRaisesRegex(ValueError, "model_selection_role"):
            self._build(output_root=output, spec_path=inconsistent_path)
        self.assertFalse(output.exists())

    def test_rejects_partial_review_coverage(self) -> None:
        inconsistent = copy.deepcopy(self.spec)
        inconsistent["review_contract"]["frame_coverage"] = "first_frame_only"
        inconsistent_path = self.root / "partial_review_spec.json"
        self._write_json(inconsistent_path, inconsistent)
        output = self.root / "partial_review_output"

        with self.assertRaisesRegex(ValueError, "frame_coverage"):
            self._build(output_root=output, spec_path=inconsistent_path)
        self.assertFalse(output.exists())

    def test_rejects_missing_extra_and_duplicate_component_decisions(self) -> None:
        missing = copy.deepcopy(self.component_decisions)
        missing["decisions"] = missing["decisions"][:1]
        missing_decisions, missing_spec = self._decision_variant(
            "missing", missing, expected_count=1
        )
        missing_output = self.root / "missing_output"
        with self.assertRaisesRegex(ValueError, "Missing required component decision"):
            self._build(
                output_root=missing_output,
                spec_path=missing_spec,
                component_decisions=missing_decisions,
            )
        self.assertFalse(missing_output.exists())

        extra = copy.deepcopy(self.component_decisions)
        extra["decisions"].append(self._extra_nonreview_decision())
        extra_decisions, extra_spec = self._decision_variant(
            "extra", extra, expected_count=3
        )
        extra_output = self.root / "extra_output"
        with self.assertRaisesRegex(ValueError, "Extra component decision"):
            self._build(
                output_root=extra_output,
                spec_path=extra_spec,
                component_decisions=extra_decisions,
            )
        self.assertFalse(extra_output.exists())

        duplicate = copy.deepcopy(self.component_decisions)
        duplicate["decisions"].append(copy.deepcopy(duplicate["decisions"][0]))
        duplicate_decisions, duplicate_spec = self._decision_variant(
            "duplicate", duplicate, expected_count=3
        )
        duplicate_output = self.root / "duplicate_output"
        with self.assertRaisesRegex(ValueError, "Duplicate component decision case_id"):
            self._build(
                output_root=duplicate_output,
                spec_path=duplicate_spec,
                component_decisions=duplicate_decisions,
            )
        self.assertFalse(duplicate_output.exists())

    def test_rejects_stale_component_analysis_and_incomplete_final_coverage(self) -> None:
        stale = copy.deepcopy(self.component_decisions)
        stale["decisions"][0]["analysis"]["raw_bbox"]["width"] = 39
        stale_decisions, stale_spec = self._decision_variant("stale", stale)
        stale_output = self.root / "stale_output"
        with self.assertRaisesRegex(ValueError, "Stale or mismatched.*analysis"):
            self._build(
                output_root=stale_output,
                spec_path=stale_spec,
                component_decisions=stale_decisions,
            )
        self.assertFalse(stale_output.exists())

        incomplete = copy.deepcopy(self.component_decisions)
        incomplete["decisions"][0]["final_component_decisions"] = []
        incomplete_decisions, incomplete_spec = self._decision_variant(
            "incomplete", incomplete
        )
        incomplete_output = self.root / "incomplete_output"
        with self.assertRaisesRegex(ValueError, "cover every current secondary component"):
            self._build(
                output_root=incomplete_output,
                spec_path=incomplete_spec,
                component_decisions=incomplete_decisions,
            )
        self.assertFalse(incomplete_output.exists())

        partial_blind = copy.deepcopy(self.component_decisions)
        partial_blind["decisions"][0]["reviews"][0]["component_decisions"] = []
        partial_blind_decisions, partial_blind_spec = self._decision_variant(
            "partial_blind", partial_blind
        )
        partial_blind_output = self.root / "partial_blind_output"
        with self.assertRaisesRegex(ValueError, r"reviews\[0\].*cover every current secondary"):
            self._build(
                output_root=partial_blind_output,
                spec_path=partial_blind_spec,
                component_decisions=partial_blind_decisions,
            )
        self.assertFalse(partial_blind_output.exists())

    def test_rejects_invalid_blind_reviews_and_final_rubric_values(self) -> None:
        variants: list[tuple[str, dict[str, object], str]] = []

        not_blind = copy.deepcopy(self.component_decisions)
        not_blind["decisions"][0]["reviews"][0]["blind_to_automatic_proposal"] = False
        variants.append(("not_blind", not_blind, "blind_to_automatic_proposal"))

        malformed_blind = copy.deepcopy(self.component_decisions)
        del malformed_blind["decisions"][0]["reviews"][0][
            "component_decisions"
        ][0]["notes"]
        variants.append(("malformed_blind", malformed_blind, "notes must be a string"))

        invalid_blind_confidence = copy.deepcopy(self.component_decisions)
        invalid_blind_confidence["decisions"][0]["reviews"][0][
            "component_decisions"
        ][0]["confidence"] = "LOW"
        variants.append(
            (
                "invalid_blind_confidence",
                invalid_blind_confidence,
                "confidence for KEEP/DROP must be HIGH or MEDIUM",
            )
        )

        unresolved = copy.deepcopy(self.component_decisions)
        unresolved["decisions"][0]["final_component_decisions"][0][
            "decision"
        ] = "UNRESOLVED"
        variants.append(("unresolved", unresolved, "KEEP or DROP"))

        low_confidence = copy.deepcopy(self.component_decisions)
        low_confidence["decisions"][0]["final_confidence"] = "LOW"
        variants.append(("low_confidence", low_confidence, "HIGH or MEDIUM"))

        invalid_rationale = copy.deepcopy(self.component_decisions)
        invalid_rationale["decisions"][0]["final_component_decisions"][0][
            "rationale_codes"
        ] = ["not_a_frozen_code"]
        variants.append(("invalid_rationale", invalid_rationale, "invalid final codes"))

        wrong_namespace = copy.deepcopy(self.component_decisions)
        wrong_namespace["decisions"][0]["final_component_decisions"][0][
            "rationale_codes"
        ] = ["K_TEMPORAL"]
        variants.append(("wrong_namespace", wrong_namespace, "invalid final codes"))

        for name, payload, message in variants:
            with self.subTest(name=name):
                decisions_path, spec_path = self._decision_variant(name, payload)
                output = self.root / f"{name}_output"
                with self.assertRaisesRegex(ValueError, message):
                    self._build(
                        output_root=output,
                        spec_path=spec_path,
                        component_decisions=decisions_path,
                    )
                self.assertFalse(output.exists())

    def test_rejects_component_decisions_hash_mismatch(self) -> None:
        changed = copy.deepcopy(self.component_decisions)
        changed["decisions"][0]["resolved_by"] = "changed_after_freeze"
        decisions_path, spec_path = self._decision_variant(
            "hash_mismatch", changed, update_hash=False
        )
        output = self.root / "component_hash_mismatch_output"

        with self.assertRaisesRegex(ValueError, "Component decisions SHA-256 mismatch"):
            self._build(
                output_root=output,
                spec_path=spec_path,
                component_decisions=decisions_path,
            )
        self.assertFalse(output.exists())

    def test_component_decisions_hash_is_optional_only_for_explicit_bootstrap_validation(self) -> None:
        bootstrap = copy.deepcopy(self.spec)
        bootstrap["source"]["component_decisions_status"] = (
            "pending_double_blind_review"
        )
        del bootstrap["source"]["component_decisions_sha256"]

        selections, totals = TOOL._validate_spec(
            bootstrap, require_component_decisions=False
        )
        self.assertEqual(len(selections), 2)
        self.assertEqual(totals["benchmark_frames"], 6)
        with self.assertRaisesRegex(ValueError, "frozen_complete"):
            TOOL._validate_spec(bootstrap)

        bootstrap_path = self.root / "bootstrap_without_decisions_hash.json"
        self._write_json(bootstrap_path, bootstrap)
        output = self.root / "bootstrap_final_build_output"
        with self.assertRaisesRegex(ValueError, "frozen_complete"):
            self._build(output_root=output, spec_path=bootstrap_path)
        self.assertFalse(output.exists())

        frozen_without_hash = copy.deepcopy(bootstrap)
        frozen_without_hash["source"]["component_decisions_status"] = "frozen_complete"
        with self.assertRaisesRegex(ValueError, "component_decisions_sha256"):
            TOOL._validate_spec(frozen_without_hash)

    def test_recomputes_consensus_and_rejects_adjudication_bypasses(self) -> None:
        def keep_row(row: dict[str, object]) -> None:
            row["decision"] = "KEEP"
            row["rationale_codes"] = ["K_CONTINUITY"]

        contradictory = copy.deepcopy(self.component_decisions)
        first = contradictory["decisions"][0]
        keep_row(first["reviews"][1]["component_decisions"][0])
        decisions_path, spec_path = self._decision_variant(
            "contradictory_without_adjudication", contradictory
        )
        with self.assertRaisesRegex(ValueError, "requires structured adjudication"):
            self._build(
                output_root=self.root / "contradictory_without_adjudication_output",
                spec_path=spec_path,
                component_decisions=decisions_path,
            )

        consensus_override = copy.deepcopy(self.component_decisions)
        first = consensus_override["decisions"][0]
        keep_row(first["final_component_decisions"][0])
        first["final_retained_component_ids"] = ["c001", "c002"]
        decisions_path, spec_path = self._decision_variant(
            "consensus_override", consensus_override
        )
        with self.assertRaisesRegex(ValueError, "overrides blind-review consensus"):
            self._build(
                output_root=self.root / "consensus_override_output",
                spec_path=spec_path,
                component_decisions=decisions_path,
            )

        adjudication_override = copy.deepcopy(contradictory)
        first = adjudication_override["decisions"][0]
        disputed_row = copy.deepcopy(first["reviews"][0]["component_decisions"][0])
        first["adjudication"] = {
            "reviewer_id": "adjudicator_c",
            "reviewer_identity_exception": None,
            "review_status": "complete",
            "case_integrity_resolution": self._integrity_record(first["frame_id"]),
            "component_decisions": [disputed_row],
            "case_notes": "Synthetic adjudication.",
        }
        keep_row(first["final_component_decisions"][0])
        first["final_retained_component_ids"] = ["c001", "c002"]
        decisions_path, spec_path = self._decision_variant(
            "adjudication_override", adjudication_override
        )
        with self.assertRaisesRegex(ValueError, "does not match adjudication"):
            self._build(
                output_root=self.root / "adjudication_override_output",
                spec_path=spec_path,
                component_decisions=decisions_path,
            )

        unnecessary_adjudication = copy.deepcopy(self.component_decisions)
        first = unnecessary_adjudication["decisions"][0]
        first["adjudication"] = {
            "reviewer_id": "adjudicator_c",
            "reviewer_identity_exception": None,
            "review_status": "complete",
            "case_integrity_resolution": self._integrity_record(first["frame_id"]),
            "component_decisions": [
                copy.deepcopy(first["final_component_decisions"][0])
            ],
            "case_notes": "Must not override consensus.",
        }
        decisions_path, spec_path = self._decision_variant(
            "unnecessary_adjudication", unnecessary_adjudication
        )
        with self.assertRaisesRegex(ValueError, "must be null"):
            self._build(
                output_root=self.root / "unnecessary_adjudication_output",
                spec_path=spec_path,
                component_decisions=decisions_path,
            )

    def test_requires_integrity_resolution_and_valid_review_evidence(self) -> None:
        variants: list[tuple[str, dict[str, object], str]] = []

        integrity_without_adjudication = copy.deepcopy(self.component_decisions)
        integrity_without_adjudication["decisions"][0]["reviews"][0][
            "case_integrity_decision"
        ] = self._integrity_record(
            integrity_without_adjudication["decisions"][0]["frame_id"],
            decision="INTEGRITY_ALERT",
        )
        variants.append(
            (
                "integrity_without_adjudication",
                integrity_without_adjudication,
                "requires structured adjudication",
            )
        )

        unresolved_final = copy.deepcopy(self.component_decisions)
        unresolved_final["decisions"][0][
            "case_integrity_resolution"
        ] = self._integrity_record(
            unresolved_final["decisions"][0]["frame_id"],
            decision="INTEGRITY_ALERT",
        )
        variants.append(
            (
                "unresolved_final_integrity",
                unresolved_final,
                "case_integrity_resolution must be OK",
            )
        )

        invalid_reviewer = copy.deepcopy(self.component_decisions)
        invalid_reviewer["decisions"][0]["reviews"][0]["reviewer_id"] = "Reviewer_A"
        variants.append(("invalid_reviewer", invalid_reviewer, "canonical lowercase"))

        duplicate_reviewer = copy.deepcopy(self.component_decisions)
        duplicate_reviewer["decisions"][0]["reviews"][1]["reviewer_id"] = (
            "blind_reviewer_a"
        )
        variants.append(("duplicate_reviewer", duplicate_reviewer, "distinct reviewers"))

        invalid_evidence = copy.deepcopy(self.component_decisions)
        invalid_evidence["decisions"][0]["reviews"][0]["component_decisions"][0][
            "evidence_frame_ids"
        ] = ["frame_9999"]
        variants.append(("invalid_evidence", invalid_evidence, "outside its stream"))

        invalid_final_evidence = copy.deepcopy(self.component_decisions)
        invalid_final_evidence["decisions"][0]["final_component_decisions"][0][
            "evidence_frame_ids"
        ] = ["frame_9999"]
        variants.append(
            ("invalid_final_evidence", invalid_final_evidence, "outside its stream")
        )

        for name, payload, message in variants:
            with self.subTest(name=name):
                decisions_path, spec_path = self._decision_variant(name, payload)
                output = self.root / f"{name}_output"
                with self.assertRaisesRegex(ValueError, message):
                    self._build(
                        output_root=output,
                        spec_path=spec_path,
                        component_decisions=decisions_path,
                    )
                self.assertFalse(output.exists())

    def test_adjudicator_must_be_independent_even_with_identity_exception(self) -> None:
        payload = copy.deepcopy(self.component_decisions)
        first = payload["decisions"][0]
        first["reviews"][1]["component_decisions"][0].update(
            {
                "decision": "KEEP",
                "rationale_codes": ["K_CONTINUITY"],
            }
        )
        first["reviews"][0]["case_integrity_decision"] = self._integrity_record(
            first["frame_id"], decision="INTEGRITY_ALERT"
        )
        first["adjudication"] = {
            "reviewer_id": "blind_reviewer_a",
            "reviewer_identity_exception": None,
            "review_status": "complete",
            "case_integrity_resolution": self._integrity_record(first["frame_id"]),
            "component_decisions": [
                copy.deepcopy(first["reviews"][0]["component_decisions"][0])
            ],
            "case_notes": "Synthetic same-person exception test.",
        }
        decisions_path, spec_path = self._decision_variant(
            "adjudicator_without_exception", payload
        )
        with self.assertRaisesRegex(ValueError, "distinct from both primary"):
            self._build(
                output_root=self.root / "adjudicator_without_exception_output",
                spec_path=spec_path,
                component_decisions=decisions_path,
            )

        first["adjudication"]["reviewer_identity_exception"] = (
            "Only reviewer available; exception recorded for audit."
        )
        decisions_path, spec_path = self._decision_variant(
            "adjudicator_with_exception", payload
        )
        output = self.root / "adjudicator_with_exception_output"
        with self.assertRaisesRegex(ValueError, "exceptions are not permitted"):
            self._build(
                output_root=output,
                spec_path=spec_path,
                component_decisions=decisions_path,
            )
        self.assertFalse(output.exists())

    def test_overlay_distinguishes_raw_and_final_bboxes(self) -> None:
        rgb_path = self.root / "overlay_source.png"
        Image.new("RGB", (50, 50), (20, 30, 40)).save(rgb_path)
        retained_mask = np.zeros((50, 50), dtype=np.bool_)
        retained_mask[20:28, 4:12] = True
        observation = TOOL.Observation(
            physical_instance_id="synthetic_instance",
            source_label=257,
            visible_pixels=64,
            visible_fraction=0.0256,
            raw_bbox=(4, 20, 30, 20),
            bbox=(4, 20, 8, 8),
            centroid=(7.5, 23.5),
            connected_component_count=2,
            border_touch=False,
            source_mask_sha256="0" * 64,
            component_review_required=True,
            automatic_retained_component_ids=("c001",),
            retained_component_ids=("c001",),
            removed_component_ids=("c002",),
            final_component_decisions=(),
            final_confidence="HIGH",
            resolved_by="primary_agreement",
            author_attention=True,
            retained_mask=retained_mask,
        )
        frame = TOOL.FrameData(
            frame_id="frame_0001",
            frame_index=1,
            source_filename="source.png",
            width=50,
            height=50,
            rgb_path=rgb_path,
            labels=np.zeros((50, 50), dtype=np.uint16),
            observations=(observation,),
        )
        output = self.root / "overlay.png"
        labels: list[str] = []
        original = TOOL._draw_text_label

        def record_label(*args, **kwargs):
            labels.append(str(args[2]))
            return original(*args, **kwargs)

        with mock.patch.object(TOOL, "_draw_text_label", side_effect=record_label):
            TOOL._render_overlay(frame, output)

        self.assertTrue(any(label.startswith("RAW source=") for label in labels))
        self.assertTrue(any(label.startswith("FINAL source=") for label in labels))
        with Image.open(output) as image:
            self.assertEqual(image.convert("RGB").getpixel((4, 39)), (0, 255, 255))

    def test_firewall_rejects_manifest_image_path_outside_stream(self) -> None:
        fixture_root = self.root / "firewall_fixture"
        analysis_root = fixture_root / "analysis_streams"
        stream_root = analysis_root / "ocid_safe_stream"
        frames_root = stream_root / "frames"
        frames_root.mkdir(parents=True)
        Image.new("RGB", (4, 3), (1, 2, 3)).save(frames_root / "frame_0001.png")
        private_root = fixture_root / "private"
        private_root.mkdir()
        Image.new("RGB", (4, 3), (255, 255, 255)).save(
            private_root / "frame_0001.png"
        )
        self._write_json(
            stream_root / "manifest.json",
            {
                "schema_version": "stream-input-0.1",
                "stream_id": "ocid_safe_stream",
                "ordering": "manifest",
                "frames": [
                    {
                        "frame_id": "frame_0001",
                        "index": 1,
                        "image_path": "../../private/frame_0001.png",
                    }
                ],
                "metadata": {"frame_size": {"width": 4, "height": 3}},
            },
        )

        try:
            TOOL._validate_analysis_firewall(analysis_root)
        except TOOL.GroundTruthFirewallError as error:
            self.assertRegex(str(error), "image_path|outside|frames")
        else:
            self.fail(
                "Ground-truth firewall accepted an image_path that resolves outside "
                "its analysis stream while a same-named decoy RGB frame was present."
            )


if __name__ == "__main__":
    unittest.main()
