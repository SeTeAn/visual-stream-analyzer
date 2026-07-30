from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from stream_analysis import BBox, ImageSize
from stream_analysis.candidates import CandidateMaskRecord, candidate_mask_digest
from tools import evaluate_ocid_masked_dinov2 as subject


def _named_input(
    candidate_id: str, frame_id: str, frame_index: int, *, stream: str = "stream",
    x: int = 10,
) -> subject.CandidateInput:
    source = SimpleNamespace(
        candidate_id=candidate_id, frame_id=frame_id, frame_index=frame_index,
        score=.7, bbox=BBox(x, 10, 10, 10),
    )
    mask_array = np.ones((10, 10), dtype=np.bool_)
    mask = CandidateMaskRecord(
        mask_ref=f"sha256:mask-{candidate_id}", mask_digest=candidate_mask_digest(mask_array),
        candidate_id=candidate_id, frame_id=frame_id,
        coordinate_bbox=source.bbox, mask=mask_array,
    )
    candidate = subject._candidate_from_selected(
        source, stream_id=stream, size=ImageSize(100, 100),
        frame_record_index=frame_index, mask=mask,
    )
    return subject.CandidateInput(stream, candidate, mask)


def _unit(value: int) -> np.ndarray:
    angle = value / 1000.0
    result = np.zeros(subject.FROZEN_EMBEDDING_DIMENSION, dtype=np.float32)
    result[0] = math.cos(angle)
    result[1] = math.sin(angle)
    result /= np.linalg.norm(result)
    return result


class FakeProvider:
    def __init__(self) -> None:
        self.batch_size = subject.FROZEN_BATCH_SIZE
        self.model_spec = SimpleNamespace(
            model_name=subject.FROZEN_MODEL_NAME,
            embedding_dimension=subject.FROZEN_EMBEDDING_DIMENSION,
            expected_checkpoint_size_bytes=123,
        )
        self.observed_batches: list[int] = []

    def embed_batch(self, batch: np.ndarray):
        self.observed_batches.append(len(batch))
        rows = np.stack([_unit(int(round(float(row[0, 0, 0])))) for row in batch])
        return SimpleNamespace(embeddings=rows)

    def provider_metadata(self):
        return {"fake": True, "batch_size": self.batch_size}

    def model_metadata(self):
        return {"fake": True, "model_name": subject.FROZEN_MODEL_NAME}


class OcidMaskedDinoTest(unittest.TestCase):
    def _synthetic_input_contract(self, root: Path, *, candidate_id: str = "a"):
        selected_path = root / "selected.json"
        selected_path.write_text("{}", encoding="utf-8")
        selected = SimpleNamespace(
            candidate_id=candidate_id, frame_id="frame_001", frame_index=0,
            score=.7, bbox=BBox(0, 0, 10, 10),
        )
        stream = SimpleNamespace(stream_id="stream", stream_directory=root, frame_count=1)
        frame = SimpleNamespace(
            frame_id="frame_001", image_size=ImageSize(10, 10), record=SimpleNamespace(index=1),
        )
        return selected_path, selected, stream, SimpleNamespace(frames=(frame,))

    def _sam_manifest(
        self, selected_path: Path, *, heldout: str = "none", candidate_id: str = "a",
        hash_value: str = "0" * 64, schema: str | None = None,
    ) -> dict:
        return {
            "schema_version": subject.SAM2_INFERENCE_SCHEMA if schema is None else schema,
            "status": "rgb_inference_completed_before_ground_truth",
            "scope": "development_only", "heldout_access": heldout,
            "selected_candidates": {"sha256": subject._sha256(selected_path)},
            "records_by_stream": {"stream": [{
                "candidate_id": candidate_id, "stream_id": "stream", "frame_id": "frame_001",
                "status": "valid", "source_bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                "cleaned_mask": {"path": "masks/mask.png", "binary_mask_sha256": hash_value},
            }]},
        }

    def _load_contract(self, *, root: Path, selected_path: Path, sam_path: Path, stream, selected, decoded):
        with mock.patch.object(subject, "FROZEN_DEVELOPMENT_STREAM_COUNT", 1), \
             mock.patch.object(subject, "FROZEN_CANDIDATE_COUNT", 1), \
             mock.patch.object(subject, "FROZEN_VALID_MASK_COUNT", 1), \
             mock.patch.object(subject, "load_selected_candidates", return_value=({"stream": (selected,)}, {})), \
             mock.patch.object(subject, "_load_stream", return_value=decoded):
            return subject.load_predicted_candidate_inputs(
                inventory=(stream,), selected_candidates_path=selected_path,
                sam2_inference_manifest_path=sam_path,
                expected_selected_candidates_sha256=subject._sha256(selected_path),
                expected_sam2_inference_sha256=subject._sha256(sam_path),
            )

    def test_frozen_constants_describe_vitb14(self) -> None:
        self.assertEqual(subject.FROZEN_MODEL_NAME, "dinov2_vitb14")
        self.assertEqual(subject.FROZEN_EMBEDDING_DIMENSION, 768)
        self.assertEqual(subject.FROZEN_BATCH_SIZE, 4)
        self.assertEqual(subject.FROZEN_CANDIDATE_COUNT, 1060)
        self.assertEqual(subject.OCCLUSION_SOURCE_LABEL, 10)
        self.assertEqual(subject.OCCLUSION_FLAG_FRAME, "frame_0020")
        self.assertEqual(
            subject.OCCLUSION_SOURCE_SEQUENCE, "ARID20/floor/bottom/seq12"
        )

    def test_model_constructor_receives_every_frozen_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifacts"; artifact.mkdir()
            (artifact / "inference_manifest.json").write_text("{}", encoding="utf-8")
            inventory = tuple(SimpleNamespace(stream_id=f"s{i}") for i in range(10))
            inputs = {item.stream_id: () for item in inventory}
            with mock.patch.object(subject, "load_development_inventory", return_value=inventory), \
                 mock.patch.object(subject, "load_predicted_candidate_inputs", return_value=(inputs, {}, {})), \
                 mock.patch.object(subject, "LocalDinoV2Provider") as provider_class, \
                 mock.patch.object(subject, "run_predicted_mask_inference", return_value={}), \
                 mock.patch.object(subject, "evaluate_persisted_inference", return_value={}), \
                 mock.patch.object(subject, "write_canonical_json_atomic"), \
                 mock.patch.object(subject, "_sha256", return_value="f" * 64):
                subject.run_ocid_representation_evaluation(
                    selected_candidates_path=root / "selected.json", sam2_inference_manifest_path=root / "sam.json",
                    expected_selected_candidates_sha256="a" * 64, expected_sam2_inference_sha256="b" * 64,
                    benchmark_spec_path=root / "spec.json", reviewed_root=root / "reviewed",
                    dino_source=root / "source", dino_checkpoint=root / "checkpoint",
                    expected_checkpoint_sha256="c" * 64, expected_checkpoint_size_bytes=999,
                    expected_source_fingerprint="d" * 64, device="cuda",
                    artifact_root=artifact, output_path=root / "report.json",
                )
            kwargs = provider_class.call_args.kwargs
            self.assertEqual(kwargs["model_name"], "dinov2_vitb14")
            self.assertEqual(kwargs["embedding_dimension"], 768)
            self.assertEqual(kwargs["batch_size"], 4)
            self.assertEqual(kwargs["expected_checkpoint_size_bytes"], 999)

    def test_chunked_inference_never_exceeds_four_and_preserves_order(self) -> None:
        inventory = tuple(SimpleNamespace(stream_id=f"s{i}", stream_directory=Path(f"s{i}")) for i in range(10))
        inputs: dict[str, tuple[subject.CandidateInput, ...]] = {}
        counter = 0
        for stream in inventory:
            rows = []
            for _ in range(106):
                rows.append(_named_input(f"c{counter:04d}", "frame_000", 0, stream=stream.stream_id))
                counter += 1
            inputs[stream.stream_id] = tuple(rows)
        fake_frame = SimpleNamespace(frame_id="frame_000")
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected.json"; selected.write_text("{}", encoding="utf-8")
            sam = root / "sam.json"; sam.write_text("{}", encoding="utf-8")
            def preprocess(_frame, item, **_kwargs):
                index = int(item.candidate_id[1:])
                return SimpleNamespace(normalized_chw=np.full((3, 1, 1), index, dtype=np.float32))
            with mock.patch.object(subject, "_load_stream", return_value=SimpleNamespace(frames=(fake_frame,))), \
                 mock.patch.object(subject, "preprocess_dinov2_candidate", side_effect=preprocess):
                manifest = subject.run_predicted_mask_inference(
                    inventory=inventory, inputs=inputs, provider=provider, artifact_root=root,
                    selected_candidates_path=selected, sam2_inference_manifest_path=sam,
                )
            self.assertLessEqual(max(provider.observed_batches), 4)
            self.assertEqual(len(provider.observed_batches), 530)
            self.assertEqual(manifest["runtime"]["by_variant"][subject.DINO_BBOX_VARIANT]["batch_count"], 265)
            with np.load(root / "embeddings.npz", allow_pickle=False) as archive:
                self.assertEqual(archive[subject.DINO_BBOX_VARIANT].shape, (1060, 768))
                self.assertTrue(np.allclose(archive[subject.DINO_BBOX_VARIANT][0], _unit(0)))
                self.assertTrue(np.allclose(archive[subject.DINO_BBOX_VARIANT][-1], _unit(1059)))

    def test_artifact_path_cannot_escape_sam2_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "masks").mkdir(); (root / "masks" / "one.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "remain under"):
                subject._under(root, "../outside.png", "mask")

    def test_input_hash_is_checked_before_manifest_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selected = root / "selected"; selected.write_text("x", encoding="utf-8")
            sam = root / "sam"; sam.write_text("not-json", encoding="utf-8")
            inventory = tuple(SimpleNamespace() for _ in range(10))
            with self.assertRaisesRegex(ValueError, "selected candidates SHA-256 mismatch"):
                subject.load_predicted_candidate_inputs(
                    inventory=inventory, selected_candidates_path=selected, sam2_inference_manifest_path=sam,
                    expected_selected_candidates_sha256="0" * 64,
                    expected_sam2_inference_sha256=subject._sha256(sam),
                )

    def test_input_contract_rejects_heldout_candidate_and_mask_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "masks").mkdir()
            full = np.ones((10, 10), dtype=np.uint8) * 255
            Image.fromarray(full, mode="L").save(root / "masks" / "mask.png")
            selected_path, selected, stream, decoded = self._synthetic_input_contract(root)
            sam = root / "sam.json"
            sam.write_text(json.dumps(self._sam_manifest(selected_path, heldout="accessed")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configured inference"):
                self._load_contract(root=root, selected_path=selected_path, sam_path=sam,
                                    stream=stream, selected=selected, decoded=decoded)
            sam.write_text(json.dumps(self._sam_manifest(selected_path, candidate_id="other")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate set mismatch"):
                self._load_contract(root=root, selected_path=selected_path, sam_path=sam,
                                    stream=stream, selected=selected, decoded=decoded)
            sam.write_text(json.dumps(self._sam_manifest(selected_path)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binary hash mismatch"):
                self._load_contract(root=root, selected_path=selected_path, sam_path=sam,
                                    stream=stream, selected=selected, decoded=decoded)

    def test_evaluation_boundary_rejects_before_loading_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "inference.json"
            bad.write_text(json.dumps({"schema_version": subject.INFERENCE_SCHEMA_VERSION, "status": "wrong"}), encoding="utf-8")
            with mock.patch.object(subject, "load_annotation") as annotations:
                with self.assertRaisesRegex(ValueError, "persisted OCID representation evaluation inference boundary"):
                    subject.evaluate_persisted_inference(inventory=(), inputs={}, inference_manifest_path=bad)
            annotations.assert_not_called()

    def test_candidate_index_and_npz_tampering_are_rejected(self) -> None:
        item = _named_input("a", "frame_001", 1)
        row = subject._candidate_index_row(0, item)
        subject._validate_candidate_index([row], [item])
        changed = dict(row); changed["candidate_id"] = "other"
        with self.assertRaisesRegex(ValueError, "candidate_id mismatch"):
            subject._validate_candidate_index([changed], [item])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.npz"
            valid = np.stack([_unit(0)])
            np.savez(path, **{subject.DINO_BBOX_VARIANT: valid,
                              subject.DINO_MASK_NEUTRAL_VARIANT: valid, "extra": valid})
            with self.assertRaisesRegex(ValueError, "exactly the two"):
                subject._load_embedding_npz(path, 1)
            np.savez(path, **{subject.DINO_BBOX_VARIANT: np.zeros((1, 768), np.float32),
                              subject.DINO_MASK_NEUTRAL_VARIANT: valid})
            with self.assertRaisesRegex(ValueError, "unit-normalized"):
                subject._load_embedding_npz(path, 1)

    def test_false_positive_gets_unique_identity(self) -> None:
        values = (_named_input("a", "frame_001", 1), _named_input("b", "frame_001", 1))
        evaluation = {"assignments": [{"candidate_id": "a", "visual_type_id": "id3", "iou": .8}]}
        with mock.patch.object(subject, "_evaluate_candidates", return_value=evaluation) as evaluator:
            identities, ious, returned = subject._assignment_identity(
                object(),
                values,
                iou_threshold=0.70,
            )
        self.assertEqual(identities["b"], "fp:stream:b")
        self.assertEqual(ious["a"], .8)
        self.assertIs(returned, evaluation)
        self.assertEqual(evaluator.call_args.kwargs["iou_threshold"], 0.70)

    def test_mask_pair_slices_have_scores_and_reference_rates(self) -> None:
        a0 = _named_input("a0", "f0", 0); a1 = _named_input("a1", "f1", 1)
        values = subject._positive_pair_mask_slice_scores(
            (a0, a1), {"a0": "same", "a1": "same"},
            {"a0": _unit(0), "a1": _unit(10)},
            {("stream", "a0"): .85, ("stream", "a1"): .75},
        )
        report = subject._summarize_mask_slice_scores(values)
        self.assertEqual(report["0.70-0.80"]["positive_pair_count"], 1)
        self.assertEqual(report["0.70-0.80"]["reference_pass_count"], 1)
        self.assertIsNotNone(report["0.70-0.80"]["score_distribution"]["mean"])

    def test_mask_iou_keys_include_stream_to_prevent_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_root = root / "grounded_sam2"
            manifest_root.mkdir()
            manifest = manifest_root / "inference_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            report = {
                "inference_manifest": {"sha256": subject._sha256(manifest)},
                "matched_observations": [
                    {"stream_id": "first", "candidate_id": "repeated", "cleaned": {"iou": .61}},
                    {"stream_id": "second", "candidate_id": "repeated", "cleaned": {"iou": .94}},
                ],
            }
            (root / "grounded_sam2_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            values = subject._mask_iou_by_candidate(
                {"sam2_inference_manifest": {"path": str(manifest)}}
            )
        self.assertEqual(values[("first", "repeated")], .61)
        self.assertEqual(values[("second", "repeated")], .94)
        self.assertEqual(len(values), 2)

    def _variant_aggregates(self, *, bbox_f1=.80, mask_f1=.81, bbox_ap=.70, mask_ap=.71,
                            bbox_margin=.10, mask_margin=.11):
        def one(f1, ap, margin):
            return {"aggregate": {
                "coverage": {"candidate_count": 1060, "valid_embedding_count": 1060,
                             "embedding_failures": 0, "hidden_fallback_count": 0},
                "scene_group_macro": {"accepted": {"f1": f1}},
                "worst_scene_group": {"accepted_f1": f1 - .05},
                "pairwise": {"ranking": {"average_precision": ap},
                             "hard_negative_margin": {"aggregate": {"mean": margin}}},
            }}
        return {subject.DINO_BBOX_VARIANT: one(bbox_f1, bbox_ap, bbox_margin),
                subject.DINO_MASK_NEUTRAL_VARIANT: one(mask_f1, mask_ap, mask_margin)}

    def test_recommendation_rule_selects_mask_only_when_all_guards_pass(self) -> None:
        selected = subject._recommendation(self._variant_aggregates())
        self.assertEqual(selected["selected_variant"], subject.DINO_MASK_NEUTRAL_VARIANT)
        fallback = subject._recommendation(self._variant_aggregates(mask_f1=.77))
        self.assertEqual(fallback["selected_variant"], subject.DINO_BBOX_VARIANT)
        tie = subject._recommendation(self._variant_aggregates(mask_f1=.799, mask_ap=.69))
        self.assertEqual(tie["selected_variant"], subject.DINO_BBOX_VARIANT)
        hidden_fallback = self._variant_aggregates()
        for value in hidden_fallback.values():
            value["aggregate"]["coverage"]["hidden_fallback_count"] = 1
        rejected = subject._recommendation(hidden_fallback)
        self.assertEqual(rejected["selected_variant"], subject.DINO_BBOX_VARIANT)
        self.assertFalse(rejected["checks"]["coverage_equal_and_no_hidden_fallback"])

    def test_pair_report_is_pooled_and_deterministic(self) -> None:
        rows = (
            subject.PairRankingRecord("q1", "g1", True, .9),
            subject.PairRankingRecord("q1", "g2", False, .2),
        )
        report = subject._pair_report(rows)
        self.assertEqual(report["ranking"]["pair_count"], 2)
        self.assertEqual(report["ranking"]["average_precision"], 1.0)
        self.assertGreater(report["hard_negative_margin"]["aggregate"]["mean"], 0)

    def test_marker_ranks_continuity_separation_and_boundary(self) -> None:
        items = (
            _named_input("m3_18", "frame_0018", 18), _named_input("m9_18", "frame_0018", 18, x=30),
            _named_input("m3_19", "frame_0019", 19), _named_input("m9_19", "frame_0019", 19, x=30),
            _named_input("other", "frame_0019", 19, x=50),
        )
        rows = [
            {"frame_id": frame, "source_label": label, "visual_type_id": f"id{label}"}
            for frame in ("frame_0018", "frame_0019") for label in (3, 9)
        ]
        annotation = SimpleNamespace(raw={"expected_element_instances": rows},
                                     frame_ids=("frame_0018", "frame_0019"))
        identities = {item.candidate.candidate_id: (
            "id3" if item.candidate.candidate_id.startswith("m3") else
            "id9" if item.candidate.candidate_id.startswith("m9") else "other"
        ) for item in items}
        embeddings = {
            "m3_18": _unit(0), "m3_19": _unit(5), "m9_18": _unit(10),
            "m9_19": _unit(15), "other": _unit(1000),
        }
        result = subject._marker_diagnostic(
            annotation, items, embeddings, identities,
            {key: .8 for key in embeddings},
            {("stream", key): .9 for key in embeddings},
        )
        self.assertTrue(result["reference_frame_reviewed_labels_verified"])
        self.assertEqual(result["same_instance_adjacent_similarity"]["003"]["count"], 1)
        self.assertEqual(len(result["co_visible_frame_ranks_and_margins"]), 4)
        self.assertEqual(result["per_frame_continuity"]["009"][1]["detector_status"], "tp")
        self.assertTrue(result["separation_interval"]["exists"])
        self.assertIn("Hypothesis-only", result["claim_boundary"])

    def test_marker_reference_frame_must_have_both_reviewed_labels(self) -> None:
        annotation = SimpleNamespace(raw={"expected_element_instances": [
            {"frame_id": "frame_0019", "source_label": 3, "visual_type_id": "id3"},
        ]}, frame_ids=("frame_0019",))
        with self.assertRaisesRegex(ValueError, "frame_0019"):
            subject._marker_diagnostic(annotation, (), {}, {}, {}, {})

    def test_partial_occlusion_diagnostic_separates_extractor_miss(self) -> None:
        first = _named_input("target_18", "frame_0018", 18, stream=subject.MARKER_STREAM)
        second = _named_input("target_19", "frame_0019", 19, stream=subject.MARKER_STREAM)
        annotation = SimpleNamespace(
            raw={"expected_element_instances": [
                {"frame_id": frame, "source_label": 10, "visual_type_id": "id10"}
                for frame in ("frame_0018", "frame_0019", "frame_0020")
            ]},
            frame_ids=("frame_0018", "frame_0019", "frame_0020"),
        )
        visibility = {"trajectory": [
            {"frame_id": "frame_0018", "visible_pixels": 100, "relative_to_instance_max": 1.0},
            {"frame_id": "frame_0019", "visible_pixels": 60, "relative_to_instance_max": .6},
            {"frame_id": "frame_0020", "visible_pixels": 5, "relative_to_instance_max": .05},
        ]}
        result = subject._partial_occlusion_diagnostic(
            annotation,
            (first, second),
            {"target_18": _unit(0), "target_19": _unit(10)},
            {"target_18": "id10", "target_19": "id10"},
            {"target_18": .9, "target_19": .8},
            {
                (subject.MARKER_STREAM, "target_18"): .85,
                (subject.MARKER_STREAM, "target_19"): .75,
            },
            visibility,
        )
        self.assertEqual(result["expected_observation_count"], 3)
        self.assertEqual(result["detected_observation_count"], 2)
        self.assertEqual(result["missed_frames"], ["frame_0020"])
        self.assertEqual(result["flagged_frame_detector_status"], "miss")
        self.assertEqual(result["same_instance_adjacent_similarity"]["count"], 1)
        self.assertEqual(result["adjacent_reference_pass_count"], 1)

    def test_visibility_proxy_requires_preexisting_development_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ocid_root = Path(temporary) / "data" / "ocid"
            reviewed = ocid_root / "derived" / "reviewed"
            stream_root = reviewed / "analysis_streams" / subject.MARKER_STREAM
            annotation_root = reviewed / "evaluation_annotations" / subject.MARKER_STREAM
            timeline_root = reviewed / "review" / "timelines"
            label_root = (
                ocid_root / "raw" / "OCID-dataset" / "ARID20" / "floor"
                / "bottom" / "seq12" / "label"
            )
            for path in (stream_root, annotation_root, timeline_root, label_root):
                path.mkdir(parents=True, exist_ok=True)
            annotation_path = annotation_root / "annotation.json"
            annotation_path.write_text("{}", encoding="utf-8")
            frames = (
                ("frame_0019", "source_19.png"),
                ("frame_0020", "source_20.png"),
            )
            (stream_root / "manifest.json").write_text(
                json.dumps({"frames": [
                    {"frame_id": frame, "metadata": {"ocid_source_filename": filename}}
                    for frame, filename in frames
                ]}),
                encoding="utf-8",
            )
            previous = np.ones((10, 10), dtype=np.uint8) * 10
            current = np.ones((10, 10), dtype=np.uint8) * 11
            current[0, :5] = 10
            Image.fromarray(previous, mode="L").save(label_root / "source_19.png")
            Image.fromarray(current, mode="L").save(label_root / "source_20.png")
            target_id = f"{subject.MARKER_STREAM}__label_010"
            (timeline_root / f"{subject.MARKER_STREAM}.csv").write_text(
                "frame_id,severe_interframe_visibility_drop_ids,automatic_flags\n"
                f"frame_0019,,\n"
                f"frame_0020,{target_id},relative_visible_area_below_0_10;severe_interframe_visibility_drop\n",
                encoding="utf-8",
            )
            annotation = SimpleNamespace(
                path=annotation_path,
                frame_ids=tuple(frame for frame, _filename in frames),
                raw={
                    "metadata": {
                        "role": "development",
                        "source_sequence": "ARID20/floor/bottom/seq12",
                        "audited_support_label": 1,
                    },
                    "expected_element_instances": [
                        {"frame_id": frame, "source_label": 10}
                        for frame, _filename in frames
                    ],
                },
            )
            stream = SimpleNamespace(
                stream_id=subject.MARKER_STREAM,
                stream_directory=stream_root,
            )
            result = subject._source_label_visibility_proxy(stream, annotation)
            annotation.raw["metadata"]["source_sequence"] = "ARID20/table/bottom/seq02"
            with self.assertRaisesRegex(ValueError, "frozen development target"):
                subject._source_label_visibility_proxy(stream, annotation)
        self.assertEqual(result["maximum_visible_pixels"], 100)
        self.assertEqual(result["flagged_visible_pixels"], 5)
        self.assertEqual(result["flagged_relative_to_instance_max"], .05)
        self.assertEqual(
            result["final_transition"]["lost_pixels_reassigned_to_other_object_labels"], 95
        )

    def test_marker_contact_sheet_uses_rgb_and_predicted_mask(self) -> None:
        first = _named_input("m3", "frame_0019", 19, stream=subject.MARKER_STREAM)
        second = _named_input("m9", "frame_0019", 19, stream=subject.MARKER_STREAM, x=30)
        annotation = SimpleNamespace(raw={"expected_element_instances": [
            {"frame_id": "frame_0019", "source_label": 3, "visual_type_id": "id3"},
            {"frame_id": "frame_0019", "source_label": 9, "visual_type_id": "id9"},
        ]})
        frame = SimpleNamespace(frame_id="frame_0019", image_size=ImageSize(100, 100),
                                rgb_bytes=bytes(100 * 100 * 3))
        with tempfile.TemporaryDirectory() as temporary:
            result = subject._write_marker_contact_sheet(
                decoded=SimpleNamespace(frames=(frame,)), items=(first, second),
                identities={"m3": "id3", "m9": "id9"}, annotation=annotation,
                artifact_root=Path(temporary),
            )
            self.assertTrue((Path(temporary) / result["relative_path"]).is_file())
            self.assertEqual(len(result["sha256"]), 64)

    def test_help_never_runs_inference(self) -> None:
        with mock.patch.object(subject, "run_ocid_representation_evaluation") as runner:
            with self.assertRaises(SystemExit) as error, contextlib.redirect_stdout(io.StringIO()):
                subject.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
