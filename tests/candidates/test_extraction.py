import hashlib
import unittest
from pathlib import Path

import numpy as np

from stream_analysis import (
    BBox,
    CandidateExtractionConfig,
    CandidateExtractionResult,
    DataProvenance,
    DecodedFrame,
    DecodedStream,
    Fingerprint,
    FrameCandidateDiagnostics,
    FrameRecord,
    HysteresisMaskConfig,
    ImageSize,
    ImageStream,
    ManifestFrameEntry,
    ManifestLoadRequest,
    ManifestLoadResult,
    MorphologyCleanupConfig,
    ProducerProvenance,
    RecordEnvelope,
    StageConfig,
    StageContext,
    BackgroundStrategy,
    extract_candidates,
    load_decoded_stream,
    semantic_config_digest,
)
from stream_analysis.candidates import CandidateExtractionSnapshot, candidate_mask_digest


def _base_frame(height: int = 16, width: int = 18) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 25
    image[:, :, 1] = 45
    image[:, :, 2] = 70
    return image


def _config(**kwargs: object) -> CandidateExtractionConfig:
    values = {
        "hysteresis": HysteresisMaskConfig(
            weak_threshold=5.0,
            strong_threshold=10.0,
            connectivity=8,
        ),
        "morphology": MorphologyCleanupConfig(),
        "min_component_area_ratio": 0.0,
        "bbox_padding": 0,
        "border_touching_policy": "allow",
    }
    values.update(kwargs)
    return CandidateExtractionConfig(**values)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:synthetic-input",
    )


def _fingerprint(data: bytes) -> Fingerprint:
    return Fingerprint(algorithm="sha256", value=hashlib.sha256(data).hexdigest())


def _decoded_stream(frames: list[np.ndarray], *, stream_id: str = "synthetic_stream") -> DecodedStream:
    if not frames:
        raise ValueError("frames must not be empty")
    root = Path.cwd().resolve()
    producer = _producer()
    manifest_frames: list[ManifestFrameEntry] = []
    decoded_frames: list[DecodedFrame] = []
    frame_records: list[FrameRecord] = []
    frame_digests: dict[str, Fingerprint] = {}
    height, width, _channels = frames[0].shape

    for index, image in enumerate(frames):
        frame_id = f"frame_{index + 1:03d}"
        image_size = ImageSize(width=width, height=height)
        relative_path = f"frames/{frame_id}.png"
        resolved_path = root / relative_path
        manifest_frames.append(
            ManifestFrameEntry(
                frame_id=frame_id,
                index=index,
                manifest_position=index,
                image_path=relative_path,
                resolved_path=resolved_path,
            )
        )
        envelope = RecordEnvelope(
            record_id=frame_id,
            schema_version="frame-record-1.0",
            stream_id=stream_id,
            producer=producer,
            context=StageContext(frame_id=frame_id),
        )
        record = FrameRecord(
            envelope=envelope,
            frame_id=frame_id,
            index=index,
            image_path=relative_path,
            image_size=image_size,
        )
        frame_records.append(record)
        rgb_bytes = image.astype(np.uint8, copy=True).tobytes()
        digest = _fingerprint(rgb_bytes)
        frame_digests[frame_id] = digest
        decoded_frames.append(
            DecodedFrame(
                record=record,
                rgb_bytes=rgb_bytes,
                raw_digest=digest,
                image_format="PNG",
                resolved_path=resolved_path,
            )
        )

    manifest = ManifestLoadResult(
        stream_root=root,
        manifest_path=root / "synthetic_manifest.json",
        manifest_digest=_fingerprint(b"synthetic-manifest"),
        producer=producer,
        stream_id=stream_id,
        ordering="manifest",
        frames=tuple(manifest_frames),
    )
    stream = ImageStream(
        envelope=RecordEnvelope(
            record_id=stream_id,
            schema_version="image-stream-1.0",
            stream_id=stream_id,
            producer=producer,
            context=StageContext(),
        ),
        frames=tuple(frame_records),
    )
    return DecodedStream(
        manifest=manifest,
        stream=stream,
        frames=tuple(decoded_frames),
        data_provenance=DataProvenance(
            manifest_digest=manifest.manifest_digest,
            frame_content_digests=frame_digests,
            candidate_snapshot_digest=None,
        ),
    )


class CandidateExtractionTest(unittest.TestCase):
    def test_empty_foreground_produces_diagnostics_and_warning(self) -> None:
        base = _base_frame()
        snapshot = extract_candidates(
            _decoded_stream([base, base.copy(), base.copy()]),
            _config(),
        )

        self.assertIsInstance(snapshot, CandidateExtractionSnapshot)
        self.assertIsInstance(snapshot.result, CandidateExtractionResult)
        self.assertEqual(snapshot.result.candidates, ())
        self.assertEqual(len(snapshot.result.frame_diagnostics), 3)
        self.assertEqual(
            {warning.code for warning in snapshot.result.warnings},
            {"NO_CANDIDATES_IN_FRAME"},
        )
        self.assertTrue(
            all(
                diagnostic.summary["accepted_candidate_count"] == 0
                for diagnostic in snapshot.result.frame_diagnostics
            )
        )

    def test_single_object_produces_one_candidate_record(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[4:8, 5:10] = (230, 20, 20)
        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            _config(),
        )

        candidates = [item for item in snapshot.result.candidates if item.frame_id == "frame_002"]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.candidate_id, "cand_frame_002_001")
        self.assertEqual(candidate.bbox, BBox(5, 4, 5, 4))
        self.assertEqual(candidate.center, candidate.bbox.center)
        self.assertIsNotNone(candidate.mask)
        self.assertIn(candidate.mask.mask_ref, snapshot.masks)
        self.assertEqual(candidate.mask.coordinate_bbox, candidate.bbox)

    def test_multiple_objects_are_sorted_deterministically(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[10:14, 2:6] = (230, 20, 20)
        object_frame[2:5, 11:15] = (20, 230, 20)
        stream = _decoded_stream([object_frame, base, base.copy()])

        first = extract_candidates(stream, _config()).result.candidates
        second = extract_candidates(stream, _config()).result.candidates

        self.assertEqual(tuple(item.candidate_id for item in first), tuple(item.candidate_id for item in second))
        self.assertEqual([item.candidate_id for item in first if item.frame_id == "frame_001"], [
            "cand_frame_001_001",
            "cand_frame_001_002",
        ])
        self.assertEqual(first[0].bbox, BBox(11, 2, 4, 3))
        self.assertEqual(first[1].bbox, BBox(2, 10, 4, 4))

    def test_bbox_uses_half_open_semantics_and_padding_is_clipped(self) -> None:
        base = _base_frame(height=10, width=10)
        object_frame = base.copy()
        object_frame[0:3, 0:3] = (230, 20, 20)
        snapshot = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(bbox_padding=2, border_touching_policy="allow"),
        )

        candidate = snapshot.result.candidates[0]
        self.assertEqual(candidate.bbox, BBox(0, 0, 5, 5))
        self.assertTrue(candidate.bbox.is_within(ImageSize(10, 10)))

    def test_small_components_are_rejected(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[5, 5] = (230, 20, 20)
        snapshot = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(min_component_area_ratio=0.01),
        )

        diagnostic = snapshot.result.frame_diagnostics[0]
        self.assertEqual(snapshot.result.candidates, ())
        self.assertEqual(diagnostic.summary["rejected_small_count"], 1)
        self.assertEqual(diagnostic.summary["accepted_candidate_count"], 0)

    def test_giant_component_policy_rejects_or_marks_component(self) -> None:
        base = _base_frame(height=10, width=10)
        object_frame = base.copy()
        object_frame[:, :] = (230, 20, 20)

        rejected = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(
                background_strategy="stream_model",
                giant_component_area_ratio=0.5,
                giant_component_policy="reject",
            ),
        )
        self.assertEqual(rejected.result.candidates, ())
        self.assertIn("GIANT_COMPONENT_REJECTED", {warning.code for warning in rejected.result.warnings})
        self.assertEqual(rejected.result.frame_diagnostics[0].summary["rejected_giant_count"], 1)

        marked = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(
                background_strategy="stream_model",
                giant_component_area_ratio=0.5,
                giant_component_policy="warn",
            ),
        )
        self.assertEqual(len(marked.result.candidates), 1)
        self.assertIn("GIANT_COMPONENT", marked.result.candidates[0].quality_flags)

    def test_border_touching_policy_rejects_component(self) -> None:
        base = _base_frame(height=12, width=12)
        object_frame = base.copy()
        object_frame[2:6, 0:3] = (230, 20, 20)
        snapshot = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(border_touching_policy="reject"),
        )

        self.assertEqual(snapshot.result.candidates, ())
        self.assertIn(
            "BORDER_TOUCHING_COMPONENT_REJECTED",
            {warning.code for warning in snapshot.result.warnings},
        )
        self.assertEqual(snapshot.result.frame_diagnostics[0].summary["rejected_border_count"], 1)

    def test_giant_warning_is_not_renamed_when_border_policy_rejects(self) -> None:
        base = _base_frame(height=10, width=10)
        object_frame = base.copy()
        object_frame[:, :] = (230, 20, 20)

        snapshot = extract_candidates(
            _decoded_stream([object_frame, base, base.copy()]),
            _config(
                background_strategy="stream_model",
                giant_component_area_ratio=0.5,
                giant_component_policy="warn",
                border_touching_policy="reject",
            ),
        )

        codes = {warning.code for warning in snapshot.result.warnings}
        self.assertEqual(snapshot.result.candidates, ())
        self.assertIn("GIANT_COMPONENT", codes)
        self.assertNotIn("GIANT_COMPONENT_REJECTED", codes)
        self.assertIn("BORDER_TOUCHING_COMPONENT_REJECTED", codes)
        diagnostic = snapshot.result.frame_diagnostics[0].summary
        self.assertEqual(diagnostic["rejected_giant_count"], 0)
        self.assertEqual(diagnostic["rejected_border_count"], 1)

    def test_warning_policy_is_not_renamed_by_small_area_rejection(self) -> None:
        base = _base_frame(height=10, width=10)

        giant_warning_frame = base.copy()
        giant_warning_frame[4:6, 4:6] = (230, 20, 20)
        giant_snapshot = extract_candidates(
            _decoded_stream([giant_warning_frame, base, base.copy()]),
            _config(
                min_component_area_ratio=0.10,
                giant_component_area_ratio=0.01,
                giant_component_policy="warn",
            ),
        )
        giant_codes = {warning.code for warning in giant_snapshot.result.warnings}
        self.assertIn("GIANT_COMPONENT", giant_codes)
        self.assertNotIn("GIANT_COMPONENT_REJECTED", giant_codes)
        self.assertEqual(
            giant_snapshot.result.frame_diagnostics[0].summary["rejected_small_count"],
            1,
        )

        border_warning_frame = base.copy()
        border_warning_frame[0:2, 4:6] = (230, 20, 20)
        border_snapshot = extract_candidates(
            _decoded_stream([border_warning_frame, base, base.copy()]),
            _config(
                min_component_area_ratio=0.10,
                border_touching_policy="warn",
            ),
        )
        border_codes = {warning.code for warning in border_snapshot.result.warnings}
        self.assertIn("BORDER_TOUCHING_COMPONENT", border_codes)
        self.assertNotIn("BORDER_TOUCHING_COMPONENT_REJECTED", border_codes)
        self.assertEqual(
            border_snapshot.result.frame_diagnostics[0].summary["rejected_small_count"],
            1,
        )

    def test_relative_filters_have_same_behavior_at_different_frame_sizes(self) -> None:
        outcomes: list[tuple[int, int, int]] = []
        for scale in (1, 2):
            size = 20 * scale
            base = _base_frame(height=size, width=size)
            accepted_frame = base.copy()
            accepted_frame[4 * scale : 8 * scale, 6 * scale : 10 * scale] = (230, 20, 20)
            rejected_frame = base.copy()
            rejected_frame[4 * scale : 6 * scale, 6 * scale : 10 * scale] = (230, 20, 20)
            oversized_frame = base.copy()
            oversized_frame[4 * scale : 10 * scale, 6 * scale : 12 * scale] = (230, 20, 20)
            config = _config(
                min_component_area_ratio=0.03,
                max_component_area_ratio=0.05,
                min_bbox_side_ratio=0.15,
            )

            accepted = extract_candidates(
                _decoded_stream([accepted_frame, base, base.copy()]),
                config,
            )
            rejected = extract_candidates(
                _decoded_stream([rejected_frame, base, base.copy()]),
                config,
            )
            oversized = extract_candidates(
                _decoded_stream([oversized_frame, base, base.copy()]),
                config,
            )
            outcomes.append(
                (
                    len(accepted.result.candidates),
                    rejected.result.frame_diagnostics[0].summary["rejected_small_count"],
                    oversized.result.frame_diagnostics[0].summary["rejected_large_count"],
                )
            )

        self.assertEqual(outcomes, [(1, 1, 1), (1, 1, 1)])

    def test_relative_filter_ratios_are_validated(self) -> None:
        invalid_configs = (
            {"min_component_area_ratio": -0.01},
            {"min_component_area_ratio": 1.01},
            {"max_component_area_ratio": 0.0},
            {"max_component_area_ratio": 1.01},
            {"min_component_area_ratio": 0.20, "max_component_area_ratio": 0.10},
            {"min_bbox_side_ratio": -0.01},
            {"min_bbox_side_ratio": 1.01},
        )

        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _config(**kwargs)

    def test_relative_filter_parameters_change_semantic_config_digest(self) -> None:
        base = _config()
        variants = (
            _config(min_component_area_ratio=0.01),
            _config(max_component_area_ratio=0.80),
            _config(min_bbox_side_ratio=0.02),
            _config(background_strategy="stream_model"),
            _config(background_border_ratio=0.12),
            _config(
                morphology=MorphologyCleanupConfig(
                    open_kernel_size=3,
                    close_kernel_size=5,
                    kernel_shape="ellipse",
                )
            ),
        )

        for variant in variants:
            with self.subTest(parameters=variant.semantic_parameters):
                self.assertNotEqual(base.config_digest, variant.config_digest)

        self.assertEqual(
            set(base.semantic_parameters).intersection(
                {"min_component_area", "max_component_area", "min_bbox_width", "min_bbox_height"}
            ),
            set(),
        )

    def test_development_defaults_are_explicit_semantic_parameters(self) -> None:
        config = CandidateExtractionConfig()
        previous_like = CandidateExtractionConfig(
            config_version="2.0",
            background_strategy="stream_model",
            hysteresis=HysteresisMaskConfig(
                weak_threshold=12.0,
                strong_threshold=20.0,
                connectivity=8,
            ),
            morphology=MorphologyCleanupConfig(),
            min_bbox_side_ratio=0.0,
        )

        self.assertEqual(config.config_version, "3.2")
        self.assertEqual(config.background_strategy, "frame_border_median")
        self.assertEqual(config.background_border_ratio, 0.08)
        self.assertEqual(config.hysteresis.weak_threshold, 10.0)
        self.assertEqual(config.hysteresis.strong_threshold, 16.0)
        self.assertEqual(config.morphology.open_kernel_size, 3)
        self.assertEqual(config.morphology.close_kernel_size, 7)
        self.assertEqual(config.morphology.kernel_shape, "ellipse")
        self.assertEqual(config.min_bbox_side_ratio, 0.045)
        self.assertNotEqual(config.config_digest, previous_like.config_digest)

    def test_background_strategy_and_border_ratio_are_validated(self) -> None:
        self.assertEqual(BackgroundStrategy.__args__, ("stream_model", "frame_border_median"))
        invalid_configs = (
            {"background_strategy": "unknown"},
            {"background_border_ratio": 0.0},
            {"background_border_ratio": -0.01},
            {"background_border_ratio": 0.51},
        )

        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _config(**kwargs)

    def test_frame_border_background_does_not_emit_disappearance_ghost(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[20:42, 30:56] = (230, 20, 20)

        snapshot = extract_candidates(
            _decoded_stream([object_frame, object_frame.copy(), base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        per_frame = {
            diagnostic.frame_id: diagnostic.summary["accepted_candidate_count"]
            for diagnostic in snapshot.result.frame_diagnostics
        }
        self.assertEqual(per_frame["frame_001"], 1)
        self.assertEqual(per_frame["frame_002"], 1)
        self.assertEqual(per_frame["frame_003"], 0)
        self.assertEqual(
            snapshot.result.frame_diagnostics[0].summary["background_strategy"],
            "frame_border_median",
        )
        self.assertEqual(
            snapshot.result.frame_diagnostics[0].summary["background_frame_count"],
            1,
        )

    def test_stream_model_strategy_keeps_legacy_stream_background_available(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[20:42, 30:56] = (230, 20, 20)

        snapshot = extract_candidates(
            _decoded_stream([object_frame, object_frame.copy(), base.copy()]),
            CandidateExtractionConfig(
                background_strategy="stream_model",
                bbox_padding=0,
            ),
        )

        per_frame = {
            diagnostic.frame_id: diagnostic.summary["accepted_candidate_count"]
            for diagnostic in snapshot.result.frame_diagnostics
        }
        self.assertEqual(per_frame["frame_001"], 0)
        self.assertEqual(per_frame["frame_002"], 0)
        self.assertEqual(per_frame["frame_003"], 1)
        self.assertEqual(
            snapshot.result.frame_diagnostics[0].summary["background_strategy"],
            "stream_model",
        )
        self.assertEqual(
            snapshot.result.frame_diagnostics[0].summary["background_frame_count"],
            3,
        )

    def test_development_defaults_filter_narrow_noise_without_losing_object(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[16:34, 20:42] = (230, 20, 20)
        object_frame[12:55, 70:74] = (20, 230, 20)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        self.assertEqual(frame_candidates[0].bbox, BBox(20, 16, 22, 18))
        diagnostic = snapshot.result.frame_diagnostics[1].summary
        self.assertEqual(diagnostic["rejected_small_count"], 1)

    def test_development_morphology_keeps_split_body_as_one_candidate(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[20:38, 20:34] = (230, 20, 20)
        object_frame[20:38, 36:50] = (230, 20, 20)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        candidate = frame_candidates[0]
        self.assertLessEqual(candidate.bbox.left, 20)
        self.assertGreaterEqual(candidate.bbox.right, 50)
        self.assertEqual(
            snapshot.result.frame_diagnostics[1].summary["accepted_candidate_count"],
            1,
        )

    def test_default_closing_keeps_nearby_thin_handle_in_object_bbox(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[20:50, 20:45] = (80, 185, 200)
        object_frame[30:42, 50:54] = (80, 185, 200)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        self.assertEqual(frame_candidates[0].bbox, BBox(20, 20, 34, 30))
        self.assertEqual(
            snapshot.result.frame_diagnostics[1].summary["rejected_small_count"],
            0,
        )

    def test_development_side_filter_scales_with_frame_size(self) -> None:
        outcomes: list[tuple[int, int]] = []
        for scale in (1, 2):
            height = 80 * scale
            width = 100 * scale
            base = _base_frame(height=height, width=width)
            object_frame = base.copy()
            object_frame[
                20 * scale : 32 * scale,
                20 * scale : 30 * scale,
            ] = (230, 20, 20)
            narrow_frame = base.copy()
            narrow_frame[
                20 * scale : 44 * scale,
                60 * scale : 64 * scale,
            ] = (20, 230, 20)

            accepted = extract_candidates(
                _decoded_stream([base, object_frame, base.copy()]),
                CandidateExtractionConfig(bbox_padding=0),
            )
            rejected = extract_candidates(
                _decoded_stream([base, narrow_frame, base.copy()]),
                CandidateExtractionConfig(bbox_padding=0),
            )
            outcomes.append(
                (
                    len(
                        [
                            item
                            for item in accepted.result.candidates
                            if item.frame_id == "frame_002"
                        ]
                    ),
                    rejected.result.frame_diagnostics[1].summary[
                        "rejected_small_count"
                    ],
                )
            )

        self.assertEqual(outcomes, [(1, 1), (1, 1)])

    def test_weak_contrast_object_on_dark_gradient_background_is_kept(self) -> None:
        height = 80
        width = 100
        base = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                base[y, x] = (28 + x // 12, 30 + y // 16, 45)
        object_frame = base.copy()
        object_frame[24:44, 32:57] = (78, 80, 92)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        self.assertEqual(frame_candidates[0].bbox, BBox(32, 24, 25, 20))

    def test_object_with_thin_detail_is_not_rejected_by_default_side_gate(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[30:52, 38:62] = (230, 80, 40)
        object_frame[20:32, 48:52] = (230, 80, 40)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        self.assertLessEqual(frame_candidates[0].bbox.top, 20)
        self.assertGreaterEqual(frame_candidates[0].bbox.bottom, 52)

    def test_object_with_internal_colors_and_contours_stays_single_candidate(self) -> None:
        base = _base_frame(height=80, width=100)
        object_frame = base.copy()
        object_frame[20:56, 24:72] = (40, 190, 120)
        object_frame[29:38, 34:44] = base[29:38, 34:44]
        object_frame[40:49, 52:62] = (220, 60, 90)

        snapshot = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            CandidateExtractionConfig(bbox_padding=0),
        )

        frame_candidates = [
            item
            for item in snapshot.result.candidates
            if item.frame_id == "frame_002"
        ]
        self.assertEqual(len(frame_candidates), 1)
        self.assertEqual(frame_candidates[0].bbox, BBox(24, 20, 48, 36))
        self.assertGreaterEqual(frame_candidates[0].geometry.values["contour_count"], 1)

    def test_dark_gradient_background_without_objects_stays_empty(self) -> None:
        height = 80
        width = 100
        base = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                base[y, x] = (28 + x // 16, 30 + y // 20, 45)

        snapshot = extract_candidates(
            _decoded_stream([base, base.copy(), base.copy()]),
            CandidateExtractionConfig(),
        )

        self.assertEqual(snapshot.result.candidates, ())

    def test_mask_digest_is_stable_and_changes_with_mask_content(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[4:8, 5:10] = (230, 20, 20)
        stream = _decoded_stream([base, object_frame, base.copy()])

        first = extract_candidates(stream, _config())
        second = extract_candidates(stream, _config())
        first_candidate = first.result.candidates[0]
        second_candidate = second.result.candidates[0]
        self.assertEqual(first_candidate.mask.mask_digest, second_candidate.mask.mask_digest)

        mask_record = first.masks[first_candidate.mask.mask_ref]
        changed = mask_record.mask.copy()
        changed[0, 0] = not changed[0, 0]
        self.assertNotEqual(candidate_mask_digest(mask_record.mask), candidate_mask_digest(changed))

    def test_geometry_metadata_contains_expected_fields(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[4:8, 5:10] = (230, 20, 20)
        candidate = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            _config(),
        ).result.candidates[0]

        values = candidate.geometry.values
        expected = {
            "area",
            "bbox_area",
            "foreground_fill_ratio",
            "mask_centroid_x",
            "mask_centroid_y",
            "aspect_ratio",
            "area_ratio_to_frame",
            "border_touching",
            "giant_component",
            "contour_count",
            "hole_count",
        }
        self.assertTrue(expected.issubset(values))
        self.assertEqual(values["area"], 20)
        self.assertEqual(values["bbox_area"], 20.0)
        self.assertEqual(values["foreground_fill_ratio"], 1.0)

    def test_extraction_preserves_decoded_stream_bytes(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[4:8, 5:10] = (230, 20, 20)
        stream = _decoded_stream([base, object_frame, base.copy()])
        before = tuple(frame.rgb_bytes for frame in stream.frames)

        _ = extract_candidates(stream, _config())

        self.assertEqual(before, tuple(frame.rgb_bytes for frame in stream.frames))

    def test_no_label_leakage_in_public_records(self) -> None:
        base = _base_frame()
        object_frame = base.copy()
        object_frame[4:8, 5:10] = (230, 20, 20)
        result = extract_candidates(
            _decoded_stream([base, object_frame, base.copy()]),
            _config(),
        ).result

        forbidden = {"expected_class", "expected_bbox", "visual_type"}
        self.assertTrue(forbidden.isdisjoint(result.frame_diagnostics[1].summary))
        self.assertTrue(forbidden.isdisjoint(result.candidates[0].geometry.values))

    def test_package_exports_are_available(self) -> None:
        import stream_analysis
        import stream_analysis.candidates as candidates

        for name in (
            "BackgroundStrategy",
            "CandidateExtractionConfig",
            "CandidateExtractionSnapshot",
            "CandidateMaskRecord",
            "candidate_mask_digest",
            "extract_candidates",
        ):
            with self.subTest(name=name):
                self.assertIn(name, stream_analysis.__all__)
                self.assertIn(name, candidates.__all__)
                self.assertTrue(hasattr(stream_analysis, name))
                self.assertTrue(hasattr(candidates, name))

    def test_read_only_probe_smoke_has_frame_diagnostics(self) -> None:
        input_config = StageConfig(
            stage_id="stream_input",
            schema_id="stream_analysis.stream_input_config.v1",
            config_version="1.0",
        )
        producer = ProducerProvenance(
            producer_stage="stream_input",
            producer_version="1.0",
            config_version=input_config.config_version,
            config_digest=semantic_config_digest(input_config),
        )
        extraction_config = CandidateExtractionConfig()
        for stream_root in (
            Path("data/streams/probe_01_stationery"),
            Path("data/streams/probe_02_fruits_vegetables_berries"),
            Path("data/streams/probe_03_tableware"),
            Path("data/streams/probe_04_technical_tools"),
        ):
            with self.subTest(stream_root=stream_root.as_posix()):
                decoded = load_decoded_stream(
                    ManifestLoadRequest(stream_root=stream_root, producer=producer)
                )
                snapshot = extract_candidates(decoded, extraction_config)
                self.assertIsInstance(snapshot.result, CandidateExtractionResult)
                self.assertTrue(
                    all(
                        isinstance(item, FrameCandidateDiagnostics)
                        for item in snapshot.result.frame_diagnostics
                    )
                )
                self.assertEqual(
                    len(snapshot.result.frame_diagnostics),
                    len(decoded.frames),
                )


if __name__ == "__main__":
    unittest.main()
