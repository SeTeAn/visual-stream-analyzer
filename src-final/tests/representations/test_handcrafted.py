import hashlib
import unittest
from pathlib import Path

import numpy as np

from stream_analysis import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    DataProvenance,
    DecodedFrame,
    DecodedStream,
    Fingerprint,
    FrameCandidateDiagnostics,
    FrameRecord,
    GeometryFeatureMetadata,
    ImageSize,
    ImageStream,
    ManifestFrameEntry,
    ManifestLoadResult,
    MaskReference,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
    ValidityStatus,
)
from stream_analysis.candidates import (
    CandidateExtractionSnapshot,
    CandidateMaskRecord,
    candidate_mask_digest,
)
from stream_analysis.representations import (
    BBOX_FEATURE_NAMES,
    HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID,
    HANDCRAFTED_BBOX_VARIANT,
    HANDCRAFTED_MASK_FEATURE_SCHEMA_ID,
    HANDCRAFTED_MASK_VARIANT,
    MASK_FEATURE_NAMES,
    HandcraftedRepresentationConfig,
    build_handcrafted_representations,
    srgb_uint8_to_normalized_cielab,
)


def _fingerprint(data: bytes) -> Fingerprint:
    return Fingerprint(algorithm="sha256", value=hashlib.sha256(data).hexdigest())


def _producer(stage: str, digest: str) -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage=stage,
        producer_version="1.0",
        config_version="1.0",
        config_digest=digest,
    )


def make_fixture(
    mask: np.ndarray,
    *,
    foreground_color: tuple[int, int, int] = (220, 40, 30),
    background_color: tuple[int, int, int] = (20, 45, 70),
    bbox_x: int = 4,
    bbox_y: int = 3,
    frame_width: int | None = None,
    frame_height: int | None = None,
    stream_id: str = "synthetic_stream",
    snapshot_stream_id: str | None = None,
    candidate_id: str = "cand_frame_001_001",
    custom_crop: np.ndarray | None = None,
    candidate_warning_ids: tuple[str, ...] = (),
    mask_warning_ids: tuple[str, ...] = (),
    mask_error_ids: tuple[str, ...] = (),
    mask_validity_status: ValidityStatus = ValidityStatus.VALID,
) -> tuple[DecodedStream, CandidateExtractionSnapshot]:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    frame_width = frame_width or (bbox_x + width + 5)
    frame_height = frame_height or (bbox_y + height + 5)
    frame = np.empty((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:, :] = background_color
    crop = np.empty((height, width, 3), dtype=np.uint8)
    crop[:, :] = background_color
    crop[mask] = foreground_color
    if custom_crop is not None:
        custom = np.asarray(custom_crop, dtype=np.uint8)
        if custom.shape != crop.shape:
            raise ValueError("custom_crop shape must match mask")
        crop = custom.copy()
    frame[bbox_y:bbox_y + height, bbox_x:bbox_x + width] = crop

    input_producer = _producer("stream_input", "sha256:synthetic-input")
    frame_id = "frame_001"
    image_size = ImageSize(width=frame_width, height=frame_height)
    frame_record = FrameRecord(
        envelope=RecordEnvelope(
            record_id=frame_id,
            schema_version="frame-record-1.0",
            stream_id=stream_id,
            producer=input_producer,
            context=StageContext(frame_id=frame_id),
        ),
        frame_id=frame_id,
        index=0,
        image_path="frames/frame_001.png",
        image_size=image_size,
    )
    rgb_bytes = frame.tobytes()
    raw_digest = _fingerprint(rgb_bytes)
    decoded_frame = DecodedFrame(
        record=frame_record,
        rgb_bytes=rgb_bytes,
        raw_digest=raw_digest,
        image_format="PNG",
        resolved_path=(Path.cwd() / "frames" / "frame_001.png").resolve(),
    )
    manifest_digest = _fingerprint(b"synthetic-manifest")
    manifest = ManifestLoadResult(
        stream_root=Path.cwd().resolve(),
        manifest_path=(Path.cwd() / "synthetic-manifest.json").resolve(),
        manifest_digest=manifest_digest,
        producer=input_producer,
        stream_id=stream_id,
        ordering="manifest",
        frames=(
            ManifestFrameEntry(
                frame_id=frame_id,
                index=0,
                manifest_position=0,
                image_path="frames/frame_001.png",
                resolved_path=decoded_frame.resolved_path,
            ),
        ),
    )
    decoded = DecodedStream(
        manifest=manifest,
        stream=ImageStream(
            envelope=RecordEnvelope(
                record_id=stream_id,
                schema_version="image-stream-1.0",
                stream_id=stream_id,
                producer=input_producer,
                context=StageContext(),
            ),
            frames=(frame_record,),
        ),
        frames=(decoded_frame,),
        data_provenance=DataProvenance(
            manifest_digest=manifest_digest,
            frame_content_digests={frame_id: raw_digest},
            candidate_snapshot_digest=None,
        ),
    )

    snapshot_stream_id = snapshot_stream_id or stream_id
    candidate_producer = _producer("candidate_extraction", "sha256:synthetic-candidates")
    bbox = BBox(bbox_x, bbox_y, width, height)
    mask_digest = candidate_mask_digest(mask)
    mask_ref = f"mask:{candidate_id}"
    mask_reference = MaskReference(
        mask_ref=mask_ref,
        mask_digest=mask_digest,
        producer_version="1.0",
        coordinate_bbox=bbox,
        validity_status=mask_validity_status,
        warning_ids=mask_warning_ids,
        error_ids=mask_error_ids,
    )
    candidate = CandidateRecord(
        envelope=RecordEnvelope(
            record_id=candidate_id,
            schema_version="candidate-record-1.0",
            stream_id=snapshot_stream_id,
            producer=candidate_producer,
            context=StageContext(frame_id=frame_id, candidate_id=candidate_id),
            warning_ids=candidate_warning_ids,
        ),
        candidate_id=candidate_id,
        frame_id=frame_id,
        frame_index=0,
        frame_size=image_size,
        bbox=bbox,
        center=bbox.center,
        geometry=GeometryFeatureMetadata(
            feature_schema_id="candidate_geometry_v1",
            producer_version="1.0",
            config_digest=candidate_producer.config_digest,
            values={
                "area": int(np.count_nonzero(mask)),
                "bbox_area": bbox.area,
                "foreground_fill_ratio": np.count_nonzero(mask) / bbox.area,
                "area_ratio_to_frame": np.count_nonzero(mask) / image_size.area,
            },
        ),
        candidate_source="synthetic_candidate",
        mask=mask_reference,
        candidate_confidence=0.8,
        quality_flags=(),
    )
    diagnostic = FrameCandidateDiagnostics(
        envelope=RecordEnvelope(
            record_id="candidate_diag_frame_001",
            schema_version="candidate-frame-diagnostics-1.0",
            stream_id=snapshot_stream_id,
            producer=candidate_producer,
            context=StageContext(frame_id=frame_id),
        ),
        frame_id=frame_id,
        frame_index=0,
        image_size=image_size,
    )
    result = CandidateExtractionResult(
        envelope=RecordEnvelope(
            record_id=f"candidate_result_{snapshot_stream_id}",
            schema_version="candidate-extraction-result-1.0",
            stream_id=snapshot_stream_id,
            producer=candidate_producer,
            context=StageContext(),
        ),
        extractor_source="synthetic_candidate",
        candidates=(candidate,),
        frame_diagnostics=(diagnostic,),
    )
    snapshot = CandidateExtractionSnapshot(
        result=result,
        masks={
            mask_ref: CandidateMaskRecord(
                mask_ref=mask_ref,
                mask_digest=mask_digest,
                candidate_id=candidate_id,
                frame_id=frame_id,
                coordinate_bbox=bbox,
                mask=mask,
            )
        },
    )
    return decoded, snapshot


def build_record(
    mask: np.ndarray,
    *,
    variant: str = HANDCRAFTED_MASK_VARIANT,
    config: HandcraftedRepresentationConfig | None = None,
    **fixture_kwargs: object,
):
    effective = config or HandcraftedRepresentationConfig(variant=variant)
    decoded, snapshot = make_fixture(mask, **fixture_kwargs)
    batch = build_handcrafted_representations(decoded, snapshot, effective)
    return batch.records[0], batch, effective


def groups(record) -> dict[str, object]:
    return {group.group_name: group for group in record.payload.feature_groups}


class HandcraftedRepresentationTest(unittest.TestCase):
    def test_exact_feature_names_and_groups_for_both_variants(self) -> None:
        mask = np.zeros((24, 28), dtype=bool)
        mask[4:20, 5:23] = True
        for variant, expected_schema, expected_names in (
            (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_MASK_FEATURE_SCHEMA_ID, MASK_FEATURE_NAMES),
            (HANDCRAFTED_BBOX_VARIANT, HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID, BBOX_FEATURE_NAMES),
        ):
            with self.subTest(variant=variant):
                record, _batch, _config = build_record(mask, variant=variant)
                self.assertEqual(record.payload.feature_schema_id, expected_schema)
                self.assertEqual(
                    tuple(record.preprocessing_metadata.details["feature_order"]),
                    expected_names,
                )
                self.assertEqual(
                    tuple(group.group_name for group in record.payload.feature_groups),
                    ("color", "shape", "size", "structure"),
                )
                actual_names = {
                    name
                    for group in record.payload.feature_groups
                    for name in group.values
                }
                self.assertEqual(actual_names, set(expected_names))

    def test_reference_cielab_black_white_and_red(self) -> None:
        rgb = np.array(((0, 0, 0), (255, 255, 255), (255, 0, 0)), dtype=np.uint8)
        before = rgb.copy()
        lab = srgb_uint8_to_normalized_cielab(rgb)
        self.assertEqual(lab.dtype, np.float64)
        np.testing.assert_array_equal(rgb, before)
        np.testing.assert_allclose(lab[0], (0.0, 128 / 255, 128 / 255), atol=1e-8)
        np.testing.assert_allclose(lab[1], (1.0, 128 / 255, 128 / 255), atol=2e-6)
        expected_red = (53.2408 / 100, (80.0925 + 128) / 255, (67.2032 + 128) / 255)
        np.testing.assert_allclose(lab[2], expected_red, atol=5e-5)

    def test_mask_topology_and_radial_profiles_are_bounded(self) -> None:
        yy, xx = np.ogrid[:40, :40]
        radius = np.sqrt((xx - 19.5) ** 2 + (yy - 19.5) ** 2)
        ring = (radius <= 14) & (radius >= 6)
        record, _batch, _config = build_record(ring)
        feature_groups = groups(record)
        shape = feature_groups["shape"].values
        structure = feature_groups["structure"]
        self.assertGreater(shape["hc_hole_count_norm"], 0.0)
        self.assertGreater(shape["hc_hole_area_ratio"], 0.0)
        self.assertAlmostEqual(
            sum(shape[f"hc_radial_mass_{index}"] for index in range(1, 5)),
            1.0,
            places=12,
        )
        if structure.valid:
            self.assertAlmostEqual(
                sum(structure.values[f"hc_radial_edge_{index}"] for index in range(1, 5)),
                1.0,
                places=12,
            )
        for group in feature_groups.values():
            for value in group.values.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_filled_shape_has_no_holes(self) -> None:
        mask = np.zeros((30, 34), dtype=bool)
        mask[5:25, 7:27] = True
        record, _batch, _config = build_record(mask)
        shape = groups(record)["shape"].values
        self.assertEqual(shape["hc_hole_count_norm"], 0.0)
        self.assertEqual(shape["hc_hole_area_ratio"], 0.0)

    def test_shape_features_are_stable_for_right_angle_rotation_and_scale(self) -> None:
        base = np.zeros((28, 36), dtype=bool)
        base[7:21, 8:28] = True
        rotated = np.rot90(base)
        scaled = np.kron(base, np.ones((2, 2), dtype=bool))
        records = [build_record(mask)[0] for mask in (base, rotated, scaled)]
        shapes = [groups(record)["shape"].values for record in records]
        for name in (
            "hc_elongation",
            "hc_circularity",
            "hc_solidity",
            "hc_hole_count_norm",
            "hc_hole_area_ratio",
        ):
            self.assertAlmostEqual(shapes[0][name], shapes[1][name], delta=0.03)
            self.assertAlmostEqual(shapes[0][name], shapes[2][name], delta=0.08)
        for other in shapes[1:]:
            radial_difference = 0.5 * sum(
                abs(shapes[0][f"hc_radial_mass_{index}"] - other[f"hc_radial_mass_{index}"])
                for index in range(1, 5)
            )
            self.assertLess(radial_difference, 0.08)

    def test_empty_or_fragmented_mask_is_invalid_without_bbox_fallback(self) -> None:
        for mask, code in (
            (np.zeros((16, 16), dtype=bool), "MASK_TOO_SMALL"),
            (
                np.pad(np.ones((3, 3), dtype=bool), ((1, 12), (1, 12)))
                | np.pad(np.ones((3, 3), dtype=bool), ((12, 1), (12, 1))),
                "MASK_COMPONENT_COUNT_INVALID",
            ),
        ):
            with self.subTest(code=code):
                record, batch, _config = build_record(mask)
                self.assertIs(record.envelope.validity_status, ValidityStatus.INVALID)
                self.assertIsNone(record.payload)
                self.assertEqual(batch.errors[0].code, code)
                self.assertEqual(record.input_variant, HANDCRAFTED_MASK_VARIANT)
                self.assertNotEqual(record.input_variant, HANDCRAFTED_BBOX_VARIANT)

    def test_new_warning_inherits_candidate_and_mask_warning_lineage(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        uniform_crop = np.zeros((16, 16, 3), dtype=np.uint8)
        _record, batch, _config = build_record(
            mask,
            custom_crop=uniform_crop,
            candidate_warning_ids=("warn_candidate_quality",),
            mask_warning_ids=("warn_mask_quality",),
        )

        self.assertEqual(len(batch.warnings), 1)
        warning = batch.warnings[0]
        self.assertEqual(warning.code, "EMPTY_EDGE_STRUCTURE")
        self.assertEqual(
            warning.upstream_warning_ids,
            ("warn_candidate_quality", "warn_mask_quality"),
        )
        self.assertEqual(warning.upstream_error_ids, ())

    def test_new_error_inherits_candidate_and_mask_diagnostic_lineage(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        _record, batch, _config = build_record(
            mask,
            candidate_warning_ids=("warn_candidate_quality",),
            mask_warning_ids=("warn_mask_quality",),
            mask_error_ids=("err_mask_invalid",),
            mask_validity_status=ValidityStatus.INVALID,
        )

        self.assertEqual(len(batch.errors), 1)
        error = batch.errors[0]
        self.assertEqual(error.code, "MASK_REQUIRED")
        self.assertEqual(
            error.upstream_warning_ids,
            ("warn_candidate_quality", "warn_mask_quality"),
        )
        self.assertEqual(error.upstream_error_ids, ("err_mask_invalid",))

    def test_stream_snapshot_contract_is_checked(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        decoded, snapshot = make_fixture(mask, snapshot_stream_id="other_stream")
        with self.assertRaisesRegex(ValueError, "same stream_id"):
            build_handcrafted_representations(decoded, snapshot)

    def test_semantic_digest_is_deterministic_and_variant_specific(self) -> None:
        first = HandcraftedRepresentationConfig()
        second = HandcraftedRepresentationConfig()
        bbox = HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT)
        self.assertEqual(first.config_digest, second.config_digest)
        self.assertNotEqual(first.config_digest, bbox.config_digest)
        bbox_changed_mask_only = HandcraftedRepresentationConfig(
            variant=HANDCRAFTED_BBOX_VARIANT,
            mask_dilation_pixels=4,
            max_holes=9,
        )
        self.assertEqual(bbox.config_digest, bbox_changed_mask_only.config_digest)

    def test_heldout_calibration_defaults_are_versioned_and_explicit(self) -> None:
        mask = HandcraftedRepresentationConfig(variant=HANDCRAFTED_MASK_VARIANT)
        bbox = HandcraftedRepresentationConfig(variant=HANDCRAFTED_BBOX_VARIANT)

        self.assertEqual(mask.config_version, "1.1")
        self.assertEqual(bbox.config_version, "1.1")
        self.assertEqual(mask.group_weights["shape"], 0.40)
        self.assertEqual(mask.group_weights["structure"], 0.40)
        self.assertEqual(mask.group_weights["color"], 0.15)
        self.assertEqual(mask.group_weights["size"], 0.05)
        self.assertEqual(bbox.group_weights["shape"], 0.25)
        self.assertEqual(bbox.group_weights["structure"], 0.55)
        self.assertEqual(bbox.group_weights["color"], 0.15)
        self.assertEqual(bbox.group_weights["size"], 0.05)

    def test_build_is_deterministic_and_does_not_mutate_inputs(self) -> None:
        mask = np.zeros((20, 22), dtype=bool)
        mask[3:17, 4:18] = True
        decoded, snapshot = make_fixture(mask)
        before_bytes = decoded.frames[0].rgb_bytes
        before_mask = snapshot.mask_for_candidate("cand_frame_001_001").mask.copy()
        first = build_handcrafted_representations(decoded, snapshot)
        second = build_handcrafted_representations(decoded, snapshot)
        self.assertEqual(first, second)
        self.assertEqual(decoded.frames[0].rgb_bytes, before_bytes)
        np.testing.assert_array_equal(
            snapshot.mask_for_candidate("cand_frame_001_001").mask,
            before_mask,
        )
        with self.assertRaises(ValueError):
            snapshot.mask_for_candidate("cand_frame_001_001").mask[0, 0] = True

    def test_working_values_are_not_rounded_to_six_decimals(self) -> None:
        mask = np.ones((9, 11), dtype=bool)
        record, _batch, _config = build_record(mask, foreground_color=(37, 129, 211))
        color = groups(record)["color"].values["hc_lab_l"]
        self.assertNotEqual(color, round(color, 6))


if __name__ == "__main__":
    unittest.main()
