from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from stream_analysis import (
    ErrorRecord,
    ProducerProvenance,
    RecordEnvelope,
    Severity,
    StageContext,
    ValidityStatus,
    WarningRecord,
)


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer_stage="stream_input",
        producer_version="1.0",
        config_version="stream_input_v1",
        config_digest="sha256:abc123",
    )


class CommonContractsTest(unittest.TestCase):
    def test_valid_record_envelope(self) -> None:
        envelope = RecordEnvelope(
            record_id="frame_record_001",
            schema_version="frame-record-1.0",
            stream_id="probe_01",
            producer=_producer(),
            context=StageContext(frame_id="frame_001"),
            warning_ids=["warning_001"],
            provenance_refs=["manifest_001"],
        )

        self.assertIs(envelope.validity_status, ValidityStatus.VALID)
        self.assertEqual(envelope.warning_ids, ("warning_001",))
        self.assertEqual(envelope.provenance_refs, ("manifest_001",))

    def test_invalid_record_requires_error_lineage(self) -> None:
        envelope = RecordEnvelope(
            record_id="frame_record_001",
            schema_version="frame-record-1.0",
            stream_id="probe_01",
            producer=_producer(),
            validity_status=ValidityStatus.INVALID,
            error_ids=("error_001",),
        )
        self.assertIs(envelope.validity_status, ValidityStatus.INVALID)

        with self.assertRaisesRegex(ValueError, "at least one error"):
            RecordEnvelope(
                record_id="frame_record_002",
                schema_version="frame-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                validity_status=ValidityStatus.INVALID,
            )

    def test_valid_record_rejects_error_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid record"):
            RecordEnvelope(
                record_id="frame_record_001",
                schema_version="frame-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                error_ids=("error_001",),
            )

    def test_validity_does_not_mix_other_status_axes(self) -> None:
        envelope = RecordEnvelope(
            record_id="stream_record_001",
            schema_version="stream-record-1.0",
            stream_id="probe_01",
            producer=_producer(),
        )
        self.assertFalse(hasattr(envelope, "decision_status"))
        self.assertFalse(hasattr(envelope, "emission_status"))
        self.assertFalse(hasattr(envelope, "run_status"))
        self.assertEqual({item.value for item in ValidityStatus}, {"valid", "invalid"})

    def test_status_requires_typed_enum(self) -> None:
        with self.assertRaisesRegex(TypeError, "ValidityStatus"):
            RecordEnvelope(
                record_id="stream_record_001",
                schema_version="stream-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                validity_status="valid",  # type: ignore[arg-type]
            )

    def test_invalid_ids_and_versions_are_rejected(self) -> None:
        for record_id in ("", " bad", "bad/id", "bad id"):
            with self.subTest(record_id=record_id), self.assertRaises(ValueError):
                RecordEnvelope(
                    record_id=record_id,
                    schema_version="stream-record-1.0",
                    stream_id="probe_01",
                    producer=_producer(),
                )

        with self.assertRaises(ValueError):
            ProducerProvenance(
                producer_stage="stream_input",
                producer_version=" ",
                config_version="stream_input_v1",
                config_digest="digest",
            )

    def test_duplicate_and_self_provenance_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            RecordEnvelope(
                record_id="stream_record_001",
                schema_version="stream-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                warning_ids=("warning_001", "warning_001"),
            )
        with self.assertRaisesRegex(ValueError, "itself"):
            RecordEnvelope(
                record_id="stream_record_001",
                schema_version="stream-record-1.0",
                stream_id="probe_01",
                producer=_producer(),
                provenance_refs=("stream_record_001",),
            )

    def test_warning_lineage_and_metadata_are_immutable(self) -> None:
        warning = WarningRecord(
            record_id="warning_002",
            schema_version="diagnostic-record-1.0",
            stream_id="probe_01",
            code="LOW_CONTRAST",
            stage="stream_input",
            message="Input contrast is low.",
            producer=_producer(),
            context=StageContext(frame_id="frame_001"),
            metadata={"levels": [1, 2], "details": {"source": "frame"}},
            provenance_refs=("frame_record_001",),
            upstream_warning_ids=("warning_001",),
        )

        self.assertEqual(warning.id, "warning_002")
        self.assertEqual(warning.upstream_warning_ids, ("warning_001",))
        self.assertEqual(warning.metadata["levels"], (1, 2))
        with self.assertRaises(TypeError):
            warning.metadata["new"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            warning.metadata["details"]["source"] = "other"  # type: ignore[index]

    def test_error_lineage_is_preserved(self) -> None:
        error = ErrorRecord(
            record_id="error_002",
            schema_version="diagnostic-record-1.0",
            stream_id="probe_01",
            code="FRAME_INVALID",
            stage="stream_input",
            message="Frame record is invalid.",
            producer=_producer(),
            upstream_warning_ids=("warning_001",),
            upstream_error_ids=("error_001",),
        )
        self.assertEqual(error.upstream_warning_ids, ("warning_001",))
        self.assertEqual(error.upstream_error_ids, ("error_001",))

    def test_diagnostics_reject_self_reference_on_every_lineage_axis(self) -> None:
        diagnostic_types = (
            (WarningRecord, "warning_self", "SELF_WARNING"),
            (ErrorRecord, "error_self", "SELF_ERROR"),
        )
        lineage_fields = (
            "provenance_refs",
            "upstream_warning_ids",
            "upstream_error_ids",
        )

        for diagnostic_type, record_id, code in diagnostic_types:
            for lineage_field in lineage_fields:
                with (
                    self.subTest(
                        diagnostic_type=diagnostic_type.__name__,
                        lineage_field=lineage_field,
                    ),
                    self.assertRaisesRegex(ValueError, lineage_field),
                ):
                    diagnostic_type(
                        record_id=record_id,
                        schema_version="diagnostic-record-1.0",
                        stream_id="probe_01",
                        code=code,
                        stage="stream_input",
                        message="Self-reference must be rejected.",
                        producer=_producer(),
                        **{lineage_field: (record_id,)},
                    )

    def test_warning_and_error_severity_axes_are_checked(self) -> None:
        with self.assertRaisesRegex(TypeError, "Severity"):
            WarningRecord(
                record_id="warning_000",
                schema_version="diagnostic-record-1.0",
                stream_id="probe_01",
                code="UNTYPED_SEVERITY",
                stage="stream_input",
                message="Untyped severity.",
                producer=_producer(),
                severity="warning",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "WarningRecord severity"):
            WarningRecord(
                record_id="warning_001",
                schema_version="diagnostic-record-1.0",
                stream_id="probe_01",
                code="BAD_WARNING",
                stage="stream_input",
                message="Wrong severity.",
                producer=_producer(),
                severity=Severity.ERROR,
            )
        with self.assertRaisesRegex(ValueError, "ErrorRecord severity"):
            ErrorRecord(
                record_id="error_001",
                schema_version="diagnostic-record-1.0",
                stream_id="probe_01",
                code="BAD_ERROR",
                stage="stream_input",
                message="Wrong severity.",
                producer=_producer(),
                severity=Severity.WARNING,
            )

    def test_diagnostic_stage_must_match_producer(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            WarningRecord(
                record_id="warning_001",
                schema_version="diagnostic-record-1.0",
                stream_id="probe_01",
                code="STAGE_MISMATCH",
                stage="candidate_extraction",
                message="Stage mismatch.",
                producer=_producer(),
            )

    def test_contracts_are_frozen(self) -> None:
        envelope = RecordEnvelope(
            record_id="stream_record_001",
            schema_version="stream-record-1.0",
            stream_id="probe_01",
            producer=_producer(),
        )
        with self.assertRaises(FrozenInstanceError):
            envelope.stream_id = "probe_02"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
