from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stream_analysis import ErrorRecord, OutputCollisionError, RunStatus, StageContext
import stream_analysis.orchestration as orchestration_module
from stream_analysis.orchestration import AnalyzeRequest, AnalyzeRunError, run_analysis

try:
    from ._helpers import create_tiny_stream, environment, pipeline_config, source
except ImportError:
    from _helpers import create_tiny_stream, environment, pipeline_config, source


class OrchestrationTest(unittest.TestCase):
    def request(self, stream: Path, output: Path, run_id: str) -> AnalyzeRequest:
        return AnalyzeRequest(
            stream_directory=stream, output_root=output, run_id=run_id,
            config=pipeline_config(), source_provenance=source(),
            environment_provenance=environment(),
        )

    def diagnostic_request(self, stream: Path, output: Path, run_id: str) -> AnalyzeRequest:
        return AnalyzeRequest(
            stream_directory=stream, output_root=output, run_id=run_id,
            config=pipeline_config(diagnostic_level="standard"),
            source_provenance=source(),
            environment_provenance=environment(),
        )

    def test_primary_run_writes_compact_non_overwriting_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            outcome = run_analysis(self.request(stream, root / "outputs", "run_001"))
            self.assertIn(outcome.result.run_status, {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS})
            names = {path.relative_to(outcome.artifacts.run_directory).as_posix() for path in outcome.artifacts.files}
            self.assertTrue({"run_manifest.json", "candidate_manifest.json", "stream_analysis.json", "report.txt", "runtime_status.json"}.issubset(names))
            self.assertEqual(len([name for name in names if name.startswith("overlays/")]), 2)
            encoded = (outcome.artifacts.run_directory / "stream_analysis.json").read_text(encoding="utf-8")
            for forbidden in ("embedding\"", "annotation", "gt_mapping", "evaluation_metrics"):
                self.assertNotIn(forbidden, encoded.casefold())
            artifact_refs = {
                item.artifact_kind: item.reference for item in outcome.result.artifacts
            }
            self.assertEqual(
                artifact_refs["stream_analysis"], "stream_analysis.json"
            )
            manifest = json.loads(
                (outcome.artifacts.run_directory / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            ledger_ids = {
                item["record_id"]
                for kind in ("warnings", "errors")
                for item in manifest["diagnostics"][kind]
            }
            self.assertTrue(
                set(outcome.result.envelope.warning_ids).issubset(ledger_ids)
            )
            self.assertTrue(
                set(outcome.result.envelope.error_ids).issubset(ledger_ids)
            )
            report = (outcome.artifacts.run_directory / "report.txt").read_text(encoding="utf-8")
            self.assertIn(f"Обработано кадров: {len(outcome.result.frame_ids)}", report)
            self.assertIn(f"Сформировано событий изменений: {len(outcome.result.change_events)}", report)
            self.assertIn("не использует разметку (ground truth)", report.casefold())
            self.assertNotIn("precision:", report.casefold())
            with self.assertRaises(OutputCollisionError):
                run_analysis(self.request(stream, root / "outputs", "run_001"))

    def test_standard_diagnostics_write_compact_pair_scores_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            outcome = run_analysis(self.diagnostic_request(stream, root / "outputs", "run_diag"))
            names = {path.relative_to(outcome.artifacts.run_directory).as_posix() for path in outcome.artifacts.files}
            self.assertIn("diagnostics/pair_scores.json", names)
            artifact_refs = {
                item.artifact_kind: item.reference for item in outcome.result.artifacts
            }
            self.assertEqual(
                artifact_refs["diagnostic_pair_scores"], "diagnostics/pair_scores.json"
            )
            manifest = json.loads(
                (outcome.artifacts.run_directory / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["provenance"]["output_schema_versions"]["diagnostic_pair_scores"],
                "stream_analysis.diagnostic_pair_scores.v1",
            )
            payload = json.loads(
                (outcome.artifacts.run_directory / "diagnostics" / "pair_scores.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["schema_id"], "stream_analysis.diagnostic_pair_scores.v1")
            self.assertEqual(payload["run_id"], "run_diag")
            self.assertEqual(payload["stream_id"], "tiny_stream")
            pair_keys = [
                (
                    item["frame_pair"]["from_frame_index"],
                    item["frame_pair"]["to_frame_index"],
                    item["from_candidate_id"],
                    item["to_candidate_id"],
                    item["pair_id"],
                )
                for item in payload["pairs"]
            ]
            self.assertEqual(pair_keys, sorted(pair_keys))
            encoded = json.dumps(payload).casefold()
            for forbidden in (
                "embedding_vector", "dino_embedding_values",
                "feature_vector", "handcrafted_working_values",
                "annotation", "gt_mapping", "evaluation_metrics",
            ):
                self.assertNotIn(forbidden, encoded)

    def test_pair_score_diagnostics_are_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            outcome = run_analysis(self.request(stream, root / "outputs", "run_compact"))
            self.assertFalse((outcome.artifacts.run_directory / "diagnostics" / "pair_scores.json").exists())
            artifact_kinds = {item.artifact_kind for item in outcome.result.artifacts}
            self.assertNotIn("diagnostic_pair_scores", artifact_kinds)

    def test_fatal_input_creates_failed_run_without_false_primary_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(AnalyzeRunError) as caught:
                run_analysis(self.request(root / "missing", root / "outputs", "failed_001"))
            self.assertEqual(caught.exception.code, "INPUT_FAILED")
            run_dir = root / "outputs" / "runs" / "failed_001"
            status = json.loads((run_dir / "runtime_status.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(status["run_status"], "failed")
            self.assertIn("requested_provenance", manifest)
            self.assertFalse((run_dir / "stream_analysis.json").exists())

    def test_output_error_after_staging_is_converted_to_clean_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            with patch.object(Image.Image, "save", side_effect=OSError("simulated output failure")):
                with self.assertRaises(AnalyzeRunError) as caught:
                    run_analysis(self.request(stream, root / "outputs", "failed_output"))
            self.assertEqual(caught.exception.code, "ANALYZE_FAILED")
            runs = root / "outputs" / "runs"
            self.assertTrue((runs / "failed_output" / "runtime_status.json").is_file())
            self.assertEqual(list(runs.glob(".failed_output.*")), [])

    def test_local_stage_error_produces_partial_not_failed_run(self) -> None:
        original = orchestration_module.build_change_events

        def with_error(*args, **kwargs):
            batch = original(*args, **kwargs)
            config = kwargs["config"]
            error = ErrorRecord(
                record_id="event_error:synthetic_partial",
                schema_version="event-error-1.0",
                stream_id=kwargs["stream_id"], code="SYNTHETIC_PARTIAL",
                stage="events", message="Synthetic local event error.",
                producer=config.producer, context=StageContext(),
            )
            return replace(batch, errors=(*batch.errors, error))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            with patch("stream_analysis.orchestration.build_change_events", side_effect=with_error):
                outcome = run_analysis(self.request(stream, root / "outputs", "partial_001"))
            self.assertIs(outcome.result.run_status, RunStatus.PARTIAL)
            self.assertEqual(outcome.result.status_summary["error_count"], 1)
            self.assertTrue((outcome.artifacts.run_directory / "stream_analysis.json").is_file())
            manifest = json.loads(
                (outcome.artifacts.run_directory / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["diagnostics"]["errors"][0]["record_id"],
                "event_error:synthetic_partial",
            )

    def test_run_id_must_be_a_portable_unambiguous_path_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = create_tiny_stream(root / "stream")
            for run_id in ("valid:logical", "name.", "CON", "nul.txt", "LPT1"):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    self.request(stream, root / "outputs", run_id)
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
