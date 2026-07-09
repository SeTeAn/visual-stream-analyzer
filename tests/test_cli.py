from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stream_analysis.cli import (
    EXIT_OK, EXIT_OUTPUT_COLLISION, EXIT_USAGE_OR_CONFIG, main,
)
import stream_analysis

try:
    from .integration._helpers import config_document, create_tiny_stream
except ImportError:
    from integration._helpers import config_document, create_tiny_stream


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.stream = create_tiny_stream(self.root / "stream")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config_document()), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def common(self):
        return [str(self.stream), "--config", str(self.config),
                "--representation-family", "handcrafted",
                "--variant", "handcrafted_bbox_v1"]

    def test_validate_and_analyze_commands(self) -> None:
        self.assertEqual(main(["validate", *self.common()]), EXIT_OK)
        arguments = ["analyze", *self.common(), "--output-root", str(self.root / "outputs"), "--run-id", "cli_run"]
        self.assertEqual(main(arguments), EXIT_OK)
        self.assertEqual(main(arguments), EXIT_OUTPUT_COLLISION)

    def test_cli_diagnostic_level_standard_writes_pair_scores(self) -> None:
        arguments = [
            "analyze",
            *self.common(),
            "--diagnostic-level",
            "standard",
            "--output-root",
            str(self.root / "outputs"),
            "--run-id",
            "cli_diag",
        ]
        self.assertEqual(main(arguments), EXIT_OK)
        self.assertTrue(
            (
                self.root
                / "outputs"
                / "runs"
                / "cli_diag"
                / "diagnostics"
                / "pair_scores.json"
            ).is_file()
        )

    def test_family_variant_mismatch_is_config_error(self) -> None:
        arguments = ["validate", str(self.stream), "--config", str(self.config),
                     "--representation-family", "handcrafted", "--variant", "wrong_variant"]
        self.assertEqual(main(arguments), EXIT_USAGE_OR_CONFIG)

    def test_public_orchestration_and_reporting_exports(self) -> None:
        for name in (
            "AnalyzePipelineConfig", "AnalyzeRequest", "AnalyzeOutcome",
            "AnalyzeRunError", "DinoV2AssetPaths", "run_analysis",
            "load_analyze_configuration", "RunArtifacts", "build_text_report",
            "render_primary_overlays", "RUN_MANIFEST_SCHEMA_ID",
            "CANDIDATE_MANIFEST_SCHEMA_ID", "PAIR_SCORES_SCHEMA_ID",
            "RUNTIME_STATUS_SCHEMA_ID", "build_pair_scores_payload",
        ):
            self.assertIn(name, stream_analysis.__all__)
            self.assertTrue(hasattr(stream_analysis, name))


if __name__ == "__main__":
    unittest.main()
