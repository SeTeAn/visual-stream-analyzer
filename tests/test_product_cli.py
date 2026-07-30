from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from stream_analysis.product_cli import (
    EXIT_ANALYSIS_FAILED,
    EXIT_INPUT_ERROR,
    EXIT_OUTPUT_COLLISION,
    _producer,
    main,
)


class ProductCliTests(unittest.TestCase):
    def test_analyze_refuses_existing_output_before_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing"
            output.mkdir()
            code = main(
                [
                    "analyze",
                    str(root / "missing-stream"),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, EXIT_OUTPUT_COLLISION)

    def test_missing_stream_is_reported_as_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            code = main(
                [
                    "analyze",
                    str(root / "missing-stream"),
                    "--output",
                    str(root / "output"),
                ]
            )
        self.assertEqual(code, EXIT_INPUT_ERROR)

    def test_unexpected_runtime_error_is_sanitized(self) -> None:
        stderr = StringIO()
        with patch(
            "stream_analysis.product_cli._run_analyze",
            side_effect=RuntimeError("private C:/machine/path"),
        ), redirect_stderr(stderr):
            code = main(["analyze", "stream", "--output", "output"])
        self.assertEqual(code, EXIT_ANALYSIS_FAILED)
        self.assertNotIn("C:/machine/path", stderr.getvalue())

    def test_manifest_producer_uses_the_stream_input_stage(self) -> None:
        self.assertEqual(_producer("a" * 64).producer_stage, "stream_input")


if __name__ == "__main__":
    unittest.main()
