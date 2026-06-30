from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stream_analysis.evaluation import write_evaluation_artifacts


class EvaluationArtifactsTest(unittest.TestCase):
    def test_rejects_non_portable_evaluation_ids_without_filesystem_side_effects(self) -> None:
        invalid_ids = (
            "nested/path",
            r"nested\path",
            "bad:name",
            "CON",
            "nul.txt",
            "LPT1",
            "ends.",
            "x" * 256,
        )
        for evaluation_id in invalid_ids:
            with self.subTest(evaluation_id=evaluation_id):
                with tempfile.TemporaryDirectory() as temporary:
                    output_root = Path(temporary) / "outputs"

                    with self.assertRaises(ValueError):
                        write_evaluation_artifacts(
                            output_root=output_root,
                            evaluation_id=evaluation_id,
                            report={},
                            error_ledger=[],
                            summary_text="",
                            manifest={},
                        )

                    self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
