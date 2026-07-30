from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stream_analysis.contracts import BBox
from stream_analysis.runtime.config import load_runtime_profile
from stream_analysis.runtime.engine import (
    RuntimeMaskRecord,
    RuntimePipelineError,
    execute_pipeline,
    resolve_aggregate_mask_records,
)
from stream_analysis.runtime.sam2 import GroundedMaskResult, MaskQuality
from stream_analysis.runtime.workspace import RuntimeWorkspace, StoredMask


def _mask_record(candidate_id: str, mask: np.ndarray) -> RuntimeMaskRecord:
    rows, columns = np.nonzero(mask)
    bbox = BBox(
        float(columns.min()),
        float(rows.min()),
        float(columns.max() - columns.min() + 1),
        float(rows.max() - rows.min() + 1),
    )
    quality = MaskQuality(
        selected_mask_index=0,
        predicted_iou=1.0,
        object_score_logit=None,
        raw_foreground_pixels=int(mask.sum()),
        cleaned_foreground_pixels=int(mask.sum()),
        removed_component_count=0,
        removed_foreground_pixels=0,
    )
    result = GroundedMaskResult(
        source_index=0,
        source_bbox=bbox,
        status="valid",
        raw_mask=mask,
        cleaned_mask=mask,
        mask_bbox=bbox,
        quality=quality,
        fallback_reason=None,
    )
    return RuntimeMaskRecord(
        candidate_id=candidate_id,
        frame_id="frame_0001",
        frame_index=1,
        score=1.0,
        phrase="object",
        source_bbox=bbox,
        result=result,
        artifacts=StoredMask(raw_path=Path("raw.png"), cleaned_path=Path("cleaned.png")),
    )


class RuntimeEngineTests(unittest.TestCase):
    def test_aggregate_resolution_removes_one_global_mask(self) -> None:
        global_mask = np.ones((20, 20), dtype=np.bool_)
        first = np.zeros((20, 20), dtype=np.bool_)
        first[2:6, 2:6] = True
        second = np.zeros((20, 20), dtype=np.bool_)
        second[12:16, 12:16] = True
        third = np.zeros((20, 20), dtype=np.bool_)
        third[2:6, 12:16] = True
        kept, report = resolve_aggregate_mask_records(
            (
                _mask_record("global", global_mask),
                _mask_record("first", first),
                _mask_record("second", second),
                _mask_record("third", third),
            )
        )
        self.assertEqual(set(kept), {"first", "second", "third"})
        self.assertEqual(report["removed_candidate_count"], 1)
        self.assertNotIn("ground_truth", " ".join(report))

    def test_execute_pipeline_orders_the_three_model_stages(self) -> None:
        decoded = SimpleNamespace(stream=SimpleNamespace(stream_id="stream"))
        detector = SimpleNamespace(decoded=decoded)
        masks = SimpleNamespace(decoded=decoded)
        analysis = SimpleNamespace(decoded=decoded)
        order: list[str] = []

        def detector_stage(*args, **kwargs):
            order.append("grounding_dino")
            return (detector,), 1.0

        def mask_stage(*args, **kwargs):
            order.append("sam2")
            return (masks,), 2.0

        def analysis_stage(*args, **kwargs):
            order.append("dinov2")
            return (analysis,), 3.0

        with tempfile.TemporaryDirectory() as directory, patch(
            "stream_analysis.runtime.engine.run_detector_streams", detector_stage
        ), patch(
            "stream_analysis.runtime.engine.run_mask_refinement_streams", mask_stage
        ), patch(
            "stream_analysis.runtime.engine.run_analysis_streams", analysis_stage
        ):
            outcome = execute_pipeline(
                (decoded,),
                profile=load_runtime_profile(),
                assets=SimpleNamespace(),
                workspace=RuntimeWorkspace(Path(directory)),
            )
        self.assertEqual(order, ["grounding_dino", "sam2", "dinov2"])
        self.assertEqual(len(outcome.streams), 1)

    def test_execute_pipeline_wraps_unexpected_model_errors(self) -> None:
        decoded = SimpleNamespace(stream=SimpleNamespace(stream_id="stream"))
        with tempfile.TemporaryDirectory() as directory, patch(
            "stream_analysis.runtime.engine.run_detector_streams",
            side_effect=RuntimeError("private C:/machine/path"),
        ):
            with self.assertRaisesRegex(
                RuntimePipelineError, "Grounding DINO stage failed"
            ) as caught:
                execute_pipeline(
                    (decoded,),
                    profile=load_runtime_profile(),
                    assets=SimpleNamespace(),
                    workspace=RuntimeWorkspace(Path(directory)),
                )
        self.assertNotIn("C:/machine/path", str(caught.exception))

    def test_workspace_round_trips_binary_masks_and_embeddings(self) -> None:
        mask = np.zeros((5, 7), dtype=np.bool_)
        mask[1:4, 2:6] = True
        with tempfile.TemporaryDirectory() as directory:
            workspace = RuntimeWorkspace(Path(directory))
            stored = workspace.store_masks(
                stream_id="stream",
                frame_id="frame_0001",
                candidate_id="candidate",
                source_index=0,
                raw_mask=mask,
                cleaned_mask=mask,
            )
            np.testing.assert_array_equal(workspace.load_mask(stored.cleaned_path), mask)
            embeddings = workspace.store_embeddings(
                stream_id="stream", embeddings=np.ones((2, 3), dtype=np.float32)
            )
            self.assertEqual(embeddings.shape, (2, 3))
            self.assertTrue(embeddings.path.is_file())

    def test_workspace_removes_raw_mask_when_cleaned_mask_write_fails(self) -> None:
        mask = np.ones((3, 3), dtype=np.bool_)
        with tempfile.TemporaryDirectory() as directory:
            workspace = RuntimeWorkspace(Path(directory))
            original = workspace._write_mask
            calls = 0

            def fail_second(path, value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("cleaned mask write failed")
                original(path, value)

            with patch.object(workspace, "_write_mask", side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    workspace.store_masks(
                        stream_id="stream",
                        frame_id="frame_0001",
                        candidate_id="candidate",
                        source_index=0,
                        raw_mask=mask,
                        cleaned_mask=mask,
                    )
            self.assertFalse(any(Path(directory).rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
