from __future__ import annotations

import json
import unittest
from pathlib import Path

from stream_analysis.runtime.config import (
    RuntimeConfigurationError,
    load_runtime_profile,
    require_portable_component,
    resolve_runtime_assets,
    runtime_profile_from_sections,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RuntimeProfileTests(unittest.TestCase):
    def test_public_profile_is_exact_neutral_copy_of_fixed_sections(self) -> None:
        public = load_runtime_profile()
        protocol = json.loads(
            (REPOSITORY_ROOT / "data/ocid/benchmark/ocid_pipeline_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        adapted = runtime_profile_from_sections(
            profile_id="visual_stream_analyzer_v1",
            models=protocol["models"],
            pipeline=protocol["pipeline"],
            paths_include_models_prefix=True,
        )
        self.assertEqual(public.sha256, adapted.sha256)
        self.assertEqual(dict(public.models), dict(adapted.models))
        self.assertEqual(dict(public.pipeline), dict(adapted.pipeline))

    def test_portable_component_rejects_windows_and_path_names(self) -> None:
        for value in ("CON", "frame/001", "frame\\001", "frame.", ".."):
            with self.subTest(value=value), self.assertRaises(RuntimeConfigurationError):
                require_portable_component(value, "component")
        require_portable_component("frame_0001", "component")

    def test_missing_runtime_paths_do_not_disclose_absolute_locations(self) -> None:
        secret = REPOSITORY_ROOT / "private-machine" / "models"
        with self.assertRaises(RuntimeConfigurationError) as caught:
            resolve_runtime_assets(load_runtime_profile(), secret)
        self.assertNotIn(str(secret), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
