from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import stream_analysis
from stream_analysis import (
    ANALYSIS_CONFIG_SCHEMA_ID,
    CANONICAL_SEMANTIC_ENCODING_ID,
    EVALUATION_CONFIG_SCHEMA_ID,
    AnalysisRunConfig,
    EvaluationConfig,
    EvaluationDataRole,
    EvaluationPolicy,
    StageConfig,
    canonical_evaluation_bytes,
    canonical_semantic_bytes,
    evaluation_config_digest,
    semantic_config_digest,
)


STAGE_IDS = (
    "stream_input",
    "candidate_extraction",
    "representation",
    "scorer",
    "matching",
    "grouping",
    "events",
    "reporting",
    "runtime_policy",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Include the pre-F02B non-package contract suite in root discovery."""

    contract_suite = unittest.TestLoader().discover(
        str(Path(__file__).parent / "contracts"),
        pattern=pattern or "test_*.py",
    )
    tests.addTests(contract_suite)
    return tests


def _stage(
    stage_id: str,
    *,
    semantic: dict | None = None,
    runtime: dict | None = None,
    diagnostic: dict | None = None,
) -> StageConfig:
    return StageConfig(
        stage_id=stage_id,
        schema_id=f"stream_analysis.{stage_id}_config.v1",
        config_version="1.0",
        semantic_parameters=semantic or {},
        runtime_parameters=runtime or {},
        diagnostic_parameters=diagnostic or {},
    )


def _analysis_config(
    *,
    representation: StageConfig | None = None,
    runtime_policy: StageConfig | None = None,
) -> AnalysisRunConfig:
    configs = {stage_id: _stage(stage_id) for stage_id in STAGE_IDS}
    if representation is not None:
        configs["representation"] = representation
    if runtime_policy is not None:
        configs["runtime_policy"] = runtime_policy
    return AnalysisRunConfig(config_version="1.0", **configs)


def _evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        config_version="1.0",
        metric_policy=EvaluationPolicy(
            policy_id="metric_policy",
            policy_version="1.0",
            parameters={"candidate_iou": "v1", "event_set": "v1"},
        ),
        invalid_status_policy=EvaluationPolicy(
            policy_id="strict_status",
            policy_version="1.0",
            parameters={"uncertain": "error", "withheld": "fn"},
        ),
        mapping_policy=EvaluationPolicy(
            policy_id="type_mapping",
            policy_version="1.0",
            parameters={"scope": "per_stream"},
        ),
        data_role=EvaluationDataRole.PROBE_DEVELOPMENT,
        selection_freeze_metadata={"selection_protocol": "loso_v1", "frozen": False},
        artifact_policy=EvaluationPolicy(
            policy_id="evaluation_artifacts",
            policy_version="1.0",
            parameters={"error_ledger": True},
        ),
    )


class ConfigurationTest(unittest.TestCase):
    def test_mapping_insertion_order_does_not_change_digest(self) -> None:
        first = _stage("representation", semantic={"beta": 2, "alpha": 1})
        second = _stage("representation", semantic={"alpha": 1, "beta": 2})
        self.assertEqual(canonical_semantic_bytes(first), canonical_semantic_bytes(second))
        self.assertEqual(semantic_config_digest(first), semantic_config_digest(second))

    def test_sequence_order_and_semantic_changes_change_digest(self) -> None:
        ordered = _stage("representation", semantic={"groups": ["color", "shape"]})
        reversed_order = _stage("representation", semantic={"groups": ["shape", "color"]})
        changed = _stage("representation", semantic={"groups": ["color", "texture"]})
        self.assertNotEqual(semantic_config_digest(ordered), semantic_config_digest(reversed_order))
        self.assertNotEqual(semantic_config_digest(ordered), semantic_config_digest(changed))

    def test_runtime_and_diagnostic_changes_do_not_change_semantic_digest(self) -> None:
        first = _stage(
            "representation",
            semantic={"variant": "bbox"},
            runtime={"device": "cpu", "batch_size": 1},
            diagnostic={"level": "minimal"},
        )
        second = _stage(
            "representation",
            semantic={"variant": "bbox"},
            runtime={"device": "cuda", "batch_size": 32},
            diagnostic={"level": "full"},
        )
        self.assertEqual(semantic_config_digest(first), semantic_config_digest(second))
        self.assertEqual(
            semantic_config_digest(_analysis_config(representation=first)),
            semantic_config_digest(_analysis_config(representation=second)),
        )

    def test_config_values_are_frozen_and_reject_unsupported_values(self) -> None:
        source = {"nested": {"values": [1, 2]}}
        config = _stage("representation", semantic=source)
        source["nested"]["values"].append(3)
        self.assertEqual(config.semantic_parameters["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            config.semantic_parameters["new"] = 1  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            config.config_version = "2.0"  # type: ignore[misc]

        invalid_values = (
            float("nan"),
            float("inf"),
            {"unordered"},
            b"bytes",
            object(),
        )
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(
                (TypeError, ValueError)
            ):
                _stage("representation", semantic={"value": value})
        with self.assertRaises(ValueError):
            StageConfig(
                stage_id="representation",
                schema_id="stream_analysis.representation_config.v1",
                config_version="1.0",
                semantic_parameters={1: "bad"},  # type: ignore[dict-item]
            )

    def test_float_bool_int_and_unicode_encoding_are_unambiguous(self) -> None:
        negative_zero = _stage("representation", semantic={"value": -0.0})
        positive_zero = _stage("representation", semantic={"value": 0.0})
        bool_value = _stage("representation", semantic={"value": True})
        int_value = _stage("representation", semantic={"value": 1})
        unicode_value = _stage("representation", semantic={"описание": "повторение"})

        self.assertEqual(
            canonical_semantic_bytes(negative_zero),
            canonical_semantic_bytes(positive_zero),
        )
        self.assertNotEqual(semantic_config_digest(bool_value), semantic_config_digest(int_value))
        encoded = canonical_semantic_bytes(unicode_value)
        self.assertIn("повторение".encode("utf-8"), encoded)
        self.assertFalse(encoded.endswith(b"\n"))
        self.assertIn(CANONICAL_SEMANTIC_ENCODING_ID.encode("utf-8"), encoded)

    def test_digest_is_stable_across_repeated_calls(self) -> None:
        config = _analysis_config(
            representation=_stage(
                "representation",
                semantic={"weight": 0.12345678901234566, "variant": "bbox"},
            )
        )
        digests = {semantic_config_digest(config) for _ in range(10)}
        self.assertEqual(len(digests), 1)
        self.assertRegex(digests.pop(), r"^sha256:[0-9a-f]{64}$")

    def test_analysis_config_rejects_evaluator_and_annotation_metadata(self) -> None:
        forbidden_keys = (
            "annotation_path",
            "annotation_digest",
            "annotationPath",
            "annotationDigest",
            "groundTruthLabels",
            "ground_truth_mapping",
            "gtBoxes",
            "gt_labels",
            "gtMapping",
            "evaluationResults",
            "evaluation_data_role",
            "evaluation_metrics",
            "evaluationMetrics",
            "mapping_policy",
            "evaluatorMappingPolicy",
            "final_data_role",
        )
        for key in forbidden_keys:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "evaluator-only"):
                _analysis_config(
                    representation=_stage("representation", semantic={key: "forbidden"})
                )
        names = {item.name for item in fields(AnalysisRunConfig)}
        self.assertTrue(
            names.isdisjoint(
                {"evaluation_config", "annotation_path", "annotation_digest", "metrics"}
            )
        )

    def test_analyze_leakage_check_is_recursive_on_all_stage_axes(self) -> None:
        forbidden_keys = (
            "groundTruthLabels",
            "ground_truth_mapping",
            "gtBoxes",
            "evaluationResults",
            "evaluation_data_role",
        )
        axes = ("semantic", "runtime", "diagnostic")
        for axis in axes:
            for key in forbidden_keys:
                stage_arguments = {
                    axis: {"outer": [{"inner": {key: "forbidden"}}]}
                }
                with (
                    self.subTest(axis=axis, key=key),
                    self.assertRaisesRegex(ValueError, "evaluator-only"),
                ):
                    _analysis_config(
                        representation=_stage("representation", **stage_arguments)
                    )

    def test_analysis_config_requires_all_named_stage_configs(self) -> None:
        config = _analysis_config()
        self.assertEqual(tuple(stage.stage_id for stage in config.stage_configs), STAGE_IDS)
        values = {item.name: getattr(config, item.name) for item in fields(AnalysisRunConfig)}
        values["scorer"] = _stage("matching")
        with self.assertRaisesRegex(ValueError, "scorer.stage_id"):
            AnalysisRunConfig(**values)

    def test_evaluation_config_is_separate_versioned_and_deterministic(self) -> None:
        config = _evaluation_config()
        self.assertIs(config.data_role, EvaluationDataRole.PROBE_DEVELOPMENT)
        self.assertNotIsInstance(config, AnalysisRunConfig)
        self.assertEqual(
            evaluation_config_digest(config),
            evaluation_config_digest(config),
        )
        self.assertEqual(
            canonical_evaluation_bytes(config),
            canonical_evaluation_bytes(config),
        )
        with self.assertRaises(TypeError):
            config.selection_freeze_metadata["frozen"] = True  # type: ignore[index]

    def test_evaluation_config_allows_gt_and_evaluation_metadata(self) -> None:
        config = _evaluation_config()
        values = {item.name: getattr(config, item.name) for item in fields(EvaluationConfig)}
        values["selection_freeze_metadata"] = {
            "nested": [
                {
                    "groundTruthLabels": "v1",
                    "ground_truth_mapping": "per_stream",
                    "gtBoxes": True,
                    "evaluationResults": "frozen",
                    "evaluation_data_role": "probe_development",
                }
            ]
        }
        accepted = EvaluationConfig(**values)
        nested = accepted.selection_freeze_metadata["nested"][0]
        self.assertEqual(nested["evaluation_data_role"], "probe_development")

    def test_schema_identifiers_and_package_exports(self) -> None:
        self.assertEqual(ANALYSIS_CONFIG_SCHEMA_ID, "stream_analysis.analysis_config.v1")
        self.assertEqual(EVALUATION_CONFIG_SCHEMA_ID, "stream_analysis.evaluation_config.v1")
        for name in (
            "StageConfig",
            "AnalysisRunConfig",
            "EvaluationConfig",
            "semantic_config_digest",
        ):
            self.assertIn(name, stream_analysis.__all__)
            self.assertTrue(hasattr(stream_analysis, name))


if __name__ == "__main__":
    unittest.main()
