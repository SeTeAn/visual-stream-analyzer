from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

import stream_analysis
from stream_analysis import (
    RUN_PROVENANCE_SCHEMA_ID,
    AnalyzeRunProvenance,
    DataProvenance,
    EnvironmentProvenance,
    EvaluationRunProvenance,
    Fingerprint,
    ModelProvenance,
    RunRole,
    RunTimestamps,
    SourceProvenance,
    StageConfigProvenance,
)


def _fingerprint(character: str = "a") -> Fingerprint:
    return Fingerprint(algorithm="sha256", value=character * 64)


def _timestamps() -> RunTimestamps:
    return RunTimestamps(
        started_at="2026-06-20T10:00:00+03:00",
        finished_at="2026-06-20T10:00:01+03:00",
    )


def _source() -> SourceProvenance:
    return SourceProvenance(revision="source_revision_1", tree_fingerprint=_fingerprint("1"))


def _data() -> DataProvenance:
    return DataProvenance(
        manifest_digest=_fingerprint("2"),
        frame_content_digests={
            "frame_002": _fingerprint("4"),
            "frame_001": _fingerprint("3"),
        },
        candidate_snapshot_digest=_fingerprint("5"),
    )


def _stage_provenance() -> tuple[StageConfigProvenance, ...]:
    return (
        StageConfigProvenance(
            stage_id="representation",
            schema_id="stream_analysis.representation_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("6"),
            runtime_parameters={"device": "cpu", "batch_size": 4},
            diagnostic_parameters={"level": "standard"},
        ),
        StageConfigProvenance(
            stage_id="matching",
            schema_id="stream_analysis.matching_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("7"),
        ),
    )


def _models() -> tuple[ModelProvenance, ...]:
    return (
        ModelProvenance(
            model_id="dinov2_vits14",
            model_version="source_revision_1",
            source_fingerprint=_fingerprint("8"),
            checkpoint_fingerprint=_fingerprint("9"),
            metadata={"loading": "offline"},
        ),
    )


def _environment() -> EnvironmentProvenance:
    return EnvironmentProvenance(
        dependency_versions={"python": "3.11.9", "torch": "2.5.1+cu121"},
        environment_metadata={"platform": "windows"},
        requested_device="cuda",
        resolved_device="cpu",
        dtype="float32",
        determinism_enabled=True,
        determinism_metadata={"seed": 42, "deterministic_algorithms": True},
    )


def _analyze(role: RunRole = RunRole.PRIMARY_ANALYZE) -> AnalyzeRunProvenance:
    return AnalyzeRunProvenance(
        run_id="prediction_run_001",
        role=role,
        timestamps=_timestamps(),
        source=_source(),
        data=_data(),
        stage_configs=_stage_provenance(),
        models=_models(),
        environment=_environment(),
        output_schema_versions={"primary_stream_result": "1.0"},
        parent_run_ids=(),
    )


def _evaluation() -> EvaluationRunProvenance:
    return EvaluationRunProvenance(
        run_id="evaluation_run_001",
        role=RunRole.EVALUATION,
        parent_prediction_run_id="prediction_run_001",
        timestamps=_timestamps(),
        source=_source(),
        data=_data(),
        annotation_digest=_fingerprint("b"),
        evaluation_config_digest=_fingerprint("c"),
        stage_configs=_stage_provenance(),
        models=_models(),
        environment=_environment(),
        evaluator_version="1.0",
        output_schema_versions={"evaluation_report": "1.0"},
    )


class ProvenanceTest(unittest.TestCase):
    def test_fingerprint_has_explicit_algorithm_and_validated_value(self) -> None:
        fingerprint = Fingerprint(algorithm="sha256", value="A" * 64)
        self.assertEqual(fingerprint.algorithm, "sha256")
        self.assertEqual(fingerprint.value, "a" * 64)
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            Fingerprint(algorithm="sha256", value="short")
        with self.assertRaisesRegex(ValueError, "algorithm"):
            Fingerprint(algorithm="SHA-256", value="a" * 64)

    def test_timestamps_require_offsets_and_forward_order(self) -> None:
        timestamps = _timestamps()
        self.assertTrue(timestamps.started_at.endswith("+03:00"))
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            RunTimestamps(
                started_at="2026-06-20T10:00:00",
                finished_at="2026-06-20T10:00:01",
            )
        with self.assertRaisesRegex(ValueError, "must not precede"):
            RunTimestamps(
                started_at="2026-06-20T10:00:01+03:00",
                finished_at="2026-06-20T10:00:00+03:00",
            )

    def test_analyze_provenance_covers_explicit_run_metadata_and_is_immutable(self) -> None:
        provenance = _analyze()
        self.assertEqual(provenance.schema_id, RUN_PROVENANCE_SCHEMA_ID)
        self.assertEqual(
            tuple(stage.stage_id for stage in provenance.stage_configs),
            ("matching", "representation"),
        )
        self.assertEqual(tuple(provenance.data.frame_content_digests), ("frame_001", "frame_002"))
        self.assertEqual(provenance.environment.requested_device, "cuda")
        self.assertEqual(provenance.environment.resolved_device, "cpu")
        representation_stage = next(
            stage for stage in provenance.stage_configs if stage.stage_id == "representation"
        )
        self.assertEqual(representation_stage.runtime_parameters["batch_size"], 4)
        self.assertEqual(representation_stage.diagnostic_parameters["level"], "standard")
        with self.assertRaises(TypeError):
            representation_stage.runtime_parameters["batch_size"] = 8  # type: ignore[index]
        with self.assertRaises(TypeError):
            provenance.environment.dependency_versions["python"] = "other"  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            provenance.run_id = "other"  # type: ignore[misc]

    def test_diagnostic_and_ablation_roles_are_explicit(self) -> None:
        for role in (
            RunRole.DEVELOPMENT_DIAGNOSTIC,
            RunRole.FINAL_DIAGNOSTIC,
            RunRole.ABLATION,
        ):
            with self.subTest(role=role):
                self.assertIs(_analyze(role).role, role)
        with self.assertRaisesRegex(ValueError, "cannot use the evaluation role"):
            _analyze(RunRole.EVALUATION)

    def test_analyze_provenance_has_no_annotation_boundary(self) -> None:
        provenance = _analyze()
        names = {item.name for item in fields(AnalyzeRunProvenance)}
        self.assertNotIn("annotation_digest", names)
        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(AnalyzeRunProvenance)
        }
        values["annotation_digest"] = _fingerprint("d")
        with self.assertRaises(TypeError):
            AnalyzeRunProvenance(**values)  # type: ignore[arg-type]

        bad_stage = StageConfigProvenance(
            stage_id="representation",
            schema_id="stream_analysis.representation_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("6"),
            runtime_parameters={"annotation_digest": "sha256:forbidden"},
        )
        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(AnalyzeRunProvenance)
        }
        values["stage_configs"] = (bad_stage,)
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            AnalyzeRunProvenance(**values)

    def test_analyze_provenance_rejects_gt_evaluation_aliases_in_all_metadata(self) -> None:
        provenance = _analyze()

        def assert_rejected(**updates: object) -> None:
            values = {
                item.name: getattr(provenance, item.name)
                for item in fields(AnalyzeRunProvenance)
            }
            values.update(updates)
            with self.assertRaisesRegex(ValueError, "evaluator-only"):
                AnalyzeRunProvenance(**values)

        stage_runtime = StageConfigProvenance(
            stage_id="representation",
            schema_id="stream_analysis.representation_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("6"),
            runtime_parameters={"outer": [{"groundTruthLabels": "v1"}]},
        )
        assert_rejected(stage_configs=(stage_runtime,))

        stage_diagnostic = StageConfigProvenance(
            stage_id="representation",
            schema_id="stream_analysis.representation_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("6"),
            diagnostic_parameters={"outer": ({"ground_truth_mapping": "v1"},)},
        )
        assert_rejected(stage_configs=(stage_diagnostic,))

        model = provenance.models[0]
        model_values = {item.name: getattr(model, item.name) for item in fields(ModelProvenance)}
        model_values["metadata"] = {"nested": {"gtBoxes": True}}
        assert_rejected(models=(ModelProvenance(**model_values),))

        environment_values = {
            item.name: getattr(provenance.environment, item.name)
            for item in fields(EnvironmentProvenance)
        }
        environment_values["environment_metadata"] = {
            "nested": [{"evaluationResults": "forbidden"}]
        }
        assert_rejected(environment=EnvironmentProvenance(**environment_values))

        environment_values = {
            item.name: getattr(provenance.environment, item.name)
            for item in fields(EnvironmentProvenance)
        }
        environment_values["environment_metadata"] = {
            "annotationDigest": "sha256:forbidden"
        }
        assert_rejected(environment=EnvironmentProvenance(**environment_values))

        environment_values = {
            item.name: getattr(provenance.environment, item.name)
            for item in fields(EnvironmentProvenance)
        }
        environment_values["determinism_metadata"] = {
            "nested": {"evaluation_data_role": "forbidden"}
        }
        assert_rejected(environment=EnvironmentProvenance(**environment_values))

        environment_values = {
            item.name: getattr(provenance.environment, item.name)
            for item in fields(EnvironmentProvenance)
        }
        environment_values["dependency_versions"] = {"groundTruthLabels": "1.0"}
        assert_rejected(environment=EnvironmentProvenance(**environment_values))
        assert_rejected(output_schema_versions={"evaluationResults": "1.0"})

    def test_evaluation_provenance_allows_gt_evaluation_metadata(self) -> None:
        provenance = _evaluation()
        stage = StageConfigProvenance(
            stage_id="evaluator",
            schema_id="stream_analysis.evaluator_config.v1",
            config_version="1.0",
            semantic_digest=_fingerprint("e"),
            runtime_parameters={"groundTruthLabels": "v1"},
            diagnostic_parameters={"ground_truth_mapping": "per_stream"},
        )
        model = provenance.models[0]
        model_values = {item.name: getattr(model, item.name) for item in fields(ModelProvenance)}
        model_values["metadata"] = {"gtBoxes": True}
        environment_values = {
            item.name: getattr(provenance.environment, item.name)
            for item in fields(EnvironmentProvenance)
        }
        environment_values["environment_metadata"] = {
            "evaluationResults": "frozen"
        }
        environment_values["determinism_metadata"] = {
            "evaluation_data_role": "probe_development"
        }
        environment_values["dependency_versions"] = {"groundTruthLabels": "1.0"}
        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(EvaluationRunProvenance)
        }
        values["stage_configs"] = (stage,)
        values["models"] = (ModelProvenance(**model_values),)
        values["environment"] = EnvironmentProvenance(**environment_values)
        values["output_schema_versions"] = {"evaluationResults": "1.0"}
        accepted = EvaluationRunProvenance(**values)
        self.assertEqual(
            accepted.environment.environment_metadata["evaluationResults"],
            "frozen",
        )

    def test_evaluation_provenance_requires_annotation_and_parent_prediction(self) -> None:
        provenance = _evaluation()
        self.assertIs(provenance.role, RunRole.EVALUATION)
        self.assertEqual(provenance.parent_prediction_run_id, "prediction_run_001")
        self.assertIsInstance(provenance.annotation_digest, Fingerprint)

        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(EvaluationRunProvenance)
        }
        values.pop("annotation_digest")
        with self.assertRaises(TypeError):
            EvaluationRunProvenance(**values)  # type: ignore[arg-type]
        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(EvaluationRunProvenance)
        }
        values.pop("parent_prediction_run_id")
        with self.assertRaises(TypeError):
            EvaluationRunProvenance(**values)  # type: ignore[arg-type]
        values = {
            item.name: getattr(provenance, item.name)
            for item in fields(EvaluationRunProvenance)
        }
        values["role"] = RunRole.PRIMARY_ANALYZE
        with self.assertRaisesRegex(ValueError, "evaluation role"):
            EvaluationRunProvenance(**values)

    def test_parent_run_ids_are_unique_and_not_self_referential(self) -> None:
        provenance = _analyze()
        values = {item.name: getattr(provenance, item.name) for item in fields(AnalyzeRunProvenance)}
        values["parent_run_ids"] = ("parent_002", "parent_001")
        with_parents = AnalyzeRunProvenance(**values)
        self.assertEqual(with_parents.parent_run_ids, ("parent_001", "parent_002"))
        values["parent_run_ids"] = ("prediction_run_001",)
        with self.assertRaisesRegex(ValueError, "run_id itself"):
            AnalyzeRunProvenance(**values)

    def test_provenance_package_exports(self) -> None:
        for name in (
            "Fingerprint",
            "AnalyzeRunProvenance",
            "EvaluationRunProvenance",
            "RunRole",
        ):
            self.assertIn(name, stream_analysis.__all__)
            self.assertTrue(hasattr(stream_analysis, name))


if __name__ == "__main__":
    unittest.main()
