"""Preregistered development-only prototype evaluation for OCID visual types.

The runner consumes the persisted Gate C2 predicted-mask DINOv2 embeddings.  It
does not run inference and validates the development-only boundary before it
opens reviewed annotations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from stream_analysis.evaluation import load_annotation

from tools import evaluate_ocid_gate_c2_masked_dinov2 as gate_c2
from tools.ocid_gate_c1_common import (
    DEFAULT_BENCHMARK_SPEC,
    DEFAULT_REVIEWED_ROOT,
    DevelopmentStream,
    GateC1ContractError,
    load_development_inventory,
    write_canonical_json_atomic,
)


SCHEMA_VERSION = "ocid-gate-c2-1-visual-type-prototypes.v1"
HYPOTHESIS_SCHEMA_VERSION = "ocid-visual-type-hypotheses-1.0"
STATUS = "author_confirmed_preregistered_before_gate_c2_1_scores"
VARIANT = "mask_neutral_letterbox_v1"
FROZEN_GATE = 0.80
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPOTHESES_SPEC = REPOSITORY_ROOT / "data" / "ocid" / "benchmark" / "ocid_visual_type_hypotheses_v1.json"
DEFAULT_GATE_C2_INFERENCE_MANIFEST = REPOSITORY_ROOT / ".local_outputs" / "gate_c2" / "development" / "dinov2" / "inference_manifest.json"


@dataclass(frozen=True, slots=True)
class Prototype:
    stream_id: str
    source_label: int
    candidate_ids: tuple[str, ...]
    mean: np.ndarray
    medoid: np.ndarray
    medoid_candidate_id: str


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateC1ContractError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise GateC1ContractError(f"{label} must be a JSON object")
    return value


def _path_digest(path: Path, expected: object, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.casefold()):
        raise GateC1ContractError(f"{label} expected SHA-256 must contain 64 hexadecimal digits")
    actual = gate_c2._sha256(path.resolve(strict=True))
    if actual != expected.casefold():
        raise GateC1ContractError(f"{label} SHA-256 mismatch: expected {expected.casefold()}, got {actual}")
    return actual


def _as_labels(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise GateC1ContractError(f"{label} must contain exactly two positive integer source labels")
    if value[0] == value[1]:
        raise GateC1ContractError(f"{label} source labels must be distinct")
    return (value[0], value[1])


def _validate_hypotheses(spec: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != HYPOTHESIS_SCHEMA_VERSION:
        raise GateC1ContractError("unsupported visual-type hypotheses schema")
    if spec.get("status") != STATUS or spec.get("scope") != "development_only" or spec.get("heldout_access") != "none":
        raise GateC1ContractError("visual-type hypotheses must be confirmed development-only with heldout_access=none")
    if spec.get("benchmark_id") != benchmark.get("benchmark_id"):
        raise GateC1ContractError("visual-type hypotheses benchmark_id mismatch")
    source = spec.get("embedding_source")
    policy = spec.get("evaluation_policy")
    hypotheses = spec.get("hypotheses")
    checks = spec.get("hard_negative_checks")
    if not isinstance(source, Mapping) or source.get("variant") != VARIANT or source.get("prototype_policy") != "l2_normalized_mean_v1" or source.get("secondary_prototype_policy") != "observed_medoid_v1" or source.get("similarity_policy") != "shifted_cosine_v1" or source.get("frozen_visual_gate") != FROZEN_GATE:
        raise GateC1ContractError("visual-type hypotheses violate frozen embedding policy")
    if not isinstance(policy, Mapping) or not isinstance(hypotheses, list) or not isinstance(checks, list):
        raise GateC1ContractError("visual-type hypotheses are missing evaluation policy or cases")
    expected_roles = {"primary_positive": 4, "challenge_positive": 1, "uncertain_diagnostic": 2}
    counts = {key: 0 for key in expected_roles}
    known_streams = {stream for group in benchmark.get("roles", {}).get("development", []) for stream in group.get("member_streams", [])}
    known_ids = {gate_c2_name for gate_c2_name in (_stream_id(value) for value in known_streams)}
    seen_ids: set[str] = set()
    for item in hypotheses:
        if not isinstance(item, Mapping) or not isinstance(item.get("hypothesis_id"), str) or item["hypothesis_id"] in seen_ids:
            raise GateC1ContractError("every visual-type hypothesis requires a unique hypothesis_id")
        seen_ids.add(item["hypothesis_id"])
        role = item.get("role")
        if role not in expected_roles:
            raise GateC1ContractError("visual-type hypothesis has unsupported role")
        counts[role] += 1
        _as_labels(item.get("source_labels"), f"hypothesis {item['hypothesis_id']}")
        streams = item.get("stream_ids")
        if not isinstance(streams, list) or len(streams) != 2 or len(set(streams)) != 2 or not all(isinstance(stream, str) and stream in known_ids for stream in streams):
            raise GateC1ContractError(f"hypothesis {item['hypothesis_id']} must name exactly two development streams")
    if counts != expected_roles:
        raise GateC1ContractError("visual-type hypothesis roles do not match the preregistered protocol")
    if policy.get("primary_case_count") != 8 or policy.get("primary_directed_query_count") != 16:
        raise GateC1ContractError("visual-type primary-case counts do not match the preregistered protocol")
    support = policy.get("support_rule")
    if not isinstance(support, Mapping) or support != {
        "required_available_primary_cases": 8,
        "minimum_passed_primary_cases": 7,
        "minimum_primary_hypotheses_passing_both_views": 3,
        "required_hard_negative_checks_passed": 2,
    }:
        raise GateC1ContractError("visual-type support rule does not match the preregistered protocol")
    if len(checks) != 2:
        raise GateC1ContractError("visual-type protocol requires exactly two hard-negative checks")
    for item in checks:
        if not isinstance(item, Mapping) or not isinstance(item.get("stream_id"), str) or item["stream_id"] not in known_ids:
            raise GateC1ContractError("hard-negative check must use one development stream")
        _as_labels(item.get("positive_source_labels"), "hard-negative positive_source_labels")
        negative = item.get("negative_source_label")
        if isinstance(negative, bool) or not isinstance(negative, int) or negative <= 0:
            raise GateC1ContractError("hard-negative negative_source_label must be positive integer")


def _stream_id(canonical_name: str) -> str:
    return "ocid_" + canonical_name.lower().replace("/", "_")


def _validate_gate_c2_manifest(manifest: Mapping[str, Any], hypotheses: Mapping[str, Any], manifest_path: Path) -> Path:
    source = hypotheses["embedding_source"]
    _path_digest(manifest_path, source.get("gate_c2_inference_manifest_sha256"), "Gate C2 inference manifest")
    if (
        manifest.get("schema_version") != gate_c2.INFERENCE_SCHEMA_VERSION
        or manifest.get("embedding_schema_version") != gate_c2.EMBEDDING_SCHEMA_VERSION
        or manifest.get("status") != "rgb_predicted_mask_inference_completed_before_ground_truth"
        or manifest.get("scope") != "development_only_gate_c2"
        or manifest.get("heldout_access") != "none"
        or manifest.get("variants") != list(gate_c2.SUPPORTED_VARIANTS)
    ):
        raise GateC1ContractError("Gate C2 manifest is not the frozen development-only persisted inference")
    artifact = manifest.get("embedding_artifact")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        raise GateC1ContractError("Gate C2 manifest is missing embedding artifact provenance")
    npz_path = Path(artifact["path"]).resolve(strict=True)
    expected = source.get("embedding_npz_sha256")
    _path_digest(npz_path, expected, "Gate C2 embedding NPZ")
    if artifact.get("sha256") != expected:
        raise GateC1ContractError("Gate C2 manifest embedding SHA-256 does not match preregistration")
    return npz_path


def _load_gate_c2_inputs(
    *, inventory: Sequence[DevelopmentStream], manifest: Mapping[str, Any], npz_path: Path,
) -> tuple[dict[str, tuple[gate_c2.CandidateInput, ...]], dict[str, np.ndarray]]:
    selected = manifest.get("selected_candidates")
    sam2 = manifest.get("sam2_inference_manifest")
    if not isinstance(selected, Mapping) or not isinstance(sam2, Mapping) or not isinstance(selected.get("path"), str) or not isinstance(sam2.get("path"), str):
        raise GateC1ContractError("Gate C2 manifest is missing selected-candidate or SAM2 provenance")
    inputs, _selected, _sam2 = gate_c2.load_predicted_candidate_inputs(
        inventory=inventory,
        selected_candidates_path=Path(selected["path"]),
        sam2_inference_manifest_path=Path(sam2["path"]),
        expected_selected_candidates_sha256=str(selected.get("sha256", "")),
        expected_sam2_inference_sha256=str(sam2.get("sha256", "")),
    )
    expected = [item for stream in inventory for item in inputs[stream.stream_id]]
    if len(expected) != gate_c2.FROZEN_CANDIDATE_COUNT:
        raise GateC1ContractError("Gate C2.1 requires the exact frozen Gate C2 candidate inventory")
    gate_c2._validate_candidate_index(manifest.get("candidate_index"), expected)
    embeddings = gate_c2._load_embedding_npz(npz_path, len(expected))
    return inputs, embeddings


def _unit_mean(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise GateC1ContractError("cannot form a prototype without matched observations")
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= 0:
        raise GateC1ContractError("prototype mean must have a non-zero finite norm")
    return np.asarray(mean / norm, dtype=np.float32)


def _shifted_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return (float(np.clip(np.dot(left.astype(np.float64), right.astype(np.float64)), -1.0, 1.0)) + 1.0) / 2.0


def _prototype(stream_id: str, source_label: int, observations: Sequence[tuple[str, np.ndarray]]) -> Prototype:
    ordered = tuple(sorted(observations, key=lambda item: item[0]))
    if not ordered:
        raise GateC1ContractError("cannot create a prototype without observations")
    mean = _unit_mean([value for _identifier, value in ordered])
    averages = [
        (sum(_shifted_cosine(vector, other) for _other_id, other in ordered) / len(ordered), identifier, vector)
        for identifier, vector in ordered
    ]
    # Sorting makes the documented candidate-id tie-break explicit and stable.
    _score, medoid_id, medoid = min(averages, key=lambda row: (-row[0], row[1]))
    return Prototype(stream_id, source_label, tuple(identifier for identifier, _ in ordered), mean, medoid.copy(), medoid_id)


def _rank(query: Prototype, counterpart_label: int, gallery: Mapping[int, Prototype]) -> tuple[int | None, float | None, int | None, float | None, str | None]:
    rows = sorted(
        ((_shifted_cosine(query.mean, value.mean), label) for label, value in gallery.items() if label != query.source_label),
        key=lambda row: (-row[0], row[1]),
    )
    counterpart_score = next((score for score, label in rows if label == counterpart_label), None)
    counterpart_rank = next((index + 1 for index, (_score, label) in enumerate(rows) if label == counterpart_label), None)
    others = [(score, label) for score, label in rows if label != counterpart_label]
    if not others:
        return counterpart_rank, counterpart_score, None, None, None
    hardest_score, hardest_label = others[0]
    return counterpart_rank, counterpart_score, hardest_score, counterpart_score - hardest_score if counterpart_score is not None else None, f"{hardest_label:03d}"


def _case_report(
    *, hypothesis: Mapping[str, Any], stream_id: str, prototypes: Mapping[int, Prototype],
    primary_minimum: int, diagnostic_minimum: int, gallery_minimum: int,
) -> dict[str, Any]:
    left, right = _as_labels(hypothesis["source_labels"], f"hypothesis {hypothesis['hypothesis_id']}")
    role = str(hypothesis["role"])
    minimum = diagnostic_minimum if role == "uncertain_diagnostic" else primary_minimum
    coverage = {
        f"{label:03d}": {
            "matched_observation_count": len(prototypes[label].candidate_ids) if label in prototypes else 0,
            "minimum_required": minimum,
            "eligible": label in prototypes and len(prototypes[label].candidate_ids) >= minimum,
            "gallery_eligible": label in prototypes and len(prototypes[label].candidate_ids) >= gallery_minimum,
        }
        for label in (left, right)
    }
    available = all(value["eligible"] for value in coverage.values())
    base = {
        "hypothesis_id": hypothesis["hypothesis_id"], "role": role, "stream_id": stream_id,
        "source_labels": [left, right], "coverage": coverage, "available": available,
        "weak_evidence": role == "uncertain_diagnostic" and any(value["matched_observation_count"] < primary_minimum for value in coverage.values()),
    }
    if not available or left not in prototypes or right not in prototypes:
        return {**base, "status": "unavailable", "case_pass": False, "reason": "insufficient_matched_observations"}
    gallery = {
        label: value
        for label, value in prototypes.items()
        if len(value.candidate_ids) >= gallery_minimum
    }
    # A preregistered uncertain pair may be calculated from one observation
    # per member and is then explicitly marked as weak evidence.  Keep the
    # rest of the comparison gallery at the stricter two-observation minimum.
    gallery[left] = prototypes[left]
    gallery[right] = prototypes[right]
    mean_score = _shifted_cosine(prototypes[left].mean, prototypes[right].mean)
    medoid_score = _shifted_cosine(prototypes[left].medoid, prototypes[right].medoid)
    directed: dict[str, Any] = {}
    for query, counterpart in ((left, right), (right, left)):
        rank, score, hardest, margin, hardest_label = _rank(prototypes[query], counterpart, gallery)
        directed[f"{query:03d}_to_{counterpart:03d}"] = {
            "counterpart_rank": rank, "counterpart_score": score,
            "hardest_other_score": hardest, "hardest_other_source_label": hardest_label,
            "counterpart_minus_hardest_other_margin": margin,
            "rank_one": rank == 1, "strictly_positive_margin": margin is not None and margin > 0,
        }
    passed = all(value["rank_one"] and value["strictly_positive_margin"] for value in directed.values())
    return {
        **base, "status": "available", "primary_mean_pair_score": mean_score,
        "secondary_medoid_pair_score": medoid_score, "frozen_gate": FROZEN_GATE,
        "mean_pair_passes_frozen_gate": mean_score >= FROZEN_GATE,
        "directed_queries": directed, "case_pass": passed,
    }


def _hard_negative_report(
    *, check: Mapping[str, Any], prototypes: Mapping[int, Prototype], gallery_minimum: int,
) -> dict[str, Any]:
    left, right = _as_labels(check["positive_source_labels"], "hard-negative positive_source_labels")
    negative = int(check["negative_source_label"])
    labels = (left, right, negative)
    unavailable = [label for label in labels if label not in prototypes or len(prototypes[label].candidate_ids) < gallery_minimum]
    base = {"check_id": check.get("check_id"), "stream_id": check["stream_id"], "positive_source_labels": [left, right], "negative_source_label": negative}
    if unavailable:
        return {**base, "available": False, "status": "unavailable", "missing_gallery_labels": unavailable, "check_pass": False}
    positive = _shifted_cosine(prototypes[left].mean, prototypes[right].mean)
    negative_left = _shifted_cosine(prototypes[negative].mean, prototypes[left].mean)
    negative_right = _shifted_cosine(prototypes[negative].mean, prototypes[right].mean)
    return {
        **base, "available": True, "status": "available", "positive_pair_score": positive,
        "negative_to_first_positive_score": negative_left,
        "negative_to_second_positive_score": negative_right,
        "positive_minus_max_negative_margin": positive - max(negative_left, negative_right),
        "check_pass": positive > negative_left and positive > negative_right,
    }


def _support(primary: Sequence[Mapping[str, Any]], hard_negatives: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    rule = policy["support_rule"]
    available = sum(bool(case.get("available")) for case in primary)
    passed = sum(bool(case.get("case_pass")) for case in primary)
    by_hypothesis: dict[str, list[Mapping[str, Any]]] = {}
    for case in primary:
        by_hypothesis.setdefault(str(case["hypothesis_id"]), []).append(case)
    both_views = sum(len(cases) == 2 and all(bool(case.get("case_pass")) for case in cases) for cases in by_hypothesis.values())
    hard_passed = sum(bool(case.get("check_pass")) for case in hard_negatives)
    checks = {
        "all_required_primary_cases_available": available == rule["required_available_primary_cases"],
        "minimum_primary_cases_passed": passed >= rule["minimum_passed_primary_cases"],
        "minimum_primary_hypotheses_passing_both_views": both_views >= rule["minimum_primary_hypotheses_passing_both_views"],
        "required_hard_negative_checks_passed": hard_passed >= rule["required_hard_negative_checks_passed"],
    }
    return {
        "rule": dict(rule), "available_primary_case_count": available, "passed_primary_case_count": passed,
        "primary_hypotheses_passing_both_views": both_views,
        "passed_hard_negative_check_count": hard_passed, "checks": checks,
        "supported": all(checks.values()),
    }


def _separation(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives: list[float] = []
    competitors: list[float] = []
    for case in primary:
        if not case.get("available"):
            continue
        positives.append(float(case["primary_mean_pair_score"]))
        for query in case["directed_queries"].values():
            if query["hardest_other_score"] is not None:
                competitors.append(float(query["hardest_other_score"]))
    minimum_positive = min(positives) if positives else None
    maximum_competitor = max(competitors) if competitors else None
    gap = (
        minimum_positive - maximum_competitor
        if minimum_positive is not None and maximum_competitor is not None
        else None
    )
    return {
        "positive_pair_count": len(positives), "competitor_query_count": len(competitors),
        "minimum_positive_score": minimum_positive, "maximum_competitor_score": maximum_competitor,
        "minimum_positive_minus_maximum_competitor_gap": gap,
        "strict_interval_exists": minimum_positive is not None and maximum_competitor is not None and minimum_positive > maximum_competitor,
        "interpretation": "Diagnostic only; no production visual-type threshold is fit or selected.",
    }


def _label_identities(annotation: Any) -> dict[int, str]:
    rows = annotation.raw.get("expected_element_instances")
    if not isinstance(rows, list):
        raise GateC1ContractError("reviewed annotation is missing expected_element_instances")
    result: dict[int, set[str]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("source_label"), int) and isinstance(row.get("visual_type_id"), str):
            result.setdefault(int(row["source_label"]), set()).add(str(row["visual_type_id"]))
    invalid = [label for label, identities in result.items() if len(identities) != 1]
    if invalid:
        raise GateC1ContractError(f"source labels must resolve to one evaluation-side physical identity: {invalid}")
    return {label: next(iter(identities)) for label, identities in result.items()}


def _prototypes_for_stream(
    *, stream: DevelopmentStream, inputs: Sequence[gate_c2.CandidateInput],
    embeddings: Mapping[str, np.ndarray], positions: Sequence[int], annotation: Any,
) -> tuple[dict[int, Prototype], dict[int, list[gate_c2.CandidateInput]]]:
    identities, _ious, _evaluation = gate_c2._assignment_identity(annotation, inputs)
    labels_by_identity = {identity: label for label, identity in _label_identities(annotation).items()}
    observations: dict[int, list[tuple[str, np.ndarray]]] = {}
    items_by_label: dict[int, list[gate_c2.CandidateInput]] = {}
    for item, position in zip(inputs, positions, strict=True):
        identity = identities.get(item.candidate.candidate_id)
        label = labels_by_identity.get(identity)
        if label is None:  # Evaluation-only false positives never make prototypes.
            continue
        observations.setdefault(label, []).append((item.candidate.candidate_id, embeddings[VARIANT][position].copy()))
        items_by_label.setdefault(label, []).append(item)
    return ({label: _prototype(stream.stream_id, label, values) for label, values in observations.items()}, items_by_label)


def _write_contact_sheet(
    *, cases: Sequence[Mapping[str, Any]], items_by_stream: Mapping[str, Mapping[int, Sequence[gate_c2.CandidateInput]]],
    inventory: Mapping[str, DevelopmentStream], artifact_root: Path,
) -> dict[str, Any]:
    panels: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    colors = ((255, 190, 0), (0, 220, 255))
    for case in sorted(cases, key=lambda row: (str(row["hypothesis_id"]), str(row["stream_id"]))):
        if not case.get("available"):
            continue
        stream_id = str(case["stream_id"]); left, right = (int(value) for value in case["source_labels"])
        by_label = items_by_stream[stream_id]
        common = sorted({item.candidate.frame_id for item in by_label.get(left, ())}.intersection(item.candidate.frame_id for item in by_label.get(right, ())))
        if not common:
            continue
        frame_id = common[0]
        decoded = gate_c2._load_stream(inventory[stream_id].stream_directory)
        frame = next(frame for frame in decoded.frames if frame.frame_id == frame_id)
        image = Image.frombytes("RGB", (frame.image_size.width, frame.image_size.height), frame.rgb_bytes).convert("RGBA")
        for label, color in zip((left, right), colors, strict=True):
            for item in by_label[label]:
                if item.candidate.frame_id != frame_id:
                    continue
                x, y, width, height = map(int, (item.candidate.bbox.x, item.candidate.bbox.y, item.candidate.bbox.width, item.candidate.bbox.height))
                alpha = Image.fromarray(item.mask_record.mask.astype(np.uint8) * 105, mode="L")
                layer = Image.new("RGBA", image.size, color + (0,))
                layer.paste(color + (105,), (x, y, x + width, y + height), alpha)
                image = Image.alpha_composite(image, layer)
                draw = ImageDraw.Draw(image)
                draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color + (255,), width=2)
                draw.text((x + 2, max(0, y - 12)), f"{label:03d}", fill=color + (255,))
        image = image.convert("RGB")
        image.thumbnail((480, 360))
        panels.append(image)
        metadata.append({
            "hypothesis_id": case["hypothesis_id"],
            "stream_id": stream_id,
            "frame_id": frame_id,
            "evaluation_assigned_source_labels": [left, right],
        })
    root = artifact_root.resolve(strict=False); root.mkdir(parents=True, exist_ok=True)
    destination = root / "visual_type_prototype_contact_sheet.png"
    if not panels:
        Image.new("RGB", (1, 1), (0, 0, 0)).save(destination, format="PNG")
    else:
        width = max(panel.width for panel in panels); height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width * min(2, len(panels)), height * ((len(panels) + 1) // 2)), (20, 20, 20))
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 2) * width, (index // 2) * height))
        canvas.save(destination, format="PNG", optimize=True)
    return {"relative_path": destination.relative_to(root).as_posix(), "sha256": gate_c2._sha256(destination), "panel_count": len(metadata), "panels": metadata, "pixel_sources": "development RGB plus predicted candidate masks only"}


def run_gate_c2_1(
    *, hypotheses_spec_path: Path, gate_c2_inference_manifest_path: Path,
    benchmark_spec_path: Path, reviewed_root: Path, artifact_root: Path, output_path: Path,
) -> dict[str, Any]:
    # This order is intentional: no reviewed annotation is opened before all
    # preregistration, scope, manifest and persisted-embedding checks complete.
    hypotheses = _object(hypotheses_spec_path, "visual-type hypotheses")
    benchmark = _object(benchmark_spec_path, "benchmark spec")
    _validate_hypotheses(hypotheses, benchmark)
    c2_manifest = _object(gate_c2_inference_manifest_path, "Gate C2 inference manifest")
    npz_path = _validate_gate_c2_manifest(c2_manifest, hypotheses, gate_c2_inference_manifest_path)
    inventory = load_development_inventory(benchmark_spec_path, reviewed_root)
    inputs, embeddings = _load_gate_c2_inputs(inventory=inventory, manifest=c2_manifest, npz_path=npz_path)
    policy = hypotheses["evaluation_policy"]
    primary_minimum = int(policy["primary_minimum_matched_observations_per_instance"])
    diagnostic_minimum = int(policy["diagnostic_minimum_matched_observations_per_instance"])
    gallery_minimum = int(policy["gallery_minimum_matched_observations_per_instance"])
    cursor = 0; prototypes: dict[str, dict[int, Prototype]] = {}; items_by_stream: dict[str, dict[int, list[gate_c2.CandidateInput]]] = {}
    inventory_by_id = {stream.stream_id: stream for stream in inventory}
    for stream in inventory:
        values = inputs[stream.stream_id]; positions = list(range(cursor, cursor + len(values))); cursor += len(values)
        annotation = load_annotation(stream.annotation_path, manifest_path=stream.stream_directory / "manifest.json")
        prototypes[stream.stream_id], items_by_stream[stream.stream_id] = _prototypes_for_stream(stream=stream, inputs=values, embeddings=embeddings, positions=positions, annotation=annotation)
    cases = [
        _case_report(hypothesis=hypothesis, stream_id=stream_id, prototypes=prototypes[stream_id], primary_minimum=primary_minimum, diagnostic_minimum=diagnostic_minimum, gallery_minimum=gallery_minimum)
        for hypothesis in hypotheses["hypotheses"] for stream_id in hypothesis["stream_ids"]
    ]
    primary = [case for case in cases if case["role"] == "primary_positive"]
    challenge = [case for case in cases if case["role"] == "challenge_positive"]
    uncertain = [case for case in cases if case["role"] == "uncertain_diagnostic"]
    hard_negative = [_hard_negative_report(check=check, prototypes=prototypes[check["stream_id"]], gallery_minimum=gallery_minimum) for check in hypotheses["hard_negative_checks"]]
    contact_sheet = _write_contact_sheet(cases=cases, items_by_stream=items_by_stream, inventory=inventory_by_id, artifact_root=artifact_root)
    report = {
        "schema_version": SCHEMA_VERSION, "experiment_id": hypotheses["experiment_id"], "status": "completed_development_only", "scope": "development_only", "heldout_access": "none",
        "provenance": {
            "hypotheses_spec": {"path": hypotheses_spec_path.resolve().as_posix(), "sha256": gate_c2._sha256(hypotheses_spec_path.resolve())},
            "benchmark_spec": {"path": benchmark_spec_path.resolve().as_posix(), "sha256": gate_c2._sha256(benchmark_spec_path.resolve())},
            "gate_c2_inference_manifest": {"path": gate_c2_inference_manifest_path.resolve().as_posix(), "sha256": gate_c2._sha256(gate_c2_inference_manifest_path.resolve())},
            "embedding_npz": {"path": npz_path.as_posix(), "sha256": gate_c2._sha256(npz_path)},
            "variant": VARIANT,
        },
        "primary_positive_cases": primary, "challenge_positive_cases": challenge,
        "uncertain_diagnostics": uncertain, "apple_hard_negative_checks": hard_negative,
        "support_decision": _support(primary, hard_negative, policy),
        "positive_vs_competitor_separation": _separation(primary), "contact_sheet": contact_sheet,
        "limitations": [
            "Author visual-type hypotheses are not OCID semantic labels or universal ground truth.",
            "False-positive candidates do not form prototypes; extractor misses can make a case unavailable.",
            "Top and bottom are paired views, not independent physical-instance examples.",
            "The 0.80 gate and separation interval are diagnostics only; no production threshold is fit.",
            "Contact-sheet pixels use development RGB and predicted masks only; reviewed masks are never rendered.",
            "No held-out RGB, predictions, embeddings, overlays, annotations, or metrics are read.",
        ],
    }
    write_canonical_json_atomic(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses-spec", type=Path, default=DEFAULT_HYPOTHESES_SPEC)
    parser.add_argument("--gate-c2-inference-manifest", type=Path, default=DEFAULT_GATE_C2_INFERENCE_MANIFEST)
    parser.add_argument("--benchmark-spec", type=Path, default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_gate_c2_1(hypotheses_spec_path=args.hypotheses_spec, gate_c2_inference_manifest_path=args.gate_c2_inference_manifest, benchmark_spec_path=args.benchmark_spec, reviewed_root=args.reviewed_root, artifact_root=args.artifact_root, output_path=args.output)
    except (GateC1ContractError, OSError, RuntimeError, ValueError) as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
