"""Coordinate deterministic double-blind OCID component review.

The tool has three fail-closed stages:

``prepare``
    Validate a frozen component review pack and publish two identical blind
    assignment passes split into four stream-preserving shards.
``merge-primary``
    Validate the eight completed assignments, preserve both raw reviews, and
    publish the adjudication queue for every non-agreed component.
``finalize``
    Validate completed adjudication and publish the exact decisions ledger
    consumed by the Gate B benchmark builder.

This is annotation/evaluation tooling.  It never reads model predictions or
computes model metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))

from stream_analysis.evaluation.review_contract import canonical_reviewer_id


PACK_SCHEMA = "ocid-component-review-pack-2.0"
PACK_SCOPE = "evaluation_only_annotation_review"
PACK_STATUS = "pending_review"
DECISIONS_SCHEMA = "ocid-component-review-decisions-1.0"
POLICY_ID = "largest_plus_reviewed_components_v1"
ASSIGNMENT_SCHEMA = "ocid-component-blind-assignment-1.0"
PRIMARY_MERGE_SCHEMA = "ocid-component-primary-merge-1.0"
ADJUDICATION_SCHEMA = "ocid-component-adjudication-1.0"
COORDINATION_SUMMARY_SCHEMA = "ocid-component-review-coordination-summary-1.0"
AUTHOR_PREVIEW_SCHEMA = "ocid-component-author-preview-1.0"
AUTHOR_DECISION_SCHEMA = "ocid-component-author-decision-queue-1.0"
ISOLATED_ARTIFACTS_SCHEMA = "ocid-component-isolated-artifacts-1.0"

PASSES = ("pass_a", "pass_b")
SHARD_IDS = ("shard_01", "shard_02", "shard_03", "shard_04")
PRIMARY_DECISIONS = frozenset(
    {"KEEP", "DROP", "UNRESOLVED", "INTEGRITY_ALERT"}
)
FINAL_DECISIONS = frozenset({"KEEP", "DROP"})
FINAL_CONFIDENCES = frozenset({"HIGH", "MEDIUM"})
INTEGRITY_DECISIONS = frozenset({"OK", "INTEGRITY_ALERT"})
INTEGRITY_CODES = frozenset(
    {"I_C001_INVALID", "I_LABEL_SWITCH", "I_MULTIPLE_OBJECTS"}
)
FORBIDDEN_ARTIFACT_TOKENS = (
    "prediction",
    "metric",
    "embedding",
    "extractor",
    "analysis_streams",
)
RATIONALE_CODES: dict[str, frozenset[str]] = {
    "KEEP": frozenset({"K_TEMPORAL", "K_OCCLUSION", "K_CONTINUITY"}),
    "DROP": frozenset(
        {
            "D_OTHER_REGION",
            "D_TEMPORAL_OTHER",
            "D_TRANSIENT_ARTIFACT",
            "D_BOUNDARY_BLEED",
        }
    ),
    "UNRESOLVED": frozenset(
        {"A_OCCLUDED", "A_NO_TEMPORAL", "A_CONFLICT", "A_IMAGE_QUALITY"}
    ),
    "INTEGRITY_ALERT": frozenset(
        {"I_C001_INVALID", "I_LABEL_SWITCH", "I_MULTIPLE_OBJECTS"}
    ),
}


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array.")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer.")
    return value


def _digest(value: object, context: str) -> str:
    text = _text(value, context)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return text


def _reviewer_id(value: object, context: str) -> str:
    return canonical_reviewer_id(value, context)


def _forbid_output_artifact(relative: str, context: str) -> None:
    lowered = relative.casefold()
    if any(token in lowered for token in FORBIDDEN_ARTIFACT_TOKENS):
        raise ValueError(f"{context} contains a forbidden model-output artifact: {relative}.")


def _load_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} must be a file: {resolved}.")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {context}: {resolved}.") from exc
    return _object(value, context), raw


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _safe_relative_path(value: object, context: str) -> Path:
    text = _text(value, context)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a safe relative path.")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_pack_artifact(pack_root: Path, value: object, context: str) -> Path:
    relative = _safe_relative_path(value, context)
    resolved = (pack_root / relative).resolve(strict=True)
    if not _inside(resolved, pack_root):
        raise ValueError(f"{context} resolves outside the review pack.")
    return resolved


def _atomic_publish(destination: Path, build: Any) -> None:
    destination = Path(destination).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        build(staging)
        if destination.exists():
            raise FileExistsError(f"Output root appeared during build: {destination}.")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _artifact_records(root: Path, *, exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded = set(exclude)
    records: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    return records


def _pack_fingerprint(manifest_bytes: bytes) -> str:
    return hashlib.sha256(manifest_bytes).hexdigest()


def _copy_sanitized_blind_visual(
    source: Path, destination: Path, template_case: Mapping[str, Any]
) -> None:
    """Copy blind evidence while removing any legacy area/proposal table."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    except (UnidentifiedImageError, OSError):
        # Tiny non-image fixtures remain useful for schema-only unit tests. Real
        # review packs are PNG and always take the sanitizing path above.
        shutil.copyfile(source, destination)
        return
    draw = ImageDraw.Draw(image)
    table_x = 1070 if image.width >= 1200 else max(1, round(image.width * 0.64))
    table_y = 350 if image.height >= 500 else max(1, round(image.height * 0.37))
    draw.rectangle((table_x, table_y, image.width, image.height), fill=(12, 12, 12))
    font = ImageFont.load_default()
    analysis = _object(template_case.get("analysis"), "template case analysis")
    lines = [
        f"source_sequence: {template_case['source_sequence']}",
        f"source: {template_case['source_filename']} | label={template_case['source_label']}",
        f"mask_sha256: {template_case['source_mask_sha256']}",
        f"raw_bbox: {analysis['raw_bbox']}",
        "",
        "ID    bbox(x,y,w,h)",
    ]
    for raw_component in _array(analysis.get("components"), "analysis.components"):
        component = _object(raw_component, "analysis component")
        bbox = _object(component.get("bbox"), "component.bbox")
        lines.append(
            f"{component['id']:<5} "
            f"({bbox['x']},{bbox['y']},{bbox['width']},{bbox['height']})"
        )
    y = table_y + 12
    for line in lines:
        draw.text((table_x + 12, y), line, fill=(235, 235, 235), font=font)
        y += 18
    image.save(destination, format="PNG", compress_level=6, optimize=False)


def _validate_pack(
    review_pack: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    root = Path(review_pack).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Review pack must be a directory: {root}.")
    manifest, manifest_bytes = _load_json(
        root / "component_review_manifest.json", "component review manifest"
    )
    if manifest.get("schema_version") != PACK_SCHEMA:
        raise ValueError("Unsupported component review manifest schema_version.")
    if manifest.get("scope") != PACK_SCOPE:
        raise ValueError("Component review manifest has an invalid scope.")
    if manifest.get("status") != PACK_STATUS:
        raise ValueError("Component review manifest status must be pending_review.")
    benchmark_id = _text(manifest.get("benchmark_id"), "manifest.benchmark_id")
    policy = _object(manifest.get("policy"), "manifest.policy")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("Component review manifest policy_id is not frozen policy.")
    boundary = _object(
        manifest.get("ground_truth_boundary"), "manifest.ground_truth_boundary"
    )
    if (
        boundary.get("scope") != PACK_SCOPE
        or boundary.get("model_predictions_accessed") is not False
        or boundary.get("metrics_computed") is not False
        or boundary.get("forbidden_consumer") != "analyze"
    ):
        raise ValueError("Component review manifest violates the GT boundary.")
    protocol = _object(manifest.get("review_protocol"), "manifest.review_protocol")
    if (
        protocol.get("primary_review_count") != 2
        or protocol.get("primary_review_is_blind") is not True
        or protocol.get("component_decision_values")
        != ["KEEP", "DROP", "UNRESOLVED", "INTEGRITY_ALERT"]
    ):
        raise ValueError("Component review manifest has an invalid review protocol.")

    template_descriptor = _object(
        manifest.get("decision_template"), "manifest.decision_template"
    )
    template_path = _resolve_pack_artifact(
        root, template_descriptor.get("path"), "manifest.decision_template.path"
    )
    expected_template_sha = _digest(
        template_descriptor.get("sha256"), "manifest.decision_template.sha256"
    )
    if _sha256_path(template_path) != expected_template_sha:
        raise ValueError("Decision template SHA-256 does not match the manifest.")
    if template_path.stat().st_size != _integer(
        template_descriptor.get("size_bytes"),
        "manifest.decision_template.size_bytes",
    ):
        raise ValueError("Decision template size does not match the manifest.")

    artifacts = _array(manifest.get("artifacts"), "manifest.artifacts")
    if _integer(manifest.get("artifact_count"), "manifest.artifact_count") != len(
        artifacts
    ):
        raise ValueError("Manifest artifact_count does not match artifacts.")
    seen_artifacts: set[str] = set()
    for position, raw_record in enumerate(artifacts):
        context = f"manifest.artifacts[{position}]"
        record = _object(raw_record, context)
        relative = _safe_relative_path(record.get("path"), f"{context}.path").as_posix()
        if relative in seen_artifacts:
            raise ValueError(f"Duplicate manifest artifact path: {relative}.")
        _forbid_output_artifact(relative, context)
        seen_artifacts.add(relative)
        path = _resolve_pack_artifact(root, relative, f"{context}.path")
        if path.stat().st_size != _integer(
            record.get("size_bytes"), f"{context}.size_bytes"
        ):
            raise ValueError(f"Artifact size mismatch: {relative}.")
        expected_sha = _digest(record.get("sha256"), f"{context}.sha256")
        if _sha256_path(path) != expected_sha:
            raise ValueError(f"Artifact SHA-256 mismatch: {relative}.")
    if template_path.relative_to(root).as_posix() not in seen_artifacts:
        raise ValueError("Decision template is absent from the artifact inventory.")
    actual_artifacts = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "component_review_manifest.json"
    }
    if actual_artifacts != seen_artifacts:
        missing = sorted(seen_artifacts - actual_artifacts)
        unregistered = sorted(actual_artifacts - seen_artifacts)
        raise ValueError(
            "Manifest artifact inventory is not exact; "
            f"missing={missing}, unregistered={unregistered}."
        )

    template, _ = _load_json(template_path, "decision template")
    if template.get("schema_version") != DECISIONS_SCHEMA:
        raise ValueError("Unsupported decision template schema_version.")
    if template.get("benchmark_id") != benchmark_id:
        raise ValueError("Decision template benchmark_id mismatch.")
    if template.get("policy_id") != POLICY_ID:
        raise ValueError("Decision template policy_id mismatch.")
    if template.get("review_status") != "pending":
        raise ValueError("Decision template review_status must be pending.")
    provenance = _object(manifest.get("provenance"), "manifest.provenance")
    source_sha = _digest(
        provenance.get("source_fingerprint_sha256"),
        "manifest.provenance.source_fingerprint_sha256",
    )
    if template.get("source_fingerprint_sha256") != source_sha:
        raise ValueError("Decision template source fingerprint mismatch.")

    manifest_cases = _array(manifest.get("cases"), "manifest.cases")
    template_cases = _array(template.get("decisions"), "decision_template.decisions")
    totals = _object(manifest.get("totals"), "manifest.totals")
    if _integer(totals.get("cases"), "manifest.totals.cases") != len(manifest_cases):
        raise ValueError("Manifest case total mismatch.")
    if len(template_cases) != len(manifest_cases):
        raise ValueError("Decision template and manifest case counts differ.")
    manifest_ids = [_text(item.get("case_id"), "manifest case_id") for item in map(lambda x: _object(x, "manifest case"), manifest_cases)]
    template_ids = [_text(item.get("case_id"), "template case_id") for item in map(lambda x: _object(x, "template decision"), template_cases)]
    if len(set(manifest_ids)) != len(manifest_ids) or set(manifest_ids) != set(template_ids):
        raise ValueError("Manifest/template case coverage is not exact and unique.")
    for raw_case in manifest_cases:
        case = _object(raw_case, "manifest case")
        blind = _safe_relative_path(
            case.get("blind_visual_path"), "manifest case blind_visual_path"
        ).as_posix()
        if not blind.startswith("cases_blind/") or blind not in seen_artifacts:
            raise ValueError("Every manifest case must reference an inventoried blind image.")
    return root, manifest, template, _pack_fingerprint(manifest_bytes)


def _component_ids(entry: Mapping[str, Any]) -> list[str]:
    analysis = _object(entry.get("analysis"), "decision.analysis")
    components = _array(analysis.get("components"), "decision.analysis.components")
    ids = [_text(_object(row, "component").get("id"), "component.id") for row in components]
    if not ids or ids[0] != "c001" or len(ids) != len(set(ids)):
        raise ValueError("Component analysis must have unique ordered IDs starting at c001.")
    return ids


def _case_maps(
    manifest: Mapping[str, Any], template: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for raw in _array(manifest.get("cases"), "manifest.cases"):
        row = _object(raw, "manifest case")
        case_id = _text(row.get("case_id"), "manifest case_id")
        manifest_by_id[case_id] = row
    template_by_id: dict[str, dict[str, Any]] = {}
    for raw in _array(template.get("decisions"), "template.decisions"):
        row = _object(raw, "template decision")
        case_id = _text(row.get("case_id"), "template case_id")
        if row.get("source_mask_sha256") != row.get("analysis", {}).get(
            "source_mask_sha256"
        ):
            raise ValueError(f"Source mask hash mismatch inside case {case_id}.")
        _digest(row.get("source_mask_sha256"), f"{case_id}.source_mask_sha256")
        _component_ids(row)
        template_by_id[case_id] = row
    return manifest_by_id, template_by_id


def _balanced_stream_shards(
    cases: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_stream[_text(case.get("stream_id"), "case.stream_id")].append(case)
    streams = sorted(by_stream.items(), key=lambda item: (-len(item[1]), item[0]))
    shard_cases: list[list[Mapping[str, Any]]] = [[] for _ in SHARD_IDS]
    totals = [0] * len(SHARD_IDS)
    for _stream_id, stream_cases in streams:
        shard_index = min(range(len(SHARD_IDS)), key=lambda index: (totals[index], index))
        ordered = sorted(stream_cases, key=lambda row: _text(row.get("case_id"), "case_id"))
        shard_cases[shard_index].extend(ordered)
        totals[shard_index] += len(ordered)
    return shard_cases


def _allowed_evidence_by_case(
    manifest: Mapping[str, Any],
) -> dict[str, list[str]]:
    cases = [_object(raw, "manifest case") for raw in _array(manifest.get("cases"), "manifest.cases")]
    by_stream: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        stream_id = _text(case.get("stream_id"), "manifest case stream_id")
        by_stream[stream_id].add(_text(case.get("frame_id"), "manifest case frame_id"))
    result: dict[str, list[str]] = {}
    for case in cases:
        case_id = _text(case.get("case_id"), "manifest case_id")
        stream_id = _text(case.get("stream_id"), "manifest case stream_id")
        result[case_id] = sorted(by_stream[stream_id])
    return result


def _blind_case(
    template_case: Mapping[str, Any],
    *,
    blind_visual_path: str,
    allowed_evidence_frame_ids: Sequence[str],
) -> dict[str, Any]:
    analysis = _object(template_case.get("analysis"), "template case analysis")
    components = _array(analysis.get("components"), "analysis.components")
    secondary: list[dict[str, Any]] = []
    for raw in components[1:]:
        component = _object(raw, "analysis component")
        secondary.append(
            {
                "component_id": _text(component.get("id"), "component.id"),
                "bbox": copy.deepcopy(_object(component.get("bbox"), "component.bbox")),
                "binary_mask_sha256": _digest(
                    component.get("binary_mask_sha256"), "component.binary_mask_sha256"
                ),
            }
        )
    return {
        "case_id": template_case["case_id"],
        "role": template_case["role"],
        "stream_id": template_case["stream_id"],
        "source_sequence": template_case["source_sequence"],
        "frame_id": template_case["frame_id"],
        "frame_index": template_case["frame_index"],
        "source_label": template_case["source_label"],
        "source_mask_sha256": template_case["source_mask_sha256"],
        "blind_visual_path": blind_visual_path,
        "allowed_evidence_frame_ids": list(allowed_evidence_frame_ids),
        "secondary_components": secondary,
        "case_integrity_decision": {
            "decision": None,
            "confidence": None,
            "rationale_codes": [],
            "evidence_frame_ids": [],
            "notes": "",
        },
        "component_decisions": None,
        "case_notes": "",
    }


def prepare_assignments(*, review_pack: Path, output_root: Path) -> None:
    pack_root, manifest, template, fingerprint = _validate_pack(review_pack)
    manifest_by_id, template_by_id = _case_maps(manifest, template)
    allowed_by_case = _allowed_evidence_by_case(manifest)
    ordered_cases = [template_by_id[key] for key in sorted(template_by_id)]
    shards = _balanced_stream_shards(ordered_cases)

    def build(staging: Path) -> None:
        for pass_id in PASSES:
            for shard_id, cases in zip(SHARD_IDS, shards):
                assignment_cases: list[dict[str, Any]] = []
                for case in cases:
                    case_id = str(case["case_id"])
                    manifest_case = manifest_by_id[case_id]
                    source_visual = _resolve_pack_artifact(
                        pack_root,
                        manifest_case["blind_visual_path"],
                        f"{case_id}.blind_visual_path",
                    )
                    relative_visual = (
                        Path(pass_id)
                        / "evidence"
                        / str(case["stream_id"])
                        / source_visual.name
                    ).as_posix()
                    destination_visual = staging / relative_visual
                    _copy_sanitized_blind_visual(
                        source_visual, destination_visual, case
                    )
                    assignment_cases.append(
                        _blind_case(
                            case,
                            blind_visual_path=relative_visual,
                            allowed_evidence_frame_ids=allowed_by_case[case_id],
                        )
                    )
                payload = {
                    "schema_version": ASSIGNMENT_SCHEMA,
                    "review_pack_fingerprint_sha256": fingerprint,
                    "benchmark_id": template["benchmark_id"],
                    "policy_id": POLICY_ID,
                    "pass_id": pass_id,
                    "shard_id": shard_id,
                    "blind_to_automatic_proposal": True,
                    "reviewer_id": None,
                    "review_status": "pending",
                    "case_count": len(cases),
                    "cases": assignment_cases,
                }
                _write_json(staging / pass_id / f"{shard_id}.json", payload)
        totals = [len(shard) for shard in shards]
        summary = {
            "schema_version": COORDINATION_SUMMARY_SCHEMA,
            "stage": "assignments_prepared",
            "review_pack_fingerprint_sha256": fingerprint,
            "benchmark_id": template["benchmark_id"],
            "pass_count": 2,
            "shard_count_per_pass": 4,
            "case_count_per_pass": len(ordered_cases),
            "shard_case_counts": dict(zip(SHARD_IDS, totals)),
            "blind_to_automatic_proposal": True,
            "model_predictions_accessed": False,
            "metrics_computed": False,
            "physically_isolated_primary_package": True,
        }
        _write_json(staging / "summary.json", summary)
        records = _artifact_records(staging)
        _write_json(
            staging / "artifact_hashes.json",
            {
                "schema_version": ISOLATED_ARTIFACTS_SCHEMA,
                "self_excluded": "artifact_hashes.json",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )

    _atomic_publish(output_root, build)


def _validate_decision_row(
    raw: object,
    *,
    expected_component_id: str,
    context: str,
    final_only: bool = False,
    allowed_evidence_frame_ids: Iterable[str] = (),
) -> dict[str, Any]:
    row = _object(raw, context)
    if row.get("component_id") != expected_component_id:
        raise ValueError(f"{context} component_id coverage/order mismatch.")
    decision = row.get("decision")
    allowed = FINAL_DECISIONS if final_only else PRIMARY_DECISIONS
    if decision not in allowed:
        raise ValueError(f"{context} has an invalid decision.")
    confidence = row.get("confidence")
    if decision in FINAL_DECISIONS:
        if confidence not in FINAL_CONFIDENCES:
            raise ValueError(f"{context} KEEP/DROP confidence must be HIGH or MEDIUM.")
    elif confidence != "LOW":
        raise ValueError(f"{context} unresolved/integrity confidence must be LOW.")
    codes = _array(row.get("rationale_codes"), f"{context}.rationale_codes")
    if not codes or len(codes) != len(set(codes)):
        raise ValueError(f"{context}.rationale_codes must be non-empty and unique.")
    for code in codes:
        if code not in RATIONALE_CODES[str(decision)]:
            raise ValueError(f"{context} contains an invalid rubric rationale code.")
    evidence = _array(row.get("evidence_frame_ids"), f"{context}.evidence_frame_ids")
    allowed_evidence = set(allowed_evidence_frame_ids)
    normalized_evidence = [
        _text(value, f"{context}.evidence_frame_ids item") for value in evidence
    ]
    if not normalized_evidence or not set(normalized_evidence).issubset(allowed_evidence):
        raise ValueError(
            f"{context}.evidence_frame_ids must be a non-empty subset of allowed stream context."
        )
    if len(normalized_evidence) != len(set(normalized_evidence)):
        raise ValueError(f"{context}.evidence_frame_ids must be unique.")
    if not isinstance(row.get("notes"), str) or not row["notes"].strip():
        raise ValueError(f"{context}.notes must be a non-empty string.")
    return copy.deepcopy(row)


def _validate_integrity_decision(
    raw: object,
    *,
    context: str,
    allowed_evidence_frame_ids: Iterable[str],
) -> dict[str, Any]:
    row = _object(raw, context)
    decision = row.get("decision")
    if decision not in INTEGRITY_DECISIONS:
        raise ValueError(f"{context}.decision must be OK or INTEGRITY_ALERT.")
    confidence = row.get("confidence")
    codes = _array(row.get("rationale_codes"), f"{context}.rationale_codes")
    if decision == "OK":
        if confidence not in FINAL_CONFIDENCES or codes:
            raise ValueError(
                f"{context} OK requires HIGH|MEDIUM confidence and no integrity codes."
            )
    else:
        if confidence != "LOW" or not codes or not set(codes).issubset(INTEGRITY_CODES):
            raise ValueError(
                f"{context} INTEGRITY_ALERT requires LOW confidence and a valid integrity code."
            )
        if len(codes) != len(set(codes)):
            raise ValueError(f"{context}.rationale_codes must be unique.")
    evidence = _array(row.get("evidence_frame_ids"), f"{context}.evidence_frame_ids")
    normalized_evidence = [
        _text(value, f"{context}.evidence_frame_ids item") for value in evidence
    ]
    if (
        not normalized_evidence
        or len(normalized_evidence) != len(set(normalized_evidence))
        or not set(normalized_evidence).issubset(set(allowed_evidence_frame_ids))
    ):
        raise ValueError(
            f"{context}.evidence_frame_ids must be a non-empty unique subset of allowed stream context."
        )
    if not isinstance(row.get("notes"), str) or not row["notes"].strip():
        raise ValueError(f"{context}.notes must be a non-empty string.")
    return copy.deepcopy(row)


def _completed_assignments(
    assignments_root: Path,
    *,
    fingerprint: str,
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    template_by_id: Mapping[str, Mapping[str, Any]],
    allowed_by_case: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(assignments_root).resolve(strict=True)
    expected_relatives = {
        (Path(pass_id) / f"{shard_id}.json").as_posix()
        for pass_id in PASSES
        for shard_id in SHARD_IDS
    }
    expected_relatives.update({"summary.json", "artifact_hashes.json"})
    expected_visuals: dict[tuple[str, str], str] = {}
    for pass_id in PASSES:
        for case_id, manifest_case in manifest_by_id.items():
            source_name = _safe_relative_path(
                manifest_case.get("blind_visual_path"), f"{case_id}.blind_visual_path"
            ).name
            relative = (
                Path(pass_id)
                / "evidence"
                / str(template_by_id[case_id]["stream_id"])
                / source_name
            ).as_posix()
            expected_relatives.add(relative)
            expected_visuals[(pass_id, case_id)] = relative
    actual_relatives = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    for relative in actual_relatives:
        _forbid_output_artifact(relative, "assignments root")
    if actual_relatives != expected_relatives:
        raise ValueError(
            "Assignments root inventory mismatch; "
            f"missing={sorted(expected_relatives - actual_relatives)}, "
            f"unexpected={sorted(actual_relatives - expected_relatives)}."
        )
    by_pass: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in PASSES}
    reviewer_by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for pass_id in PASSES:
        for shard_id in SHARD_IDS:
            path = root / pass_id / f"{shard_id}.json"
            assignment, _ = _load_json(path, f"{pass_id}/{shard_id} assignment")
            expected_assignment_fields = {
                "schema_version",
                "review_pack_fingerprint_sha256",
                "benchmark_id",
                "policy_id",
                "pass_id",
                "shard_id",
                "blind_to_automatic_proposal",
                "reviewer_id",
                "review_status",
                "case_count",
                "cases",
            }
            if set(assignment) != expected_assignment_fields:
                raise ValueError(f"Assignment fields are not exact in {path}.")
            if assignment.get("schema_version") != ASSIGNMENT_SCHEMA:
                raise ValueError(f"Invalid assignment schema in {path}.")
            if assignment.get("review_pack_fingerprint_sha256") != fingerprint:
                raise ValueError(f"Pack fingerprint mismatch in {path}.")
            if assignment.get("pass_id") != pass_id or assignment.get("shard_id") != shard_id:
                raise ValueError(f"Assignment identity mismatch in {path}.")
            if assignment.get("blind_to_automatic_proposal") is not True:
                raise ValueError(f"Assignment is not blind in {path}.")
            if assignment.get("review_status") != "complete":
                raise ValueError(f"Assignment review_status must be complete: {path}.")
            reviewer_id = _reviewer_id(
                assignment.get("reviewer_id"), f"{path}.reviewer_id"
            )
            cases = _array(assignment.get("cases"), f"{path}.cases")
            if assignment.get("case_count") != len(cases):
                raise ValueError(f"Assignment case_count mismatch in {path}.")
            for raw_case in cases:
                case = _object(raw_case, f"{path} case")
                expected_case_fields = {
                    "case_id",
                    "role",
                    "stream_id",
                    "source_sequence",
                    "frame_id",
                    "frame_index",
                    "source_label",
                    "source_mask_sha256",
                    "blind_visual_path",
                    "allowed_evidence_frame_ids",
                    "secondary_components",
                    "case_integrity_decision",
                    "component_decisions",
                    "case_notes",
                }
                if set(case) != expected_case_fields:
                    raise ValueError(f"Blind assignment case fields are not exact in {path}.")
                case_id = _text(case.get("case_id"), "assignment case_id")
                if case_id not in template_by_id:
                    raise ValueError(f"Unexpected assignment case: {case_id}.")
                if case_id in by_pass[pass_id]:
                    raise ValueError(f"Duplicate case in {pass_id}: {case_id}.")
                template_case = template_by_id[case_id]
                for field in (
                    "role",
                    "stream_id",
                    "source_sequence",
                    "frame_id",
                    "frame_index",
                    "source_label",
                ):
                    if case.get(field) != template_case.get(field):
                        raise ValueError(f"Immutable blind case field mismatch for {case_id}: {field}.")
                if case.get("source_mask_sha256") != template_case.get("source_mask_sha256"):
                    raise ValueError(f"Source mask hash mismatch for {case_id}.")
                expected_visual = expected_visuals[(pass_id, case_id)]
                if case.get("blind_visual_path") != expected_visual:
                    raise ValueError(f"Blind visual path mismatch for {case_id}.")
                if case.get("allowed_evidence_frame_ids") != list(
                    allowed_by_case[case_id]
                ):
                    raise ValueError(f"Allowed evidence frames mismatch for {case_id}.")
                blind_components = _array(
                    case.get("secondary_components"),
                    f"{case_id}.secondary_components",
                )
                for component in blind_components:
                    fields = set(_object(component, "secondary component"))
                    if fields != {"component_id", "bbox", "binary_mask_sha256"}:
                        raise ValueError(
                            f"Blind component evidence leaks forbidden fields for {case_id}."
                        )
                secondary_ids = _component_ids(template_case)[1:]
                source_components = _array(
                    _object(template_case.get("analysis"), "template analysis").get(
                        "components"
                    ),
                    "template components",
                )[1:]
                expected_blind_components = [
                    {
                        "component_id": _text(component.get("id"), "component.id"),
                        "bbox": copy.deepcopy(_object(component.get("bbox"), "component.bbox")),
                        "binary_mask_sha256": _digest(
                            component.get("binary_mask_sha256"),
                            "component.binary_mask_sha256",
                        ),
                    }
                    for component in map(lambda item: _object(item, "component"), source_components)
                ]
                if blind_components != expected_blind_components:
                    raise ValueError(f"Blind component evidence mismatch for {case_id}.")
                integrity = _validate_integrity_decision(
                    case.get("case_integrity_decision"),
                    context=f"{case_id}.case_integrity_decision",
                    allowed_evidence_frame_ids=allowed_by_case[case_id],
                )
                decisions = _array(
                    case.get("component_decisions"),
                    f"{case_id}.component_decisions",
                )
                if len(decisions) != len(secondary_ids):
                    raise ValueError(f"Incomplete component coverage for {case_id}.")
                rows = [
                    _validate_decision_row(
                        raw,
                        expected_component_id=component_id,
                        context=f"{case_id}.{component_id}",
                        allowed_evidence_frame_ids=allowed_by_case[case_id],
                    )
                    for raw, component_id in zip(decisions, secondary_ids)
                ]
                if not isinstance(case.get("case_notes"), str):
                    raise ValueError(f"{case_id}.case_notes must be a string.")
                review = {
                    "reviewer_id": reviewer_id,
                    "pass_id": pass_id,
                    "blind_to_automatic_proposal": True,
                    "case_integrity_decision": integrity,
                    "component_decisions": rows,
                    "case_notes": case["case_notes"],
                }
                by_pass[pass_id][case_id] = review
                reviewer_by_case[case_id][pass_id] = reviewer_id
    expected_ids = set(template_by_id)
    for pass_id in PASSES:
        if set(by_pass[pass_id]) != expected_ids:
            missing = sorted(expected_ids - set(by_pass[pass_id]))
            extras = sorted(set(by_pass[pass_id]) - expected_ids)
            raise ValueError(
                f"{pass_id} case coverage mismatch; missing={missing}, extras={extras}."
            )
    for case_id, reviewers in reviewer_by_case.items():
        if reviewers["pass_a"].casefold() == reviewers["pass_b"].casefold():
            raise ValueError(f"Primary reviewers must be distinct for case {case_id}.")
    return by_pass


def _agreed_final_row(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(first["decision"])
    confidence = (
        "HIGH"
        if first.get("confidence") == second.get("confidence") == "HIGH"
        else "MEDIUM"
    )
    rationale = sorted(set(first["rationale_codes"]) | set(second["rationale_codes"]))
    evidence = sorted(set(first["evidence_frame_ids"]) | set(second["evidence_frame_ids"]))
    notes = " | ".join(
        text for text in (str(first.get("notes", "")), str(second.get("notes", ""))) if text
    )
    return {
        "component_id": first["component_id"],
        "decision": decision,
        "rationale_codes": rationale,
        "confidence": confidence,
        "evidence_frame_ids": evidence,
        "notes": notes,
    }


def _agreed_integrity_row(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    if first.get("decision") != "OK" or second.get("decision") != "OK":
        raise ValueError("Only two OK integrity decisions can be merged automatically.")
    return {
        "decision": "OK",
        "confidence": (
            "HIGH"
            if first.get("confidence") == second.get("confidence") == "HIGH"
            else "MEDIUM"
        ),
        "rationale_codes": [],
        "evidence_frame_ids": sorted(
            set(first["evidence_frame_ids"]) | set(second["evidence_frame_ids"])
        ),
        "notes": " | ".join(
            text
            for text in (str(first.get("notes", "")), str(second.get("notes", "")))
            if text
        ),
    }


def merge_primary_reviews(
    *, review_pack: Path, assignments_root: Path, output_root: Path
) -> None:
    _, manifest, template, fingerprint = _validate_pack(review_pack)
    manifest_by_id, template_by_id = _case_maps(manifest, template)
    allowed_by_case = _allowed_evidence_by_case(manifest)
    reviews = _completed_assignments(
        assignments_root,
        fingerprint=fingerprint,
        manifest_by_id=manifest_by_id,
        template_by_id=template_by_id,
        allowed_by_case=allowed_by_case,
    )
    merged_cases: list[dict[str, Any]] = []
    adjudication_cases: list[dict[str, Any]] = []
    disputed_component_total = 0
    for case_id in sorted(template_by_id):
        template_case = template_by_id[case_id]
        review_a = reviews["pass_a"][case_id]
        review_b = reviews["pass_b"][case_id]
        rows_a = {row["component_id"]: row for row in review_a["component_decisions"]}
        rows_b = {row["component_id"]: row for row in review_b["component_decisions"]}
        integrity_a = review_a["case_integrity_decision"]
        integrity_b = review_b["case_integrity_decision"]
        integrity_agreed = (
            _agreed_integrity_row(integrity_a, integrity_b)
            if integrity_a["decision"] == integrity_b["decision"] == "OK"
            else None
        )
        integrity_disputed = integrity_agreed is None
        agreed: list[dict[str, Any]] = []
        disputed: list[str] = []
        for component_id in _component_ids(template_case)[1:]:
            first = rows_a[component_id]
            second = rows_b[component_id]
            if (
                first["decision"] == second["decision"]
                and first["decision"] in FINAL_DECISIONS
            ):
                agreed.append(_agreed_final_row(first, second))
            else:
                disputed.append(component_id)
        merged_cases.append(
            {
                "case_id": case_id,
                "source_mask_sha256": template_case["source_mask_sha256"],
                "reviews": [review_a, review_b],
                "agreed_case_integrity_resolution": integrity_agreed,
                "case_integrity_disputed": integrity_disputed,
                "agreed_final_component_decisions": agreed,
                "disputed_component_ids": disputed,
            }
        )
        if disputed or integrity_disputed:
            disputed_component_total += len(disputed)
            components = {
                row["id"]: row
                for row in _array(template_case["analysis"]["components"], "components")
            }
            adjudication_cases.append(
                {
                    "case_id": case_id,
                    "role": template_case["role"],
                    "stream_id": template_case["stream_id"],
                    "frame_id": template_case["frame_id"],
                    "frame_index": template_case["frame_index"],
                    "source_label": template_case["source_label"],
                    "source_mask_sha256": template_case["source_mask_sha256"],
                    "allowed_evidence_frame_ids": list(allowed_by_case[case_id]),
                    "adjudication_visual_path": manifest_by_id[case_id][
                        "adjudication_visual_path"
                    ],
                    "automatic_retained_component_ids": copy.deepcopy(
                        template_case["automatic_retained_component_ids"]
                    ),
                    "disputed_components": [
                        {
                            "component_id": component_id,
                            "area": components[component_id]["area"],
                            "bbox": copy.deepcopy(components[component_id]["bbox"]),
                        }
                        for component_id in disputed
                    ],
                    "primary_reviews": [review_a, review_b],
                    "reviewer_id": None,
                    "reviewer_identity_exception": None,
                    "review_status": "pending",
                    "case_integrity_resolution": None,
                    "component_decisions": None,
                    "case_notes": "",
                }
            )
    merge_payload = {
        "schema_version": PRIMARY_MERGE_SCHEMA,
        "review_pack_fingerprint_sha256": fingerprint,
        "benchmark_id": template["benchmark_id"],
        "policy_id": POLICY_ID,
        "review_status": "primary_complete",
        "case_count": len(merged_cases),
        "disputed_case_count": len(adjudication_cases),
        "disputed_component_count": disputed_component_total,
        "cases": merged_cases,
    }
    adjudication_payload = {
        "schema_version": ADJUDICATION_SCHEMA,
        "review_pack_fingerprint_sha256": fingerprint,
        "benchmark_id": template["benchmark_id"],
        "policy_id": POLICY_ID,
        "blind_to_automatic_proposal": False,
        "review_status": "pending" if adjudication_cases else "complete",
        "case_count": len(adjudication_cases),
        "cases": adjudication_cases,
    }

    def build(staging: Path) -> None:
        _write_json(staging / "primary_merge.json", merge_payload)
        _write_json(staging / "adjudication_template.json", adjudication_payload)
        _write_json(
            staging / "summary.json",
            {
                "schema_version": COORDINATION_SUMMARY_SCHEMA,
                "stage": "primary_reviews_merged",
                "review_pack_fingerprint_sha256": fingerprint,
                "case_count": len(merged_cases),
                "disputed_case_count": len(adjudication_cases),
                "disputed_component_count": disputed_component_total,
                "automatic_proposal_revealed_only_after_both_passes": True,
                "model_predictions_accessed": False,
                "metrics_computed": False,
            },
        )
        _write_json(staging / "artifact_hashes.json", _artifact_records(staging))

    _atomic_publish(output_root, build)


def _recompute_primary_resolution(
    row: Mapping[str, Any],
    *,
    template_case: Mapping[str, Any],
    allowed_evidence_frame_ids: Sequence[str],
) -> dict[str, Any]:
    case_id = _text(row.get("case_id"), "primary merge case_id")
    if row.get("source_mask_sha256") != template_case.get("source_mask_sha256"):
        raise ValueError(f"Primary merge source mask mismatch for {case_id}.")
    raw_reviews = _array(row.get("reviews"), f"{case_id}.reviews")
    if len(raw_reviews) != 2:
        raise ValueError(f"{case_id} must preserve exactly two primary reviews.")
    reviews_by_pass: dict[str, dict[str, Any]] = {}
    component_ids = _component_ids(template_case)[1:]
    for raw_review in raw_reviews:
        review = _object(raw_review, f"{case_id} primary review")
        pass_id = review.get("pass_id")
        if pass_id not in PASSES or pass_id in reviews_by_pass:
            raise ValueError(f"{case_id} primary reviews must cover pass_a and pass_b exactly.")
        reviewer = _reviewer_id(review.get("reviewer_id"), f"{case_id}.reviewer_id")
        if review.get("blind_to_automatic_proposal") is not True:
            raise ValueError(f"{case_id} primary review is not blind.")
        integrity = _validate_integrity_decision(
            review.get("case_integrity_decision"),
            context=f"{case_id}.{pass_id}.case_integrity_decision",
            allowed_evidence_frame_ids=allowed_evidence_frame_ids,
        )
        raw_decisions = _array(
            review.get("component_decisions"), f"{case_id}.{pass_id}.component_decisions"
        )
        if len(raw_decisions) != len(component_ids):
            raise ValueError(f"{case_id} primary component coverage mismatch.")
        decisions = [
            _validate_decision_row(
                raw_decision,
                expected_component_id=component_id,
                context=f"{case_id}.{pass_id}.{component_id}",
                allowed_evidence_frame_ids=allowed_evidence_frame_ids,
            )
            for raw_decision, component_id in zip(raw_decisions, component_ids)
        ]
        if not isinstance(review.get("case_notes"), str):
            raise ValueError(f"{case_id}.{pass_id}.case_notes must be a string.")
        reviews_by_pass[str(pass_id)] = {
            "reviewer_id": reviewer,
            "pass_id": pass_id,
            "blind_to_automatic_proposal": True,
            "case_integrity_decision": integrity,
            "component_decisions": decisions,
            "case_notes": review["case_notes"],
        }
    first = reviews_by_pass["pass_a"]
    second = reviews_by_pass["pass_b"]
    if first["reviewer_id"].casefold() == second["reviewer_id"].casefold():
        raise ValueError(f"{case_id} primary reviewer IDs are not distinct.")
    first_rows = {item["component_id"]: item for item in first["component_decisions"]}
    second_rows = {item["component_id"]: item for item in second["component_decisions"]}
    agreed: list[dict[str, Any]] = []
    disputed: list[str] = []
    for component_id in component_ids:
        row_a = first_rows[component_id]
        row_b = second_rows[component_id]
        if row_a["decision"] == row_b["decision"] and row_a["decision"] in FINAL_DECISIONS:
            agreed.append(_agreed_final_row(row_a, row_b))
        else:
            disputed.append(component_id)
    integrity_a = first["case_integrity_decision"]
    integrity_b = second["case_integrity_decision"]
    agreed_integrity = (
        _agreed_integrity_row(integrity_a, integrity_b)
        if integrity_a["decision"] == integrity_b["decision"] == "OK"
        else None
    )
    return {
        "case_id": case_id,
        "source_mask_sha256": template_case["source_mask_sha256"],
        "reviews": [first, second],
        "agreed_case_integrity_resolution": agreed_integrity,
        "case_integrity_disputed": agreed_integrity is None,
        "agreed_final_component_decisions": agreed,
        "disputed_component_ids": disputed,
    }


def _load_primary_merge(
    path: Path,
    *,
    fingerprint: str,
    template_by_id: Mapping[str, Mapping[str, Any]],
    allowed_by_case: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload, _ = _load_json(path, "primary merge")
    if payload.get("schema_version") != PRIMARY_MERGE_SCHEMA:
        raise ValueError("Unsupported primary merge schema.")
    if payload.get("review_pack_fingerprint_sha256") != fingerprint:
        raise ValueError("Primary merge pack fingerprint mismatch.")
    if payload.get("review_status") != "primary_complete":
        raise ValueError("Primary merge review_status must be primary_complete.")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in _array(payload.get("cases"), "primary_merge.cases"):
        row = _object(raw, "primary merge case")
        case_id = _text(row.get("case_id"), "primary merge case_id")
        if case_id in by_id:
            raise ValueError(f"Duplicate primary merge case: {case_id}.")
        if case_id not in template_by_id:
            raise ValueError(f"Unexpected primary merge case: {case_id}.")
        recomputed = _recompute_primary_resolution(
            row,
            template_case=template_by_id[case_id],
            allowed_evidence_frame_ids=allowed_by_case[case_id],
        )
        for field in (
            "agreed_case_integrity_resolution",
            "case_integrity_disputed",
            "agreed_final_component_decisions",
            "disputed_component_ids",
        ):
            if row.get(field) != recomputed[field]:
                raise ValueError(
                    f"Primary merge derived field was tampered for {case_id}: {field}."
                )
        by_id[case_id] = recomputed
    if set(by_id) != set(template_by_id):
        raise ValueError("Primary merge case coverage mismatch.")
    disputed_cases = sum(
        bool(row["disputed_component_ids"] or row["case_integrity_disputed"])
        for row in by_id.values()
    )
    disputed_components = sum(len(row["disputed_component_ids"]) for row in by_id.values())
    if payload.get("case_count") != len(by_id):
        raise ValueError("Primary merge case_count mismatch.")
    if payload.get("disputed_case_count") != disputed_cases:
        raise ValueError("Primary merge disputed_case_count was tampered.")
    if payload.get("disputed_component_count") != disputed_components:
        raise ValueError("Primary merge disputed_component_count was tampered.")
    return payload, by_id


def _completed_adjudication(
    path: Path,
    *,
    fingerprint: str,
    primary_by_id: Mapping[str, Mapping[str, Any]],
    allowed_by_case: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    payload, _ = _load_json(path, "completed adjudication")
    if payload.get("schema_version") != ADJUDICATION_SCHEMA:
        raise ValueError("Unsupported adjudication schema.")
    if payload.get("review_pack_fingerprint_sha256") != fingerprint:
        raise ValueError("Adjudication pack fingerprint mismatch.")
    expected_cases = {
        case_id: list(row["disputed_component_ids"])
        for case_id, row in primary_by_id.items()
        if row["disputed_component_ids"] or row["case_integrity_disputed"]
    }
    payload_status = payload.get("review_status")
    if payload_status not in {"complete", "author_decision_required"}:
        raise ValueError(
            "Adjudication review_status must be complete or author_decision_required."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for raw in _array(payload.get("cases"), "adjudication.cases"):
        row = _object(raw, "adjudication case")
        case_id = _text(row.get("case_id"), "adjudication case_id")
        if case_id not in expected_cases or case_id in by_id:
            raise ValueError(f"Unexpected or duplicate adjudication case: {case_id}.")
        case_status = row.get("review_status")
        if case_status not in {"complete", "author_decision_required"}:
            raise ValueError(f"Adjudication case {case_id} has an invalid status.")
        reviewer_id = _reviewer_id(row.get("reviewer_id"), f"{case_id}.reviewer_id")
        primary_reviewers = {
            str(review["reviewer_id"]).casefold()
            for review in primary_by_id[case_id]["reviews"]
        }
        identity_exception = row.get("reviewer_identity_exception")
        if identity_exception not in (None, ""):
            raise ValueError(
                f"Adjudicator identity exceptions are not permitted for {case_id}."
            )
        if reviewer_id.casefold() in primary_reviewers:
            raise ValueError(
                f"Adjudicator for {case_id} must be distinct from both primary reviewers."
            )
        decisions = _array(row.get("component_decisions"), f"{case_id}.component_decisions")
        expected_components = expected_cases[case_id]
        if len(decisions) != len(expected_components):
            raise ValueError(f"Adjudication component coverage mismatch for {case_id}.")
        validated = [
            _validate_decision_row(
                raw_decision,
                expected_component_id=component_id,
                context=f"adjudication {case_id}.{component_id}",
                final_only=case_status == "complete",
                allowed_evidence_frame_ids=allowed_by_case[case_id],
            )
            for raw_decision, component_id in zip(decisions, expected_components)
        ]
        integrity = _validate_integrity_decision(
            row.get("case_integrity_resolution"),
            context=f"adjudication {case_id}.case_integrity_resolution",
            allowed_evidence_frame_ids=allowed_by_case[case_id],
        )
        needs_author = integrity["decision"] != "OK" or any(
            item["decision"] not in FINAL_DECISIONS for item in validated
        )
        if case_status == "complete" and needs_author:
            raise ValueError(
                f"Adjudication case {case_id} cannot be complete with unresolved integrity/components."
            )
        if case_status == "author_decision_required" and not needs_author:
            raise ValueError(
                f"Adjudication case {case_id} claims author attention without an unresolved decision."
            )
        if not isinstance(row.get("case_notes"), str) or not row["case_notes"].strip():
            raise ValueError(f"{case_id}.case_notes must be a string.")
        normalized = copy.deepcopy(row)
        normalized["reviewer_id"] = reviewer_id
        normalized["case_integrity_resolution"] = integrity
        normalized["component_decisions"] = validated
        by_id[case_id] = normalized
    if set(by_id) != set(expected_cases):
        raise ValueError("Adjudication case coverage is not exact.")
    expected_payload_status = (
        "author_decision_required"
        if any(row["review_status"] == "author_decision_required" for row in by_id.values())
        else "complete"
    )
    if payload_status != expected_payload_status:
        raise ValueError("Adjudication top-level review_status does not match case statuses.")
    return by_id


def _bbox_edges(bbox: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    return x, y, x + width, y + height


def _bbox_union(components: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    edges = [_bbox_edges(_object(component.get("bbox"), "component.bbox")) for component in components]
    left = min(edge[0] for edge in edges)
    top = min(edge[1] for edge in edges)
    right = max(edge[2] for edge in edges)
    bottom = max(edge[3] for edge in edges)
    return {
        "x": int(left),
        "y": int(top),
        "width": int(right - left),
        "height": int(bottom - top),
    }


def _bbox_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    a_left, a_top, a_right, a_bottom = _bbox_edges(first)
    b_left, b_top, b_right, b_bottom = _bbox_edges(second)
    width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    height = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    intersection = width * height
    area_a = max(0.0, a_right - a_left) * max(0.0, a_bottom - a_top)
    area_b = max(0.0, b_right - b_left) * max(0.0, b_bottom - b_top)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _edge_move_large(
    raw_bbox: Mapping[str, Any], final_bbox: Mapping[str, Any], mask_shape: Sequence[Any]
) -> bool:
    if len(mask_shape) != 2:
        raise ValueError("analysis.mask_shape must contain height and width.")
    frame_height = float(_integer(mask_shape[0], "mask height"))
    frame_width = float(_integer(mask_shape[1], "mask width"))
    raw = _bbox_edges(raw_bbox)
    final = _bbox_edges(final_bbox)
    return (
        abs(raw[0] - final[0]) > 0.1 * frame_width
        or abs(raw[2] - final[2]) > 0.1 * frame_width
        or abs(raw[1] - final[1]) > 0.1 * frame_height
        or abs(raw[3] - final[3]) > 0.1 * frame_height
    )


def finalize_decisions(
    *,
    review_pack: Path,
    primary_merge: Path,
    adjudication: Path,
    output_root: Path,
) -> None:
    _, manifest, template, fingerprint = _validate_pack(review_pack)
    manifest_by_id, template_by_id = _case_maps(manifest, template)
    allowed_by_case = _allowed_evidence_by_case(manifest)
    _, primary_by_id = _load_primary_merge(
        primary_merge,
        fingerprint=fingerprint,
        template_by_id=template_by_id,
        allowed_by_case=allowed_by_case,
    )
    adjudication_by_id = _completed_adjudication(
        adjudication,
        fingerprint=fingerprint,
        primary_by_id=primary_by_id,
        allowed_by_case=allowed_by_case,
    )
    author_required = [
        case_id
        for case_id, row in adjudication_by_id.items()
        if row["review_status"] == "author_decision_required"
    ]
    if author_required:
        queue_cases = []
        for case_id in sorted(author_required):
            primary = primary_by_id[case_id]
            adjudication_record = adjudication_by_id[case_id]
            queue_cases.append(
                {
                    "case_id": case_id,
                    "role": template_by_id[case_id]["role"],
                    "stream_id": template_by_id[case_id]["stream_id"],
                    "frame_id": template_by_id[case_id]["frame_id"],
                    "source_label": template_by_id[case_id]["source_label"],
                    "allowed_evidence_frame_ids": list(allowed_by_case[case_id]),
                    "blind_visual_path": manifest_by_id[case_id]["blind_visual_path"],
                    "adjudication_visual_path": manifest_by_id[case_id][
                        "adjudication_visual_path"
                    ],
                    "primary_reviews": copy.deepcopy(primary["reviews"]),
                    "adjudication": copy.deepcopy(adjudication_record),
                    "unresolved_component_ids": [
                        row["component_id"]
                        for row in adjudication_record["component_decisions"]
                        if row["decision"] not in FINAL_DECISIONS
                    ],
                    "case_integrity_requires_author": (
                        adjudication_record["case_integrity_resolution"]["decision"]
                        != "OK"
                    ),
                }
            )
        queue = {
            "schema_version": AUTHOR_DECISION_SCHEMA,
            "review_pack_fingerprint_sha256": fingerprint,
            "benchmark_id": template["benchmark_id"],
            "status": "author_decision_required",
            "case_count": len(queue_cases),
            "cases": queue_cases,
        }

        def build_author_queue(staging: Path) -> None:
            _write_json(staging / "author_decision_queue.json", queue)
            _write_json(
                staging / "summary.json",
                {
                    "schema_version": COORDINATION_SUMMARY_SCHEMA,
                    "stage": "author_decision_required",
                    "review_pack_fingerprint_sha256": fingerprint,
                    "review_status": "author_decision_required",
                    "author_decision_count": len(queue_cases),
                    "complete_component_decisions_emitted": False,
                    "model_predictions_accessed": False,
                    "metrics_computed": False,
                },
            )
            _write_json(staging / "artifact_hashes.json", _artifact_records(staging))

        _atomic_publish(output_root, build_author_queue)
        return
    final_entries: list[dict[str, Any]] = []
    preview: list[dict[str, Any]] = []
    for case_id in sorted(template_by_id):
        source = copy.deepcopy(template_by_id[case_id])
        primary = primary_by_id[case_id]
        component_order = _component_ids(source)
        agreed = {
            row["component_id"]: copy.deepcopy(row)
            for row in _array(
                primary.get("agreed_final_component_decisions"),
                f"{case_id}.agreed_final_component_decisions",
            )
        }
        adjudication_record = adjudication_by_id.get(case_id)
        adjudicated = (
            {
                row["component_id"]: copy.deepcopy(row)
                for row in adjudication_record["component_decisions"]
            }
            if adjudication_record is not None
            else {}
        )
        final_rows: list[dict[str, Any]] = []
        for component_id in component_order[1:]:
            if component_id in agreed:
                final_rows.append(agreed[component_id])
            elif component_id in adjudicated:
                final_rows.append(adjudicated[component_id])
            else:
                raise ValueError(f"No final decision for {case_id}.{component_id}.")
        integrity_resolution = (
            copy.deepcopy(adjudication_record["case_integrity_resolution"])
            if adjudication_record is not None
            else copy.deepcopy(primary["agreed_case_integrity_resolution"])
        )
        if integrity_resolution is None or integrity_resolution["decision"] != "OK":
            raise ValueError(f"Case integrity is not resolved for {case_id}.")
        retained = ["c001"] + [
            row["component_id"] for row in final_rows if row["decision"] == "KEEP"
        ]
        component_by_id = {
            row["id"]: row for row in _array(source["analysis"]["components"], "components")
        }
        final_bbox = _bbox_union([component_by_id[item] for item in retained])
        raw_bbox = _object(source["analysis"].get("raw_bbox"), "analysis.raw_bbox")
        iou = _bbox_iou(raw_bbox, final_bbox)
        automatic = list(source["automatic_retained_component_ids"])
        reasons: list[str] = []
        if adjudication_record is not None:
            reasons.append("ADJUDICATED")
        if iou < 0.75:
            reasons.append("RAW_FINAL_BBOX_IOU_LT_0_75")
        if retained != automatic:
            reasons.append("AUTOMATIC_PROPOSAL_OVERRIDDEN")
        if _edge_move_large(raw_bbox, final_bbox, source["analysis"]["mask_shape"]):
            reasons.append("BBOX_EDGE_MOVE_GT_10_PERCENT_FRAME")
        final_confidence = (
            "MEDIUM" if any(row["confidence"] == "MEDIUM" for row in final_rows) else "HIGH"
        )
        source["review_status"] = "complete"
        source["reviews"] = copy.deepcopy(primary["reviews"])
        source["case_integrity_resolution"] = integrity_resolution
        source["adjudication"] = (
            {
                "reviewer_id": adjudication_record["reviewer_id"],
                "reviewer_identity_exception": adjudication_record.get(
                    "reviewer_identity_exception"
                ),
                "component_decisions": copy.deepcopy(
                    adjudication_record["component_decisions"]
                ),
                "case_integrity_resolution": copy.deepcopy(
                    adjudication_record["case_integrity_resolution"]
                ),
                "review_status": adjudication_record["review_status"],
                "case_notes": adjudication_record["case_notes"],
            }
            if adjudication_record is not None
            else None
        )
        source["final_component_decisions"] = final_rows
        source["final_retained_component_ids"] = retained
        source["final_confidence"] = final_confidence
        source["resolved_by"] = (
            "adjudication" if adjudication_record is not None else "primary_agreement"
        )
        source["author_attention"] = bool(reasons)
        source["notes"] = (
            adjudication_record["case_notes"] if adjudication_record is not None else ""
        )
        final_entries.append(source)
        if reasons:
            preview.append(
                {
                    "case_id": case_id,
                    "role": source["role"],
                    "stream_id": source["stream_id"],
                    "frame_id": source["frame_id"],
                    "source_label": source["source_label"],
                    "reasons": reasons,
                    "raw_bbox": copy.deepcopy(raw_bbox),
                    "final_bbox": final_bbox,
                    "raw_final_bbox_iou": iou,
                    "blind_visual_path": manifest_by_id[case_id]["blind_visual_path"],
                    "adjudication_visual_path": manifest_by_id[case_id][
                        "adjudication_visual_path"
                    ],
                }
            )
    ledger = copy.deepcopy(template)
    ledger["review_status"] = "complete"
    ledger["decisions"] = final_entries
    author_queue = {
        "schema_version": AUTHOR_PREVIEW_SCHEMA,
        "review_pack_fingerprint_sha256": fingerprint,
        "benchmark_id": template["benchmark_id"],
        "status": "ready",
        "case_count": len(preview),
        "cases": preview,
    }

    def build(staging: Path) -> None:
        ledger_path = staging / "component_decisions.json"
        _write_json(ledger_path, ledger)
        _write_json(staging / "author_preview_queue.json", author_queue)
        _write_json(
            staging / "summary.json",
            {
                "schema_version": COORDINATION_SUMMARY_SCHEMA,
                "stage": "decisions_finalized",
                "review_pack_fingerprint_sha256": fingerprint,
                "decision_count": len(final_entries),
                "author_preview_count": len(preview),
                "component_decisions_sha256": _sha256_path(ledger_path),
                "review_status": "complete",
                "model_predictions_accessed": False,
                "metrics_computed": False,
            },
        )
        _write_json(staging / "artifact_hashes.json", _artifact_records(staging))

    _atomic_publish(output_root, build)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare two blind assignment passes.")
    prepare.add_argument("--review-pack", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)

    merge = subparsers.add_parser(
        "merge-primary", help="Merge completed blind assignments."
    )
    merge.add_argument("--review-pack", type=Path, required=True)
    merge.add_argument("--assignments-root", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="Finalize the complete component decisions ledger."
    )
    finalize.add_argument("--review-pack", type=Path, required=True)
    finalize.add_argument("--primary-merge", type=Path, required=True)
    finalize.add_argument("--adjudication", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_assignments(review_pack=args.review_pack, output_root=args.output_root)
    elif args.command == "merge-primary":
        merge_primary_reviews(
            review_pack=args.review_pack,
            assignments_root=args.assignments_root,
            output_root=args.output_root,
        )
    else:
        finalize_decisions(
            review_pack=args.review_pack,
            primary_merge=args.primary_merge,
            adjudication=args.adjudication,
            output_root=args.output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
