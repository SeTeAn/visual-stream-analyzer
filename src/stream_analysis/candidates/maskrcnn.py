"""Verified local Torchvision Mask R-CNN candidate-extraction provider."""

from __future__ import annotations

import gc
import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from ..config import StageConfig, semantic_config_digest
from ..contracts import (
    BBox,
    CandidateExtractionResult,
    CandidateRecord,
    FrameCandidateDiagnostics,
    GeometryFeatureMetadata,
    MaskReference,
    ProducerProvenance,
    RecordEnvelope,
    StageContext,
    ValidityStatus,
    WarningRecord,
)
from ..input import DecodedStream
from .diagnostics import CandidateExtractionSnapshot, CandidateMaskRecord, candidate_mask_digest
from .extraction import (
    CANDIDATE_EXTRACTION_RESULT_SCHEMA_VERSION,
    CANDIDATE_GEOMETRY_SCHEMA_ID,
    CANDIDATE_RECORD_SCHEMA_VERSION,
    CANDIDATE_WARNING_SCHEMA_VERSION,
    FRAME_CANDIDATE_DIAGNOSTICS_SCHEMA_VERSION,
)


MASKRCNN_CONFIG_SCHEMA_ID = "stream_analysis.maskrcnn_candidate_extraction_config.v1"
MASKRCNN_MODEL_NAME = "maskrcnn_resnet50_fpn_v2"
MASKRCNN_MODEL_VERSION = "torchvision-coco-v1"
MASKRCNN_CANDIDATE_SOURCE = "torchvision_maskrcnn_coco_v1"

DevicePolicy = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True, slots=True)
class MaskRCNNCandidateExtractionConfig:
    expected_checkpoint_sha256: str
    expected_checkpoint_size_bytes: int
    expected_torchvision_version: str
    score_threshold: float = 0.10
    mask_threshold: float = 0.50
    class_agnostic_nms_iou: float | None = 0.50
    max_detections_per_frame: int = 100
    device_policy: DevicePolicy = "auto"
    config_version: str = "1.0"
    model_name: str = MASKRCNN_MODEL_NAME
    model_version: str = MASKRCNN_MODEL_VERSION
    candidate_source: str = MASKRCNN_CANDIDATE_SOURCE
    geometry_feature_schema_id: str = CANDIDATE_GEOMETRY_SCHEMA_ID

    def __post_init__(self) -> None:
        digest = self.expected_checkpoint_sha256.casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("expected_checkpoint_sha256 must contain 64 lowercase hex digits.")
        object.__setattr__(self, "expected_checkpoint_sha256", digest)
        if (
            isinstance(self.expected_checkpoint_size_bytes, bool)
            or not isinstance(self.expected_checkpoint_size_bytes, int)
            or self.expected_checkpoint_size_bytes <= 0
        ):
            raise ValueError("expected_checkpoint_size_bytes must be a positive integer.")
        for name in (
            "expected_torchvision_version",
            "config_version",
            "model_name",
            "model_version",
            "candidate_source",
            "geometry_feature_schema_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(char.isspace() for char in value):
                raise ValueError(f"{name} must be a non-empty token without whitespace.")
        if self.model_name != MASKRCNN_MODEL_NAME:
            raise ValueError(f"model_name must remain {MASKRCNN_MODEL_NAME!r} for v1.")
        for name in ("score_threshold", "mask_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        if self.class_agnostic_nms_iou is not None:
            value = self.class_agnostic_nms_iou
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("class_agnostic_nms_iou must be a real number or None.")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("class_agnostic_nms_iou must be finite and in [0, 1].")
            object.__setattr__(self, "class_agnostic_nms_iou", value)
        if (
            isinstance(self.max_detections_per_frame, bool)
            or not isinstance(self.max_detections_per_frame, int)
            or self.max_detections_per_frame <= 0
        ):
            raise ValueError("max_detections_per_frame must be a positive integer.")
        if self.device_policy not in {"cpu", "cuda", "auto"}:
            raise ValueError("device_policy must be cpu, cuda or auto.")

    @property
    def semantic_parameters(self) -> Mapping[str, object]:
        return {
            "provider_family": "torchvision_maskrcnn",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "semantic_classes": "COCO_closed_vocabulary",
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "expected_checkpoint_size_bytes": self.expected_checkpoint_size_bytes,
            "expected_torchvision_version": self.expected_torchvision_version,
            "score_threshold": self.score_threshold,
            "mask_threshold": self.mask_threshold,
            "class_agnostic_nms_iou": self.class_agnostic_nms_iou,
            "max_detections_per_frame": self.max_detections_per_frame,
            "bbox_policy": "model_bbox_integral_clipped",
            "mask_policy": "optional_thresholded_model_mask_inside_bbox",
            "candidate_source": self.candidate_source,
            "geometry_feature_schema_id": self.geometry_feature_schema_id,
        }

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            stage_id="candidate_extraction",
            schema_id=MASKRCNN_CONFIG_SCHEMA_ID,
            config_version=self.config_version,
            semantic_parameters=self.semantic_parameters,
            runtime_parameters={"device_policy": self.device_policy},
        )

    @property
    def config_digest(self) -> str:
        return semantic_config_digest(self.to_stage_config())

    @property
    def producer(self) -> ProducerProvenance:
        return ProducerProvenance(
            producer_stage="candidate_extraction",
            producer_version=self.model_version,
            config_version=self.config_version,
            config_digest=self.config_digest,
        )

    @property
    def source_identity_sha256(self) -> str:
        identity = f"torchvision:{self.expected_torchvision_version}:{self.model_name}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class MaskRCNNPrediction:
    bbox: BBox
    score: float
    label_id: int
    local_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BBox):
            raise TypeError("bbox must be BBox.")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and in [0, 1].")
        if isinstance(self.label_id, bool) or not isinstance(self.label_id, int) or self.label_id <= 0:
            raise ValueError("label_id must be a positive integer.")
        object.__setattr__(self, "score", score)
        if self.local_mask is not None:
            mask = np.asarray(self.local_mask)
            shape = (int(self.bbox.height), int(self.bbox.width))
            if mask.dtype != np.dtype(np.bool_) or mask.shape != shape:
                raise ValueError("local_mask must be bool and match bbox geometry.")
            mask = np.ascontiguousarray(mask.copy())
            mask.flags["WRITEABLE"] = False
            object.__setattr__(self, "local_mask", mask)


@dataclass(frozen=True, slots=True)
class MaskRCNNFrameOutput:
    predictions: tuple[MaskRCNNPrediction, ...]
    runtime_details: Mapping[str, object]

    def __post_init__(self) -> None:
        predictions = tuple(self.predictions)
        if any(not isinstance(item, MaskRCNNPrediction) for item in predictions):
            raise TypeError("predictions must contain MaskRCNNPrediction values.")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "runtime_details", MappingProxyType(dict(self.runtime_details)))


class MaskRCNNProviderProtocol(Protocol):
    config: MaskRCNNCandidateExtractionConfig
    resolved_device: str | None

    def predict(self, rgb: np.ndarray) -> MaskRCNNFrameOutput: ...

    def close(self) -> None: ...


class LocalMaskRCNNProvider:
    """Lazy, local-checkpoint-only Torchvision provider."""

    def __init__(self, *, checkpoint_path: str | Path, config: MaskRCNNCandidateExtractionConfig) -> None:
        if not isinstance(config, MaskRCNNCandidateExtractionConfig):
            raise TypeError("config must be MaskRCNNCandidateExtractionConfig.")
        self.config = config
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        if self.checkpoint_path.stat().st_size != config.expected_checkpoint_size_bytes:
            raise ValueError("Mask R-CNN checkpoint size mismatch.")
        if _checkpoint_sha256(self.checkpoint_path) != config.expected_checkpoint_sha256:
            raise ValueError("Mask R-CNN checkpoint SHA-256 mismatch.")
        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        if torchvision.__version__ != self.config.expected_torchvision_version:
            raise RuntimeError("Installed torchvision version does not match candidate config.")
        if self.config.device_policy == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Mask R-CNN but is unavailable.")
        device = (
            "cuda"
            if self.config.device_policy == "cuda"
            or (self.config.device_policy == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            box_score_thresh=0.0,
            box_detections_per_img=self.config.max_detections_per_frame,
        )
        state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        self._model = model
        self._torch = torch
        self._resolved_device = device

    def predict(self, rgb: np.ndarray) -> MaskRCNNFrameOutput:
        self._ensure_model()
        assert self._model is not None and self._torch is not None and self._resolved_device is not None
        image = np.asarray(rgb)
        if image.dtype != np.dtype(np.uint8) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3) and dtype uint8.")
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(
            device=self._resolved_device,
            dtype=torch.float32,
        ) / 255.0
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model([tensor])[0]
        if self._resolved_device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        boxes = output["boxes"].detach().to("cpu").numpy()
        scores = output["scores"].detach().to("cpu").numpy()
        labels = output["labels"].detach().to("cpu").numpy()
        masks = output["masks"].detach().to("cpu").numpy()[:, 0]
        height, width = image.shape[:2]
        predictions: list[MaskRCNNPrediction] = []
        for box_values, score, label, mask_values in zip(boxes, scores, labels, masks, strict=True):
            if float(score) < self.config.score_threshold:
                continue
            bbox = _integral_box(box_values, width=width, height=height)
            if bbox is None:
                continue
            x, y = int(bbox.x), int(bbox.y)
            local_mask = np.asarray(
                mask_values[y:y + int(bbox.height), x:x + int(bbox.width)]
                >= self.config.mask_threshold,
                dtype=np.bool_,
            )
            predictions.append(
                MaskRCNNPrediction(
                    bbox=bbox,
                    score=float(score),
                    label_id=int(label),
                    local_mask=local_mask if np.any(local_mask) else None,
                )
            )
        predictions.sort(
            key=lambda item: (
                item.bbox.y,
                item.bbox.x,
                item.bbox.height,
                item.bbox.width,
                -item.score,
                item.label_id,
            )
        )
        del tensor, output
        return MaskRCNNFrameOutput(
            predictions=tuple(predictions),
            runtime_details={
                "inference_seconds": elapsed,
                "resolved_device": self._resolved_device,
                "dtype": "float32",
                "raw_output_count": len(scores),
            },
        )

    def close(self) -> None:
        model, torch = self._model, self._torch
        self._model = None
        self._torch = None
        if model is not None:
            model.to("cpu")
            del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_maskrcnn_candidates(
    decoded_stream: DecodedStream,
    provider: MaskRCNNProviderProtocol,
    config: MaskRCNNCandidateExtractionConfig,
) -> CandidateExtractionSnapshot:
    if not isinstance(decoded_stream, DecodedStream):
        raise TypeError("decoded_stream must be DecodedStream.")
    if not isinstance(config, MaskRCNNCandidateExtractionConfig):
        raise TypeError("config must be MaskRCNNCandidateExtractionConfig.")
    if provider.config != config:
        raise ValueError("provider config must equal candidate extraction config.")
    producer = config.producer
    stream_id = decoded_stream.stream.stream_id
    candidates: list[CandidateRecord] = []
    masks: dict[str, CandidateMaskRecord] = {}
    diagnostics: list[FrameCandidateDiagnostics] = []
    warnings: list[WarningRecord] = []
    for frame in decoded_stream.frames:
        rgb = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(
            frame.image_size.height,
            frame.image_size.width,
            3,
        ).copy()
        output = provider.predict(rgb)
        ordered_predictions = tuple(sorted(output.predictions, key=_prediction_sort_key))
        if config.class_agnostic_nms_iou is not None:
            ordered_predictions = _class_agnostic_nms_predictions(
                ordered_predictions,
                config.class_agnostic_nms_iou,
            )
        frame_warning_ids: list[str] = []
        for index, prediction in enumerate(ordered_predictions, start=1):
            candidate_id = f"cand_{frame.frame_id}_{index:03d}"
            mask_reference = None
            foreground_area = int(prediction.bbox.area)
            if prediction.local_mask is not None:
                foreground_area = int(np.count_nonzero(prediction.local_mask))
                mask_ref = f"mask:{candidate_id}"
                mask_digest = candidate_mask_digest(prediction.local_mask)
                mask_reference = MaskReference(
                    mask_ref=mask_ref,
                    mask_digest=mask_digest,
                    producer_version=config.model_version,
                    coordinate_bbox=prediction.bbox,
                    validity_status=ValidityStatus.VALID,
                )
                masks[mask_ref] = CandidateMaskRecord(
                    mask_ref=mask_ref,
                    mask_digest=mask_digest,
                    candidate_id=candidate_id,
                    frame_id=frame.frame_id,
                    coordinate_bbox=prediction.bbox,
                    mask=prediction.local_mask,
                )
            bbox_area = int(prediction.bbox.area)
            candidates.append(
                CandidateRecord(
                    envelope=RecordEnvelope(
                        record_id=candidate_id,
                        schema_version=CANDIDATE_RECORD_SCHEMA_VERSION,
                        stream_id=stream_id,
                        producer=producer,
                        context=StageContext(frame_id=frame.frame_id, candidate_id=candidate_id),
                    ),
                    candidate_id=candidate_id,
                    frame_id=frame.frame_id,
                    frame_index=frame.record.index,
                    frame_size=frame.image_size,
                    bbox=prediction.bbox,
                    center=prediction.bbox.center,
                    geometry=GeometryFeatureMetadata(
                        feature_schema_id=config.geometry_feature_schema_id,
                        producer_version=config.model_version,
                        config_digest=config.config_digest,
                        values={
                            "area": foreground_area,
                            "bbox_area": bbox_area,
                            "aspect_ratio": prediction.bbox.width / prediction.bbox.height,
                            "foreground_fill_ratio": foreground_area / bbox_area,
                            "area_ratio_to_frame": foreground_area / frame.image_size.area,
                            "model_label_id": prediction.label_id,
                            "model_score": prediction.score,
                        },
                    ),
                    candidate_source=config.candidate_source,
                    mask=mask_reference,
                    candidate_confidence=prediction.score,
                )
            )
        if not ordered_predictions:
            warning_id = f"warn_{frame.frame_id}_no_candidates"
            warnings.append(
                WarningRecord(
                    record_id=warning_id,
                    schema_version=CANDIDATE_WARNING_SCHEMA_VERSION,
                    stream_id=stream_id,
                    code="NO_CANDIDATES_IN_FRAME",
                    stage="candidate_extraction",
                    message="Mask R-CNN produced no candidates above the configured threshold.",
                    producer=producer,
                    context=StageContext(frame_id=frame.frame_id),
                    metadata={"frame_index": frame.record.index},
                )
            )
            frame_warning_ids.append(warning_id)
        diagnostics.append(
            FrameCandidateDiagnostics(
                envelope=RecordEnvelope(
                    record_id=f"candidate_diag_{frame.frame_id}",
                    schema_version=FRAME_CANDIDATE_DIAGNOSTICS_SCHEMA_VERSION,
                    stream_id=stream_id,
                    producer=producer,
                    context=StageContext(frame_id=frame.frame_id),
                    warning_ids=tuple(frame_warning_ids),
                ),
                frame_id=frame.frame_id,
                frame_index=frame.record.index,
                image_size=frame.image_size,
                summary={
                    "accepted_candidate_count": len(ordered_predictions),
                    "provider_family": "torchvision_maskrcnn",
                    "model_name": config.model_name,
                    "semantic_classes": "COCO_closed_vocabulary",
                    "class_agnostic_nms_iou": config.class_agnostic_nms_iou,
                    **dict(output.runtime_details),
                },
            )
        )
    result = CandidateExtractionResult(
        envelope=RecordEnvelope(
            record_id=f"candidate_result_{stream_id}",
            schema_version=CANDIDATE_EXTRACTION_RESULT_SCHEMA_VERSION,
            stream_id=stream_id,
            producer=producer,
            context=StageContext(),
            warning_ids=tuple(item.record_id for item in warnings),
        ),
        extractor_source=config.candidate_source,
        candidates=tuple(candidates),
        frame_diagnostics=tuple(diagnostics),
        warnings=tuple(warnings),
    )
    return CandidateExtractionSnapshot(result=result, masks=masks)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_sort_key(item: MaskRCNNPrediction) -> tuple[float, ...]:
    return (
        item.bbox.y,
        item.bbox.x,
        item.bbox.height,
        item.bbox.width,
        -item.score,
        float(item.label_id),
    )


def _class_agnostic_nms_predictions(
    predictions: tuple[MaskRCNNPrediction, ...],
    iou_threshold: float,
) -> tuple[MaskRCNNPrediction, ...]:
    kept: list[MaskRCNNPrediction] = []
    for candidate in sorted(
        predictions,
        key=lambda item: (-item.score, *_prediction_sort_key(item)),
    ):
        if all(_bbox_iou(candidate.bbox, existing.bbox) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return tuple(sorted(kept, key=_prediction_sort_key))


def _bbox_iou(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _integral_box(values: np.ndarray, *, width: int, height: int) -> BBox | None:
    if np.asarray(values).shape != (4,) or not np.isfinite(values).all():
        return None
    left = max(0, min(width, int(np.floor(float(values[0])))))
    top = max(0, min(height, int(np.floor(float(values[1])))))
    right = max(0, min(width, int(np.ceil(float(values[2])))))
    bottom = max(0, min(height, int(np.ceil(float(values[3])))))
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right - left, bottom - top)


__all__ = [
    "MASKRCNN_CANDIDATE_SOURCE",
    "MASKRCNN_CONFIG_SCHEMA_ID",
    "MASKRCNN_MODEL_NAME",
    "MASKRCNN_MODEL_VERSION",
    "LocalMaskRCNNProvider",
    "MaskRCNNCandidateExtractionConfig",
    "MaskRCNNFrameOutput",
    "MaskRCNNPrediction",
    "MaskRCNNProviderProtocol",
    "extract_maskrcnn_candidates",
]
