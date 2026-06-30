"""Pure grouped distance for compatible handcrafted representation records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..contracts import (
    HandcraftedFeatureGroup,
    HandcraftedPayload,
    RepresentationFamily,
    RepresentationRecord,
    ValidityStatus,
)
from .handcrafted import (
    BBOX_COLOR_FEATURE_NAMES,
    BBOX_SHAPE_FEATURE_NAMES,
    BBOX_SIZE_FEATURE_NAMES,
    BBOX_STRUCTURE_FEATURE_NAMES,
    FEATURE_GROUP_NAMES,
    HANDCRAFTED_BBOX_FEATURE_SCHEMA_ID,
    HANDCRAFTED_BBOX_VARIANT,
    HANDCRAFTED_MASK_FEATURE_SCHEMA_ID,
    HANDCRAFTED_MASK_VARIANT,
    HANDCRAFTED_REPRESENTATION_TYPE,
    MASK_COLOR_FEATURE_NAMES,
    MASK_SHAPE_FEATURE_NAMES,
    MASK_SIZE_FEATURE_NAMES,
    MASK_STRUCTURE_FEATURE_NAMES,
    HandcraftedRepresentationConfig,
)


@dataclass(frozen=True, slots=True)
class HandcraftedDistanceResult:
    """Bounded visual distance with explicit group contributions."""

    variant: str
    semantic_config_digest: str
    group_distances: Mapping[str, float]
    omitted_groups: tuple[str, ...]
    distance: float
    similarity: float

    def __post_init__(self) -> None:
        frozen: dict[str, float] = {}
        for name in sorted(self.group_distances, key=lambda value: value.encode("utf-8")):
            value = _bounded(self.group_distances[name], f"group_distances.{name}")
            frozen[name] = value
        distance = _bounded(self.distance, "distance")
        similarity = _bounded(self.similarity, "similarity")
        if not math.isclose(similarity, 1.0 - distance, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("similarity must equal 1 - distance.")
        object.__setattr__(self, "group_distances", MappingProxyType(frozen))
        object.__setattr__(self, "omitted_groups", tuple(self.omitted_groups))
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "similarity", similarity)


def handcrafted_distance(
    left: RepresentationRecord,
    right: RepresentationRecord,
    config: HandcraftedRepresentationConfig,
) -> HandcraftedDistanceResult:
    """Compare two compatible records without position or decision thresholds."""

    if not isinstance(config, HandcraftedRepresentationConfig):
        raise TypeError("config must be HandcraftedRepresentationConfig.")
    _validate_record(left, config, "left")
    _validate_record(right, config, "right")
    if left.input_variant != right.input_variant:
        raise ValueError("Handcrafted variants are incompatible.")
    if left.semantic_config_digest != right.semantic_config_digest:
        raise ValueError("Handcrafted semantic config digests are incompatible.")
    assert isinstance(left.payload, HandcraftedPayload)
    assert isinstance(right.payload, HandcraftedPayload)
    left_groups = _groups(left.payload)
    right_groups = _groups(right.payload)

    if config.variant == HANDCRAFTED_MASK_VARIANT:
        group_distances, omitted = _mask_group_distances(
            left_groups,
            right_groups,
            config,
        )
    else:
        group_distances, omitted = _bbox_group_distances(
            left,
            right,
            left_groups,
            right_groups,
            config,
        )

    weights = config.group_weights
    available_weight = sum(weights[name] for name in group_distances)
    if available_weight <= 0.0:
        raise ValueError("At least one positive-weight feature group is required.")
    distance = sum(weights[name] * value for name, value in group_distances.items()) / available_weight
    distance = min(1.0, max(0.0, float(distance)))
    return HandcraftedDistanceResult(
        variant=config.variant,
        semantic_config_digest=config.config_digest,
        group_distances=group_distances,
        omitted_groups=omitted,
        distance=distance,
        similarity=1.0 - distance,
    )


def _validate_record(
    record: RepresentationRecord,
    config: HandcraftedRepresentationConfig,
    field_name: str,
) -> None:
    if not isinstance(record, RepresentationRecord):
        raise TypeError(f"{field_name} must be RepresentationRecord.")
    if record.envelope.validity_status is not ValidityStatus.VALID or record.payload is None:
        raise ValueError(f"{field_name} must be a valid representation record.")
    if record.family is not RepresentationFamily.HANDCRAFTED:
        raise ValueError(f"{field_name} must belong to the handcrafted family.")
    if record.representation_type != HANDCRAFTED_REPRESENTATION_TYPE:
        raise ValueError(f"{field_name} has an incompatible representation_type.")
    if record.input_variant != config.variant:
        raise ValueError(f"{field_name} input_variant does not match config.variant.")
    if record.representation_version != config.representation_version:
        raise ValueError(f"{field_name} representation_version is incompatible.")
    if record.semantic_config_digest != config.config_digest:
        raise ValueError(f"{field_name} semantic config digest does not match config.")
    if not isinstance(record.payload, HandcraftedPayload):
        raise TypeError(f"{field_name} payload must be HandcraftedPayload.")
    if record.payload.feature_schema_id != config.feature_schema_id:
        raise ValueError(f"{field_name} feature schema is incompatible.")


def _groups(payload: HandcraftedPayload) -> Mapping[str, HandcraftedFeatureGroup]:
    groups = {group.group_name: group for group in payload.feature_groups}
    if set(groups) != set(FEATURE_GROUP_NAMES):
        raise ValueError("Handcrafted payload must contain shape, structure, color and size groups.")
    return groups


def _mask_group_distances(
    left: Mapping[str, HandcraftedFeatureGroup],
    right: Mapping[str, HandcraftedFeatureGroup],
    config: HandcraftedRepresentationConfig,
) -> tuple[dict[str, float], tuple[str, ...]]:
    _require_valid_group(left["shape"], "shape")
    _require_valid_group(right["shape"], "shape")
    _require_valid_group(left["size"], "size")
    _require_valid_group(right["size"], "size")
    result: dict[str, float] = {
        "shape": _mask_shape_distance(left["shape"], right["shape"], config),
        "size": _size_distance(
            left["size"],
            right["size"],
            MASK_SIZE_FEATURE_NAMES[0],
            config,
        ),
    }
    omitted: list[str] = []
    if left["structure"].valid and right["structure"].valid:
        result["structure"] = _structure_distance(
            left["structure"],
            right["structure"],
            MASK_STRUCTURE_FEATURE_NAMES,
            config,
        )
    else:
        omitted.append("structure")
    if left["color"].valid and right["color"].valid:
        result["color"] = _color_distance(
            left["color"],
            right["color"],
            MASK_COLOR_FEATURE_NAMES,
        )
    else:
        omitted.append("color")
    return result, tuple(sorted(omitted))


def _bbox_group_distances(
    left_record: RepresentationRecord,
    right_record: RepresentationRecord,
    left: Mapping[str, HandcraftedFeatureGroup],
    right: Mapping[str, HandcraftedFeatureGroup],
    config: HandcraftedRepresentationConfig,
) -> tuple[dict[str, float], tuple[str, ...]]:
    for name in FEATURE_GROUP_NAMES:
        _require_valid_group(left[name], name)
        _require_valid_group(right[name], name)
    _require_keys(left["shape"], BBOX_SHAPE_FEATURE_NAMES)
    _require_keys(right["shape"], BBOX_SHAPE_FEATURE_NAMES)
    shape = abs(
        left["shape"].values["hcb_box_elongation"]
        - right["shape"].values["hcb_box_elongation"]
    )
    structure = _bbox_structure_distance(
        left_record,
        right_record,
        left["structure"],
        right["structure"],
        config,
    )
    return {
        "shape": _bounded(shape, "bbox shape distance"),
        "structure": structure,
        "color": _color_distance(left["color"], right["color"], BBOX_COLOR_FEATURE_NAMES),
        "size": _size_distance(
            left["size"],
            right["size"],
            BBOX_SIZE_FEATURE_NAMES[0],
            config,
        ),
    }, ()


def _mask_shape_distance(
    left: HandcraftedFeatureGroup,
    right: HandcraftedFeatureGroup,
    config: HandcraftedRepresentationConfig,
) -> float:
    _require_keys(left, MASK_SHAPE_FEATURE_NAMES)
    _require_keys(right, MASK_SHAPE_FEATURE_NAMES)
    scalar_names = MASK_SHAPE_FEATURE_NAMES[:5]
    radial_names = MASK_SHAPE_FEATURE_NAMES[5:]
    scalar = sum(abs(left.values[name] - right.values[name]) for name in scalar_names) / len(scalar_names)
    radial = 0.5 * sum(abs(left.values[name] - right.values[name]) for name in radial_names)
    ws, wr = config.shape_component_weights
    return _bounded(ws * scalar + wr * radial, "shape distance")


def _structure_distance(
    left: HandcraftedFeatureGroup,
    right: HandcraftedFeatureGroup,
    names: tuple[str, ...],
    config: HandcraftedRepresentationConfig,
) -> float:
    _require_keys(left, names)
    _require_keys(right, names)
    density = abs(left.values[names[0]] - right.values[names[0]])
    radial = 0.5 * sum(abs(left.values[name] - right.values[name]) for name in names[1:])
    wd, wr = config.structure_component_weights
    return _bounded(wd * density + wr * radial, "structure distance")


def _bbox_structure_distance(
    left_record: RepresentationRecord,
    right_record: RepresentationRecord,
    left: HandcraftedFeatureGroup,
    right: HandcraftedFeatureGroup,
    config: HandcraftedRepresentationConfig,
) -> float:
    _require_keys(left, BBOX_STRUCTURE_FEATURE_NAMES)
    _require_keys(right, BBOX_STRUCTURE_FEATURE_NAMES)
    density = abs(left.values["hcb_edge_density"] - right.values["hcb_edge_density"])
    left_valid = _bbox_edge_profile_valid(left_record)
    right_valid = _bbox_edge_profile_valid(right_record)
    if not left_valid and not right_valid:
        radial = 0.0
    elif left_valid != right_valid:
        radial = 1.0
    else:
        radial = 0.5 * sum(
            abs(left.values[name] - right.values[name])
            for name in BBOX_STRUCTURE_FEATURE_NAMES[1:]
        )
    wd, wr = config.structure_component_weights
    return _bounded(wd * density + wr * radial, "bbox structure distance")


def _bbox_edge_profile_valid(record: RepresentationRecord) -> bool:
    value = record.preprocessing_metadata.details.get("edge_profile_valid")
    if not isinstance(value, bool):
        raise ValueError("BBox preprocessing metadata must contain boolean edge_profile_valid.")
    return value


def _color_distance(
    left: HandcraftedFeatureGroup,
    right: HandcraftedFeatureGroup,
    names: tuple[str, str, str],
) -> float:
    _require_keys(left, names)
    _require_keys(right, names)
    value = math.sqrt(sum((left.values[name] - right.values[name]) ** 2 for name in names)) / math.sqrt(3.0)
    return _bounded(value, "color distance")


def _size_distance(
    left: HandcraftedFeatureGroup,
    right: HandcraftedFeatureGroup,
    name: str,
    config: HandcraftedRepresentationConfig,
) -> float:
    _require_keys(left, (name,))
    _require_keys(right, (name,))
    left_value = left.values[name]
    right_value = right.values[name]
    if left_value <= 0.0 or right_value <= 0.0:
        raise ValueError("Size features must be positive.")
    value = abs(
        math.log((left_value + config.epsilon) / (right_value + config.epsilon))
    ) / math.log(config.size_ratio_cap)
    if math.isfinite(value) and value > 1.0:
        value = 1.0
    return _bounded(value, "size distance")


def _require_valid_group(group: HandcraftedFeatureGroup, name: str) -> None:
    if not group.valid:
        raise ValueError(f"Feature group {name!r} must be valid.")


def _require_keys(group: HandcraftedFeatureGroup, names: tuple[str, ...]) -> None:
    if set(group.values) != set(names):
        raise ValueError(
            f"Feature group {group.group_name!r} must contain exactly {names!r}."
        )


def _bounded(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    if number < -1e-12 or number > 1.0 + 1e-12:
        raise ValueError(f"{field_name} must be in [0, 1].")
    return min(1.0, max(0.0, number))


__all__ = ["HandcraftedDistanceResult", "handcrafted_distance"]
