"""Strict, serialisable schema for Planning constraint graphs.

The graph is deliberately much smaller than a general-purpose planning language.
It is safe to populate from an LLM/VLM because unknown fields, constraint kinds,
phases, relations, and parameter names are rejected before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "planning.constraint-graph.v1"


class SchemaError(ValueError):
    """Raised when a graph is outside the executable Planning language."""


class Relation(StrEnum):
    ON = "on"
    IN = "in"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    NEXT_TO = "next_to"
    BETWEEN = "between"
    CENTER = "center"
    UNDER = "under"
    LEFTMOST = "leftmost"
    RIGHTMOST = "rightmost"
    FRONTMOST = "frontmost"
    BACKMOST = "backmost"
    MIDDLE = "middle"
    MIDDLE_PART = "middle_part"
    TOPMOST = "topmost"
    BOTTOMMOST = "bottommost"
    TOP_PART = "top_part"
    BOTTOM_PART = "bottom_part"
    FIRST = "first"
    SECOND = "second"
    FARTHEST_FROM = "farthest_from"


class Phase(StrEnum):
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    VERIFY = "verify"


class ConstraintKind(StrEnum):
    GRASP_BINDING = "grasp_binding"
    COLLISION_CLEARANCE = "collision_clearance"
    SMOOTHNESS = "smoothness"
    REACHABILITY = "reachability"
    ABOVE = "above"
    GOAL_RELATION = "goal_relation"


_PARAMETERS_BY_KIND: dict[ConstraintKind, frozenset[str]] = {
    ConstraintKind.GRASP_BINDING: frozenset({"pregrasp_distance"}),
    ConstraintKind.COLLISION_CLEARANCE: frozenset({"clearance", "tool_radius"}),
    ConstraintKind.SMOOTHNESS: frozenset({"order"}),
    ConstraintKind.REACHABILITY: frozenset({"margin"}),
    ConstraintKind.ABOVE: frozenset({"margin"}),
    ConstraintKind.GOAL_RELATION: frozenset(
        {
            "relation",
            "xy_margin",
            "vertical_offset",
            "inside_margin",
            "relative_offset",
            "world_offset_x_fraction",
            "world_offset_y_fraction",
            "target_subregion",
        }
    ),
}


def _only_keys(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise SchemaError(f"unknown {where} field(s): {sorted(unknown)}")


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SchemaError(f"{name} must be finite and non-negative")
    return result


def _normalise_label(label: str) -> str:
    label = " ".join(label.lower().replace("_", " ").strip().split())
    if not label or len(label) > 96:
        raise SchemaError("entity label must contain 1..96 characters")
    return label


@dataclass(frozen=True)
class SpatialSelector:
    """Observable geometric selector used when multiple entities share a label."""

    relation: Relation
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {
            Relation.ON,
            Relation.IN,
            Relation.LEFT_OF,
            Relation.RIGHT_OF,
            Relation.FRONT_OF,
            Relation.BEHIND,
            Relation.NEXT_TO,
            Relation.BETWEEN,
            Relation.CENTER,
            Relation.LEFTMOST,
            Relation.RIGHTMOST,
            Relation.FRONTMOST,
            Relation.BACKMOST,
            Relation.MIDDLE,
            Relation.MIDDLE_PART,
            Relation.TOPMOST,
            Relation.BOTTOMMOST,
            Relation.TOP_PART,
            Relation.BOTTOM_PART,
            Relation.FIRST,
            Relation.SECOND,
            Relation.FARTHEST_FROM,
        }
        if self.relation not in allowed:
            raise SchemaError(f"unsupported selector relation: {self.relation}")
        refs = tuple(_normalise_label(x) for x in self.references)
        reference_free = {
            Relation.CENTER,
            Relation.LEFTMOST,
            Relation.RIGHTMOST,
            Relation.FRONTMOST,
            Relation.BACKMOST,
            Relation.MIDDLE,
            Relation.MIDDLE_PART,
            Relation.TOPMOST,
            Relation.BOTTOMMOST,
            Relation.TOP_PART,
            Relation.BOTTOM_PART,
            Relation.FIRST,
            Relation.SECOND,
        }
        expected = 2 if self.relation == Relation.BETWEEN else (0 if self.relation in reference_free else 1)
        if len(refs) != expected:
            raise SchemaError(
                f"selector {self.relation.value!r} requires {expected} reference label(s), got {len(refs)}"
            )
        object.__setattr__(self, "references", refs)

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation.value, "references": list(self.references)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpatialSelector":
        _only_keys(value, {"relation", "references"}, "selector")
        if "relation" not in value:
            raise SchemaError("selector.relation is required")
        refs = value.get("references", [])
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise SchemaError("selector.references must be an array")
        try:
            relation = Relation(value["relation"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid selector relation: {value.get('relation')!r}") from exc
        return cls(relation=relation, references=tuple(str(x) for x in refs))


@dataclass(frozen=True)
class EntityRef:
    role: str
    label: str
    selector: SpatialSelector | None = None

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise SchemaError("entity role must be 'source' or 'target'")
        object.__setattr__(self, "label", _normalise_label(self.label))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "label": self.label}
        if self.selector is not None:
            result["selector"] = self.selector.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityRef":
        _only_keys(value, {"role", "label", "selector"}, "entity")
        if not isinstance(value.get("role"), str) or not isinstance(value.get("label"), str):
            raise SchemaError("entity.role and entity.label are required strings")
        selector_raw = value.get("selector")
        if selector_raw is not None and not isinstance(selector_raw, Mapping):
            raise SchemaError("entity.selector must be an object")
        selector = SpatialSelector.from_dict(selector_raw) if selector_raw is not None else None
        return cls(role=value["role"], label=value["label"], selector=selector)


@dataclass(frozen=True)
class ConstraintSpec:
    constraint_id: str
    kind: ConstraintKind
    phases: tuple[Phase, ...]
    subject: str
    reference: str | None
    weight: float
    tolerance: float
    parameters: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.constraint_id or len(self.constraint_id) > 64:
            raise SchemaError("constraint_id must contain 1..64 characters")
        if not self.phases:
            raise SchemaError(f"constraint {self.constraint_id!r} has no phases")
        if self.subject not in {"end_effector", "source"}:
            raise SchemaError(f"unsupported constraint subject: {self.subject!r}")
        if self.reference not in {None, "source", "target", "scene", "workspace"}:
            raise SchemaError(f"unsupported constraint reference: {self.reference!r}")
        object.__setattr__(self, "weight", _finite_nonnegative(self.weight, "weight"))
        object.__setattr__(self, "tolerance", _finite_nonnegative(self.tolerance, "tolerance"))
        params = dict(self.parameters)
        unknown = set(params) - _PARAMETERS_BY_KIND[self.kind]
        if unknown:
            raise SchemaError(
                f"constraint {self.constraint_id!r} has forbidden parameter(s): {sorted(unknown)}"
            )
        for name, raw in params.items():
            if name == "relation":
                try:
                    Relation(raw)
                except (TypeError, ValueError) as exc:
                    raise SchemaError(f"invalid goal relation: {raw!r}") from exc
            elif name == "target_subregion":
                if raw not in {
                    "left",
                    "right",
                    "front",
                    "back",
                    "upper_shelf",
                    "lower_shelf",
                }:
                    raise SchemaError(
                        "parameters.target_subregion must be a supported planar "
                        "or shelf-cavity region"
                    )
            elif name in {"world_offset_x_fraction", "world_offset_y_fraction"}:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise SchemaError(f"parameters.{name} must be numeric")
                signed = float(raw)
                if not math.isfinite(signed) or abs(signed) > 1.0:
                    raise SchemaError(
                        f"parameters.{name} must be finite and lie in [-1, 1]"
                    )
                params[name] = signed
            else:
                params[name] = _finite_nonnegative(raw, f"parameters.{name}")
        object.__setattr__(self, "parameters", params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "phases": [x.value for x in self.phases],
            "subject": self.subject,
            "reference": self.reference,
            "weight": self.weight,
            "tolerance": self.tolerance,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintSpec":
        allowed = {
            "constraint_id",
            "kind",
            "phases",
            "subject",
            "reference",
            "weight",
            "tolerance",
            "parameters",
        }
        _only_keys(value, allowed, "constraint")
        missing = allowed - {"reference", "parameters"} - set(value)
        if missing:
            raise SchemaError(f"missing constraint field(s): {sorted(missing)}")
        raw_phases = value["phases"]
        if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
            raise SchemaError("constraint.phases must be an array")
        raw_parameters = value.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise SchemaError("constraint.parameters must be an object")
        try:
            kind = ConstraintKind(value["kind"])
            phases = tuple(Phase(x) for x in raw_phases)
        except (TypeError, ValueError) as exc:
            raise SchemaError("invalid constraint kind or phase") from exc
        return cls(
            constraint_id=str(value["constraint_id"]),
            kind=kind,
            phases=phases,
            subject=str(value["subject"]),
            reference=value.get("reference"),
            weight=value["weight"],
            tolerance=value["tolerance"],
            parameters=dict(raw_parameters),
        )


@dataclass(frozen=True)
class ConstraintGraph:
    task_text: str
    source: EntityRef
    target: EntityRef
    goal_relation: Relation
    phases: tuple[Phase, ...]
    constraints: tuple[ConstraintSpec, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported schema version: {self.schema_version!r}")
        if not isinstance(self.task_text, str) or not self.task_text.strip() or len(self.task_text) > 512:
            raise SchemaError("task_text must contain 1..512 characters")
        if self.source.role != "source" or self.target.role != "target":
            raise SchemaError("source/target entity roles do not match graph slots")
        if self.goal_relation not in {
            Relation.ON,
            Relation.IN,
            Relation.LEFT_OF,
            Relation.RIGHT_OF,
            Relation.FRONT_OF,
            Relation.BEHIND,
            Relation.UNDER,
        }:
            raise SchemaError(f"unsupported executable goal relation: {self.goal_relation.value!r}")
        if tuple(dict.fromkeys(self.phases)) != self.phases:
            raise SchemaError("graph phases must be unique and ordered")
        required = (
            Phase.APPROACH,
            Phase.GRASP,
            Phase.LIFT,
            Phase.TRANSFER,
            Phase.PLACE,
            Phase.RELEASE,
            Phase.RETREAT,
            Phase.VERIFY,
        )
        if self.phases != required:
            raise SchemaError("graph must use the fixed safe phase sequence")
        ids = [x.constraint_id for x in self.constraints]
        if len(ids) != len(set(ids)):
            raise SchemaError("constraint_id values must be unique")
        if not any(x.kind == ConstraintKind.GOAL_RELATION for x in self.constraints):
            raise SchemaError("graph has no goal_relation constraint")

    def constraints_for(self, phase: Phase) -> tuple[ConstraintSpec, ...]:
        return tuple(x for x in self.constraints if phase in x.phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_text": self.task_text,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "goal_relation": self.goal_relation.value,
            "phases": [x.value for x in self.phases],
            "constraints": [x.to_dict() for x in self.constraints],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintGraph":
        allowed = {
            "schema_version",
            "task_text",
            "source",
            "target",
            "goal_relation",
            "phases",
            "constraints",
        }
        _only_keys(value, allowed, "graph")
        missing = allowed - set(value)
        if missing:
            raise SchemaError(f"missing graph field(s): {sorted(missing)}")
        if not isinstance(value["source"], Mapping) or not isinstance(value["target"], Mapping):
            raise SchemaError("graph source and target must be objects")
        raw_phases, raw_constraints = value["phases"], value["constraints"]
        if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
            raise SchemaError("graph.phases must be an array")
        if not isinstance(raw_constraints, Sequence) or isinstance(raw_constraints, (str, bytes)):
            raise SchemaError("graph.constraints must be an array")
        try:
            goal_relation = Relation(value["goal_relation"])
            phases = tuple(Phase(x) for x in raw_phases)
        except (TypeError, ValueError) as exc:
            raise SchemaError("invalid graph goal relation or phase") from exc
        constraints: list[ConstraintSpec] = []
        for raw in raw_constraints:
            if not isinstance(raw, Mapping):
                raise SchemaError("each constraint must be an object")
            constraints.append(ConstraintSpec.from_dict(raw))
        return cls(
            schema_version=str(value["schema_version"]),
            task_text=str(value["task_text"]),
            source=EntityRef.from_dict(value["source"]),
            target=EntityRef.from_dict(value["target"]),
            goal_relation=goal_relation,
            phases=phases,
            constraints=tuple(constraints),
        )

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """Return a compact schema suitable for constrained model decoding."""

        relation_values = [x.value for x in Relation]
        phase_values = [x.value for x in Phase]
        kind_values = [x.value for x in ConstraintKind]
        entity = {
            "type": "object",
            "additionalProperties": False,
            "required": ["role", "label"],
            "properties": {
                "role": {"enum": ["source", "target"]},
                "label": {"type": "string", "minLength": 1, "maxLength": 96},
                "selector": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["relation", "references"],
                    "properties": {
                        "relation": {"enum": relation_values},
                        "references": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 96},
                            "maxItems": 2,
                        },
                    },
                },
            },
        }
        constraint = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "constraint_id",
                "kind",
                "phases",
                "subject",
                "reference",
                "weight",
                "tolerance",
                "parameters",
            ],
            "properties": {
                "constraint_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "kind": {"enum": kind_values},
                "phases": {"type": "array", "minItems": 1, "items": {"enum": phase_values}},
                "subject": {"enum": ["end_effector", "source"]},
                "reference": {"enum": [None, "source", "target", "scene", "workspace"]},
                "weight": {"type": "number", "minimum": 0},
                "tolerance": {"type": "number", "minimum": 0},
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pregrasp_distance": {"type": "number", "minimum": 0},
                        "clearance": {"type": "number", "minimum": 0},
                        "tool_radius": {"type": "number", "minimum": 0},
                        "order": {"type": "number", "minimum": 0},
                        "margin": {"type": "number", "minimum": 0},
                        "relation": {"enum": relation_values},
                        "xy_margin": {"type": "number", "minimum": 0},
                        "vertical_offset": {"type": "number", "minimum": 0},
                        "inside_margin": {"type": "number", "minimum": 0},
                        "relative_offset": {"type": "number", "minimum": 0},
                    },
                },
            },
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PlanningConstraintGraph",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "task_text",
                "source",
                "target",
                "goal_relation",
                "phases",
                "constraints",
            ],
            "properties": {
                "schema_version": {"const": SCHEMA_VERSION},
                "task_text": {"type": "string", "minLength": 1, "maxLength": 512},
                "source": entity,
                "target": entity,
                "goal_relation": {
                    "enum": [
                        Relation.ON.value,
                        Relation.IN.value,
                        Relation.LEFT_OF.value,
                        Relation.RIGHT_OF.value,
                        Relation.FRONT_OF.value,
                        Relation.BEHIND.value,
                    ]
                },
                "phases": {"type": "array", "items": {"enum": phase_values}},
                "constraints": {"type": "array", "minItems": 1, "items": constraint},
            },
        }


SAFE_PHASES = tuple(Phase)
