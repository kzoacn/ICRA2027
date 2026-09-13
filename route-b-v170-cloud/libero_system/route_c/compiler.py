"""Deterministic language-to-constraint templates for LIBERO pick/place tasks."""

from __future__ import annotations

import re
from typing import Iterable

from .schema import (
    ConstraintGraph,
    ConstraintKind,
    ConstraintSpec,
    EntityRef,
    Phase,
    Relation,
    SAFE_PHASES,
    SchemaError,
    SpatialSelector,
)


class UnsupportedTaskLanguage(SchemaError):
    """The deterministic grammar cannot compile the requested operation."""


def _clean(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return " ".join(text.split())


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip())


def _known_label(clause: str, labels: tuple[str, ...]) -> str | None:
    matches = [label for label in labels if re.search(rf"\b{re.escape(label)}\b", clause)]
    return max(matches, key=lambda x: (len(x.split()), len(x))) if matches else None


class TemplateConstraintCompiler:
    """Compile frozen task templates; never asks a model for executable code.

    ``known_labels`` is optional. Supplying the open-vocabulary detector's label
    catalogue makes noun extraction unambiguous while keeping execution fully
    sensor based.
    """

    _SOURCE_SELECTORS: tuple[tuple[str, Relation], ...] = (
        (r"\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)$", Relation.BETWEEN),
        (r"\s+next\s+to\s+(?:the\s+)?(.+)$", Relation.NEXT_TO),
        (r"\s+to\s+the\s+left\s+of\s+(?:the\s+)?(.+)$", Relation.LEFT_OF),
        (r"\s+to\s+the\s+right\s+of\s+(?:the\s+)?(.+)$", Relation.RIGHT_OF),
        (r"\s+in\s+front\s+of\s+(?:the\s+)?(.+)$", Relation.FRONT_OF),
        (r"\s+behind\s+(?:the\s+)?(.+)$", Relation.BEHIND),
        (r"\s+on(?:\s+top\s+of)?\s+(?:the\s+)?(.+)$", Relation.ON),
        (r"\s+in(?:side)?\s+(?:the\s+)?(.+)$", Relation.IN),
        (r"\s+from\s+table\s+cent(?:er|re)$", Relation.CENTER),
        (r"\s+at\s+the\s+(?:table\s+)?cent(?:er|re)$", Relation.CENTER),
    )

    _GOAL_PREFIXES: tuple[tuple[str, Relation], ...] = (
        (r"^(?:to\s+)?the\s+left\s+of\s+(?:the\s+)?(.+)$", Relation.LEFT_OF),
        (r"^(?:to\s+)?the\s+right\s+of\s+(?:the\s+)?(.+)$", Relation.RIGHT_OF),
        (r"^in\s+front\s+of\s+(?:the\s+)?(.+)$", Relation.FRONT_OF),
        (r"^behind\s+(?:the\s+)?(.+)$", Relation.BEHIND),
        (r"^on(?:\s+top\s+of)?\s+(?:the\s+)?(.+)$", Relation.ON),
        (r"^in(?:side)?\s+(?:the\s+)?(.+)$", Relation.IN),
    )

    def __init__(self, known_labels: Iterable[str] = ()) -> None:
        labels = {_clean(x) for x in known_labels if _clean(x)}
        self._known_labels = tuple(sorted(labels, key=lambda x: (-len(x.split()), -len(x), x)))

    def compile(self, task_text: str) -> ConstraintGraph:
        text = _clean(task_text)
        if not text:
            raise UnsupportedTaskLanguage("task language is empty")

        source_clause: str
        goal_clause: str
        match = re.match(
            r"^(?:pick(?:\s+up)?|grasp|take)\s+(?:the\s+)?(.+?)\s+and\s+"
            r"(?:place|put)\s+(?:it|them)\s+(.+)$",
            text,
        )
        if match:
            source_clause, goal_clause = match.group(1), match.group(2)
        else:
            direct = re.match(r"^(?:place|put)\s+(?:the\s+)?(.+?)\s+((?:on|in|to|behind).+)$", text)
            if not direct:
                raise UnsupportedTaskLanguage(
                    "supported templates are pick/grasp ... and place/put it on/in/relative-to ..."
                )
            source_clause, goal_clause = direct.group(1), direct.group(2)

        source, selector = self._parse_source(source_clause)
        relation, target = self._parse_goal(goal_clause)
        if source == target:
            raise UnsupportedTaskLanguage("source and target resolve to the same noun phrase")
        return self._build_graph(task_text.strip(), source, selector, target, relation)

    def _parse_source(self, clause: str) -> tuple[str, SpatialSelector | None]:
        known = _known_label(clause, self._known_labels)
        selector: SpatialSelector | None = None
        bare = clause
        for pattern, relation in self._SOURCE_SELECTORS:
            match = re.search(pattern, clause)
            if match:
                bare = clause[: match.start()]
                refs = tuple(_strip_article(x) for x in match.groups() if x is not None)
                if self._known_labels:
                    refs = tuple(_known_label(x, self._known_labels) or x for x in refs)
                selector = SpatialSelector(relation=relation, references=refs)
                break
        label = known or _strip_article(bare)
        if not label:
            raise UnsupportedTaskLanguage("could not extract source entity")
        return label, selector

    def _parse_goal(self, clause: str) -> tuple[Relation, str]:
        for pattern, relation in self._GOAL_PREFIXES:
            match = re.match(pattern, clause)
            if match:
                target_clause = match.group(1)
                known = _known_label(target_clause, self._known_labels)
                target = known or _strip_article(target_clause)
                target = re.sub(r"\s+(?:please|now)$", "", target).strip()
                if not target:
                    break
                return relation, target
        raise UnsupportedTaskLanguage(f"unsupported placement relation in: {clause!r}")

    @staticmethod
    def _build_graph(
        task_text: str,
        source: str,
        selector: SpatialSelector | None,
        target: str,
        relation: Relation,
    ) -> ConstraintGraph:
        motion_phases = (
            Phase.APPROACH,
            Phase.GRASP,
            Phase.LIFT,
            Phase.TRANSFER,
            Phase.PLACE,
            Phase.RETREAT,
        )
        constraints = (
            ConstraintSpec(
                "bind-grasp",
                ConstraintKind.GRASP_BINDING,
                (Phase.APPROACH, Phase.GRASP, Phase.LIFT, Phase.TRANSFER, Phase.PLACE),
                "end_effector",
                "source",
                80.0,
                0.008,
                {"pregrasp_distance": 0.09},
            ),
            ConstraintSpec(
                "scene-clearance",
                ConstraintKind.COLLISION_CLEARANCE,
                motion_phases,
                "end_effector",
                "scene",
                100.0,
                0.005,
                {"clearance": 0.025, "tool_radius": 0.025},
            ),
            ConstraintSpec(
                "trajectory-smoothness",
                ConstraintKind.SMOOTHNESS,
                motion_phases,
                "end_effector",
                None,
                2.0,
                0.0,
                {"order": 2.0},
            ),
            ConstraintSpec(
                "workspace-reachability",
                ConstraintKind.REACHABILITY,
                motion_phases,
                "end_effector",
                "workspace",
                120.0,
                0.002,
                {"margin": 0.01},
            ),
            ConstraintSpec(
                "safe-height",
                ConstraintKind.ABOVE,
                (Phase.LIFT, Phase.TRANSFER),
                "source",
                "scene",
                25.0,
                0.01,
                {"margin": 0.08},
            ),
            ConstraintSpec(
                "placement-goal",
                ConstraintKind.GOAL_RELATION,
                (Phase.PLACE, Phase.VERIFY),
                "source",
                "target",
                150.0,
                0.015,
                {
                    "relation": relation.value,
                    "xy_margin": 0.012,
                    "vertical_offset": 0.006,
                    "inside_margin": 0.015,
                    "relative_offset": 0.09,
                },
            ),
        )
        return ConstraintGraph(
            task_text=task_text,
            source=EntityRef("source", source, selector),
            target=EntityRef("target", target),
            goal_relation=relation,
            phases=SAFE_PHASES,
            constraints=constraints,
        )

