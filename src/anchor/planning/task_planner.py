"""Strict instruction-only high-level planning for Planning.

The planner deliberately has no benchmark, task-id, BDDL, simulator, or
evaluator dependency.  It turns a complete natural-language instruction into
an ordered tuple of typed semantic goals.  Motion/grasp/contact executors may
consume those goals later, but cannot add back semantics that were silently
dropped here: every supported grammar branch matches the entire instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


PLANNER_SCHEMA_VERSION = "planning.instruction-plan.v1"


class TaskPlanningError(ValueError):
    """Base class for invalid or unsupported high-level task language."""


class UnsupportedTaskPlan(TaskPlanningError):
    """Raised when the complete instruction is outside the closed grammar."""


class AtomicGoalKind(StrEnum):
    PLACE = "place"
    PLACE_GROUP = "place_group"
    OPEN = "open"
    CLOSE = "close"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    PUSH = "push"
    STACK = "stack"


class PlannerRelation(StrEnum):
    ON = "on"
    IN = "in"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    UNDER = "under"


class SelectorKind(StrEnum):
    """Sensor-resolvable selection within same-label entity hypotheses."""

    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    MIDDLE = "middle"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    FIRST = "first"
    SECOND = "second"
    ON = "on"
    IN = "in"
    NEXT_TO = "next_to"
    BETWEEN = "between"


class PlacementRegion(StrEnum):
    """A language-selected subregion of a sensed placement target.

    The shelf values name vertically distinct cavities of the public
    two-layer shelf asset.  They are deliberately different from planar
    caddy directions and from the exterior ``ON`` surface of the fixture.
    """

    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    UPPER_SHELF = "upper_shelf"
    LOWER_SHELF = "lower_shelf"


_POSITIONAL_SELECTORS = frozenset(
    {
        SelectorKind.LEFT,
        SelectorKind.RIGHT,
        SelectorKind.FRONT,
        SelectorKind.BACK,
        SelectorKind.MIDDLE,
        SelectorKind.TOP,
        SelectorKind.BOTTOM,
    }
)


def _normalise(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return " ".join(text.split())


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip())


@dataclass(frozen=True)
class TaskEntityRef:
    """An open-vocabulary label plus an optional sensor-derived selector."""

    label: str
    selector: EntitySelector | None = None

    def __post_init__(self) -> None:
        label = _normalise(self.label)
        label = _ENTITY_ALIASES.get(label, label)
        if not label or len(label) > 96:
            raise TaskPlanningError("entity label must contain 1..96 characters")
        if label in {"it", "them"}:
            raise TaskPlanningError("pronouns must be resolved before creating a task entity")
        if self.selector is not None and not isinstance(self.selector, EntitySelector):
            raise TaskPlanningError("entity selector must be an EntitySelector")
        object.__setattr__(self, "label", label)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"label": self.label}
        if self.selector is not None:
            result["selector"] = self.selector.to_dict()
        return result


@dataclass(frozen=True)
class EntitySelector:
    kind: SelectorKind
    references: tuple[TaskEntityRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectorKind):
            raise TaskPlanningError("selector kind must be a SelectorKind")
        references = tuple(self.references)
        if not all(isinstance(ref, TaskEntityRef) for ref in references):
            raise TaskPlanningError("selector references must be TaskEntityRef values")
        if self.kind is SelectorKind.BETWEEN:
            valid = len(references) == 2
        elif self.kind in {SelectorKind.ON, SelectorKind.IN, SelectorKind.NEXT_TO}:
            valid = len(references) == 1
        elif self.kind in _POSITIONAL_SELECTORS:
            # A drawer may retain its language parent ("top drawer of the
            # cabinet"); ordinary left/right/front/back selectors have none.
            valid = len(references) <= 1
        else:
            valid = not references
        if not valid:
            raise TaskPlanningError(
                f"selector {self.kind.value!r} has an invalid reference count"
            )
        object.__setattr__(self, "references", references)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "references": [reference.to_dict() for reference in self.references],
        }


@dataclass(frozen=True)
class AtomicGoal:
    """A fully typed semantic operation, independent of a concrete executor."""

    kind: AtomicGoalKind
    subjects: tuple[TaskEntityRef, ...]
    target: TaskEntityRef | None = None
    relation: PlannerRelation | None = None
    target_region: PlacementRegion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AtomicGoalKind):
            raise TaskPlanningError("goal kind must be an AtomicGoalKind")
        subjects = tuple(self.subjects)
        if not subjects or not all(isinstance(x, TaskEntityRef) for x in subjects):
            raise TaskPlanningError("goal subjects must contain typed task entities")
        object.__setattr__(self, "subjects", subjects)

        unary = {
            AtomicGoalKind.OPEN,
            AtomicGoalKind.CLOSE,
            AtomicGoalKind.TURN_ON,
            AtomicGoalKind.TURN_OFF,
        }
        relational = {
            AtomicGoalKind.PLACE,
            AtomicGoalKind.PLACE_GROUP,
            AtomicGoalKind.PUSH,
            AtomicGoalKind.STACK,
        }
        if self.kind in unary:
            if len(subjects) != 1 or self.target is not None or self.relation is not None:
                raise TaskPlanningError(f"{self.kind.value} must have one subject and no target")
            if self.target_region is not None:
                raise TaskPlanningError(f"{self.kind.value} cannot have a placement region")
        elif self.kind in relational:
            if not isinstance(self.target, TaskEntityRef) or not isinstance(
                self.relation, PlannerRelation
            ):
                raise TaskPlanningError(f"{self.kind.value} requires a typed target and relation")
            expected_group = self.kind is AtomicGoalKind.PLACE_GROUP
            if expected_group != (len(subjects) >= 2):
                requirement = "two or more" if expected_group else "exactly one"
                raise TaskPlanningError(f"{self.kind.value} requires {requirement} subject(s)")
            if self.kind is AtomicGoalKind.STACK and self.relation is not PlannerRelation.ON:
                raise TaskPlanningError("stack relation must be on")
            if self.kind is AtomicGoalKind.PUSH and self.relation not in {
                PlannerRelation.LEFT_OF,
                PlannerRelation.RIGHT_OF,
                PlannerRelation.FRONT_OF,
            }:
                raise TaskPlanningError("push requires a planar relative relation")
            if self.target_region is not None and self.kind not in {
                AtomicGoalKind.PLACE,
                AtomicGoalKind.PLACE_GROUP,
            }:
                raise TaskPlanningError("only placement goals may select a target region")
            if self.target_region is not None and self.relation not in {
                PlannerRelation.ON,
                PlannerRelation.IN,
            }:
                raise TaskPlanningError("target regions require an on/in placement relation")
        else:  # pragma: no cover - closed enum, defensive against future additions
            raise TaskPlanningError(f"unsupported goal kind: {self.kind!r}")

    @property
    def subject(self) -> TaskEntityRef:
        if len(self.subjects) != 1:
            raise TaskPlanningError("group goals have no singular subject")
        return self.subjects[0]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind.value,
            "subjects": [subject.to_dict() for subject in self.subjects],
        }
        if self.target is not None:
            result["target"] = self.target.to_dict()
        if self.relation is not None:
            result["relation"] = self.relation.value
        if self.target_region is not None:
            result["target_region"] = self.target_region.value
        return result


@dataclass(frozen=True)
class TaskPlan:
    instruction: str
    goals: tuple[AtomicGoal, ...]
    schema_version: str = PLANNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise TaskPlanningError(f"unsupported planner schema: {self.schema_version!r}")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise TaskPlanningError("instruction cannot be empty")
        if len(self.instruction) > 512:
            raise TaskPlanningError("instruction is too long")
        goals = tuple(self.goals)
        if not goals or not all(isinstance(goal, AtomicGoal) for goal in goals):
            raise TaskPlanningError("a task plan requires typed atomic goals")
        object.__setattr__(self, "goals", goals)

    @property
    def required_entities(self) -> tuple[str, ...]:
        labels: list[str] = []

        def visit(entity: TaskEntityRef) -> None:
            if entity.label not in labels:
                labels.append(entity.label)
            if entity.selector is not None:
                for reference in entity.selector.references:
                    visit(reference)

        for goal in self.goals:
            for subject in goal.subjects:
                visit(subject)
            if goal.target is not None:
                visit(goal.target)
        return tuple(labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "instruction": self.instruction,
            "goals": [goal.to_dict() for goal in self.goals],
        }


_KNOWN_LABELS = frozenset(
    {
        "alphabet soup",
        "basket",
        "bbq sauce",
        "black bowl",
        "book",
        "bowl",
        "butter",
        "cabinet",
        "cabinet shelf",
        "caddy",
        "chocolate pudding",
        "cookie box",
        "cream cheese",
        "cream cheese box",
        "drawer",
        "frying pan",
        "ketchup",
        "microwave",
        "milk",
        "moka pot",
        "orange juice",
        "plate",
        "rack",
        "ramekin",
        "red mug",
        "salad dressing",
        "shelf",
        "stove",
        "tomato sauce",
        "tray",
        "white bowl",
        "white mug",
        "wine bottle",
        "wine rack",
        "wooden cabinet",
        "yellow and white mug",
    }
)
_LABELS_LONGEST_FIRST = tuple(sorted(_KNOWN_LABELS, key=lambda x: (-len(x), x)))
_SELECTOR_FROM_WORD = {
    "left": SelectorKind.LEFT,
    "right": SelectorKind.RIGHT,
    "front": SelectorKind.FRONT,
    "back": SelectorKind.BACK,
    "middle": SelectorKind.MIDDLE,
    "top": SelectorKind.TOP,
    "bottom": SelectorKind.BOTTOM,
}
_REGION_FROM_WORD = {
    "left": PlacementRegion.LEFT,
    "right": PlacementRegion.RIGHT,
    "front": PlacementRegion.FRONT,
    "back": PlacementRegion.BACK,
}

# Natural-language surface aliases are resolved before any sensor query is
# produced.  LIBERO calls the public ``wooden_two_layer_shelf`` fixture both
# "shelf" and "cabinet shelf"; keeping one canonical entity label lets
# ``on top of`` address its exterior while ``on`` / ``under`` still select the
# distinct upper and lower cavities in ``_parse_goal_phrase``.
_ENTITY_ALIASES = {
    "shelf": "cabinet shelf",
    "cream cheese box": "cream cheese",
}


def _selector(kind: SelectorKind, *references: TaskEntityRef) -> EntitySelector:
    return EntitySelector(kind, tuple(references))


class PlanningTaskPlanner:
    """Compile complete supported instructions using language alone."""

    def plan(self, instruction: str) -> TaskPlan:
        text = _normalise(instruction)
        if not text:
            raise UnsupportedTaskPlan("instruction cannot be empty")

        goals = self._parse_complete(text)
        if not goals:
            raise UnsupportedTaskPlan(f"instruction produced no goals: {instruction!r}")
        return TaskPlan(instruction=instruction.strip(), goals=tuple(goals))

    def compile(self, instruction: str) -> TaskPlan:
        """Compiler-style alias for callers that already use Planning compilers."""

        return self.plan(instruction)

    def _parse_complete(self, text: str) -> list[AtomicGoal]:
        # A stacked pair is subsequently moved as one group.  ``them`` is
        # resolved now, rather than exposed as an executable free-form token.
        match = re.fullmatch(r"stack (.+) and place them (.+)", text)
        if match:
            top, bottom = self._parse_stack_body(match.group(1))
            relation, target, region = self._parse_goal_phrase(match.group(2))
            return [
                self._relational(AtomicGoalKind.STACK, top, bottom, PlannerRelation.ON),
                AtomicGoal(
                    AtomicGoalKind.PLACE_GROUP,
                    (top, bottom),
                    target,
                    relation,
                    region,
                ),
            ]

        # ``both`` is expanded in language order.  For two same-label objects,
        # ordinal selectors ensure the executor cannot select the same track
        # twice while remaining agnostic to benchmark instance names.
        if text.startswith("put both "):
            return self._parse_both(text)

        match = re.fullmatch(r"turn (on|off) (?:the )?(.+?) and put (.+)", text)
        if match:
            fixture = self._parse_entity_phrase(match.group(2))
            turn_kind = (
                AtomicGoalKind.TURN_ON if match.group(1) == "on" else AtomicGoalKind.TURN_OFF
            )
            placement = self._parse_put_body(match.group(3), pronoun=fixture)
            return [AtomicGoal(turn_kind, (fixture,)), placement]

        match = re.fullmatch(r"close (?:the )?(.+?) and put (.+)", text)
        if match:
            fixture = self._parse_entity_phrase(match.group(1))
            placement = self._parse_put_body(match.group(2), pronoun=fixture)
            return [AtomicGoal(AtomicGoalKind.CLOSE, (fixture,)), placement]

        match = re.fullmatch(r"close (?:the )?(.+?) and open (?:the )?(.+)", text)
        if match:
            first = self._parse_entity_phrase(match.group(1))
            second = self._parse_entity_phrase(match.group(2))
            return [
                AtomicGoal(AtomicGoalKind.CLOSE, (first,)),
                AtomicGoal(AtomicGoalKind.OPEN, (second,)),
            ]

        match = re.fullmatch(r"open (?:the )?(.+?) and put (.+)", text)
        if match:
            fixture = self._parse_entity_phrase(match.group(1))
            placement = self._parse_put_body(
                match.group(2), pronoun=fixture, implicit_target=fixture
            )
            return [AtomicGoal(AtomicGoalKind.OPEN, (fixture,)), placement]

        match = re.fullmatch(r"(put .+) and close it", text)
        if match:
            placement = self._parse_put_clause(match.group(1))
            assert placement.target is not None
            return [placement, AtomicGoal(AtomicGoalKind.CLOSE, (placement.target,))]

        if text.startswith("put ") and " and put " in text:
            first_text, second_body = text.split(" and put ", maxsplit=1)
            return [self._parse_put_clause(first_text), self._parse_put_body(second_body)]

        match = re.fullmatch(r"pick(?: up)? (.+) and (?:place|put) it (.+)", text)
        if match:
            source = self._parse_entity_phrase(match.group(1))
            relation, target, region = self._parse_goal_phrase(match.group(2))
            return [self._relational(AtomicGoalKind.PLACE, source, target, relation, region)]

        match = re.fullmatch(r"(open|close) (?:the )?(.+)", text)
        if match:
            subject = self._parse_entity_phrase(match.group(2))
            kind = AtomicGoalKind.OPEN if match.group(1) == "open" else AtomicGoalKind.CLOSE
            return [AtomicGoal(kind, (subject,))]

        match = re.fullmatch(r"turn (on|off) (?:the )?(.+)", text)
        if match:
            subject = self._parse_entity_phrase(match.group(2))
            kind = AtomicGoalKind.TURN_ON if match.group(1) == "on" else AtomicGoalKind.TURN_OFF
            return [AtomicGoal(kind, (subject,))]

        match = re.fullmatch(r"push (?:the )?(.+?) (to the (?:left|right|front) of .+)", text)
        if match:
            source = self._parse_entity_phrase(match.group(1))
            relation, target, region = self._parse_goal_phrase(match.group(2))
            if region is not None:
                raise UnsupportedTaskPlan("push targets cannot use a placement subregion")
            return [self._relational(AtomicGoalKind.PUSH, source, target, relation)]

        match = re.fullmatch(r"stack (.+)", text)
        if match:
            top, bottom = self._parse_stack_body(match.group(1))
            return [self._relational(AtomicGoalKind.STACK, top, bottom, PlannerRelation.ON)]

        if text.startswith("put "):
            return [self._parse_put_clause(text)]

        raise UnsupportedTaskPlan(f"unsupported complete instruction: {text!r}")

    def _parse_both(self, text: str) -> list[AtomicGoal]:
        match = re.fullmatch(r"put both (.+) (on|in) (?:the )?(.+)", text)
        if match is None:
            raise UnsupportedTaskPlan(f"incomplete both placement: {text!r}")
        subjects_text, preposition, target_text = match.groups()
        relation, target, region = self._parse_goal_phrase(f"{preposition} {target_text}")

        if _strip_article(subjects_text) == "moka pots":
            subjects = (
                TaskEntityRef("moka pot", _selector(SelectorKind.FIRST)),
                TaskEntityRef("moka pot", _selector(SelectorKind.SECOND)),
            )
        else:
            subjects = self._parse_explicit_pair(subjects_text)
        return [
            self._relational(AtomicGoalKind.PLACE, subject, target, relation, region)
            for subject in subjects
        ]

    def _parse_explicit_pair(self, text: str) -> tuple[TaskEntityRef, TaskEntityRef]:
        for first_label in _LABELS_LONGEST_FIRST:
            for second_label in _LABELS_LONGEST_FIRST:
                pattern = (
                    rf"(?:the )?{re.escape(first_label)} and "
                    rf"(?:the )?{re.escape(second_label)}"
                )
                if re.fullmatch(pattern, text):
                    return TaskEntityRef(first_label), TaskEntityRef(second_label)
        raise UnsupportedTaskPlan(f"both must name exactly two supported entities: {text!r}")

    def _parse_stack_body(self, text: str) -> tuple[TaskEntityRef, TaskEntityRef]:
        # Try every ``on`` boundary and accept only a boundary whose two sides
        # are complete entity references.  This remains safe for phrases such
        # as "book on the left" if that grammar is extended later.
        candidates: list[tuple[TaskEntityRef, TaskEntityRef]] = []
        for boundary in re.finditer(r"\s+on\s+", text):
            try:
                top = self._parse_entity_phrase(text[: boundary.start()])
                bottom = self._parse_entity_phrase(text[boundary.end() :])
            except UnsupportedTaskPlan:
                continue
            candidates.append((top, bottom))
        if len(candidates) != 1:
            raise UnsupportedTaskPlan(f"stack requires one unambiguous on relation: {text!r}")
        return candidates[0]

    def _parse_put_clause(self, text: str) -> AtomicGoal:
        match = re.fullmatch(r"put (.+)", text)
        if match is None:
            raise UnsupportedTaskPlan(f"expected a complete put clause: {text!r}")
        return self._parse_put_body(match.group(1))

    def _parse_put_body(
        self,
        body: str,
        *,
        pronoun: TaskEntityRef | None = None,
        implicit_target: TaskEntityRef | None = None,
    ) -> AtomicGoal:
        body = _strip_article(body)
        relation_pattern = (
            r"on top of|to the left of|to the right of|to the front of|under|inside|on|in"
        )
        match = re.fullmatch(rf"(.+?) ({relation_pattern})(?: (.+))?", body)
        if match is None:
            raise UnsupportedTaskPlan(f"put clause lacks a complete placement relation: {body!r}")

        source_text, relation_text, target_text = match.groups()
        source = self._parse_entity_phrase(source_text)
        goal_tail = relation_text
        if target_text:
            goal_tail += f" {target_text}"
        relation, target, destination_region = self._parse_goal_phrase(
            goal_tail,
            pronoun=pronoun,
            implicit_target=implicit_target,
        )
        return self._relational(
            AtomicGoalKind.PLACE,
            source,
            target,
            relation,
            destination_region,
        )

    def _parse_goal_phrase(
        self,
        text: str,
        *,
        pronoun: TaskEntityRef | None = None,
        implicit_target: TaskEntityRef | None = None,
    ) -> tuple[PlannerRelation, TaskEntityRef, PlacementRegion | None]:
        prefixes = (
            ("on top of", PlannerRelation.ON),
            ("to the left of", PlannerRelation.LEFT_OF),
            ("to the right of", PlannerRelation.RIGHT_OF),
            ("to the front of", PlannerRelation.FRONT_OF),
            ("under", PlannerRelation.UNDER),
            ("inside", PlannerRelation.IN),
            ("on", PlannerRelation.ON),
            ("in", PlannerRelation.IN),
        )
        normalised = _normalise(text)
        for prefix, relation in prefixes:
            if normalised == prefix:
                target_text = ""
            elif normalised.startswith(prefix + " "):
                target_text = normalised[len(prefix) + 1 :]
            else:
                continue

            target_text = _strip_article(target_text)
            if target_text == "it":
                if pronoun is None:
                    raise UnsupportedTaskPlan("unbound pronoun 'it'")
                if (
                    relation is PlannerRelation.ON
                    and pronoun.label == "drawer"
                    and pronoun.selector is not None
                    and pronoun.selector.references
                ):
                    # In "close the top drawer of the cabinet and put X on
                    # top of it", the closed drawer is a selected part of the
                    # named cabinet; its exposed top is the cabinet top.  Keep
                    # that language parent instead of inventing a surface on
                    # the closed drawer front.
                    return relation, pronoun.selector.references[0], None
                return relation, pronoun, None
            if not target_text:
                if implicit_target is None:
                    raise UnsupportedTaskPlan(f"placement target is missing after {prefix!r}")
                return relation, implicit_target, None
            target, region = self._parse_destination(target_text)
            if (
                prefix in {"on", "under"}
                and target.label in {"cabinet shelf", "shelf"}
            ):
                # Bare ``on/under [cabinet] shelf`` names one of the two
                # storage cavities of the public two-layer fixture.  Keep the
                # cavity explicit all the way through execution.  The earlier
                # and syntactically distinct ``on top of`` prefix remains an
                # exterior ON relation.
                target = TaskEntityRef("cabinet shelf")
                relation = PlannerRelation.IN
                region = (
                    PlacementRegion.UPPER_SHELF
                    if prefix == "on"
                    else PlacementRegion.LOWER_SHELF
                )
            return relation, target, region
        raise UnsupportedTaskPlan(f"unsupported placement goal: {text!r}")

    def _parse_destination(
        self, text: str
    ) -> tuple[TaskEntityRef, PlacementRegion | None]:
        compartment = re.fullmatch(
            r"(?:the )?(front|back|left|right) compartment of (?:the )?caddy", text
        )
        if compartment:
            return TaskEntityRef("caddy"), _REGION_FROM_WORD[compartment.group(1)]
        return self._parse_entity_phrase(text), None

    def _parse_entity_phrase(self, text: str) -> TaskEntityRef:
        phrase = _strip_article(_normalise(text))
        if not phrase:
            raise UnsupportedTaskPlan("entity phrase cannot be empty")
        if phrase in {"it", "them"}:
            raise UnsupportedTaskPlan(f"unbound pronoun: {phrase!r}")
        phrase = _ENTITY_ALIASES.get(phrase, phrase)

        drawer = re.fullmatch(
            r"(top|middle|bottom) drawer(?: of (?:the )?(wooden cabinet|cabinet))?", phrase
        )
        if drawer:
            references = (TaskEntityRef(drawer.group(2)),) if drawer.group(2) else ()
            return TaskEntityRef(
                "drawer", _selector(_SELECTOR_FROM_WORD[drawer.group(1)], *references)
            )

        prefix = re.fullmatch(
            r"(left|right|front|back|middle) (black bowl|bowl|moka pot|book|plate)",
            phrase,
        )
        if prefix:
            return TaskEntityRef(
                prefix.group(2), _selector(_SELECTOR_FROM_WORD[prefix.group(1)])
            )

        suffix = re.fullmatch(
            r"(.+?) (?:at|in|on) the (left|right|front|back|middle)", phrase
        )
        if suffix and suffix.group(1) in _KNOWN_LABELS:
            return TaskEntityRef(
                suffix.group(1), _selector(_SELECTOR_FROM_WORD[suffix.group(2)])
            )

        center = re.fullmatch(r"(.+?) from (?:the )?table cent(?:er|re)", phrase)
        if center and center.group(1) in _KNOWN_LABELS:
            return TaskEntityRef(center.group(1), _selector(SelectorKind.CENTER))

        between = re.fullmatch(r"(.+?) between (?:the )?(.+?) and (?:the )?(.+)", phrase)
        if between and between.group(1) in _KNOWN_LABELS:
            first = self._parse_entity_phrase(between.group(2))
            second = self._parse_entity_phrase(between.group(3))
            return TaskEntityRef(
                between.group(1), _selector(SelectorKind.BETWEEN, first, second)
            )

        next_to = re.fullmatch(r"(.+?) next to (?:the )?(.+)", phrase)
        if next_to and next_to.group(1) in _KNOWN_LABELS:
            reference = self._parse_entity_phrase(next_to.group(2))
            return TaskEntityRef(
                next_to.group(1), _selector(SelectorKind.NEXT_TO, reference)
            )

        for word, kind in (("on", SelectorKind.ON), ("in", SelectorKind.IN)):
            match = re.fullmatch(rf"(.+?) {word} (?:the )?(.+)", phrase)
            if match and match.group(1) in _KNOWN_LABELS:
                reference = self._parse_entity_phrase(match.group(2))
                return TaskEntityRef(match.group(1), _selector(kind, reference))

        if phrase in _KNOWN_LABELS:
            return TaskEntityRef(phrase)
        raise UnsupportedTaskPlan(f"unsupported complete entity phrase: {text!r}")

    @staticmethod
    def _relational(
        kind: AtomicGoalKind,
        source: TaskEntityRef,
        target: TaskEntityRef,
        relation: PlannerRelation,
        region: PlacementRegion | None = None,
    ) -> AtomicGoal:
        return AtomicGoal(kind, (source,), target, relation, region)


__all__ = [
    "AtomicGoal",
    "AtomicGoalKind",
    "EntitySelector",
    "PLANNER_SCHEMA_VERSION",
    "PlacementRegion",
    "PlannerRelation",
    "PlanningTaskPlanner",
    "SelectorKind",
    "TaskEntityRef",
    "TaskPlan",
    "TaskPlanningError",
    "UnsupportedTaskPlan",
]
