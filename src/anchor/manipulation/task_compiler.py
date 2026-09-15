"""Constrained LIBERO language-to-skill compiler.

The deterministic grammar in this module consumes language only. It never
looks at a benchmark name, task/init id, BDDL, simulator state, reward, or an
evaluator predicate. Known benchmark phrasings are compiled into an explicit
ordered skill graph; an optional frozen semantic resolver may handle
paraphrases, but its output is accepted only after the same entity and skill
allow-list validation.

Compilation support and execution support are intentionally separate. The IR
can faithfully represent all official LIBERO task language even when the
ANCHOR grasp/place executor does not yet implement a represented skill.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class SkillKind(str, Enum):
    PICK = "pick"
    PLACE_ON = "place_on"
    PLACE_IN = "place_in"
    PLACE_RELATIVE = "place_relative"
    OPEN = "open"
    CLOSE = "close"
    PUSH_TO = "push_to"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    STACK = "stack"


class SelectorRelation(str, Enum):
    """Observable relation used to disambiguate same-label instances.

    A selector is language metadata, not a simulator predicate. The scene
    adapter resolves it from sensor-derived 3-D geometry before execution.
    LEFT/RIGHT/FRONT/BACK/MIDDLE are zero-reference order statistics over the
    detected instances of one label, rather than hidden benchmark identities.
    """

    BETWEEN = "between"
    NEXT_TO = "next_to"
    ON = "on"
    IN = "in"
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    MIDDLE = "middle"


_ZERO_REFERENCE_SELECTORS = frozenset(
    {
        SelectorRelation.CENTER,
        SelectorRelation.LEFT,
        SelectorRelation.RIGHT,
        SelectorRelation.FRONT,
        SelectorRelation.BACK,
        SelectorRelation.MIDDLE,
    }
)


@dataclass(frozen=True)
class EntitySelector:
    relation: SelectorRelation
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = (
            2
            if self.relation is SelectorRelation.BETWEEN
            else (0 if self.relation in _ZERO_REFERENCE_SELECTORS else 1)
        )
        if len(self.references) != expected:
            raise TaskCompilationError(
                f"selector {self.relation.value!r} requires {expected} reference(s)"
            )


@dataclass(frozen=True)
class EntityBinding:
    """One language-bound entity, optionally selected from repeated instances."""

    label: str
    selector: EntitySelector | None = None


EXECUTABLE_GRASP_PLACE_SKILLS = frozenset(
    {
        SkillKind.PICK,
        SkillKind.PLACE_ON,
        SkillKind.PLACE_IN,
        SkillKind.PLACE_RELATIVE,
        SkillKind.STACK,
    }
)

EXECUTABLE_SOURCE_SELECTORS = frozenset(
    # All of these are resolved from the measured multi-instance point cloud.
    # Rank selectors deliberately require enough visible candidates; they do
    # not silently degrade to the only detected instance.
    SelectorRelation
)

EXECUTABLE_TARGET_SELECTORS = EXECUTABLE_SOURCE_SELECTORS

EXECUTABLE_PLACEMENT_RELATIONS: Mapping[SkillKind, frozenset[str | None]] = {
    SkillKind.PLACE_ON: frozenset({None, "front", "back"}),
    SkillKind.PLACE_IN: frozenset(
        {
            None,
            "front",
            "back",
            "front_compartment",
            "back_compartment",
            "left_compartment",
            "right_compartment",
            "upper_shelf",
            "lower_shelf",
        }
    ),
    SkillKind.PLACE_RELATIVE: frozenset(
        {"left_of", "right_of", "front_of", "under"}
    ),
    SkillKind.STACK: frozenset({None}),
}


DEFAULT_ENTITIES = frozenset(
    {
        "alphabet soup",
        "bbq sauce",
        "basket",
        "black bowl",
        "book",
        "bottom drawer",
        "bowl",
        "butter",
        "cabinet",
        "cabinet shelf",
        "caddy",
        "chocolate pudding",
        "cookie box",
        "cream cheese",
        "cream cheese box",
        "frying pan",
        "ketchup",
        "microwave",
        "middle drawer",
        "milk",
        "moka pot",
        "orange juice",
        "plate",
        "rack",
        "ramekin",
        "red mug",
        "salad dressing",
        "stove",
        "tomato sauce",
        "top drawer",
        "tray",
        "white bowl",
        "white mug",
        "wine bottle",
        "wine rack",
        "wooden cabinet",
        "yellow and white mug",
    }
)


# Surface forms are still language-only. They canonicalise wording variants
# before a sensor query is ever produced.
DEFAULT_ENTITY_ALIASES: Mapping[str, str] = {
    "moka pots": "moka pot",
    "shelf": "cabinet shelf",
    "cream cheese box": "cream cheese",
}


@dataclass(frozen=True)
class SkillStep:
    kind: SkillKind
    subject: str
    target: str | None = None
    relation: str | None = None
    selector: EntitySelector | None = None
    target_selector: EntitySelector | None = None
    companions: tuple[EntityBinding, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    steps: tuple[SkillStep, ...]

    @property
    def required_entities(self) -> tuple[str, ...]:
        entities: list[str] = []

        def append(entity: str | None) -> None:
            if entity is not None and entity not in entities:
                entities.append(entity)

        for step in self.steps:
            append(step.subject)
            append(step.target)
            for selector in (step.selector, step.target_selector):
                if selector is not None:
                    for entity in selector.references:
                        append(entity)
            for companion in step.companions:
                append(companion.label)
                if companion.selector is not None:
                    for entity in companion.selector.references:
                        append(entity)
        return tuple(entities)

    @property
    def grasp_place_executable(self) -> bool:
        """Whether the current ANCHOR executor can consume this exact IR.

        This covers only skills implemented by ``AnchorController``.  Contact
        skills are handled by the common-policy dispatcher and therefore make
        this narrower property false.  This generic property also excludes
        companion/group bindings; ``anchor_executable`` separately validates
        the ordered formed-stack protocol implemented by the sequential
        controller's stack, proof-lift, and final-placement sensor gates.
        """

        return all(
            step.kind in EXECUTABLE_GRASP_PLACE_SKILLS
            and not step.companions
            and (
                step.selector is None
                or step.selector.relation in EXECUTABLE_SOURCE_SELECTORS
            )
            and (
                step.target_selector is None
                or step.target_selector.relation in EXECUTABLE_TARGET_SELECTORS
            )
            and (
                step.kind is SkillKind.PICK
                and step.relation is None
                or step.kind in EXECUTABLE_PLACEMENT_RELATIONS
                and step.relation in EXECUTABLE_PLACEMENT_RELATIONS[step.kind]
            )
            for step in self.steps
        )

    @property
    def anchor_executable(self) -> bool:
        """Whether the deployed ANCHOR dispatcher can execute every step.

        The check is language/IR-only and runs before any motion.  Unsupported
        fixture families and ambiguous mechanics fail closed, so a partially
        supported compound instruction is never started.
        """

        return anchor_execution_issue(self) is None

    @property
    def represented_skill_kinds(self) -> tuple[SkillKind, ...]:
        return tuple(dict.fromkeys(step.kind for step in self.steps))


class TaskCompilationError(ValueError):
    pass


class UnknownEntityError(TaskCompilationError):
    pass


class UnsupportedInstructionError(TaskCompilationError):
    pass


def anchor_execution_issue(spec: TaskSpec) -> str | None:
    """Return the first reason the deployed ANCHOR stack cannot execute ``spec``.

    This is intentionally conservative.  It validates the complete ordered
    graph before the dispatcher emits motion, preventing a compound command
    from being half executed before an unsupported suffix is discovered.
    """

    held_subject: str | None = None
    held_selector: EntitySelector | None = None
    formed_group: tuple[str, EntitySelector | None, EntitySelector | None] | None = None
    held_group: tuple[str, EntitySelector | None, EntitySelector | None] | None = None
    for index, step in enumerate(spec.steps):
        prefix = f"step {index} ({step.kind.value})"
        if step.kind is SkillKind.PICK:
            if held_subject is not None:
                return f"{prefix}: cannot pick while {held_subject!r} is held"
            if (
                step.selector is not None
                and step.selector.relation not in EXECUTABLE_SOURCE_SELECTORS
            ):
                return f"{prefix}: source selector is not implemented"
            if step.relation == "carry_stack":
                if formed_group is None:
                    return f"{prefix}: no sensor-verified formed stack is available"
                label, top_selector, base_selector = formed_group
                expected_companions = (EntityBinding(label, top_selector),)
                if (
                    step.subject != label
                    or step.selector != base_selector
                    or step.companions != expected_companions
                ):
                    return f"{prefix}: formed-stack bindings do not match verified stack"
                held_group = formed_group
            elif step.relation is not None or step.companions:
                return f"{prefix}: unsupported grouped-object pick structure"
            held_subject = step.subject
            held_selector = step.selector
            continue

        if step.kind in {
            SkillKind.PLACE_ON,
            SkillKind.PLACE_IN,
            SkillKind.PLACE_RELATIVE,
            SkillKind.STACK,
        }:
            if held_subject != step.subject:
                return (
                    f"{prefix}: expected held {step.subject!r}, got "
                    f"{held_subject!r}"
                )
            if step.target is None:
                return f"{prefix}: placement target is missing"
            if held_group is not None and step.relation != "carry_stack":
                return f"{prefix}: verified formed stack requires grouped placement"
            if step.relation == "carry_stack":
                if step.kind is not SkillKind.PLACE_IN or held_group is None:
                    return f"{prefix}: formed stacks support only verified PLACE_IN"
                label, top_selector, _base_selector = held_group
                if step.companions != (EntityBinding(label, top_selector),):
                    return f"{prefix}: formed-stack placement bindings do not match"
            else:
                if step.companions:
                    return f"{prefix}: unsupported grouped-object placement structure"
                allowed_relations = EXECUTABLE_PLACEMENT_RELATIONS[step.kind]
                if step.relation not in allowed_relations:
                    return f"{prefix}: placement relation {step.relation!r} is not implemented"
            if (
                step.target_selector is not None
                and step.target_selector.relation not in EXECUTABLE_TARGET_SELECTORS
            ):
                return f"{prefix}: target selector is not implemented"
            if step.kind is SkillKind.STACK:
                if step.selector != held_selector:
                    return f"{prefix}: stack source binding changed after pick"
                formed_group = (
                    step.subject,
                    step.selector,
                    step.target_selector,
                )
            elif held_group is not None:
                formed_group = None
            held_group = None
            held_subject = None
            held_selector = None
            continue

        if held_subject is not None:
            return f"{prefix}: contact motion while holding an object is unsupported"
        if step.kind in {SkillKind.OPEN, SkillKind.CLOSE}:
            if not (
                step.subject.endswith(" drawer")
                or step.subject == "microwave"
            ):
                return (
                    f"{prefix}: no RGB-D open/close mechanic exists for "
                    f"fixture {step.subject!r}"
                )
            continue
        if step.kind is SkillKind.TURN_ON:
            if step.subject != "stove":
                return f"{prefix}: only the stove turn-on mechanic exists"
            continue
        if step.kind is SkillKind.TURN_OFF:
            if step.subject != "stove":
                return f"{prefix}: only the stove turn-off mechanic exists"
            continue
        if step.kind is SkillKind.PUSH_TO:
            if not (
                step.subject == "plate"
                and step.target == "stove"
                and step.relation == "front_of"
                and step.selector is None
                and step.target_selector is None
            ):
                return f"{prefix}: only plate-to-front-of-stove push is implemented"
            continue
        return f"{prefix}: skill is not implemented"

    if held_subject is not None:
        return f"task ends while {held_subject!r} is still held"
    return None


class FrozenSemanticResolver(Protocol):
    """Optional adapter for a frozen LLM/VLM with structured output."""

    def resolve(
        self,
        instruction: str,
        *,
        allowed_entities: Sequence[str],
        allowed_skills: Sequence[str],
    ) -> Sequence[Mapping[str, str | None]]:
        """Return mappings with keys ``kind``, ``subject``, and optional ``target``."""


def _normalize(text: str) -> str:
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", _normalize(text))


@dataclass(frozen=True)
class _PlacementGoal:
    kind: SkillKind
    target: EntityBinding
    relation: str | None = None


class TaskCompiler:
    """Compile benchmark language into a small, validated ordered skill graph."""

    def __init__(
        self,
        *,
        entities: Sequence[str] = tuple(DEFAULT_ENTITIES),
        resolver: FrozenSemanticResolver | None = None,
    ) -> None:
        normalized = {_normalize(entity) for entity in entities}
        if not normalized:
            raise ValueError("entities cannot be empty")
        self.entities = frozenset(normalized)
        aliases = {
            _normalize(surface): _normalize(canonical)
            for surface, canonical in DEFAULT_ENTITY_ALIASES.items()
            if _normalize(canonical) in self.entities
        }
        mentions = [(entity, aliases.get(entity, entity)) for entity in self.entities]
        mentions.extend(aliases.items())
        self._entity_mentions = tuple(
            sorted(mentions, key=lambda item: (-len(item[0]), item[0], item[1]))
        )
        self.resolver = resolver

    def compile(self, instruction: str) -> TaskSpec:
        normalized = _normalize(instruction)
        if not normalized:
            raise UnsupportedInstructionError("instruction cannot be empty")
        try:
            steps = self._parse_known(normalized)
        except TaskCompilationError:
            if self.resolver is None:
                raise
            steps = self._validate_resolver_output(
                self.resolver.resolve(
                    normalized,
                    allowed_entities=sorted(self.entities),
                    allowed_skills=[kind.value for kind in SkillKind],
                )
            )
        if not steps:
            raise UnsupportedInstructionError(f"instruction produced no skills: {instruction!r}")
        return TaskSpec(instruction=instruction, steps=tuple(steps))

    def _parse_known(self, text: str) -> list[SkillStep]:
        # STACK is handled before generic placement because its optional
        # ``place them`` continuation refers to the newly formed pair.
        stack = re.fullmatch(
            r"stack the (.+?) on the (.+?)(?: and place them (.+))?", text
        )
        if stack:
            top = self._entity_binding(stack.group(1))
            bottom = self._entity_binding(stack.group(2))
            steps = [
                SkillStep(SkillKind.PICK, top.label, selector=top.selector),
                SkillStep(
                    SkillKind.STACK,
                    top.label,
                    bottom.label,
                    selector=top.selector,
                    target_selector=bottom.selector,
                ),
            ]
            continuation = stack.group(3)
            if continuation is not None:
                goal = self._parse_goal(continuation)
                companion = EntityBinding(top.label, top.selector)
                steps.extend(
                    [
                        SkillStep(
                            SkillKind.PICK,
                            bottom.label,
                            relation="carry_stack",
                            selector=bottom.selector,
                            companions=(companion,),
                        ),
                        self._placement_step(
                            bottom,
                            goal,
                            relation_override="carry_stack",
                            companions=(companion,),
                        ),
                    ]
                )
            return steps

        # Two explicitly named objects share one destination.
        both_named = re.fullmatch(r"put both the (.+?) and the (.+)", text)
        if both_named:
            second_source, goal_clause = self._split_direct_move(both_named.group(2))
            return [
                *self._movement_steps(both_named.group(1), goal_clause),
                *self._movement_steps(second_source, goal_clause),
            ]

        # The official plural task denotes both visible instances of one
        # label. LEFT/RIGHT are sensor-resolved ranks, not task identities.
        both_instances = re.fullmatch(r"put both (?:the )?moka pots (.+)", text)
        if both_instances:
            goal_clause = both_instances.group(1)
            return [
                *self._movement_steps(
                    "moka pot", goal_clause, forced_selector=SelectorRelation.LEFT
                ),
                *self._movement_steps(
                    "moka pot", goal_clause, forced_selector=SelectorRelation.RIGHT
                ),
            ]

        # Resolve ``close it`` to the destination of the preceding placement.
        if text.startswith("put ") and text.endswith(" and close it"):
            movement = self._parse_put(text[: -len(" and close it")])
            destination = self._last_placement_target(movement)
            return [*movement, SkillStep(SkillKind.CLOSE, destination)]

        # Two independent placement clauses must remain two complete
        # pick/place sequences. The delimiter does not collide with names
        # such as "yellow and white mug".
        if text.startswith("put the ") and " and put the " in text:
            first, second = text.split(" and put the ", maxsplit=1)
            return [
                *self._parse_put(first),
                *self._parse_put(f"put the {second}"),
            ]

        # Fixture action followed by another action or a placement. ``it``
        # normally refers to the fixture; for "on top of it" after a drawer
        # action, the official wording refers to the containing cabinet.
        compound = re.fullmatch(
            r"(open|close|turn on|turn off) the (.+?) and (.+)", text
        )
        if compound:
            first = self._fixture_step(compound.group(1), compound.group(2))
            rest = compound.group(3)
            if rest.startswith("put "):
                return [
                    first,
                    *self._parse_put(
                        rest,
                        antecedent=first.subject,
                        antecedent_top=first.target or first.subject,
                    ),
                ]
            return [first, *self._parse_known(rest)]

        pickup = re.fullmatch(
            r"pick up the (.+?) and (?:place|put) it (.+)", text
        )
        if pickup:
            return self._movement_steps(pickup.group(1), pickup.group(2))

        if text.startswith("put "):
            return self._parse_put(text)

        if text.startswith("open the "):
            return [self._fixture_step("open", text[len("open the ") :])]
        if text.startswith("close the "):
            return [self._fixture_step("close", text[len("close the ") :])]

        push = re.fullmatch(r"push the (.+?) to the front of the (.+)", text)
        if push:
            subject = self._entity_binding(push.group(1))
            target = self._entity_binding(push.group(2))
            return [
                SkillStep(
                    SkillKind.PUSH_TO,
                    subject.label,
                    target.label,
                    relation="front_of",
                    selector=subject.selector,
                    target_selector=target.selector,
                )
            ]

        if text.startswith("turn on the "):
            return [self._fixture_step("turn on", text[len("turn on the ") :])]
        if text.startswith("turn off the "):
            return [self._fixture_step("turn off", text[len("turn off the ") :])]

        raise UnsupportedInstructionError(f"unsupported instruction grammar: {text!r}")

    def _parse_put(
        self,
        text: str,
        *,
        antecedent: str | None = None,
        antecedent_top: str | None = None,
    ) -> list[SkillStep]:
        match = re.fullmatch(r"put (?:the )?(.+)", text)
        if match is None:
            raise UnsupportedInstructionError(f"unsupported put clause: {text!r}")
        source_clause, goal_clause = self._split_direct_move(match.group(1))
        return self._movement_steps(
            source_clause,
            goal_clause,
            antecedent=antecedent,
            antecedent_top=antecedent_top,
        )

    @staticmethod
    def _split_direct_move(body: str) -> tuple[str, str]:
        body = _normalize(body)
        # In the official wording, "the bowl/butter at the front/back" is an
        # instance qualifier even though the destination preposition follows
        # immediately ("... at the back in the drawer", for example).  Keep
        # the rank on the PICK and emit an ordinary placement relation.
        regional = re.fullmatch(
            r"(.+?) at the (front|back) (in|on) (?:the )?(.+)", body
        )
        if regional:
            return (
                f"{regional.group(1)} at the {regional.group(2)}",
                f"{regional.group(3)} the {regional.group(4)}",
            )

        relation = re.fullmatch(
            r"(.+?) ((?:(?:to the (?:front|left|right) of|under|on top of|on|inside|in) .+)|inside)",
            body,
        )
        if relation:
            return relation.group(1), relation.group(2)
        raise UnsupportedInstructionError(
            f"could not separate moved entity and destination: {body!r}"
        )

    def _movement_steps(
        self,
        source_clause: str,
        goal_clause: str,
        *,
        antecedent: str | None = None,
        antecedent_top: str | None = None,
        forced_selector: SelectorRelation | None = None,
    ) -> list[SkillStep]:
        source = self._entity_binding(source_clause)
        if forced_selector is not None:
            source = EntityBinding(source.label, EntitySelector(forced_selector))
        goal = self._parse_goal(
            goal_clause,
            antecedent=antecedent,
            antecedent_top=antecedent_top,
        )
        return [
            SkillStep(SkillKind.PICK, source.label, selector=source.selector),
            self._placement_step(source, goal),
        ]

    @staticmethod
    def _placement_step(
        source: EntityBinding,
        goal: _PlacementGoal,
        *,
        relation_override: str | None = None,
        companions: tuple[EntityBinding, ...] = (),
    ) -> SkillStep:
        relation = goal.relation
        if relation_override is not None:
            relation = (
                relation_override
                if relation is None
                else f"{relation_override}:{relation}"
            )
        return SkillStep(
            goal.kind,
            source.label,
            goal.target.label,
            relation=relation,
            target_selector=goal.target.selector,
            companions=companions,
        )

    def _parse_goal(
        self,
        clause: str,
        *,
        antecedent: str | None = None,
        antecedent_top: str | None = None,
    ) -> _PlacementGoal:
        clause = _normalize(clause)

        regional = re.fullmatch(
            r"at the (front|back) (in|on) (?:the )?(.+)", clause
        )
        if regional:
            kind = SkillKind.PLACE_IN if regional.group(2) == "in" else SkillKind.PLACE_ON
            target = self._target_binding(regional.group(3), antecedent)
            return _PlacementGoal(kind, target, regional.group(1))

        compartment = re.fullmatch(
            r"in (?:the )?(front|back|left|right) compartment of (?:the )?(.+)",
            clause,
        )
        if compartment:
            return _PlacementGoal(
                SkillKind.PLACE_IN,
                self._target_binding(compartment.group(2), antecedent),
                f"{compartment.group(1)}_compartment",
            )

        relative = re.fullmatch(
            r"to the (front|left|right) of (?:the )?(.+)", clause
        )
        if relative:
            relation = {
                "front": "front_of",
                "left": "left_of",
                "right": "right_of",
            }[relative.group(1)]
            return _PlacementGoal(
                SkillKind.PLACE_RELATIVE,
                self._target_binding(relative.group(2), antecedent),
                relation,
            )

        under = re.fullmatch(r"under (?:the )?(.+)", clause)
        if under:
            under_target = _strip_article(under.group(1))
            if under_target in {"cabinet shelf", "shelf"}:
                return _PlacementGoal(
                    SkillKind.PLACE_IN,
                    self._target_binding("cabinet shelf", antecedent),
                    "lower_shelf",
                )
            return _PlacementGoal(
                SkillKind.PLACE_RELATIVE,
                self._target_binding(under.group(1), antecedent),
                "under",
            )

        on_top = re.fullmatch(r"on top of (?:the )?(.+)", clause)
        if on_top:
            return _PlacementGoal(
                SkillKind.PLACE_ON,
                self._target_binding(
                    on_top.group(1), antecedent_top or antecedent
                ),
            )

        # In LIBERO instructions, "on the cabinet shelf" names the support
        # surface inside the cabinet's upper bay.  It is mechanically distinct
        # from "on top of the cabinet/shelf", which is handled above.  Keep an
        # explicit relation so execution must visually ground that inner
        # region instead of treating the whole fixture AABB as an ON target.
        inner_shelf = re.fullmatch(r"on (?:the )?(?:cabinet shelf|shelf)", clause)
        if inner_shelf:
            return _PlacementGoal(
                SkillKind.PLACE_IN,
                self._target_binding("cabinet shelf", antecedent),
                "upper_shelf",
            )

        on = re.fullmatch(r"on (?:the )?(.+)", clause)
        if on:
            return _PlacementGoal(
                SkillKind.PLACE_ON,
                self._target_binding(on.group(1), antecedent),
            )

        inside = re.fullmatch(r"inside(?: (?:the )?(.+))?", clause)
        if inside:
            target_clause = inside.group(1) or "it"
            return _PlacementGoal(
                SkillKind.PLACE_IN,
                self._target_binding(target_clause, antecedent),
            )

        in_target = re.fullmatch(r"in (?:the )?(.+)", clause)
        if in_target:
            return _PlacementGoal(
                SkillKind.PLACE_IN,
                self._target_binding(in_target.group(1), antecedent),
            )

        raise UnsupportedInstructionError(f"unsupported placement clause: {clause!r}")

    def _target_binding(
        self, clause: str, antecedent: str | None
    ) -> EntityBinding:
        clause = _strip_article(clause)
        if clause == "it":
            if antecedent is None:
                raise UnsupportedInstructionError("pronoun 'it' has no antecedent")
            return EntityBinding(antecedent)
        if clause == "them":
            raise UnsupportedInstructionError(
                "plural pronoun requires an explicit grouped-object continuation"
            )
        return self._entity_binding(clause)

    def _fixture_step(self, action: str, clause: str) -> SkillStep:
        binding = self._entity_binding(clause)
        kind = {
            "open": SkillKind.OPEN,
            "close": SkillKind.CLOSE,
            "turn on": SkillKind.TURN_ON,
            "turn off": SkillKind.TURN_OFF,
        }[action]
        parent = self._second_entity(clause, binding.label)
        return SkillStep(
            kind,
            binding.label,
            target=parent,
            selector=binding.selector,
        )

    @staticmethod
    def _last_placement_target(steps: Sequence[SkillStep]) -> str:
        placement_kinds = {
            SkillKind.PLACE_ON,
            SkillKind.PLACE_IN,
            SkillKind.PLACE_RELATIVE,
            SkillKind.STACK,
        }
        for step in reversed(steps):
            if step.kind in placement_kinds and step.target is not None:
                return step.target
        raise UnsupportedInstructionError("close-it continuation has no destination antecedent")

    def _entity_match(self, clause: str) -> tuple[str, int, int]:
        normalized = _normalize(clause)
        padded = f" {normalized} "
        matches: list[tuple[int, int, str]] = []
        for surface, canonical in self._entity_mentions:
            token = f" {surface} "
            index = padded.find(token)
            if index >= 0:
                matches.append((index, -len(surface), canonical))
        if not matches:
            raise UnknownEntityError(f"no allowed entity found in clause: {clause!r}")
        index, negative_length, canonical = min(matches)
        # The one-character left padding makes its token index equal to the
        # corresponding start offset in the unpadded normalized string.
        start = index
        end = start - negative_length
        return canonical, start, end

    def _first_entity(self, clause: str) -> str:
        return self._entity_match(clause)[0]

    def _second_entity(self, clause: str, first: str) -> str | None:
        normalized = _normalize(clause)
        _canonical, _start, end = self._entity_match(normalized)
        remainder = normalized[end:].strip()
        if not remainder:
            return None
        try:
            second = self._first_entity(remainder)
        except UnknownEntityError:
            return None
        return second if second != first else None

    def _entity_binding(self, clause: str) -> EntityBinding:
        subject, start, end = self._entity_match(clause)
        normalized = _normalize(clause)
        prefix = normalized[:start].strip()
        suffix = normalized[end:].strip()
        selector = self._parse_selector(prefix, suffix)
        return EntityBinding(subject, selector)

    def _parse_selector(
        self, prefix: str, suffix: str
    ) -> EntitySelector | None:
        qualifier = _normalize(suffix)

        between = re.search(r"\bbetween (?:the )?(.+?) and (?:the )?(.+)$", qualifier)
        if between:
            return EntitySelector(
                SelectorRelation.BETWEEN,
                (self._first_entity(between.group(1)), self._first_entity(between.group(2))),
            )
        next_to = re.search(r"\bnext to (?:the )?(.+)$", qualifier)
        if next_to:
            return EntitySelector(
                SelectorRelation.NEXT_TO, (self._first_entity(next_to.group(1)),)
            )
        if re.search(r"\bfrom (?:the )?table cent(?:er|re)$", qualifier):
            return EntitySelector(SelectorRelation.CENTER)

        ranks = {
            "left": SelectorRelation.LEFT,
            "right": SelectorRelation.RIGHT,
            "front": SelectorRelation.FRONT,
            "back": SelectorRelation.BACK,
            "middle": SelectorRelation.MIDDLE,
        }
        prefix_rank = re.search(r"(?:^| )(left|right|front|back|middle)$", _normalize(prefix))
        suffix_rank = re.fullmatch(
            r"(?:on|at|in) (?:the )?(left|right|front|back|middle)", qualifier
        )
        rank = suffix_rank or prefix_rank
        if rank:
            return EntitySelector(ranks[rank.group(1)])

        on = re.search(r"\bon(?: top of)? (?:the )?(.+)$", qualifier)
        if on:
            return EntitySelector(SelectorRelation.ON, (self._first_entity(on.group(1)),))
        inside = re.search(r"\bin(?:side)? (?:the )?(.+)$", qualifier)
        if inside:
            return EntitySelector(SelectorRelation.IN, (self._first_entity(inside.group(1)),))
        # Some LIBERO versions say "top layer" while task filenames say
        # "top drawer". Both are observable parts of the same fixture.
        if "top layer" in qualifier and "wooden cabinet" in qualifier:
            return EntitySelector(SelectorRelation.IN, ("wooden cabinet",))
        return None

    def _validate_resolver_output(
        self, rows: Sequence[Mapping[str, str | None]]
    ) -> list[SkillStep]:
        validated: list[SkillStep] = []
        target_required = {
            SkillKind.PLACE_ON,
            SkillKind.PLACE_IN,
            SkillKind.PLACE_RELATIVE,
            SkillKind.PUSH_TO,
            SkillKind.STACK,
        }
        for index, row in enumerate(rows):
            extra = set(row) - {"kind", "subject", "target", "relation"}
            if extra:
                raise TaskCompilationError(
                    f"resolver step {index} has unexpected keys: {sorted(extra)}"
                )
            try:
                kind = SkillKind(str(row["kind"]))
            except (KeyError, ValueError) as exc:
                raise TaskCompilationError(f"resolver step {index} has invalid kind") from exc
            subject = _normalize(str(row.get("subject") or ""))
            target_raw = row.get("target")
            target = _normalize(str(target_raw)) if target_raw else None
            if subject not in self.entities:
                raise UnknownEntityError(f"resolver returned unknown subject: {subject!r}")
            if target is not None and target not in self.entities:
                raise UnknownEntityError(f"resolver returned unknown target: {target!r}")
            if kind in target_required and target is None:
                raise TaskCompilationError(f"resolver step {index} requires a target")
            relation_raw = row.get("relation")
            relation = str(relation_raw) if relation_raw else None
            validated.append(SkillStep(kind, subject, target, relation))
        return validated
