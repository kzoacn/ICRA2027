"""Small language compiler for LIBERO-Goal contact tasks."""

from __future__ import annotations

import re

from .schema import GoalSkillKind, GoalSkillPlan, GoalSkillStep


class GoalTaskCompiler:
    """Compile contact language while leaving grasp/place to ANCHOR."""

    def compile(self, instruction: str) -> GoalSkillPlan:
        text = " ".join(instruction.lower().strip().split())
        if not text:
            raise ValueError("instruction cannot be empty")
        steps: list[GoalSkillStep] = []
        drawer = re.search(r"open (?:the )?(top|middle|bottom) drawer", text)
        close_drawer = re.search(r"close (?:the )?(top|middle|bottom) drawer", text)
        if drawer:
            level = drawer.group(1)
            steps.append(
                GoalSkillStep(GoalSkillKind.OPEN_DRAWER, f"{level} drawer", "cabinet", level)
            )
        if close_drawer:
            level = close_drawer.group(1)
            steps.append(
                GoalSkillStep(GoalSkillKind.CLOSE_DRAWER, f"{level} drawer", "cabinet", level)
            )
        if "open the microwave" in text:
            steps.append(
                GoalSkillStep(GoalSkillKind.OPEN_MICROWAVE, "microwave")
            )
        if "close the microwave" in text:
            steps.append(
                GoalSkillStep(GoalSkillKind.CLOSE_MICROWAVE, "microwave")
            )
        if "turn on" in text and "stove" in text:
            steps.append(
                GoalSkillStep(
                    GoalSkillKind.TURN_KNOB,
                    "stove knob",
                    "stove",
                    turn_direction=1,
                )
            )
        if "turn off" in text and "stove" in text:
            steps.append(
                GoalSkillStep(
                    GoalSkillKind.TURN_KNOB,
                    "stove knob",
                    "stove",
                    turn_direction=-1,
                )
            )
        if "push" in text and "plate" in text and "front" in text and "stove" in text:
            steps.append(GoalSkillStep(GoalSkillKind.PUSH_OBJECT, "plate", "front of stove"))
        if drawer and any(token in text for token in ("put ", "place ")) and "bowl" in text:
            steps.append(
                GoalSkillStep(
                    GoalSkillKind.ANCHOR_PLACE_IN,
                    "black bowl" if "black bowl" in text else "bowl",
                    f"{drawer.group(1)} drawer",
                    drawer.group(1),
                )
            )
        if not steps:
            raise ValueError(f"unsupported Goal contact instruction: {instruction!r}")
        return GoalSkillPlan(instruction, tuple(steps))
