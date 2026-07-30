"""A composable guardrail pipeline with typed per-stage payloads.

Each stage receives exactly one payload type, so a guardrail can be written
against a concrete object instead of defensively probing an untyped blob:

===========  ==========================================================
Stage        Payload
===========  ==========================================================
``INPUT``    ``str`` -- the raw incoming user message
``CONTEXT``  ``ScopedContext`` -- assembled context, pre-compile
``TOOL``     ``ToolProposal`` -- proposed tool + args, pre-execution
``OUTPUT``   ``tuple[str, list[ToolResult]]`` -- final text plus every
             ToolResult gathered during the turn, for evidence-grounding
===========  ==========================================================

The OUTPUT stage runs only when a turn resolves to FINAL. A clarifying question
is not a claim and has no tool evidence to check against.

A guardrail signals a problem by raising :class:`GuardrailViolation`.
:class:`~agent_harness.loop.LoopController` catches it and converts it into a
clean ``LoopResult`` rather than letting it reach the calling process.

This module intentionally imports nothing from the rest of the package: payload
types are documented, not enforced at import time, which keeps the guardrail
layer free of import cycles.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

__all__ = [
    "GuardrailStage",
    "GuardrailViolation",
    "Guardrail",
    "GuardrailPipeline",
]


class GuardrailStage(str, Enum):
    """Where in a turn a guardrail runs.

    Values are the lowercase stage names, which is what appears in a
    ``stopped_reason`` of ``"guardrail_violation:{stage}"``.
    """

    INPUT = "input"
    CONTEXT = "context"
    TOOL = "tool"
    OUTPUT = "output"


class GuardrailViolation(Exception):
    """Raised by a guardrail check to stop the turn.

    Carries the stage that rejected, so a single handler can identify the origin
    without needing a separate try/except per call site.
    """

    def __init__(self, stage: GuardrailStage, reason: str) -> None:
        super().__init__(f"{stage.value}: {reason}")
        self.stage = stage
        self.reason = reason


@dataclass
class Guardrail:
    """One named check bound to one stage.

    ``check`` receives the payload type for its stage and raises
    :class:`GuardrailViolation` to reject. Sync and async checks are both
    accepted; a sync check is simply not awaited.
    """

    name: str
    stage: GuardrailStage
    check: Callable[[Any], Any]


class GuardrailPipeline:
    """Guardrails grouped by stage, run in registration order."""

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self._by_stage: dict[GuardrailStage, list[Guardrail]] = {
            stage: [] for stage in GuardrailStage
        }
        for guardrail in guardrails or []:
            self.add(guardrail)

    def add(self, guardrail: Guardrail) -> GuardrailPipeline:
        """Register a guardrail. Returns ``self`` so calls can be chained."""
        self._by_stage[guardrail.stage].append(guardrail)
        return self

    def for_stage(self, stage: GuardrailStage) -> list[Guardrail]:
        """The guardrails registered for ``stage``, in registration order."""
        return list(self._by_stage[stage])

    async def run(self, stage: GuardrailStage, payload: Any) -> None:
        """Run every guardrail for ``stage`` against ``payload``.

        Raises the first :class:`GuardrailViolation` encountered; later
        guardrails for that stage do not run.
        """
        for guardrail in self._by_stage[stage]:
            outcome = guardrail.check(payload)
            if inspect.isawaitable(outcome):
                await outcome
