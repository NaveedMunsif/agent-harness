"""The bounded agent loop.

Three distinct exits, all decided by the harness and never by the model:

* **TOOL_CALL** -- validate, execute, record, fold the result into context, iterate.
* **CLARIFICATION** -- return the model's question immediately. Return-based, not
  pause-in-place: nothing suspends, blocks, or holds an open coroutine. The
  caller displays the question, collects an answer through whatever transport it
  owns, and calls :meth:`LoopController.handle_turn` again with that answer and
  the same ``current_frame``. No state is held between calls beyond what is
  passed back in.
* **FINAL** -- run the OUTPUT guardrail, format for the channel, return.

Plus the forced stops -- turn, tool-call and wall-clock budgets, and
repeated-call detection -- which return ``final_text=None`` and skip both the
OUTPUT guardrail and output formatting, because there is no answer to check or
render.

The model can propose ``final`` all it likes; the loop still ends only when
``LoopController`` says so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, model_validator

from .context import ContextEngine, ScopedContext
from .guardrails import GuardrailPipeline, GuardrailStage, GuardrailViolation
from .memory import MemoryKind, MemoryStore
from .output import OutputFormatter, RenderChannel
from .prompt import CompiledPrompt, PromptCompiler
from .state import TaskFrame
from .tools import ToolError, ToolGateway, ToolProposal, ToolResult

__all__ = [
    "TurnStepType",
    "LLMTurnStep",
    "LoopLimits",
    "LoopBudget",
    "LoopResult",
    "LoopController",
    "DEFAULT_OUTPUT_REQUIREMENT",
]

DEFAULT_OUTPUT_REQUIREMENT = "Answer concisely and ground every claim in tool results."

CallLLM = Callable[[CompiledPrompt], Awaitable["LLMTurnStep"]]


class TurnStepType(str, Enum):
    """What the model proposed doing next."""

    TOOL_CALL = "tool_call"
    CLARIFICATION = "clarification"
    FINAL = "final"


class LLMTurnStep(BaseModel):
    """One step proposed by the model.

    Field invariants are enforced at construction, so a malformed model or mock
    response fails loudly here rather than as a ``NoneType`` error deep inside
    :class:`~agent_harness.tools.ToolGateway`.
    """

    step_type: TurnStepType
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _check_fields_match_step_type(self) -> LLMTurnStep:
        if self.step_type == TurnStepType.TOOL_CALL:
            if not self.tool_name:
                raise ValueError("step_type 'tool_call' requires a non-empty tool_name")
            if self.tool_arguments is None:
                raise ValueError("step_type 'tool_call' requires tool_arguments (may be {})")
        elif self.step_type in (TurnStepType.CLARIFICATION, TurnStepType.FINAL):
            if not self.text:
                raise ValueError(f"step_type {self.step_type.value!r} requires non-empty text")
        return self


@dataclass
class LoopLimits:
    """The turn's budget. Every field is a hard ceiling."""

    max_turns: int = 6
    max_tool_calls: int = 4
    max_seconds: float = 30.0

    def new_budget(self) -> LoopBudget:
        """A fresh budget, with its clock started now."""
        return LoopBudget(limits=self)


@dataclass
class LoopBudget:
    """Live consumption against a :class:`LoopLimits`.

    ``seen_proposals`` holds ``(tool_name, canonical_json_args)`` pairs rather
    than ``ToolProposal`` objects, which are unhashable because ``arguments`` is a
    dict.
    """

    limits: LoopLimits
    turns: int = 0
    tool_calls: int = 0
    seen_proposals: set[tuple[str, str]] = field(default_factory=set)
    #: Proposals the model has already been told it repeated. A repeat is worth
    #: one correction before it counts as a loop -- see ``warn_repeat``.
    warned_proposals: set[tuple[str, str]] = field(default_factory=set)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def out_of_time(self) -> bool:
        return self.elapsed >= self.limits.max_seconds

    def turns_exhausted(self) -> bool:
        return self.turns >= self.limits.max_turns

    def tool_calls_exhausted(self) -> bool:
        return self.tool_calls >= self.limits.max_tool_calls

    def record_turn(self) -> None:
        """Count one model round trip. A CLARIFICATION counts, like any turn."""
        self.turns += 1

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    @staticmethod
    def proposal_key(proposal: ToolProposal) -> tuple[str, str]:
        """Canonical identity of a proposal: its name plus its sorted arguments.

        ``default=str`` keeps arguments containing ``datetime``, ``UUID`` or
        ``Decimal`` from raising ``TypeError`` during canonicalization.
        """
        return (
            proposal.tool_name,
            json.dumps(proposal.arguments, sort_keys=True, default=str),
        )

    def register_proposal(self, proposal: ToolProposal) -> bool:
        """Record ``proposal``; return ``True`` if an identical one was already seen."""
        key = self.proposal_key(proposal)
        if key in self.seen_proposals:
            return True
        self.seen_proposals.add(key)
        return False

    def warn_repeat(self, proposal: ToolProposal) -> bool:
        """``True`` the first time this exact repeat is reported, ``False`` after.

        A model that asks for the same call twice is usually stuck rather than
        looping -- it wanted something the result did not contain and looked
        again. Telling it so is more useful than ending the turn, so the first
        repeat buys a correction. A second one is a loop, and the caller stops.
        """
        key = self.proposal_key(proposal)
        if key in self.warned_proposals:
            return False
        self.warned_proposals.add(key)
        return True


@dataclass
class LoopResult:
    """The outcome of one ``handle_turn`` call.

    ``final_text`` carries the final answer *or* the clarifying question -- check
    ``stopped_reason`` to tell which. It is ``None`` for every forced stop.

    ``stopped_reason`` is one of ``"final_answer"``, ``"clarification_needed"``,
    ``"max_turns_exceeded"``, ``"max_tool_calls_exceeded"``,
    ``"max_seconds_exceeded"``, ``"repeated_call_detected"``, or
    ``f"guardrail_violation:{stage}"`` for a stage in
    ``{input, context, tool, output}``.

    ``violation_reason`` carries the ``GuardrailViolation.reason`` when a
    guardrail ended the turn, and is ``None`` otherwise. It exists because
    ``stopped_reason`` names only the *stage*: a caller that wants to tell a user
    why they were refused would otherwise have to wrap every guardrail itself,
    since a violation aborts before any ``ToolResult`` is recorded.
    """

    final_text: str | None
    task_frame: TaskFrame | None
    tool_results: list[ToolResult]
    turns_used: int
    stopped_reason: str
    violation_reason: str | None = None


class LoopController:
    """Owns the turn: context, prompt, tools, guardrails, budget, write-back."""

    def __init__(
        self,
        context_engine: ContextEngine,
        prompt_compiler: PromptCompiler,
        tool_gateway: ToolGateway,
        guardrails: GuardrailPipeline,
        memory: MemoryStore,
        call_llm: CallLLM,
        output_formatter: OutputFormatter | None = None,
    ) -> None:
        self.context_engine = context_engine
        self.prompt_compiler = prompt_compiler
        self.tool_gateway = tool_gateway
        self.guardrails = guardrails
        self.memory = memory
        self.call_llm = call_llm
        self.output_formatter = output_formatter

    async def handle_turn(
        self,
        session_id: str,
        message: str,
        current_frame: TaskFrame | None = None,
        allowed_tools: list[str] | None = None,
        output_requirement: str = DEFAULT_OUTPUT_REQUIREMENT,
        render_channel: RenderChannel = RenderChannel.PLAIN_TEXT,
        limits: LoopLimits | None = None,
    ) -> LoopResult:
        """Run one bounded turn to one of its exits.

        Not coroutine-safe for a single session: if concurrent calls for the same
        ``session_id`` are possible, the caller serializes them (e.g. one
        ``asyncio.Lock`` per session).
        """
        budget = (limits or LoopLimits()).new_budget()
        tool_results: list[ToolResult] = []
        frame: TaskFrame | None = current_frame

        # One wrapper for every stage: GuardrailViolation.stage already identifies
        # the origin, so a single handler covers all four call sites with no risk
        # of missing one.
        try:
            await self.guardrails.run(GuardrailStage.INPUT, message)

            context = await self.context_engine.build(session_id, message, current_frame)
            # The frame build() returns IS the frame for the rest of the turn.
            frame = context.task_frame

            await self.guardrails.run(GuardrailStage.CONTEXT, context)

            tools = self.tool_gateway.allowed_tools(allowed_tools)

            while True:
                if budget.out_of_time():
                    return await self._forced_stop(
                        session_id, "max_seconds_exceeded", frame, tool_results, budget
                    )
                if budget.turns_exhausted():
                    return await self._forced_stop(
                        session_id, "max_turns_exceeded", frame, tool_results, budget
                    )

                prompt = self.prompt_compiler.compile(
                    context, tools, message, output_requirement
                )
                step = await self.call_llm(prompt)
                budget.record_turn()

                if step.step_type == TurnStepType.TOOL_CALL:
                    if budget.tool_calls_exhausted():
                        return await self._forced_stop(
                            session_id,
                            "max_tool_calls_exceeded",
                            frame,
                            tool_results,
                            budget,
                        )

                    proposal = ToolProposal(
                        tool_name=step.tool_name or "",
                        arguments=step.tool_arguments or {},
                    )

                    if budget.register_proposal(proposal):
                        # Correct the model once rather than ending the turn on
                        # the spot. The result it is asking for again is already
                        # in this context, so saying so lets it answer from what
                        # it has; killing the turn leaves the caller with
                        # ``final_text=None`` and a user with nothing. The note
                        # goes into episodic context, the same channel every
                        # other piece of evidence arrives through. No tool runs,
                        # so no budget is spent, and ``max_turns`` still bounds
                        # the whole exchange.
                        if budget.warn_repeat(proposal):
                            context.episodic.append(
                                f"tool {proposal.tool_name} was already called this "
                                f"turn with exactly these arguments; its result is "
                                f"above. Answer from it, or do something different -- "
                                f"calling it again will return the same thing."
                            )
                            continue
                        return await self._forced_stop(
                            session_id,
                            "repeated_call_detected",
                            frame,
                            tool_results,
                            budget,
                        )

                    await self.guardrails.run(GuardrailStage.TOOL, proposal)

                    try:
                        result = await self.tool_gateway.execute(
                            session_id, proposal, allowed_tools
                        )
                    except ToolError as exc:
                        # A policy refusal is evidence too: record it and let the
                        # model adapt. The budget still bounds the retrying.
                        result = ToolResult(
                            proposal_id=proposal.id,
                            tool_name=proposal.tool_name,
                            ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )

                    budget.record_tool_call()
                    tool_results.append(result)

                    # Write-as-you-go, before anything can force a stop: a tool
                    # that caused a real side effect must already be recorded.
                    summary = _describe(result)
                    await self.memory.remember(
                        session_id,
                        MemoryKind.EPISODIC,
                        summary,
                        event="tool_call",
                        tool=result.tool_name,
                        ok=result.ok,
                        proposal_id=result.proposal_id,
                    )
                    # Folded into episodic context so the next iteration of THIS
                    # loop sees the evidence through the normal history section.
                    context.episodic.append(summary)
                    continue

                if step.step_type == TurnStepType.CLARIFICATION:
                    question = step.text or ""
                    # No OUTPUT guardrail: a question is not a claim, and there is
                    # no evidence to ground it against. No formatting either --
                    # the calling UI decides how to present a question.
                    await self._record_turn_completion(session_id, message, question)
                    return LoopResult(
                        final_text=question,
                        task_frame=frame,
                        tool_results=tool_results,
                        turns_used=budget.turns,
                        stopped_reason="clarification_needed",
                    )

                # FINAL
                answer = step.text or ""
                await self.guardrails.run(GuardrailStage.OUTPUT, (answer, list(tool_results)))
                if self.output_formatter is not None:
                    answer = await self.output_formatter.format(answer, render_channel)
                await self._record_turn_completion(session_id, message, answer)
                return LoopResult(
                    final_text=answer,
                    task_frame=frame,
                    tool_results=tool_results,
                    turns_used=budget.turns,
                    stopped_reason="final_answer",
                )

        except GuardrailViolation as violation:
            # Guardrails fail safe: the caller gets a clean LoopResult, not an
            # exception to handle at the edge of its own process.
            reason = f"guardrail_violation:{violation.stage.value}"
            await self.memory.remember(
                session_id,
                MemoryKind.EPISODIC,
                f"Turn terminated by {violation.stage.value} guardrail: {violation.reason}",
                event="guardrail_violation",
                stage=violation.stage.value,
                reason=violation.reason,
                stopped_reason=reason,
            )
            return LoopResult(
                final_text=None,
                task_frame=frame,
                tool_results=tool_results,
                turns_used=budget.turns,
                stopped_reason=reason,
                violation_reason=violation.reason,
            )

    async def _forced_stop(
        self,
        session_id: str,
        stopped_reason: str,
        frame: TaskFrame | None,
        tool_results: list[ToolResult],
        budget: LoopBudget,
    ) -> LoopResult:
        """Abandon the turn on a budget breach: no answer, no guardrail, no format.

        Still writes an episodic record, so the audit trail shows the forced stop
        alongside the tool calls that already ran.
        """
        await self.memory.remember(
            session_id,
            MemoryKind.EPISODIC,
            f"Turn terminated without an answer ({stopped_reason}) after "
            f"{budget.turns} turn(s) and {budget.tool_calls} tool call(s).",
            event="forced_stop",
            stopped_reason=stopped_reason,
        )
        return LoopResult(
            final_text=None,
            task_frame=frame,
            tool_results=tool_results,
            turns_used=budget.turns,
            stopped_reason=stopped_reason,
        )

    async def _record_turn_completion(
        self, session_id: str, message: str, response: str
    ) -> None:
        """One consolidated episodic record per resolved turn.

        Human-readable, alongside the granular per-tool-call records.
        """
        await self.memory.remember(
            session_id,
            MemoryKind.EPISODIC,
            f"user: {message}\nassistant: {response}",
            event="turn_complete",
        )


def _describe(result: ToolResult) -> str:
    """Render a ToolResult for episodic memory and the next prompt."""
    if result.ok:
        return f"tool {result.tool_name} succeeded: {result.data}"
    return f"tool {result.tool_name} failed: {result.error}"
