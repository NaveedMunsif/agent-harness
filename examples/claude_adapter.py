"""Connect the harness to a real model: Claude, via the Anthropic SDK.

The other examples inject a fake ``call_llm`` so they run with no API key. This
one makes real API calls, and exists to show the only piece the library
deliberately leaves to you: the translation layer.

    CompiledPrompt  ->  Anthropic Messages API  ->  LLMTurnStep

Three mappings do all the work:

* a ``tool_use`` block          -> ``TurnStepType.TOOL_CALL``
* a call to the ``ask_user`` tool -> ``TurnStepType.CLARIFICATION``
* a text-only response          -> ``TurnStepType.FINAL``

``ask_user`` is a sentinel: it is declared to the model as a normal tool but is
never registered with the :class:`ToolGateway`, so the adapter intercepts it and
the harness never tries to execute it. That is what lets the model *ask* through
the same native tool-calling channel it uses to *act*, instead of us pattern
matching on prose.

Note what the adapter closes over. ``CompiledPrompt`` renders tools into a text
section for the model to read, so the callable alone cannot rebuild a JSON
schema -- ``make_call_llm`` therefore takes the same ``list[Tool]`` you hand the
gateway and derives the schemas from it. One list, two consumers.

``anthropic`` is not a dependency of this library; it is needed only to run this
file:

    pip install agent-harnessed anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      # setx on Windows
    python examples/claude_adapter.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from anthropic import AsyncAnthropic

from agent_harness import (
    CompiledPrompt,
    ContextEngine,
    Guardrail,
    GuardrailPipeline,
    GuardrailStage,
    GuardrailViolation,
    InMemoryBackend,
    LLMTurnStep,
    LoopController,
    LoopLimits,
    MemoryStore,
    PlainTextFormatter,
    PromptCompiler,
    Tool,
    ToolGateway,
    ToolResult,
    TurnStepType,
)

MODEL = "claude-opus-5"

# Declared to the model, never registered with the gateway. A call to it is a
# question, not an action.
ASK_USER = "ask_user"

ASK_USER_SCHEMA: dict[str, Any] = {
    "name": ASK_USER,
    "description": (
        "Ask the user one question and stop. Use this when a required detail is "
        "missing and you cannot proceed without it. Do not guess."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The single question to put to the user.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}


# --- the domain: same order lookup as the README, so the shape is familiar ----
ORDERS = {"A-1001": {"status": "shipped", "carrier": "UPS", "eta": "2026-08-03"}}


async def lookup_order(session_id: str, arguments: dict) -> dict:
    order_id = arguments["order_id"]
    if order_id not in ORDERS:
        raise KeyError(f"no such order: {order_id}")
    return {"order_id": order_id, **ORDERS[order_id], "internal_note": "flagged for QA"}


lookup = Tool(
    name="lookup_order",
    description="Look up an order's status, carrier and ETA by its id (e.g. A-1001).",
    parameters=["order_id"],
    execute=lookup_order,
    read_only=True,
    max_calls_per_session=5,
    redact_fields=["internal_note"],
)


def _schema_for(tool: Tool) -> dict[str, Any]:
    """Derive an Anthropic tool schema from a harness ``Tool``.

    ``Tool.parameters`` is a list of names with no types attached, so every
    parameter is declared as a string and all of them are required -- the most
    faithful schema the library's own metadata supports. A tool needing richer
    types would carry its own schema; this keeps the two definitions in sync for
    the common case instead of duplicating them by hand.
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": {
                name: {"type": "string"} for name in tool.parameters
            },
            "required": list(tool.parameters),
            "additionalProperties": False,
        },
    }


def make_call_llm(client: AsyncAnthropic, tools: list[Tool], model: str = MODEL):
    """Build the ``async (CompiledPrompt) -> LLMTurnStep`` callable the loop wants."""
    schemas = [_schema_for(tool) for tool in tools] + [ASK_USER_SCHEMA]

    async def call_llm(prompt: CompiledPrompt) -> LLMTurnStep:
        # The whole compiled prompt goes in as one user turn: `render()` already
        # leads with the ROLE block, and it is the library's own contract for
        # flattening. A production adapter would lift `prompt.role` into the
        # `system` field instead, so the stable part of the prefix can be cached
        # independently of the volatile sections beneath it.
        response = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            # Claude Opus 5's safety classifiers can decline a request; this
            # re-runs a declined one on Anthropic's recommended fallback rather
            # than handing you an empty response.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt.render()}],
            tools=schemas,
        )

        # Check this before touching `content`: on a refusal the list is empty
        # (declined before any output) or holds a partial answer worth discarding.
        if response.stop_reason == "refusal":
            raise RuntimeError(
                "model declined the request "
                f"(category={getattr(response.stop_details, 'category', None)})"
            )

        text_parts: list[str] = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == ASK_USER:
                    return LLMTurnStep(
                        step_type=TurnStepType.CLARIFICATION,
                        text=block.input.get("question", "Could you clarify?"),
                    )
                return LLMTurnStep(
                    step_type=TurnStepType.TOOL_CALL,
                    tool_name=block.name,
                    tool_arguments=dict(block.input),
                )
            if block.type == "text":
                text_parts.append(block.text)
            # thinking blocks and anything else are not part of the mapping

        answer = "\n".join(part for part in text_parts if part).strip()
        if not answer:
            # LLMTurnStep rejects a FINAL with no text, so fail here where the
            # cause is still visible rather than inside the validator.
            raise RuntimeError(
                f"no tool call and no text in response (stop_reason={response.stop_reason})"
            )
        return LLMTurnStep(step_type=TurnStepType.FINAL, text=answer)

    return call_llm


# --- the rest is ordinary harness wiring, unchanged from the fake-model case --
def require_evidence(payload: tuple[str, list[ToolResult]]) -> None:
    """No shipping claim without a successful lookup behind it."""
    _text, results = payload
    if not any(result.ok for result in results):
        raise GuardrailViolation(GuardrailStage.OUTPUT, "claim with no tool evidence")


class SupportContext(ContextEngine):
    async def extract_intent(self, message: str) -> tuple[str, dict]:
        entities = {}
        for token in message.replace("?", " ").replace(",", " ").split():
            if token.upper() in ORDERS:
                entities["order_id"] = token.upper()
        return "track_order", entities


def build_controller(client: AsyncAnthropic) -> LoopController:
    memory = MemoryStore(InMemoryBackend())
    guardrails = GuardrailPipeline()
    guardrails.add(Guardrail("evidence", GuardrailStage.OUTPUT, require_evidence))

    return LoopController(
        context_engine=SupportContext(memory),
        prompt_compiler=PromptCompiler(
            role=(
                "You are a concise order-support agent. Look up an order before "
                "making any claim about it. If no order id is available, use the "
                f"{ASK_USER} tool rather than guessing."
            )
        ),
        tool_gateway=ToolGateway([lookup]),
        guardrails=guardrails,
        memory=memory,
        # The gateway and the adapter are given the same tool list.
        call_llm=make_call_llm(client, tools=[lookup]),
        output_formatter=PlainTextFormatter(),
    )


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY first")

    client = AsyncAnthropic()
    controller = build_controller(client)
    limits = LoopLimits(max_turns=6, max_tool_calls=3, max_seconds=120.0)

    # Turn 1: the order id is present, so the model should look it up and answer.
    resolved = await controller.handle_turn(
        "session-1", "Where is order A-1001?", limits=limits
    )
    print("stopped_reason :", resolved.stopped_reason)
    print("final_text     :", resolved.final_text)
    print("turns_used     :", resolved.turns_used)
    if resolved.tool_results:
        # internal_note came back [REDACTED] -- the gateway scrubbed it before
        # the harness, memory, or the next prompt could observe it.
        print("tool_data      :", resolved.tool_results[0].data)

    # Turn 2: no order id anywhere, so the model should ask instead of guessing.
    asked = await controller.handle_turn(
        "session-2", "Can you check on my order?", limits=limits
    )
    print()
    print("stopped_reason :", asked.stopped_reason)  # clarification_needed
    print("question       :", asked.final_text)

    # The caller owns the wait. Hand the same frame back with the answer and the
    # turn continues rather than starting over.
    answered = await controller.handle_turn(
        "session-2",
        "It's A-1001 - where is it?",
        current_frame=asked.task_frame,
        limits=limits,
    )
    print("stopped_reason :", answered.stopped_reason)
    print("final_text     :", answered.final_text)


if __name__ == "__main__":
    asyncio.run(main())
