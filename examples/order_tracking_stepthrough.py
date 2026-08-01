"""``order_tracking.py``, paused at every step so you can watch the loop run.

    python examples/order_tracking_stepthrough.py

Same agent, same output -- but it prints the compiled prompt each iteration and
waits for Enter between stages, so the order of events is something you observe
rather than infer. Useful once, when the loop is still abstract.

No API key required: the "model" is a local function that reads the compiled
prompt and returns a typed LLMTurnStep, which is exactly the contract a real
client satisfies.
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import sys
from typing import Any

# Lets the example run straight from a checkout, before `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agent_harness import (  # noqa: E402
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
    MemoryKind,
    MemoryStore,
    PlainTextFormatter,
    PromptCompiler,
    RenderChannel,
    Tool,
    ToolGateway,
    ToolResult,
    TurnStepType,
)

SESSION = "demo-session"
ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "UPS", "eta": "2026-08-03"},
    "B-2002": {"status": "processing", "carrier": None, "eta": "2026-08-09"},
}
ORDER_ID = re.compile(r"\b([A-Z]-\d{4})\b")


def pause(label: str) -> None:
    input(f"\n--- [step: {label}] press Enter to continue ---")


class SupportContextEngine(ContextEngine):
    """Real intent extraction: keyword routing plus order-number resolution.

    Resolving the order number into ``entities`` is what lets the harness tell
    "still talking about A-1001" from "now asking about B-2002" -- the latter is a
    pivot even though the intent is identical.
    """

    async def extract_intent(self, message: str) -> tuple[str, dict[str, Any]]:
        lowered = message.lower()
        if any(word in lowered for word in ("refund", "money back")):
            intent = "refund_order"
        elif any(word in lowered for word in ("track", "where", "status", "shipped", "check")):
            intent = "track_order"
        else:
            intent = "general_enquiry"

        entities: dict[str, Any] = {}
        found = ORDER_ID.search(message.upper())
        if found:
            entities["order_id"] = found.group(1)
        return intent, entities


async def authorize_lookup(session_id: str, arguments: dict[str, Any]) -> bool:
    """Per-session authorization. A real one would check the session's account."""
    return bool(session_id)


async def lookup_order(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    order_id = str(arguments.get("order_id", "")).upper()
    print(f"    [tool] lookup_order(order_id={order_id!r}) executing...")
    order = ORDERS.get(order_id)
    if order is None:
        raise LookupError(f"no such order: {order_id}")
    # internal_note is returned by the backend but never allowed out: it is named
    # in redact_fields, so the gateway scrubs it before anyone can see it.
    result = {"order_id": order_id, **order, "internal_note": "flagged for QA audit"}
    print(f"    [tool] raw result (pre-redaction): {result}")
    pause("tool executed")
    return result


lookup_order_tool = Tool(
    name="lookup_order",
    description="Look up the shipping status of an order by its id.",
    parameters=["order_id"],
    execute=lookup_order,
    authorize=authorize_lookup,
    read_only=True,
    max_calls_per_session=5,
    redact_fields=["internal_note"],
)


def reject_oversized_input(message: str) -> None:
    if len(message) > 2_000:
        raise GuardrailViolation(GuardrailStage.INPUT, "message too long")


def require_tool_evidence(payload: tuple[str, list[ToolResult]]) -> None:
    """The point of the OUTPUT stage: no shipping claims without a lookup.

    Receives the final text *and* every ToolResult from the turn, so it can check
    the answer against the evidence that was actually gathered.
    """
    text, results = payload
    claims_status = any(word in text.lower() for word in ("shipped", "processing", "eta"))
    if claims_status and not any(result.ok for result in results):
        raise GuardrailViolation(GuardrailStage.OUTPUT, "status claim with no successful lookup")


async def call_llm(prompt: CompiledPrompt) -> LLMTurnStep:
    """Stand-in for a model client: async (CompiledPrompt) -> LLMTurnStep.

    It reads the prompt the harness compiled, which is the whole point -- the
    harness decides what the model can see, and this reads only that.
    """
    history = prompt.sections.get("history", "")
    order_id = _extract(prompt.sections.get("task_frame", ""), "order_id")

    print("\n    [llm] compiled prompt sections seen this iteration:")
    for name, text in prompt.sections.items():
        print(f"      {name}: {text!r}")

    # Second iteration: the tool result was folded into episodic history, so the
    # evidence is right there in the prompt. Answer from it -- but only from
    # evidence about *this* order. History spans earlier turns too, and grounding
    # a claim in a previous order's lookup is exactly what the OUTPUT guardrail
    # exists to catch.
    evidence = _evidence_for(history, order_id)
    if evidence:
        status = _extract(evidence, "status")
        carrier = _extract(evidence, "carrier")
        eta = _extract(evidence, "eta")
        if status == "shipped":
            text = f"Order {order_id} has shipped with {carrier} and should arrive {eta}."
        else:
            text = f"Order {order_id} is still {status}; estimated delivery is {eta}."
        print(f"    [llm] decision: FINAL -> {text!r}")
        pause("llm decided FINAL")
        return LLMTurnStep(step_type=TurnStepType.FINAL, text=text)

    # First iteration: no usable evidence yet, so ask for a lookup.
    if order_id:
        print(f"    [llm] decision: TOOL_CALL -> lookup_order(order_id={order_id!r})")
        pause("llm decided TOOL_CALL")
        return LLMTurnStep(
            step_type=TurnStepType.TOOL_CALL,
            tool_name="lookup_order",
            tool_arguments={"order_id": order_id},
        )
    print("    [llm] decision: CLARIFICATION (no order id found)")
    pause("llm decided CLARIFICATION")
    return LLMTurnStep(
        step_type=TurnStepType.CLARIFICATION,
        text="Which order number should I look up?",
    )


def _evidence_for(history: str, order_id: str) -> str:
    """The most recent successful lookup line for ``order_id``, if any."""
    if not order_id:
        return ""
    matches = [
        line
        for line in history.splitlines()
        if "tool lookup_order succeeded" in line and f"'order_id': '{order_id}'" in line
    ]
    return matches[-1] if matches else ""


def _extract(blob: str, key: str) -> str:
    """Pull ``key=value`` or ``'key': 'value'`` out of a rendered prompt section."""
    for pattern in (rf"{key}=([^\s,]+)", rf"'{key}': '([^']*)'"):
        found = re.search(pattern, blob)
        if found:
            return found.group(1)
    return ""


def build_controller(backend: InMemoryBackend) -> LoopController:
    memory = MemoryStore(backend)
    guardrails = GuardrailPipeline()
    guardrails.add(Guardrail("input-size", GuardrailStage.INPUT, reject_oversized_input))
    guardrails.add(Guardrail("evidence", GuardrailStage.OUTPUT, require_tool_evidence))

    return LoopController(
        context_engine=SupportContextEngine(memory),
        prompt_compiler=PromptCompiler(
            role="You are a concise order-support agent.", version="1.0.0"
        ),
        tool_gateway=ToolGateway([lookup_order_tool]),
        guardrails=guardrails,
        memory=memory,
        call_llm=call_llm,
        output_formatter=PlainTextFormatter(),
    )


async def main() -> None:
    backend = InMemoryBackend()
    controller = build_controller(backend)

    # Seed the durable knowledge the agent should already have.
    await MemoryStore(backend).remember(
        SESSION, MemoryKind.SEMANTIC, "Customer is on the Pro plan; free returns apply."
    )

    print("=" * 70)
    print("TOOL CALL -> FINAL")
    print("=" * 70)

    message = "Hi, where is my order A-1001?"
    print(f"\nuser: {message}\n")
    pause("about to call handle_turn (this runs the whole loop below)")

    result = await controller.handle_turn(
        SESSION,
        message,
        current_frame=None,
        output_requirement="Answer in one sentence. Cite only what the tools returned.",
        render_channel=RenderChannel.PLAIN_TEXT,
        limits=LoopLimits(max_turns=6, max_tool_calls=4, max_seconds=30.0),
    )

    print(f"agent: {result.final_text}\n")
    print(f"stopped_reason : {result.stopped_reason}")
    print(f"turns_used     : {result.turns_used}")
    print(f"intent         : {result.task_frame.intent}")
    print(f"entities       : {result.task_frame.entities}")
    for tool_result in result.tool_results:
        print(f"tool           : {tool_result.tool_name} ok={tool_result.ok} {tool_result.data}")
    pause("turn finished, about to inspect episodic memory")

    print("\n--- episodic memory after the turn " + "-" * 34)
    # Two records: the granular tool call (written the moment it completed) and
    # the consolidated turn summary. internal_note is absent from both.
    for record in backend.all_records():
        if record.kind is not MemoryKind.EPISODIC:
            continue
        event = record.metadata.get("event")
        body = record.content.replace("\n", " | ")
        print(f"[{event}] {body}")
    pause("about to run the forced-stop scenario (max_tool_calls=0)")

    print("\n--- a forced stop still records what already ran " + "-" * 21)
    # max_tool_calls=0 stops before the first lookup; note there is no answer and
    # no turn_complete record, but the forced stop itself is on the audit trail.
    strict_backend = InMemoryBackend()
    strict = build_controller(strict_backend)
    forced = await strict.handle_turn(
        SESSION, "where is my order B-2002?", limits=LoopLimits(max_tool_calls=0)
    )
    print(f"final_text     : {forced.final_text}")
    print(f"stopped_reason : {forced.stopped_reason}")
    for record in strict_backend.all_records():
        print(f"[{record.metadata.get('event')}] {record.content}")


if __name__ == "__main__":
    asyncio.run(main())
