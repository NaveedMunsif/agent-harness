# agent-harness

A clean, async-first Python library that provides the **runtime environment around an LLM**.

```
Agent = LLM + Context + Memory + Tools + Control Flow + Guardrails + State
```

The LLM reasons and proposes actions. The harness decides what it sees, validates and
executes what it asks for, controls how long the loop runs, formats the result for its
delivery channel, and hands back the final answer.

The model client is **not** part of this library. You inject an async
`(CompiledPrompt) -> LLMTurnStep` callable, so `agent-harness` depends on `pydantic>=2.0`
and nothing else — no LLM SDK, no templating engine, no vector store.

## Install

```bash
pip install agent-harnessed
```

The distribution is published as `agent-harnessed`; the import name is `agent_harness`:

```python
from agent_harness import LoopController, ToolGateway
```

For local development:

```bash
pip install -e ".[dev]"
```

## Architecture

| Module | Responsibility | Key types |
| --- | --- | --- |
| `state` | Task continuity. Explicit pivot-vs-continuation, never inferred from chat history. | `TaskFrame` |
| `memory` | Typed recall: durable facts, what happened, how to do things. Backends are injected. | `MemoryKind`, `MemoryRecord`, `MemoryBackend`, `InMemoryBackend`, `MemoryStore` |
| `context` | Decides what the model gets to see this turn, and owns the pivot decision. | `ContextEngine`, `ScopedContext` |
| `prompt` | Versioned prompt assembly — a compiled artifact, not scattered f-strings. | `PromptCompiler`, `CompiledPrompt` |
| `tools` | The only path from a proposal to a real side effect: allowlist, authorize, rate-limit, redact. | `Tool`, `ToolGateway`, `ToolProposal`, `ToolResult` |
| `guardrails` | Composable checks with a typed payload per stage. | `GuardrailPipeline`, `Guardrail`, `GuardrailStage`, `GuardrailViolation` |
| `output` | Channel-aware rendering of the final answer. Injected like the model client. | `RenderChannel`, `OutputFormatter`, `PlainTextFormatter` |
| `loop` | The bounded reasoning loop. Owns every exit decision and all memory write-back. | `LoopController`, `LoopLimits`, `LoopResult`, `LLMTurnStep`, `TurnStepType` |

### The loop has exactly three exits — and the LLM controls none of them

| `stopped_reason` | `final_text` | OUTPUT guardrail | Formatter |
| --- | --- | --- | --- |
| `final_answer` | the answer | ✅ runs | ✅ applied |
| `clarification_needed` | the question | ❌ skipped | ❌ skipped |
| `max_turns_exceeded` | `None` | ❌ skipped | ❌ skipped |
| `max_tool_calls_exceeded` | `None` | ❌ skipped | ❌ skipped |
| `max_seconds_exceeded` | `None` | ❌ skipped | ❌ skipped |
| `repeated_call_detected` | `None` | ❌ skipped | ❌ skipped |
| `guardrail_violation:{input\|context\|tool\|output}` | `None` | — | ❌ skipped |

A model can emit `final` on every turn; the loop still ends only when `LoopController`
says so. The OUTPUT guardrail is skipped for a clarification because **a question is not
a claim** — there is nothing to ground it against.

### Guardrail payload contract

Each stage's `check()` receives exactly one type, so guardrails are written against
concrete objects rather than an untyped blob:

| Stage | Payload |
| --- | --- |
| `INPUT` | `str` — the raw incoming user message |
| `CONTEXT` | `ScopedContext` — assembled context, pre-compile |
| `TOOL` | `ToolProposal` — proposed tool + args, pre-execution |
| `OUTPUT` | `tuple[str, list[ToolResult]]` — final text plus every `ToolResult` from the turn |

A violation never reaches your process: `LoopController` catches `GuardrailViolation`,
records it, and returns a clean `LoopResult` with `final_text=None`. Output guardrails
must fail safe when evidence is missing, not crash the caller.

## Order tracking: tool call → final answer

Runnable as-is — no API key, no network.

```python
import asyncio

from agent_harness import (
    ContextEngine, Guardrail, GuardrailPipeline, GuardrailStage, GuardrailViolation,
    InMemoryBackend, LLMTurnStep, LoopController, LoopLimits, MemoryStore,
    PlainTextFormatter, PromptCompiler, Tool, ToolGateway, TurnStepType,
)

ORDERS = {"A-1001": {"status": "shipped", "carrier": "UPS", "eta": "2026-08-03"}}


# --- a tool: the gateway authorizes, rate-limits and redacts around this -------
async def lookup_order(session_id: str, arguments: dict) -> dict:
    order_id = arguments["order_id"]
    # internal_note is returned by the backend but never allowed out.
    return {"order_id": order_id, **ORDERS[order_id], "internal_note": "flagged for QA"}


lookup = Tool(
    name="lookup_order",
    description="Look up an order's status by id.",
    parameters=["order_id"],
    execute=lookup_order,
    read_only=True,
    max_calls_per_session=5,
    redact_fields=["internal_note"],
)


# --- a guardrail: no shipping claims without a successful lookup ---------------
def require_evidence(payload: tuple) -> None:
    text, results = payload
    if not any(result.ok for result in results):
        raise GuardrailViolation(GuardrailStage.OUTPUT, "claim with no tool evidence")


# --- context: resolving the order number is what makes pivots precise ----------
class SupportContext(ContextEngine):
    async def extract_intent(self, message: str) -> tuple[str, dict]:
        entities = {"order_id": "A-1001"} if "A-1001" in message else {}
        return "track_order", entities


# --- your model client goes here: async (CompiledPrompt) -> LLMTurnStep --------
async def call_llm(prompt) -> LLMTurnStep:
    if "tool lookup_order succeeded" in prompt.sections.get("history", ""):
        return LLMTurnStep(step_type=TurnStepType.FINAL, text="Order A-1001 has shipped via UPS.")
    if "order_id" not in prompt.sections.get("task_frame", ""):
        return LLMTurnStep(
            step_type=TurnStepType.CLARIFICATION, text="Which order number should I look up?"
        )
    return LLMTurnStep(
        step_type=TurnStepType.TOOL_CALL,
        tool_name="lookup_order",
        tool_arguments={"order_id": "A-1001"},
    )


def build_controller(backend: InMemoryBackend) -> LoopController:
    memory = MemoryStore(backend)
    guardrails = GuardrailPipeline()
    guardrails.add(Guardrail("evidence", GuardrailStage.OUTPUT, require_evidence))

    return LoopController(
        context_engine=SupportContext(memory),
        prompt_compiler=PromptCompiler(role="You are a concise order-support agent."),
        tool_gateway=ToolGateway([lookup]),
        guardrails=guardrails,
        memory=memory,
        call_llm=call_llm,
        output_formatter=PlainTextFormatter(),
    )


async def main() -> None:
    backend = InMemoryBackend()
    controller = build_controller(backend)

    result = await controller.handle_turn(
        "session-1", "Where is order A-1001?", limits=LoopLimits(max_tool_calls=2)
    )

    print(result.stopped_reason)              # final_answer
    print(result.final_text)                  # Order A-1001 has shipped via UPS.
    print(result.turns_used)                  # 2
    print(result.tool_results[0].data)        # ... 'internal_note': '[REDACTED]'

    for record in backend.all_records():
        print(record.metadata.get("event"), "->", record.content.replace("\n", " | "))


asyncio.run(main())
```

Output:

```
final_answer
Order A-1001 has shipped via UPS.
2
{'order_id': 'A-1001', 'status': 'shipped', 'carrier': 'UPS', 'eta': '2026-08-03', 'internal_note': '[REDACTED]'}
tool_call -> tool lookup_order succeeded: {'order_id': 'A-1001', 'status': 'shipped', 'carrier': 'UPS', 'eta': '2026-08-03', 'internal_note': '[REDACTED]'}
turn_complete -> user: Where is order A-1001? | assistant: Order A-1001 has shipped via UPS.
```

Two things worth noticing. The redacted field never reaches memory *or* the next
prompt — redaction happens inside the gateway, before anything can observe the raw
value. And there is no separate tool-history channel: the `ToolResult` is folded into
`context.episodic`, so iteration 2 sees the evidence through the ordinary history
section.

See [`examples/order_tracking.py`](examples/order_tracking.py) for the fuller version,
including a forced stop.

## Clarification: a return, not a suspension

When the model needs to ask something, `handle_turn` **returns immediately**. It does
not suspend, block, or hold an open coroutine. The caller owns the wait — print the
question, collect an answer over whatever transport it has (HTTP request, websocket,
SMS, tomorrow), then call `handle_turn` again with that answer and the frame it got
back. The controller keeps no state between calls beyond what you pass in.

Continuing with the same `controller` as above — the message names no order, so the
frame carries no `order_id` and the model asks instead of guessing:

```python
controller = build_controller(InMemoryBackend())

asked = await controller.handle_turn("session-1", "Can you check on my order?")

assert asked.stopped_reason == "clarification_needed"
assert asked.final_text == "Which order number should I look up?"   # plain text, unformatted
assert asked.tool_results == []

# ... your transport waits here, for however long it takes ...

resolved = await controller.handle_turn(
    "session-1",
    "It's A-1001 - where is it?",
    current_frame=asked.task_frame,      # hand the same frame back
)

assert resolved.stopped_reason == "final_answer"
```

Because the intent matches and no entity conflicts, that second turn is a
**continuation**: `ContextEngine.build` extends the frame via `merged_with()` and
`is_pivot` is `False`. Ask about a *different* order and the same intent now carries a
conflicting `order_id`, so the frame is rebuilt via `TaskFrame.fresh()` with
`is_pivot=True` — dropping the abandoned task's plan and in-flight tool ids.

See [`examples/clarification_flow.py`](examples/clarification_flow.py) for all three
turns end to end.

## What the episodic record looks like

Memory write-back is a correctness requirement, not an optimization. Tools that caused
real side effects are recorded **the moment they complete**, so nothing about a forced
stop can erase them.

| Exit | Episodic records written |
| --- | --- |
| tool call → final | one `tool_call` per executed tool, then one `turn_complete` holding the user message and the final answer |
| clarification | one `turn_complete` holding the user message and the question — no tool records, since no tool ran |
| `max_*_exceeded` / `repeated_call_detected` | every `tool_call` that already ran, then one `forced_stop` naming the specific `stopped_reason` |
| `guardrail_violation:*` | every `tool_call` that already ran, then one `guardrail_violation` naming the stage and reason |

So a forced stop after two lookups leaves `[tool_call, tool_call, forced_stop]` — no
answer was produced, but the audit trail shows both executions *and* why the turn was
cut short. A resolved turn never loses that granularity either: the consolidated
`turn_complete` record sits alongside the per-tool records, not instead of them.

## Out of scope

- **The model client.** Injected as an async callable. This library ships no model client.
- **Pause-in-place clarification.** Return-based only, as described above.
- **Long-term persistence.** Beyond in-process state, that is the caller's job — implement
  `MemoryBackend` against your own store.
- **Concurrency safety.** A single instance is not guaranteed coroutine-safe. If you might
  call `handle_turn` concurrently for the same session, serialize it yourself (one
  `asyncio.Lock` per session).

## Tests

```bash
pytest
```

94 tests covering tool authorization/redaction/rate-limiting, pivot-vs-continuation
semantics and the `fresh()`/`merged_with()` branch in `ContextEngine.build`, all three
loop exits, every forced stop (including that already-executed `ToolResult`s survive
one), guardrail violations converting to clean results at each stage, `LLMTurnStep`
validation rejecting malformed steps, repeated-call detection over non-JSON-native
arguments, and the per-stage guardrail payload contract.
