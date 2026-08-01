# agent-harness

[![PyPI](https://img.shields.io/pypi/v/agent-harnessed.svg)](https://pypi.org/project/agent-harnessed/)
[![Python](https://img.shields.io/pypi/pyversions/agent-harnessed.svg)](https://pypi.org/project/agent-harnessed/)
[![License: MIT](https://img.shields.io/pypi/l/agent-harnessed.svg)](https://github.com/NaveedMunsif/agent-harness/blob/main/LICENSE)

```bash
pip install agent-harnessed
```

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

The distribution is published as `agent-harnessed`; the import name is `agent_harness`:

```python
from agent_harness import LoopController, ToolGateway
```

New here? [**GETTING_STARTED.md**](https://github.com/NaveedMunsif/agent-harness/blob/main/GETTING_STARTED.md)
walks from an empty folder to a real agent calling your own tool, in about ten minutes.

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

## A complete agent, wired to a real model

An order-support agent over Claude. Everything below is the whole program.

```bash
pip install agent-harnessed anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # setx on Windows
```

```python
import asyncio

from anthropic import AsyncAnthropic

from agent_harness import (
    ContextEngine, GuardrailPipeline, InMemoryBackend, LLMTurnStep, LoopController,
    MemoryStore, PromptCompiler, Tool, ToolGateway, TurnStepType,
)

ORDERS = {"A-1001": {"status": "shipped", "carrier": "UPS", "eta": "2026-08-03"}}


async def lookup_order(session_id: str, arguments: dict) -> dict:
    return ORDERS[str(arguments["order_id"]).upper()]


lookup = Tool(
    name="lookup_order",
    description="Look up an order's status by its id, e.g. A-1001.",
    parameters=["order_id"],
    execute=lookup_order,
    read_only=True,
)


def make_call_llm(client: AsyncAnthropic, tools: list[Tool]):
    """The only piece this library leaves to you: prompt in, LLMTurnStep out."""
    schemas = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in tool.parameters},
                "required": list(tool.parameters),
            },
        }
        for tool in tools
    ]

    async def call_llm(prompt) -> LLMTurnStep:
        response = await client.messages.create(
            model="claude-sonnet-5",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt.render()}],
            tools=schemas,
        )
        for block in response.content:
            if block.type == "tool_use":
                return LLMTurnStep(
                    step_type=TurnStepType.TOOL_CALL,
                    tool_name=block.name,
                    tool_arguments=dict(block.input),
                )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMTurnStep(step_type=TurnStepType.FINAL, text=text.strip())

    return call_llm


async def main() -> None:
    memory = MemoryStore(InMemoryBackend())
    controller = LoopController(
        context_engine=ContextEngine(memory),
        prompt_compiler=PromptCompiler(role="You are a concise order-support agent."),
        tool_gateway=ToolGateway([lookup]),
        guardrails=GuardrailPipeline(),
        memory=memory,
        call_llm=make_call_llm(AsyncAnthropic(), [lookup]),
    )

    result = await controller.handle_turn("session-1", "Where is order A-1001?")

    print(result.stopped_reason)        # final_answer
    print(result.final_text)            # Order A-1001 has shipped with UPS.
    print(result.tool_results[0].data)  # {'status': 'shipped', 'carrier': 'UPS', ...}


asyncio.run(main())
```

Two round trips to the model, and you wrote neither of them. The first returns a
`tool_use` block, so the gateway runs `lookup_order` and folds the result into the
next prompt as history; the second sees that evidence and answers. `handle_turn`
returns once, when `LoopController` decides the turn is over.

### The adapter is the whole integration

`make_call_llm` above is the entire model-specific surface. Three mappings do all
the work:

| The model does this | The adapter returns |
| --- | --- |
| emits a `tool_use` block | `TurnStepType.TOOL_CALL` |
| calls the `ask_user` tool | `TurnStepType.CLARIFICATION` |
| replies with text only | `TurnStepType.FINAL` |

`ask_user` is a **sentinel**: declared to the model as an ordinary tool, never
registered with the `ToolGateway`. The adapter intercepts it, so the model asks a
question through the same native tool-calling channel it uses to act — no parsing
prose to guess whether an answer was really a question. See
[`examples/claude_adapter.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/claude_adapter.py).

Tool schemas are derived from the same `list[Tool]` handed to the gateway, so what
the model is told about and what the gateway will actually run cannot drift apart.

Nothing here is Claude-specific beyond the SDK call. Any model with native tool
calling maps the same three ways; one without it needs the adapter to parse a
structured response instead.

## Examples

```bash
python examples/refund_agent.py --session naveed --trace
```

| File | What it shows |
| --- | --- |
| [`refund_agent.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/refund_agent.py) | **Start here.** Money on the line: three independent protections on a refund, an LLM intent classifier, and `--trace` printing every stage of the pipeline and where a turn stopped. |
| [`sqlite_agent.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/sqlite_agent.py) | The smallest real agent — a model, a database, two tools. No intents, no guardrails, no subclassing. |
| [`claude_adapter.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/claude_adapter.py) | The translation layer on its own, including the `ask_user` sentinel and refusal handling. |
| [`order_tracking.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/order_tracking.py) | The loop with **no API key** — a local function stands in for the model, so you can watch redaction, memory write-back and a forced stop for free. |

In none of them does the model write SQL. It names a tool and supplies a value;
every statement is parameterized and lives in your code.

## Clarification: a return, not a suspension

When the model needs to ask something, `handle_turn` **returns immediately**. It does
not suspend, block, or hold an open coroutine. The caller owns the wait — print the
question, collect an answer over whatever transport it has (HTTP request, websocket,
SMS, tomorrow), then call `handle_turn` again with that answer and the frame it got
back. The controller keeps no state between calls beyond what you pass in.

```python
asked = await controller.handle_turn("session-1", "Can you check on my order?")
# asked.stopped_reason == "clarification_needed"
# asked.final_text     == "Which order number should I look up?"

# ... your transport waits here, for however long it takes ...

resolved = await controller.handle_turn(
    "session-1",
    "It's A-1001 - where is it?",
    current_frame=asked.task_frame,      # hand the same frame back
)
# resolved.stopped_reason == "final_answer"
```

Because the intent matches and no entity conflicts, that second turn is a
**continuation**: `ContextEngine.build` extends the frame via `merged_with()` and
`is_pivot` is `False`. Ask about a *different* order and the same intent now carries a
conflicting `order_id`, so the frame is rebuilt via `TaskFrame.fresh()` with
`is_pivot=True` — dropping the abandoned task's plan and in-flight tool ids.

See [`examples/clarification_flow.py`](https://github.com/NaveedMunsif/agent-harness/blob/main/examples/clarification_flow.py) for all three
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

99 tests covering tool authorization/redaction/rate-limiting, pivot-vs-continuation
semantics and the `fresh()`/`merged_with()` branch in `ContextEngine.build`, all three
loop exits, every forced stop (including that already-executed `ToolResult`s survive
one), guardrail violations converting to clean results at each stage, `LLMTurnStep`
validation rejecting malformed steps, repeated-call detection over non-JSON-native
arguments, the per-stage guardrail payload contract, and that the topic-change note
reaches the prompt on a real pivot but never on a session's opening message.
