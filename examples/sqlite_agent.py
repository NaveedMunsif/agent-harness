"""Ask questions about a real database, in plain English.

The simplest complete agent: a real SQLite database, a real model, two tools.
No intent classification, no guardrails, no subclassing -- those are refinements,
and this file is here to show that none of them are required.

    pip install agent-harnessed anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      # setx on Windows
    python examples/sqlite_agent.py

Then just talk to it:

    you: where is order A-1001?
    you: how much did it cost?
    you: what else is still processing?

One thing to notice: the model never writes SQL. It picks a tool and supplies an
argument; the SQL lives in this file, parameterized. A model that cannot express
a query also cannot express `DROP TABLE`.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import sqlite3
import sys
from typing import Any

from anthropic import AsyncAnthropic

# Lets the example run straight from a checkout, before `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agent_harness import (  # noqa: E402
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
    PromptCompiler,
    Tool,
    ToolGateway,
    ToolProposal,
    ToolResult,
    TurnStepType,
)

DB_PATH = pathlib.Path(__file__).parent / "shop.db"
MODEL = "claude-opus-5"
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
SESSION = "demo-session"

ORDER_ID = re.compile(r"[A-Z]-\d{4}")
STATUSES = ("shipped", "processing", "delivered", "cancelled")
INTENTS = ("track_order", "refund_order", "cancel_order", "general_enquiry")

SEED = [
    ("A-1001", "Naveed", "Mechanical keyboard", "shipped", 129.99),
    ("A-1002", "Sara", "USB-C cable", "processing", 12.50),
    ("A-1003", "Naveed", "Monitor stand", "delivered", 45.00),
    ("A-1004", "Omar", "Laptop sleeve", "processing", 25.00),
    ("A-1005", "Sara", "Webcam", "cancelled", 89.00),
]


def setup_database() -> None:
    """Create and seed the demo database. Safe to re-run."""
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer TEXT NOT NULL,
            item     TEXT NOT NULL,
            status   TEXT NOT NULL,
            total    REAL NOT NULL
        )
        """
    )
    connection.execute("DELETE FROM orders")
    connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", SEED)
    connection.commit()
    connection.close()


# --- the tools: ordinary async functions that hit the database ---------------
async def get_order(session_id: str, arguments: dict) -> dict:
    order_id = str(arguments["order_id"]).upper()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    # Parameterized. The model supplies a value, never SQL.
    row = connection.execute(
        "SELECT * FROM orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    connection.close()
    if row is None:
        raise LookupError(f"no order with id {order_id}")
    return dict(row)


async def orders_by_status(session_id: str, arguments: dict) -> dict:
    status = str(arguments["status"]).lower()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT order_id, customer, item, total FROM orders WHERE status = ?",
        (status,),
    ).fetchall()
    connection.close()
    return {"status": status, "count": len(rows), "orders": [dict(r) for r in rows]}


TOOLS = [
    Tool(
        name="get_order",
        description="Get one order's full details by its id, e.g. A-1001.",
        parameters=["order_id"],
        execute=get_order,
        read_only=True,
        max_calls_per_session=20,
    ),
    Tool(
        name="orders_by_status",
        description=(
            "List every order with a given status. "
            "Valid statuses: shipped, processing, delivered, cancelled."
        ),
        parameters=["status"],
        execute=orders_by_status,
        read_only=True,
        max_calls_per_session=20,
    ),
]


# --- context: regex for structured ids, a small model for open-ended intent --
class ShopContextEngine(ContextEngine):
    """Entities by regex, intent by a small fast model.

    The split matters. An order id is *structured*, so a regex beats a model at
    it: deterministic, free, instant, and incapable of hallucinating one. Intent
    is open-ended -- "kab tak aayega", "where's my stuff", "has it left the
    warehouse" all mean track_order -- so that half needs a model.

    Anything outside INTENTS becomes "", and a falsy intent never diverges, so an
    uncertain classification *merges into* the live frame rather than replacing
    it. Failing safe here means keeping the task, not guessing a new one.
    """

    def __init__(self, memory: MemoryStore, client: AsyncAnthropic) -> None:
        super().__init__(memory)
        self.client = client

    async def extract_intent(self, message: str) -> tuple[str, dict[str, Any]]:
        entities: dict[str, Any] = {}
        found = ORDER_ID.search(message.upper())
        if found:
            entities["order_id"] = found.group(0)

        try:
            response = await self.client.messages.create(
                model=CLASSIFIER_MODEL,
                max_tokens=64,
                system=(
                    "Classify the customer message into exactly one label:\n"
                    + "\n".join(f"- {label}" for label in INTENTS)
                    + "\nReply with the label only, nothing else."
                ),
                messages=[{"role": "user", "content": message}],
            )
            label = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip().lower()
        except Exception:
            # A classifier outage must not destroy the turn.
            label = ""

        return (label if label in INTENTS else ""), entities


# --- guardrails: one per stage that actually matters here --------------------
def reject_oversized_input(message: str) -> None:
    """INPUT: cheapest possible check, before any model is paid."""
    if len(message) > 2_000:
        raise GuardrailViolation(GuardrailStage.INPUT, "message too long")


def validate_tool_arguments(proposal: ToolProposal) -> None:
    """TOOL: runs after the model proposes, before anything executes.

    This is the stage that stops an invented order id from ever reaching the
    database -- the model can ask for 'X-9999', but it never gets run.
    """
    if proposal.tool_name == "get_order":
        order_id = str(proposal.arguments.get("order_id", "")).upper()
        if not ORDER_ID.fullmatch(order_id):
            raise GuardrailViolation(
                GuardrailStage.TOOL, f"malformed order id: {order_id!r}"
            )
    if proposal.tool_name == "orders_by_status":
        status = str(proposal.arguments.get("status", "")).lower()
        if status not in STATUSES:
            raise GuardrailViolation(GuardrailStage.TOOL, f"unknown status: {status!r}")


def require_evidence(payload: tuple[str, list[ToolResult]]) -> None:
    """OUTPUT: no claim about an order without a successful lookup behind it.

    Only fires when the answer actually makes an order claim, so a greeting or a
    "I can't help with that" still gets through.
    """
    text, results = payload
    claims = bool(ORDER_ID.search(text.upper())) or any(
        word in text.lower() for word in STATUSES
    )
    if claims and not any(result.ok for result in results):
        raise GuardrailViolation(
            GuardrailStage.OUTPUT, "order claim with no successful lookup"
        )


# --- the adapter: CompiledPrompt -> Claude -> LLMTurnStep --------------------
def make_call_llm(client: AsyncAnthropic, tools: list[Tool]):
    schemas = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {name: {"type": "string"} for name in tool.parameters},
                "required": list(tool.parameters),
                "additionalProperties": False,
            },
        }
        for tool in tools
    ]

    async def call_llm(prompt) -> LLMTurnStep:
        response = await client.messages.create(
            model=MODEL,
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


def build_controller(client: AsyncAnthropic, backend: InMemoryBackend) -> LoopController:
    memory = MemoryStore(backend)

    guardrails = GuardrailPipeline()
    guardrails.add(Guardrail("input-size", GuardrailStage.INPUT, reject_oversized_input))
    guardrails.add(Guardrail("tool-args", GuardrailStage.TOOL, validate_tool_arguments))
    guardrails.add(Guardrail("evidence", GuardrailStage.OUTPUT, require_evidence))

    return LoopController(
        context_engine=ShopContextEngine(memory, client),
        prompt_compiler=PromptCompiler(
            role=(
                "You are a shop assistant answering questions about orders. "
                "Use the tools to look things up; never invent an order. "
                "Answer in one or two short sentences."
            )
        ),
        tool_gateway=ToolGateway(TOOLS),
        guardrails=guardrails,
        memory=memory,
        call_llm=make_call_llm(client, TOOLS),
    )


async def seed_memory(backend: InMemoryBackend) -> None:
    """Fill the two kinds the harness does *not* write for you.

    The loop writes EPISODIC itself (every tool call, every completed turn).
    SEMANTIC and PROCEDURAL are the caller's job -- in production these come from
    a CRM row and a policy document, not from literals in a source file.
    """
    store = MemoryStore(backend)
    await store.remember(
        SESSION, MemoryKind.SEMANTIC, "Naveed is a Pro member; free returns apply."
    )
    await store.remember(
        SESSION, MemoryKind.SEMANTIC, "Store currency is USD; prices exclude tax."
    )
    await store.remember(
        SESSION,
        MemoryKind.PROCEDURAL,
        "For a cancelled order, say the refund lands in 5 working days.",
    )


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY first")

    setup_database()
    backend = InMemoryBackend()
    await seed_memory(backend)
    controller = build_controller(AsyncAnthropic(), backend)
    limits = LoopLimits(max_turns=6, max_tool_calls=4, max_seconds=60.0)

    print("Ask about the orders. Blank line or Ctrl-C to quit.")
    print("Try: where is order A-1001?  /  what is still processing?")
    print("     multi-request and other languages both work.\n")

    # Carrying the frame between turns is what makes "how much did it cost?"
    # resolve against the order discussed a moment ago.
    frame = None
    while True:
        try:
            question = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        result = await controller.handle_turn(
            SESSION, question, current_frame=frame, limits=limits
        )
        frame = result.task_frame

        if result.final_text:
            print(f"agent: {result.final_text}")
        else:
            # No answer is a real outcome, not an error: a guardrail refused, or
            # a budget ran out. stopped_reason always says which.
            print(f"agent: (no answer -- {result.stopped_reason})")

        for tool_result in result.tool_results:
            print(f"       [used {tool_result.tool_name}: {tool_result.data}]")
        print(f"       [intent={frame.intent!r} entities={frame.entities}]")
        print()


if __name__ == "__main__":
    asyncio.run(main())
