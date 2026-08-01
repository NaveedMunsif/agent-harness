"""Issue refunds, with three independent protections on the money.

A refund is the first thing in a support agent that can actually cost you. So
this example is less about the happy path than about the three *different*
mechanisms standing between a model's proposal and a debited account:

===  ===================  ==================================================
#    Mechanism            Where it lives
===  ===================  ==================================================
1    ownership            ``Tool.authorize`` on ``issue_refund``
2    amount ceiling       a TOOL-stage guardrail ($500)
3    no double refund     an atomic conditional UPDATE, capped by
                          ``max_calls_per_session``
===  ===================  ==================================================

Every one of them is enforced by a code path, not by reading the model's prose.
There is deliberately no OUTPUT guardrail scanning the answer for "a refund was
issued" claims: you cannot verify a natural-language claim without parsing
natural language, and a keyword heuristic that fires on the wrong sentence is
worse than no check at all, because it teaches you to switch guardrails off.

They are deliberately not four flavours of one check. Each sits at a different
point in the turn, and -- this is the part worth internalising -- they therefore
produce *different* ``stopped_reason`` values:

* A TOOL guardrail raises :class:`GuardrailViolation`, which aborts the whole
  turn. ``final_text`` is ``None`` and ``stopped_reason`` is
  ``guardrail_violation:tool``. The model never gets to speak.
* ``authorize`` and the tool's own status check raise inside the gateway. The
  loop catches those (see ``ToolError`` handling in ``loop.py``) and turns them
  into a **failed ToolResult** -- evidence the model must then explain. The turn
  still resolves normally, so ``stopped_reason`` is ``final_answer`` with an
  ``ok=False`` result attached.

Both are blocks. Only one is a *stop*. Reading ``stopped_reason`` alone is not
enough; you read it together with ``tool_results``.

    pip install agent-harnessed anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      # setx on Windows

    python examples/refund_agent.py          # interactive
    python examples/refund_agent.py --demo   # every scenario in sequence

The model never writes SQL. It picks a tool and supplies a value; every statement
in this file is parameterized and lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

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
    LoopResult,
    MemoryKind,
    MemoryStore,
    PromptCompiler,
    Tool,
    GatewayEvent,
    ToolGateway,
    ToolProposal,
    ToolResult,
    TurnStepType,
)

DB_PATH = pathlib.Path(__file__).parent / "refunds.db"
# Two model sizes on purpose: the loop juggles several constraints at once,
# while picking one label out of four is what Haiku is good at.
MODEL = "claude-sonnet-5"
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

REFUND_CEILING = 500.00

ORDER_ID = re.compile(r"[A-Z]-\d{4}")
STATUSES = ("delivered", "processing", "refunded", "cancelled")
INTENTS = ("track_order", "refund_order", "cancel_order", "general_enquiry")

# No authentication -- this is a demo. In production the customer comes from a
# verified session claim, never a lookup table or a model-supplied argument.
SESSIONS = {"sess-naveed": "Naveed", "sess-sara": "Sara", "sess-omar": "Omar"}

SEED = [
    ("A-1001", "Naveed", "Mechanical keyboard", "delivered", 129.99),
    ("A-1002", "Naveed", "USB-C cable", "processing", 12.50),
    ("A-1003", "Naveed", "Monitor stand", "refunded", 45.00),
    ("A-1004", "Omar", "Laptop sleeve", "delivered", 890.00),
    ("A-1005", "Sara", "Webcam", "cancelled", 89.00),
    ("A-1006", "Sara", "Monitor stand", "processing", 99.00),
    ("A-1007", "Naveed", "Wireless mouse", "delivered", 800.00),


]


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def setup_database() -> None:
    """Create and re-seed the demo database. Safe to re-run.

    The re-seed matters here in a way it does not for a read-only example: the
    demo issues a real refund, so without resetting, a second run would start
    from an already-refunded A-1001 and scenario 1 would not be reproducible.
    """
    connection = connect()
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


def fetch_order(order_id: str) -> sqlite3.Row | None:
    """One order row, or ``None``. Used by tools, authorize and guardrails alike."""
    connection = connect()
    try:
        return connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),)
        ).fetchone()
    finally:
        connection.close()


# --- the tools ---------------------------------------------------------------
async def authorize_read(session_id: str, arguments: dict) -> bool | str:
    """You may only read an order that is yours.

    Reads need this as much as writes do: without it, anyone who guesses an id
    sees another customer's item and total. Easy to forget precisely because
    ``read_only=True`` reads as harmless.

    Missing and not-yours are refused in the same words, for the same reason as
    :func:`authorize_refund` -- different wording is an enumeration oracle.
    """
    customer = SESSIONS.get(session_id)
    if customer is None:
        return "this session is not signed in"
    row = fetch_order(str(arguments.get("order_id", "")))
    if row is None or row["customer"] != customer:
        return "that order is not on this account"
    return True


async def get_order(session_id: str, arguments: dict) -> dict:
    """One order. Ownership is settled by ``authorize_read`` before this runs."""
    return dict(fetch_order(str(arguments["order_id"]).upper()))


async def orders_by_status(session_id: str, arguments: dict) -> dict:
    """List *this customer's* orders with a given status.

    Scoped in the WHERE clause rather than by an ``authorize`` callback, because
    a listing has no single order to authorize against -- the filter is the
    authorization. This is why ``execute`` receives ``session_id`` at all.
    """
    status = str(arguments["status"]).lower()
    customer = SESSIONS.get(session_id)
    if customer is None:
        raise PermissionError(f"unknown session: {session_id}")
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT order_id, item, total FROM orders "
            "WHERE status = ? AND customer = ?",
            (status, customer),
        ).fetchall()
    finally:
        connection.close()
    return {"status": status, "count": len(rows), "orders": [dict(r) for r in rows]}


async def my_orders(session_id: str, arguments: dict) -> dict:
    """Every order belonging to this customer, whatever its status.

    Exists because ``orders_by_status`` demands a status and so cannot answer
    "show me my orders": the model guesses one, gets nothing, guesses again, and
    trips ``repeated_call_detected``. A question users will obviously ask that
    maps to no tool shows up there first.
    """
    customer = SESSIONS.get(session_id)
    if customer is None:
        raise PermissionError(f"unknown session: {session_id}")
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT order_id, item, status, total FROM orders WHERE customer = ? "
            "ORDER BY order_id",
            (customer,),
        ).fetchall()
    finally:
        connection.close()
    return {"count": len(rows), "orders": [dict(r) for r in rows]}


async def issue_refund(session_id: str, arguments: dict) -> dict:
    """Refund one delivered order. **Has side effects.**

    Protection 3, second half. The status check is not a ``SELECT`` followed by
    an ``UPDATE`` -- it is one conditional ``UPDATE`` whose ``WHERE`` clause
    carries the precondition, so two concurrent turns cannot both observe
    'delivered' and both refund. ``rowcount`` reports whether this call was the
    one that won.
    """
    order_id = str(arguments["order_id"]).upper()
    row = fetch_order(order_id)
    if row is None:
        raise LookupError(f"no order with id {order_id}")

    connection = connect()
    try:
        cursor = connection.execute(
            "UPDATE orders SET status = 'refunded' "
            "WHERE order_id = ? AND status = 'delivered'",
            (order_id,),
        )
        connection.commit()
    finally:
        connection.close()

    if cursor.rowcount != 1:
        raise ValueError(
            f"order {order_id} is {row['status']}, not delivered -- "
            "only a delivered order can be refunded"
        )

    return {
        "order_id": order_id,
        "customer": row["customer"],
        "refunded_amount": row["total"],
        "status": "refunded",
    }


async def authorize_refund(session_id: str, arguments: dict) -> bool | str:
    """Protection 1: you may only refund an order that is yours.

    This is why ``authorize`` takes ``session_id`` rather than reading it out of
    ``arguments``: the customer identity comes from the session the harness was
    called with, so a model that puts ``"customer": "Omar"`` in its arguments
    cannot promote itself.

    Returning a string refuses *and* explains. A bare ``False`` can only produce
    "authorization declined", which a model reads as transient and retries.
    """
    customer = SESSIONS.get(session_id)
    if customer is None:
        return "this session is not signed in"
    row = fetch_order(str(arguments.get("order_id", "")))
    # "Does not exist" and "belongs to someone else" are refused in the same
    # words: different wording lets anyone walk the id space to find real orders.
    # Log the distinction here if you need it; never put it in the reply.
    if row is None or row["customer"] != customer:
        return "that order is not on this account"
    return True


TOOLS = [
    Tool(
        name="my_orders",
        description=(
            "List every order belonging to this customer, with status and total. "
            "Use this for open questions like 'what did I order' or 'show my "
            "orders', where no order id or status has been given."
        ),
        parameters=[],
        execute=my_orders,
        read_only=True,
        max_calls_per_session=20,
    ),
    Tool(
        name="get_order",
        description="Get one order's full details by its id, e.g. A-1001.",
        parameters=["order_id"],
        execute=get_order,
        authorize=authorize_read,
        read_only=True,
        max_calls_per_session=20,
    ),
    Tool(
        name="orders_by_status",
        description=(
            "List this customer's orders with a given status. "
            "Valid statuses: delivered, processing, refunded, cancelled."
        ),
        parameters=["status"],
        execute=orders_by_status,
        read_only=True,
        max_calls_per_session=20,
    ),
    Tool(
        name="issue_refund",
        description=(
            # Says what the tool does, not which calls get refused: this text
            # lands in the prompt, so listing rules here makes the model refuse
            # on its own and the gateway is never exercised.
            "Refund one order and return the amount refunded."
        ),
        parameters=["order_id"],
        execute=issue_refund,
        authorize=authorize_refund,
        # Not read_only: the prompt renders this as '[has side effects]', and
        # that label is the model's only cue that this tool is different.
        read_only=False,
        # A blast-radius cap, not the double-refund guard -- that is the atomic
        # UPDATE in issue_refund. Five rather than one because the gateway counts
        # at dispatch, so an attempt refused by the status check still spends
        # budget; the ceiling and authorize fire earlier and cost nothing.
        max_calls_per_session=5,
    ),
]


# --- guardrails: one per stage that matters here -----------------------------
def reject_oversized_input(message: str) -> None:
    """INPUT: the cheapest possible check, before any model is paid."""
    if len(message) > 2_000:
        raise GuardrailViolation(GuardrailStage.INPUT, "message too long")


def refund_under_ceiling(proposal: ToolProposal) -> None:
    """TOOL, protection 2: no refund over $500 without a human.

    A TOOL guardrail sees only the :class:`ToolProposal`, so it looks the amount
    up itself rather than trusting a number the model supplied. It has no
    ``session_id``, which is why ownership belongs in ``authorize`` instead.
    It also runs *before* the gateway, so on an order that is both someone
    else's and over the ceiling, this fires first.
    """
    if proposal.tool_name != "issue_refund":
        return
    row = fetch_order(str(proposal.arguments.get("order_id", "")))
    if row is not None and row["total"] > REFUND_CEILING:
        raise GuardrailViolation(
            GuardrailStage.TOOL,
            f"refund of ${row['total']:.2f} exceeds the ${REFUND_CEILING:.2f} "
            "automatic limit and needs human approval",
        )


# --- context: regex for structured ids, a small model for open-ended intent --
class RefundContext(ContextEngine):
    """Entities by regex, intent by a small fast model.

    An order id is *structured*, so a regex beats a model at it: deterministic,
    free, instant, and incapable of hallucinating one. Intent is open-ended, so
    that half needs a model.

    Anything outside ``INTENTS``, and any exception, becomes ``""``. A falsy
    intent never diverges, so an uncertain classification merges into the live
    frame instead of destroying it -- failing safe here means keeping the task,
    not guessing a new one.
    """

    def __init__(self, memory: MemoryStore, client: AsyncAnthropic) -> None:
        super().__init__(memory)
        self.client = client

    async def build(
        self,
        session_id: str,
        message: str,
        current_frame=None,
        semantic_query: str = "",
        procedural_query: str = "",
    ):
        """Delegate, then report the state decision and what memory returned.

        Traced here rather than around ``MemoryStore.recall`` because the three
        kinds are only worth reading together. A pivot is the irreversible part:
        it drops the previous frame's plan and in-flight tool ids.
        """
        context = await super().build(
            session_id, message, current_frame, semantic_query, procedural_query
        )
        trace(
            "state",
            "fresh frame (pivot -- earlier task dropped)"
            if context.is_pivot
            else "continued the existing frame",
            "info",
        )
        trace(
            "memory",
            f"recalled {len(context.semantic)} semantic, "
            f"{len(context.procedural)} procedural, "
            f"{len(context.episodic)} episodic",
            "info",
        )
        return context

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
            label = (
                "".join(b.text for b in response.content if b.type == "text")
                .strip()
                .lower()
            )
        except Exception:
            # A classifier outage must not destroy the turn.
            label = ""

        intent = label if label in INTENTS else ""
        trace(
            "context",
            f"intent={intent or '(unknown)'} entities={entities or '{}'}",
            "info",
        )
        return intent, entities


# --- tracing: where in the turn did it actually stop? ------------------------
#
# `stopped_reason` says a turn ended, not which checks it passed first. Each
# stage is wrapped where it actually runs, which works because every extension
# point in this library is a plain callable you supply.
@dataclass
class TraceEvent:
    stage: str
    detail: str
    status: str = "ok"  # ok | fail | info


TRACE: list[TraceEvent] = []


def trace(stage: str, detail: str, status: str = "ok") -> None:
    TRACE.append(TraceEvent(stage, detail, status))


def traced_guardrail(name: str, stage: GuardrailStage, check, applies=None):
    """Wrap a guardrail so all three outcomes are recorded, then re-raise.

    ``applies`` separates "checked and fine" from "not this guardrail's
    business": a TOOL guardrail runs on every proposal, so without it the refund
    ceiling reports 'passed' on a plain listing.
    """

    def wrapped(payload):
        if applies is not None and not applies(payload):
            trace(f"{stage.value} guardrail", f"{name}: not applicable", "info")
            return
        try:
            check(payload)
        except GuardrailViolation as violation:
            trace(f"{stage.value} guardrail", f"{name}: {violation.reason}", "fail")
            raise
        trace(f"{stage.value} guardrail", f"{name}: passed")

    return wrapped


class TracingMemoryStore(MemoryStore):
    """Reports what the harness writes back, which is the audit trail forming.

    Only writes are traced. Reads are reported once, from ``RefundContext.build``,
    where all three kinds have been retrieved and can be counted together.
    """

    async def remember(self, session_id, kind, content, **metadata):
        record = await super().remember(session_id, kind, content, **metadata)
        event = metadata.get("event")
        if event:  # seeding has no event; only the loop's own writes do
            trace("memory", f"wrote {kind.value}: {event}", "info")
        return record


class TracingPromptCompiler(PromptCompiler):
    """Reports the compiled artifact: its version and which sections it carries.

    Worth seeing because the section list *is* the answer to "what could the
    model possibly know this turn".
    """

    def compile(self, context, tools, user_message, output_requirement=""):
        prompt = super().compile(context, tools, user_message, output_requirement)
        trace(
            "prompt",
            f"v{prompt.version}, {len(prompt.sections)} sections: "
            + ", ".join(prompt.sections),
            "info",
        )
        return prompt


def on_gateway_event(event: GatewayEvent) -> None:
    """Record every decision the gateway makes.

    This replaces wrapping ``authorize`` and ``execute`` by hand. It also covers
    the two stages that have no callable to wrap at all -- allowlisting and the
    per-session rate limit -- which previously had to be guessed at from the
    error string on a failed ``ToolResult``.
    """
    detail = f"{event.tool_name}: {event.detail}" if event.detail else (
        f"{event.tool_name}: {'ok' if event.ok else 'refused'}"
    )
    trace(event.stage, detail, "ok" if event.ok else "fail")


def render_trace(result: LoopResult) -> str:
    """The recorded stages, in the order they ran, with where it stopped."""
    marks = {"ok": "PASS", "fail": "STOP", "info": "  ->"}
    lines = ["   " + "-" * 66, "   how the turn went"]
    for event in TRACE:
        lines.append(f"   {marks[event.status]:>4}  {event.stage:<18} {event.detail}")

    lines.append(f"   {'==':>4}  {'stopped':<18} {result.stopped_reason}")
    lines.append("   " + "-" * 66)
    return "\n".join(lines)


# --- the adapter: CompiledPrompt -> Claude -> LLMTurnStep --------------------
def _schema_for(tool: Tool) -> dict[str, Any]:
    """Derive an Anthropic tool schema from a harness ``Tool``.

    ``Tool.parameters`` carries names without types, so every parameter is a
    required string -- the most faithful schema the metadata supports. Deriving
    it keeps the gateway's tools and the model's tools from drifting apart.
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": {name: {"type": "string"} for name in tool.parameters},
            "required": list(tool.parameters),
            "additionalProperties": False,
        },
    }


def make_call_llm(client: AsyncAnthropic, tools: list[Tool], model: str = MODEL):
    """Build the ``async (CompiledPrompt) -> LLMTurnStep`` callable the loop wants."""
    schemas = [_schema_for(tool) for tool in tools]

    async def call_llm(prompt: CompiledPrompt) -> LLMTurnStep:
        response = await client.messages.create(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt.render()}],
            tools=schemas,
        )
        for block in response.content:
            if block.type == "tool_use":
                args = ", ".join(f"{k}={v}" for k, v in dict(block.input).items())
                # One prompt is compiled per iteration immediately before this
                # call, so counting them numbers the turns without extra state.
                turn = sum(1 for event in TRACE if event.stage == "prompt")
                trace("model", f"turn {turn}: proposes {block.name}({args})", "info")
                return LLMTurnStep(
                    step_type=TurnStepType.TOOL_CALL,
                    tool_name=block.name,
                    tool_arguments=dict(block.input),
                )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        turn = sum(1 for event in TRACE if event.stage == "prompt")
        trace("model", f"turn {turn}: answers", "info")
        if not text:
            # LLMTurnStep rejects a FINAL with no text, so fail here where the
            # cause is still visible rather than inside the validator.
            raise RuntimeError(
                f"no tool call and no text (stop_reason={response.stop_reason})"
            )
        return LLMTurnStep(step_type=TurnStepType.FINAL, text=text)

    return call_llm


ROLE = (
    "You are a refund agent for an online shop. Look an order up before making "
    "any claim about it, then call issue_refund when the customer asks for a "
    "refund. The system decides whether a refund is permitted -- do not "
    "pre-judge eligibility yourself. "
    # Keep this: an order that is not the customer's fails the *read*, so
    # issue_refund is never proposed and ownership and the ceiling never run.
    "When the customer asks for a refund, call issue_refund straight away with "
    "the order id. Do not call get_order first to check whether it is allowed; "
    "issue_refund performs its own checks and will refuse if it must. "
    # authorize returns only a bool, so the gateway's wording says nothing
    # useful. describe_stop handles this too; this lets the model phrase it.
    "If a tool reports 'authorization declined', that order is not on this "
    "customer's account. Say exactly that and ask them to check the order id. "
    "Never say whether the order exists or who it belongs to. Do not try again. "
    # Tool results are folded into HISTORY, which reads as the past -- without
    # this, a refund issued a second ago is reported as 'already' done.
    "Results in HISTORY may be from the call you just made. Report a refund you "
    "issued in this turn as newly completed -- never say it had 'already' "
    "happened. "
    "If a refund is refused, say plainly that it "
    "was refused and why -- never tell a customer a refund was issued unless the "
    "issue_refund tool actually succeeded. "
    # A failed call looks transient, and a successful one that lacks the answer
    # invites a second look. Both trip repeated_call_detected.
    "Never call the same tool with the same arguments twice in one turn -- the "
    "earlier result is already in your history. If a call failed, explain the "
    "failure. If it succeeded but does not contain what the customer asked for, "
    "say plainly that the information is not available. Do not look again, and "
    "never invent a value that a tool did not return. "
    # cancel_order is classified but has no tool. An intent you cannot serve is
    # still worth recognising: it lets you decline instead of improvising.
    "You cannot cancel orders. If a customer asks to cancel one, say so and "
    "point them at customer service -- do not look the order up first. "
    "Answer in one or two short sentences."
)


def build_controller(
    client: AsyncAnthropic,
    backend: InMemoryBackend,
    tools: list[Tool] | None = None,
) -> LoopController:
    tools = TOOLS if tools is None else tools
    memory = TracingMemoryStore(backend)

    guardrails = GuardrailPipeline()
    guardrails.add(
        Guardrail(
            "input-size",
            GuardrailStage.INPUT,
            traced_guardrail("input-size", GuardrailStage.INPUT, reject_oversized_input),
        )
    )
    guardrails.add(
        Guardrail(
            "refund-ceiling",
            GuardrailStage.TOOL,
            traced_guardrail(
                "refund-ceiling",
                GuardrailStage.TOOL,
                refund_under_ceiling,
                applies=lambda proposal: proposal.tool_name == "issue_refund",
            ),
        )
    )

    return LoopController(
        context_engine=RefundContext(memory, client),
        prompt_compiler=TracingPromptCompiler(role=ROLE),
        tool_gateway=ToolGateway(tools, on_event=on_gateway_event),
        guardrails=guardrails,
        memory=memory,
        # The gateway and the adapter are handed the same list, so the two
        # definitions cannot drift.
        call_llm=make_call_llm(client, tools),
    )


async def seed_memory(backend: InMemoryBackend, session_id: str) -> None:
    """Fill the two kinds the harness does *not* write for you.

    The loop writes EPISODIC itself. In production these two come from a CRM row
    and a policy document, not from literals in a source file.
    """
    store = MemoryStore(backend)
    await store.remember(
        session_id,
        MemoryKind.SEMANTIC,
        f"The signed-in customer is {SESSIONS.get(session_id, 'unknown')}. "
        "Store currency is USD; prices exclude tax.",
    )
    # Deliberately omits the eligibility rules: a model told what is forbidden
    # refuses on its own, leaving the gateway idle and its behaviour unproven.
    await store.remember(
        session_id,
        MemoryKind.PROCEDURAL,
        "To refund: look the order up, then call issue_refund. "
        "The system enforces eligibility.",
    )


# --- the demo ----------------------------------------------------------------
# Each scenario isolates one protection, so every *other* protection has to
# pass: the ownership case uses a cheap order owned by someone else, the ceiling
# case one the caller owns. The success runs last, so earlier scenarios see a
# database the demo has not changed yet.
SCENARIOS = [
    ("sess-naveed", "refund order A-1006", "blocked -- Sara's order (ownership)"),
    ("sess-omar", "refund order A-1004", "blocked -- over the $500 ceiling"),
    ("sess-naveed", "refund order A-1002", "blocked -- processing, not delivered"),
    ("sess-naveed", "refund order A-1003", "blocked -- already refunded"),
    ("sess-naveed", "refund order A-1001", "succeeds -- delivered, his, under the ceiling"),
]


async def run_demo(client: AsyncAnthropic) -> None:
    """Run every scenario, each in its own session and its own controller.

    A fresh controller per scenario keeps them independent: one scenario's spent
    ``max_calls_per_session`` budget or accumulated memory must not decide the
    next one's outcome. The database is deliberately *shared* across them,
    because scenario 1's refund really does happen and the rest should see it.
    """
    setup_database()

    for index, (session_id, question, expectation) in enumerate(SCENARIOS, start=1):
        backend = InMemoryBackend()
        await seed_memory(backend, session_id)
        controller = build_controller(client, backend)

        TRACE.clear()
        result = await controller.handle_turn(
            session_id,
            question,
            limits=LoopLimits(max_turns=6, max_tool_calls=4, max_seconds=120.0),
        )

        print("=" * 72)
        print(f"{index}. {session_id} ({SESSIONS[session_id]}): {question!r}")
        print(f"   expected: {expectation}")
        print("-" * 72)
        print(f"   STOPPED_REASON : {result.stopped_reason}")
        print(f"   answer         : {result.final_text or describe_stop(result)}")
        print(f"   turns_used     : {result.turns_used}")
        for tool_result in result.tool_results:
            status = "ok" if tool_result.ok else "FAILED"
            payload = tool_result.data if tool_result.ok else tool_result.error
            print(f"   tool {tool_result.tool_name} [{status}]: {payload}")
        # The demo always traces: seeing which protection fired, and which ones
        # the turn got past first, is the entire point of these scenarios.
        print(render_trace(result))
        print()

    print("=" * 72)
    print("final database state")
    print("-" * 72)
    connection = connect()
    try:
        for row in connection.execute(
            "SELECT order_id, customer, status, total FROM orders ORDER BY order_id"
        ):
            print(
                f"   {row['order_id']}  {row['customer']:<7}"
                f"  {row['status']:<11} ${row['total']:.2f}"
            )
    finally:
        connection.close()


# Derived from `read_only=False`, the harness's own marker for "this one has
# consequences", so the set cannot drift.
WRITE_TOOLS = {tool.name for tool in TOOLS if not tool.read_only}


# Keyed by exception class name, which is what LoopController puts in front of
# the colon when it converts a ToolError into a failed ToolResult.
# ``ToolAuthorizationFailed`` is deliberately absent: ``authorize`` now returns
# its own reason, so the error already carries a sentence written for a customer.
# Mapping it here would replace that with something vaguer. What remains are the
# refusals the gateway phrases itself, in its own technical terms.
_GATEWAY_REFUSALS = {
    "ToolRateLimited": (
        "You have reached the refund limit for this session. "
        "Customer service can help with anything further."
    ),
    "ToolNotAllowed": "I am not able to do that here.",
}


def describe_stop(result: LoopResult) -> str:
    """What to say when the loop ends with no answer.

    Every exit without ``final_text`` still carries its reason: a guardrail names
    the stage, and a forced stop leaves the tool results that led to it. Printing
    "(no answer)" throws that away and makes a correctly-blocked turn look like a
    crash. Telling the model not to retry is advice; the loop's detector is
    enforcement, and the two disagree often enough to matter.
    """
    # A completed side effect outranks everything and is checked first: a repeat
    # after a successful refund would otherwise fall through to "not in my
    # records", telling the customer nothing happened after the money has gone.
    for tool_result in result.tool_results:
        if tool_result.ok and tool_result.tool_name in WRITE_TOOLS:
            data = tool_result.data if isinstance(tool_result.data, dict) else {}
            amount = data.get("refunded_amount")
            order_id = data.get("order_id", "your order")
            if amount is not None:
                return f"Your refund for order {order_id} has been issued: ${amount:.2f}."
            return f"{tool_result.tool_name} completed for {order_id}."

    for tool_result in reversed(result.tool_results):
        if not tool_result.ok and tool_result.error:
            # Strip the exception class name: 'ValueError: order A-1005 is
            # cancelled' is not a sentence to show a customer.
            kind, _, detail = tool_result.error.partition(": ")
            # Gateway refusals are the exception: `authorize` returns a bare
            # bool, so all it can say is "authorization declined". The exception
            # type knows exactly what happened, so map it here.
            if kind in _GATEWAY_REFUSALS:
                return _GATEWAY_REFUSALS[kind]
            return (detail or tool_result.error).strip()

    # Every call succeeded and the loop still stopped: the model looked twice.
    # Say what was established rather than giving a generic apology.
    if result.stopped_reason == "repeated_call_detected" and result.tool_results:
        looked_up = ", ".join(
            sorted({t.tool_name for t in result.tool_results if t.ok})
        )
        if looked_up:
            return (
                f"I checked {looked_up} and that is all the detail I hold for "
                "this order -- what you asked for is not in my records. "
                "Customer service can help with anything beyond it."
            )

    if result.violation_reason:
        return result.violation_reason
    if result.stopped_reason.startswith("guardrail_violation"):
        stage = result.stopped_reason.split(":", 1)[-1]
        return f"That request was refused by a {stage} policy check."
    if result.stopped_reason.endswith("_exceeded"):
        return "That took too many steps, so I stopped. Try narrowing the request."
    return "I could not complete that."


def resolve_session(requested: str | None) -> str:
    """Session id from ``--session``, or an interactive pick.

    Accepts either the session id or the customer name, case-insensitively, so
    ``--session sara`` and ``--session sess-sara`` both work.
    """
    if requested:
        wanted = requested.strip().lower()
        for session_id, customer in SESSIONS.items():
            if wanted in (session_id.lower(), customer.lower()):
                return session_id
        known = ", ".join(f"{c} ({s})" for s, c in SESSIONS.items())
        raise SystemExit(f"unknown session {requested!r}. Known: {known}")

    print("Who are you?")
    for number, session_id in enumerate(SESSIONS, start=1):
        print(f"  {number}. {SESSIONS[session_id]} ({session_id})")
    choice = input(f"pick 1-{len(SESSIONS)}: ").strip()
    try:
        return list(SESSIONS)[int(choice) - 1]
    except (ValueError, IndexError):
        raise SystemExit(f"pick a number between 1 and {len(SESSIONS)}")


async def run_interactive(
    client: AsyncAnthropic,
    requested: str | None = None,
    keep_db: bool = False,
    show_trace: bool = False,
) -> None:
    session_id = resolve_session(requested)

    # --keep-db is what makes the two-terminal demo work: refund as one customer,
    # restart as another, and the first refund is still there.
    if not keep_db or not DB_PATH.exists():
        setup_database()
    backend = InMemoryBackend()
    await seed_memory(backend, session_id)
    controller = build_controller(client, backend)
    limits = LoopLimits(max_turns=6, max_tool_calls=4, max_seconds=120.0)

    print(f"\nYou are {SESSIONS[session_id]} ({session_id}).")
    print("Blank line or Ctrl-C to quit.")
    print("Try: what did I order?  /  refund order A-1001  /  refund order A-1004\n")

    # Carrying the frame between turns is what makes "refund it" resolve against
    # the order discussed a moment ago.
    frame = None
    while True:
        try:
            question = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        TRACE.clear()
        result = await controller.handle_turn(
            session_id, question, current_frame=frame, limits=limits
        )
        frame = result.task_frame

        print(f"agent: {result.final_text or describe_stop(result)}")

        for tool_result in result.tool_results:
            payload = tool_result.data if tool_result.ok else tool_result.error
            print(f"       [{tool_result.tool_name}: {payload}]")
        if show_trace:
            print(render_trace(result))
        else:
            print(f"       [stopped_reason={result.stopped_reason}]")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the refund scenarios instead of chatting",
    )
    parser.add_argument(
        "--session",
        metavar="WHO",
        help="sign in as a customer without being prompted, "
        "by name or session id (e.g. --session sara)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="after each turn, show which stage of the pipeline passed and "
        "where it stopped",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="do not re-seed the database on start, so refunds persist "
        "across runs while you switch sessions",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY first")

    client = AsyncAnthropic()
    if args.demo:
        await run_demo(client)
    else:
        await run_interactive(
            client, args.session, keep_db=args.keep_db, show_trace=args.trace
        )


if __name__ == "__main__":
    asyncio.run(main())
