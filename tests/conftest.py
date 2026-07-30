"""Shared test doubles: a scripted LLM, deterministic intent extraction, tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_harness import (
    ContextEngine,
    Guardrail,
    GuardrailPipeline,
    GuardrailStage,
    InMemoryBackend,
    LLMTurnStep,
    LoopController,
    MemoryKind,
    MemoryRecord,
    MemoryStore,
    PromptCompiler,
    RenderChannel,
    Tool,
    ToolGateway,
    TurnStepType,
)

SESSION = "session-1"


# --------------------------------------------------------------------------- LLMs


class ScriptedLLM:
    """Returns pre-set steps in order and records the prompts it was given."""

    def __init__(self, steps: list[LLMTurnStep]) -> None:
        self.steps = list(steps)
        self.prompts: list[Any] = []

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    async def __call__(self, prompt: Any) -> LLMTurnStep:
        self.prompts.append(prompt)
        if not self.steps:
            raise AssertionError("call_llm invoked more times than the script allows")
        return self.steps.pop(0)


class ToolSpamLLM:
    """Proposes a *distinct* tool call every turn.

    Distinct arguments matter: it ensures a budget limit stops the loop rather
    than repeated-call detection firing first.
    """

    def __init__(self, tool_name: str = "lookup_order") -> None:
        self.tool_name = tool_name
        self.prompts: list[Any] = []

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    async def __call__(self, prompt: Any) -> LLMTurnStep:
        self.prompts.append(prompt)
        return LLMTurnStep(
            step_type=TurnStepType.TOOL_CALL,
            tool_name=self.tool_name,
            tool_arguments={"attempt": len(self.prompts)},
        )


def tool_call(name: str, **arguments: Any) -> LLMTurnStep:
    return LLMTurnStep(
        step_type=TurnStepType.TOOL_CALL, tool_name=name, tool_arguments=arguments
    )


def final(text: str) -> LLMTurnStep:
    return LLMTurnStep(step_type=TurnStepType.FINAL, text=text)


def clarification(text: str) -> LLMTurnStep:
    return LLMTurnStep(step_type=TurnStepType.CLARIFICATION, text=text)


# ------------------------------------------------------------------- context/memory


class KeywordContextEngine(ContextEngine):
    """Deterministic intent extraction so pivot tests can be driven from text.

    ``"track order=A-1"`` -> intent ``"track"``, entities ``{"order": "A-1"}``.
    """

    async def extract_intent(self, message: str) -> tuple[str, dict[str, Any]]:
        tokens = message.split()
        intent = tokens[0].lower() if tokens else ""
        entities = {}
        for token in tokens[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                entities[key] = value
        return intent, entities


class RecordingBackend(InMemoryBackend):
    """InMemoryBackend that logs every query it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[tuple[MemoryKind, str, int]] = []

    async def query(
        self, session_id: str, kind: MemoryKind, query: str, limit: int
    ) -> list[MemoryRecord]:
        self.queries.append((kind, query, limit))
        return await super().query(session_id, kind, query, limit)

    def queries_for(self, kind: MemoryKind) -> list[tuple[MemoryKind, str, int]]:
        return [entry for entry in self.queries if entry[0] == kind]


# ------------------------------------------------------------------------ formatter


class UppercaseFormatter:
    """Visibly transforms text, so a test can prove formatting was applied."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, RenderChannel]] = []

    async def format(
        self,
        content: str,
        channel: RenderChannel = RenderChannel.PLAIN_TEXT,
        schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append((content, channel))
        return content.upper()


# ---------------------------------------------------------------------------- tools


def lookup_order_tool(
    *,
    authorize: Any = None,
    max_calls_per_session: int | None = None,
    redact_fields: list[str] | None = None,
    raises: Exception | None = None,
) -> Tool:
    """A read-only order lookup whose result carries a field worth redacting."""
    calls: list[dict[str, Any]] = []

    async def execute(session_id: str, arguments: dict[str, Any]) -> Any:
        calls.append({"session_id": session_id, **arguments})
        if raises is not None:
            raise raises
        return {
            "order_id": arguments.get("order_id", "A-1001"),
            "status": "shipped",
            "card_number": "4111111111111111",
        }

    tool = Tool(
        name="lookup_order",
        description="Look up an order's status by id.",
        parameters=["order_id"],
        execute=execute,
        authorize=authorize,
        read_only=True,
        max_calls_per_session=max_calls_per_session,
        redact_fields=redact_fields or [],
    )
    # Exposed for assertions about whether the tool body actually ran.
    tool.calls = calls  # type: ignore[attr-defined]
    return tool


# -------------------------------------------------------------------------- harness


@dataclass
class Harness:
    controller: LoopController
    memory: MemoryStore
    backend: InMemoryBackend
    gateway: ToolGateway
    pipeline: GuardrailPipeline
    llm: Any
    engine: ContextEngine
    tools: list[Tool] = field(default_factory=list)

    def episodic(self) -> list[MemoryRecord]:
        return [r for r in self.backend.all_records() if r.kind == MemoryKind.EPISODIC]

    def episodic_events(self) -> list[str]:
        return [str(r.metadata.get("event")) for r in self.episodic()]


def build_harness(
    *,
    steps: list[LLMTurnStep] | None = None,
    call_llm: Any = None,
    tools: list[Tool] | None = None,
    guardrails: list[Guardrail] | None = None,
    formatter: Any = None,
    backend: InMemoryBackend | None = None,
    role: str = "You are a support agent.",
) -> Harness:
    backend = backend or InMemoryBackend()
    memory = MemoryStore(backend)
    engine = KeywordContextEngine(memory)
    gateway = ToolGateway(list(tools or []))
    pipeline = GuardrailPipeline(list(guardrails or []))
    llm = call_llm if call_llm is not None else ScriptedLLM(list(steps or []))
    controller = LoopController(
        context_engine=engine,
        prompt_compiler=PromptCompiler(role),
        tool_gateway=gateway,
        guardrails=pipeline,
        memory=memory,
        call_llm=llm,
        output_formatter=formatter,
    )
    return Harness(
        controller=controller,
        memory=memory,
        backend=backend,
        gateway=gateway,
        pipeline=pipeline,
        llm=llm,
        engine=engine,
        tools=list(tools or []),
    )


class PayloadRecorder:
    """Captures the payload handed to each guardrail stage."""

    def __init__(self) -> None:
        self.seen: dict[GuardrailStage, list[Any]] = {stage: [] for stage in GuardrailStage}

    def guardrail(self, stage: GuardrailStage) -> Guardrail:
        async def check(payload: Any) -> None:
            self.seen[stage].append(payload)

        return Guardrail(name=f"record-{stage.value}", stage=stage, check=check)

    def ran(self, stage: GuardrailStage) -> bool:
        return bool(self.seen[stage])


@pytest.fixture
def recorder() -> PayloadRecorder:
    return PayloadRecorder()
