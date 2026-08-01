"""agent-harness: the runtime environment around an LLM.

    Agent = LLM + Context + Memory + Tools + Control Flow + Guardrails + State

The LLM reasons and proposes actions. The harness decides what it sees, validates
and executes what it asks for, controls how long the loop runs, formats the result
for its delivery channel, and hands back the final answer.

The model client is *not* part of this library: pass an async
``(CompiledPrompt) -> LLMTurnStep`` callable as ``call_llm``.
"""

from .context import ContextEngine, ScopedContext
from .guardrails import (
    Guardrail,
    GuardrailPipeline,
    GuardrailStage,
    GuardrailViolation,
)
from .loop import (
    DEFAULT_OUTPUT_REQUIREMENT,
    LLMTurnStep,
    LoopBudget,
    LoopController,
    LoopLimits,
    LoopResult,
    TurnStepType,
)
from .memory import (
    InMemoryBackend,
    MemoryBackend,
    MemoryKind,
    MemoryRecord,
    MemoryStore,
)
from .output import OutputFormatter, PlainTextFormatter, RenderChannel
from .prompt import CompiledPrompt, PromptCompiler
from .state import TaskFrame
from .tools import (
    REDACTED,
    GatewayEvent,
    Tool,
    ToolAuthorizationFailed,
    ToolError,
    ToolGateway,
    ToolNotAllowed,
    ToolProposal,
    ToolRateLimited,
    ToolResult,
)

__version__ = "0.3.0"

__all__ = [
    # state
    "TaskFrame",
    # memory
    "MemoryKind",
    "MemoryRecord",
    "MemoryBackend",
    "InMemoryBackend",
    "MemoryStore",
    # context
    "ContextEngine",
    "ScopedContext",
    # prompt
    "CompiledPrompt",
    "PromptCompiler",
    # tools
    "Tool",
    "ToolProposal",
    "ToolResult",
    "ToolGateway",
    "GatewayEvent",
    "ToolError",
    "ToolNotAllowed",
    "ToolAuthorizationFailed",
    "ToolRateLimited",
    "REDACTED",
    # guardrails
    "Guardrail",
    "GuardrailPipeline",
    "GuardrailStage",
    "GuardrailViolation",
    # output
    "RenderChannel",
    "OutputFormatter",
    "PlainTextFormatter",
    # loop
    "TurnStepType",
    "LLMTurnStep",
    "LoopController",
    "LoopLimits",
    "LoopBudget",
    "LoopResult",
    "DEFAULT_OUTPUT_REQUIREMENT",
    "__version__",
]
