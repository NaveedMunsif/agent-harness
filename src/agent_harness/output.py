"""Channel-aware output formatting.

The harness decides *what* the final text says; the formatter decides how it
looks on the channel it is being delivered to. Formatters are injected by the
caller exactly like ``call_llm`` and :class:`~agent_harness.memory.MemoryBackend`,
so this library keeps no dependency on any templating or markup engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = ["RenderChannel", "OutputFormatter", "PlainTextFormatter"]


class RenderChannel(str, Enum):
    """The delivery channel a final answer is being rendered for."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    JSON = "json"
    EMAIL_HTML = "email_html"
    VOICE_SSML = "voice_ssml"


@runtime_checkable
class OutputFormatter(Protocol):
    """Renders final answer text for a channel.

    Only applied to FINAL answers, after the OUTPUT guardrail passes. A
    clarifying question is always returned as plain text -- the calling UI
    decides how to present a question.
    """

    async def format(
        self,
        content: str,
        channel: RenderChannel,
        schema: dict[str, Any] | None = None,
    ) -> str: ...


class PlainTextFormatter:
    """Reference implementation: a passthrough.

    The default, so the library has no hard dependency on any markup engine.
    Swap in your own formatter to get Markdown, SSML, schema-validated JSON, or
    templated HTML.
    """

    async def format(
        self,
        content: str,
        channel: RenderChannel = RenderChannel.PLAIN_TEXT,
        schema: dict[str, Any] | None = None,
    ) -> str:
        return content
