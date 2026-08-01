"""The tool gateway: the only path from a model's proposal to a real side effect.

The model *proposes* (:class:`ToolProposal`); the gateway decides. Every call
passes allowlisting, authorization and per-session rate limiting before it
executes, and result fields named in ``Tool.redact_fields`` are scrubbed before
the :class:`ToolResult` leaves the gateway -- so secrets never reach episodic
memory or the next prompt.

Two distinct failure modes, deliberately:

* **Policy** failures (not allowed / not authorized / rate limited) raise a
  :class:`ToolError`. The call never ran.
* **Execution** failures (the tool's own code raised) come back as a
  ``ToolResult`` with ``ok=False`` and ``error`` set. The call did run, so it is
  recordable evidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

__all__ = [
    "Tool",
    "ToolProposal",
    "ToolResult",
    "ToolError",
    "ToolNotAllowed",
    "ToolAuthorizationFailed",
    "ToolRateLimited",
    "ToolGateway",
    "GatewayEvent",
    "REDACTED",
]

REDACTED = "[REDACTED]"

ExecuteFn = Callable[[str, dict[str, Any]], Awaitable[Any]]
#: Return ``True`` to allow, ``False`` to refuse, or a **string** to refuse with
#: that string as the reason. A bare ``False`` can only ever produce a generic
#: "authorization declined", which tells neither the caller nor a model *why* --
#: and an unexplained refusal reads like a transient error worth retrying.
AuthorizeFn = Callable[[str, dict[str, Any]], Awaitable["bool | str"]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Tool:
    """A capability the model may propose, plus the policy that governs it.

    ``execute`` and ``authorize`` both take ``(session_id, arguments)`` so a tool
    can make per-session decisions -- tenant scoping, per-user permissions --
    without the gateway having to smuggle ``session_id`` in through the model's
    arguments.

    ``read_only`` defaults to ``False``: a tool must explicitly claim it has no
    side effects. ``redact_fields`` names keys scrubbed from the result data.
    """

    name: str
    description: str
    parameters: list[str]
    execute: ExecuteFn
    authorize: AuthorizeFn | None = None
    read_only: bool = False
    max_calls_per_session: int | None = None
    redact_fields: list[str] = field(default_factory=list)


@dataclass
class ToolProposal:
    """A model's request to call a tool. Not yet validated, not yet run."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class ToolResult:
    """The outcome of a dispatched tool call, post-redaction."""

    proposal_id: str
    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None
    executed_at: datetime = field(default_factory=_utcnow)


@dataclass
class GatewayEvent:
    """One decision the gateway made on its way to running a tool.

    Three of the gateway's four jobs -- allowlisting, rate limiting, redaction --
    have no callable you can wrap, so from the outside they are invisible: a
    rate-limited call and a call that was never proposed look the same in a
    ``LoopResult``. Passing ``on_event`` to :class:`ToolGateway` makes each
    decision observable without changing what the gateway does.

    ``stage`` is one of ``"allowlist"``, ``"authorize"``, ``"rate_limit"`` or
    ``"execute"``. ``ok`` reports whether that stage let the call proceed.
    """

    stage: str
    tool_name: str
    session_id: str
    ok: bool
    detail: str | None = None


GatewayHook = Callable[[GatewayEvent], None]


class ToolError(Exception):
    """Base class for policy refusals raised before a tool runs."""


class ToolNotAllowed(ToolError):
    """Unregistered tool, or one excluded by this turn's allowlist."""


class ToolAuthorizationFailed(ToolError):
    """The tool's own ``authorize`` callable declined this call."""


class ToolRateLimited(ToolError):
    """The tool's ``max_calls_per_session`` budget is spent."""


class ToolGateway:
    """Registry plus enforcement point for tool execution.

    ``on_event`` receives a :class:`GatewayEvent` for every decision made here.
    It is called synchronously and its return value is ignored; an exception
    raised inside it will propagate, so keep it to recording.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        on_event: GatewayHook | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._call_counts: dict[tuple[str, str], int] = {}
        self._on_event = on_event
        for tool in tools or []:
            self.register(tool)

    def _emit(
        self,
        stage: str,
        tool_name: str,
        session_id: str,
        ok: bool,
        detail: str | None = None,
    ) -> None:
        if self._on_event is not None:
            self._on_event(GatewayEvent(stage, tool_name, session_id, ok, detail))

    def register(self, tool: Tool) -> Tool:
        """Register (or replace) a tool by name. Returns the tool."""
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        """The registered tool, or ``None``."""
        return self._tools.get(name)

    def allowed_tools(self, allowlist: list[str] | None = None) -> list[Tool]:
        """Registered tools, optionally narrowed to ``allowlist``.

        ``None`` means "no restriction this turn" and returns everything; an
        empty list means "no tools this turn" and returns nothing.
        """
        tools = list(self._tools.values())
        if allowlist is None:
            return tools
        permitted = set(allowlist)
        return [tool for tool in tools if tool.name in permitted]

    def call_count(self, session_id: str, tool_name: str) -> int:
        """How many times ``tool_name`` has been dispatched for this session."""
        return self._call_counts.get((session_id, tool_name), 0)

    async def execute(
        self,
        session_id: str,
        proposal: ToolProposal,
        allowlist: list[str] | None = None,
    ) -> ToolResult:
        """Validate and run ``proposal``.

        Raises :class:`ToolNotAllowed`, :class:`ToolAuthorizationFailed` or
        :class:`ToolRateLimited` when policy refuses -- in those cases nothing
        ran. If the tool itself raises, that is captured as a ``ToolResult`` with
        ``ok=False``, because a call that ran is evidence even when it failed.
        """
        tool = self._tools.get(proposal.tool_name)
        if tool is None:
            reason = f"unknown tool: {proposal.tool_name!r}"
            self._emit("allowlist", proposal.tool_name, session_id, False, reason)
            raise ToolNotAllowed(reason)

        if allowlist is not None and proposal.tool_name not in set(allowlist):
            reason = f"tool not in allowlist: {proposal.tool_name!r}"
            self._emit("allowlist", tool.name, session_id, False, reason)
            raise ToolNotAllowed(reason)

        if tool.authorize is not None:
            verdict = await tool.authorize(session_id, proposal.arguments)
            # A string is a refusal *carrying its reason*. Tested before
            # truthiness on purpose: every non-empty string is truthy, so
            # checking ``if not verdict`` first would let "not your order"
            # through as an approval.
            if isinstance(verdict, str) or not verdict:
                reason = verdict if isinstance(verdict, str) and verdict else (
                    f"authorization declined for {proposal.tool_name!r}"
                )
                self._emit("authorize", tool.name, session_id, False, reason)
                raise ToolAuthorizationFailed(reason)
            self._emit("authorize", tool.name, session_id, True)

        key = (session_id, tool.name)
        if tool.max_calls_per_session is not None:
            if self._call_counts.get(key, 0) >= tool.max_calls_per_session:
                reason = (
                    f"{proposal.tool_name!r} exceeded "
                    f"{tool.max_calls_per_session} call(s) per session"
                )
                self._emit("rate_limit", tool.name, session_id, False, reason)
                raise ToolRateLimited(reason)
            self._emit("rate_limit", tool.name, session_id, True)

        # Counted at dispatch, not on success: a call that errored partway may
        # still have caused side effects, so it must consume budget.
        self._call_counts[key] = self._call_counts.get(key, 0) + 1

        try:
            data = await tool.execute(session_id, proposal.arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced as failed evidence
            error = f"{type(exc).__name__}: {exc}"
            self._emit("execute", tool.name, session_id, False, error)
            return ToolResult(
                proposal_id=proposal.id,
                tool_name=tool.name,
                ok=False,
                error=error,
            )

        self._emit("execute", tool.name, session_id, True)
        return ToolResult(
            proposal_id=proposal.id,
            tool_name=tool.name,
            ok=True,
            data=_redact(data, tool.redact_fields),
        )


def _redact(data: Any, redact_fields: list[str]) -> Any:
    """Replace top-level ``redact_fields`` keys in a dict result with ``REDACTED``.

    Non-dict results pass through untouched -- there are no named fields to
    scrub. Redaction is applied inside the gateway so no caller can accidentally
    observe, log, or remember the raw value.
    """
    if not redact_fields or not isinstance(data, dict):
        return data
    targets = set(redact_fields)
    return {key: (REDACTED if key in targets else value) for key, value in data.items()}
