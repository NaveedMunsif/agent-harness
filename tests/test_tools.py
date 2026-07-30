"""ToolGateway: authorization, redaction, rate limiting, allowlisting."""

from __future__ import annotations

import pytest

from agent_harness import (
    REDACTED,
    Tool,
    ToolAuthorizationFailed,
    ToolGateway,
    ToolNotAllowed,
    ToolProposal,
    ToolRateLimited,
)
from conftest import SESSION, lookup_order_tool


def gateway_with(tool: Tool) -> ToolGateway:
    gateway = ToolGateway()
    gateway.register(tool)
    return gateway


async def test_successful_call_returns_ok_result_with_data():
    tool = lookup_order_tool()
    gateway = gateway_with(tool)

    result = await gateway.execute(SESSION, ToolProposal("lookup_order", {"order_id": "A-1"}))

    assert result.ok is True
    assert result.error is None
    assert result.data["order_id"] == "A-1"
    assert result.tool_name == "lookup_order"
    # session_id is threaded through to the tool body.
    assert tool.calls == [{"session_id": SESSION, "order_id": "A-1"}]


# ------------------------------------------------------------------- authorization


async def test_authorization_declined_raises_and_never_executes():
    async def deny(session_id: str, arguments: dict) -> bool:
        return False

    tool = lookup_order_tool(authorize=deny)
    gateway = gateway_with(tool)

    with pytest.raises(ToolAuthorizationFailed):
        await gateway.execute(SESSION, ToolProposal("lookup_order", {"order_id": "A-1"}))

    assert tool.calls == []


async def test_authorization_granted_allows_the_call():
    seen: list[tuple[str, dict]] = []

    async def allow(session_id: str, arguments: dict) -> bool:
        seen.append((session_id, arguments))
        return True

    tool = lookup_order_tool(authorize=allow)
    result = await gateway_with(tool).execute(SESSION, ToolProposal("lookup_order", {"x": 1}))

    assert result.ok is True
    assert seen == [(SESSION, {"x": 1})]


async def test_authorization_can_decide_per_session():
    async def only_owner(session_id: str, arguments: dict) -> bool:
        return session_id == "owner"

    gateway = gateway_with(lookup_order_tool(authorize=only_owner))

    assert (await gateway.execute("owner", ToolProposal("lookup_order", {}))).ok is True
    with pytest.raises(ToolAuthorizationFailed):
        await gateway.execute("stranger", ToolProposal("lookup_order", {}))


# ----------------------------------------------------------------------- redaction


async def test_redaction_scrubs_named_fields_and_leaves_the_rest():
    tool = lookup_order_tool(redact_fields=["card_number"])

    result = await gateway_with(tool).execute(
        SESSION, ToolProposal("lookup_order", {"order_id": "A-1"})
    )

    assert result.data["card_number"] == REDACTED
    assert result.data["order_id"] == "A-1"
    assert result.data["status"] == "shipped"


async def test_redaction_of_absent_field_is_a_no_op():
    tool = lookup_order_tool(redact_fields=["ssn"])

    result = await gateway_with(tool).execute(SESSION, ToolProposal("lookup_order", {}))

    assert "ssn" not in result.data
    assert result.data["card_number"] == "4111111111111111"


async def test_redaction_leaves_non_dict_results_untouched():
    async def execute(session_id: str, arguments: dict) -> str:
        return "a plain string"

    tool = Tool(
        name="echo",
        description="echo",
        parameters=[],
        execute=execute,
        redact_fields=["card_number"],
    )

    result = await gateway_with(tool).execute(SESSION, ToolProposal("echo", {}))

    assert result.data == "a plain string"


# --------------------------------------------------------------------- rate limits


async def test_rate_limit_raises_once_the_session_budget_is_spent():
    tool = lookup_order_tool(max_calls_per_session=2)
    gateway = gateway_with(tool)

    assert (await gateway.execute(SESSION, ToolProposal("lookup_order", {}))).ok is True
    assert (await gateway.execute(SESSION, ToolProposal("lookup_order", {}))).ok is True

    with pytest.raises(ToolRateLimited):
        await gateway.execute(SESSION, ToolProposal("lookup_order", {}))

    assert gateway.call_count(SESSION, "lookup_order") == 2
    assert len(tool.calls) == 2


async def test_rate_limit_budget_is_per_session():
    gateway = gateway_with(lookup_order_tool(max_calls_per_session=1))

    assert (await gateway.execute("s1", ToolProposal("lookup_order", {}))).ok is True
    # A different session has its own untouched budget.
    assert (await gateway.execute("s2", ToolProposal("lookup_order", {}))).ok is True
    with pytest.raises(ToolRateLimited):
        await gateway.execute("s1", ToolProposal("lookup_order", {}))


async def test_failed_execution_still_consumes_rate_limit_budget():
    # A call that raised partway may still have caused side effects, so it must
    # count against the budget.
    gateway = gateway_with(
        lookup_order_tool(max_calls_per_session=1, raises=RuntimeError("upstream down"))
    )

    result = await gateway.execute(SESSION, ToolProposal("lookup_order", {}))
    assert result.ok is False

    with pytest.raises(ToolRateLimited):
        await gateway.execute(SESSION, ToolProposal("lookup_order", {}))


async def test_unlimited_by_default():
    gateway = gateway_with(lookup_order_tool())
    for _ in range(5):
        assert (await gateway.execute(SESSION, ToolProposal("lookup_order", {}))).ok is True


# -------------------------------------------------------------------- allowlisting


async def test_unknown_tool_is_not_allowed():
    gateway = gateway_with(lookup_order_tool())

    with pytest.raises(ToolNotAllowed):
        await gateway.execute(SESSION, ToolProposal("delete_everything", {}))


async def test_registered_tool_excluded_by_allowlist_is_refused():
    tool = lookup_order_tool()
    gateway = gateway_with(tool)

    with pytest.raises(ToolNotAllowed):
        await gateway.execute(SESSION, ToolProposal("lookup_order", {}), allowlist=["refund"])

    assert tool.calls == []


async def test_allowlist_containing_the_tool_permits_it():
    gateway = gateway_with(lookup_order_tool())
    result = await gateway.execute(
        SESSION, ToolProposal("lookup_order", {}), allowlist=["lookup_order"]
    )
    assert result.ok is True


def test_allowed_tools_filters_by_allowlist():
    gateway = ToolGateway([lookup_order_tool()])
    gateway.register(
        Tool(name="refund", description="refund", parameters=[], execute=lookup_order_tool().execute)
    )

    assert {t.name for t in gateway.allowed_tools()} == {"lookup_order", "refund"}
    assert [t.name for t in gateway.allowed_tools(["refund"])] == ["refund"]
    # An empty allowlist means "no tools this turn", distinct from None.
    assert gateway.allowed_tools([]) == []


# ------------------------------------------------------------- execution failures


async def test_tool_raising_becomes_a_failed_result_not_an_exception():
    gateway = gateway_with(lookup_order_tool(raises=RuntimeError("upstream down")))

    result = await gateway.execute(SESSION, ToolProposal("lookup_order", {}))

    assert result.ok is False
    assert result.data is None
    assert "RuntimeError" in result.error
    assert "upstream down" in result.error


async def test_result_carries_the_originating_proposal_id():
    proposal = ToolProposal("lookup_order", {})
    result = await gateway_with(lookup_order_tool()).execute(SESSION, proposal)
    assert result.proposal_id == proposal.id
