"""LoopController: the three exits, the forced stops, and step validation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from agent_harness import (
    LLMTurnStep,
    LoopLimits,
    RenderChannel,
    TaskFrame,
    TurnStepType,
)
from conftest import (
    SESSION,
    ScriptedLLM,
    ToolSpamLLM,
    UppercaseFormatter,
    build_harness,
    clarification,
    final,
    lookup_order_tool,
    tool_call,
)

# ------------------------------------------------------- LLMTurnStep invariants


def test_tool_call_requires_a_tool_name():
    with pytest.raises(ValidationError, match="requires a non-empty tool_name"):
        LLMTurnStep(step_type=TurnStepType.TOOL_CALL, tool_arguments={"order_id": "A-1"})


def test_tool_call_requires_arguments_even_if_empty():
    with pytest.raises(ValidationError, match="requires tool_arguments"):
        LLMTurnStep(step_type=TurnStepType.TOOL_CALL, tool_name="lookup_order")

    # An empty dict is a legitimate argument set; None is not.
    assert LLMTurnStep(
        step_type=TurnStepType.TOOL_CALL, tool_name="lookup_order", tool_arguments={}
    ).tool_arguments == {}


def test_blank_tool_name_is_rejected():
    with pytest.raises(ValidationError):
        LLMTurnStep(step_type=TurnStepType.TOOL_CALL, tool_name="", tool_arguments={})


@pytest.mark.parametrize("step_type", [TurnStepType.FINAL, TurnStepType.CLARIFICATION])
def test_text_bearing_steps_require_text(step_type):
    with pytest.raises(ValidationError, match="requires non-empty text"):
        LLMTurnStep(step_type=step_type)
    with pytest.raises(ValidationError, match="requires non-empty text"):
        LLMTurnStep(step_type=step_type, text="")


def test_well_formed_steps_construct():
    assert tool_call("lookup_order", order_id="A-1").tool_name == "lookup_order"
    assert final("Shipped.").text == "Shipped."
    assert clarification("Which order?").text == "Which order?"


# --------------------------------------------------------------- the FINAL exit


async def test_tool_call_then_final_applies_guardrail_and_formatter():
    formatter = UppercaseFormatter()
    harness = build_harness(
        steps=[tool_call("lookup_order", order_id="A-1"), final("Order A-1 shipped.")],
        tools=[lookup_order_tool()],
        formatter=formatter,
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert result.final_text == "ORDER A-1 SHIPPED."
    assert formatter.calls == [("Order A-1 shipped.", RenderChannel.PLAIN_TEXT)]
    assert result.turns_used == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].ok is True


async def test_render_channel_is_passed_to_the_formatter():
    formatter = UppercaseFormatter()
    harness = build_harness(steps=[final("hi")], formatter=formatter)

    await harness.controller.handle_turn(
        SESSION, "greet", render_channel=RenderChannel.MARKDOWN
    )

    assert formatter.calls == [("hi", RenderChannel.MARKDOWN)]


async def test_without_a_formatter_the_raw_text_is_returned():
    harness = build_harness(steps=[final("Order A-1 shipped.")])

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.final_text == "Order A-1 shipped."


async def test_final_returns_the_active_frame_from_context_build():
    harness = build_harness(steps=[final("done")])

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert isinstance(result.task_frame, TaskFrame)
    assert result.task_frame.intent == "track"
    assert result.task_frame.entities == {"order": "A-1"}


async def test_final_writes_tool_and_turn_completion_records():
    harness = build_harness(
        steps=[tool_call("lookup_order", order_id="A-1"), final("Order A-1 shipped.")],
        tools=[lookup_order_tool()],
    )

    await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert harness.episodic_events() == ["tool_call", "turn_complete"]
    records = harness.episodic()
    assert "tool lookup_order succeeded" in records[0].content
    assert records[0].metadata["tool"] == "lookup_order"
    assert records[0].metadata["ok"] is True
    # The consolidated record is the human-readable summary of the whole turn.
    assert "user: track order=A-1" in records[1].content
    assert "assistant: Order A-1 shipped." in records[1].content


async def test_tool_evidence_reaches_the_next_prompt_through_episodic_context():
    # There is no separate tool_history channel: results are folded into
    # context.episodic before the recompile.
    harness = build_harness(
        steps=[tool_call("lookup_order", order_id="A-1"), final("Shipped.")],
        tools=[lookup_order_tool()],
    )

    await harness.controller.handle_turn(SESSION, "track order=A-1")

    first, second = harness.llm.prompts
    assert "tool lookup_order succeeded" not in first.render()
    assert "tool lookup_order succeeded" in second.render()


async def test_redacted_fields_never_reach_memory_or_the_next_prompt():
    harness = build_harness(
        steps=[tool_call("lookup_order", order_id="A-1"), final("Shipped.")],
        tools=[lookup_order_tool(redact_fields=["card_number"])],
    )

    await harness.controller.handle_turn(SESSION, "track order=A-1")

    everything = "\n".join(r.content for r in harness.episodic())
    everything += harness.llm.prompts[1].render()
    assert "4111111111111111" not in everything
    assert "[REDACTED]" in everything


# ------------------------------------------------------- the CLARIFICATION exit


async def test_clarification_returns_the_question_and_skips_output_handling():
    formatter = UppercaseFormatter()
    harness = build_harness(
        steps=[clarification("Which order number?")], formatter=formatter
    )

    result = await harness.controller.handle_turn(SESSION, "track my order")

    assert result.stopped_reason == "clarification_needed"
    assert result.final_text == "Which order number?"
    # A question is always plain text; the calling UI decides presentation.
    assert formatter.calls == []
    # It counts as a turn, like any other model round trip.
    assert result.turns_used == 1
    assert result.tool_results == []


async def test_clarification_is_a_plain_return_not_a_suspension():
    harness = build_harness(steps=[clarification("Which order?"), final("Shipped.")])

    result = await harness.controller.handle_turn(SESSION, "track my order")

    # Returned without consuming the rest of the script: the loop stopped dead,
    # it did not hold a coroutine open waiting for an answer.
    assert result.stopped_reason == "clarification_needed"
    assert harness.llm.call_count == 1
    assert len(harness.llm.steps) == 1


async def test_clarification_writes_a_turn_completion_record():
    harness = build_harness(steps=[clarification("Which order number?")])

    await harness.controller.handle_turn(SESSION, "track my order")

    assert harness.episodic_events() == ["turn_complete"]
    assert "assistant: Which order number?" in harness.episodic()[0].content


async def test_caller_resolves_a_clarification_by_calling_handle_turn_again():
    harness = build_harness(
        steps=[
            clarification("Which order number?"),
            tool_call("lookup_order", order_id="A-1"),
            final("Order A-1 shipped."),
        ],
        tools=[lookup_order_tool()],
    )

    asked = await harness.controller.handle_turn(SESSION, "track my order")
    assert asked.stopped_reason == "clarification_needed"

    # The caller passes the answer back as a new message with the same frame.
    resolved = await harness.controller.handle_turn(
        SESSION, "track order=A-1", asked.task_frame
    )

    assert resolved.stopped_reason == "final_answer"
    assert resolved.final_text == "Order A-1 shipped."
    # Continuation, not a pivot: the frame carried the intent forward.
    assert resolved.task_frame.intent == "track"
    assert resolved.task_frame.entities == {"order": "A-1"}


# ------------------------------------------------------------- the forced stops


async def test_max_tool_calls_exceeded_keeps_the_already_written_tool_records():
    harness = build_harness(call_llm=ToolSpamLLM(), tools=[lookup_order_tool()])

    result = await harness.controller.handle_turn(
        SESSION, "track order=A-1", limits=LoopLimits(max_turns=10, max_tool_calls=2)
    )

    assert result.stopped_reason == "max_tool_calls_exceeded"
    assert result.final_text is None
    assert len(result.tool_results) == 2

    # Write-as-you-go: both executions were recorded *before* the forced stop, so
    # real side effects are never erased by the loop being cut short.
    assert harness.episodic_events() == ["tool_call", "tool_call", "forced_stop"]
    assert "max_tool_calls_exceeded" in harness.episodic()[-1].content
    assert harness.episodic()[-1].metadata["stopped_reason"] == "max_tool_calls_exceeded"


async def test_max_turns_exceeded_stops_the_loop():
    harness = build_harness(call_llm=ToolSpamLLM(), tools=[lookup_order_tool()])

    result = await harness.controller.handle_turn(
        SESSION, "track order=A-1", limits=LoopLimits(max_turns=2, max_tool_calls=10)
    )

    assert result.stopped_reason == "max_turns_exceeded"
    assert result.final_text is None
    assert result.turns_used == 2
    assert harness.episodic_events()[-1] == "forced_stop"


async def test_max_seconds_exceeded_stops_before_any_model_call():
    harness = build_harness(steps=[final("never reached")])

    result = await harness.controller.handle_turn(
        SESSION, "track order=A-1", limits=LoopLimits(max_seconds=0.0)
    )

    assert result.stopped_reason == "max_seconds_exceeded"
    assert result.final_text is None
    assert harness.llm.call_count == 0


async def test_forced_stops_write_no_turn_completion_record():
    harness = build_harness(steps=[final("never reached")])

    await harness.controller.handle_turn(
        SESSION, "track", limits=LoopLimits(max_seconds=0.0)
    )

    assert "turn_complete" not in harness.episodic_events()


async def test_forced_stop_skips_the_output_formatter():
    formatter = UppercaseFormatter()
    harness = build_harness(steps=[final("x")], formatter=formatter)

    await harness.controller.handle_turn(
        SESSION, "track", limits=LoopLimits(max_seconds=0.0)
    )

    assert formatter.calls == []


# ------------------------------------------------------ repeated-call detection


async def test_a_repeated_proposal_is_corrected_rather_than_ending_the_turn():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("lookup_order", order_id="A-1"),  # repeat: corrected
            final("Order A-1 has shipped."),
        ],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert result.final_text == "Order A-1 has shipped."
    # The repeat never executed, so there is still exactly one result.
    assert len(result.tool_results) == 1
    assert harness.episodic_events() == ["tool_call", "turn_complete"]


async def test_a_repeat_after_the_correction_stops_the_loop():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("lookup_order", order_id="A-1"),  # corrected
            tool_call("lookup_order", order_id="A-1"),  # still looping: stop
            final("never reached"),
        ],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "repeated_call_detected"
    assert result.final_text is None
    assert len(result.tool_results) == 1
    assert harness.episodic_events() == ["tool_call", "forced_stop"]


async def test_the_correction_reaches_the_next_prompt():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("lookup_order", order_id="A-1"),
            final("Order A-1 has shipped."),
        ],
        tools=[lookup_order_tool()],
    )

    await harness.controller.handle_turn(SESSION, "track order=A-1")

    # The third compile is the one that happened after the correction was
    # appended, so the model can see it before answering.
    assert "was already called this turn" in harness.llm.prompts[-1].sections["history"]


async def test_argument_order_does_not_disguise_a_repeat():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1", carrier="ups"),
            tool_call("lookup_order", carrier="ups", order_id="A-1"),
            tool_call("lookup_order", order_id="A-1", carrier="ups"),
        ],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "repeated_call_detected"


async def test_different_arguments_are_not_a_repeat():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("lookup_order", order_id="B-2"),
            final("Both shipped."),
        ],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert len(result.tool_results) == 2


async def test_non_json_native_arguments_do_not_raise_a_type_error():
    # datetime / UUID / Decimal are unhashable-adjacent hazards: canonicalizing
    # with default=str is what keeps this from blowing up.
    exotic = {
        "when": datetime(2026, 7, 29, tzinfo=timezone.utc),
        "trace": uuid.UUID("12345678-1234-5678-1234-567812345678"),
        "amount": Decimal("10.50"),
    }
    harness = build_harness(
        steps=[
            LLMTurnStep(
                step_type=TurnStepType.TOOL_CALL,
                tool_name="lookup_order",
                tool_arguments=dict(exotic),
            ),
            LLMTurnStep(
                step_type=TurnStepType.TOOL_CALL,
                tool_name="lookup_order",
                tool_arguments=dict(exotic),
            ),
            LLMTurnStep(
                step_type=TurnStepType.TOOL_CALL,
                tool_name="lookup_order",
                tool_arguments=dict(exotic),
            ),
        ],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "repeated_call_detected"


async def test_the_same_arguments_to_a_different_tool_are_not_a_repeat():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("other", order_id="A-1"),
            final("done"),
        ],
        tools=[lookup_order_tool()],
    )
    other = lookup_order_tool()
    other.name = "other"
    harness.gateway.register(other)

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert len(result.tool_results) == 2


# ------------------------------------------------------------- tool policy errors


async def test_a_policy_refusal_becomes_failed_evidence_and_the_loop_continues():
    harness = build_harness(
        steps=[
            tool_call("lookup_order", order_id="A-1"),
            tool_call("lookup_order", order_id="B-2"),
            final("Only got one of them."),
        ],
        tools=[lookup_order_tool(max_calls_per_session=1)],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert result.tool_results[0].ok is True
    assert result.tool_results[1].ok is False
    assert "ToolRateLimited" in result.tool_results[1].error
    assert harness.episodic_events() == ["tool_call", "tool_call", "turn_complete"]


async def test_an_unknown_tool_proposal_is_recorded_and_survivable():
    harness = build_harness(
        steps=[tool_call("delete_everything"), final("I can't do that.")],
        tools=[lookup_order_tool()],
    )

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    assert result.stopped_reason == "final_answer"
    assert result.tool_results[0].ok is False
    assert "ToolNotAllowed" in result.tool_results[0].error


async def test_allowed_tools_narrows_what_the_model_is_offered():
    harness = build_harness(
        steps=[final("done")],
        tools=[lookup_order_tool()],
    )

    await harness.controller.handle_turn(SESSION, "track", allowed_tools=[])

    assert "lookup_order" not in harness.llm.prompts[0].render()


# ---------------------------------------------------------------------- defaults


async def test_default_limits_are_used_when_none_are_supplied():
    harness = build_harness(call_llm=ToolSpamLLM(), tools=[lookup_order_tool()])

    result = await harness.controller.handle_turn(SESSION, "track order=A-1")

    # Default max_tool_calls=4 bites before default max_turns=6.
    assert result.stopped_reason == "max_tool_calls_exceeded"
    assert len(result.tool_results) == 4


async def test_the_default_output_requirement_reaches_the_prompt():
    harness = build_harness(steps=[final("done")])

    await harness.controller.handle_turn(SESSION, "track")

    assert "ground every claim in tool results" in harness.llm.prompts[0].render()


async def test_scripted_llm_is_never_over_called():
    # Guards the test doubles themselves: an over-called script would silently
    # mask a loop that failed to stop.
    harness = build_harness(steps=[final("done")])
    await harness.controller.handle_turn(SESSION, "track")
    assert isinstance(harness.llm, ScriptedLLM)
    assert harness.llm.steps == []
