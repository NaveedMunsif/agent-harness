"""A CLARIFICATION exit, resolved by the caller calling back in. No API key required.

    python examples/clarification_flow.py

The thing to notice: ``handle_turn`` *returns* when the model needs to ask
something. It does not suspend, block, or hold an open coroutine. The caller owns
the wait -- print the question, collect an answer over whatever transport it has
(HTTP request, websocket, SMS, next day), then call ``handle_turn`` again with
that answer and the frame it got back. The controller keeps no state between
calls beyond what is passed back in.

Reuses the harness from ``order_tracking.py`` so the two examples are the same
system, not two different ones.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agent_harness import InMemoryBackend, MemoryKind, TaskFrame  # noqa: E402
from order_tracking import SESSION, build_controller  # noqa: E402


def show(label: str, result) -> None:
    print(f"\n  <- {label}")
    print(f"     stopped_reason : {result.stopped_reason}")
    print(f"     text           : {result.final_text}")
    print(f"     intent         : {result.task_frame.intent}")
    print(f"     entities       : {result.task_frame.entities}")
    print(f"     turns_used     : {result.turns_used}")


async def main() -> None:
    backend = InMemoryBackend()
    controller = build_controller(backend)

    print("=" * 70)
    print("CLARIFICATION -> CALLER ANSWERS -> FINAL")
    print("=" * 70)

    # -- Turn 1: too vague to act on. ---------------------------------------
    first = "Can you check on my order?"
    print(f"\nuser: {first}")
    asked = await controller.handle_turn(SESSION, first, current_frame=None)
    show("clarification_needed", asked)

    assert asked.stopped_reason == "clarification_needed"
    assert asked.tool_results == []
    print("\n     No OUTPUT guardrail ran and no formatter was applied: a question")
    print("     is not a claim, so there is nothing to ground or render.")

    # The caller now owns the wait. Nothing is held open on the harness side --
    # this could be a new HTTP request an hour later.
    print(f"\nagent: {asked.final_text}")

    # -- Turn 2: the user's answer, carried back in with the same frame. -----
    answer = "It's B-2002 - where is it?"
    print(f"\nuser: {answer}")
    resolved = await controller.handle_turn(
        SESSION, answer, current_frame=asked.task_frame  # <- the same frame
    )
    show("final_answer", resolved)

    assert resolved.stopped_reason == "final_answer"
    print("\n     Continuation, not a pivot: same intent, no conflicting entity, so")
    print("     ContextEngine.build extended the frame via merged_with().")
    print(f"\nagent: {resolved.final_text}")

    # -- Turn 3: a genuine pivot. -------------------------------------------
    # Same intent, but a different resolved order number conflicts with the one
    # held in the frame -- so the frame is rebuilt via TaskFrame.fresh() and the
    # old task's plan and in-flight tool ids are dropped.
    pivot_message = "Actually, where is order A-1001?"
    print(f"\nuser: {pivot_message}")
    print(
        "     diverges_from -> "
        f"{resolved.task_frame.diverges_from('track_order', {'order_id': 'A-1001'})}"
    )
    pivoted = await controller.handle_turn(
        SESSION, pivot_message, current_frame=resolved.task_frame
    )
    show("final_answer (after pivot)", pivoted)
    # A stale-evidence answer would trip the OUTPUT guardrail instead: the model
    # must look up A-1001 rather than reuse B-2002's result from turn 2.
    assert pivoted.stopped_reason == "final_answer", pivoted.stopped_reason
    assert len(pivoted.tool_results) == 1
    print(f"\nagent: {pivoted.final_text}")

    # -- The audit trail across all three turns. ----------------------------
    print("\n--- episodic memory after all three turns " + "-" * 28)
    print("    turn 1 (clarification): one turn_complete record, no tool records")
    print("    turns 2-3 (final)     : a tool_call record then a turn_complete record")
    print()
    for index, record in enumerate(backend.all_records(), start=1):
        if record.kind is not MemoryKind.EPISODIC:
            continue
        event = record.metadata.get("event")
        body = record.content.replace("\n", " | ")
        print(f"    {index}. [{event}] {body}")

    frame: TaskFrame = pivoted.task_frame
    print(f"\n    final frame: intent={frame.intent!r} entities={frame.entities}")


if __name__ == "__main__":
    asyncio.run(main())
