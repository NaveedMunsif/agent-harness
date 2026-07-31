"""TaskFrame pivot-versus-continuation semantics, and the branch in ContextEngine.build."""

from __future__ import annotations

from agent_harness import (
    MemoryKind,
    MemoryStore,
    PromptCompiler,
    ScopedContext,
    TaskFrame,
)
from conftest import SESSION, KeywordContextEngine, RecordingBackend


def frame(**overrides) -> TaskFrame:
    base = dict(
        session_id=SESSION,
        intent="track",
        entities={"order": "A-1"},
        active_plan=["lookup", "summarize"],
        pending_tool_call_ids={"call-1"},
    )
    return TaskFrame(**{**base, **overrides})


# -------------------------------------------------------------- construction


def test_mutable_fields_are_optional_at_construction():
    minimal = TaskFrame(session_id=SESSION, intent="track")

    assert minimal.entities == {}
    assert minimal.active_plan == []
    assert minimal.pending_tool_call_ids == set()


def test_intent_is_optional_too():
    assert TaskFrame(session_id=SESSION).intent is None


# --------------------------------------------------------------- diverges_from


def test_a_different_intent_diverges():
    assert frame().diverges_from("refund", {}) is True


def test_the_same_intent_with_no_entity_conflict_continues():
    assert frame().diverges_from("track", {"order": "A-1"}) is False


def test_a_conflicting_shared_entity_diverges():
    # Same intent, different order number: a new task, not a continuation.
    assert frame().diverges_from("track", {"order": "B-2"}) is True


def test_an_entity_present_on_only_one_side_does_not_diverge():
    assert frame().diverges_from("track", {"carrier": "ups"}) is False
    assert frame(entities={"order": "A-1", "carrier": "ups"}).diverges_from(
        "track", {"order": "A-1"}
    ) is False


def test_a_frame_with_no_intent_never_diverges():
    assert frame(intent=None).diverges_from("refund", {"order": "Z-9"}) is False


def test_a_falsy_new_intent_carries_no_intent_information():
    # "" means the caller derived no intent, not "the intent changed to nothing".
    assert frame().diverges_from("", {"order": "A-1"}) is False
    assert frame().diverges_from(None, {}) is False


def test_new_entities_defaults_to_empty():
    assert frame().diverges_from("track") is False


# ---------------------------------------------------------------- merged_with


def test_merged_with_lets_new_entity_values_win():
    merged = frame().merged_with("track", {"order": "A-1", "carrier": "ups"})

    assert merged.entities == {"order": "A-1", "carrier": "ups"}


def test_merged_with_keeps_the_existing_intent_when_none_is_supplied():
    assert frame().merged_with("", {}).intent == "track"
    assert frame().merged_with(None, {}).intent == "track"
    assert frame().merged_with("track_v2", {}).intent == "track_v2"


def test_merged_with_preserves_the_plan_unless_one_is_supplied():
    assert frame().merged_with("track", {}).active_plan == ["lookup", "summarize"]
    assert frame().merged_with("track", {}, active_plan=["refund"]).active_plan == ["refund"]
    # An explicitly empty plan does clear it.
    assert frame().merged_with("track", {}, active_plan=[]).active_plan == []


def test_merged_with_carries_over_pending_tool_call_ids():
    assert frame().merged_with("track", {}).pending_tool_call_ids == {"call-1"}


def test_merged_with_does_not_mutate_the_original():
    original = frame()
    original.merged_with("track", {"carrier": "ups"}, active_plan=["x"])

    assert original.entities == {"order": "A-1"}
    assert original.active_plan == ["lookup", "summarize"]


# --------------------------------------------------------------------- fresh


def test_fresh_drops_plan_and_pending_calls():
    pivoted = TaskFrame.fresh(SESSION, "refund", {"order": "B-2"})

    assert pivoted.session_id == SESSION
    assert pivoted.intent == "refund"
    assert pivoted.entities == {"order": "B-2"}
    assert pivoted.active_plan == []
    assert pivoted.pending_tool_call_ids == set()


def test_fresh_accepts_no_entities():
    assert TaskFrame.fresh(SESSION, "refund").entities == {}


# ------------------------------------------------- ContextEngine.build branch


def engine() -> tuple[KeywordContextEngine, RecordingBackend]:
    backend = RecordingBackend()
    return KeywordContextEngine(MemoryStore(backend)), backend


async def test_build_with_no_current_frame_is_a_pivot_via_fresh():
    ctx_engine, _ = engine()

    context = await ctx_engine.build(SESSION, "track order=A-1", None)

    assert isinstance(context, ScopedContext)
    assert context.is_pivot is True
    assert context.task_frame.intent == "track"
    assert context.task_frame.entities == {"order": "A-1"}
    assert context.task_frame.active_plan == []


async def test_build_on_a_matching_frame_continues_via_merged_with():
    ctx_engine, _ = engine()
    current = frame(entities={"order": "A-1"})

    context = await ctx_engine.build(SESSION, "track carrier=ups", current)

    assert context.is_pivot is False
    # merged_with semantics: entities folded together, plan and pending ids kept.
    assert context.task_frame.entities == {"order": "A-1", "carrier": "ups"}
    assert context.task_frame.active_plan == ["lookup", "summarize"]
    assert context.task_frame.pending_tool_call_ids == {"call-1"}


async def test_build_on_a_divergent_intent_pivots_via_fresh():
    ctx_engine, _ = engine()

    context = await ctx_engine.build(SESSION, "refund order=A-1", frame())

    assert context.is_pivot is True
    assert context.task_frame.intent == "refund"
    # fresh() semantics: the abandoned task's plan and in-flight calls are gone.
    assert context.task_frame.active_plan == []
    assert context.task_frame.pending_tool_call_ids == set()


async def test_build_on_a_conflicting_entity_pivots_even_with_the_same_intent():
    ctx_engine, _ = engine()

    context = await ctx_engine.build(SESSION, "track order=B-2", frame())

    assert context.is_pivot is True
    assert context.task_frame.entities == {"order": "B-2"}
    assert context.task_frame.active_plan == []


# ------------------------------------------- the topic-change note in the prompt


TOPIC_NOTE = "note: the user changed topic"


async def test_first_turn_is_a_pivot_but_had_no_prior_frame():
    ctx_engine, _ = engine()

    context = await ctx_engine.build(SESSION, "track order=A-1", None)

    # A fresh frame was built, but not *away from* anything.
    assert context.is_pivot is True
    assert context.had_prior_frame is False


async def test_a_real_pivot_had_a_prior_frame():
    ctx_engine, _ = engine()

    context = await ctx_engine.build(SESSION, "refund order=A-1", frame())

    assert context.is_pivot is True
    assert context.had_prior_frame is True


async def test_opening_message_is_not_told_the_topic_changed():
    """The bug: every session's first prompt claimed the user changed topic."""
    ctx_engine, _ = engine()
    context = await ctx_engine.build(SESSION, "track order=A-1", None)

    compiled = PromptCompiler(role="r").compile(context, [], "track order=A-1")

    assert TOPIC_NOTE not in compiled.sections["task_frame"]
    assert TOPIC_NOTE not in compiled.render()


async def test_an_actual_topic_change_still_says_so():
    ctx_engine, _ = engine()
    context = await ctx_engine.build(SESSION, "refund order=A-1", frame())

    compiled = PromptCompiler(role="r").compile(context, [], "refund order=A-1")

    assert TOPIC_NOTE in compiled.sections["task_frame"]


async def test_a_continuation_never_says_so():
    ctx_engine, _ = engine()
    context = await ctx_engine.build(SESSION, "track carrier=ups", frame())

    compiled = PromptCompiler(role="r").compile(context, [], "track carrier=ups")

    assert context.is_pivot is False
    assert TOPIC_NOTE not in compiled.sections["task_frame"]


async def test_build_populates_each_memory_kind():
    ctx_engine, _ = engine()
    memory = ctx_engine.memory
    await memory.remember(SESSION, MemoryKind.SEMANTIC, "customer is on the Pro plan")
    await memory.remember(SESSION, MemoryKind.EPISODIC, "user asked about order A-1")
    await memory.remember(SESSION, MemoryKind.PROCEDURAL, "verify identity before refunding")

    context = await ctx_engine.build(SESSION, "track order=A-1", None)

    assert context.semantic == ["customer is on the Pro plan"]
    assert context.episodic == ["user asked about order A-1"]
    assert context.procedural == ["verify identity before refunding"]


async def test_build_scopes_memory_to_the_session():
    ctx_engine, _ = engine()
    await ctx_engine.memory.remember("other-session", MemoryKind.SEMANTIC, "someone else's fact")

    context = await ctx_engine.build(SESSION, "track", None)

    assert context.semantic == []


async def test_retrieval_queries_fall_back_to_the_derived_intent():
    ctx_engine, backend = engine()

    await ctx_engine.build(SESSION, "track order=A-1", None)

    assert backend.queries_for(MemoryKind.SEMANTIC)[0][1] == "track"
    assert backend.queries_for(MemoryKind.PROCEDURAL)[0][1] == "track"
    # Episodic retrieval is recency-ordered, not query-driven.
    assert backend.queries_for(MemoryKind.EPISODIC)[0][1] == ""


async def test_explicit_queries_override_the_intent_fallback():
    ctx_engine, backend = engine()

    await ctx_engine.build(
        SESSION, "track order=A-1", None, semantic_query="shipping", procedural_query="escalation"
    )

    assert backend.queries_for(MemoryKind.SEMANTIC)[0][1] == "shipping"
    assert backend.queries_for(MemoryKind.PROCEDURAL)[0][1] == "escalation"


async def test_empty_query_returns_the_most_recent_records():
    backend = RecordingBackend()
    memory = MemoryStore(backend)
    for index in range(5):
        await memory.remember(SESSION, MemoryKind.EPISODIC, f"event {index}")

    recalled = await memory.recall(SESSION, MemoryKind.EPISODIC, "", limit=2)

    assert [record.content for record in recalled] == ["event 4", "event 3"]
