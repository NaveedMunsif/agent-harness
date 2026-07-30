"""Task-frame state: what the agent currently believes the user is doing.

A :class:`TaskFrame` is the small, explicit unit of continuity across turns. The
harness never infers continuity from raw chat history; it compares the newly
derived intent/entities against the live frame and either *continues* it
(:meth:`TaskFrame.merged_with`) or *pivots* to a clean one
(:meth:`TaskFrame.fresh`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["TaskFrame"]


class TaskFrame(BaseModel):
    """The agent's working understanding of the current task.

    All mutable-typed fields use ``default_factory`` so they are optional at
    construction: ``TaskFrame(session_id="s1", intent="track_order")`` is valid.

    ``intent`` is nullable so a frame can exist before any intent has been
    derived. :meth:`diverges_from` treats such a frame as never diverging, which
    is what makes "first message of a session" a continuation rather than a
    pivot away from nothing.
    """

    session_id: str
    intent: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    active_plan: list[str] = Field(default_factory=list)
    pending_tool_call_ids: set[str] = Field(default_factory=set)

    def diverges_from(
        self,
        new_intent: str | None,
        new_entities: dict[str, Any] | None = None,
    ) -> bool:
        """Return ``True`` if this frame should be abandoned for a fresh one.

        Divergence means either:

        * ``new_intent`` names a *different* intent than the one held here, or
        * a key present in **both** ``self.entities`` and ``new_entities`` has a
          conflicting value (e.g. the user switched to a different order number).

        A falsy ``new_intent`` carries no intent information and so never
        diverges on its own -- it is the signal that the caller could not derive
        an intent this turn, and pairs with :meth:`merged_with` keeping the
        existing intent. A frame whose own ``intent`` is ``None`` has nothing to
        diverge *from* and always returns ``False``.
        """
        if self.intent is None:
            return False

        if new_intent and new_intent != self.intent:
            return True

        for key, value in (new_entities or {}).items():
            if key in self.entities and self.entities[key] != value:
                return True

        return False

    def merged_with(
        self,
        new_intent: str | None,
        new_entities: dict[str, Any] | None = None,
        active_plan: list[str] | None = None,
    ) -> TaskFrame:
        """Continuation path: fold new information into a copy of this frame.

        New entity values take precedence over held ones. The existing intent is
        kept when ``new_intent`` is falsy. ``active_plan`` is replaced only when
        a plan is actually supplied, so a caller that has nothing to say about
        the plan cannot accidentally erase it. ``pending_tool_call_ids`` carries
        over -- in-flight work survives a continuation.
        """
        return TaskFrame(
            session_id=self.session_id,
            intent=new_intent or self.intent,
            entities={**self.entities, **(new_entities or {})},
            active_plan=list(active_plan) if active_plan is not None else list(self.active_plan),
            pending_tool_call_ids=set(self.pending_tool_call_ids),
        )

    @classmethod
    def fresh(
        cls,
        session_id: str,
        intent: str | None,
        entities: dict[str, Any] | None = None,
    ) -> TaskFrame:
        """Pivot path: a clean frame with no plan and no in-flight tool calls.

        Deliberately drops ``active_plan`` and ``pending_tool_call_ids``: a plan
        built for the abandoned task is not evidence about the new one.
        """
        return cls(
            session_id=session_id,
            intent=intent,
            entities=dict(entities or {}),
            active_plan=[],
            pending_tool_call_ids=set(),
        )
