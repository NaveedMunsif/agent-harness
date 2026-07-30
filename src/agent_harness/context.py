"""Context selection: deciding what the model gets to see this turn.

The model does not read memory; the harness reads memory *for* it. Every turn,
:meth:`ContextEngine.build` derives the intent, decides pivot-versus-continuation
explicitly, retrieves each memory kind, and hands back one
:class:`ScopedContext`.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .memory import MemoryKind, MemoryStore
from .state import TaskFrame

__all__ = ["ScopedContext", "ContextEngine"]

_WHITESPACE = re.compile(r"\s+")


class ScopedContext(BaseModel):
    """Exactly the context assembled for one turn.

    ``task_frame`` is the *active* frame for the remainder of the turn -- built by
    :meth:`ContextEngine.build` via either ``TaskFrame.fresh`` (pivot) or
    ``TaskFrame.merged_with`` (continuation). ``LoopController`` adopts it as-is
    and returns it as ``LoopResult.task_frame``; there is no separate replacement
    step.
    """

    task_frame: TaskFrame
    semantic: list[str] = Field(default_factory=list)
    episodic: list[str] = Field(default_factory=list)
    procedural: list[str] = Field(default_factory=list)
    is_pivot: bool = False


class ContextEngine:
    """Assembles per-turn context and owns the pivot decision."""

    def __init__(
        self,
        memory: MemoryStore,
        semantic_limit: int = 5,
        episodic_limit: int = 5,
        procedural_limit: int = 3,
    ) -> None:
        self.memory = memory
        self.semantic_limit = semantic_limit
        self.episodic_limit = episodic_limit
        self.procedural_limit = procedural_limit

    async def extract_intent(self, message: str) -> tuple[str, dict[str, Any]]:
        """Derive ``(intent, entities)`` from a raw user message. **Override point.**

        The default is deliberately trivial: the normalized message text as the
        intent and no entities. That means every materially different message
        reads as a new intent, which is safe but coarse. Real deployments
        subclass this and plug in a classifier, a router, or a small LLM call --
        that is where entity resolution (order numbers, account ids) belongs, and
        entities are what make :meth:`TaskFrame.diverges_from` precise.
        """
        return _WHITESPACE.sub(" ", message).strip().lower(), {}

    async def build(
        self,
        session_id: str,
        message: str,
        current_frame: TaskFrame | None = None,
        semantic_query: str = "",
        procedural_query: str = "",
    ) -> ScopedContext:
        """Assemble context for one turn, resolving pivot versus continuation.

        The frame is rebuilt from scratch via ``TaskFrame.fresh`` when there is no
        current frame or when the new intent/entities diverge from it
        (``is_pivot=True``); otherwise it is extended via
        ``current_frame.merged_with`` (``is_pivot=False``).

        Semantic and procedural retrieval fall back to the derived intent when no
        explicit query is given (``semantic_query or intent``). Episodic
        retrieval is recency-ordered rather than query-driven -- recent history is
        wanted because it is recent, not because it matches a search string,
        which is why there is no ``episodic_query`` parameter.
        """
        intent, entities = await self.extract_intent(message)

        is_pivot = current_frame is None or current_frame.diverges_from(intent, entities)
        if is_pivot:
            task_frame = TaskFrame.fresh(session_id, intent, entities)
        else:
            assert current_frame is not None  # narrowed by is_pivot
            task_frame = current_frame.merged_with(intent, entities)

        semantic = await self.memory.recall(
            session_id, MemoryKind.SEMANTIC, semantic_query or intent, self.semantic_limit
        )
        episodic = await self.memory.recall(
            session_id, MemoryKind.EPISODIC, "", self.episodic_limit
        )
        procedural = await self.memory.recall(
            session_id,
            MemoryKind.PROCEDURAL,
            procedural_query or intent,
            self.procedural_limit,
        )

        return ScopedContext(
            task_frame=task_frame,
            semantic=[record.content for record in semantic],
            episodic=[record.content for record in episodic],
            procedural=[record.content for record in procedural],
            is_pivot=is_pivot,
        )
