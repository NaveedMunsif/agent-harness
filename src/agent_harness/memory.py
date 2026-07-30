"""Typed memory: semantic, episodic, procedural.

Memory is typed because the three kinds answer different questions and must be
retrieved differently:

* ``SEMANTIC`` -- durable facts ("customer is on the Pro plan").
* ``EPISODIC`` -- what happened ("looked up order A-1001, it shipped"). The
  harness writes these itself; append-only by convention.
* ``PROCEDURAL`` -- how to do things ("to refund, verify the order first").

Backends are injected. :class:`InMemoryBackend` is the reference implementation
so the library needs no vector store to run.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "MemoryKind",
    "MemoryRecord",
    "MemoryBackend",
    "InMemoryBackend",
    "MemoryStore",
]

_TOKEN = re.compile(r"\w+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryKind(str, Enum):
    """The three retrieval-distinct kinds of memory."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryRecord(BaseModel):
    """One stored memory."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: MemoryKind
    session_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


@runtime_checkable
class MemoryBackend(Protocol):
    """Storage and retrieval for :class:`MemoryRecord` objects.

    **Empty-query contract.** If ``query`` is an empty string, a backend returns
    the ``limit`` most recent records for that ``session_id`` and ``kind`` rather
    than attempting a relevance match or returning nothing. In practice
    :meth:`~agent_harness.context.ContextEngine.build` falls back to the derived
    intent (``semantic_query or intent``), so an empty string rarely reaches a
    backend -- but custom backends must still honour this.
    """

    async def write(self, record: MemoryRecord) -> None: ...

    async def query(
        self,
        session_id: str,
        kind: MemoryKind,
        query: str,
        limit: int,
    ) -> list[MemoryRecord]: ...


class InMemoryBackend:
    """Reference backend: in-process list with token-overlap relevance.

    Ranking is ``(overlap score, recency)`` descending, so an empty query -- which
    scores every record zero -- naturally degrades to "the ``limit`` most recent",
    satisfying the empty-query contract without a special case. Records are never
    filtered out for scoring zero; a caller asking for ``limit`` records gets up
    to that many whenever the session has them.

    Swap in a vector store for real relevance; this exists so the library has no
    hard retrieval dependency.
    """

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    async def write(self, record: MemoryRecord) -> None:
        self._records.append(record)

    async def query(
        self,
        session_id: str,
        kind: MemoryKind,
        query: str,
        limit: int,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []

        terms = set(_TOKEN.findall(query.lower()))
        candidates = [
            (position, record)
            for position, record in enumerate(self._records)
            if record.session_id == session_id and record.kind == kind
        ]

        def rank(entry: tuple[int, MemoryRecord]) -> tuple[float, int]:
            position, record = entry
            if not terms:
                return (0.0, position)
            hits = terms & set(_TOKEN.findall(record.content.lower()))
            return (len(hits) / len(terms), position)

        candidates.sort(key=rank, reverse=True)
        return [record for _, record in candidates[:limit]]

    def all_records(self) -> list[MemoryRecord]:
        """Every record, in write order. Useful for tests and audit dumps."""
        return list(self._records)


class MemoryStore:
    """Thin typed façade over a :class:`MemoryBackend`."""

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        self.backend: MemoryBackend = backend or InMemoryBackend()

    async def remember(
        self,
        session_id: str,
        kind: MemoryKind,
        content: str,
        **metadata: Any,
    ) -> MemoryRecord:
        """Write one memory and return the stored record.

        Episodic writes are append-only by convention -- the harness records what
        happened and never rewrites it.
        """
        record = MemoryRecord(
            kind=kind,
            session_id=session_id,
            content=content,
            metadata=dict(metadata),
        )
        await self.backend.write(record)
        return record

    async def recall(
        self,
        session_id: str,
        kind: MemoryKind,
        query: str = "",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Retrieve up to ``limit`` records. See the empty-query contract above."""
        return await self.backend.query(session_id, kind, query, limit)
