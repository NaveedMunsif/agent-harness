"""Versioned prompt assembly.

A prompt is a compiled artifact with a version, not a format string scattered
through application code -- so a change to prompt structure is a reviewable,
attributable change.

**No separate tool-history channel.** ``compile`` takes no ``tool_history``
parameter. Within a single turn, ``LoopController`` folds each executed
``ToolResult`` into ``context.episodic`` before recompiling, so evidence from
earlier iterations of the same loop reaches the model through the ordinary
episodic section. One path for "what happened", not two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .context import ScopedContext
from .tools import Tool

__all__ = ["CompiledPrompt", "PromptCompiler"]


@dataclass
class CompiledPrompt:
    """An assembled, versioned prompt. ``sections`` preserves insertion order."""

    version: str
    role: str
    sections: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """Flatten to the string handed to the model."""
        blocks = [f"# ROLE (prompt v{self.version})", self.role]
        for name, body in self.sections.items():
            if not body:
                continue
            blocks.append(f"# {name.upper().replace('_', ' ')}")
            blocks.append(body)
        return "\n\n".join(blocks)


class PromptCompiler:
    """Turns a :class:`ScopedContext` plus this turn's inputs into a prompt.

    Empty sections are omitted rather than rendered as empty headings, so the
    model is never shown a "TOOLS" heading with nothing under it.
    """

    def __init__(self, role: str, version: str = "1.0.0") -> None:
        self.role = role
        self.version = version

    def compile(
        self,
        context: ScopedContext,
        tools: list[Tool],
        user_message: str,
        output_requirement: str = "",
    ) -> CompiledPrompt:
        sections: dict[str, str] = {}

        frame = context.task_frame
        frame_lines = [f"session: {frame.session_id}", f"intent: {frame.intent or '(unknown)'}"]
        if frame.entities:
            frame_lines.append(
                "entities: " + ", ".join(f"{k}={v}" for k, v in sorted(frame.entities.items()))
            )
        if frame.active_plan:
            frame_lines.append("plan: " + " -> ".join(frame.active_plan))
        # Only a pivot *away from something* is a topic change. A fresh frame
        # built because the session just started has no earlier topic to have
        # left, and saying otherwise tells the model something untrue on the
        # opening message of every conversation.
        if context.is_pivot and context.had_prior_frame:
            frame_lines.append("note: the user changed topic; earlier task state was dropped.")
        sections["task_frame"] = "\n".join(frame_lines)

        if context.semantic:
            sections["known_facts"] = _bullets(context.semantic)
        if context.procedural:
            sections["procedures"] = _bullets(context.procedural)
        if context.episodic:
            sections["history"] = _bullets(context.episodic)

        if tools:
            sections["tools"] = "\n".join(
                f"- {tool.name}({', '.join(tool.parameters)}): {tool.description}"
                f"{'' if tool.read_only else ' [has side effects]'}"
                for tool in tools
            )

        if output_requirement:
            sections["output_requirement"] = output_requirement

        sections["user_message"] = user_message

        return CompiledPrompt(version=self.version, role=self.role, sections=sections)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
