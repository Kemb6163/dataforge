"""Incremental dataset statistics tracker.

All dict-based stats are capped to prevent unbounded RAM growth.
Overflow entries are bucketed into "_other".
"""

from __future__ import annotations

from typing import Any

from dataforge.core.types import Example, DatasetStats


_MAX_TOOL_ENTRIES = 500
_MAX_ROLE_ENTRIES = 20
_MAX_STRUCTURE_ENTRIES = 50


class StatsTracker:
    """Incremental statistics tracker with bounded memory."""

    def __init__(self):
        self._total = 0
        self._by_tool: dict[str, int] = {}
        self._by_role: dict[str, int] = {}
        self._multi_turn = 0
        self._no_tool_calls = 0
        self._parallel_tool_calls = 0
        self._response_structures: dict[str, int] = {}
        self._error_handling = 0
        self._total_messages = 0

    @property
    def total(self) -> int:
        return self._total

    def ingest(self, ex: Example) -> None:
        """Process one example."""
        self._total += 1
        msgs = ex.messages

        self._total_messages += len(msgs)

        # Count roles
        for msg in msgs:
            role = msg.get("role", "unknown")
            self._increment_capped(self._by_role, role, _MAX_ROLE_ENTRIES)

        # Count tool calls
        has_tool_call = False
        has_parallel = False
        user_turns = sum(1 for m in msgs if m.get("role") == "user")

        for msg in msgs:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    has_tool_call = True
                    if len(tool_calls) > 1:
                        has_parallel = True
                    for tc in tool_calls:
                        name = tc.get("function", {}).get("name", "unknown")
                        self._increment_capped(self._by_tool, name, _MAX_TOOL_ENTRIES)

        if not has_tool_call:
            self._no_tool_calls += 1

        if has_parallel:
            self._parallel_tool_calls += 1

        if user_turns > 1:
            self._multi_turn += 1

        # Detect error handling examples
        for msg in msgs:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and '"error": true' in content.lower():
                    self._error_handling += 1
                    break

        # Track conversation structure
        structure = self._extract_structure(msgs)
        self._increment_capped(self._response_structures, structure, _MAX_STRUCTURE_ENTRIES)

    @property
    def stats(self) -> DatasetStats:
        """Return current stats snapshot."""
        return DatasetStats(
            total=self._total,
            by_tool=dict(self._by_tool),
            by_role=dict(self._by_role),
            multi_turn=self._multi_turn,
            no_tool_calls=self._no_tool_calls,
            parallel_tool_calls=self._parallel_tool_calls,
            response_structures=dict(self._response_structures),
            error_handling=self._error_handling,
            avg_messages_per_example=(
                self._total_messages / self._total if self._total > 0 else 0
            ),
            total_messages=self._total_messages,
        )

    def _extract_structure(self, msgs: list[dict[str, Any]]) -> str:
        """Extract a structural signature for the conversation."""
        parts = []
        for msg in msgs:
            role = msg.get("role", "?")
            if role == "system":
                continue
            if role == "assistant" and msg.get("tool_calls"):
                n = len(msg["tool_calls"])
                parts.append(f"TC:{n}")
            elif role == "tool":
                parts.append("TR")
            else:
                parts.append(role[0].upper())
        return "|".join(parts)

    @staticmethod
    def _increment_capped(d: dict[str, int], key: str, max_entries: int) -> None:
        """Increment a counter, bucketing into '_other' if over max entries."""
        if key in d:
            d[key] += 1
        elif len(d) < max_entries:
            d[key] = 1
        else:
            d["_other"] = d.get("_other", 0) + 1
