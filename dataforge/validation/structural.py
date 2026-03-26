"""Structural validation for training examples.

Checks message role sequences, tool call matching, and tool name leaks.
"""

from __future__ import annotations

import re
from typing import Any

from dataforge.core.types import Example


def validate_example(
    ex: Example,
    idx: int,
    tool_names: list[str] | None = None,
) -> list[str]:
    """Validate a single training example for structural correctness.

    Args:
        ex: The Example to validate.
        idx: Example index (for error messages).
        tool_names: Known tool names — if provided, checks for tool name leaks
                    in assistant text content.

    Returns:
        List of error/warning strings. Empty list = valid.
    """
    errors: list[str] = []
    msgs = ex.messages

    if not msgs:
        errors.append(f"[{idx}] Empty message list")
        return errors

    # Check role sequence validity
    prev_role = None
    pending_tool_calls: dict[str, str] = {}  # call_id -> tool_name

    for i, msg in enumerate(msgs):
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errors.append(f"[{idx}] msg[{i}]: invalid role '{role}'")
            continue

        if role == "system" and i != 0:
            errors.append(f"[{idx}] msg[{i}]: system message not at position 0")

        if role == "tool":
            call_id = msg.get("tool_call_id")
            if not call_id:
                errors.append(f"[{idx}] msg[{i}]: tool msg missing tool_call_id")
            elif call_id not in pending_tool_calls:
                errors.append(f"[{idx}] msg[{i}]: tool_call_id '{call_id}' has no matching tool_call")
            else:
                del pending_tool_calls[call_id]

        # Track tool calls from assistant messages
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                tc_id = tc.get("id")
                tc_name = tc.get("function", {}).get("name", "")
                if tc_id:
                    if tc_id in pending_tool_calls:
                        errors.append(f"[{idx}] msg[{i}]: duplicate tool_call id '{tc_id}'")
                    pending_tool_calls[tc_id] = tc_name

                # Validate tool name exists in known tools
                if tool_names and tc_name and tc_name not in tool_names:
                    errors.append(f"[{idx}] msg[{i}]: unknown tool '{tc_name}'")

        prev_role = role

    # Check for unresolved tool calls
    if pending_tool_calls:
        for call_id, name in pending_tool_calls.items():
            errors.append(f"[{idx}] unresolved tool_call '{call_id}' ({name})")

    # Check for tool name leaks in assistant text content
    if tool_names:
        _check_tool_name_leaks(ex, idx, tool_names, errors)

    return errors


def _check_tool_name_leaks(
    ex: Example,
    idx: int,
    tool_names: list[str],
    errors: list[str],
) -> None:
    """Warn if assistant text content contains tool function names.

    This is a warning, not a hard error — but models that learn tool names
    in natural text tend to hallucinate tool calls.
    """
    for i, msg in enumerate(ex.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content or not isinstance(content, str):
            continue
        # Only check text content, not tool_calls
        if msg.get("tool_calls"):
            continue

        content_lower = content.lower()
        for tn in tool_names:
            # Match tool names as whole words (not substrings of other words)
            pattern = r'\b' + re.escape(tn) + r'\b'
            if re.search(pattern, content_lower):
                errors.append(
                    f"[{idx}] msg[{i}] WARNING: tool name '{tn}' found in assistant text"
                )
