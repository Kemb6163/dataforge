"""Message builders for OpenAI/Hermes tool-calling format.

All functions produce dicts compatible with the OpenAI chat-completions schema
used by most fine-tuning frameworks (trl, axolotl, LLaMA-Factory, etc.).
"""

from __future__ import annotations

import uuid
from typing import Any

_call_counter: int = 0


def reset_call_counter() -> None:
    """Reset the global call counter. Call between generators."""
    global _call_counter
    _call_counter = 0


def make_call_id(prefix: str = "", rng: Any | None = None) -> str:
    """Generate a unique tool call ID.

    Format: call_{prefix}_{NNNN}_{hex4}
    - prefix prevents cross-generator collisions
    - hex suffix (from rng) prevents collisions when merging datasets

    Args:
        prefix: Generator category name (e.g. "menu_search").
        rng: A random.Random instance for deterministic hex suffix.
    """
    global _call_counter
    _call_counter += 1
    counter_str = f"{_call_counter:04d}"
    if rng is not None:
        hex_suffix = f"{rng.randint(0, 0xFFFF):04x}"
    else:
        hex_suffix = uuid.uuid4().hex[:4]
    if prefix:
        return f"call_{prefix}_{counter_str}_{hex_suffix}"
    return f"call_{counter_str}_{hex_suffix}"


def system_msg(content: str) -> dict[str, Any]:
    """Create a system message."""
    return {"role": "system", "content": content}


def user_msg(content: str) -> dict[str, Any]:
    """Create a user message."""
    return {"role": "user", "content": content}


def assistant_msg(content: str) -> dict[str, Any]:
    """Create an assistant text message (no tool calls)."""
    return {"role": "assistant", "content": content}


def tool_call_msg(
    name: str,
    arguments: dict[str, Any],
    call_id: str | None = None,
    prefix: str = "",
    rng: Any | None = None,
) -> dict[str, Any]:
    """Create an assistant message with a single tool call.

    Args:
        name: Tool function name.
        arguments: Tool arguments dict.
        call_id: Explicit call ID. If None, auto-generated.
        prefix: Category prefix for auto-generated IDs.
        rng: RNG for deterministic ID generation.
    """
    if call_id is None:
        call_id = make_call_id(prefix=prefix, rng=rng)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def multi_tool_call_msg(
    calls: list[tuple[str, dict[str, Any]]],
    prefix: str = "",
    rng: Any | None = None,
) -> dict[str, Any]:
    """Create an assistant message with parallel tool calls.

    Args:
        calls: List of (tool_name, arguments) tuples.
        prefix: Category prefix for auto-generated IDs.
        rng: RNG for deterministic ID generation.
    """
    tool_calls = []
    for name, arguments in calls:
        call_id = make_call_id(prefix=prefix, rng=rng)
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }


def tool_result_msg(call_id: str, result: Any) -> dict[str, Any]:
    """Create a tool response message.

    Args:
        call_id: The tool_call ID this result corresponds to.
        result: The tool's return value (will be stored as-is).
    """
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result if isinstance(result, str) else str(result),
    }


def example(messages: list[dict[str, Any]], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """Wrap a message list with an optional system prompt.

    If system_prompt is provided and messages[0] is not already a system msg,
    prepends it.
    """
    if system_prompt and (not messages or messages[0].get("role") != "system"):
        return [system_msg(system_prompt)] + messages
    return messages
