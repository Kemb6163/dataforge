"""Response style variation for natural-sounding training data.

Provides multiple response styles and structural variations to prevent
the trained model from learning a single monotonous response pattern.
"""

from __future__ import annotations

import random
from typing import Any


RESPONSE_STYLES: dict[str, dict[str, str]] = {
    "professional": {
        "greeting": "Here are the results:",
        "no_result": "No results were found matching your criteria.",
        "closing": "Is there anything else I can help you with?",
        "error_ack": "I encountered an issue while processing your request.",
        "transition": "Based on the information retrieved,",
    },
    "friendly": {
        "greeting": "Great question! Here's what I found:",
        "no_result": "Hmm, I couldn't find anything matching that. Want to try different criteria?",
        "closing": "Let me know if you need anything else!",
        "error_ack": "Oops, something went wrong on my end.",
        "transition": "So here's the deal:",
    },
    "technical": {
        "greeting": "Results:",
        "no_result": "Query returned zero results for the given parameters.",
        "closing": "Awaiting further instructions.",
        "error_ack": "The operation returned an error.",
        "transition": "The retrieved data indicates:",
    },
    "concise": {
        "greeting": "Found it:",
        "no_result": "Nothing found.",
        "closing": "Need anything else?",
        "error_ack": "Got an error.",
        "transition": "Here's what that means:",
    },
}

# Weighted structure types: how to assemble a response
STRUCTURE_WEIGHTS: list[tuple[str, float]] = [
    ("full", 0.35),          # greeting + body + closing
    ("no_closure", 0.25),    # greeting + body (no closing)
    ("direct", 0.30),        # body only
    ("monophrase", 0.10),    # single compact sentence
]

_structure_names = [s[0] for s in STRUCTURE_WEIGHTS]
_structure_cum_weights = []
_cum = 0.0
for _, w in STRUCTURE_WEIGHTS:
    _cum += w
    _structure_cum_weights.append(_cum)


def get_style_names() -> list[str]:
    """Return available style names."""
    return list(RESPONSE_STYLES.keys())


def pick_style(rng: random.Random, custom_styles: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """Pick a random response style.

    Args:
        rng: Seeded RNG.
        custom_styles: Optional user-defined styles merged with defaults.
    """
    styles = dict(RESPONSE_STYLES)
    if custom_styles:
        styles.update(custom_styles)
    name = rng.choice(list(styles.keys()))
    return styles[name]


def pick_structure(rng: random.Random) -> str:
    """Pick a weighted random structure type."""
    r = rng.random()
    for name, cum_w in zip(_structure_names, _structure_cum_weights):
        if r < cum_w:
            return name
    return _structure_names[-1]


def build_response(
    body: str,
    style: dict[str, str],
    structure: str,
    rng: random.Random,
    no_result: bool = False,
) -> str:
    """Assemble a response from body text, style, and structure.

    Args:
        body: The main content of the response.
        style: Style dict with greeting/closing/etc phrases.
        structure: One of "full", "no_closure", "direct", "monophrase".
        rng: Seeded RNG for minor variation choices.
        no_result: If True, use the no_result phrase instead of greeting+body.
    """
    if no_result:
        return style.get("no_result", "No results found.")

    if structure == "monophrase":
        return body.split(".")[0].strip() + "." if "." in body else body

    if structure == "direct":
        return body

    parts = []
    if structure in ("full", "no_closure"):
        greeting = style.get("greeting", "")
        if greeting:
            parts.append(greeting)

    parts.append(body)

    if structure == "full":
        closing = style.get("closing", "")
        if closing and rng.random() > 0.15:
            parts.append(closing)

    return "\n\n".join(parts)


def format_tool_results(
    results: list[dict[str, Any]],
    rng: random.Random,
    style: dict[str, str] | None = None,
    structure: str | None = None,
    custom_styles: dict[str, dict[str, str]] | None = None,
    no_result: bool = False,
    formatter: Any = None,
) -> str:
    """High-level helper: format tool results into a styled response.

    Args:
        results: List of result dicts to present.
        rng: Seeded RNG.
        style: Explicit style dict. If None, picks randomly.
        structure: Explicit structure. If None, picks randomly.
        custom_styles: User-defined styles to merge.
        no_result: If True, return no-result phrasing.
        formatter: Optional callable(results) -> str for custom body formatting.
    """
    if style is None:
        style = pick_style(rng, custom_styles)
    if structure is None:
        structure = pick_structure(rng)

    if no_result or not results:
        return build_response("", style, structure, rng, no_result=True)

    if formatter:
        body = formatter(results)
    else:
        body = _default_format(results, rng)

    return build_response(body, style, structure, rng)


def _default_format(results: list[dict[str, Any]], rng: random.Random) -> str:
    """Default formatter: bullet list of result summaries."""
    if len(results) == 1:
        item = results[0]
        if isinstance(item, dict) and "name" in item:
            return f"I found **{item['name']}**. " + ", ".join(
                f"{k}: {v}" for k, v in item.items() if k != "name"
            )
        return str(item)

    lines = []
    for item in results[:10]:
        if isinstance(item, dict) and "name" in item:
            lines.append(f"- **{item['name']}**")
        else:
            lines.append(f"- {item}")
    if len(results) > 10:
        lines.append(f"...and {len(results) - 10} more results.")
    return "\n".join(lines)
