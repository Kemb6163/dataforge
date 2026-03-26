"""Tests for response style variation."""

import random
from dataforge.core.styles import (
    pick_style, pick_structure, build_response, format_tool_results,
    RESPONSE_STYLES, STRUCTURE_WEIGHTS, get_style_names,
)


def test_pick_style_returns_dict():
    rng = random.Random(42)
    style = pick_style(rng)
    assert isinstance(style, dict)
    assert "greeting" in style
    assert "closing" in style


def test_pick_style_with_custom():
    rng = random.Random(42)
    custom = {"custom": {"greeting": "Hi!", "closing": "Bye!", "no_result": "Nope."}}
    # With enough tries, should eventually pick the custom style
    styles_seen = set()
    for _ in range(100):
        s = pick_style(random.Random(random.randint(0, 9999)), custom)
        if s.get("greeting") == "Hi!":
            styles_seen.add("custom")
    # Custom style should be pickable
    assert "custom" in styles_seen


def test_pick_structure_returns_valid():
    rng = random.Random(42)
    valid = {s[0] for s in STRUCTURE_WEIGHTS}
    for _ in range(50):
        structure = pick_structure(random.Random(random.randint(0, 9999)))
        assert structure in valid


def test_build_response_full():
    style = RESPONSE_STYLES["professional"]
    rng = random.Random(42)
    resp = build_response("Here is info.", style, "full", rng)
    assert "Here is info." in resp
    # Full structure should have greeting
    assert style["greeting"] in resp


def test_build_response_direct():
    style = RESPONSE_STYLES["professional"]
    rng = random.Random(42)
    resp = build_response("Direct body.", style, "direct", rng)
    assert resp == "Direct body."


def test_build_response_no_result():
    style = RESPONSE_STYLES["friendly"]
    rng = random.Random(42)
    resp = build_response("", style, "full", rng, no_result=True)
    assert resp == style["no_result"]


def test_build_response_monophrase():
    style = RESPONSE_STYLES["concise"]
    rng = random.Random(42)
    resp = build_response("First sentence. Second sentence.", style, "monophrase", rng)
    assert resp == "First sentence."


def test_format_tool_results_empty():
    rng = random.Random(42)
    resp = format_tool_results([], rng)
    assert isinstance(resp, str)
    assert len(resp) > 0  # Should return no-result text


def test_get_style_names():
    names = get_style_names()
    assert "professional" in names
    assert "friendly" in names
    assert "technical" in names
    assert "concise" in names
