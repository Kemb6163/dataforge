"""Tests for structural validation."""

from dataforge.core.types import Example
from dataforge.core.messages import (
    user_msg, assistant_msg, tool_call_msg, tool_result_msg, system_msg,
)
from dataforge.validation.structural import validate_example


def test_valid_simple_conversation():
    ex = Example(messages=[
        user_msg("Hello"),
        assistant_msg("Hi there!"),
    ])
    errors = validate_example(ex, 0)
    assert errors == []


def test_valid_tool_call():
    tc = tool_call_msg("search", {"q": "test"}, call_id="call_001")
    ex = Example(messages=[
        user_msg("Search for test"),
        tc,
        tool_result_msg("call_001", '{"results": []}'),
        assistant_msg("No results found."),
    ])
    errors = validate_example(ex, 0, tool_names=["search"])
    assert errors == []


def test_empty_messages():
    ex = Example(messages=[])
    errors = validate_example(ex, 0)
    assert any("Empty" in e for e in errors)


def test_invalid_role():
    ex = Example(messages=[{"role": "wizard", "content": "Magic!"}])
    errors = validate_example(ex, 0)
    assert any("invalid role" in e for e in errors)


def test_system_not_at_position_zero():
    ex = Example(messages=[
        user_msg("Hi"),
        system_msg("I am system"),
    ])
    errors = validate_example(ex, 0)
    assert any("position 0" in e for e in errors)


def test_unresolved_tool_call():
    tc = tool_call_msg("search", {"q": "test"}, call_id="call_001")
    ex = Example(messages=[
        user_msg("Search"),
        tc,
        assistant_msg("Here are results."),
    ])
    errors = validate_example(ex, 0)
    assert any("unresolved" in e for e in errors)


def test_tool_result_without_call():
    ex = Example(messages=[
        user_msg("Hi"),
        tool_result_msg("call_999", "result"),
    ])
    errors = validate_example(ex, 0)
    assert any("no matching" in e for e in errors)


def test_unknown_tool_name():
    tc = tool_call_msg("unknown_tool", {}, call_id="call_001")
    ex = Example(messages=[
        user_msg("Do something"),
        tc,
        tool_result_msg("call_001", "done"),
    ])
    errors = validate_example(ex, 0, tool_names=["search", "get_info"])
    assert any("unknown tool" in e for e in errors)


def test_tool_name_leak_warning():
    ex = Example(messages=[
        user_msg("Search for food"),
        assistant_msg("I'll use search_menu to find that for you."),
    ])
    errors = validate_example(ex, 0, tool_names=["search_menu"])
    assert any("WARNING" in e and "search_menu" in e for e in errors)


def test_no_tool_name_leak_in_tool_calls():
    """Tool calls themselves should not trigger tool name leak warnings."""
    tc = tool_call_msg("search_menu", {"q": "pasta"}, call_id="call_001")
    ex = Example(messages=[
        user_msg("Find pasta"),
        tc,
        tool_result_msg("call_001", "found"),
        assistant_msg("Here are the pasta options."),
    ])
    errors = validate_example(ex, 0, tool_names=["search_menu"])
    # No warnings expected — tool name is in tool_call, not in plain text
    assert not any("WARNING" in e for e in errors)
