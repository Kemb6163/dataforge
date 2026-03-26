"""Tests for message builders."""

import random
from dataforge.core.messages import (
    system_msg, user_msg, assistant_msg, tool_call_msg, multi_tool_call_msg,
    tool_result_msg, make_call_id, reset_call_counter, example,
)


def test_system_msg():
    msg = system_msg("You are helpful.")
    assert msg["role"] == "system"
    assert msg["content"] == "You are helpful."


def test_user_msg():
    msg = user_msg("Hello")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello"


def test_assistant_msg():
    msg = assistant_msg("Hi there!")
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi there!"


def test_tool_call_msg():
    msg = tool_call_msg("search", {"query": "test"}, call_id="call_001")
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_001"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == {"query": "test"}


def test_multi_tool_call_msg():
    msg = multi_tool_call_msg([
        ("tool_a", {"x": 1}),
        ("tool_b", {"y": 2}),
    ])
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 2
    assert msg["tool_calls"][0]["function"]["name"] == "tool_a"
    assert msg["tool_calls"][1]["function"]["name"] == "tool_b"


def test_tool_result_msg():
    msg = tool_result_msg("call_001", '{"result": 42}')
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_001"
    assert msg["content"] == '{"result": 42}'


def test_make_call_id_uniqueness():
    reset_call_counter()
    ids = {make_call_id(prefix="test", rng=random.Random(42)) for _ in range(100)}
    assert len(ids) == 100  # All unique


def test_make_call_id_with_prefix():
    reset_call_counter()
    rng = random.Random(42)
    cid = make_call_id(prefix="menu", rng=rng)
    assert cid.startswith("call_menu_")


def test_reset_call_counter():
    reset_call_counter()
    rng = random.Random(42)
    id1 = make_call_id(rng=rng)
    reset_call_counter()
    rng2 = random.Random(42)
    id2 = make_call_id(rng=rng2)
    # After reset, counter starts over, so first 4 digits match
    assert id1.split("_")[1] == id2.split("_")[1]  # same counter value


def test_example_prepends_system():
    msgs = [user_msg("Hi"), assistant_msg("Hello")]
    result = example(msgs, "You are a bot.")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a bot."
    assert len(result) == 3


def test_example_no_duplicate_system():
    msgs = [system_msg("Existing"), user_msg("Hi")]
    result = example(msgs, "New system")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "Existing"
    assert len(result) == 2


def test_example_no_system():
    msgs = [user_msg("Hi")]
    result = example(msgs, None)
    assert len(result) == 1
    assert result[0]["role"] == "user"
