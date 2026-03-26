"""Integration test: generate + validate the restaurant example."""

import json
import sys
import os
import tempfile
from pathlib import Path

# Add examples to path for import resolution
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "restaurant"))

from dataforge.core.types import Example
from dataforge.core.messages import reset_call_counter
from dataforge.validation.structural import validate_example
from dataforge.validation.stats import StatsTracker
from dataforge.validation.template_detection import TemplateChecker


def _load_tools():
    tools_path = Path(__file__).parent.parent / "examples" / "restaurant" / "tools.json"
    with open(tools_path) as f:
        tools_list = json.load(f)
    tools_dict = {}
    tool_names = []
    for t in tools_list:
        name = t["function"]["name"]
        tools_dict[name] = t
        tool_names.append(name)
    return tools_dict, tool_names


def _get_config():
    return {
        "seed": 42,
        "language": "en",
        "styles": {},
        "error_injection": {"enabled": True, "base_rate": 0.10, "burst_probability": 0.13},
    }


def test_menu_search_generation():
    """MenuSearchGenerator produces valid examples."""
    from generators.menu_search import MenuSearchGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()
    gen = MenuSearchGenerator(config, tools_dict)

    reset_call_counter()
    examples = list(gen.generate())
    assert len(examples) == gen.expected_count()

    errors = []
    for i, ex in enumerate(examples):
        errs = validate_example(ex, i, tool_names)
        hard = [e for e in errs if "WARNING" not in e]
        errors.extend(hard)
    assert len(errors) == 0, f"Validation errors: {errors[:5]}"


def test_reservations_generation():
    """ReservationGenerator produces valid multi-turn examples."""
    from generators.reservations import ReservationGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()
    gen = ReservationGenerator(config, tools_dict)

    reset_call_counter()
    examples = list(gen.generate())
    assert len(examples) == gen.expected_count()

    # Check multi-turn: most should have >2 user messages
    multi_turn_count = sum(
        1 for ex in examples
        if sum(1 for m in ex.messages if m.get("role") == "user") > 1
    )
    assert multi_turn_count > 30, f"Expected >30 multi-turn, got {multi_turn_count}"


def test_order_management_parallel_calls():
    """OrderManagementGenerator produces parallel tool call examples."""
    from generators.order_management import OrderManagementGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()
    gen = OrderManagementGenerator(config, tools_dict)

    reset_call_counter()
    examples = list(gen.generate())
    assert len(examples) == gen.expected_count()

    parallel_count = sum(
        1 for ex in examples
        if any(len(m.get("tool_calls", [])) > 1 for m in ex.messages)
    )
    assert parallel_count >= 40, f"Expected >=40 parallel call examples, got {parallel_count}"


def test_reviews_no_tool_restraint():
    """ReviewGenerator produces no-tool examples for out-of-scope requests."""
    from generators.reviews import ReviewGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()
    gen = ReviewGenerator(config, tools_dict)

    reset_call_counter()
    examples = list(gen.generate())
    assert len(examples) == gen.expected_count()

    no_tool = sum(
        1 for ex in examples
        if not any(m.get("tool_calls") for m in ex.messages)
    )
    assert no_tool >= 50, f"Expected >=50 no-tool examples, got {no_tool}"


def test_full_restaurant_dataset():
    """Full restaurant dataset passes basic structural validation."""
    from generators.menu_search import MenuSearchGenerator
    from generators.reservations import ReservationGenerator
    from generators.order_management import OrderManagementGenerator
    from generators.reviews import ReviewGenerator
    from generators.complex_scenarios import ComplexScenarioGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()

    generators = [
        MenuSearchGenerator(config, tools_dict),
        ReservationGenerator(config, tools_dict),
        OrderManagementGenerator(config, tools_dict),
        ReviewGenerator(config, tools_dict),
        ComplexScenarioGenerator(config, tools_dict),
    ]

    stats = StatsTracker()
    checker = TemplateChecker()
    total_errors = 0
    total_examples = 0

    for gen in generators:
        reset_call_counter()
        for ex in gen.generate():
            errs = validate_example(ex, total_examples, tool_names)
            hard = [e for e in errs if "WARNING" not in e]
            total_errors += len(hard)
            stats.ingest(ex)
            checker.ingest(ex)
            total_examples += 1

    s = stats.stats
    assert total_errors == 0, f"Got {total_errors} validation errors"
    assert s.total >= 500, f"Expected >=500 examples, got {s.total}"
    assert s.multi_turn >= 30, f"Expected >=30 multi-turn, got {s.multi_turn}"
    assert s.no_tool_calls >= 50, f"Expected >=50 no-tool, got {s.no_tool_calls}"
    assert s.parallel_tool_calls >= 20, f"Expected >=20 parallel, got {s.parallel_tool_calls}"

    # Check all tools are represented
    for tool_name in tool_names:
        assert tool_name in s.by_tool, f"Tool '{tool_name}' not used in any example"


def test_deterministic_generation():
    """Same seed produces identical output."""
    from generators.menu_search import MenuSearchGenerator

    tools_dict, tool_names = _load_tools()
    config = _get_config()

    gen1 = MenuSearchGenerator(config, tools_dict)
    gen2 = MenuSearchGenerator(config, tools_dict)

    reset_call_counter()
    examples1 = [json.dumps(ex.to_dict(), sort_keys=True) for ex in gen1.generate()]
    reset_call_counter()
    examples2 = [json.dumps(ex.to_dict(), sort_keys=True) for ex in gen2.generate()]

    assert examples1 == examples2
