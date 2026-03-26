"""Tests for quality gates."""

from dataforge.core.types import DatasetStats
from dataforge.validation.quality_gates import (
    run_quality_gates, QualityGateConfig, parse_gate_config,
)


def _make_stats(**kwargs) -> DatasetStats:
    defaults = {
        "total": 600,
        "multi_turn": 50,
        "no_tool_calls": 60,
        "parallel_tool_calls": 30,
        "error_handling": 20,
        "by_tool": {"search": 200, "create": 150, "update": 100},
        "response_structures": {"A": 200, "B": 200, "C": 200},
    }
    defaults.update(kwargs)
    return DatasetStats(**defaults)


def test_all_gates_pass():
    stats = _make_stats()
    config = QualityGateConfig(min_total=500)
    results = run_quality_gates(stats, config, tool_names=["search", "create", "update"])
    assert all(r.passed for r in results)


def test_min_total_fails():
    stats = _make_stats(total=100)
    config = QualityGateConfig(min_total=500)
    results = run_quality_gates(stats, config)
    total_gate = next(r for r in results if r.gate == "min_total")
    assert not total_gate.passed


def test_min_multi_turn_fails():
    stats = _make_stats(multi_turn=5)
    config = QualityGateConfig(min_multi_turn=30)
    results = run_quality_gates(stats, config)
    gate = next(r for r in results if r.gate == "min_multi_turn")
    assert not gate.passed


def test_min_no_tool_fails():
    stats = _make_stats(no_tool_calls=10)
    config = QualityGateConfig(min_no_tool=50)
    results = run_quality_gates(stats, config)
    gate = next(r for r in results if r.gate == "min_no_tool")
    assert not gate.passed


def test_min_parallel_fails():
    stats = _make_stats(parallel_tool_calls=5)
    config = QualityGateConfig(min_parallel=20)
    results = run_quality_gates(stats, config)
    gate = next(r for r in results if r.gate == "min_parallel")
    assert not gate.passed


def test_closure_ratio_fails():
    stats = _make_stats(response_structures={"A": 500, "B": 100})
    config = QualityGateConfig(max_closure_ratio=0.65)
    results = run_quality_gates(stats, config)
    gate = next(r for r in results if r.gate == "max_closure_ratio")
    assert not gate.passed


def test_require_all_tools_fails():
    stats = _make_stats(by_tool={"search": 200, "create": 150})
    config = QualityGateConfig(require_all_tools=True)
    results = run_quality_gates(stats, config, tool_names=["search", "create", "missing_tool"])
    gate = next(r for r in results if r.gate == "require_all_tools")
    assert not gate.passed
    assert "missing_tool" in gate.message


def test_error_handling_fails():
    stats = _make_stats(error_handling=2)
    config = QualityGateConfig(min_error_handling=10)
    results = run_quality_gates(stats, config)
    gate = next(r for r in results if r.gate == "min_error_handling")
    assert not gate.passed


def test_parse_gate_config_defaults():
    config = parse_gate_config(None)
    assert config.min_total == 500
    assert config.require_all_tools is True


def test_parse_gate_config_override():
    config = parse_gate_config({"min_total": 100, "require_all_tools": False})
    assert config.min_total == 100
    assert config.require_all_tools is False
