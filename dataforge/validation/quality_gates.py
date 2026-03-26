"""Configurable quality gates for dataset validation.

All thresholds come from config — nothing is hardcoded. Sensible defaults
are provided for projects that don't specify them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataforge.core.types import DatasetStats


@dataclass
class QualityGateConfig:
    """Quality gate thresholds. All are optional with sensible defaults."""

    min_total: int = 500
    min_multi_turn: int = 30
    min_no_tool: int = 50
    min_parallel: int = 20
    max_closure_ratio: float = 0.65
    require_all_tools: bool = True
    min_error_handling: int = 10


@dataclass
class GateResult:
    """Result of a single quality gate check."""

    gate: str
    passed: bool
    message: str
    actual: Any = None
    threshold: Any = None


def parse_gate_config(raw: dict[str, Any] | None) -> QualityGateConfig:
    """Parse quality gate config from a dict (e.g. from YAML)."""
    if not raw:
        return QualityGateConfig()
    return QualityGateConfig(
        min_total=raw.get("min_total", 500),
        min_multi_turn=raw.get("min_multi_turn", 30),
        min_no_tool=raw.get("min_no_tool", 50),
        min_parallel=raw.get("min_parallel", 20),
        max_closure_ratio=raw.get("max_closure_ratio", 0.65),
        require_all_tools=raw.get("require_all_tools", True),
        min_error_handling=raw.get("min_error_handling", 10),
    )


def run_quality_gates(
    stats: DatasetStats,
    config: QualityGateConfig | None = None,
    tool_names: list[str] | None = None,
) -> list[GateResult]:
    """Run all quality gates against dataset statistics.

    Args:
        stats: Aggregated dataset statistics.
        config: Gate thresholds. Uses defaults if None.
        tool_names: Expected tool names (for require_all_tools check).

    Returns:
        List of GateResult objects.
    """
    if config is None:
        config = QualityGateConfig()

    results: list[GateResult] = []

    # Gate 1: Minimum total examples
    results.append(GateResult(
        gate="min_total",
        passed=stats.total >= config.min_total,
        message=f"Total examples: {stats.total} (minimum: {config.min_total})",
        actual=stats.total,
        threshold=config.min_total,
    ))

    # Gate 2: Minimum multi-turn conversations
    results.append(GateResult(
        gate="min_multi_turn",
        passed=stats.multi_turn >= config.min_multi_turn,
        message=f"Multi-turn examples: {stats.multi_turn} (minimum: {config.min_multi_turn})",
        actual=stats.multi_turn,
        threshold=config.min_multi_turn,
    ))

    # Gate 3: Minimum no-tool examples (restraint training)
    results.append(GateResult(
        gate="min_no_tool",
        passed=stats.no_tool_calls >= config.min_no_tool,
        message=f"No-tool examples: {stats.no_tool_calls} (minimum: {config.min_no_tool})",
        actual=stats.no_tool_calls,
        threshold=config.min_no_tool,
    ))

    # Gate 4: Minimum parallel tool call examples
    results.append(GateResult(
        gate="min_parallel",
        passed=stats.parallel_tool_calls >= config.min_parallel,
        message=f"Parallel tool call examples: {stats.parallel_tool_calls} (minimum: {config.min_parallel})",
        actual=stats.parallel_tool_calls,
        threshold=config.min_parallel,
    ))

    # Gate 5: Response structure diversity (no single structure > max_closure_ratio)
    if stats.total > 0 and stats.response_structures:
        max_ratio = max(stats.response_structures.values()) / stats.total
        results.append(GateResult(
            gate="max_closure_ratio",
            passed=max_ratio <= config.max_closure_ratio,
            message=f"Max structure ratio: {max_ratio:.2%} (maximum: {config.max_closure_ratio:.0%})",
            actual=max_ratio,
            threshold=config.max_closure_ratio,
        ))
    else:
        results.append(GateResult(
            gate="max_closure_ratio",
            passed=True,
            message="No response structures to check",
        ))

    # Gate 6: All tools represented
    if config.require_all_tools and tool_names:
        missing_tools = [t for t in tool_names if t not in stats.by_tool]
        results.append(GateResult(
            gate="require_all_tools",
            passed=len(missing_tools) == 0,
            message=(
                f"All tools represented: {len(tool_names) - len(missing_tools)}/{len(tool_names)}"
                + (f" — missing: {', '.join(missing_tools)}" if missing_tools else "")
            ),
            actual=len(tool_names) - len(missing_tools),
            threshold=len(tool_names),
        ))

    # Gate 7: Minimum error handling examples
    results.append(GateResult(
        gate="min_error_handling",
        passed=stats.error_handling >= config.min_error_handling,
        message=f"Error handling examples: {stats.error_handling} (minimum: {config.min_error_handling})",
        actual=stats.error_handling,
        threshold=config.min_error_handling,
    ))

    return results
