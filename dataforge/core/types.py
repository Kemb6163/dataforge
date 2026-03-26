"""Type definitions for DataForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """A single SFT training example (conversation)."""

    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages}


@dataclass
class DPOPair:
    """A DPO preference pair: same prompt, chosen vs rejected continuation."""

    prompt: list[dict[str, Any]]
    chosen: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


@dataclass
class ContrastiveSet:
    """A set of ranked responses to the same prompt for DPO conversion."""

    prompt: list[dict[str, Any]]
    responses: list[dict[str, Any]]  # each has "text" and "rank" (lower = better)

    def to_dpo_pairs(self) -> list[DPOPair]:
        """Convert ranked responses to all valid DPO pairs."""
        pairs: list[DPOPair] = []
        sorted_resp = sorted(self.responses, key=lambda r: r["rank"])
        for i, chosen in enumerate(sorted_resp):
            for rejected in sorted_resp[i + 1 :]:
                pairs.append(
                    DPOPair(
                        prompt=self.prompt,
                        chosen=[{"role": "assistant", "content": chosen["text"]}],
                        rejected=[{"role": "assistant", "content": rejected["text"]}],
                    )
                )
        return pairs


@dataclass
class DatasetStats:
    """Aggregated dataset statistics."""

    total: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)
    by_role: dict[str, int] = field(default_factory=dict)
    multi_turn: int = 0
    no_tool_calls: int = 0
    parallel_tool_calls: int = 0
    response_structures: dict[str, int] = field(default_factory=dict)
    error_handling: int = 0
    avg_messages_per_example: float = 0.0
    total_messages: int = 0
