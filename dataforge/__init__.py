"""DataForge — Synthetic dataset generation toolkit for LLM fine-tuning."""

__version__ = "0.1.0"

from dataforge.core.types import Example, DPOPair, ContrastiveSet, DatasetStats
from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    tool_call_msg,
    multi_tool_call_msg,
    tool_result_msg,
    assistant_msg,
    user_msg,
    system_msg,
    make_call_id,
    reset_call_counter,
    example,
)
from dataforge.generation.base import SFTGenerator, DPOGenerator

__all__ = [
    "__version__",
    "Example",
    "DPOPair",
    "ContrastiveSet",
    "DatasetStats",
    "make_rng",
    "tool_call_msg",
    "multi_tool_call_msg",
    "tool_result_msg",
    "assistant_msg",
    "user_msg",
    "system_msg",
    "make_call_id",
    "reset_call_counter",
    "example",
    "SFTGenerator",
    "DPOGenerator",
]
