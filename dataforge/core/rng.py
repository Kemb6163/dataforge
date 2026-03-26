"""Deterministic RNG — PYTHONHASHSEED-safe.

Uses SHA-256 (not Python hash()) so the same seed produces identical output
across processes, machines, and Python versions.
"""

import hashlib
import random


def make_rng(category: str, idx: int, seed: int = 42) -> random.Random:
    """Create a deterministic RNG with zero cross-category correlation.

    Same idx in different categories produces completely different sequences.
    Same category+idx+seed always produces the same sequence regardless of
    PYTHONHASHSEED, OS, or Python version.

    Args:
        category: Generator category (e.g. "menu_search").
        idx: Example index within the generator.
        seed: Global seed for the entire dataset run.

    Returns:
        A seeded random.Random instance.
    """
    key = f"{category}:{idx}:{seed}"
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return random.Random(h)
