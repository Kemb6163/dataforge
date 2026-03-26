"""Data pool generation helpers.

Utilities for generating fake data pools (names, dates, IDs, etc.) used
by example generators to create realistic training conversations.
"""

from __future__ import annotations

import random
import string
from typing import Any


def pick(rng: random.Random, items: list[Any]) -> Any:
    """Pick a random item from a list."""
    return rng.choice(items)


def pick_n(rng: random.Random, items: list[Any], n: int, unique: bool = True) -> list[Any]:
    """Pick n items from a list.

    Args:
        rng: Seeded RNG.
        items: Source list.
        n: Number of items to pick.
        unique: If True, no duplicates (sample). If False, allow repeats (choices).
    """
    if unique:
        return rng.sample(items, min(n, len(items)))
    return [rng.choice(items) for _ in range(n)]


def fake_id(rng: random.Random, prefix: str = "", length: int = 8) -> str:
    """Generate a fake alphanumeric ID."""
    chars = string.ascii_lowercase + string.digits
    body = "".join(rng.choice(chars) for _ in range(length))
    return f"{prefix}{body}" if prefix else body


def fake_name(rng: random.Random, first_names: list[str] | None = None, last_names: list[str] | None = None) -> str:
    """Generate a fake person name."""
    firsts = first_names or [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Karen", "Leo", "Maria", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    ]
    lasts = last_names or [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
        "Moore", "Jackson", "Martin", "Lee", "Clark", "Lewis", "Walker",
    ]
    return f"{rng.choice(firsts)} {rng.choice(lasts)}"


def fake_email(rng: random.Random, name: str | None = None) -> str:
    """Generate a fake email address."""
    if name is None:
        name = fake_name(rng)
    local = name.lower().replace(" ", ".") + str(rng.randint(1, 999))
    domains = ["example.com", "testmail.org", "demo.net", "sample.io"]
    return f"{local}@{rng.choice(domains)}"


def fake_date(rng: random.Random, year: int = 2025, month_range: tuple[int, int] = (1, 12)) -> str:
    """Generate a fake date string (YYYY-MM-DD)."""
    month = rng.randint(*month_range)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def fake_time(rng: random.Random, hour_range: tuple[int, int] = (8, 22)) -> str:
    """Generate a fake time string (HH:MM)."""
    hour = rng.randint(*hour_range)
    minute = rng.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"


def fake_phone(rng: random.Random) -> str:
    """Generate a fake phone number."""
    area = rng.randint(200, 999)
    mid = rng.randint(200, 999)
    last = rng.randint(1000, 9999)
    return f"+1-{area}-{mid}-{last}"


def fake_price(rng: random.Random, min_val: float = 5.0, max_val: float = 100.0) -> float:
    """Generate a fake price with 2 decimal places."""
    return round(rng.uniform(min_val, max_val), 2)


def weighted_choice(rng: random.Random, options: list[tuple[Any, float]]) -> Any:
    """Weighted random selection.

    Args:
        rng: Seeded RNG.
        options: List of (value, weight) tuples.
    """
    values = [o[0] for o in options]
    weights = [o[1] for o in options]
    return rng.choices(values, weights=weights, k=1)[0]
