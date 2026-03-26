"""Error injection framework for realistic training data.

Injects tool errors (timeouts, permission denied, not found, etc.) at
a configurable rate with non-linear burst zones so the model learns
graceful error handling.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


ERROR_TYPES = {
    "timeout": {
        "error": True,
        "message": "Request timed out after 30000ms",
        "code": "TIMEOUT",
    },
    "empty": {
        "error": True,
        "message": "Operation returned empty result set",
        "code": "EMPTY_RESULT",
    },
    "partial": {
        "error": True,
        "message": "Partial results returned — some data may be missing",
        "code": "PARTIAL_RESULT",
    },
    "permission": {
        "error": True,
        "message": "Permission denied: insufficient access level",
        "code": "PERMISSION_DENIED",
    },
    "not_found": {
        "error": True,
        "message": "Resource not found",
        "code": "NOT_FOUND",
    },
}

# Error handling response templates (assistant recovery text)
_RECOVERY_TEMPLATES = {
    "timeout": [
        "It looks like the request timed out. Let me try a simpler search.",
        "The service is taking too long to respond. Could you try a more specific query?",
        "I'm experiencing a timeout. Let me attempt an alternative approach.",
    ],
    "empty": [
        "I didn't find any results matching those criteria. Would you like to broaden your search?",
        "No results came back for that query. Let's try adjusting the parameters.",
        "That search returned nothing. Could you try different keywords?",
    ],
    "partial": [
        "I got partial results — some information might be missing. Here's what I have so far.",
        "The results are incomplete, but let me share what I was able to retrieve.",
    ],
    "permission": [
        "It seems I don't have permission to access that resource. You may need to check your access level.",
        "Access was denied for that operation. This might require elevated permissions.",
    ],
    "not_found": [
        "I couldn't find that resource. It may have been moved or deleted.",
        "That item doesn't appear to exist. Could you double-check the identifier?",
        "Nothing was found at that location. Let me know if you have an alternative reference.",
    ],
}

# Burst zone markers — around these percentages of a generator's output,
# error rate is elevated to simulate real-world error clustering
_BURST_ZONES = [0.13, 0.42, 0.71, 0.93]
_BURST_ZONE_WIDTH = 0.04
_BURST_MULTIPLIER = 3.0


def should_inject_error(
    category: str,
    idx: int,
    total: int,
    base_rate: float = 0.10,
    seed: int = 42,
) -> str | None:
    """Decide whether to inject an error for this example.

    Uses deterministic hashing (not RNG state) so error injection is
    independent of generator execution order.

    Args:
        category: Generator category.
        idx: Example index.
        total: Total expected examples for this generator.
        base_rate: Base error injection probability (0.0-1.0).
        seed: Global seed.

    Returns:
        Error type string or None.
    """
    key = f"error:{category}:{idx}:{seed}"
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    threshold = h / 0xFFFFFFFF

    rate = base_rate
    if total > 0:
        progress = idx / total
        for zone_center in _BURST_ZONES:
            if abs(progress - zone_center) < _BURST_ZONE_WIDTH:
                rate = min(base_rate * _BURST_MULTIPLIER, 0.5)
                break

    if threshold >= rate:
        return None

    # Pick error type deterministically
    type_key = f"error_type:{category}:{idx}:{seed}"
    type_h = int.from_bytes(hashlib.sha256(type_key.encode()).digest()[:4], "big")
    error_names = list(ERROR_TYPES.keys())
    return error_names[type_h % len(error_names)]


def make_error_response(error_type: str) -> dict[str, Any]:
    """Create a tool error result dict.

    Args:
        error_type: One of the ERROR_TYPES keys.
    """
    template = ERROR_TYPES.get(error_type, ERROR_TYPES["not_found"])
    return {
        "error": True,
        "code": template["code"],
        "message": template["message"],
    }


def make_error_handling_response(error_type: str, rng: random.Random) -> str:
    """Create an assistant recovery message for a tool error.

    Args:
        error_type: The error type that occurred.
        rng: Seeded RNG for selecting a template variant.
    """
    templates = _RECOVERY_TEMPLATES.get(error_type, _RECOVERY_TEMPLATES["not_found"])
    return rng.choice(templates)
