"""Anti-template explosion detection.

Detects when generated datasets have too many similar examples, which
causes models to learn templates instead of generalizing.

Uses fixed-size data structures (Bloom filters, TopK counters, histograms)
for constant RAM regardless of dataset size.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from dataforge.core.types import Example


class TopKCounter:
    """Space-saving approximate top-K frequency counter.

    Keeps at most max_entries items. When full, prunes the bottom quartile.
    RAM usage is bounded at ~max_entries * (avg_key_size + 8 bytes).
    """

    def __init__(self, max_entries: int = 50_000):
        self.max_entries = max_entries
        self._counts: dict[str, int] = {}
        self._prune_threshold = int(max_entries * 0.75)

    def add(self, key: str, count: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + count
        if len(self._counts) > self.max_entries:
            self._prune()

    def _prune(self) -> None:
        """Remove bottom quartile by count."""
        if not self._counts:
            return
        sorted_items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        self._counts = dict(sorted_items[: self._prune_threshold])

    def top(self, k: int = 20) -> list[tuple[str, int]]:
        """Return top-k items by count."""
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:k]

    @property
    def total_tracked(self) -> int:
        return len(self._counts)


class BloomFilter:
    """Simple Bloom filter with fixed memory (1 MB default).

    False positive rate: ~0.1% at 1M items, ~1% at 10M.
    """

    def __init__(self, size_bytes: int = 1_048_576, num_hashes: int = 3):
        self._bits = bytearray(size_bytes)
        self._size = size_bytes * 8
        self._num_hashes = num_hashes
        self._count = 0

    def _hashes(self, key: str) -> list[int]:
        digest = hashlib.sha256(key.encode()).digest()
        positions = []
        for i in range(self._num_hashes):
            h = int.from_bytes(digest[i * 4 : (i + 1) * 4], "big")
            positions.append(h % self._size)
        return positions

    def add(self, key: str) -> bool:
        """Add key. Returns True if key was probably already present."""
        positions = self._hashes(key)
        was_present = all(self._bits[p // 8] & (1 << (p % 8)) for p in positions)
        for p in positions:
            self._bits[p // 8] |= 1 << (p % 8)
        if not was_present:
            self._count += 1
        return was_present

    @property
    def count(self) -> int:
        return self._count


class TemplateChecker:
    """Incremental template explosion detector.

    Fixed RAM (~8 MB total) regardless of dataset size:
    - structural_bloom: 1 MB Bloom filter for structural dedup
    - flow_bloom: 1 MB Bloom filter for conversation flow patterns
    - trigram_counter: TopK counter (~5 MB max)
    - length_histogram: 20 buckets, O(1) RAM
    """

    # Length histogram bucket boundaries (in characters)
    _LENGTH_BUCKETS = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000,
                       1200, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]

    def __init__(self):
        self.structural_bloom = BloomFilter()
        self.flow_bloom = BloomFilter()
        self.trigram_counter = TopKCounter(max_entries=50_000)
        self.length_histogram: list[int] = [0] * (len(self._LENGTH_BUCKETS) + 1)
        self._structural_dup_count = 0
        self._flow_dup_count = 0
        self._total = 0

    def ingest(self, ex: Example) -> None:
        """Process one example incrementally."""
        self._total += 1

        # 1. Structural dedup: hash of normalized last assistant response
        last_assistant = self._extract_last_assistant(ex)
        if last_assistant:
            normalized = self._normalize(last_assistant)
            struct_key = hashlib.md5(normalized.encode()).hexdigest()
            if self.structural_bloom.add(struct_key):
                self._structural_dup_count += 1

            # Length histogram
            bucket = self._length_bucket(len(last_assistant))
            self.length_histogram[bucket] += 1

            # Trigram extraction
            for trigram in self._extract_trigrams(normalized):
                self.trigram_counter.add(trigram)

        # 2. Conversation flow pattern
        flow = self._extract_flow(ex)
        if flow:
            if self.flow_bloom.add(flow):
                self._flow_dup_count += 1

    def finalize(self) -> list[str]:
        """Return warnings after all examples ingested."""
        warnings: list[str] = []

        if self._total == 0:
            return warnings

        # Structural duplicates
        dup_rate = self._structural_dup_count / self._total
        if dup_rate > 0.05:
            warnings.append(
                f"High structural duplication: {self._structural_dup_count}/{self._total} "
                f"({dup_rate:.1%}) — responses may be too similar"
            )

        # Flow pattern duplicates
        flow_dup_rate = self._flow_dup_count / self._total
        if flow_dup_rate > 0.30:
            warnings.append(
                f"High flow pattern duplication: {self._flow_dup_count}/{self._total} "
                f"({flow_dup_rate:.1%}) — conversation structures are too repetitive"
            )

        # Length clustering: any single bucket with >40% of examples
        for i, count in enumerate(self.length_histogram):
            ratio = count / self._total
            if ratio > 0.40:
                lo = self._LENGTH_BUCKETS[i - 1] if i > 0 else 0
                hi = self._LENGTH_BUCKETS[i] if i < len(self._LENGTH_BUCKETS) else "+"
                warnings.append(
                    f"Length clustering: {ratio:.1%} of responses in [{lo}-{hi}] chars"
                )

        # Trigram overuse: any trigram appearing in >15% of examples
        for trigram, count in self.trigram_counter.top(20):
            ratio = count / self._total
            if ratio > 0.15:
                warnings.append(
                    f"Overused trigram: \"{trigram}\" appears in {ratio:.1%} of examples"
                )

        return warnings

    @property
    def structural_dup_count(self) -> int:
        return self._structural_dup_count

    @property
    def flow_dup_count(self) -> int:
        return self._flow_dup_count

    def _extract_last_assistant(self, ex: Example) -> str | None:
        for msg in reversed(ex.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content and isinstance(content, str):
                    return content
        return None

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _extract_trigrams(self, text: str) -> list[str]:
        words = text.split()
        if len(words) < 3:
            return []
        return [" ".join(words[i:i+3]) for i in range(min(len(words) - 2, 30))]

    def _extract_flow(self, ex: Example) -> str:
        """Extract conversation flow pattern hash."""
        parts = []
        for msg in ex.messages:
            role = msg.get("role", "?")
            if role == "system":
                continue
            if role == "assistant" and msg.get("tool_calls"):
                tool_names = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                parts.append(f"TOOL({','.join(sorted(tool_names))})")
            elif role == "tool":
                parts.append("RESULT")
            else:
                parts.append(role.upper())
        flow_str = "->".join(parts)
        return hashlib.md5(flow_str.encode()).hexdigest()

    def _length_bucket(self, length: int) -> int:
        for i, boundary in enumerate(self._LENGTH_BUCKETS):
            if length < boundary:
                return i
        return len(self._LENGTH_BUCKETS)
