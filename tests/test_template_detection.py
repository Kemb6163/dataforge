"""Tests for anti-template explosion detection."""

from dataforge.core.types import Example
from dataforge.core.messages import user_msg, assistant_msg, tool_call_msg, tool_result_msg
from dataforge.validation.template_detection import TemplateChecker, BloomFilter, TopKCounter


def test_bloom_filter_basic():
    bf = BloomFilter(size_bytes=1024)
    assert not bf.add("hello")
    assert bf.add("hello")  # Second add returns True (probably present)
    assert bf.count == 1


def test_bloom_filter_different_keys():
    bf = BloomFilter(size_bytes=1024)
    bf.add("key1")
    bf.add("key2")
    bf.add("key3")
    assert bf.count == 3


def test_topk_counter():
    counter = TopKCounter(max_entries=10)
    for i in range(5):
        counter.add("frequent", count=10)
        counter.add("rare", count=1)
    top = counter.top(5)
    assert top[0][0] == "frequent"
    assert top[0][1] == 50
    assert top[1][0] == "rare"
    assert top[1][1] == 5


def test_topk_counter_prune():
    counter = TopKCounter(max_entries=5)
    for i in range(20):
        counter.add(f"key_{i}")
    # After pruning, should be at most max_entries
    assert counter.total_tracked <= 5


def test_template_checker_no_issues():
    """Diverse examples should produce no warnings."""
    checker = TemplateChecker()
    for i in range(50):
        ex = Example(messages=[
            user_msg(f"Question number {i} about topic {i * 7 % 13}"),
            assistant_msg(f"Answer {i}: " + "x" * (50 + i * 3)),
        ])
        checker.ingest(ex)

    warnings = checker.finalize()
    # Diverse content should not trigger warnings
    assert not any("Structural" in w.lower() for w in warnings)


def test_template_checker_detects_duplicates():
    """Identical responses should trigger structural dup warning."""
    checker = TemplateChecker()
    for i in range(100):
        ex = Example(messages=[
            user_msg(f"Question {i}"),
            assistant_msg("This is always the same response."),
        ])
        checker.ingest(ex)

    warnings = checker.finalize()
    assert any("structural" in w.lower() or "duplication" in w.lower() for w in warnings)


def test_template_checker_flow_patterns():
    """Tracks conversation flow patterns."""
    checker = TemplateChecker()
    # All examples have identical flow: USER -> TOOL_CALL -> RESULT -> ASSISTANT
    for i in range(20):
        tc = tool_call_msg("search", {"q": f"query_{i}"}, call_id=f"call_{i}")
        ex = Example(messages=[
            user_msg(f"Search {i}"),
            tc,
            tool_result_msg(f"call_{i}", f"result_{i}"),
            assistant_msg(f"Found result {i}" + " extra" * (i % 5)),
        ])
        checker.ingest(ex)

    # Same flow pattern for all — should be detected
    assert checker.flow_dup_count > 0


def test_template_checker_fixed_ram():
    """Bloom filters have fixed size regardless of input."""
    checker = TemplateChecker()
    initial_bloom_size = len(checker.structural_bloom._bits)

    for i in range(1000):
        ex = Example(messages=[
            user_msg(f"Q{i}"),
            assistant_msg(f"A{i}" + "y" * (i % 200)),
        ])
        checker.ingest(ex)

    # Bloom filter size unchanged (fixed allocation)
    assert len(checker.structural_bloom._bits) == initial_bloom_size
