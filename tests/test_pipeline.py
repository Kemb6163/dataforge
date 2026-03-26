"""Tests for the streaming pipeline."""

import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

from dataforge.core.types import Example, DPOPair
from dataforge.core.messages import user_msg, assistant_msg, tool_call_msg, tool_result_msg
from dataforge.generation.base import SFTGenerator, DPOGenerator
from dataforge.generation.pipeline import StreamingWriter


class _MockSFTGenerator(SFTGenerator):
    @property
    def category(self) -> str:
        return "mock_sft"

    @property
    def name(self) -> str:
        return "Mock SFT"

    def expected_count(self) -> int:
        return 10

    def generate(self) -> Iterator[Example]:
        for i in range(10):
            yield Example(messages=[
                user_msg(f"Question {i}"),
                assistant_msg(f"Answer {i}"),
            ])


class _MockDPOGenerator(DPOGenerator):
    @property
    def category(self) -> str:
        return "mock_dpo"

    @property
    def name(self) -> str:
        return "Mock DPO"

    def expected_count(self) -> int:
        return 5

    def generate(self) -> Iterator[DPOPair]:
        for i in range(5):
            yield DPOPair(
                prompt=[user_msg(f"Q{i}")],
                chosen=[assistant_msg(f"Good {i}")],
                rejected=[assistant_msg(f"Bad {i}")],
            )


def test_streaming_writer_split():
    """StreamingWriter splits examples into train/val files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = StreamingWriter(tmpdir, "test", train_split=0.80, seed=42)

        for i in range(100):
            ex = Example(messages=[user_msg(f"Q{i}"), assistant_msg(f"A{i}")])
            writer.write(ex)

        summary = writer.close()
        assert summary["total_written"] == 100
        assert summary["train"] + summary["val"] == 100
        # With 80% split, train should be roughly 80 (+/- some variance)
        assert 60 < summary["train"] < 95

        # Files should exist and be valid JSONL
        train_path = Path(tmpdir) / "test-train.jsonl"
        val_path = Path(tmpdir) / "test-val.jsonl"
        assert train_path.exists()
        assert val_path.exists()

        with open(train_path) as f:
            train_lines = [json.loads(l) for l in f if l.strip()]
        assert len(train_lines) == summary["train"]


def test_streaming_writer_deterministic_split():
    """Same content always goes to same split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer1 = StreamingWriter(tmpdir, "run1", train_split=0.90, seed=42)
        writer2 = StreamingWriter(tmpdir, "run2", train_split=0.90, seed=42)

        for i in range(50):
            ex = Example(messages=[user_msg(f"Q{i}"), assistant_msg(f"A{i}")])
            writer1.write(ex)
            writer2.write(ex)

        s1 = writer1.close()
        s2 = writer2.close()
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]


def test_streaming_writer_dpo():
    """DPO pairs are written correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = StreamingWriter(tmpdir, "dpo-test", train_split=1.0)

        for i in range(10):
            pair = DPOPair(
                prompt=[user_msg(f"Q{i}")],
                chosen=[assistant_msg(f"Good {i}")],
                rejected=[assistant_msg(f"Bad {i}")],
            )
            writer.write_dpo(pair)

        summary = writer.close()
        assert summary["total_written"] == 10

        train_path = Path(tmpdir) / "dpo-test-train.jsonl"
        with open(train_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 10
        assert "prompt" in lines[0]
        assert "chosen" in lines[0]
        assert "rejected" in lines[0]


def test_streaming_writer_sharegpt_format():
    """ShareGPT export format conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = StreamingWriter(tmpdir, "sg-test", export_format="sharegpt")
        ex = Example(messages=[user_msg("Hi"), assistant_msg("Hello")])
        writer.write(ex)
        summary = writer.close()

        train_path = Path(tmpdir) / "sg-test-train.jsonl"
        with open(train_path) as f:
            data = json.loads(f.readline())
        assert "conversations" in data
        assert data["conversations"][0]["from"] == "human"
        assert data["conversations"][1]["from"] == "gpt"
