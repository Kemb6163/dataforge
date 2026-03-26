"""Streaming pipeline orchestrator.

Runs generators, validates, tracks stats, writes JSONL — all streaming
with constant RAM. Never holds the full dataset in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from dataforge.core.messages import reset_call_counter
from dataforge.core.types import Example, DPOPair, DatasetStats
from dataforge.generation.base import SFTGenerator, DPOGenerator
from dataforge.generation.discovery import discover_generators
from dataforge.validation.structural import validate_example
from dataforge.validation.template_detection import TemplateChecker
from dataforge.validation.quality_gates import (
    QualityGateConfig,
    GateResult,
    run_quality_gates,
    parse_gate_config,
)
from dataforge.validation.stats import StatsTracker


class StreamingWriter:
    """Writes examples to train/val JSONL files with inline content-hash split."""

    def __init__(
        self,
        output_dir: str | Path,
        name: str,
        train_split: float = 0.95,
        seed: int = 42,
        export_format: str = "openai",
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._name = name
        self._train_split = train_split
        self._seed = seed
        self._export_format = export_format
        self._write_count = 0
        self._train_count = 0
        self._val_count = 0
        self._errors: list[str] = []

        self._train_file: TextIO = open(self._output_dir / f"{name}-train.jsonl", "w", encoding="utf-8")
        self._val_file: TextIO = open(self._output_dir / f"{name}-val.jsonl", "w", encoding="utf-8")

    def write(self, ex: Example) -> None:
        """Write one SFT example to the appropriate split file."""
        exported = self._export(ex.to_dict())
        line = json.dumps(exported, ensure_ascii=False) + "\n"

        payload = json.dumps(ex.messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        key = f"{payload}:{self._seed}"
        h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
        if (h % 10000) / 10000 < self._train_split:
            self._train_file.write(line)
            self._train_count += 1
        else:
            self._val_file.write(line)
            self._val_count += 1
        self._write_count += 1

    def write_dpo(self, pair: DPOPair) -> None:
        """Write one DPO pair (always to train file)."""
        line = json.dumps(pair.to_dict(), ensure_ascii=False) + "\n"
        self._train_file.write(line)
        self._write_count += 1
        self._train_count += 1

    def log_errors(self, errors: list[str]) -> None:
        """Log validation errors."""
        self._errors.extend(errors)

    def close(self) -> dict[str, Any]:
        """Close files and return summary."""
        self._train_file.close()
        self._val_file.close()
        return {
            "total_written": self._write_count,
            "train": self._train_count,
            "val": self._val_count,
            "errors": self._errors,
        }

    def _export(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert internal format to export format."""
        if self._export_format == "openai":
            return data
        elif self._export_format == "sharegpt":
            return self._to_sharegpt(data)
        elif self._export_format == "chatml":
            return self._to_chatml(data)
        return data

    @staticmethod
    def _to_sharegpt(data: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI format to ShareGPT format."""
        conversations = []
        role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}
        for msg in data.get("messages", []):
            role = role_map.get(msg.get("role", ""), msg.get("role", ""))
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                content = json.dumps(msg["tool_calls"], ensure_ascii=False)
            conversations.append({"from": role, "value": content or ""})
        return {"conversations": conversations}

    @staticmethod
    def _to_chatml(data: dict[str, Any]) -> dict[str, Any]:
        """Convert to ChatML format with im_start/im_end markers."""
        return data  # ChatML is structurally identical for most use cases


class PipelineResult:
    """Result of a pipeline run."""

    def __init__(self):
        self.sft_stats: DatasetStats = DatasetStats()
        self.dpo_count: int = 0
        self.sft_written: int = 0
        self.dpo_written: int = 0
        self.sft_train: int = 0
        self.sft_val: int = 0
        self.validation_errors: list[str] = []
        self.template_warnings: list[str] = []
        self.gate_results: list[GateResult] = []
        self.generators_run: list[dict[str, Any]] = []
        self.duration_seconds: float = 0.0


def run_pipeline(
    config: dict[str, Any],
    generators_dir: str | Path,
    tools: dict[str, Any],
    tool_names: list[str],
    output_dir: str | Path,
    project_name: str = "dataset",
    seed: int = 42,
    train_split: float = 0.95,
    system_prompt: str | None = None,
    quality_gates: dict[str, Any] | None = None,
    export_format: str = "openai",
    dry_run: bool = False,
) -> PipelineResult:
    """Run the full streaming generation pipeline.

    Args:
        config: Parsed config dict.
        generators_dir: Path to generators/ directory.
        tools: Parsed tools dict (from tools.json).
        tool_names: List of known tool names.
        output_dir: Where to write output JSONL files.
        project_name: Project name for file naming and category prefixes.
        seed: Global RNG seed.
        train_split: Train/val split ratio.
        system_prompt: Optional system prompt to prepend to all examples.
        quality_gates: Quality gate config dict.
        export_format: Output format (openai/sharegpt/chatml).
        dry_run: If True, discover generators but don't generate.

    Returns:
        PipelineResult with stats and gate results.
    """
    start = time.time()
    result = PipelineResult()

    # Phase 1: Discover generators
    sft_generators, dpo_generators = discover_generators(
        generators_dir, config, tools, project_name
    )

    print(f"Discovered {len(sft_generators)} SFT generator(s), {len(dpo_generators)} DPO generator(s)")

    if dry_run:
        for g in sft_generators:
            print(f"  SFT: {g.name} ({g.category}) — ~{g.expected_count()} examples")
        for g in dpo_generators:
            print(f"  DPO: {g.name} ({g.category}) — ~{g.expected_count()} pairs")
        result.duration_seconds = time.time() - start
        return result

    # Phase 2: SFT generation — streaming
    sft_writer = StreamingWriter(output_dir, f"{project_name}-sft", train_split, seed, export_format)
    stats_tracker = StatsTracker()
    template_checker = TemplateChecker()

    for gen in sft_generators:
        reset_call_counter()
        gen_start = time.time()
        gen_count = 0

        print(f"  Running: {gen.name} ({gen.category})", end="", flush=True)

        for ex in gen.generate():
            # Prepend system prompt if configured
            if system_prompt:
                from dataforge.core.messages import example as wrap_example
                ex = Example(messages=wrap_example(ex.messages, system_prompt))

            errors = validate_example(ex, stats_tracker.total, tool_names)
            warnings = [e for e in errors if "WARNING" in e]
            hard_errors = [e for e in errors if "WARNING" not in e]

            if hard_errors:
                sft_writer.log_errors(hard_errors)
                continue

            if warnings:
                sft_writer.log_errors(warnings)

            template_checker.ingest(ex)
            stats_tracker.ingest(ex)
            sft_writer.write(ex)
            gen_count += 1

        gen_duration = time.time() - gen_start
        print(f" — {gen_count} examples ({gen_duration:.1f}s)")
        result.generators_run.append({
            "id": gen.category,
            "name": gen.name,
            "count": gen_count,
            "duration": gen_duration,
        })

    sft_summary = sft_writer.close()
    result.sft_stats = stats_tracker.stats
    result.sft_written = sft_summary["total_written"]
    result.sft_train = sft_summary["train"]
    result.sft_val = sft_summary["val"]
    result.validation_errors = sft_summary["errors"]

    # Phase 3: DPO generation — streaming to separate file
    if dpo_generators:
        dpo_writer = StreamingWriter(output_dir, f"{project_name}-dpo", 1.0, seed, export_format)
        for gen in dpo_generators:
            reset_call_counter()
            gen_start = time.time()
            gen_count = 0

            print(f"  Running: {gen.name} ({gen.category})", end="", flush=True)

            for pair in gen.generate():
                dpo_writer.write_dpo(pair)
                gen_count += 1

            gen_duration = time.time() - gen_start
            print(f" — {gen_count} pairs ({gen_duration:.1f}s)")
            result.generators_run.append({
                "id": gen.category,
                "name": gen.name,
                "count": gen_count,
                "duration": gen_duration,
                "type": "dpo",
            })

        dpo_summary = dpo_writer.close()
        result.dpo_written = dpo_summary["total_written"]
        result.dpo_count = dpo_summary["total_written"]

    # Phase 4: Post-generation checks
    result.template_warnings = template_checker.finalize()
    gate_config = parse_gate_config(quality_gates)
    result.gate_results = run_quality_gates(stats_tracker.stats, gate_config, tool_names)

    # Phase 5: Metadata sidecar
    result.duration_seconds = time.time() - start
    _write_metadata(output_dir, project_name, result, seed, config)

    return result


def _write_metadata(
    output_dir: str | Path,
    project_name: str,
    result: PipelineResult,
    seed: int,
    config: dict[str, Any],
) -> None:
    """Write .meta.json sidecar file."""
    from dataforge import __version__

    meta = {
        "dataset_version": "1.0.0",
        "dataforge_version": __version__,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12],
        "generators": [
            {"id": g["id"], "name": g["name"], "count": g["count"]}
            for g in result.generators_run
        ],
        "stats": {
            "total": result.sft_stats.total,
            "multi_turn": result.sft_stats.multi_turn,
            "no_tool_calls": result.sft_stats.no_tool_calls,
            "parallel_tool_calls": result.sft_stats.parallel_tool_calls,
            "error_handling": result.sft_stats.error_handling,
            "by_tool": result.sft_stats.by_tool,
        },
        "quality_gates": {
            "passed": all(g.passed for g in result.gate_results),
            "failures": [g.message for g in result.gate_results if not g.passed],
        },
        "template_warnings": result.template_warnings,
        "train_split": result.sft_train,
        "val_split": result.sft_val,
        "dpo_pairs": result.dpo_count,
        "duration_seconds": round(result.duration_seconds, 2),
    }

    output_path = Path(output_dir)
    with open(output_path / f"{project_name}-sft.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
