"""CLI commands for DataForge: generate, validate, inspect, train, merge, init, diff, sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataforge",
        description="DataForge — Synthetic dataset generation toolkit for LLM fine-tuning",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    p_gen = sub.add_parser("generate", help="Generate dataset from config")
    p_gen.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    p_gen.add_argument("--dry-run", action="store_true", help="Discover generators without generating")
    p_gen.add_argument("--format", choices=["openai", "sharegpt", "chatml"], help="Export format override")

    # validate
    p_val = sub.add_parser("validate", help="Validate an existing JSONL dataset")
    p_val.add_argument("dataset", help="Path to JSONL file")
    p_val.add_argument("--tools", help="Path to tools.json for tool name checks")

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect dataset statistics (LLM dataset linter)")
    p_insp.add_argument("dataset", help="Path to JSONL file")

    # train
    p_train = sub.add_parser("train", help="Train LoRA adapter")
    p_train_sub = p_train.add_subparsers(dest="train_type")

    p_sft = p_train_sub.add_parser("sft", help="Train SFT LoRA")
    p_sft.add_argument("--config", "-c", default="config.yaml")
    p_sft.add_argument("--dataset", required=True, help="Path to SFT JSONL")
    p_sft.add_argument("--dry-run", action="store_true")

    p_dpo = p_train_sub.add_parser("dpo", help="Train DPO LoRA")
    p_dpo.add_argument("--config", "-c", default="config.yaml")
    p_dpo.add_argument("--adapter", required=True, help="Path to SFT adapter")
    p_dpo.add_argument("--dataset", required=True, help="Path to DPO JSONL")
    p_dpo.add_argument("--contrastive", help="Optional contrastive set JSONL")
    p_dpo.add_argument("--dry-run", action="store_true")

    # merge
    p_merge = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    p_merge.add_argument("--base", required=True, help="Base model name or path")
    p_merge.add_argument("--adapter", required=True, help="Adapter path")
    p_merge.add_argument("--output", help="Output path (default: {adapter}-merged)")

    # init
    p_init = sub.add_parser("init", help="Initialize new project from template")
    p_init.add_argument("name", help="Project directory name")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two dataset versions")
    p_diff.add_argument("file_a", help="First JSONL file")
    p_diff.add_argument("file_b", help="Second JSONL file")

    # sample
    p_sample = sub.add_parser("sample", help="Show random examples from dataset")
    p_sample.add_argument("dataset", help="Path to JSONL file")
    p_sample.add_argument("--n", type=int, default=3, help="Number of examples to show")
    p_sample.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "inspect": cmd_inspect,
        "train": cmd_train,
        "merge": cmd_merge,
        "init": cmd_init,
        "diff": cmd_diff,
        "sample": cmd_sample,
    }
    commands[args.command](args)


def cmd_generate(args: argparse.Namespace) -> None:
    from dataforge.config import load_config, load_tools
    from dataforge.generation.pipeline import run_pipeline

    config = load_config(args.config)
    config_dir = Path(args.config).parent

    tools_path = config_dir / config.tools_file
    tools_dict, tool_names = load_tools(tools_path)

    export_format = args.format or config.export_format

    raw_config = {
        "project_name": config.project_name,
        "seed": config.seed,
        "language": config.language,
        "styles": config.styles,
        "error_injection": config.error_injection.model_dump(),
    }

    print(f"DataForge v{_version()} — Generating dataset: {config.project_name}")
    print(f"  Seed: {config.seed} | Format: {export_format} | Tools: {len(tool_names)}")
    print()

    result = run_pipeline(
        config=raw_config,
        generators_dir=config_dir / config.generators_dir,
        tools=tools_dict,
        tool_names=tool_names,
        output_dir=config_dir / config.output_dir,
        project_name=config.project_name,
        seed=config.seed,
        train_split=config.dataset.train_split,
        system_prompt=config.system_prompt or None,
        quality_gates=config.quality_gates.model_dump(),
        export_format=export_format,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run complete.")
        return

    print()
    print(f"SFT: {result.sft_written} examples (train: {result.sft_train}, val: {result.sft_val})")
    if result.dpo_count:
        print(f"DPO: {result.dpo_count} pairs")

    if result.validation_errors:
        warnings = [e for e in result.validation_errors if "WARNING" in e]
        errors = [e for e in result.validation_errors if "WARNING" not in e]
        if errors:
            print(f"\nValidation errors: {len(errors)}")
            for e in errors[:10]:
                print(f"  {e}")
        if warnings:
            print(f"\nValidation warnings: {len(warnings)}")

    if result.template_warnings:
        print(f"\nTemplate warnings:")
        for w in result.template_warnings:
            print(f"  {w}")

    print(f"\nQuality Gates:")
    all_passed = True
    for g in result.gate_results:
        status = "PASS" if g.passed else "FAIL"
        symbol = "+" if g.passed else "x"
        print(f"  [{symbol}] {g.gate}: {g.message}")
        if not g.passed:
            all_passed = False

    print(f"\nDuration: {result.duration_seconds:.1f}s")
    if all_passed:
        print("All quality gates passed.")
    else:
        print("WARNING: Some quality gates failed.")
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    from dataforge.core.types import Example
    from dataforge.validation.structural import validate_example

    tool_names = None
    if args.tools:
        from dataforge.config import load_tools
        _, tool_names = load_tools(args.tools)

    path = Path(args.dataset)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    total = 0
    errors_count = 0
    all_errors: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"[line {line_num + 1}] Invalid JSON: {e}")
                errors_count += 1
                continue

            msgs = data.get("messages", data.get("conversations", []))
            ex = Example(messages=msgs)
            errs = validate_example(ex, total, tool_names)
            if errs:
                all_errors.extend(errs)
                errors_count += len([e for e in errs if "WARNING" not in e])
            total += 1

    print(f"Validated {total} examples from {path.name}")
    if all_errors:
        warnings = [e for e in all_errors if "WARNING" in e]
        hard = [e for e in all_errors if "WARNING" not in e]
        if hard:
            print(f"\nErrors ({len(hard)}):")
            for e in hard[:20]:
                print(f"  {e}")
            if len(hard) > 20:
                print(f"  ... and {len(hard) - 20} more")
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings[:10]:
                print(f"  {w}")
    else:
        print("No errors found.")


def cmd_inspect(args: argparse.Namespace) -> None:
    path = Path(args.dataset)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    # Check for sidecar metadata
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        stem = path.stem
        if stem.endswith("-train") or stem.endswith("-val"):
            base_stem = stem.rsplit("-", 1)[0]
            meta_path = path.parent / f"{base_stem}.meta.json"

    meta = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    # Scan dataset
    from dataforge.core.types import Example
    from dataforge.validation.stats import StatsTracker
    from dataforge.validation.template_detection import TemplateChecker

    stats = StatsTracker()
    checker = TemplateChecker()
    total_tokens_est = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = data.get("messages", data.get("conversations", []))
            ex = Example(messages=msgs)
            stats.ingest(ex)
            checker.ingest(ex)
            for m in msgs:
                c = m.get("content") or m.get("value") or ""
                total_tokens_est += len(str(c)) // 4

    s = stats.stats
    template_warnings = checker.finalize()

    # Print report
    print()
    print("=" * 50)
    print("  DataForge Dataset Inspector")
    print("=" * 50)
    print(f"  File: {path.name}")
    if meta:
        print(f"  Generated: {meta.get('timestamp', 'unknown')}")
        print(f"  Seed: {meta.get('seed', '?')} | DataForge v{meta.get('dataforge_version', '?')}")
    print("=" * 50)
    print()

    avg_msgs = s.avg_messages_per_example
    avg_tokens = total_tokens_est / s.total if s.total > 0 else 0
    print(f"Overview")
    print(f"  Total examples:    {s.total}")
    print(f"  Avg messages/ex:   {avg_msgs:.1f}")
    print(f"  Avg tokens/ex:     ~{int(avg_tokens):,}")
    print()

    # Tool distribution
    if s.by_tool:
        print("Tool Usage Distribution")
        max_count = max(s.by_tool.values()) if s.by_tool else 1
        for name, count in sorted(s.by_tool.items(), key=lambda x: -x[1]):
            pct = count / s.total * 100 if s.total > 0 else 0
            bar_len = int(count / max_count * 20)
            bar = "#" * bar_len
            print(f"  {name:25s} {bar:20s}  {pct:5.1f}%  ({count})")
        print()

    # Conversation patterns
    print("Conversation Patterns")
    single = s.total - s.multi_turn
    print(f"  Single-turn:     {_pct(single, s.total):6s}  ({single})")
    print(f"  Multi-turn:      {_pct(s.multi_turn, s.total):6s}  ({s.multi_turn})")
    print(f"  No-tool:         {_pct(s.no_tool_calls, s.total):6s}  ({s.no_tool_calls})")
    print(f"  Parallel calls:  {_pct(s.parallel_tool_calls, s.total):6s}  ({s.parallel_tool_calls})")
    print(f"  Error handling:  {_pct(s.error_handling, s.total):6s}  ({s.error_handling})")
    print()

    # Template similarity
    print("Template Similarity")
    print(f"  Structural dups:  {checker.structural_dup_count}")
    print(f"  Flow pattern dups: {checker.flow_dup_count}")
    if template_warnings:
        for w in template_warnings:
            print(f"  WARNING: {w}")
    else:
        print("  No template issues detected.")
    print()

    # Quality gates (if metadata available)
    if meta and "quality_gates" in meta:
        gates = meta["quality_gates"]
        if gates.get("passed"):
            n_gates = len(meta.get("stats", {}))
            print(f"Quality Gates: ALL PASSED")
        else:
            print(f"Quality Gates: FAILURES")
            for f_msg in gates.get("failures", []):
                print(f"  [x] {f_msg}")


def cmd_train(args: argparse.Namespace) -> None:
    if args.train_type == "sft":
        _cmd_train_sft(args)
    elif args.train_type == "dpo":
        _cmd_train_dpo(args)
    else:
        print("Usage: dataforge train {sft|dpo}")
        sys.exit(1)


def _cmd_train_sft(args: argparse.Namespace) -> None:
    from dataforge.config import load_config

    config = load_config(args.config)
    tc = config.training

    print(f"DataForge SFT Training")
    print(f"  Model: {tc.model}")
    print(f"  LoRA: rank={tc.lora_rank}, alpha={tc.lora_alpha}")
    print(f"  Epochs: {tc.sft.epochs}, Batch: {tc.sft.batch_size}, LR: {tc.sft.learning_rate}")
    print(f"  Max seq len: {tc.sft.max_seq_len}")
    print(f"  Dataset: {args.dataset}")

    if args.dry_run:
        print("\nDry run — config valid, training would start with above parameters.")
        return

    from dataforge.training.sft import train_sft

    train_sft(
        model_name=tc.model,
        dataset_path=args.dataset,
        output_dir=str(Path(args.dataset).parent / "sft-adapter"),
        lora_rank=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        epochs=tc.sft.epochs,
        batch_size=tc.sft.batch_size,
        learning_rate=tc.sft.learning_rate,
        max_seq_len=tc.sft.max_seq_len,
    )


def _cmd_train_dpo(args: argparse.Namespace) -> None:
    from dataforge.config import load_config

    config = load_config(args.config)
    tc = config.training

    print(f"DataForge DPO Training")
    print(f"  Model: {tc.model}")
    print(f"  SFT Adapter: {args.adapter}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Beta: {tc.dpo.beta}, Epochs: {tc.dpo.epochs}, LR: {tc.dpo.learning_rate}")

    if args.dry_run:
        print("\nDry run — config valid, DPO training would start.")
        return

    from dataforge.training.dpo import train_dpo

    train_dpo(
        model_name=tc.model,
        sft_adapter=args.adapter,
        dpo_data_path=args.dataset,
        output_dir=str(Path(args.dataset).parent / "dpo-adapter"),
        contrastive_path=args.contrastive,
        beta=tc.dpo.beta,
        epochs=tc.dpo.epochs,
        learning_rate=tc.dpo.learning_rate,
    )


def cmd_merge(args: argparse.Namespace) -> None:
    from dataforge.training.merge import merge_adapter

    output = args.output or f"{args.adapter}-merged"
    print(f"Merging adapter into base model...")
    print(f"  Base: {args.base}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Output: {output}")
    merge_adapter(args.base, args.adapter, output)
    print("Merge complete.")


def cmd_init(args: argparse.Namespace) -> None:
    project_dir = Path(args.name)
    if project_dir.exists():
        print(f"Error: directory '{args.name}' already exists")
        sys.exit(1)

    project_dir.mkdir(parents=True)
    (project_dir / "generators").mkdir()
    (project_dir / "output").mkdir()

    # config.yaml
    config_content = f'''project_name: "{args.name}"
seed: 42
language: "en"
tools_file: "tools.json"
system_prompt: "You are a helpful assistant with access to tools."

generators_dir: "generators"
output_dir: "output"

dataset:
  train_split: 0.95

quality_gates:
  min_total: 100
  min_multi_turn: 10
  min_no_tool: 10
  min_parallel: 5
  max_closure_ratio: 0.65
  require_all_tools: true

error_injection:
  enabled: true
  base_rate: 0.10
'''
    (project_dir / "config.yaml").write_text(config_content, encoding="utf-8")

    # tools.json
    tools_content = json.dumps([
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for items",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_details",
                "description": "Get details for an item",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Item ID"}
                    },
                    "required": ["id"],
                },
            },
        },
    ], indent=2)
    (project_dir / "tools.json").write_text(tools_content, encoding="utf-8")

    # generators/__init__.py
    init_content = '''"""Custom generators for this project."""
'''
    (project_dir / "generators" / "__init__.py").write_text(init_content, encoding="utf-8")

    # data_pools.py
    pools_content = '''"""Data pools for generating realistic fake data."""

from dataforge.core.rng import make_rng
from dataforge.generation.pools import fake_name, fake_id


# Add your data pools here
ITEMS = ["Item A", "Item B", "Item C", "Item D", "Item E"]
'''
    (project_dir / "data_pools.py").write_text(pools_content, encoding="utf-8")

    print(f"Project '{args.name}' initialized.")
    print(f"  {project_dir}/config.yaml")
    print(f"  {project_dir}/tools.json")
    print(f"  {project_dir}/generators/")
    print(f"  {project_dir}/data_pools.py")
    print()
    print("Next steps:")
    print("  1. Edit tools.json with your tool definitions")
    print("  2. Create generators in generators/")
    print("  3. Run: dataforge generate --config config.yaml")


def cmd_diff(args: argparse.Namespace) -> None:
    path_a = Path(args.file_a)
    path_b = Path(args.file_b)

    for p in [path_a, path_b]:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    from dataforge.core.types import Example
    from dataforge.validation.stats import StatsTracker

    def scan_file(path: Path) -> tuple[StatsTracker, int, int]:
        tracker = StatsTracker()
        total_tokens = 0
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = data.get("messages", data.get("conversations", []))
                ex = Example(messages=msgs)
                tracker.ingest(ex)
                for m in msgs:
                    c = m.get("content") or ""
                    total_tokens += len(str(c)) // 4
                count += 1
        return tracker, total_tokens, count

    ta, tok_a, cnt_a = scan_file(path_a)
    tb, tok_b, cnt_b = scan_file(path_b)
    sa = ta.stats
    sb = tb.stats

    print()
    print("=" * 50)
    print("  DataForge Dataset Diff")
    print(f"  {path_a.name} vs {path_b.name}")
    print("=" * 50)
    print()

    avg_tok_a = tok_a / cnt_a if cnt_a else 0
    avg_tok_b = tok_b / cnt_b if cnt_b else 0

    print("Overview")
    print(f"  {'':20s} {'v1':>10s} {'v2':>10s} {'delta':>15s}")
    print(f"  {'Total examples:':20s} {sa.total:>10d} {sb.total:>10d} {_delta(sa.total, sb.total):>15s}")
    print(f"  {'Avg tokens/ex:':20s} {int(avg_tok_a):>10,d} {int(avg_tok_b):>10,d} {_delta(int(avg_tok_a), int(avg_tok_b)):>15s}")
    print()

    # Tool distribution changes
    all_tools = sorted(set(list(sa.by_tool.keys()) + list(sb.by_tool.keys())))
    if all_tools:
        print("Tool Distribution Changes")
        for t in all_tools:
            ca = sa.by_tool.get(t, 0)
            cb = sb.by_tool.get(t, 0)
            pa = ca / sa.total * 100 if sa.total else 0
            pb = cb / sb.total * 100 if sb.total else 0
            diff_str = f"{pb - pa:+.1f}%"
            marker = ""
            if ca == 0:
                marker = " (NEW)"
            elif cb == 0:
                marker = " (REMOVED)"
            print(f"  {t:25s} {pa:6.1f}% -> {pb:6.1f}%  {diff_str}{marker}")
        print()

    # Pattern changes
    print("Pattern Changes")
    print(f"  {'Multi-turn:':20s} {_pct(sa.multi_turn, sa.total):>8s} -> {_pct(sb.multi_turn, sb.total):>8s}")
    print(f"  {'No-tool:':20s} {_pct(sa.no_tool_calls, sa.total):>8s} -> {_pct(sb.no_tool_calls, sb.total):>8s}")
    print(f"  {'Parallel:':20s} {_pct(sa.parallel_tool_calls, sa.total):>8s} -> {_pct(sb.parallel_tool_calls, sb.total):>8s}")
    print(f"  {'Error handling:':20s} {_pct(sa.error_handling, sa.total):>8s} -> {_pct(sb.error_handling, sb.total):>8s}")


def cmd_sample(args: argparse.Namespace) -> None:
    import random

    path = Path(args.dataset)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    examples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not examples:
        print("No valid examples found.")
        return

    rng = random.Random(args.seed)
    n = min(args.n, len(examples))
    samples = rng.sample(examples, n)

    for i, ex in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"  Example {i + 1}/{n}")
        print(f"{'='*50}")
        msgs = ex.get("messages", ex.get("conversations", []))
        for msg in msgs:
            role = msg.get("role", msg.get("from", "?"))
            content = msg.get("content", msg.get("value", ""))
            tool_calls = msg.get("tool_calls", [])

            if role == "system":
                print(f"\n  [SYSTEM] {_truncate(str(content), 80)}")
            elif role in ("user", "human"):
                print(f"\n  [USER] {content}")
            elif role in ("assistant", "gpt"):
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        print(f"\n  [TOOL CALL] {fn.get('name', '?')}({json.dumps(fn.get('arguments', {}), ensure_ascii=False)})")
                if content:
                    print(f"\n  [ASSISTANT] {content}")
            elif role == "tool":
                call_id = msg.get("tool_call_id", "?")
                print(f"\n  [TOOL RESULT] ({call_id}) {_truncate(str(content), 120)}")


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{n / total * 100:.1f}%"


def _delta(a: int, b: int) -> str:
    diff = b - a
    if a == 0:
        return f"+{diff}" if diff > 0 else str(diff)
    pct = diff / a * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff} ({sign}{pct:.1f}%)"


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _version() -> str:
    from dataforge import __version__
    return __version__


if __name__ == "__main__":
    main()
