"""DPO training module with contrastive set conversion.

Requires: torch, transformers, peft, trl, datasets
Install with: pip install dataforge[train]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataforge.core.types import DPOPair, ContrastiveSet


def convert_contrastive_to_dpo(contrastive_path: str) -> list[dict[str, Any]]:
    """Convert ranked contrastive sets to pairwise DPO preferences.

    Each contrastive set has a prompt and ranked responses (lower rank = better).
    This generates all valid preference pairs: every higher-ranked response is
    'chosen' against every lower-ranked response.

    Args:
        contrastive_path: Path to JSONL with contrastive sets.

    Returns:
        List of DPO pair dicts ready for training.
    """
    pairs: list[dict[str, Any]] = []

    with open(contrastive_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cs = ContrastiveSet(
                prompt=data["prompt"],
                responses=data["responses"],
            )
            for pair in cs.to_dpo_pairs():
                pairs.append(pair.to_dict())

    return pairs


def train_dpo(
    model_name: str,
    sft_adapter: str,
    dpo_data_path: str,
    output_dir: str,
    contrastive_path: str | None = None,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-7,
    max_seq_len: int = 4096,
    gradient_accumulation_steps: int = 8,
    warmup_ratio: float = 0.1,
    logging_steps: int = 5,
) -> None:
    """Train a DPO adapter on top of an SFT adapter.

    Args:
        model_name: Base model name (same as used for SFT).
        sft_adapter: Path to the SFT LoRA adapter.
        dpo_data_path: Path to DPO JSONL data.
        output_dir: Where to save the DPO adapter.
        contrastive_path: Optional path to contrastive sets to convert.
        beta: DPO beta parameter (controls KL penalty).
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate.
        max_seq_len: Maximum sequence length.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        logging_steps: Log every N steps.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel, LoraConfig
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not installed: {e}\n"
            "Install with: pip install dataforge[train]"
        ) from e

    # Load DPO data
    dpo_examples: list[dict[str, Any]] = []
    with open(dpo_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dpo_examples.append(json.loads(line))

    # Convert contrastive sets if provided
    if contrastive_path:
        contrastive_pairs = convert_contrastive_to_dpo(contrastive_path)
        dpo_examples.extend(contrastive_pairs)
        print(f"Added {len(contrastive_pairs)} pairs from contrastive sets")

    print(f"Total DPO pairs: {len(dpo_examples)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load SFT adapter
    model = PeftModel.from_pretrained(model, sft_adapter)
    model = model.merge_and_unload()

    # Prepare DPO dataset
    def format_messages(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    formatted = []
    for ex in dpo_examples:
        prompt_text = format_messages(ex["prompt"])
        chosen_text = format_messages(ex["prompt"] + ex["chosen"])
        rejected_text = format_messages(ex["prompt"] + ex["rejected"])
        formatted.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })

    dataset = Dataset.from_list(formatted)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_seq_len,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # New LoRA for DPO
    dpo_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=dpo_lora_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("DPO training complete.")
