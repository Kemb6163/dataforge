"""QLoRA SFT training module.

Requires: torch, transformers, peft, trl, datasets, bitsandbytes, accelerate
Install with: pip install dataforge[train]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def train_sft(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_seq_len: int = 4096,
    quantize_4bit: bool = True,
    target_mlp: bool = True,
    val_split: float = 0.05,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 100,
) -> None:
    """Train a QLoRA SFT adapter.

    Args:
        model_name: HuggingFace model name or local path.
        dataset_path: Path to JSONL training data.
        output_dir: Where to save the adapter.
        lora_rank: LoRA rank (4-64).
        lora_alpha: LoRA alpha (usually 2x rank).
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate.
        max_seq_len: Maximum sequence length.
        quantize_4bit: Use 4-bit quantization (QLoRA).
        target_mlp: Also target MLP layers (not just attention).
        val_split: Validation split ratio.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not installed: {e}\n"
            "Install with: pip install dataforge[train]"
        ) from e

    print(f"Loading model: {model_name}")

    # Quantization config
    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quantize_4bit else None,
    )

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mlp:
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Format function for chat templates
    def format_fn(example: dict) -> dict:
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_dataset = train_dataset.map(format_fn)
    if eval_dataset:
        eval_dataset = eval_dataset.map(format_fn)

    # Training config
    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_len,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save adapter
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    if eval_dataset:
        metrics = trainer.evaluate()
        print(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")

    print("SFT training complete.")
