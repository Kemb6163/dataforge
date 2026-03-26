"""LoRA adapter merge into base model.

Requires: torch, transformers, peft
Install with: pip install dataforge[train]
"""

from __future__ import annotations

from pathlib import Path


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_path: str | None = None,
) -> None:
    """Merge a LoRA adapter into the base model.

    Creates a full model with the adapter weights baked in.
    The output can be loaded without peft.

    Args:
        base_model: HuggingFace model name or local path.
        adapter_path: Path to the LoRA adapter directory.
        output_path: Where to save the merged model.
                     Defaults to {adapter_path}-merged.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not installed: {e}\n"
            "Install with: pip install dataforge[train]"
        ) from e

    if output_path is None:
        output_path = f"{adapter_path}-merged"

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    total_params = sum(p.numel() for p in model.parameters())
    size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    print(f"Merged model: {total_params:,} parameters ({size_gb:.1f} GB)")
    print("Merge complete.")
