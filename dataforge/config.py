"""YAML config loader with Pydantic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    train_split: float = Field(0.95, ge=0.5, le=1.0)


class ErrorInjectionConfig(BaseModel):
    enabled: bool = True
    base_rate: float = Field(0.10, ge=0.0, le=0.5)
    burst_probability: float = Field(0.13, ge=0.0, le=1.0)


class SFTTrainingConfig(BaseModel):
    epochs: int = Field(3, ge=1)
    batch_size: int = Field(2, ge=1)
    learning_rate: float = Field(2e-5, gt=0)
    max_seq_len: int = Field(4096, ge=256)


class DPOTrainingConfig(BaseModel):
    epochs: int = Field(1, ge=1)
    batch_size: int = Field(1, ge=1)
    learning_rate: float = Field(5e-7, gt=0)
    beta: float = Field(0.1, ge=0.01)


class TrainingConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = Field(16, ge=4)
    lora_alpha: int = Field(32, ge=4)
    sft: SFTTrainingConfig = Field(default_factory=SFTTrainingConfig)
    dpo: DPOTrainingConfig = Field(default_factory=DPOTrainingConfig)


class QualityGatesConfig(BaseModel):
    min_total: int = Field(500, ge=0)
    min_multi_turn: int = Field(30, ge=0)
    min_no_tool: int = Field(50, ge=0)
    min_parallel: int = Field(20, ge=0)
    max_closure_ratio: float = Field(0.65, ge=0.0, le=1.0)
    require_all_tools: bool = True
    min_error_handling: int = Field(10, ge=0)


class ProjectConfig(BaseModel):
    project_name: str = "my-assistant"
    seed: int = 42
    language: str = "en"
    tools_file: str = "tools.json"
    system_prompt: str = ""
    generators_dir: str = "generators"
    output_dir: str = "output"
    export_format: str = Field("openai", pattern=r"^(openai|sharegpt|chatml)$")
    styles: dict[str, dict[str, str]] = Field(default_factory=dict)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)
    error_injection: ErrorInjectionConfig = Field(default_factory=ErrorInjectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a project config from YAML.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Validated ProjectConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raw = {}

    return ProjectConfig(**raw)


def load_tools(tools_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Load tool definitions from JSON.

    Args:
        tools_path: Path to tools.json.

    Returns:
        Tuple of (full tools dict, list of tool names).
    """
    path = Path(tools_path)
    if not path.exists():
        raise FileNotFoundError(f"Tools file not found: {path}")

    with open(path, encoding="utf-8") as f:
        tools_data = json.load(f)

    # Support both list format and dict format
    if isinstance(tools_data, list):
        tools_dict = {}
        tool_names = []
        for tool in tools_data:
            name = tool.get("function", {}).get("name", tool.get("name", ""))
            if name:
                tools_dict[name] = tool
                tool_names.append(name)
        return tools_dict, tool_names
    elif isinstance(tools_data, dict):
        if "tools" in tools_data:
            return load_tools_from_list(tools_data["tools"])
        return tools_data, list(tools_data.keys())
    return {}, []


def load_tools_from_list(tools_list: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Parse a list-format tools definition."""
    tools_dict = {}
    tool_names = []
    for tool in tools_list:
        name = tool.get("function", {}).get("name", tool.get("name", ""))
        if name:
            tools_dict[name] = tool
            tool_names.append(name)
    return tools_dict, tool_names
