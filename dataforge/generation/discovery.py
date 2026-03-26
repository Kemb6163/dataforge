"""Generator discovery: auto-discover from directories and entry_points."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from dataforge.generation.base import SFTGenerator, DPOGenerator


def discover_generators(
    generators_dir: str | Path,
    config: dict,
    tools: dict,
    project_name: str = "",
) -> tuple[list[SFTGenerator], list[DPOGenerator]]:
    """Discover and instantiate all generators.

    Sources:
    1. Python files in generators_dir (local generators)
    2. Entry points registered under 'dataforge.generators' (plugins)

    Local generators take precedence over entry_points for same category.
    Duplicate categories within the same source raise ValueError.

    Args:
        generators_dir: Path to the generators/ directory.
        config: Parsed config dict.
        tools: Parsed tools dict.
        project_name: Project name for auto-prefixing categories.

    Returns:
        Tuple of (sft_generators, dpo_generators).
    """
    sft: list[SFTGenerator] = []
    dpo: list[DPOGenerator] = []
    seen_categories: dict[str, str] = {}  # category -> source file/module

    # Phase 1: Local directory discovery
    gen_path = Path(generators_dir)
    if gen_path.is_dir():
        _discover_from_directory(gen_path, config, tools, project_name, sft, dpo, seen_categories)

    # Phase 2: Entry points discovery
    _discover_from_entry_points(config, tools, sft, dpo, seen_categories)

    return sft, dpo


def _discover_from_directory(
    gen_path: Path,
    config: dict,
    tools: dict,
    project_name: str,
    sft: list[SFTGenerator],
    dpo: list[DPOGenerator],
    seen_categories: dict[str, str],
) -> None:
    """Discover generators from a local directory."""
    for py_file in sorted(gen_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"generators.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not isinstance(obj, type):
                continue
            if obj in (SFTGenerator, DPOGenerator):
                continue

            if issubclass(obj, SFTGenerator):
                instance = obj(config, tools)
                _register(instance, py_file.name, project_name, sft, seen_categories, is_sft=True)
            elif issubclass(obj, DPOGenerator):
                instance = obj(config, tools)
                _register(instance, py_file.name, project_name, dpo, seen_categories, is_sft=False)


def _discover_from_entry_points(
    config: dict,
    tools: dict,
    sft: list[SFTGenerator],
    dpo: list[DPOGenerator],
    seen_categories: dict[str, str],
) -> None:
    """Discover generators from installed entry_points."""
    try:
        if sys.version_info >= (3, 12):
            from importlib.metadata import entry_points
            eps = entry_points(group="dataforge.generators")
        else:
            from importlib.metadata import entry_points
            all_eps = entry_points()
            eps = all_eps.get("dataforge.generators", [])
    except Exception:
        return

    for ep in eps:
        try:
            cls = ep.load()
        except Exception:
            continue

        if isinstance(cls, type) and issubclass(cls, (SFTGenerator, DPOGenerator)):
            instance = cls(config, tools)
            cat = instance.category
            # Entry points don't auto-prefix — they manage their own namespacing
            if cat in seen_categories:
                # Local takes precedence over entry_points — skip
                continue

            if issubclass(cls, SFTGenerator):
                sft.append(instance)
            else:
                dpo.append(instance)
            seen_categories[cat] = f"entry_point:{ep.name}"


def _register(
    instance: Any,
    source_file: str,
    project_name: str,
    target_list: list,
    seen_categories: dict[str, str],
    is_sft: bool,
) -> None:
    """Register a generator, enforcing unique categories."""
    cat = instance.category

    # Auto-prefix local generators without a dot in their category
    if project_name and "." not in cat:
        cat = f"{project_name}.{cat}"
        # Patch the instance's category (store the namespaced version)
        instance._namespaced_category = cat

    if cat in seen_categories:
        raise ValueError(
            f"Duplicate category '{cat}' in generators: "
            f"{seen_categories[cat]} and {source_file}"
        )

    seen_categories[cat] = source_file
    target_list.append(instance)
