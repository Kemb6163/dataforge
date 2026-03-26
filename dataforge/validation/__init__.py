"""Validation: structural checks, template detection, quality gates, stats."""

from dataforge.validation.structural import validate_example
from dataforge.validation.template_detection import TemplateChecker
from dataforge.validation.quality_gates import run_quality_gates
from dataforge.validation.stats import StatsTracker

__all__ = ["validate_example", "TemplateChecker", "run_quality_gates", "StatsTracker"]
