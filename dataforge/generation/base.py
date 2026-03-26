"""Base generator classes for SFT and DPO dataset generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from dataforge.core.types import Example, DPOPair


class SFTGenerator(ABC):
    """Base class for SFT example generators.

    Subclasses must implement generate() as a generator function that yields
    Example objects one at a time for constant-RAM streaming.
    """

    def __init__(self, config: dict, tools: dict):
        self.config = config
        self.tools = tools

    @abstractmethod
    def generate(self) -> Iterator[Example]:
        """Yield SFT training examples one at a time."""
        ...

    @abstractmethod
    def expected_count(self) -> int:
        """Estimated number of examples. Used for progress logging and metadata."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Unique category ID for RNG isolation.

        MUST be unique across all generators in the project.
        Convention: '{project}.{name}' (e.g. 'restaurant.menu_search').
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...


class DPOGenerator(ABC):
    """Base class for DPO preference pair generators.

    Subclasses must implement generate() as a generator function that yields
    DPOPair objects one at a time.
    """

    def __init__(self, config: dict, tools: dict):
        self.config = config
        self.tools = tools

    @abstractmethod
    def generate(self) -> Iterator[DPOPair]:
        """Yield DPO preference pairs one at a time."""
        ...

    @abstractmethod
    def expected_count(self) -> int:
        """Estimated number of pairs."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Unique category ID for RNG isolation."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...
