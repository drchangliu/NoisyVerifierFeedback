from __future__ import annotations

from abc import ABC, abstractmethod

from nvf.analyzers.finding import Finding


class Analyzer(ABC):
    """Abstract interface for static security analyzers."""

    @abstractmethod
    def analyze(self, code: str, filename: str = "target.py") -> list[Finding]:
        """Run analysis on code string and return findings."""
        ...
