"""Execution metadata for pipeline runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ExecutionMetadata:
    """Execution metadata tracking for a Context."""

    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    node_latencies: list[tuple[str, float]] = field(default_factory=list)
    per_node_totals: dict[str, float] = field(default_factory=dict)
    node_execution_count: int = 0

    @property
    def total_duration(self) -> float | None:
        """Total duration from creation to finish, or None if still running."""
        if self.finished_at is None:
            return None
        return self.finished_at - self.created_at

    def copy(self) -> "ExecutionMetadata":
        """Create a deep copy of this metadata."""
        return ExecutionMetadata(
            created_at=self.created_at,
            finished_at=self.finished_at,
            node_latencies=list(self.node_latencies),
            per_node_totals=dict(self.per_node_totals),
            node_execution_count=self.node_execution_count,
        )

    def record_node(self, name: str, duration: float) -> None:
        """Record timing for a node execution."""
        self.node_latencies.append((name, duration))
        self.per_node_totals[name] = self.per_node_totals.get(name, 0.0) + duration
        self.node_execution_count += 1

    def reset(self) -> None:
        """Clear timing data for fresh execution."""
        self.created_at = time.time()
        self.finished_at = None
        self.node_latencies = []
        self.per_node_totals = {}
        self.node_execution_count = 0
