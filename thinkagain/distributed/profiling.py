"""Profiling infrastructure for replica call patterns.

This module provides lightweight instrumentation to track which @node functions
call which @replica classes, enabling dependency graph construction for
optimization and capacity planning.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

# Context-local storage for tracking execution context
_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_node", default=None
)

# Global profiler instance
_profiler: ReplicaProfiler | None = None
_profiler_lock = threading.Lock()


class ReplicaProfiler:
    """Tracks replica call patterns during node execution.

    This profiler captures:
    1. Which @node functions call which @replica classes
    2. Fanout ratios (average calls per request)
    3. Service times for each replica method
    4. Arrival rates for capacity planning
    """

    def __init__(self, max_service_samples: int | None = 10_000):
        # Node name -> {replica names called}
        self._node_to_replicas: dict[str, set[str]] = defaultdict(set)

        # Node name -> replica name -> call count
        self._fanout_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Node name -> execution count (for computing average fanout)
        self._node_executions: dict[str, int] = defaultdict(int)

        # Replica name -> bounded service times in seconds (for stats)
        self._service_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_service_samples)
        )
        self._max_service_samples = max_service_samples

        # Replica name -> total call count
        self._replica_calls: dict[str, int] = defaultdict(int)

        # External calls (not from a node context)
        self._external_calls: dict[str, int] = defaultdict(int)

        # Lock for thread-safe updates
        self._lock = threading.Lock()

        # Start time for computing rates
        self._start_time = time.time()

    def record_node_execution(self, node_name: str) -> None:
        """Record that a node is being executed."""
        with self._lock:
            self._node_executions[node_name] += 1

    def record_replica_call(
        self,
        replica_name: str,
        caller_node: str | None,
        duration: float | None = None,
    ) -> None:
        """Record a replica method call.

        Args:
            replica_name: Name of the @replica class being called
            caller_node: Name of the @node calling it (None if external)
            duration: Optional service time in seconds
        """
        with self._lock:
            self._replica_calls[replica_name] += 1

            if caller_node:
                # Track node -> replica dependency
                self._node_to_replicas[caller_node].add(replica_name)
                self._fanout_counts[caller_node][replica_name] += 1
            else:
                # External call (not from a node)
                self._external_calls[replica_name] += 1

            if duration is not None:
                self._service_times[replica_name].append(duration)

    def get_dependency_graph(self) -> dict[str, set[str]]:
        """Get the node -> replicas dependency graph.

        Returns:
            Dict mapping node names to the set of replica classes they call.
        """
        with self._lock:
            return {
                node: set(replicas) for node, replicas in self._node_to_replicas.items()
            }

    def get_fanout_matrix(self) -> dict[str, dict[str, float]]:
        """Get average fanout: calls from node N to replica R per execution.

        Returns:
            Dict[node_name, Dict[replica_name, avg_calls_per_execution]]
        """
        with self._lock:
            fanout = {}
            for node, replica_counts in self._fanout_counts.items():
                node_execs = self._node_executions[node]
                if node_execs > 0:
                    fanout[node] = {
                        replica: count / node_execs
                        for replica, count in replica_counts.items()
                    }
            return fanout

    def get_service_stats(self) -> dict[str, dict[str, float]]:
        """Get service time statistics for each replica.

        Returns:
            Dict[replica_name, Dict[stat_name, value]]
            Stats include: mean, p50, p95, p99 (in seconds)
        """
        try:
            import numpy as np
        except ModuleNotFoundError:
            np = None
        import math
        import statistics

        with self._lock:
            stats = {}
            for replica, times in self._service_times.items():
                if times:
                    sample = list(times)
                    if np is not None:
                        stats[replica] = {
                            "mean": float(np.mean(sample)),
                            "p50": float(np.percentile(sample, 50)),
                            "p95": float(np.percentile(sample, 95)),
                            "p99": float(np.percentile(sample, 99)),
                            "count": len(sample),
                        }
                    else:
                        sorted_times = sorted(sample)
                        mean = statistics.fmean(sorted_times)

                        def percentile(percent: float) -> float:
                            k = (len(sorted_times) - 1) * (percent / 100)
                            f = math.floor(k)
                            c = math.ceil(k)
                            if f == c:
                                return sorted_times[int(k)]
                            return sorted_times[f] + (
                                (sorted_times[c] - sorted_times[f]) * (k - f)
                            )

                        stats[replica] = {
                            "mean": float(mean),
                            "p50": float(percentile(50)),
                            "p95": float(percentile(95)),
                            "p99": float(percentile(99)),
                            "count": len(sorted_times),
                        }
            return stats

    def get_call_counts(self) -> dict[str, int]:
        """Get total call counts for each replica."""
        with self._lock:
            return dict(self._replica_calls)

    def get_external_call_counts(self) -> dict[str, int]:
        """Get external call counts (calls not from @node context)."""
        with self._lock:
            return dict(self._external_calls)

    def get_node_executions(self) -> dict[str, int]:
        """Get execution counts for each node."""
        with self._lock:
            return dict(self._node_executions)

    def get_elapsed_time(self) -> float:
        """Get elapsed time since profiler started (seconds)."""
        return time.time() - self._start_time

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._node_to_replicas.clear()
            self._fanout_counts.clear()
            self._node_executions.clear()
            self._service_times.clear()
            self._replica_calls.clear()
            self._external_calls.clear()
            self._start_time = time.time()

    def summary(self) -> dict[str, Any]:
        """Get a complete summary of profiling data."""
        return {
            "dependency_graph": self.get_dependency_graph(),
            "fanout_matrix": self.get_fanout_matrix(),
            "service_stats": self.get_service_stats(),
            "call_counts": self.get_call_counts(),
            "external_calls": self.get_external_call_counts(),
            "node_executions": self.get_node_executions(),
            "elapsed_seconds": self.get_elapsed_time(),
        }


@dataclass(frozen=True)
class ProfilingSession:
    """Active profiling session configuration."""

    profiler: "ReplicaProfiler"
    context_factory: Callable[[str], ContextManager[None]]


# Context management for thread-local execution tracking


def get_current_node() -> str | None:
    """Get the currently executing node name, if any."""
    return _current_node.get()


@contextmanager
def node_context(node_name: str):
    """Context manager for node execution tracking.

    Usage (in executor):
        with node_context(node.name):
            result = await node.execute(...)
    """
    profiler = _profiler
    if profiler is None:
        yield
        return

    token = _current_node.set(node_name)
    profiler.record_node_execution(node_name)

    try:
        yield
    finally:
        _current_node.reset(token)


# Public API for profiling control


def enable_profiling(max_service_samples: int | None = 10_000) -> ReplicaProfiler:
    """Enable replica call profiling.

    Returns:
        The active profiler instance.

    Example:
        profiler = enable_profiling()
        # ... run your workload ...
        stats = profiler.summary()
        disable_profiling()

    Note:
        To track @node executions, pass node_context as the Context
        factory (see profiling_enabled()).
    """
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            _profiler = ReplicaProfiler(max_service_samples=max_service_samples)
        return _profiler


def disable_profiling() -> None:
    """Disable replica call profiling."""
    global _profiler
    with _profiler_lock:
        _profiler = None


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled."""
    return _profiler is not None


def get_profiler() -> ReplicaProfiler | None:
    """Get the active profiler instance, if any."""
    return _profiler


def get_node_context_factory() -> Callable[[str], ContextManager[None]]:
    """Return the context factory used to track node execution."""
    return node_context


@contextmanager
def profiling_enabled(max_service_samples: int | None = 10_000):
    """Context manager for scoped profiling.

    Example:
        with profiling_enabled() as session:
            result = run(pipeline, data, context_factory=session.context_factory)
            print(session.profiler.summary())
    """
    profiler = enable_profiling(max_service_samples=max_service_samples)
    try:
        yield ProfilingSession(profiler=profiler, context_factory=node_context)
    finally:
        disable_profiling()


# Internal API for replica tracking


def record_replica_call(replica_name: str, duration: float | None = None) -> None:
    """Record a replica call (called by ReplicaSpec.get_instance).

    Args:
        replica_name: Name of the replica class
        duration: Optional service time in seconds
    """
    profiler = _profiler
    if profiler is not None:
        profiler.record_replica_call(replica_name, get_current_node(), duration)
