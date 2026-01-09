"""Transparent profiling for distributed execution.

Tracks execution metrics for @jit functions and @replicate calls to enable
automatic optimization and capacity planning.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any

# Context-local storage for tracking execution context
_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_node", default=None
)

# Global profiler instance (always active in the background)
_profiler: ExecutionProfiler | None = None
_profiler_lock = threading.Lock()


class ExecutionProfiler:
    """Tracks execution patterns for @jit and @replicate functions.

    This profiler captures:
    1. Which @jit functions call which @replicate functions
    2. Fanout ratios (average calls per execution)
    3. Execution times for nodes and replicated functions
    4. Call counts for capacity planning

    The profiler runs transparently in the background and can be queried
    at any time without disrupting execution.
    """

    def __init__(self, max_samples: int = 10_000):
        # Node name -> {replicate function names called}
        self._node_to_replicates: dict[str, set[str]] = defaultdict(set)

        # Node name -> replicate name -> call count
        self._fanout_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Node name -> execution count
        self._node_executions: dict[str, int] = defaultdict(int)

        # Function name -> execution times (bounded queue)
        self._execution_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

        # Replicate function name -> total call count
        self._replicate_calls: dict[str, int] = defaultdict(int)

        # Lock for thread-safe updates
        self._lock = threading.Lock()

        # Start time
        self._start_time = time.time()

    def record_node_start(self, node_name: str) -> None:
        """Record that a node started executing."""
        with self._lock:
            self._node_executions[node_name] += 1

    def record_node_end(self, node_name: str, duration: float) -> None:
        """Record that a node finished executing.

        Args:
            node_name: Name of the node
            duration: Execution time in seconds
        """
        with self._lock:
            self._execution_times[node_name].append(duration)

    def record_replicate_call(
        self,
        replicate_name: str,
        caller_node: str | None,
        duration: float | None = None,
    ) -> None:
        """Record a replicated function call.

        Args:
            replicate_name: Name of the @replicate function
            caller_node: Name of the calling @jit function (None if external)
            duration: Optional execution time in seconds
        """
        with self._lock:
            self._replicate_calls[replicate_name] += 1

            if caller_node:
                # Track node -> replicate dependency
                self._node_to_replicates[caller_node].add(replicate_name)
                self._fanout_counts[caller_node][replicate_name] += 1

            if duration is not None:
                self._execution_times[replicate_name].append(duration)

    def get_dependency_graph(self) -> dict[str, set[str]]:
        """Get the @jit -> @replicate dependency graph.

        Returns:
            Dict mapping node names to the set of replicate functions they call
        """
        with self._lock:
            return {
                node: set(replicates)
                for node, replicates in self._node_to_replicates.items()
            }

    def get_fanout_matrix(self) -> dict[str, dict[str, float]]:
        """Get average fanout: calls from node N to replicate R per execution.

        Returns:
            Dict[node_name, Dict[replicate_name, avg_calls_per_execution]]
        """
        with self._lock:
            fanout = {}
            for node, replicate_counts in self._fanout_counts.items():
                node_execs = self._node_executions[node]
                if node_execs > 0:
                    fanout[node] = {
                        replicate: count / node_execs
                        for replicate, count in replicate_counts.items()
                    }
            return fanout

    def get_execution_stats(self) -> dict[str, dict[str, float]]:
        """Get execution time statistics for each function.

        Returns:
            Dict[function_name, Dict[stat_name, value]]
            Stats include: mean, p50, p95, p99 (in seconds), count
        """
        import math
        import statistics

        with self._lock:
            stats = {}
            for func_name, times in self._execution_times.items():
                if times:
                    sample = list(times)
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

                    stats[func_name] = {
                        "mean": float(mean),
                        "p50": float(percentile(50)),
                        "p95": float(percentile(95)),
                        "p99": float(percentile(99)),
                        "count": len(sorted_times),
                    }
            return stats

    def get_call_counts(self) -> dict[str, int]:
        """Get total call counts for each replicated function."""
        with self._lock:
            return dict(self._replicate_calls)

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
            self._node_to_replicates.clear()
            self._fanout_counts.clear()
            self._node_executions.clear()
            self._execution_times.clear()
            self._replicate_calls.clear()
            self._start_time = time.time()

    def summary(self) -> dict[str, Any]:
        """Get a complete summary of profiling data."""
        return {
            "dependency_graph": self.get_dependency_graph(),
            "fanout_matrix": self.get_fanout_matrix(),
            "execution_stats": self.get_execution_stats(),
            "call_counts": self.get_call_counts(),
            "node_executions": self.get_node_executions(),
            "elapsed_seconds": self.get_elapsed_time(),
        }


# Profiler control


def enable_profiling(max_samples: int = 10_000) -> ExecutionProfiler:
    """Enable transparent execution profiling.

    Returns:
        The active profiler instance

    Example:
        profiler = ta.enable_profiling()
        # ... run your workload ...
        stats = profiler.summary()
        print(stats['fanout_matrix'])
    """
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            _profiler = ExecutionProfiler(max_samples=max_samples)
        return _profiler


def disable_profiling() -> None:
    """Disable execution profiling."""
    global _profiler
    with _profiler_lock:
        _profiler = None


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled."""
    return _profiler is not None


def get_profiler() -> ExecutionProfiler | None:
    """Get the active profiler instance, if any."""
    return _profiler


@contextmanager
def profile(max_samples: int = 10_000):
    """Context manager for scoped profiling.

    Example:
        with ta.profile() as profiler:
            result = await pipeline(data)
            print(profiler.summary())
    """
    profiler = enable_profiling(max_samples=max_samples)
    try:
        yield profiler
    finally:
        disable_profiling()


# Internal tracking helpers


def get_current_node() -> str | None:
    """Get the currently executing node name, if any."""
    return _current_node.get()


@contextmanager
def node_context(node_name: str):
    """Context manager for node execution tracking (internal use).

    This is used by the executor to track which node is currently running.
    """
    profiler = _profiler
    if profiler is None:
        yield
        return

    token = _current_node.set(node_name)
    profiler.record_node_start(node_name)
    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        profiler.record_node_end(node_name, duration)
        _current_node.reset(token)


def record_replicate_call(replicate_name: str, duration: float | None = None) -> None:
    """Record a replicated function call (internal use).

    Args:
        replicate_name: Name of the replicated function
        duration: Optional execution time in seconds
    """
    profiler = _profiler
    if profiler is not None:
        profiler.record_replicate_call(replicate_name, get_current_node(), duration)
