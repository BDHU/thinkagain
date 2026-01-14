"""Transparent profiling for distributed execution."""

from __future__ import annotations

import contextvars
import threading
import time
import math
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any

from .context import get_current_execution_context

_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_node", default=None
)

_profiler_lock = threading.Lock()


class ExecutionProfiler:
    def __init__(self, max_samples: int = 10_000):
        self._node_to_services: dict[str, set[str]] = defaultdict(set)
        self._fanout_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._node_executions: dict[str, int] = defaultdict(int)
        self._execution_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._service_calls: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()

    def record_node_start(self, node_name: str) -> None:
        with self._lock:
            self._node_executions[node_name] += 1

    def record_node_end(self, node_name: str, duration: float) -> None:
        with self._lock:
            self._execution_times[node_name].append(duration)

    def record_service_call(
        self,
        service_name: str,
        caller_node: str | None,
        duration: float | None = None,
    ) -> None:
        with self._lock:
            self._service_calls[service_name] += 1
            if caller_node:
                self._node_to_services[caller_node].add(service_name)
                self._fanout_counts[caller_node][service_name] += 1
            if duration is not None:
                self._execution_times[service_name].append(duration)

    def get_dependency_graph(self) -> dict[str, set[str]]:
        with self._lock:
            return {
                node: set(services) for node, services in self._node_to_services.items()
            }

    def get_fanout_matrix(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                node: {r: c / self._node_executions[node] for r, c in counts.items()}
                for node, counts in self._fanout_counts.items()
                if self._node_executions[node] > 0
            }

    def get_execution_stats(self) -> dict[str, dict[str, float]]:
        import statistics

        with self._lock:
            stats = {}
            for func_name, times in self._execution_times.items():
                if not times:
                    continue
                sample = sorted(times)
                n = len(sample)

                def pct(p: float) -> float:
                    k = (n - 1) * (p / 100)
                    f, c = math.floor(k), math.ceil(k)
                    return (
                        sample[int(k)]
                        if f == c
                        else sample[f] + (sample[c] - sample[f]) * (k - f)
                    )

                stats[func_name] = {
                    "mean": statistics.fmean(sample),
                    "p50": pct(50),
                    "p95": pct(95),
                    "p99": pct(99),
                    "count": n,
                }
            return stats

    def get_call_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._service_calls)

    def get_node_executions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._node_executions)

    def get_elapsed_time(self) -> float:
        return time.perf_counter() - self._start_time

    def reset(self) -> None:
        with self._lock:
            self._node_to_services.clear()
            self._fanout_counts.clear()
            self._node_executions.clear()
            self._execution_times.clear()
            self._service_calls.clear()
            self._start_time = time.perf_counter()

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
    ctx = get_current_execution_context()
    with _profiler_lock:
        if ctx.profiler is None:
            ctx.profiler = ExecutionProfiler(max_samples=max_samples)
        return ctx.profiler


def disable_profiling() -> None:
    """Disable execution profiling."""
    ctx = get_current_execution_context()
    with _profiler_lock:
        ctx.profiler = None


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled."""
    ctx = get_current_execution_context()
    with _profiler_lock:
        return ctx.profiler is not None


def get_profiler() -> ExecutionProfiler | None:
    """Get the active profiler instance, if any."""
    ctx = get_current_execution_context()
    with _profiler_lock:
        return ctx.profiler


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
    profiler = get_current_execution_context().profiler
    if profiler is None:
        yield
        return

    token = _current_node.set(node_name)
    profiler.record_node_start(node_name)
    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        profiler.record_node_end(node_name, duration)
        _current_node.reset(token)


def record_service_call(service_name: str, duration: float | None = None) -> None:
    """Record a service function call (internal use).

    Args:
        service_name: Name of the service function
        duration: Optional execution time in seconds
    """
    profiler = get_current_execution_context().profiler
    if profiler is not None:
        profiler.record_service_call(service_name, get_current_node(), duration)
