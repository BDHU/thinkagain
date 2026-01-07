"""Auto-scaling for distributed replicas based on profiling and optimization.

This module provides automatic scaling of replica instances based on:
- Real-time profiling data (from ReplicaProfiler)
- Optimizer recommendations (from ReplicaOptimizer)
- Cluster resource constraints (from scheduler)

The AutoScaler runs as a background asyncio task in the same process,
periodically checking metrics and adjusting deployments.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .optimizer import (
    Constraints,
    ReplicaConfig,
    ReplicaOptimizer,
    WorkloadProfile,
)
from .profiling import get_profiler
from .replica import ReplicaSpec
from .runtime import get_backend
from .manager import get_default_manager, ReplicaManager

logger = logging.getLogger(__name__)


@dataclass
class ScalingEvent:
    """Record of a scaling decision."""

    timestamp: float
    replica_name: str
    old_count: int
    new_count: int
    reason: str
    utilization: float
    arrival_rate: float


@dataclass
class ScalingStatus:
    """Current status of a replica's scaling."""

    replica_name: str
    current_instances: int
    target_instances: int
    utilization: float
    arrival_rate: float
    capacity: float
    last_scaled_at: float | None


class AutoScaler:
    """Automatic scaler for replica instances.

    Monitors profiling data and adjusts replica counts to meet target
    utilization while respecting resource constraints.

    Example:
        from thinkagain.distributed import init, replica
        from thinkagain.distributed.autoscaling import AutoScaler
        from thinkagain.distributed.optimizer import linear_throughput

        init(backend="grpc")

        @replica(
            cpus=2,
            throughput_func=linear_throughput(200.0),
            min_instances=1,
            max_instances=10,
        )
        class VectorDB:
            def search(self, query): ...

        await VectorDB.deploy(instances=1)

        # Auto-scaler will automatically discover all replicas with throughput
        scaler = AutoScaler(
            target_rps=100.0,
            check_interval=30.0,
        )

        await scaler.start()  # Background monitoring starts
        # ... your application runs ...
        await scaler.stop()
    """

    def __init__(
        self,
        target_rps: float,
        replicas: list[ReplicaSpec] | None = None,
        check_interval: float = 30.0,
        constraints: Constraints | None = None,
        cooldown_period: float = 60.0,
        scale_up_threshold: float = 0.85,
        scale_down_threshold: float = 0.50,
        manager: ReplicaManager | None = None,
    ):
        """Initialize the AutoScaler.

        Args:
            target_rps: Target requests per second (external arrival rate)
            replicas: Optional list of ReplicaSpec to auto-scale. If None, auto-discovers
                     all registered replicas with throughput configured.
            check_interval: Seconds between scaling checks (default: 30s)
            constraints: Resource constraints for optimizer (optional)
            cooldown_period: Minimum seconds between scaling actions (default: 60s)
            scale_up_threshold: Trigger scale-up if utilization > this (default: 0.85)
            scale_down_threshold: Trigger scale-down if utilization < this (default: 0.50)
            manager: ReplicaManager to use (default: global manager)
        """
        self.target_rps = target_rps
        self.check_interval = check_interval
        self.constraints = constraints or Constraints()
        self.cooldown_period = cooldown_period
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        # Determine which replicas to manage
        if manager is None:
            manager = get_default_manager()
        self._manager = manager

        if replicas is not None:
            # Use provided replicas
            self._replicas = {spec.name: spec for spec in replicas}
        else:
            # Auto-discover: find all replicas with throughput configured
            self._replicas = {}
            all_specs = self._manager.get_all()
            for full_name, spec in all_specs.items():
                if spec.throughput is not None:
                    self._replicas[spec.name] = spec

            if not self._replicas:
                logger.warning(
                    "No replicas with throughput found. "
                    "Add throughput to @replica decorators or provide replicas list."
                )

        # Runtime state
        self._running = False
        self._task: asyncio.Task | None = None
        self._history: list[ScalingEvent] = []
        self._last_scaled: dict[str, float] = {}  # replica -> timestamp
        self._current_counts: dict[str, int] = {}  # replica -> instance count

        # Profiler integration
        self._profiler = None

    def _get_replica_names(self) -> list[str]:
        """Get list of replica names being managed."""
        return list(self._replicas.keys())

    def _get_replica_configs(self) -> list[ReplicaConfig]:
        """Convert replicas to ReplicaConfig list."""
        configs = []
        for spec in self._replicas.values():
            try:
                configs.append(spec.to_replica_config())
            except ValueError as e:
                logger.warning(f"Skipping replica {spec.name}: {e}")
        return configs

    async def start(self) -> None:
        """Start the background scaling loop."""
        if self._running:
            logger.warning("AutoScaler already running")
            return

        self._running = True
        self._profiler = get_profiler()

        if self._profiler is None:
            logger.warning(
                "No profiler found. AutoScaler will make decisions based on "
                "target_rps only. For better scaling, use profiling context."
            )

        logger.info(
            f"AutoScaler started: target_rps={self.target_rps}, "
            f"check_interval={self.check_interval}s, replicas={self._get_replica_names()}"
        )

        # Start background task
        self._task = asyncio.create_task(self._scaling_loop())

    async def stop(self) -> None:
        """Stop the background scaling loop."""
        if not self._running:
            return

        logger.info("Stopping AutoScaler...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("AutoScaler stopped")

    async def _scaling_loop(self) -> None:
        """Background task that periodically checks and scales replicas."""
        while self._running:
            try:
                await self._check_and_scale()
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}", exc_traceback=True)

            await asyncio.sleep(self.check_interval)

    async def _check_and_scale(self) -> None:
        """Check current state and make scaling decisions."""
        backend = get_backend()
        if backend is None:
            logger.warning("No backend initialized, skipping scaling check")
            return

        manager = self._manager

        # Get current deployment state
        for replica_name in self._get_replica_names():
            spec = manager.get_spec(replica_name)
            if spec is None:
                logger.warning(f"Replica '{replica_name}' not registered, skipping")
                continue

            if not backend.is_deployed(spec):
                logger.debug(f"Replica '{replica_name}' not deployed, skipping")
                continue

            # Get current instance count from backend
            current_count = backend.get_instance_count(spec)
            self._current_counts[replica_name] = current_count

        # Run optimizer to get recommended counts
        result = self._compute_optimal_counts()

        if result is None:
            logger.debug("Optimizer returned no result, skipping scaling")
            return

        if result.solver_status != "OPTIMAL":
            logger.warning(f"Optimizer failed: {result.solver_status}")
            return

        # Make scaling decisions
        for replica_name, target_count in result.replica_counts.items():
            await self._maybe_scale_replica(
                replica_name=replica_name,
                current_count=self._current_counts.get(replica_name, 0),
                target_count=target_count,
                utilization=result.utilizations.get(replica_name, 0.0),
                arrival_rate=result.arrival_rates.get(replica_name, 0.0),
            )

    def _compute_optimal_counts(self):
        """Compute optimal replica counts using optimizer."""
        from .optimizer import GUROBI_AVAILABLE

        if not GUROBI_AVAILABLE:
            logger.warning(
                "Gurobi not available, cannot run optimizer. "
                "Install with: pip install gurobipy"
            )
            return None

        # Build workload profile
        if self._profiler:
            # Use actual profiling data
            fanout_matrix = self._profiler.get_fanout_matrix()
            node_executions = self._profiler.get_node_executions()

            if not fanout_matrix or not node_executions:
                logger.debug("No profiling data yet, skipping optimization")
                return None

            # Distribute target RPS across entry nodes
            entry_nodes = set(node_executions.keys())
            rps_per_node = self.target_rps / len(entry_nodes) if entry_nodes else 0.0
            external_arrivals = {node: rps_per_node for node in entry_nodes}

            workload = WorkloadProfile(
                external_arrivals=external_arrivals,
                fanout_matrix=fanout_matrix,
            )
        else:
            # Fallback: assume uniform load distribution
            # Create simple fanout: each replica gets equal share
            fanout_matrix = {
                "default_node": {name: 1.0 for name in self._get_replica_names()}
            }
            external_arrivals = {"default_node": self.target_rps}
            workload = WorkloadProfile(
                external_arrivals=external_arrivals,
                fanout_matrix=fanout_matrix,
            )

        # Convert replicas/policies to replica configs
        replica_configs = self._get_replica_configs()

        # Run optimizer
        optimizer = ReplicaOptimizer(
            configs=replica_configs,
            constraints=self.constraints,
        )

        return optimizer.optimize(workload)

    async def _maybe_scale_replica(
        self,
        replica_name: str,
        current_count: int,
        target_count: int,
        utilization: float,
        arrival_rate: float,
    ) -> None:
        """Make scaling decision for a single replica."""
        # No change needed
        if current_count == target_count:
            logger.debug(
                f"{replica_name}: No scaling needed (current={current_count}, target={target_count})"
            )
            return

        # Check cooldown period
        last_scaled = self._last_scaled.get(replica_name, 0.0)
        time_since_last = time.time() - last_scaled
        if time_since_last < self.cooldown_period:
            logger.debug(
                f"{replica_name}: In cooldown period "
                f"({time_since_last:.1f}s < {self.cooldown_period}s)"
            )
            return

        # Determine scaling direction and reason
        if target_count > current_count:
            # Scale up
            if utilization < self.scale_up_threshold:
                logger.debug(
                    f"{replica_name}: Utilization too low for scale-up "
                    f"({utilization:.2%} < {self.scale_up_threshold:.2%})"
                )
                return
            reason = f"high_utilization ({utilization:.2%})"
            action = "scale up"
        else:
            # Scale down
            if utilization > self.scale_down_threshold:
                logger.debug(
                    f"{replica_name}: Utilization too high for scale-down "
                    f"({utilization:.2%} > {self.scale_down_threshold:.2%})"
                )
                return
            reason = f"low_utilization ({utilization:.2%})"
            action = "scale down"

        # Execute scaling
        logger.info(
            f"{replica_name}: {action} from {current_count} to {target_count} instances "
            f"(utilization={utilization:.2%}, arrival_rate={arrival_rate:.1f} req/s, reason={reason})"
        )

        try:
            manager = get_default_manager()
            spec = manager.get_spec(replica_name)

            # Re-deploy with new count
            # Note: deploy() handles incremental scaling automatically
            await spec.cls.deploy(instances=target_count)

            # Record event
            event = ScalingEvent(
                timestamp=time.time(),
                replica_name=replica_name,
                old_count=current_count,
                new_count=target_count,
                reason=reason,
                utilization=utilization,
                arrival_rate=arrival_rate,
            )
            self._history.append(event)
            self._last_scaled[replica_name] = time.time()
            self._current_counts[replica_name] = target_count

            logger.info(
                f"{replica_name}: Successfully scaled to {target_count} instances"
            )

        except Exception as e:
            logger.error(f"{replica_name}: Failed to scale: {e}", exc_info=True)

    def get_status(self) -> dict[str, ScalingStatus]:
        """Get current scaling status for all replicas.

        Returns:
            Dict mapping replica name to ScalingStatus
        """
        result = self._compute_optimal_counts()

        status = {}
        for replica_name in self._get_replica_names():
            current = self._current_counts.get(replica_name, 0)
            target = (
                result.replica_counts.get(replica_name, current) if result else current
            )
            utilization = result.utilizations.get(replica_name, 0.0) if result else 0.0
            arrival_rate = (
                result.arrival_rates.get(replica_name, 0.0) if result else 0.0
            )
            capacity = result.capacities.get(replica_name, 0.0) if result else 0.0

            status[replica_name] = ScalingStatus(
                replica_name=replica_name,
                current_instances=current,
                target_instances=target,
                utilization=utilization,
                arrival_rate=arrival_rate,
                capacity=capacity,
                last_scaled_at=self._last_scaled.get(replica_name),
            )

        return status

    def get_history(self, limit: int | None = None) -> list[ScalingEvent]:
        """Get scaling event history.

        Args:
            limit: Maximum number of events to return (most recent first)

        Returns:
            List of ScalingEvent objects
        """
        events = sorted(self._history, key=lambda e: e.timestamp, reverse=True)
        if limit:
            events = events[:limit]
        return events

    def get_report(self) -> dict:
        """Get comprehensive scaling report.

        Returns:
            Dict with summary statistics and history
        """
        status = self.get_status()
        history = self.get_history()

        total_instances = sum(s.current_instances for s in status.values())
        avg_utilization = (
            sum(s.utilization for s in status.values()) / len(status) if status else 0.0
        )

        return {
            "running": self._running,
            "target_rps": self.target_rps,
            "check_interval": self.check_interval,
            "total_instances": total_instances,
            "avg_utilization": avg_utilization,
            "num_replicas": len(status),
            "status": {name: s.__dict__ for name, s in status.items()},
            "total_scaling_events": len(self._history),
            "recent_events": [
                {
                    "time": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp)
                    ),
                    "replica": e.replica_name,
                    "from": e.old_count,
                    "to": e.new_count,
                    "reason": e.reason,
                    "utilization": e.utilization,
                }
                for e in history[:10]
            ],
        }
