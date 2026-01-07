"""Optimizer for computing optimal replica counts based on profiling data.

This module uses integer programming (Gurobi) to optimize replica counts with
support for different scaling behaviors (batching, linear, sub-linear, etc.):

1. Compute arrival rates for each replica based on dependency graphs
2. Model capacity with custom throughput functions
3. Solve for optimal replica counts minimizing cost while meeting SLAs
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# Throughput function type: (num_replicas, arrival_rate) -> max_throughput
ThroughputFunc = Callable[[int, float], float]


def _require_gurobi() -> None:
    if not GUROBI_AVAILABLE:
        raise ImportError(
            "gurobipy is required for optimization. "
            "Install it with: pip install gurobipy"
        )


def linear_throughput(service_rate: float) -> ThroughputFunc:
    """Linear scaling: throughput = n × service_rate.

    Standard case for stateless services.

    Args:
        service_rate: Requests/sec per instance

    Example:
        VectorDB: each instance handles 200 req/s independently
        → linear_throughput(200.0)
    """

    def func(n: int, _arrival_rate: float) -> float:
        return n * service_rate

    return func


def batched_throughput(
    base_rate: float, batch_size: int, max_batch_efficiency: float = 1.0
) -> ThroughputFunc:
    """Batching scaling: throughput scales sub-linearly due to batching effects.

    Models systems like vLLM where:
    - Single instance can batch multiple requests
    - Adding replicas reduces per-instance batching efficiency

    Formula:
        effective_batch_per_instance = batch_size / sqrt(n)
        throughput = n × base_rate × (effective_batch / batch_size)

    Args:
        base_rate: Base throughput per instance with full batching (req/s)
        batch_size: Maximum batch size per instance
        max_batch_efficiency: Efficiency at maximum batching (default 1.0)

    Example:
        vLLM: Single instance can batch 32 requests, achieving 50 req/s
        → batched_throughput(base_rate=50.0, batch_size=32)
    """

    def func(n: int, arrival_rate: float) -> float:
        if n == 0:
            return 0.0

        # Estimate requests per instance
        per_instance_rate = arrival_rate / n

        # Effective batch size decreases as we split load
        # (fewer requests per instance → less batching opportunity)
        effective_batch = min(per_instance_rate, batch_size)

        # Batching efficiency: how much speedup we get from batching
        batch_efficiency = (effective_batch / batch_size) * max_batch_efficiency

        # Total throughput
        return n * base_rate * max(0.1, batch_efficiency)

    return func


def contention_throughput(
    base_rate: float, contention_factor: float = 0.9
) -> ThroughputFunc:
    """Sub-linear scaling due to resource contention.

    Models systems with shared resources (DB connections, locks, etc.)

    Formula:
        throughput = n × base_rate × (contention_factor ^ (n-1))

    Args:
        base_rate: Base throughput per instance (req/s)
        contention_factor: Efficiency multiplier per additional instance
                          (< 1.0 means contention, default 0.9)

    Example:
        Database with connection pooling:
        - 1 instance: 100 req/s
        - 2 instances: 180 req/s (not 200)
        - 3 instances: 243 req/s (not 300)
        → contention_throughput(100.0, contention_factor=0.9)
    """

    def func(n: int, _arrival_rate: float) -> float:
        if n == 0:
            return 0.0
        return n * base_rate * (contention_factor ** (n - 1))

    return func


def custom_throughput(func: Callable[[int], float]) -> ThroughputFunc:
    """Custom throughput function.

    Args:
        func: Function taking number of replicas, returning max throughput

    Example:
        # Measured empirically
        def my_throughput(n):
            if n == 1: return 50
            if n == 2: return 90
            if n == 3: return 120
            return 120 + (n - 3) * 25  # Linear after 3

        → custom_throughput(my_throughput)
    """

    def wrapper(n: int, _arrival_rate: float) -> float:
        return func(n)

    return wrapper


@dataclass
class ReplicaConfig:
    """Configuration for a single replica class."""

    name: str
    throughput_func: ThroughputFunc  # Function: (n_replicas, λ) → max_throughput

    # Resource requirements per instance (from @replica decorator)
    cpus_per_instance: int = 0  # CPUs per instance (0 for GPU-only)
    gpus_per_instance: int = 0  # GPUs per instance (0 for CPU-only)

    # Instance count bounds
    min_instances: int = 1
    max_instances: int = 100


@dataclass
class WorkloadProfile:
    """Workload profile from profiling data."""

    # External arrival rate (requests/sec entering the system at each node)
    external_arrivals: dict[str, float]

    # Fanout matrix: node -> replica -> avg calls per execution
    # From profiler.get_fanout_matrix()
    fanout_matrix: dict[str, dict[str, float]]


@dataclass
class Constraints:
    """Optimization constraints."""

    max_utilization: float = 0.8  # Max utilization per replica (default 80%)

    # Cluster resource limits
    max_total_cpus: int | None = None  # Total CPU budget (optional)
    max_total_gpus: int | None = None  # Total GPU budget (optional)
    max_total_instances: int | None = None  # Max total instances (optional)

    # Per-replica overrides
    per_replica_constraints: dict[str, dict] = field(
        default_factory=dict
    )  # Per-replica overrides (e.g., max_utilization, min_instances, max_instances)


@dataclass
class OptimizationResult:
    """Result of replica count optimization."""

    replica_counts: dict[str, int]  # Recommended replica counts
    arrival_rates: dict[str, float]  # Computed arrival rate per replica
    capacities: dict[str, float]  # Computed capacity per replica (req/s)
    utilizations: dict[str, float]  # Expected utilization per replica

    # Resource usage
    total_cpus: int  # Total CPUs allocated
    total_gpus: int  # Total GPUs allocated
    total_instances: int  # Total number of instances

    # Optimization metadata
    metrics: dict[str, Any]  # Additional metrics
    solver_status: str  # Gurobi solver status
    objective_value: float  # Objective function value


class DeploymentPlan:
    """Actionable deployment plan from optimization.

    Provides a clean interface to deploy optimized replica configurations
    and create auto-scalers from optimization results.

    Example:
        plan = optimize(target_rps=100.0, max_cpus=64)
        print(plan)  # Pretty-print the plan

        await plan.deploy()  # Deploy all replicas

        # Or deploy selectively
        await plan.deploy_only(['VectorDB', 'LLM'])

        # Create auto-scaler from plan
        scaler = await plan.autoscale(reoptimize_interval=60.0)
    """

    def __init__(
        self,
        result: OptimizationResult,
        target_rps: float,
        constraints: Constraints | None = None,
    ):
        """Initialize deployment plan.

        Args:
            result: Optimization result
            target_rps: Target requests per second
            constraints: Constraints used for optimization
        """
        self.result = result
        self.target_rps = target_rps
        self.constraints = constraints or Constraints()

    @property
    def replica_counts(self) -> dict[str, int]:
        """Get recommended replica counts."""
        return self.result.replica_counts

    @property
    def total_cpus(self) -> int:
        """Get total CPUs required."""
        return self.result.total_cpus

    @property
    def total_gpus(self) -> int:
        """Get total GPUs required."""
        return self.result.total_gpus

    @property
    def is_optimal(self) -> bool:
        """Check if optimization found optimal solution."""
        return self.result.solver_status == "OPTIMAL"

    async def deploy(self) -> None:
        """Deploy all replicas with optimal instance counts (async).

        Raises:
            RuntimeError: If optimization was not successful
        """
        if not self.is_optimal:
            raise RuntimeError(
                f"Cannot deploy non-optimal plan. Solver status: {self.result.solver_status}"
            )

        from .manager import get_default_manager

        manager = get_default_manager()

        # Deploy each replica with recommended count
        for replica_name, count in self.result.replica_counts.items():
            spec = manager.get_spec(replica_name)
            if spec is None:
                raise RuntimeError(f"Replica '{replica_name}' not found in registry")

            await spec.deploy(instances=count)

    def deploy_sync(self) -> None:
        """Deploy all replicas with optimal instance counts (sync).

        Raises:
            RuntimeError: If optimization was not successful
        """
        import asyncio

        asyncio.run(self.deploy())

    async def deploy_only(self, replica_names: list[str]) -> None:
        """Deploy only specified replicas (async).

        Args:
            replica_names: List of replica names to deploy
        """
        if not self.is_optimal:
            raise RuntimeError(
                f"Cannot deploy non-optimal plan. Solver status: {self.result.solver_status}"
            )

        from .manager import get_default_manager

        manager = get_default_manager()

        for replica_name in replica_names:
            if replica_name not in self.result.replica_counts:
                raise ValueError(f"Replica '{replica_name}' not in optimization result")

            spec = manager.get_spec(replica_name)
            if spec is None:
                raise RuntimeError(f"Replica '{replica_name}' not found in registry")

            count = self.result.replica_counts[replica_name]
            await spec.deploy(instances=count)

    async def autoscale(
        self,
        reoptimize_interval: float = 60.0,
        cooldown_period: float = 30.0,
    ):
        """Create and start an AutoScaler based on this plan.

        Args:
            reoptimize_interval: Seconds between re-optimization runs
            cooldown_period: Minimum seconds between scaling actions

        Returns:
            Started AutoScaler instance
        """
        from .autoscaling import AutoScaler

        # Deploy with current plan first
        await self.deploy()

        # Create auto-scaler that will re-optimize periodically
        scaler = AutoScaler(
            target_rps=self.target_rps,
            check_interval=reoptimize_interval,
            constraints=self.constraints,
            cooldown_period=cooldown_period,
        )

        await scaler.start()
        return scaler

    def __repr__(self) -> str:
        """Pretty-print the deployment plan."""
        lines = []
        lines.append("=" * 70)
        lines.append("DEPLOYMENT PLAN")
        lines.append("=" * 70)
        lines.append(f"Target RPS: {self.target_rps}")
        lines.append(f"Status: {self.result.solver_status}")
        lines.append("")
        lines.append("Resources:")
        lines.append(f"  Total CPUs: {self.result.total_cpus}")
        lines.append(f"  Total GPUs: {self.result.total_gpus}")
        lines.append(f"  Total Instances: {self.result.total_instances}")
        lines.append("")
        lines.append("Replicas:")

        for name in sorted(self.result.replica_counts.keys()):
            count = self.result.replica_counts[name]
            util = self.result.utilizations.get(name, 0.0)
            arrival = self.result.arrival_rates.get(name, 0.0)
            capacity = self.result.capacities.get(name, 0.0)

            lines.append(f"  {name}:")
            lines.append(f"    Instances: {count}")
            lines.append(f"    Utilization: {util:.1%}")
            lines.append(f"    Arrival Rate: {arrival:.1f} req/s")
            lines.append(f"    Capacity: {capacity:.1f} req/s")

        lines.append("")
        lines.append(
            f"Avg Utilization: {self.result.metrics.get('avg_utilization', 0.0):.1%}"
        )
        lines.append("=" * 70)

        return "\n".join(lines)


class ReplicaOptimizer:
    """Optimizer using Gurobi for computing optimal replica counts."""

    def __init__(
        self,
        configs: list[ReplicaConfig],
        constraints: Constraints | None = None,
    ):
        """Initialize optimizer.

        Args:
            configs: Configuration for each replica class
            constraints: Optimization constraints

        Raises:
            ImportError: If gurobipy is not installed
        """
        _require_gurobi()

        self.configs = {c.name: c for c in configs}
        self.constraints = constraints or Constraints()

    def compute_arrival_rates(self, workload: WorkloadProfile) -> dict[str, float]:
        """Compute arrival rate for each replica given workload.

        Uses the dependency graph to solve for steady-state arrival rates:
            λ_replica = Σ_node (external_λ_node × fanout_node→replica)

        Args:
            workload: Workload profile with fanout matrix

        Returns:
            Dict mapping replica name to arrival rate (req/s)
        """
        return _compute_arrival_rates(
            workload.fanout_matrix,
            workload.external_arrivals,
        )

    def optimize(self, workload: WorkloadProfile) -> OptimizationResult:
        """Compute optimal replica counts using integer programming.

        Maximizes throughput while minimizing resource usage.
        For batching and non-linear scaling, we use piecewise linear approximation.

        Formulation:
            Variables:
                n_i ∈ ℤ: number of replicas for service i

            Objective:
                minimize: Σ_i n_i  (minimize total instances)

            Constraints:
                1. Capacity: λ_i ≤ throughput_func_i(n_i, λ_i) × max_utilization
                2. Min instances: n_i ≥ min_instances_i
                3. Max instances: n_i ≤ max_instances_i
                4. CPU budget: Σ_i (cpus_i × n_i) ≤ max_total_cpus (optional)
                5. GPU budget: Σ_i (gpus_i × n_i) ≤ max_total_gpus (optional)
                6. Max instances: Σ_i n_i ≤ max_total_instances (optional)

        Args:
            workload: Workload profile from profiling

        Returns:
            OptimizationResult with optimal counts and resource allocation
        """
        # Step 1: Compute arrival rates
        arrival_rates = self.compute_arrival_rates(workload)

        # Step 2: Create Gurobi model
        model = gp.Model("replica_optimization")
        model.setParam("OutputFlag", 0)  # Suppress solver output

        # Step 3: Create decision variables (number of replicas)
        replica_vars = {}
        for replica_name in arrival_rates.keys():
            config = self.configs.get(replica_name)
            if not config:
                raise ValueError(
                    f"No config provided for replica '{replica_name}'. "
                    f"Please provide a ReplicaConfig with throughput_func."
                )

            # Integer variable for number of replicas
            replica_vars[replica_name] = model.addVar(
                vtype=GRB.INTEGER,
                lb=config.min_instances,
                ub=config.max_instances,
                name=f"n_{replica_name}",
            )

        # Step 4: Set objective (minimize total instances to maximize efficiency)
        objective = gp.quicksum(replica_vars[name] for name in replica_vars)
        model.setObjective(objective, GRB.MINIMIZE)

        # Step 5: Add capacity constraints using piecewise linear approximation

        for replica_name, arrival_rate in arrival_rates.items():
            config = self.configs[replica_name]
            max_util = self.constraints.max_utilization

            # Per-replica override
            per_replica = self.constraints.per_replica_constraints.get(replica_name, {})
            if "max_utilization" in per_replica:
                max_util = per_replica["max_utilization"]

            # Precompute throughput for each possible replica count
            # Gurobi doesn't support arbitrary functions, so we use
            # a big-M constraint approach with binary indicators

            n_var = replica_vars[replica_name]
            min_n = config.min_instances
            max_n = config.max_instances

            # For each possible value of n, add constraint:
            # If n_var == n, then throughput_func(n, λ) × max_util ≥ λ

            # Create binary indicator variables for each possible n
            indicator_vars = []
            for n in range(min_n, max_n + 1):
                indicator = model.addVar(
                    vtype=GRB.BINARY, name=f"is_{replica_name}_n{n}"
                )
                indicator_vars.append((n, indicator))

                # Link indicator to n_var: indicator == 1 iff n_var == n
                # We'll use SOS1 constraint below

            # SOS1: Exactly one indicator can be 1
            model.addSOS(
                GRB.SOS_TYPE1,
                [ind for _, ind in indicator_vars],
                list(range(len(indicator_vars))),
            )

            # Link indicators to n_var: n_var = sum(n × indicator_n)
            model.addConstr(
                n_var == gp.quicksum(n * ind for n, ind in indicator_vars),
                name=f"link_{replica_name}",
            )

            # Add capacity constraint for each n
            for n, indicator in indicator_vars:
                max_throughput = config.throughput_func(n, arrival_rate)
                required_capacity = arrival_rate

                # Big-M constraint: if indicator == 1, enforce capacity
                # required_capacity ≤ max_throughput × max_util + M × (1 - indicator)
                M = 1e6  # Large constant
                model.addConstr(
                    required_capacity
                    <= max_throughput * max_util + M * (1 - indicator),
                    name=f"capacity_{replica_name}_n{n}",
                )

        # CPU resource constraint (optional)
        if self.constraints.max_total_cpus is not None:
            total_cpus = gp.quicksum(
                self.configs[name].cpus_per_instance * replica_vars[name]
                for name in replica_vars
            )
            model.addConstr(
                total_cpus <= self.constraints.max_total_cpus,
                name="max_total_cpus",
            )

        # GPU resource constraint (optional)
        if self.constraints.max_total_gpus is not None:
            total_gpus = gp.quicksum(
                self.configs[name].gpus_per_instance * replica_vars[name]
                for name in replica_vars
            )
            model.addConstr(
                total_gpus <= self.constraints.max_total_gpus,
                name="max_total_gpus",
            )

        # Max total instances constraint (optional)
        if self.constraints.max_total_instances is not None:
            total_instances = gp.quicksum(replica_vars[name] for name in replica_vars)
            model.addConstr(
                total_instances <= self.constraints.max_total_instances,
                name="max_total_instances",
            )

        # Step 6: Solve
        model.optimize()

        # Step 7: Extract solution
        if model.status == GRB.OPTIMAL:
            replica_counts = {
                name: int(round(var.X)) for name, var in replica_vars.items()
            }

            # Compute capacities and utilizations
            capacities = {}
            utilizations = {}
            for name, count in replica_counts.items():
                config = self.configs[name]
                max_throughput = config.throughput_func(count, arrival_rates[name])
                capacities[name] = max_throughput
                utilizations[name] = (
                    arrival_rates[name] / max_throughput if max_throughput > 0 else 0.0
                )

            # Compute resource totals
            total_cpus = sum(
                replica_counts[name] * self.configs[name].cpus_per_instance
                for name in replica_counts
            )
            total_gpus = sum(
                replica_counts[name] * self.configs[name].gpus_per_instance
                for name in replica_counts
            )
            total_instances = sum(replica_counts.values())

            util_values = list(utilizations.values())
            metrics = {
                "num_replicas": len(replica_counts),
                "avg_utilization": sum(util_values) / len(util_values)
                if util_values
                else 0.0,
                "max_utilization": max(util_values) if util_values else 0.0,
                "min_utilization": min(util_values) if util_values else 0.0,
            }

            return OptimizationResult(
                replica_counts=replica_counts,
                arrival_rates=arrival_rates,
                capacities=capacities,
                utilizations=utilizations,
                total_cpus=total_cpus,
                total_gpus=total_gpus,
                total_instances=total_instances,
                metrics=metrics,
                solver_status="OPTIMAL",
                objective_value=model.objVal,
            )
        else:
            # Solver failed
            return OptimizationResult(
                replica_counts={},
                arrival_rates=arrival_rates,
                capacities={},
                utilizations={},
                total_cpus=0,
                total_gpus=0,
                total_instances=0,
                metrics={},
                solver_status=f"FAILED (status={model.status})",
                objective_value=float("inf"),
            )


def _compute_arrival_rates(
    fanout_matrix: dict[str, dict[str, float]],
    external_arrivals: dict[str, float],
) -> dict[str, float]:
    """Compute arrival rate for each replica from fanout matrix."""
    arrival_rates: dict[str, float] = defaultdict(float)
    for node, replicas in fanout_matrix.items():
        external_rate = external_arrivals.get(node, 0.0)
        for replica, fanout in replicas.items():
            arrival_rates[replica] += external_rate * fanout
    return dict(arrival_rates)


def _compute_external_arrivals(
    entry_nodes: set[str],
    external_rps: float,
) -> dict[str, float]:
    if not entry_nodes:
        return {}
    per_node = external_rps / len(entry_nodes)
    return {node: per_node for node in entry_nodes}


def compute_optimal_counts_from_profiler(
    profiler,
    external_rps: float,
    replicas: list | None = None,
    constraints: Constraints | None = None,
) -> OptimizationResult:
    """Helper function to compute optimal counts directly from profiler.

    Args:
        profiler: ReplicaProfiler instance with profiling data
        external_rps: External requests per second entering the system
        replicas: Optional list of ReplicaSpec. If None, auto-discovers from registry.
        constraints: Optimization constraints

    Returns:
        OptimizationResult with recommended counts

    Example:
        from thinkagain.distributed.profiling import profile, node_context
        from thinkagain.distributed.optimizer import (
            compute_optimal_counts_from_profiler,
            Constraints,
            linear_throughput,
            batched_throughput,
        )
        from thinkagain.distributed import replica

        @replica(cpus=2, throughput=linear_throughput(200.0))
        class VectorDB:
            ...

        @replica(gpus=1, throughput=batched_throughput(50.0, 32))
        class LLMPool:
            ...

        with profile() as profiler:
            # Run workload
            for _ in range(100):
                run(pipeline, data, context_factory=node_context)

            # Compute optimal counts (auto-discovers replicas)
            result = compute_optimal_counts_from_profiler(
                profiler,
                external_rps=100.0,
                constraints=Constraints(max_utilization=0.8),
            )

            print(f"Recommended counts: {result.replica_counts}")
            print(f"Utilizations: {result.utilizations}")
    """
    _require_gurobi()

    if replicas is None:
        # Auto-discover replicas from registry
        from .manager import get_default_manager

        manager = get_default_manager()
        all_specs = manager.get_all()
        replicas = [spec for spec in all_specs.values() if spec.throughput is not None]

        if not replicas:
            raise ValueError(
                "No replicas with throughput found. "
                "Add throughput to @replica decorators."
            )

    # Convert replicas to configs
    replica_configs = [spec.to_replica_config() for spec in replicas]

    fanout_matrix = profiler.get_fanout_matrix()
    entry_nodes = set(profiler.get_node_executions().keys())
    external_arrivals = _compute_external_arrivals(entry_nodes, external_rps)

    workload = WorkloadProfile(
        external_arrivals=external_arrivals,
        fanout_matrix=fanout_matrix,
    )
    optimizer = ReplicaOptimizer(replica_configs, constraints=constraints)
    return optimizer.optimize(workload)


# Fallback: Simple heuristic optimizer if Gurobi is not available
def compute_optimal_counts_heuristic(
    profiler,
    external_rps: float,
    service_rates: dict[str, float],
    target_utilization: float = 0.7,
) -> dict[str, int]:
    """Simple heuristic optimizer when Gurobi is not available.

    Assumes linear scaling: n_i = ceil(λ_i / (μ_i × target_utilization))

    Args:
        profiler: ReplicaProfiler instance
        external_rps: External requests per second
        service_rates: Service rates per instance (assumes linear scaling)
        target_utilization: Target utilization (default 0.7)

    Returns:
        Dict mapping replica name to recommended count
    """
    fanout_matrix = profiler.get_fanout_matrix()
    entry_nodes = set(profiler.get_node_executions().keys())
    external_arrivals = _compute_external_arrivals(entry_nodes, external_rps)

    arrival_rates = _compute_arrival_rates(fanout_matrix, external_arrivals)

    replica_counts = {}
    for replica, arrival_rate in arrival_rates.items():
        service_rate = service_rates.get(replica, 10.0)
        count = math.ceil(arrival_rate / (service_rate * target_utilization))
        replica_counts[replica] = max(1, count)

    return replica_counts


# ============================================================================
# New Clean API
# ============================================================================


def optimize(
    target_rps: float,
    profiler=None,
    constraints: Constraints | None = None,
    max_cpus: int | None = None,
    max_gpus: int | None = None,
    max_utilization: float = 0.8,
) -> DeploymentPlan:
    """Compute optimal replica deployment plan.

    This is the main entry point for optimization. It discovers all replicas
    with throughput functions configured, runs the optimizer, and returns an
    actionable deployment plan.

    Args:
        target_rps: Target requests per second (external arrival rate)
        profiler: Optional profiler instance. If None, uses global profiler
                 or assumes uniform load distribution.
        constraints: Advanced constraints (optional). Use max_cpus/max_gpus for simple cases.
        max_cpus: Maximum total CPUs (shorthand for constraints.max_total_cpus)
        max_gpus: Maximum total GPUs (shorthand for constraints.max_total_gpus)
        max_utilization: Target max utilization (default: 0.8)

    Returns:
        DeploymentPlan that can be deployed or inspected

    Example:
        # Simplest usage
        plan = optimize(target_rps=100.0, max_cpus=64, max_gpus=4)
        await plan.deploy()

        # With profiling
        with profile():
            run_workload()

        plan = optimize(target_rps=100.0)
        print(plan)  # Inspect before deploying

        # Advanced constraints
        plan = optimize(
            target_rps=100.0,
            constraints=Constraints(
                max_total_cpus=64,
                max_total_gpus=4,
                max_utilization=0.75,
                per_replica_constraints={
                    'LLM': {'max_instances': 2},
                }
            )
        )

    Raises:
        ImportError: If gurobipy is not installed
        ValueError: If no replicas with throughput configured are found
    """
    _require_gurobi()

    # Build constraints
    if constraints is None:
        constraints = Constraints(max_utilization=max_utilization)

    # Apply shorthand parameters
    if max_cpus is not None:
        constraints.max_total_cpus = max_cpus
    if max_gpus is not None:
        constraints.max_total_gpus = max_gpus

    # Auto-discover replicas with throughput configured
    from .manager import get_default_manager

    manager = get_default_manager()
    all_specs = manager.get_all()
    replicas = [spec for spec in all_specs.values() if spec.throughput is not None]

    if not replicas:
        raise ValueError(
            "No replicas with throughput configured found. "
            "Add throughput parameter to @replica decorators."
        )

    # Convert to ReplicaConfig
    replica_configs = [spec.to_replica_config() for spec in replicas]

    # Get or create profiler
    if profiler is None:
        from .profiling import get_profiler

        profiler = get_profiler()

    # Build workload profile
    if profiler is not None:
        fanout_matrix = profiler.get_fanout_matrix()
        node_executions = profiler.get_node_executions()

        if not fanout_matrix or not node_executions:
            # No profiling data, use uniform distribution
            fanout_matrix = {"default_node": {spec.name: 1.0 for spec in replicas}}
            external_arrivals = {"default_node": target_rps}
        else:
            # Use profiling data
            entry_nodes = set(node_executions.keys())
            external_arrivals = _compute_external_arrivals(entry_nodes, target_rps)
    else:
        # No profiler, assume uniform distribution
        fanout_matrix = {"default_node": {spec.name: 1.0 for spec in replicas}}
        external_arrivals = {"default_node": target_rps}

    workload = WorkloadProfile(
        external_arrivals=external_arrivals,
        fanout_matrix=fanout_matrix,
    )

    # Run optimizer
    optimizer = ReplicaOptimizer(configs=replica_configs, constraints=constraints)
    result = optimizer.optimize(workload)

    # Return deployment plan
    return DeploymentPlan(
        result=result,
        target_rps=target_rps,
        constraints=constraints,
    )


def compare_scenarios(
    target_rps: float,
    scenarios: dict[str, Constraints],
    profiler=None,
) -> dict[str, DeploymentPlan]:
    """Compare multiple deployment scenarios.

    Args:
        target_rps: Target requests per second
        scenarios: Dict mapping scenario name to constraints
        profiler: Optional profiler instance

    Returns:
        Dict mapping scenario name to deployment plan

    Example:
        scenarios = {
            "Budget": Constraints(max_total_cpus=32, max_total_gpus=2),
            "Performance": Constraints(max_utilization=0.6),
            "Balanced": Constraints(max_total_cpus=64, max_utilization=0.75),
        }

        plans = compare_scenarios(target_rps=100.0, scenarios=scenarios)

        for name, plan in plans.items():
            print(f"\n{name}:")
            print(f"  CPUs: {plan.total_cpus}, GPUs: {plan.total_gpus}")
            print(f"  Instances: {plan.result.total_instances}")
    """
    results = {}
    for name, constraints in scenarios.items():
        try:
            plan = optimize(
                target_rps=target_rps,
                profiler=profiler,
                constraints=constraints,
            )
            results[name] = plan
        except Exception as e:
            print(f"Warning: Scenario '{name}' failed: {e}")

    return results
