"""Comprehensive guide to stateful replica patterns in ThinkAgain.

This example demonstrates all the patterns for working with stateful replicas,
including:

1. Basic replica definition with @trace + @replica
2. Pure state updates with apply_replica (immutable pattern)
3. Using replicas in @jit functions (via mesh)
4. Simple replicas vs replicas with runtime state in aux
5. Replicas with mutable loops
6. Advanced patterns with scan and control flow

The key insight: Graphs are pure and immutable, but replicas can be stateful
actors. The `@trace` decorator + `decompose/compose` pattern allows pure
functional updates while maintaining the object-oriented replica interface.

Quick Start:
    python examples/stateful_replica_patterns.py
"""

import asyncio
from dataclasses import dataclass, field

import thinkagain as ta


# =============================================================================
# State Definitions
# =============================================================================


@dataclass
class RAGState:
    """State for RAG examples."""

    query: str
    documents: list[str] = field(default_factory=list)
    answer: str = ""


# =============================================================================
# Example 1: Basic Stateful Replica (Simple Pattern)
# =============================================================================


@ta.replica()
class Counter:
    """Simplest stateful replica - just mutates state directly.

    This pattern is fine for:
    - Simple counters, caches, databases
    - When you don't need pure functional updates
    - When replica is used directly (not in complex graph operations)

    Note: State mutations happen inside the replica instance.
    """

    def __init__(self, initial: int = 0):
        """Initialize counter with starting value."""
        self.count = initial
        self.call_history = []

    async def __call__(self, increment: int = 1) -> int:
        """Increment counter and return new value."""
        self.count += increment
        self.call_history.append(self.count)
        return self.count


# =============================================================================
# Example 2: Replica with @trace for Pure Updates (Immutable Pattern)
# =============================================================================


@ta.trace
@ta.replica()
class PureCounter:
    """Replica with decompose/compose for pure functional updates.

    This pattern enables:
    - Pure state transformations (no mutations)
    - Using apply_replica for @jit-compatible updates
    - Nested graph values as state
    - Better composability with graph operations

    The @trace decorator automatically implements decompose/compose for dataclass-like
    classes. For custom classes, implement the protocol manually.
    """

    def __init__(self, count: int = 0, step: int = 1):
        """Initialize with count and step size.

        These become the "children" in decompose/compose.
        """
        self.count = count
        self.step = step

    def decompose(self) -> tuple[list, None]:
        """Break replica into traceable children and static aux data.

        Returns:
            (children, aux) where children are values that can be traced
            and aux is metadata that stays constant
        """
        return [self.count, self.step], None

    @classmethod
    def compose(cls, aux, children: list) -> "PureCounter":
        """Reconstruct replica from children and aux.

        Args:
            aux: Auxiliary data from decompose (unused here)
            children: List of [count, step] from decompose

        Returns:
            New replica instance with updated state
        """
        count, step = children
        return cls(count=count, step=step)

    async def __call__(self, x: int) -> int:
        """Pure computation: count + step + x (doesn't mutate)."""
        return self.count + self.step + x


# =============================================================================
# Example 3: LLM Replica with Runtime State in Aux
# =============================================================================


@ta.trace
@ta.replica(gpus=1)
class LLM:
    """Realistic LLM replica with heavy initialization and runtime state.

    This pattern shows:
    - Separating model state (static, in aux) from generation state (dynamic, in children)
    - Heavy initialization (model loading) happens once
    - Runtime state (temperature, cache) can be updated via pure functions
    - GPU resources specified via @replica decorator
    """

    def __init__(self, model_name: str, temperature: float = 0.7):
        """Initialize LLM.

        Args:
            model_name: Name of model to load (goes in aux - never changes)
            temperature: Sampling temperature (goes in children - can be updated)
        """
        self.model_name = model_name
        self.temperature = temperature
        # Simulate heavy model loading (happens once per instance)
        self.model = self._load_model(model_name)
        self.cache = {}

    def _load_model(self, model_name: str):
        """Simulate expensive model loading."""
        print(f"  [INIT] Loading {model_name} (expensive operation)...")
        return {"name": model_name, "params": "70B"}

    def decompose(self) -> tuple[list, dict]:
        """Decompose into runtime state (children) and static state (aux).

        Children: Values that can change via pure updates (temperature, cache)
        Aux: Static initialization data (model_name, loaded model object)
        """
        children = [self.temperature, self.cache]
        aux = {
            "model_name": self.model_name,
            "model": self.model,
        }
        return children, aux

    @classmethod
    def compose(cls, aux: dict, children: list) -> "LLM":
        """Reconstruct LLM with updated runtime state."""
        temperature, cache = children
        instance = cls.__new__(cls)
        instance.model_name = aux["model_name"]
        instance.model = aux["model"]
        instance.temperature = temperature
        instance.cache = cache
        return instance

    async def __call__(self, prompt: str) -> str:
        """Generate response (pure function of prompt + state)."""
        # Simulate generation
        await asyncio.sleep(0.01)

        # Check cache
        if prompt in self.cache:
            return self.cache[prompt]

        # Generate response
        response = f"[{self.model_name}@temp={self.temperature}]: Response to '{prompt[:30]}...'"
        return response


# =============================================================================
# Example 4: Database Replica with Query State
# =============================================================================


@ta.trace
@ta.replica()
class Database:
    """Database replica with connection pool and query history.

    This pattern demonstrates:
    - Persistent connection (in aux, never changes)
    - Query history (in children, grows over time)
    - Pure functional updates for tracking queries
    """

    def __init__(self, connection_string: str):
        """Initialize database connection.

        Args:
            connection_string: Database URL (static, goes in aux)
        """
        self.connection_string = connection_string
        self.connection = self._connect(connection_string)
        self.query_history = []

    def _connect(self, conn_str: str):
        """Simulate database connection."""
        print(f"  [INIT] Connecting to database: {conn_str}")
        return {"connected": True, "url": conn_str}

    def decompose(self) -> tuple[list, dict]:
        """Decompose into query history (children) and connection (aux)."""
        children = [self.query_history]
        aux = {
            "connection_string": self.connection_string,
            "connection": self.connection,
        }
        return children, aux

    @classmethod
    def compose(cls, aux: dict, children: list) -> "Database":
        """Reconstruct database with updated query history."""
        (query_history,) = children
        instance = cls.__new__(cls)
        instance.connection_string = aux["connection_string"]
        instance.connection = aux["connection"]
        instance.query_history = query_history
        return instance

    async def __call__(self, query: str) -> list[dict]:
        """Execute query and return results."""
        await asyncio.sleep(0.01)  # Simulate query
        results = [
            {"id": 1, "data": f"Result for: {query}"},
            {"id": 2, "data": f"Another result for: {query}"},
        ]
        return results


# =============================================================================
# Example 5: Using apply_replica for Pure State Updates
# =============================================================================


@ta.node
async def increment_step(count: int, step: int, amount: int) -> tuple[list, int]:
    """Pure function to update counter state.

    Args:
        count: Current count (from decompose children[0])
        step: Current step (from decompose children[1])
        amount: Value to add to count

    Returns:
        (new_children, output) where:
        - new_children = [new_count, new_step]
        - output = result to return
    """
    new_count = count + step + amount
    new_step = step
    return [new_count, new_step], new_count


@ta.node
async def update_temperature(
    temp: float, cache: dict, new_temp: float
) -> tuple[list, str]:
    """Pure function to update LLM temperature.

    Args:
        temp: Current temperature (from decompose children[0])
        cache: Current cache (from decompose children[1])
        new_temp: New temperature value

    Returns:
        (new_children, output)
    """
    return [new_temp, cache], f"Temperature updated to {new_temp}"


@ta.node
async def add_query_to_history(query_history: list, query: str) -> tuple[list, list]:
    """Pure function to add query to history.

    Args:
        query_history: Current history (from decompose children[0])
        query: New query to add

    Returns:
        (new_children, output)
    """
    new_history = query_history + [query]
    return [new_history], new_history


# =============================================================================
# Example 6: Using Replicas in @jit Functions
# =============================================================================


# Define replica handles (must be created outside @jit)
simple_counter = Counter.init(0)
pure_counter = PureCounter.init(count=0, step=5)
llm = LLM.init("llama-70b", temperature=0.7)
database = Database.init("postgresql://localhost/mydb")


# =============================================================================
# Pipelines (Graph Composition with @jit)
# =============================================================================


# Node functions that call replicas (replicas must be called from @node, not directly in @jit)
@ta.bind_service(simple_counter=simple_counter)
@ta.node
async def call_simple_counter(x: int) -> int:
    """Call simple counter replica."""
    return await simple_counter(x)


@ta.bind_service(pure_counter=pure_counter)
@ta.node
async def call_pure_counter(x: int) -> int:
    """Call pure counter replica."""
    return await pure_counter(x)


@ta.bind_service(llm=llm)
@ta.node
async def call_llm(prompt: str) -> str:
    """Call LLM replica."""
    return await llm(prompt)


@ta.bind_service(database=database)
@ta.node
async def call_database(query: str) -> list[dict]:
    """Call database replica."""
    return await database(query)


@ta.jit
async def simple_pipeline(x: int) -> int:
    """Simple pipeline using replica via @node.

    Replicas must be called from @node functions, not directly in @jit.
    State mutations happen inside the replica.
    """
    result = await call_simple_counter(x)
    return result


@ta.jit
async def pure_pipeline(x: int) -> int:
    """Pipeline with pure replica computation.

    The replica's __call__ is pure (doesn't mutate state).
    """
    result = await call_pure_counter(x)
    return result


@ta.jit
async def llm_pipeline(prompt: str) -> str:
    """Pipeline using LLM replica."""
    response = await call_llm(prompt)
    return response


@ta.jit
async def database_pipeline(query: str) -> list[dict]:
    """Pipeline using database replica."""
    results = await call_database(query)
    return results


@ta.jit
async def multi_replica_pipeline(query: str) -> dict:
    """Complex pipeline using multiple replicas.

    This shows:
    - Composing multiple replicas
    - Mixing replicas with regular nodes
    - Building complex data flows
    """
    # Fetch data from database
    db_results = await call_database(query)

    # Process with LLM
    prompt = f"Summarize: {db_results}"
    summary = await call_llm(prompt)

    # Count the operation
    count = await call_simple_counter(1)

    return {
        "query": query,
        "results": db_results,
        "summary": summary,
        "operation_count": count,
    }


# =============================================================================
# Demonstration Functions
# =============================================================================


async def demo_basic_replicas():
    """Demo basic replica patterns."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Stateful Replica")
    print("=" * 70)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        print("\nDirect replica calls (stateful):")
        result1 = await simple_pipeline(5)
        result2 = await simple_pipeline(3)
        result3 = await simple_pipeline(2)
        print(f"  Call 1: increment by 5 -> {result1}")
        print(f"  Call 2: increment by 3 -> {result2}")
        print(f"  Call 3: increment by 2 -> {result3}")
        print(f"  Total: {result3} (cumulative)")


async def demo_pure_replicas():
    """Demo pure replica with apply_replica."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Pure Replica with apply_replica")
    print("=" * 70)

    # Create replica instance directly
    pure = PureCounter(count=10, step=5)

    print("\nUsing apply_replica for pure updates:")
    print(f"  Initial state: count={pure.count}, step={pure.step}")

    # Update 1: increment
    out1 = await ta.apply_replica(pure, increment_step, 100)
    print(f"  After increment_step(100): count={pure.count}, output={out1}")

    # Update 2: increment again
    out2 = await ta.apply_replica(pure, increment_step, 50)
    print(f"  After increment_step(50): count={pure.count}, output={out2}")

    print("\n  Note: The replica instance was mutated via pure updates!")


async def demo_llm_replica():
    """Demo LLM replica with runtime state."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: LLM Replica with Runtime State")
    print("=" * 70)

    llm_instance = LLM("gpt-4", temperature=0.5)

    print(f"\nInitial state: temperature={llm_instance.temperature}")

    mesh = ta.Mesh([ta.GpuDevice(0)])

    with mesh:
        # Use in pipeline
        print("\nCalling LLM in pipeline:")
        response = await llm_pipeline("What is machine learning?")
        print(f"  Response: {response[:60]}...")

    # Update temperature with pure function
    print("\nUpdating temperature with apply_replica:")
    msg = await ta.apply_replica(llm_instance, update_temperature, 0.9)
    print(f"  {msg}")
    print(f"  New temperature: {llm_instance.temperature}")


async def demo_database_replica():
    """Demo database replica with query history."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Database Replica with Query History")
    print("=" * 70)

    db = Database("postgresql://localhost/mydb")

    print(f"\nInitial query history: {db.query_history}")

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        print("\nExecuting queries:")
        results1 = await database_pipeline("SELECT * FROM users")
        print(f"  Query 1 results: {len(results1)} rows")

        results2 = await database_pipeline("SELECT * FROM orders")
        print(f"  Query 2 results: {len(results2)} rows")

    # Update history with pure function
    print("\nAdding to query history with apply_replica:")
    history = await ta.apply_replica(db, add_query_to_history, "SELECT * FROM products")
    print(f"  Updated history: {history}")


async def demo_multi_replica():
    """Demo complex pipeline with multiple replicas."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Replica Pipeline")
    print("=" * 70)

    mesh = ta.Mesh([ta.GpuDevice(0)])

    with mesh:
        print("\nRunning complex pipeline with multiple replicas:")
        result = await multi_replica_pipeline("machine learning")
        print(f"  Query: {result['query']}")
        print(f"  DB Results: {len(result['results'])} rows")
        print(f"  LLM Summary: {result['summary'][:50]}...")
        print(f"  Operation Count: {result['operation_count']}")


async def demo_summary():
    """Print summary of patterns."""
    print("\n" + "=" * 70)
    print("PATTERN SUMMARY")
    print("=" * 70)
    print("""
1. BASIC REPLICA (@replica only):
   - Simple stateful objects (counters, caches)
   - State mutations happen inside __call__
   - Good for straightforward use cases

2. PURE REPLICA (@trace + @replica):
   - Implement decompose/compose protocol
   - Enable pure functional updates via apply_replica
   - Better composability with graph operations
   - Children = dynamic state, Aux = static initialization

3. RUNTIME STATE IN AUX:
   - Heavy initialization (model loading) in aux
   - Dynamic state (temperature, cache) in children
   - Separates expensive setup from runtime state

4. REPLICAS IN @JIT:
   - Create handles outside @jit with Class.init()
   - Call replicas from @node functions (not directly in @jit)
   - Use @bind_service to register replica handles with @node
   - Pattern: @bind_service(svc=handle) + @node + await svc(args)
   - Works with all control flow (cond, while, scan, switch)
   - Mesh context determines execution (local/remote)

5. APPLY_REPLICA FOR PURE UPDATES:
   - await apply_replica(replica, update_fn, *args)
   - update_fn must be @jit-compatible (use @node)
   - Returns (new_children, output)
   - Enables pure functional state management

6. ADVANCED PATTERNS:
   - Loops with replicas (while_loop, scan)
   - Conditional execution (cond, switch)
   - Multi-replica composition
   - Mixing stateful and pure operations

KEY INSIGHTS:
- Graphs are PURE (immutable, traceable)
- Replicas can be STATEFUL (mutable actors)
- @trace + decompose/compose bridges the gap
- apply_replica enables pure updates on stateful objects
- Mesh controls WHERE code runs (local/remote/distributed)
- @jit controls HOW code runs (compiled graph)
""")

    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("STATEFUL REPLICA PATTERNS - COMPREHENSIVE GUIDE")
    print("=" * 70)

    # Run all demos
    await demo_basic_replicas()
    await demo_pure_replicas()
    await demo_llm_replica()
    await demo_database_replica()
    await demo_multi_replica()
    await demo_summary()

    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  - thinkagain/core/execution/replica.py")
    print("  - thinkagain/core/execution/replica_state.py")
    print("  - thinkagain/core/traceable.py")
    print("  - examples/distributed_example.py")
    print("  - tests/test_distributed.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
