"""
Graph-based pipeline supporting arbitrary cycles and conditional routing.

Provides Graph class for complex agent workflows with dynamic routing,
while maintaining simplicity for inspection and debugging.

Use Graph when you need:
- Cycles (loops back to previous nodes)
- Dynamic routing based on runtime state
- Multi-agent interactions with complex flow

For simple sequential pipelines, prefer the >> operator:
    pipeline = worker1 >> worker2 >> worker3

Each graph supports both synchronous and asynchronous execution:
- Use .run(ctx) or graph(ctx) for synchronous execution
- Use await .arun(ctx) for asynchronous execution
"""

import asyncio
from typing import Callable, Dict, Tuple, Union, Optional, Any, Iterator, Generator
from .context import Context
import warnings


# Special constant for graph termination (optional, can also just omit edges)
END = "__end__"


class StepResult:
    """
    Result of executing a single node in the graph.

    Returned by Graph.steps() for step-by-step debugging.
    """

    def __init__(self, node: str, ctx: Context, next_node: Optional[str] = None):
        """
        Initialize step result.

        Args:
            node: Name of the node that was just executed
            ctx: Context after executing this node
            next_node: Name of the next node to execute (or END/None)
        """
        self.node = node
        self.ctx = ctx
        self.next_node = next_node

    def __repr__(self) -> str:
        return f"StepResult(node='{self.node}', next='{self.next_node}')"


class Graph:
    """
    Graph-based pipeline supporting arbitrary cycles and conditional routing.

    Unlike Pipeline (which uses >> for sequential composition), Graph uses
    explicit nodes and edges to support complex workflows with cycles.

    Example - Self-correcting RAG with cycle:
        graph = Graph(name="self_correcting_rag")

        # Add nodes
        graph.add_node("retrieve", retrieve_worker)
        graph.add_node("critique", critique_worker)
        graph.add_node("refine", refine_worker)
        graph.add_node("generate", generate_worker)

        # Set starting point
        graph.set_entry("retrieve")

        # Define flow (with cycles!)
        graph.add_edge("retrieve", "critique")

        graph.add_conditional_edge(
            "critique",
            route=lambda ctx: "refine" if ctx.quality < 0.8 else "generate",
            paths={
                "refine": "refine",
                "generate": "generate"
            }
        )

        graph.add_edge("refine", "retrieve")  # Cycle back!
        graph.add_edge("generate", END)

        # Execute directly
        result = graph(Context(query="Explain quantum computing"))

        # Inspect execution
        print(result.execution_path)
        # ["retrieve", "critique", "refine", "retrieve", "critique", "generate"]
    """

    def __init__(self, name: str = "graph", max_steps: Optional[int] = None):
        """
        Initialize a new graph.

        Args:
            name: Name for this graph (used in logging)
            max_steps: Optional maximum execution steps to prevent infinite loops.
                      If None (default), no limit is enforced - user is responsible
                      for ensuring graph terminates via proper routing logic.
        """
        self.name = name
        self.nodes: Dict[str, Any] = {}  # node_name -> worker
        self.edges: Dict[str, Union[str, Tuple]] = {}  # node_name -> next or (route_fn, paths)
        self.entry_point: Optional[str] = None
        self.max_steps = max_steps

    def add_node(self, name: str, worker) -> 'Graph':
        """
        Add a node to the graph.

        Args:
            name: Unique identifier for this node
            worker: Worker instance to execute at this node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node name already exists
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = worker

        # Auto-set entry point if this is the first node
        if self.entry_point is None:
            self.entry_point = name

        return self

    def set_entry(self, name: str) -> 'Graph':
        """
        Set the starting node for graph execution.

        Args:
            name: Name of the entry node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node does not exist
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")
        self.entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> 'Graph':
        """
        Add a direct edge between two nodes.

        Args:
            from_node: Source node name
            to_node: Destination node name (or END to terminate)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If nodes don't exist or from_node already has an edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' does not exist")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' does not exist")

        if from_node in self.edges:
            raise ValueError(
                f"Node '{from_node}' already has an outgoing edge. "
                f"Use add_conditional_edge for multiple paths."
            )

        self.edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        route: Callable[[Context], str],
        paths: Dict[str, str]
    ) -> 'Graph':
        """
        Add a conditional edge that routes based on context state.

        The route function examines the context and returns a key from
        the paths dict, which maps to the next node.

        Args:
            from_node: Source node name
            route: Function that takes Context and returns a path key
            paths: Mapping of route keys to node names

        Returns:
            Self for method chaining

        Example:
            graph.add_conditional_edge(
                "critique",
                route=lambda ctx: "high" if ctx.score > 0.8 else "low",
                paths={
                    "high": "generate",
                    "low": "refine"
                }
            )

        Raises:
            ValueError: If from_node doesn't exist or already has an edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' does not exist")

        if from_node in self.edges:
            raise ValueError(
                f"Node '{from_node}' already has an outgoing edge"
            )

        # Validate all path destinations exist
        for path_key, to_node in paths.items():
            if to_node != END and to_node not in self.nodes:
                raise ValueError(
                    f"Path '{path_key}' points to non-existent node '{to_node}'"
                )

        self.edges[from_node] = (route, paths)
        return self

    def __call__(self, ctx: Context) -> Context:
        """Execute the graph starting from entry point (synchronous). Alias for run()."""
        return self.run(ctx)

    def run(self, ctx: Context) -> Context:
        """
        Execute the graph starting from entry point (synchronous).

        Args:
            ctx: Input context with initial state

        Returns:
            Context with execution results and path history

        Raises:
            ValueError: If graph is invalid or execution fails
        """
        # Lazy validation on first execution
        self._validate()

        current = self.entry_point
        execution_path = []

        ctx.log(f"[Graph] Starting: {self.name}")
        ctx.log(f"[Graph] Entry point: {current}")

        step = 0
        while True:
            if current == END:
                ctx.log(f"[Graph] Reached END after {step} steps")
                break

            # Execute current node
            worker = self.nodes[current]
            ctx.log(f"[Graph] Executing node: {current}")

            try:
                ctx = worker(ctx)
            except Exception as e:
                ctx.log(f"[Graph] Error in node '{current}': {e}")
                raise

            execution_path.append(current)

            # Determine next node
            if current not in self.edges:
                ctx.log(f"[Graph] Node '{current}' has no outgoing edge, terminating")
                break

            edge = self.edges[current]

            if isinstance(edge, tuple):  # Conditional edge
                route_fn, edge_map = edge

                try:
                    route_result = route_fn(ctx)
                except Exception as e:
                    ctx.log(f"[Graph] Error in routing function: {e}")
                    raise

                ctx.log(f"[Graph] Conditional routing from '{current}': '{route_result}'")

                if route_result in edge_map:
                    current = edge_map[route_result]
                elif route_result == END:
                    current = END
                else:
                    available = list(edge_map.keys()) + [END]
                    raise ValueError(
                        f"Route function returned '{route_result}' but no matching edge. "
                        f"Available paths: {available}"
                    )
            else:  # Direct edge
                next_node = edge
                ctx.log(f"[Graph] Direct edge: '{current}' → '{next_node}'")
                current = next_node

            # Check max_steps limit if set
            step += 1
            if self.max_steps is not None and step >= self.max_steps:
                ctx.log(f"[Graph] WARNING: Terminated after max_steps limit ({self.max_steps})")
                ctx.log(f"[Graph] This may indicate an infinite loop or insufficient max_steps")
                break

        # Store execution metadata
        ctx.execution_path = execution_path
        ctx.total_steps = len(execution_path)

        ctx.log(f"[Graph] Completed: {self.name}")
        ctx.log(f"[Graph] Total steps: {ctx.total_steps}")
        return ctx

    async def arun(self, ctx: Context) -> Context:
        """
        Execute the graph starting from entry point (asynchronous).

        Args:
            ctx: Input context with initial state

        Returns:
            Context with execution results and path history

        Raises:
            ValueError: If graph is invalid or execution fails
        """
        # Lazy validation on first execution
        self._validate()

        current = self.entry_point
        execution_path = []

        ctx.log(f"[Graph] Starting: {self.name}")
        ctx.log(f"[Graph] Entry point: {current}")

        step = 0
        while True:
            if current == END:
                ctx.log(f"[Graph] Reached END after {step} steps")
                break

            # Execute current node (await if async)
            worker = self.nodes[current]
            ctx.log(f"[Graph] Executing node: {current}")

            try:
                # Call acall() for async execution if available
                if hasattr(worker, 'acall'):
                    ctx = await worker.acall(ctx)
                else:
                    ctx = await worker(ctx)
            except Exception as e:
                ctx.log(f"[Graph] Error in node '{current}': {e}")
                raise

            execution_path.append(current)

            # Determine next node
            if current not in self.edges:
                ctx.log(f"[Graph] Node '{current}' has no outgoing edge, terminating")
                break

            edge = self.edges[current]

            if isinstance(edge, tuple):  # Conditional edge
                route_fn, edge_map = edge

                try:
                    # Allow async routing functions
                    if asyncio.iscoroutinefunction(route_fn):
                        route_result = await route_fn(ctx)
                    else:
                        route_result = route_fn(ctx)
                except Exception as e:
                    ctx.log(f"[Graph] Error in routing function: {e}")
                    raise

                ctx.log(f"[Graph] Conditional routing from '{current}': '{route_result}'")

                if route_result in edge_map:
                    current = edge_map[route_result]
                elif route_result == END:
                    current = END
                else:
                    available = list(edge_map.keys()) + [END]
                    raise ValueError(
                        f"Route function returned '{route_result}' but no matching edge. "
                        f"Available paths: {available}"
                    )
            else:  # Direct edge
                next_node = edge
                ctx.log(f"[Graph] Direct edge: '{current}' → '{next_node}'")
                current = next_node

            # Check max_steps limit if set
            step += 1
            if self.max_steps is not None and step >= self.max_steps:
                ctx.log(f"[Graph] WARNING: Terminated after max_steps limit ({self.max_steps})")
                ctx.log(f"[Graph] This may indicate an infinite loop or insufficient max_steps")
                break

        # Store execution metadata
        ctx.execution_path = execution_path
        ctx.total_steps = len(execution_path)

        ctx.log(f"[Graph] Completed: {self.name}")
        ctx.log(f"[Graph] Total steps: {ctx.total_steps}")
        return ctx

    def steps(self, ctx: Context) -> Generator[StepResult, None, None]:
        """
        Execute graph step-by-step, yielding after each node execution.

        This allows for interactive debugging, state inspection, and
        control flow visualization during execution.

        Args:
            ctx: Input context with initial state

        Yields:
            StepResult containing the node name, context, and next node

        Example:
            for step in graph.steps(ctx):
                print(f"Executed: {step.node}")
                print(f"State: {step.ctx.data}")
                if step.node == "critique":
                    print(f"Quality: {step.ctx.quality}")

            # Can also modify state between steps
            for step in graph.steps(ctx):
                if step.node == "critique" and step.ctx.quality < 0.5:
                    step.ctx.quality = 0.6  # Override for testing
        """
        # Lazy validation on first execution
        self._validate()

        current = self.entry_point

        ctx.log(f"[Graph] Starting: {self.name} (step-by-step mode)")
        ctx.log(f"[Graph] Entry point: {current}")

        step_count = 0
        while True:
            if current == END:
                ctx.log(f"[Graph] Reached END after {step_count} steps")
                break

            # Execute current node
            worker = self.nodes[current]
            ctx.log(f"[Graph] Executing node: {current}")

            try:
                ctx = worker(ctx)
            except Exception as e:
                ctx.log(f"[Graph] Error in node '{current}': {e}")
                raise

            # Determine next node
            next_node = None
            if current not in self.edges:
                ctx.log(f"[Graph] Node '{current}' has no outgoing edge, will terminate")
                next_node = None
            else:
                edge = self.edges[current]

                if isinstance(edge, tuple):  # Conditional edge
                    route_fn, edge_map = edge

                    try:
                        route_result = route_fn(ctx)
                    except Exception as e:
                        ctx.log(f"[Graph] Error in routing function: {e}")
                        raise

                    ctx.log(f"[Graph] Conditional routing from '{current}': '{route_result}'")

                    if route_result in edge_map:
                        next_node = edge_map[route_result]
                    elif route_result == END:
                        next_node = END
                    else:
                        available = list(edge_map.keys()) + [END]
                        raise ValueError(
                            f"Route function returned '{route_result}' but no matching edge. "
                            f"Available paths: {available}"
                        )
                else:  # Direct edge
                    next_node = edge
                    ctx.log(f"[Graph] Direct edge: '{current}' → '{next_node}'")

            # Yield control to caller with current state
            yield StepResult(node=current, ctx=ctx, next_node=next_node)

            # Check if we should continue
            if next_node is None or next_node == END:
                break

            current = next_node
            step_count += 1

            # Check max_steps limit if set
            if self.max_steps is not None and step_count >= self.max_steps:
                ctx.log(f"[Graph] WARNING: Terminated after max_steps limit ({self.max_steps})")
                ctx.log(f"[Graph] This may indicate an infinite loop or insufficient max_steps")
                break

        ctx.log(f"[Graph] Completed: {self.name} (step-by-step mode)")

    async def asteps(self, ctx: Context):
        """
        Execute graph step-by-step asynchronously, yielding after each node execution.

        This allows for interactive debugging, state inspection, and
        control flow visualization during async execution.

        Args:
            ctx: Input context with initial state

        Yields:
            StepResult containing the node name, context, and next node

        Example:
            async for step in graph.asteps(ctx):
                print(f"Executed: {step.node}")
                print(f"State: {step.ctx.data}")
                if step.node == "critique":
                    print(f"Quality: {step.ctx.quality}")
        """
        # Lazy validation on first execution
        self._validate()

        current = self.entry_point

        ctx.log(f"[Graph] Starting: {self.name} (step-by-step mode)")
        ctx.log(f"[Graph] Entry point: {current}")

        step_count = 0
        while True:
            if current == END:
                ctx.log(f"[Graph] Reached END after {step_count} steps")
                break

            # Execute current node (await if async)
            worker = self.nodes[current]
            ctx.log(f"[Graph] Executing node: {current}")

            try:
                # Call acall() for async execution if available
                if hasattr(worker, 'acall'):
                    ctx = await worker.acall(ctx)
                else:
                    ctx = await worker(ctx)
            except Exception as e:
                ctx.log(f"[Graph] Error in node '{current}': {e}")
                raise

            # Determine next node
            next_node = None
            if current not in self.edges:
                ctx.log(f"[Graph] Node '{current}' has no outgoing edge, will terminate")
                next_node = None
            else:
                edge = self.edges[current]

                if isinstance(edge, tuple):  # Conditional edge
                    route_fn, edge_map = edge

                    try:
                        # Allow async routing functions
                        if asyncio.iscoroutinefunction(route_fn):
                            route_result = await route_fn(ctx)
                        else:
                            route_result = route_fn(ctx)
                    except Exception as e:
                        ctx.log(f"[Graph] Error in routing function: {e}")
                        raise

                    ctx.log(f"[Graph] Conditional routing from '{current}': '{route_result}'")

                    if route_result in edge_map:
                        next_node = edge_map[route_result]
                    elif route_result == END:
                        next_node = END
                    else:
                        available = list(edge_map.keys()) + [END]
                        raise ValueError(
                            f"Route function returned '{route_result}' but no matching edge. "
                            f"Available paths: {available}"
                        )
                else:  # Direct edge
                    next_node = edge
                    ctx.log(f"[Graph] Direct edge: '{current}' → '{next_node}'")

            # Yield control to caller with current state
            yield StepResult(node=current, ctx=ctx, next_node=next_node)

            # Check if we should continue
            if next_node is None or next_node == END:
                break

            current = next_node
            step_count += 1

            # Check max_steps limit if set
            if self.max_steps is not None and step_count >= self.max_steps:
                ctx.log(f"[Graph] WARNING: Terminated after max_steps limit ({self.max_steps})")
                ctx.log(f"[Graph] This may indicate an infinite loop or insufficient max_steps")
                break

        ctx.log(f"[Graph] Completed: {self.name} (step-by-step mode)")

    def _validate(self):
        """Validate graph structure (called lazily on first execution)."""
        if self.entry_point is None:
            raise ValueError("Entry point not set. Use set_entry() or add the first node.")

        # Detect unreachable nodes (warning only)
        reachable = self._find_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            warnings.warn(f"Unreachable nodes detected: {unreachable}")

        # Detect nodes without outgoing edges (warning only)
        dead_ends = [node for node in self.nodes if node not in self.edges]
        if dead_ends:
            warnings.warn(
                f"Nodes without outgoing edges (potential dead ends): {dead_ends}. "
                f"Consider adding edges to END or other nodes."
            )

    def _find_reachable_nodes(self) -> set:
        """BFS to find all reachable nodes from entry point."""
        reachable = set()
        queue = [self.entry_point]

        while queue:
            current = queue.pop(0)
            if current in reachable or current == END:
                continue

            reachable.add(current)

            # Find outgoing edges
            if current in self.edges:
                edge = self.edges[current]
                if isinstance(edge, tuple):  # Conditional edge
                    route_fn, edge_map = edge
                    queue.extend([v for v in edge_map.values() if v != END])
                else:  # Direct edge
                    if edge != END:
                        queue.append(edge)

        return reachable

    def visualize(self) -> str:
        """
        Generate Mermaid diagram syntax for the graph.

        Returns:
            Mermaid diagram code that can be rendered or saved

        Example:
            print(graph.visualize())
            # Copy output to mermaid.live or GitHub markdown
        """
        lines = ["graph TD"]
        lines.append(f"    START([START]) --> {self.entry_point}")

        # Add nodes
        for node_name in self.nodes:
            # Use rounded rectangles for regular nodes
            lines.append(f"    {node_name}[{node_name}]")

        # Add END node
        lines.append(f"    END([END])")

        # Add edges
        for from_node, edge in self.edges.items():
            if isinstance(edge, tuple):  # Conditional
                route_fn, edge_map = edge
                for label, to_node in edge_map.items():
                    if to_node == END:
                        lines.append(f"    {from_node} -->|{label}| END")
                    else:
                        lines.append(f"    {from_node} -->|{label}| {to_node}")
            else:  # Direct
                if edge == END:
                    lines.append(f"    {from_node} --> END")
                else:
                    lines.append(f"    {from_node} --> {edge}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export graph structure as dictionary."""
        edges_dict = {}
        for k, v in self.edges.items():
            if isinstance(v, tuple):
                route_fn, edge_map = v
                fn_name = route_fn.__name__ if hasattr(route_fn, '__name__') else "lambda"
                edges_dict[k] = {
                    "type": "conditional",
                    "function": fn_name,
                    "paths": edge_map
                }
            else:
                edges_dict[k] = {"type": "direct", "to": v}

        return {
            "type": "Graph",
            "name": self.name,
            "entry_point": self.entry_point,
            "max_steps": self.max_steps,
            "nodes": {
                name: worker.to_dict() if hasattr(worker, 'to_dict')
                else {"type": "Unknown", "name": str(worker)}
                for name, worker in self.nodes.items()
            },
            "edges": edges_dict
        }

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, entry='{self.entry_point}')"
