"""Regression tests for self-referencing graph vulnerabilities."""

import re

import pytest

from thinkagain.core.graph import END, Graph
from thinkagain.core.worker import Worker
from thinkagain.core.context import Context


class DummyWorker(Worker):
    """Simple worker for wiring graphs."""

    async def arun(
        self, ctx: Context
    ) -> Context:  # pragma: no cover - trivial passthrough
        return ctx


def test_direct_self_reference_raises_value_error() -> None:
    graph = Graph(name="self_ref")
    worker = DummyWorker(name="worker")

    graph.add_node("worker", worker)
    graph.add_node("self", graph)
    graph.add_edge("worker", "self")
    graph.add_edge("self", END)
    graph.set_entry("worker")

    with pytest.raises(ValueError, match=re.compile("cycle detected", re.IGNORECASE)):
        graph.compile(flatten=True)


def test_indirect_self_reference_is_detected() -> None:
    g1 = Graph(name="graph1")
    g2 = Graph(name="graph2")
    worker = DummyWorker(name="worker")

    g1.add_node("worker", worker)
    g1.add_node("g2", g2)
    g1.add_edge("worker", "g2")
    g1.add_edge("g2", END)
    g1.set_entry("worker")

    g2.add_node("g1", g1)
    g2.add_edge("g1", END)
    g2.set_entry("g1")

    with pytest.raises(ValueError, match=re.compile("cycle detected", re.IGNORECASE)):
        g1.compile(flatten=True)


def test_normal_nested_graph_compiles() -> None:
    outer = Graph(name="outer")
    inner = Graph(name="inner")
    worker1 = DummyWorker(name="worker1")
    worker2 = DummyWorker(name="worker2")

    inner.add_node("w2", worker2)
    inner.add_edge("w2", END)
    inner.set_entry("w2")

    outer.add_node("w1", worker1)
    outer.add_node("subgraph", inner)
    outer.add_edge("w1", "subgraph")
    outer.add_edge("subgraph", END)
    outer.set_entry("w1")

    compiled = outer.compile(flatten=True)
    assert "subgraph__w2" in compiled.nodes
