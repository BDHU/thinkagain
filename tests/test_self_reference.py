"""Regression tests for self-referencing graph vulnerabilities."""

import re

import pytest

from thinkagain import Graph, END, Worker, Context


class DummyWorker(Worker):
    """Simple worker for wiring graphs."""

    async def arun(self, ctx: Context) -> Context:
        return ctx


def test_direct_self_reference_raises_value_error() -> None:
    graph = Graph(name="self_ref")
    worker = DummyWorker(name="worker")

    graph.add("worker", worker)
    graph.add("self", graph)
    graph.edge("worker", "self")
    graph.edge("self", END)
    graph.set_entry("worker")

    with pytest.raises(ValueError, match=re.compile("cycle detected", re.IGNORECASE)):
        graph.compile()


def test_indirect_self_reference_is_detected() -> None:
    g1 = Graph(name="graph1")
    g2 = Graph(name="graph2")
    worker = DummyWorker(name="worker")

    g1.add("worker", worker)
    g1.add("g2", g2)
    g1.edge("worker", "g2")
    g1.edge("g2", END)
    g1.set_entry("worker")

    g2.add("g1", g1)
    g2.edge("g1", END)
    g2.set_entry("g1")

    with pytest.raises(ValueError, match=re.compile("cycle detected", re.IGNORECASE)):
        g1.compile()


def test_normal_nested_graph_compiles() -> None:
    outer = Graph(name="outer")
    inner = Graph(name="inner")
    worker1 = DummyWorker(name="worker1")
    worker2 = DummyWorker(name="worker2")

    inner.add("w2", worker2)
    inner.edge("w2", END)
    inner.set_entry("w2")

    outer.add("w1", worker1)
    outer.add("subgraph", inner)
    outer.edge("w1", "subgraph")
    outer.edge("subgraph", END)
    outer.set_entry("w1")

    compiled = outer.compile()
    assert "subgraph__w2" in compiled.nodes
