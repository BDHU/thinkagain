import asyncio

from thinkagain import Context, Executable, Graph, END, async_executable


class _SuffixExecutable(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.value.append("suffix")
        return ctx


def test_async_worker_decorator_runs_and_composes() -> None:
    calls: list[str] = []

    @async_executable
    async def starter(ctx: Context) -> Context:
        calls.append("starter")
        ctx.value.append("starter")
        return ctx

    assert isinstance(starter, Executable)
    assert starter.name == "starter"

    # Build pipeline using Graph
    graph = Graph()
    graph.add("starter", starter)
    graph.add("suffix", _SuffixExecutable())
    graph.edge("starter", "suffix")
    graph.edge("suffix", END)

    ctx = asyncio.run(graph.compile().arun(Context(value=[])))

    assert ctx.value == ["starter", "suffix"]
    assert calls == ["starter"]
