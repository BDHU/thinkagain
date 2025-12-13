import asyncio

from thinkagain import Context, Worker, async_worker


class _SuffixWorker(Worker):
    async def arun(self, ctx: Context) -> Context:
        ctx.value.append("suffix")
        return ctx


def test_async_worker_decorator_runs_and_composes() -> None:
    calls: list[str] = []

    @async_worker
    async def starter(ctx: Context) -> Context:
        calls.append("starter")
        ctx.value.append("starter")
        return ctx

    assert isinstance(starter, Worker)
    assert starter.name == "starter"

    pipeline = starter >> _SuffixWorker()
    ctx = asyncio.run(pipeline.arun(Context(value=[])))

    assert ctx.value == ["starter", "suffix"]
    assert calls == ["starter"]
