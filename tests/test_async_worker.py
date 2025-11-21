import pytest

from thinkagain import Context, Worker, async_worker


class _SuffixWorker(Worker):
    def __call__(self, ctx: Context) -> Context:  # pragma: no cover - trivial
        ctx.value.append("suffix")
        return ctx


@pytest.mark.asyncio
async def test_async_worker_decorator_runs_and_composes() -> None:
    calls: list[str] = []

    @async_worker
    async def starter(ctx: Context) -> Context:
        calls.append("starter")
        ctx.value.append("starter")
        return ctx

    assert isinstance(starter, Worker)
    assert starter.name == "starter"

    pipeline = starter >> _SuffixWorker()
    ctx = await pipeline.arun(Context(value=[]))

    assert ctx.value == ["starter", "suffix"]
    assert calls == ["starter"]
