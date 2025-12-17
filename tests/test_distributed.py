"""Tests for distributed execution."""

import pytest
import thinkagain
from thinkagain import node, run, worker, launch, shutdown, runtime, WorkerServiceError
from thinkagain.distributed import (
    get_worker_spec,
    get_all_workers,
    clear_worker_registry,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear worker registry before each test."""
    clear_worker_registry()
    yield
    clear_worker_registry()


class TestWorkerDecorator:
    def test_worker_registers_class(self):
        @worker
        class MyWorker:
            pass

        spec = get_worker_spec("MyWorker")
        assert spec is not None
        assert spec.cls is MyWorker
        assert spec.n == 1

    def test_worker_with_n(self):
        @worker(n=4)
        class MyWorker:
            pass

        spec = get_worker_spec("MyWorker")
        assert spec is not None
        assert spec.n == 4

    def test_get_all_workers(self):
        @worker
        class Worker1:
            pass

        @worker(n=2)
        class Worker2:
            pass

        all_workers = get_all_workers()
        assert "Worker1" in all_workers
        assert "Worker2" in all_workers
        assert len(all_workers) == 2


class TestWorkerLaunch:
    def test_launch_creates_instances(self):
        @worker(n=3)
        class MyWorker:
            pass

        MyWorker.launch()
        spec = get_worker_spec("MyWorker")
        assert len(spec._instances) == 3

    def test_launch_with_custom_instances(self):
        @worker(n=2)
        class MyWorker:
            def __init__(self, value: int = 0):
                self.value = value

        custom = [MyWorker(10), MyWorker(20)]
        MyWorker.launch(custom)

        spec = get_worker_spec("MyWorker")
        assert len(spec._instances) == 2
        assert spec._instances[0].value == 10
        assert spec._instances[1].value == 20

    def test_shutdown_clears_instances(self):
        @worker(n=2)
        class MyWorker:
            pass

        MyWorker.launch()
        spec = get_worker_spec("MyWorker")
        assert len(spec._instances) == 2

        MyWorker.shutdown()
        assert len(spec._instances) == 0

    def test_global_launch(self):
        @worker(n=2)
        class WorkerA:
            pass

        @worker(n=3)
        class WorkerB:
            pass

        launch()

        spec_a = get_worker_spec("WorkerA")
        spec_b = get_worker_spec("WorkerB")
        assert len(spec_a._instances) == 2
        assert len(spec_b._instances) == 3

    def test_global_shutdown(self):
        @worker(n=2)
        class WorkerA:
            pass

        @worker(n=3)
        class WorkerB:
            pass

        launch()
        shutdown()

        spec_a = get_worker_spec("WorkerA")
        spec_b = get_worker_spec("WorkerB")
        assert len(spec_a._instances) == 0
        assert len(spec_b._instances) == 0

    def test_global_launch_skips_already_launched(self):
        @worker(n=2)
        class MyWorker:
            def __init__(self, value: int = 0):
                self.value = value

        # Launch with custom instances first
        MyWorker.launch([MyWorker(10), MyWorker(20)])

        # Global launch should not overwrite
        launch()

        spec = get_worker_spec("MyWorker")
        assert len(spec._instances) == 2
        assert spec._instances[0].value == 10
        assert spec._instances[1].value == 20


class TestWorkerServiceAccess:
    def test_non_node_method_raises_error(self):
        @worker
        class MyWorker:
            def helper(self):
                return 42

        with pytest.raises(WorkerServiceError) as exc_info:
            MyWorker.helper
        assert "not a @node service" in str(exc_info.value)

    def test_node_method_allowed(self):
        @worker
        class MyWorker:
            @node
            async def process(self, ctx):
                return ctx

        # Should not raise
        assert MyWorker.process is not None

    def test_private_method_allowed(self):
        @worker
        class MyWorker:
            def _helper(self):
                return 42

        # Private methods are allowed (accessed on instances internally)
        # This should not raise when accessed on class
        assert callable(MyWorker._helper)

    def test_helper_works_on_instance(self):
        @worker
        class MyWorker:
            def helper(self):
                return 42

            @node
            async def process(self, ctx):
                ctx.set("value", self.helper())
                return ctx

        def pipeline(ctx):
            ctx = MyWorker.process(ctx)
            return ctx

        MyWorker.launch([MyWorker()])
        result = run(pipeline, {})
        assert result.get("value") == 42


class TestNodeMethodDetection:
    def test_function_node_is_not_method(self):
        @node
        async def my_func(ctx):
            return ctx

        assert not my_func.is_method
        assert my_func.class_name is None

    def test_method_node_is_method(self):
        class MyClass:
            @node
            async def my_method(self, ctx):
                return ctx

        assert MyClass.my_method.is_method
        assert MyClass.my_method.class_name == "MyClass"

    def test_bound_method_detection(self):
        class MyClass:
            @node
            async def my_method(self, ctx):
                return ctx

        obj = MyClass()
        bound = obj.my_method

        assert bound.is_bound
        assert bound.is_method

    def test_unbound_method_not_bound(self):
        class MyClass:
            @node
            async def my_method(self, ctx):
                return ctx

        assert not MyClass.my_method.is_bound


class TestWorkerExecution:
    def test_run_stateless_pipeline(self):
        @node
        async def add_one(ctx):
            ctx.set("value", ctx.get("value", 0) + 1)
            return ctx

        def pipeline(ctx):
            ctx = add_one(ctx)
            return ctx

        result = run(pipeline, {"value": 5})
        assert result.get("value") == 6

    def test_run_worker_pipeline(self):
        @worker
        class Multiplier:
            def __init__(self, factor: int = 10):
                self.factor = factor

            @node
            async def multiply(self, ctx):
                ctx.set("value", ctx.get("value") * self.factor)
                return ctx

        @node
        async def add_one(ctx):
            ctx.set("value", ctx.get("value") + 1)
            return ctx

        def pipeline(ctx):
            ctx = add_one(ctx)
            ctx = Multiplier.multiply(ctx)
            return ctx

        Multiplier.launch([Multiplier(factor=10)])

        result = run(pipeline, {"value": 5})
        assert result.get("value") == 60  # (5+1) * 10

    def test_round_robin_distribution(self):
        @worker(n=3)
        class Counter:
            def __init__(self, name: str = ""):
                self.name = name
                self.call_count = 0

            @node
            async def count(self, ctx):
                self.call_count += 1
                ctx.set("last_worker", self.name)
                return ctx

        def pipeline(ctx):
            ctx = Counter.count(ctx)
            return ctx

        workers = [Counter("a"), Counter("b"), Counter("c")]
        Counter.launch(workers)

        # Run multiple times
        results = []
        for _ in range(6):
            result = run(pipeline, {})
            results.append(result.get("last_worker"))

        # Should round-robin through workers
        assert results == ["a", "b", "c", "a", "b", "c"]
        assert all(w.call_count == 2 for w in workers)

    def test_missing_worker_instance_error(self):
        from thinkagain.core import NodeExecutionError

        @worker
        class MyWorker:
            @node
            async def process(self, ctx):
                return ctx

        def pipeline(ctx):
            ctx = MyWorker.process(ctx)
            return ctx

        # Don't launch any instances

        with pytest.raises(NodeExecutionError) as exc_info:
            run(pipeline, {})
        assert "No instances available" in str(exc_info.value.cause)


class TestConditionalPipelines:
    def test_conditional_with_stateless_before(self):
        """Conditional after stateless node should work."""

        @worker
        class FastWorker:
            @node
            async def process(self, ctx):
                ctx.set("result", "fast")
                return ctx

        @worker
        class SlowWorker:
            @node
            async def process(self, ctx):
                ctx.set("result", "slow")
                return ctx

        @node
        async def set_flag(ctx):
            ctx.set("use_fast", ctx.get("fast", False))
            return ctx

        def pipeline(ctx):
            ctx = set_flag(ctx)
            if ctx.get("use_fast"):
                ctx = FastWorker.process(ctx)
            else:
                ctx = SlowWorker.process(ctx)
            return ctx

        FastWorker.launch([FastWorker()])
        SlowWorker.launch([SlowWorker()])

        result_fast = run(pipeline, {"fast": True})
        assert result_fast.get("result") == "fast"

        result_slow = run(pipeline, {"fast": False})
        assert result_slow.get("result") == "slow"


class TestAsyncExecution:
    @pytest.mark.asyncio
    async def test_arun_pipeline(self):
        from thinkagain import arun

        @worker(n=2)
        class MyWorker:
            def __init__(self, value: int = 0):
                self.value = value

            @node
            async def add(self, ctx):
                ctx.set("result", ctx.get("result", 0) + self.value)
                return ctx

        def pipeline(ctx):
            ctx = MyWorker.add(ctx)
            ctx = MyWorker.add(ctx)
            return ctx

        MyWorker.launch([MyWorker(10), MyWorker(20)])

        result = await arun(pipeline, {})
        # Round-robin: first call uses 10, second uses 20
        assert result.get("result") == 30


class TestRuntimeContextManager:
    def test_runtime_launches_and_shuts_down(self):
        @worker(n=2)
        class MyWorker:
            @node
            async def process(self, ctx):
                ctx.set("processed", True)
                return ctx

        def pipeline(ctx):
            ctx = MyWorker.process(ctx)
            return ctx

        spec = get_worker_spec("MyWorker")
        assert len(spec._instances) == 0

        with runtime():
            assert len(spec._instances) == 2
            result = run(pipeline, {})
            assert result.get("processed") is True

        assert len(spec._instances) == 0

    def test_runtime_shuts_down_on_exception(self):
        @worker(n=1)
        class MyWorker:
            @node
            async def fail(self, ctx):
                raise ValueError("intentional error")

        def pipeline(ctx):
            ctx = MyWorker.fail(ctx)
            return ctx

        spec = get_worker_spec("MyWorker")

        with pytest.raises(thinkagain.NodeExecutionError):
            with runtime():
                assert len(spec._instances) == 1
                run(pipeline, {})

        # Should still shut down after exception
        assert len(spec._instances) == 0

    def test_unified_imports(self):
        """Verify all main APIs are accessible from thinkagain package."""
        assert thinkagain.node is not None
        assert thinkagain.worker is not None
        assert thinkagain.run is not None
        assert thinkagain.arun is not None
        assert thinkagain.launch is not None
        assert thinkagain.shutdown is not None
        assert thinkagain.runtime is not None
        assert thinkagain.Context is not None
        assert thinkagain.Node is not None
        assert thinkagain.NodeExecutionError is not None
        assert thinkagain.WorkerServiceError is not None
