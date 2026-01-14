"""ActorHandle - Ray-style handle for replica actors with .method.go() support."""

from typing import Any
from uuid import UUID, uuid4

from .replica import ReplicaConfig, ReplicaHandle
from ..runtime.object_ref import ObjectRef
from ..runtime import get_current_runtime
from ..runtime.task import ActorTask

__all__ = ["ActorHandle", "RemoteMethod"]


class RemoteMethod:
    """Proxy for actor.method.go() calls.

    This enables the Ray-style actor API:
        actor = MyClass.init(args)
        ref = actor.method.go(x)
        result = await ref
    """

    def __init__(self, actor_handle: "ActorHandle", method_name: str):
        """Initialize the remote method proxy.

        Args:
            actor_handle: The actor handle this method belongs to
            method_name: Name of the method to call
        """
        self._actor_handle = actor_handle
        self._method_name = method_name

    def go(self, *args: Any, **kwargs: Any) -> ObjectRef:
        """Submit method call to scheduler as an actor task.

        Returns an ObjectRef immediately. The method will be executed on
        the actor instance when its dependencies are ready.

        Args:
            *args: Positional arguments (may contain ObjectRefs)
            **kwargs: Keyword arguments (may contain ObjectRefs)

        Returns:
            ObjectRef that will contain the result when execution completes

        Notes:
            If no runtime context is active, a default local runtime is used.

        Example:
            llm = LLM.init("llama")
            ref = llm.generate.go("hello")
            result = await ref
        """
        runtime = get_current_runtime()
        runtime.ensure_started()

        # Create actor task
        task = ActorTask(
            task_id=uuid4(),
            fn=None,  # Will be resolved from actor instance
            args=args,
            kwargs=kwargs,
            actor_handle=self._actor_handle,
            method_name=self._method_name,
        )

        # Submit and return ObjectRef
        return runtime.scheduler.submit_task(task)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call for backward compatibility.

        This allows calling methods directly without .go():
            result = await actor.method(x)

        For dynamic execution, use .go() instead:
            ref = actor.method.go(x)
            result = await ref
        """
        # For now, route through .go() for simplicity
        # This could be optimized to bypass scheduler for direct calls
        ref = self.go(*args, **kwargs)
        return await ref

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"RemoteMethod({self._actor_handle._replica_class.__name__}.{self._method_name})"


class ActorHandle:
    """Ray-style handle to a replica actor with .method.go() support.

    This wraps a ReplicaHandle and adds RemoteMethod proxies for each
    public method, enabling the Ray actor API:

    Example:
        @replica(gpus=1)
        class LLM:
            def __init__(self, model: str):
                self.model = model
                self.engine = load_model(model)

            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

            async def set_temperature(self, temp: float):
                self.temperature = temp

        # Create actor handle
        llm = LLM.init("llama")

        # Call methods with .go()
        async def pipeline(query: str):
            ref1 = llm.generate.go(query)
            result1 = await ref1

            await llm.set_temperature.go(0.9)

            ref2 = llm.generate.go(query)
            result2 = await ref2
            return result2
    """

    def __init__(
        self,
        replica_class: type,
        init_args: tuple,
        init_kwargs: dict,
        config: ReplicaConfig,
    ):
        """Initialize the actor handle.

        Args:
            replica_class: The class to instantiate
            init_args: Positional arguments for __init__
            init_kwargs: Keyword arguments for __init__
            config: Replica configuration (gpus, backend, etc.)
        """
        self._replica_class = replica_class
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._config = config
        self._uuid: UUID = uuid4()
        self._deployed = False

        # Create the underlying ReplicaHandle
        self._replica_handle = ReplicaHandle(
            replica_class=replica_class,
            init_args=init_args,
            init_kwargs=init_kwargs,
            config=config,
        )

        # Add .go() to each public method
        self._create_remote_methods()

    def _create_remote_methods(self):
        """Create RemoteMethod proxies for all public methods."""
        # Special case: if class has __call__, create a .go() shortcut
        if hasattr(self._replica_class, "__call__"):
            call_method = RemoteMethod(self, "__call__")
            # Set both as attributes
            setattr(self, "__call__", call_method)
            # Also set .go as a shortcut to __call__.go
            setattr(self, "go", call_method.go)

        # Get all callable attributes from the class (not instance)
        for method_name in dir(self._replica_class):
            # Skip private methods and special methods (except __call__ handled above)
            if method_name.startswith("_"):
                continue

            # Get the attribute from the class
            attr = getattr(self._replica_class, method_name)

            # Only wrap callable methods (functions, not properties, etc.)
            if callable(attr):
                # Create RemoteMethod proxy
                remote_method = RemoteMethod(self, method_name)
                setattr(self, method_name, remote_method)

    @property
    def replica_handle(self) -> ReplicaHandle:
        """Get the underlying ReplicaHandle.

        Useful for backward compatibility with code that expects ReplicaHandle.
        """
        return self._replica_handle

    @property
    def config(self) -> ReplicaConfig:
        """Get the replica configuration (gpus, backend, etc.)."""
        return self._config

    @property
    def uuid(self) -> UUID:
        """Get the unique ID for this actor handle."""
        return self._uuid

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"ActorHandle({self._replica_class.__name__}, uuid={self._uuid})"

    def __hash__(self):
        """Hash by UUID for unique identity."""
        return hash(self._uuid)
