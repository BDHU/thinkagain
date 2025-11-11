from typing import Any, Optional


class Context:
    """
    A context object that holds state and tracks history as it passes through workers.

    Provides:
    - Dictionary-like attribute access for data
    - History/logging capability for debugging
    - Full transparency of pipeline execution

    Example:
        ctx = Context(query="What is ML?", top_k=5)
        ctx.documents = ["doc1", "doc2"]
        ctx.log("Retrieved documents")
        print(ctx.history)  # See all logs
    """

    def __init__(self, **kwargs):
        """
        Initialize context with keyword arguments.

        Args:
            **kwargs: Initial context data (e.g., query="What is ML?")
        """
        object.__setattr__(self, '_data', kwargs)
        object.__setattr__(self, '_history', [])

    def __getattr__(self, key: str) -> Any:
        """Get attribute from context data."""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any):
        """Set attribute in context data."""
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return self._data.get(key, default)

    def log(self, message: str):
        """Add a log message to history for debugging."""
        self._history.append(message)

    @property
    def history(self) -> list[str]:
        """Get full execution history."""
        return self._history.copy()

    @property
    def data(self) -> dict:
        """Get all context data as dictionary."""
        return self._data.copy()

    def __repr__(self) -> str:
        return f"Context({self._data})"

    def __str__(self) -> str:
        return f"Context with {len(self._data)} fields: {list(self._data.keys())}"

    def copy(self) -> 'Context':
        """
        Create a shallow copy of the context.

        Useful for parallel execution where multiple workers need
        independent copies of the context.

        Returns:
            New Context with copied data and history
        """
        new_ctx = Context(**self._data.copy())
        new_ctx._history = self._history.copy()
        return new_ctx
