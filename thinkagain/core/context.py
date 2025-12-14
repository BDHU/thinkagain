from typing import Any, List, Optional


class Context:
    """
    State container that passes through executables.

    Provides dictionary-like attribute access and execution history tracking.

    Example:
        ctx = Context(query="What is ML?", top_k=5)
        ctx.documents = ["doc1", "doc2"]
        ctx.log("Retrieved documents")
    """

    def __init__(self, _history: Optional[List[str]] = None, **kwargs):
        object.__setattr__(self, "_data", kwargs)
        object.__setattr__(self, "_history", _history if _history is not None else [])

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def log(self, message: str):
        self._history.append(message)

    @property
    def history(self) -> List[str]:
        return self._history.copy()

    @property
    def data(self) -> dict:
        return self._data.copy()

    def copy(self) -> "Context":
        """Shallow copy of context with independent data and history."""
        return Context(_history=self._history.copy(), **self._data.copy())

    def __repr__(self) -> str:
        return f"Context({self._data})"

    def __str__(self) -> str:
        return f"Context with {len(self._data)} fields: {list(self._data.keys())}"
