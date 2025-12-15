from typing import Any, Dict, Iterable, Iterator, List, Optional


class Context:
    """
    State container that passes through executables.

    Provides dictionary-like attribute access and execution history tracking.

    Accessing a missing attribute raises ``AttributeError``; use ``get()`` for
    optional values.

    Example:
        ctx = Context(query="What is ML?", top_k=5)
        ctx.documents = ["doc1", "doc2"]
        ctx.log("Retrieved documents")
    """

    def __init__(self, _history: Optional[List[str]] = None, **kwargs: Any):
        object.__setattr__(self, "_data", dict(kwargs))
        object.__setattr__(self, "_history", list(_history) if _history is not None else [])

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._data.items()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def log(self, message: str) -> None:
        self._history.append(message)

    @property
    def history(self) -> List[str]:
        return self._history.copy()

    @property
    def data(self) -> Dict[str, Any]:
        return self._data.copy()

    def copy(self) -> "Context":
        """Shallow copy of context with independent data and history."""
        return Context(_history=self._history.copy(), **self._data.copy())

    def __repr__(self) -> str:
        return f"Context({self._data})"

    def __str__(self) -> str:
        return f"Context with {len(self._data)} fields: {list(self._data.keys())}"
