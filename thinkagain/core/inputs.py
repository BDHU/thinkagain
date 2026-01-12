"""Input bundling utilities for pure pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass
from typing import Any

from .traceable import trace
from .tracing import node


def make_inputs(name: str = "Inputs", **fields) -> type:
    """Create a traced dataclass for pipeline inputs.

    Args:
        name: Name for the generated class (default: "Inputs")
        **fields: field_name=type pairs

    Returns:
        A traced dataclass type with the specified fields

    Example:
        >>> Inputs = ta.make_inputs(query=str, llm=ta.ReplicaHandle)
        >>> inputs = Inputs(query="test", llm=llm_handle)
    """
    fields_list = [(fname, ftype) for fname, ftype in fields.items()]
    cls = make_dataclass(name, fields_list, frozen=False)
    return trace(cls)


@trace
@dataclass
class Bundle:
    """Lightweight traced container for pipeline inputs.

    Use with runtime operations via top-level helpers:
        small = await ta.subset(inputs, 'query', 'llm')
        enriched = await ta.extend(inputs, docs=docs)
    """

    _data: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize bundle with keyword arguments."""
        object.__setattr__(self, "_data", kwargs)

    def __getattr__(self, name: str) -> Any:
        """Access bundled values as attributes."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name not in self._data:
            raise AttributeError(
                f"Bundle has no attribute '{name}'. "
                f"Available: {', '.join(self._data.keys())}"
            )
        return self._data[name]

    def __getitem__(self, key: str) -> Any:
        """Access bundled values as dict items."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get bundled value with optional default."""
        return self._data.get(key, default)

    def keys(self):
        """Return bundle keys."""
        return self._data.keys()

    def decompose(self) -> tuple[list[Any], dict[str, Any]]:
        """Decompose bundle into traceable children."""
        keys = tuple(sorted(self._data.keys()))
        children = [self._data[k] for k in keys]
        return children, {"keys": keys}

    @classmethod
    def compose(cls, aux: dict[str, Any], children: list[Any]) -> Bundle:
        """Reconstruct bundle from traced children."""
        kwargs = dict(zip(aux["keys"], children))
        return cls(**kwargs)

    def __repr__(self) -> str:
        """String representation showing bundled values."""
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self._data.items()))
        return f"Bundle({items})"


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert Bundle or dataclass to dict."""
    if isinstance(obj, Bundle):
        return obj._data
    elif is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    else:
        raise TypeError(f"Expected Bundle or dataclass, got {type(obj).__name__}")


class BundleOps:
    """Namespace for Bundle operations that work with Bundle or dataclass instances."""

    @staticmethod
    @node
    async def subset(bundle: Any, *keys: str) -> Bundle:
        """Subset a bundle or dataclass (runtime operation, creates graph node).

        Works with both Bundle instances and traced dataclasses.
        Raises KeyError if any requested key is missing.

        Example:
            >>> retrieval = await ta.subset(inputs, 'query', 'db')
        """
        data = _to_dict(bundle)
        missing = [k for k in keys if k not in data]
        if missing:
            missing_list = ", ".join(missing)
            available = ", ".join(sorted(data.keys()))
            raise KeyError(
                f"Missing bundle keys: {missing_list}. Available: {available}"
            )
        return Bundle(**{k: data[k] for k in keys})

    @staticmethod
    @node
    async def extend(bundle: Any, **fields: Any) -> Bundle:
        """Extend a bundle or dataclass with new fields (runtime operation).

        Works with both Bundle instances and traced dataclasses.

        Example:
            >>> enriched = await ta.extend(inputs, docs=docs)
        """
        data = _to_dict(bundle)
        return Bundle(**{**data, **fields})

    @staticmethod
    @node
    async def replace(bundle: Any, **updates: Any) -> Bundle:
        """Replace fields in a bundle or dataclass (runtime operation).

        Works with both Bundle instances and traced dataclasses.

        Example:
            >>> updated = await ta.replace(inputs, query="new")
        """
        data = _to_dict(bundle)
        return Bundle(**{**data, **updates})

    @staticmethod
    @node
    async def get(bundle: Any, key: str, default: Any = None) -> Any:
        """Get a single field from bundle or dataclass (runtime operation).

        Works with both Bundle instances and traced dataclasses.

        Example:
            >>> query = await ta.get(inputs, 'query')
        """
        data = _to_dict(bundle)
        return data.get(key, default)

    @staticmethod
    @node
    async def unpack(bundle: Any, *keys: str) -> tuple[Any, ...]:
        """Unpack multiple fields from bundle or dataclass (runtime operation).

        Returns values as a tuple in the order keys were specified.
        Works with both Bundle instances and traced dataclasses.
        Raises KeyError if any requested key is missing.

        Example:
            >>> query, db = await ta.unpack(inputs, 'query', 'db')
        """
        data = _to_dict(bundle)
        missing = [k for k in keys if k not in data]
        if missing:
            missing_list = ", ".join(missing)
            available = ", ".join(sorted(data.keys()))
            raise KeyError(
                f"Missing bundle keys: {missing_list}. Available: {available}"
            )
        return tuple(data[k] for k in keys)


# Create singleton namespace
bundle = BundleOps()
