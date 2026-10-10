"""Small helpers to build (nested, frozen) dataclasses from YAML/dicts."""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml

T = TypeVar("T")


def _convert(typ: Any, value: Any) -> Any:
    origin = get_origin(typ)
    if dataclasses.is_dataclass(typ) and isinstance(value, dict):
        return from_dict(typ, value)  # type: ignore[arg-type]
    if origin is tuple:
        args = get_args(typ)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        return tuple(_convert(a, v) for a, v in zip(args, value, strict=True))
    if origin in (typing.Union, types.UnionType):
        for arg in get_args(typ):
            if arg is type(None) and value is None:
                return None
            if dataclasses.is_dataclass(arg) and isinstance(value, dict):
                return from_dict(arg, value)  # type: ignore[arg-type]
        return value
    if typ is float and isinstance(value, int):
        return float(value)
    return value


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Instantiate dataclass ``cls`` from ``data``; unknown keys raise ``ValueError``."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {name: _convert(hints[name], value) for name, value in data.items()}
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Convert (nested) dataclasses to plain dicts/lists suitable for JSON/YAML."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple | list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(cls: type[T], path: str | Path | None) -> T:
    return cls() if path is None else from_dict(cls, load_yaml(path))


def replace(obj: T, **changes: Any) -> T:
    return dataclasses.replace(obj, **changes)  # type: ignore[type-var]
