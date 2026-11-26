"""
Encode messages to bytes and back.

The format is compact JSON: `{"type": <message class name>, "body": {...}}`,
with nested models (`Job`, `Status`, ...) as objects of their fields and
tuples as arrays.
"""

from __future__ import annotations

import collections.abc
import json
import types
import typing
from dataclasses import fields, is_dataclass
from functools import cache
from typing import Any

from torchsched.types.messages import Message

_PRIMITIVES = (str, int, float, bool)


def encode(msg: Message) -> bytes:
    if not isinstance(msg, Message) or Message.registry.get(
        type(msg).__name__
    ) is not type(msg):
        raise TypeError(f"not a registered message: {msg!r}")
    payload = {"type": type(msg).__name__, "body": _to_data(msg)}
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()


def decode(data: bytes) -> Message:
    """
    Raises ValueError for anything that is not a valid encoded message.
    """
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"not JSON: {error}") from None
    if not isinstance(payload, dict) or set(payload) != {"type", "body"}:
        raise ValueError("expected an object with exactly 'type' and 'body'")
    cls = (
        Message.registry.get(payload["type"])
        if isinstance(payload["type"], str)
        else None
    )
    if cls is None:
        raise ValueError(f"unknown message type {payload['type']!r}")
    try:
        return _from_data(cls, payload["body"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {cls.__name__}: {error}") from None


def _to_data(value: Any) -> Any:
    if value is None or isinstance(value, _PRIMITIVES):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_to_data(item) for item in value]
    if isinstance(value, collections.abc.Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings")
        return {key: _to_data(item) for key, item in value.items()}
    raise TypeError(f"cannot encode {type(value).__name__}")


@cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _from_data(tp: Any, data: Any) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)

    if tp is type(None):
        if data is not None:
            raise TypeError(f"expected null, got {data!r}")
        return None
    if origin in (typing.Union, types.UnionType):
        if data is None and type(None) in args:
            return None
        options = [arg for arg in args if arg is not type(None)]
        if len(options) != 1:
            raise TypeError(f"unsupported union {tp}")
        return _from_data(options[0], data)
    if tp is bool:
        if not isinstance(data, bool):
            raise TypeError(f"expected a boolean, got {data!r}")
        return data
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise TypeError(f"expected an integer, got {data!r}")
        return data
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise TypeError(f"expected a number, got {data!r}")
        return float(data)
    if tp is str:
        if not isinstance(data, str):
            raise TypeError(f"expected a string, got {data!r}")
        return data
    if origin is tuple:
        if not isinstance(data, list):
            raise TypeError(f"expected an array, got {data!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_data(args[0], item) for item in data)
        if len(data) != len(args):
            raise TypeError(f"expected {len(args)} items, got {len(data)}")
        return tuple(_from_data(arg, item) for arg, item in zip(args, data))
    if origin in (collections.abc.Mapping, dict):
        if not isinstance(data, dict):
            raise TypeError(f"expected an object, got {data!r}")
        key_type, value_type = args
        return {
            _from_data(key_type, key): _from_data(value_type, item)
            for key, item in data.items()
        }
    if isinstance(tp, type) and is_dataclass(tp):
        if not isinstance(data, dict):
            raise TypeError(f"expected an object for {tp.__name__}, got {data!r}")
        hints = _hints(tp)
        names = {f.name for f in fields(tp)}
        if set(data) != names:
            missing, extra = names - set(data), set(data) - names
            raise TypeError(
                f"{tp.__name__} fields: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )
        return tp(**{name: _from_data(hints[name], data[name]) for name in names})
    raise TypeError(f"cannot decode type {tp}")
