"""Strict JSON output: non-finite numbers become null and NaN can never be written."""

import json
import math

import numpy as np


def finite(value):
    """A JSON-ready copy of value: NaN and infinities become None, numpy values and sets plain Python.

    Args:
        value: Nested dicts, lists, tuples, sets, numbers, strings, numpy scalars or arrays.

    Returns:
        The same structure with only JSON types and finite numbers.
    """
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [finite(item) for item in sorted(value)]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value, **kwargs) -> str:
    """json.dumps of finite(value) with allow_nan=False.

    Args:
        value: What to serialise.
        **kwargs: Passed on to json.dumps (indent, separators).

    Returns:
        The JSON text.
    """
    return json.dumps(finite(value), allow_nan=False, **kwargs)
