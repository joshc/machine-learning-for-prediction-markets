"""Stable strict JSON, including explicit string representations of infinities."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from math import isfinite
from typing import Any


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
        return round(value, 6)
    return value


def emit(title: str, result: Any) -> None:
    print(json.dumps(json_value({"demonstration": title, "data": "SYNTHETIC / ILLUSTRATIVE",
                                "live_edge_claim": False, "result": result}),
                     indent=2, sort_keys=True, allow_nan=False))
