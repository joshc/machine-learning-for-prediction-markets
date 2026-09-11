"""Explicit numeric grids and UTC availability checks."""

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
from math import isfinite

CENT = Decimal("0.01")
PRICE_TICK = Decimal("0.0001")
ZERO = Decimal("0")
ONE = Decimal("1")


def decimal(value: Decimal | str | int) -> Decimal:
    """Reject floats: cash inputs must have an unambiguous decimal spelling."""
    if isinstance(value, bool) or not isinstance(value, (Decimal, str, int)):
        raise ValueError("use Decimal, a decimal string, or an integer, not float")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("decimal must be finite")
    return result


def money(value: Decimal | str | int) -> Decimal:
    result = decimal(value)
    try:
        rounded = result.quantize(CENT)
    except InvalidOperation as exc:
        raise ValueError("cash exceeds the supported decimal range") from exc
    if result != rounded:
        raise ValueError("cash must be an exact number of cents")
    return rounded


def price(value: Decimal | str | int) -> Decimal:
    result = decimal(value)
    if not ZERO <= result <= ONE or result != result.quantize(PRICE_TICK):
        raise ValueError("price/payout must be in [0, 1] on the 0.0001 grid")
    return result


def quantity(value: int, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("quantity must be a whole integer")
    if value < (0 if allow_zero else 1):
        raise ValueError("quantity must be positive" if not allow_zero else "quantity is negative")
    return value


def notional(qty: int, unit_price: Decimal) -> Decimal:
    return (quantity(qty) * price(unit_price)).quantize(CENT, rounding=ROUND_HALF_UP)


def ceil_cash(value: Decimal) -> Decimal:
    return decimal(value).quantize(CENT, rounding=ROUND_CEILING)


def probability(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("probability must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probability must be a finite numeric value") from exc
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    return result


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value


def parse_utc(value: str) -> datetime:
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def available(published_at: datetime, observed_at: datetime, cutoff: datetime) -> bool:
    """A publication is usable only after both publication and local observation."""
    return max(utc(published_at), utc(observed_at)) <= utc(cutoff)
