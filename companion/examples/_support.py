"""Shared fixture construction only; accounting and models live in the package."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable

from prediction_market_lab import Contract, FeeSchedule, Level, OrderBook, PaperBroker, Side
from prediction_market_lab.data import load_recurring
from prediction_market_lab.validation import parse_utc

NOW = parse_utc("2025-07-01T12:00:00Z")
FEES = FeeSchedule(Decimal("0.01"))


def contract(name: str = "yes", *, event: str = "classroom-event", rules: str = "classroom-v1") -> Contract:
    return Contract(name, event, rules)


def book(name: str = "yes", *, bid: str = "0.38", ask: str = "0.40",
         bid_size: int = 100, ask_size: int = 100, snapshot: str = "toy-book",
         age_seconds: int = 0) -> OrderBook:
    return OrderBook(snapshot, name, NOW - timedelta(seconds=age_seconds),
                     (Level(Decimal(bid), bid_size),), (Level(Decimal(ask), ask_size),))


def canonical_position() -> PaperBroker:
    broker = PaperBroker("10.00", FEES)
    broker.register(contract())
    order = broker.submit("yes", Side.BUY, 10, Decimal("0.40"), NOW)
    broker.take(order, book(), NOW)
    return broker


def recurring():
    return load_recurring(Path(__file__).resolve().parents[1] / "data" / "fixtures" / "recurring_events.csv")


def rejected(operation: Callable) -> dict[str, str]:
    try:
        operation()
    except ValueError as exc:
        return {"status": "REJECTED", "reason": str(exc)}
    raise AssertionError("negative teaching case unexpectedly succeeded")
