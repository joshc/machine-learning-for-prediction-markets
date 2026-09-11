"""Venue-neutral USD classroom contracts, finite books and invented fee schedules."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from .validation import ZERO, ceil_cash, decimal, notional, price, quantity, utc


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Contract:
    contract_id: str
    event_id: str
    rules_id: str
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.contract_id, self.event_id, self.rules_id)):
            raise ValueError("contract, event and rules identifiers are required")
        if self.currency != "USD":
            raise ValueError("this classroom ledger supports USD only")


@dataclass(frozen=True)
class Settlement:
    contract_id: str
    payout: Decimal
    settled_at: datetime
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payout", price(self.payout))
        utc(self.settled_at)
        if not self.contract_id or not self.reason.strip():
            raise ValueError("settlement must identify the contract and reason")

    @property
    def binary_label(self) -> int | None:
        """Exceptional fractional payouts are not fractional binary ML labels."""
        return int(self.payout) if self.payout in (0, 1) else None


@dataclass(frozen=True)
class FeeSchedule:
    """Invented flat-unit plus notional-rate fee, rounded UP once per fill.

    No venue fee schedule is represented. Splitting fills can increase fees.
    """

    per_unit: Decimal = ZERO
    notional_rate: Decimal = ZERO
    description: str = "invented classroom fee, not a venue schedule"

    def __post_init__(self) -> None:
        for name in ("per_unit", "notional_rate"):
            value = decimal(getattr(self, name))
            if value < 0:
                raise ValueError("fees cannot be negative")
            object.__setattr__(self, name, value)
        if not self.description.strip():
            raise ValueError("fee provenance is required")

    def total(self, qty: int, unit_price: Decimal) -> Decimal:
        return ceil_cash(quantity(qty) * (self.per_unit + price(unit_price) * self.notional_rate))

    def buy_reserve(self, qty: int, limit: Decimal) -> Decimal:
        """Worst-case cash for one-unit fills, including per-fill rounding."""
        return quantity(qty) * (ceil_cash(price(limit)) + self.total(1, limit))


@dataclass(frozen=True)
class Level:
    price: Decimal
    quantity: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", price(self.price))
        quantity(self.quantity)


@dataclass(frozen=True)
class OrderBook:
    snapshot_id: str
    contract_id: str
    observed_at: datetime
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    provenance: str = "explicitly synthetic classroom snapshot"

    def __post_init__(self) -> None:
        utc(self.observed_at)
        if not self.snapshot_id or not self.contract_id or not self.provenance.strip():
            raise ValueError("snapshot identity, contract and provenance are required")
        for name, reverse in (("bids", True), ("asks", False)):
            levels = tuple(getattr(self, name))
            if any(not isinstance(level, Level) for level in levels):
                raise ValueError("book sides require Level values")
            prices = [level.price for level in levels]
            if prices != sorted(set(prices), reverse=reverse):
                raise ValueError("book levels must be unique and sorted best first")
            object.__setattr__(self, name, levels)
        if self.bids and self.asks and self.bids[0].price > self.asks[0].price:
            raise ValueError("crossed book is ambiguous")

    def validate_at(self, now: datetime, max_age: timedelta) -> None:
        utc(now)
        if max_age < timedelta(0):
            raise ValueError("maximum quote age cannot be negative")
        if not timedelta(0) <= now - self.observed_at <= max_age:
            raise ValueError("stale or future quote")

    def executable(self, side: Side) -> tuple[Level, ...]:
        if not isinstance(side, Side):
            raise ValueError("use Side.BUY or Side.SELL")
        return self.asks if side is Side.BUY else self.bids

    def midpoint(self) -> Decimal:
        if not self.bids or not self.asks:
            raise ValueError("midpoint needs both sides")
        return (self.bids[0].price + self.asks[0].price) / 2


def sweep_cost(book: OrderBook, side: Side, qty: int, fees: FeeSchedule) -> Decimal:
    """Full-size gross cash outlay (buy) or net receipt (sell), one fill per level.

    Stateless candidate calculation, not a fill. Raises if displayed depth is short.
    """
    remaining, total = quantity(qty), ZERO
    for level in book.executable(side):
        take = min(remaining, level.quantity)
        gross, fee = notional(take, level.price), fees.total(take, level.price)
        if side is Side.SELL and fee > gross:
            raise ValueError("sell proceeds do not cover fees")
        total += gross + fee if side is Side.BUY else gross - fee
        remaining -= take
        if remaining == 0:
            return total
    raise ValueError("insufficient executable depth for the requested full size")


def executable_limit(book: OrderBook, side: Side, qty: int) -> Decimal:
    """Worst displayed price needed for the requested full size."""
    remaining = quantity(qty)
    for level in book.executable(side):
        remaining -= level.quantity
        if remaining <= 0:
            return level.price
    raise ValueError("insufficient executable depth for the requested full size")
