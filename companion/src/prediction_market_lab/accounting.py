"""Conservative, in-memory paper ledger. Explicit fills only; never live orders."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from .contracts import Contract, FeeSchedule, Level, OrderBook, Settlement, Side
from .validation import CENT, ZERO, money, notional, price, quantity, utc


@dataclass
class _Order:
    order_id: str
    contract_id: str
    side: Side
    quantity: int
    limit: Decimal
    submitted_at: datetime
    filled: int = 0
    cancelled: bool = False

    @property
    def remaining(self) -> int:
        return 0 if self.cancelled else self.quantity - self.filled


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    contract_id: str
    side: Side
    quantity: int
    limit: Decimal
    filled: int
    remaining: int
    cancelled: bool


@dataclass(frozen=True)
class Fill:
    order_id: str
    contract_id: str
    side: Side
    quantity: int
    price: Decimal
    notional: Decimal
    fee: Decimal
    filled_at: datetime
    evidence: str


@dataclass(frozen=True)
class LedgerEntry:
    at: datetime
    kind: str
    contract_id: str
    cash_delta: Decimal
    inventory_delta: int


@dataclass(frozen=True)
class Mark:
    cash: Decimal
    inventory_liquidation_value: Decimal
    equity: Decimal
    realized_profit: Decimal
    marked_profit: Decimal
    unrealized_profit: Decimal


class PaperBroker:
    """Whole-unit, long-only paper orders with FIFO cent-denominated cost basis.

    Taker fills consume a finite identified snapshot. Resting quote touches do
    nothing. Explicit user-supplied synthetic maker fills must obey limits.
    Cash notionals round half-up per fill; fees round up per fill. Buy reserves
    assume the worst case of separate one-unit fills. No borrowing, shorting,
    settlement fee, interest, or automatic queue/fill inference is implemented.
    """

    def __init__(self, cash: Decimal | str | int, fees: FeeSchedule,
                 *, max_quote_age: timedelta = timedelta(seconds=5)) -> None:
        self.initial_cash = money(cash)
        if self.initial_cash < 0 or max_quote_age < timedelta(0):
            raise ValueError("cash and quote age must be nonnegative")
        self.fees, self.max_quote_age = fees, max_quote_age
        self._cash = self.initial_cash
        self._contracts: dict[str, Contract] = {}
        self._orders: dict[str, _Order] = {}
        self._inventory: dict[str, int] = {}
        self._lots: dict[str, list[list[int | Decimal]]] = {}
        self._fills: list[Fill] = []
        self._ledger: list[LedgerEntry] = []
        self._settled: set[str] = set()
        self._snapshots: dict[str, OrderBook] = {}
        self._used_depth: dict[tuple[str, Side, int], int] = {}
        self._last_at: datetime | None = None
        self._realized = ZERO

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def ledger(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._ledger)

    @property
    def realized_profit(self) -> Decimal:
        return self._realized

    @property
    def reserved_cash(self) -> Decimal:
        return sum((self.fees.buy_reserve(o.remaining, o.limit)
                    for o in self._orders.values() if o.side is Side.BUY and o.remaining), ZERO)

    @property
    def available_cash(self) -> Decimal:
        return self.cash - self.reserved_cash

    def inventory(self, contract_id: str) -> int:
        return self._inventory.get(contract_id, 0)

    def register(self, contract: Contract) -> None:
        if contract.contract_id in self._contracts and self._contracts[contract.contract_id] != contract:
            raise ValueError("contract metadata cannot silently change")
        self._contracts[contract.contract_id] = contract

    def status(self, order_id: str) -> OrderStatus:
        o = self._orders[order_id]
        return OrderStatus(o.order_id, o.contract_id, o.side, o.quantity, o.limit,
                           o.filled, o.remaining, o.cancelled)

    def _time(self, now: datetime) -> None:
        utc(now)
        if self._last_at is not None and now < self._last_at:
            raise ValueError("ledger timestamps cannot go backwards")

    def _open_contract(self, contract_id: str) -> None:
        if contract_id not in self._contracts or contract_id in self._settled:
            raise ValueError("unknown or already settled contract")

    def validate_book(self, book: OrderBook, now: datetime) -> None:
        """Validate a paper input without modifying the ledger."""
        self._time(now)
        self._open_contract(book.contract_id)
        book.validate_at(now, self.max_quote_age)
        old = self._snapshots.get(book.snapshot_id)
        if old is not None and old != book:
            raise ValueError("snapshot identity reused for different data")

    def _remaining_depth(self, book: OrderBook, side: Side, index: int) -> int:
        return (book.executable(side)[index].quantity
                - self._used_depth.get((book.snapshot_id, side, index), 0))

    def remaining_book(self, book: OrderBook, now: datetime) -> OrderBook:
        """Read-only screening view; execution must receive the original snapshot."""
        self.validate_book(book, now)

        def remaining(side: Side) -> tuple[Level, ...]:
            levels = []
            for index, level in enumerate(book.executable(side)):
                depth = self._remaining_depth(book, side, index)
                if depth:
                    levels.append(Level(level.price, depth))
            return tuple(levels)

        return replace(book, bids=remaining(Side.SELL), asks=remaining(Side.BUY))

    def submit(self, contract_id: str, side: Side, qty: int, limit: Decimal,
               now: datetime) -> str:
        self._time(now)
        self._open_contract(contract_id)
        quantity(qty)
        limit = price(limit)
        if not isinstance(side, Side):
            raise ValueError("use Side.BUY or Side.SELL")
        if side is Side.BUY:
            if self.fees.buy_reserve(qty, limit) > self.available_cash:
                raise ValueError("insufficient unreserved funded cash")
        else:
            reserved_units = sum(o.remaining for o in self._orders.values()
                                 if o.contract_id == contract_id and o.side is Side.SELL)
            if qty > self.inventory(contract_id) - reserved_units:
                raise ValueError("naked or doubly reserved sell rejected")
        order_id = f"paper-{len(self._orders) + 1:04d}"
        self._orders[order_id] = _Order(order_id, contract_id, side, qty, limit, now)
        self._last_at = now
        return order_id

    def cancel(self, order_id: str, now: datetime) -> None:
        self._time(now)
        order = self._orders[order_id]
        if not order.remaining:
            raise ValueError("order is already closed")
        order.cancelled = True
        self._last_at = now
        self._ledger.append(LedgerEntry(now, "cancel", order.contract_id, ZERO, 0))

    def _fill(self, order_id: str, qty: int, unit_price: Decimal,
              now: datetime, evidence: str) -> Fill:
        self._time(now)
        order = self._orders[order_id]
        self._open_contract(order.contract_id)
        quantity(qty)
        unit_price = price(unit_price)
        if qty > order.remaining or now < order.submitted_at:
            raise ValueError("closed order, overfill or fill before submission")
        if ((order.side is Side.BUY and unit_price > order.limit)
                or (order.side is Side.SELL and unit_price < order.limit)):
            raise ValueError("fill violates limit")
        gross, fee = notional(qty, unit_price), self.fees.total(qty, unit_price)
        cash_delta = -gross - fee if order.side is Side.BUY else gross - fee
        if order.side is Side.SELL and cash_delta < 0:
            raise ValueError("sell proceeds do not cover fees")
        release = (self.fees.buy_reserve(qty, order.limit) if order.side is Side.BUY else ZERO)
        if self.cash + cash_delta < self.reserved_cash - release:
            raise ValueError("fill would consume another order's funded reserve")
        if order.side is Side.BUY:
            self._lots.setdefault(order.contract_id, []).append([qty, -cash_delta])
            inventory_delta = qty
        else:
            if qty > self.inventory(order.contract_id):
                raise ValueError("insufficient inventory")
            self._realized += cash_delta - self._consume_basis(order.contract_id, qty)
            inventory_delta = -qty
        order.filled += qty
        self._cash += cash_delta
        self._inventory[order.contract_id] = self.inventory(order.contract_id) + inventory_delta
        fill = Fill(order_id, order.contract_id, order.side, qty, unit_price, gross, fee, now, evidence)
        self._fills.append(fill)
        self._ledger.append(LedgerEntry(now, "fill", order.contract_id, cash_delta, inventory_delta))
        self._last_at = now
        self.reconcile()
        return fill

    def synthetic_fill(self, order_id: str, qty: int, unit_price: Decimal,
                       now: datetime, *, evidence: str) -> Fill:
        if not evidence.strip():
            raise ValueError("an explicitly supplied synthetic fill needs an explanation")
        return self._fill(order_id, qty, unit_price, now, f"USER-SUPPLIED SYNTHETIC: {evidence}")

    def take(self, order_id: str, book: OrderBook, now: datetime) -> tuple[Fill, ...]:
        """Execute against finite visible depth under an instantaneous paper assumption."""
        self.validate_book(book, now)
        order = self._orders[order_id]
        self._open_contract(order.contract_id)
        if book.contract_id != order.contract_id or not order.remaining:
            raise ValueError("wrong book or closed order")
        self._snapshots[book.snapshot_id] = book
        fills = []
        for index, level in enumerate(book.executable(order.side)):
            if not order.remaining:
                break
            if ((order.side is Side.BUY and level.price > order.limit)
                    or (order.side is Side.SELL and level.price < order.limit)):
                break
            key = (book.snapshot_id, order.side, index)
            remaining_depth = self._remaining_depth(book, order.side, index)
            qty = min(order.remaining, remaining_depth)
            if qty:
                fills.append(self._fill(order_id, qty, level.price, now,
                                        f"finite snapshot {book.snapshot_id}; paper latency assumption"))
                self._used_depth[key] = self._used_depth.get(key, 0) + qty
        self._last_at = now
        return tuple(fills)

    def observe_resting(self, order_id: str, book: OrderBook, now: datetime) -> str:
        self.validate_book(book, now)
        order = self._orders[order_id]
        if order.contract_id != book.contract_id or not order.remaining:
            raise ValueError("wrong book or closed order")
        self._last_at = now
        return "NO_INFERRED_FILL: quotes alone do not establish queue position or a trade"

    def _consume_basis(self, contract_id: str, qty: int) -> Decimal:
        remaining, basis = qty, ZERO
        lots = self._lots[contract_id]
        while remaining:
            lot_qty, lot_cost = lots[0]
            take = min(remaining, int(lot_qty))
            cost = (lot_cost if take == lot_qty else
                    (lot_cost * take / lot_qty).quantize(CENT, rounding=ROUND_HALF_UP))
            basis += cost
            remaining -= take
            if take == lot_qty:
                lots.pop(0)
            else:
                lots[0] = [int(lot_qty) - take, lot_cost - cost]
        return basis

    def settle(self, settlement: Settlement) -> None:
        self._time(settlement.settled_at)
        self._open_contract(settlement.contract_id)
        contract_id = settlement.contract_id
        qty = self.inventory(contract_id)
        receipt = notional(qty, settlement.payout) if qty else ZERO
        basis = self._consume_basis(contract_id, qty) if qty else ZERO
        self._realized += receipt - basis
        self._cash += receipt
        self._inventory[contract_id] = 0
        for order in self._orders.values():
            if order.contract_id == contract_id and order.remaining:
                order.cancelled = True
        self._settled.add(contract_id)
        self._ledger.append(LedgerEntry(settlement.settled_at, "settlement",
                                        contract_id, receipt, -qty))
        self._last_at = settlement.settled_at
        self.reconcile()

    def mark(self, books: dict[str, OrderBook], now: datetime) -> Mark:
        """Conservative full liquidation bid mark, net of exit fees.

        Reject missing, stale, wrong-contract or insufficient remaining bid depth.
        This hypothetical liquidation does not consume depth or realize P&L.
        """
        self._time(now)
        liquidation = ZERO
        for contract_id, qty in self._inventory.items():
            if not qty:
                continue
            if contract_id not in books:
                raise ValueError("missing inventory mark")
            book = books[contract_id]
            if book.contract_id != contract_id:
                raise ValueError("wrong contract in inventory mark")
            book.validate_at(now, self.max_quote_age)
            if book.snapshot_id in self._snapshots and self._snapshots[book.snapshot_id] != book:
                raise ValueError("snapshot identity reused for different data")
            remaining = qty
            for index, level in enumerate(book.bids):
                depth = self._remaining_depth(book, Side.SELL, index)
                take = min(remaining, depth)
                if take:
                    proceeds = notional(take, level.price) - self.fees.total(take, level.price)
                    if proceeds < 0:
                        raise ValueError("sell proceeds do not cover liquidation fees")
                    liquidation += proceeds
                    remaining -= take
                if not remaining:
                    break
            if remaining:
                raise ValueError("insufficient bid depth for full liquidation mark")
        equity = self.cash + liquidation
        profit = equity - self.initial_cash
        return Mark(self.cash, liquidation, equity, self.realized_profit,
                    profit, profit - self.realized_profit)

    def reconcile(self) -> dict[str, Decimal | int]:
        expected_cash = self.initial_cash + sum((entry.cash_delta for entry in self._ledger), ZERO)
        if self.cash != expected_cash or self.available_cash < 0:
            raise AssertionError("cash reconciliation failed")
        for contract_id in self._contracts:
            expected = sum(entry.inventory_delta for entry in self._ledger
                           if entry.contract_id == contract_id)
            lot_qty = sum(int(lot[0]) for lot in self._lots.get(contract_id, []))
            if expected != self.inventory(contract_id) or expected != lot_qty or expected < 0:
                raise AssertionError("inventory reconciliation failed")
        if any(not 0 <= o.filled <= o.quantity for o in self._orders.values()):
            raise AssertionError("order overfill")
        basis = sum((lot[1] for lots in self._lots.values() for lot in lots), ZERO)
        if self.cash + basis - self.initial_cash != self.realized_profit:
            raise AssertionError("cost-basis/profit reconciliation failed")
        return {"cash": self.cash, "reserved_cash": self.reserved_cash,
                "available_cash": self.available_cash, "inventory_units": sum(self._inventory.values()),
                "open_cost_basis": basis, "realized_profit": self.realized_profit}
