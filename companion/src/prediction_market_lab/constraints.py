"""Explicit payoff compatibility checks, not title-based claims of arbitrage."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping, Sequence

from .contracts import Contract, FeeSchedule, OrderBook, Side, executable_limit, sweep_cost
from .validation import ZERO, money, notional, price, quantity


@dataclass(frozen=True)
class BasketLeg:
    contract: Contract
    book: OrderBook
    payouts: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        if self.contract.contract_id != self.book.contract_id:
            raise ValueError("leg contract does not match executable book")
        if not self.payouts or any(not state for state in self.payouts):
            raise ValueError("an explicitly complete state space is required")
        object.__setattr__(self, "payouts", {state: price(payout) for state, payout in self.payouts.items()})


@dataclass(frozen=True)
class BasketResult:
    status: str
    reason: str
    quantity_each: int
    total_cost: Decimal | None
    minimum_payout: Decimal | None
    minimum_net_profit: Decimal | None
    leg_risk: str = "non-atomic legs can fail; quoted basket is not guaranteed execution"
    required_cash: Decimal | None = None


def _compatible(legs: Sequence[BasketLeg], compatibility_verified: bool) -> bool:
    if compatibility_verified is not True or not legs:
        return False
    first = legs[0]
    return all((leg.contract.event_id, leg.contract.rules_id, leg.contract.currency, set(leg.payouts))
               == (first.contract.event_id, first.contract.rules_id, first.contract.currency, set(first.payouts))
               for leg in legs)


def scan_basket(legs: Sequence[BasketLeg], qty: int, fees: FeeSchedule, capital: Decimal,
                now: datetime, *, compatibility_verified: bool,
                max_quote_age: timedelta = timedelta(seconds=5)) -> BasketResult:
    quantity(qty)
    capital = money(capital)
    if capital < 0:
        raise ValueError("negative capital")
    if len(legs) < 2 or len({leg.contract.contract_id for leg in legs}) != len(legs):
        raise ValueError("basket requires at least two distinct contracts")
    if not _compatible(legs, compatibility_verified):
        return BasketResult("REJECTED", "unverified or mismatched events, rules, currency or state space",
                            0, None, None, None)
    try:
        for leg in legs:
            leg.book.validate_at(now, max_quote_age)
        cost = sum((sweep_cost(leg.book, Side.BUY, qty, fees) for leg in legs), ZERO)
        reserve = sum(
            (fees.buy_reserve(qty, executable_limit(leg.book, Side.BUY, qty)) for leg in legs), ZERO,
        )
    except ValueError as exc:
        return BasketResult("NO_TRADE", str(exc), 0, None, None, None)
    payouts = {state: sum((notional(qty, leg.payouts[state]) for leg in legs), ZERO)
               for state in legs[0].payouts}
    minimum_payout = min(payouts.values())
    profit = minimum_payout - cost
    required_cash = max(cost, reserve)
    if required_cash > capital:
        return BasketResult(
            "NO_TRADE", "funded capital cannot reserve all basket orders before execution",
            0, cost, minimum_payout, profit, required_cash=required_cash,
        )
    if profit <= 0:
        return BasketResult("NO_TRADE", "no positive payoff floor after executable costs",
                            0, cost, minimum_payout, profit, required_cash=required_cash)
    return BasketResult("PAPER_CANDIDATE_ONLY", "conditional on complete verified payout states and all legs filling",
                        qty, cost, minimum_payout, profit, required_cash=required_cash)


def payoff_dominates(superset: BasketLeg, subset: BasketLeg, *, compatibility_verified: bool) -> bool:
    if not _compatible((superset, subset), compatibility_verified):
        raise ValueError("cannot compare unverified payoff maps")
    return all(superset.payouts[state] >= subset.payouts[state] for state in superset.payouts)
