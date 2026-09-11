"""Cash risk budgets and explicitly conditional binary sizing illustrations."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

from .contracts import FeeSchedule, OrderBook, Side, executable_limit, sweep_cost
from .validation import ZERO, decimal, money, notional, price, probability, quantity


@dataclass(frozen=True)
class RiskBudget:
    max_order_cost: Decimal
    max_committed_cash: Decimal

    def __post_init__(self) -> None:
        for name in ("max_order_cost", "max_committed_cash"):
            value = money(getattr(self, name))
            if value < 0:
                raise ValueError("risk budget cannot be negative")
            object.__setattr__(self, name, value)

    def permits(self, cost: Decimal, available_cash: Decimal, committed_cash: Decimal) -> bool:
        cost, available_cash, committed_cash = map(money, (cost, available_cash, committed_cash))
        if min(cost, available_cash, committed_cash) < 0:
            raise ValueError("negative risk state")
        return cost <= min(self.max_order_cost, available_cash) and cost + committed_cash <= self.max_committed_cash


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    quantity: int
    executable_cost: Decimal | None
    modeled_expected_profit: Decimal | None


def screen_buy_yes(model_probability: float, book: OrderBook, qty: int, fees: FeeSchedule,
                   budget: RiskBudget, available_cash: Decimal, committed_cash: Decimal,
                   *, minimum_modeled_profit: Decimal = ZERO) -> Decision:
    """Classroom binary USD 1 hold-to-settlement screen, not empirical evidence.

    Caller must separately check freshness, rule certainty and data provenance.
    Spread is already included by using executable asks; it is not deducted again.
    """
    p, qty = probability(model_probability), quantity(qty)
    threshold = decimal(minimum_modeled_profit)
    if threshold < 0:
        raise ValueError("minimum modeled profit cannot be negative")
    try:
        cost = sweep_cost(book, Side.BUY, qty, fees)
    except ValueError as exc:
        return Decision("NO_TRADE", str(exc), 0, None, None)
    expected = Decimal(str(p)) * qty - cost
    if expected <= threshold:
        return Decision("NO_TRADE", "modeled edge does not exceed the stated cost buffer", 0, cost, expected)
    reserve = fees.buy_reserve(qty, executable_limit(book, Side.BUY, qty))
    if not budget.permits(max(cost, reserve), available_cash, committed_cash):
        return Decision("NO_TRADE", "cash/reservation or loss budget fails", 0, cost, expected)
    return Decision("PAPER_CANDIDATE_ONLY", "model-conditional arithmetic, not live-edge evidence",
                    qty, cost, expected)


def fractional_kelly(p: float, effective_unit_cost: float, fraction: float = 0.25) -> float:
    """Fraction of bankroll SPENT, not units or payout exposure.

    Assumes a binary USD 1 payout, known p, all-in constant cost c in (0,1),
    divisible units, no exit/settlement fee and no correlated other holdings.
    Repeated bets, parameter uncertainty and affordable loss require separate work.
    """
    p, c, fraction = map(probability, (p, effective_unit_cost, fraction))
    if not 0 < c < 1:
        raise ValueError("effective binary cost must be strictly between zero and one")
    return fraction * max(0.0, min(1.0, (p - c) / (1 - c)))


@dataclass(frozen=True)
class Exposure:
    contract_id: str
    quantity: int
    paid_cost: Decimal

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract identifier required")
        quantity(self.quantity)
        object.__setattr__(self, "paid_cost", money(self.paid_cost))
        if self.paid_cost < 0:
            raise ValueError("negative paid cost")


def scenario_stress(exposures: Sequence[Exposure],
                    scenarios: Mapping[str, Mapping[str, Decimal]]) -> dict[str, Decimal]:
    if not exposures or not scenarios:
        raise ValueError("nonempty exposures and explicitly joint scenarios required")
    total_cost = sum((item.paid_cost for item in exposures), ZERO)
    result = {}
    for name, payouts in scenarios.items():
        if not name or any(item.contract_id not in payouts for item in exposures):
            raise ValueError("scenario must name every held contract")
        receipts = sum((notional(item.quantity, price(payouts[item.contract_id])) for item in exposures), ZERO)
        result[name] = receipts - total_cost
    return result
