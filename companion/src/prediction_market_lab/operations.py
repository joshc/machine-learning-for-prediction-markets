"""Fail-closed checks for a paper decision; never a funded-order client."""

from dataclasses import dataclass
from datetime import datetime

from .accounting import PaperBroker
from .contracts import OrderBook
from .risk import Decision, RiskBudget, screen_buy_yes


@dataclass(frozen=True)
class PaperInputs:
    contract_id: str
    probability: float | None
    book: OrderBook | None
    rules_verified: bool
    features_available: bool
    fee_assumptions_explicit: bool


def paper_decision(broker: PaperBroker, inputs: PaperInputs, budget: RiskBudget,
                   qty: int, now: datetime) -> Decision:
    if (inputs.rules_verified is not True or inputs.features_available is not True
            or inputs.fee_assumptions_explicit is not True):
        return Decision("NO_TRADE", "ambiguous rules, unavailable features or unspecified fees", 0, None, None)
    if inputs.book is None or inputs.probability is None:
        return Decision("NO_TRADE", "missing quote or probability", 0, None, None)
    if inputs.book.contract_id != inputs.contract_id:
        return Decision("NO_TRADE", "quote contract mismatch", 0, None, None)
    try:
        remaining_book = broker.remaining_book(inputs.book, now)
        state = broker.reconcile()
        committed = state["open_cost_basis"] + broker.reserved_cash
        return screen_buy_yes(inputs.probability, remaining_book, qty, broker.fees, budget,
                              broker.available_cash, committed)
    except (ValueError, AssertionError) as exc:
        return Decision("NO_TRADE", f"monitor stopped: {exc}", 0, None, None)
