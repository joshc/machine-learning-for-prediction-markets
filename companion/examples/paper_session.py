"""Fail-closed paper monitor with a reconciled synthetic finite-depth fill."""

from dataclasses import replace
from decimal import Decimal

from prediction_market_lab import PaperBroker, Side
from prediction_market_lab.operations import PaperInputs, paper_decision
from prediction_market_lab.reporting import emit
from prediction_market_lab.risk import RiskBudget

from _support import FEES, NOW, book, contract


def main() -> None:
    broker = PaperBroker("10.00", FEES)
    broker.register(contract())
    budget = RiskBudget("5.00", "5.00")
    inputs = PaperInputs("yes", 0.55, book(), True, True, True)
    decisions = {}
    for name, item in {
        "fresh_explicit": inputs,
        "stale": replace(inputs, book=book(age_seconds=30, snapshot="stale")),
        "missing": replace(inputs, book=None),
        "ambiguous_rules": replace(inputs, rules_verified=False),
        "invalid_probability": replace(inputs, probability=float("nan")),
    }.items():
        decisions[name] = paper_decision(broker, item, budget, 10, NOW)
    if decisions["fresh_explicit"].status == "PAPER_CANDIDATE_ONLY":
        order = broker.submit("yes", Side.BUY, 10, Decimal("0.40"), NOW)
        broker.take(order, inputs.book, NOW)
    emit("Offline paper session with explicit stopping conditions", {
        "decisions": decisions, "ledger": broker.ledger, "reconciliation": broker.reconcile(),
        "marked_not_realized": broker.mark({"yes": book()}, NOW),
        "operating_boundary": "in-memory deterministic replay; no daemon, credential loading or external order endpoint",
    })


if __name__ == "__main__":
    main()
