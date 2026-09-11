"""Inventory caps stop quoting even after a favorable-looking spread."""

from decimal import Decimal

from prediction_market_lab import PaperBroker, Side
from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.operations import PaperInputs, paper_decision
from prediction_market_lab.reporting import emit
from prediction_market_lab.risk import RiskBudget

from _support import FEES, NOW, book, contract


def main() -> None:
    broker = PaperBroker("10.00", FEES)
    broker.register(contract())
    bid_order = broker.submit("yes", Side.BUY, 10, Decimal("0.40"), NOW)
    broker.synthetic_fill(bid_order, 10, Decimal("0.40"), NOW, evidence="invented one-sided flow")
    liquidation = broker.mark({"yes": book(bid="0.25", ask="0.28", snapshot="inventory-stress")}, NOW)
    decision = paper_decision(broker, PaperInputs("yes", 0.60, book(), True, True, True),
                              RiskBudget("5.00", "5.00"), 10, NOW)
    emit("Market-making quickstart: inventory and stopping, not claimed income", {
        "filled_buy_units": broker.inventory("yes"), "filled_sell_units": 0,
        "inventory_liquidation_mark": liquidation,
        "additional_bid_budget_check": decision,
        "stop_reason": "inventory concentration and one-sided fills outweigh a displayed spread",
        "evidence_gate": synthetic_evidence(),
        "unmeasured": ["queue priority", "fill-conditioned adverse selection", "cancellation latency"],
    })


if __name__ == "__main__":
    main()
