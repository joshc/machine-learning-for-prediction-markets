"""Finite depth, funded reservations, partial fills and no quote-touch inference."""

from dataclasses import replace
from decimal import Decimal

from prediction_market_lab import Level, PaperBroker, Side
from prediction_market_lab.reporting import emit

from _support import FEES, NOW, book, contract, rejected


def main() -> None:
    broker = PaperBroker("3.00", FEES)
    broker.register(contract())
    order = broker.submit("yes", Side.BUY, 5, Decimal("0.45"), NOW)
    reserved = broker.reserved_cash
    double_spend = rejected(lambda: broker.submit("yes", Side.BUY, 2, Decimal("0.45"), NOW))
    depth = replace(book(), asks=(Level(Decimal("0.40"), 2), Level(Decimal("0.44"), 1)))
    fills = broker.take(order, depth, NOW)
    repeated = broker.take(order, depth, NOW)
    touch = broker.observe_resting(order, book(snapshot="touch"), NOW)
    partial = broker.status(order)
    broker.cancel(order, NOW)
    emit("Conservative paper execution replay", {
        "reserved_before_fills": reserved, "double_spend": double_spend,
        "finite_depth_fills": fills, "remaining_after_depth": partial.remaining,
        "same_snapshot_extra_fills": len(repeated), "resting_quote_touch": touch,
        "after_cancel": broker.reconcile(),
        "overfill_or_closed_order": rejected(lambda: broker.synthetic_fill(order, 3, Decimal("0.40"), NOW,
                                                                            evidence="invalid supplied fill")),
        "naked_sell": rejected(lambda: broker.submit("yes", Side.SELL, 4, Decimal("0.38"), NOW)),
        "latency_limitation": "taker snapshot fill is an explicit zero-latency paper assumption, not a historical guarantee",
    })


if __name__ == "__main__":
    main()
