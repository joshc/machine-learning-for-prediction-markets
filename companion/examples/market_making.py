"""User-supplied synthetic maker fills: spread capture can lose to adverse selection."""

from datetime import timedelta
from decimal import Decimal

from prediction_market_lab import PaperBroker, Settlement, Side
from prediction_market_lab.reporting import emit

from _support import FEES, NOW, book, contract


def maker_path(*, adverse: bool) -> dict:
    broker = PaperBroker("10.00", FEES)
    broker.register(contract())
    buy = broker.submit("yes", Side.BUY, 10, Decimal("0.40"), NOW)
    no_inference = broker.observe_resting(buy, book(), NOW)
    broker.synthetic_fill(buy, 10, Decimal("0.40"), NOW,
                          evidence="invented incoming sell fills our resting bid")
    sell = broker.submit("yes", Side.SELL, 10, Decimal("0.46"), NOW)
    if not adverse:
        broker.synthetic_fill(sell, 10, Decimal("0.46"), NOW,
                              evidence="invented incoming buy fills our ask")
        return {"realized_profit": broker.realized_profit, "inventory": broker.inventory("yes"),
                "quote_touch_policy": no_inference}
    broker.cancel(sell, NOW)
    mark = broker.mark({"yes": book(bid="0.25", ask="0.28", snapshot="adverse")}, NOW)
    broker.settle(Settlement("yes", Decimal(0), NOW + timedelta(days=1), "invented NO outcome"))
    return {"before_settlement_bid_liquidation_mark": mark,
            "final_realized_profit": broker.realized_profit, "inventory": broker.inventory("yes"),
            "quote_touch_policy": no_inference}


def main() -> None:
    emit("Market-making paths, not automatic spread income", {
        "favorable_supplied_fills": maker_path(adverse=False),
        "adverse_supplied_buy_no_exit": maker_path(adverse=True),
        "limitation": "neither fill probabilities nor the conditional outcome process was estimated",
    })


if __name__ == "__main__":
    main()
