"""Stable LMSR prices and finite trade cost, not a real venue's order book."""

from prediction_market_lab.lmsr import cost, prices, trade_cost
from prediction_market_lab.reporting import emit

from _support import rejected


def main() -> None:
    emit("LMSR liquidity B, not order-book bid b", {
        "liquidity_B": 10, "initial_marginal_prices": prices([0, 0], 10).tolist(),
        "cost_to_buy_ten_yes": trade_cost([0, 0], [10, 0], 10),
        "final_marginal_prices": prices([10, 0], 10).tolist(),
        "stable_large_position_cost": cost([10000, 9999], 10),
        "invalid_B": rejected(lambda: cost([0, 0], 0)),
        "limitation": "marginal price times size is not the finite LMSR trade cost; fees omitted",
    })


if __name__ == "__main__":
    main()
