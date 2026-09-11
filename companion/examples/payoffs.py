"""Canonical exact-dollar accounting and exceptional payout, not an exchange fee."""

from datetime import timedelta
from decimal import Decimal

from prediction_market_lab import Settlement, Side
from prediction_market_lab.reporting import emit
from prediction_market_lab.risk import RiskBudget, screen_buy_yes
from prediction_market_lab.scoring import brier_score

from _support import FEES, NOW, book, canonical_position, rejected


def main() -> None:
    results = {"fee_assumption": FEES.description + "; flat USD 0.01 per unit per fill"}
    for name, payout in (("YES", "1"), ("NO", "0"), ("exceptional_fractional", "0.5")):
        broker = canonical_position()
        settlement = Settlement("yes", Decimal(payout), NOW + timedelta(days=1), name)
        broker.settle(settlement)
        results[name] = {"profit": broker.realized_profit, "ending_cash": broker.cash,
                         "binary_ML_label": settlement.binary_label}
    resale = canonical_position()
    sale = resale.submit("yes", Side.SELL, 10, Decimal("0.46"), NOW)
    resale.take(sale, book(bid="0.46", ask="0.48", snapshot="resale"), NOW)
    decision = screen_buy_yes(0.55, book(), 10, FEES, RiskBudget("5.00", "5.00"),
                              Decimal("10.00"), Decimal("0.00"))
    results.update({"initial_cost": decision.executable_cost,
                    "modeled_expected_profit_at_p_0_55": decision.modeled_expected_profit,
                    "resale_realized_profit": resale.realized_profit,
                    "fractional_label_rejected": rejected(lambda: brier_score([0.5], [0.55])),
                    "spread_counted_once": "entry uses ask; resale uses bid; no extra spread deduction"})
    emit("Contract cash flows, not an empirical edge", results)


if __name__ == "__main__":
    main()
