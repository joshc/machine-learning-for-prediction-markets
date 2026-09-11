"""Binary fractional Kelly assumptions and correlated joint loss scenarios."""

from decimal import Decimal

from prediction_market_lab.reporting import emit
from prediction_market_lab.risk import Exposure, RiskBudget, fractional_kelly, scenario_stress

from _support import rejected


def main() -> None:
    exposures = [Exposure("event-a", 10, Decimal("4.10")), Exposure("event-b", 10, Decimal("4.10"))]
    stress = scenario_stress(exposures, {
        "shared_bad_news_both_NO": {"event-a": Decimal(0), "event-b": Decimal(0)},
        "both_YES": {"event-a": Decimal(1), "event-b": Decimal(1)},
        "divergent": {"event-a": Decimal(1), "event-b": Decimal(0)},
    })
    emit("Sizing is conditional; affordable joint loss comes first", {
        "quarter_Kelly_fraction_of_cash_spent_p_0_55_cost_0_41": fractional_kelly(0.55, 0.41, 0.25),
        "no_edge_fraction": fractional_kelly(0.39, 0.41, 0.25),
        "joint_scenario_profit": stress,
        "second_position_passes_5_dollar_total_budget": RiskBudget("5.00", "5.00").permits(
            Decimal("4.10"), Decimal("5.90"), Decimal("4.10")),
        "invalid_binary_cost": rejected(lambda: fractional_kelly(0.55, 1, 0.25)),
        "assumptions": "known p, binary USD 1 payout, cost includes all costs, divisible units; not a real allocation recommendation",
    })


if __name__ == "__main__":
    main()
