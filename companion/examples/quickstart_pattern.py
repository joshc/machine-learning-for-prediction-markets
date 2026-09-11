"""Predeclared high/low market-probability calibration pattern, not a trade rule."""

from prediction_market_lab.data import latest_per_event, teaching_split
from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.reporting import emit
from prediction_market_lab.scoring import brier_score

from _support import recurring


def main() -> None:
    split = teaching_split(recurring())
    train, test = latest_per_event(split.train), latest_per_event(split.test)
    reference = sum(row.outcome for row in train) / len(train)
    groups = {}
    for name, high in (("market_below_half", False), ("market_at_least_half", True)):
        rows = [row for row in test if (row.market_probability >= 0.5) == high]
        groups[name] = {"events": len(rows), "observed_yes_rate": sum(r.outcome for r in rows) / len(rows),
                        "market_brier": brier_score([r.outcome for r in rows], [r.market_probability for r in rows]),
                        "training_reference_brier": brier_score([r.outcome for r in rows], [reference] * len(rows))}
    emit("Pattern research quickstart: predeclared split, baselines and refusal", {
        "hypothesis": "compare observed rate with market forecasts in two fixed probability bins",
        "analysis_unit": "one latest snapshot per held-out event", "groups": groups,
        "no_trade_baseline_profit": "0.00", "evidence_gate": synthetic_evidence(),
        "next_real_research_requirement": "timestamped independent events, costs, trial registry and dependence-aware uncertainty",
    })


if __name__ == "__main__":
    main()
