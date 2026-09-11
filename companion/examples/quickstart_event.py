"""Timestamped synthetic event forecast through model-conditional paper screening."""

from dataclasses import replace

from prediction_market_lab import PaperBroker
from prediction_market_lab.data import latest_per_event, teaching_split
from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.models import fit_forecaster
from prediction_market_lab.operations import PaperInputs, paper_decision
from prediction_market_lab.reporting import emit
from prediction_market_lab.risk import RiskBudget

from _support import FEES, NOW, book, contract, recurring


def main() -> None:
    split = teaching_split(recurring())
    model = fit_forecaster(split)
    event = latest_per_event(split.test)[0]
    p = round(float(model.predict([event])["fixed_equal_ensemble"][0]), 6)
    broker = PaperBroker("10.00", FEES)
    broker.register(contract())
    # These are new invented books for arithmetic, not books observed for the old event.
    cheap, costly = book(ask="0.40"), book(bid="0.90", ask="0.95", snapshot="expensive")
    inputs = PaperInputs("yes", p, cheap, True, event.features_available(), True)
    budget = RiskBudget("5.00", "5.00")
    emit("Recurring-event workflow without backdated execution claims", {
        "source_event": event.event_id, "feature_publication": event.published_at,
        "feature_observation": event.observed_at, "historical_decision": event.decision_at,
        "forecast": p, "new_synthetic_quote_time": NOW,
        "forecast_rounding": "six decimals for this printed arithmetic; scoring uses unrounded model probabilities",
        "illustrative_screen": paper_decision(broker, inputs, budget, 10, NOW),
        "costly_no_trade": paper_decision(broker, replace(inputs, book=costly), budget, 10, NOW),
        "evidence_gate": synthetic_evidence(),
        "limitation": "hypothetical quotes are not joined to this historical event and cannot form a P&L backtest",
    })


if __name__ == "__main__":
    main()
