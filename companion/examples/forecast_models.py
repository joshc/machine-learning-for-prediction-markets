"""Chronological logistic/boosting baselines and held-out calibration."""

from prediction_market_lab.data import latest_per_event, teaching_split
from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.models import evaluate_forecaster, fit_forecaster
from prediction_market_lab.reporting import emit

from _support import recurring


def main() -> None:
    split = teaching_split(recurring())
    model = fit_forecaster(split)
    event = latest_per_event(split.test)[0].event_id
    updates = [row for row in split.test if row.event_id == event]
    forecasts = model.predict(updates)["logistic"]
    emit("Models, chronological evaluation, temporal updates and a fixed ensemble", {
        "train_events": len(model.trained_event_ids),
        "validation_events_for_calibration": len(model.calibrated_event_ids),
        "test": evaluate_forecaster(model, split.test),
        "same_event_updates_not_independent_trials": [
            {"event": row.event_id, "decision_at": row.decision_at,
             "horizon_hours": row.horizon_hours, "logistic_probability": float(p)}
            for row, p in zip(updates, forecasts)],
        "temporal_update_limitation": "frozen latest-horizon model rescored at two snapshots; early-horizon calibration is unvalidated",
        "evidence_gate": synthetic_evidence(),
        "selection_warning": "fixed toy settings; do not tune on these repeatedly displayed test scores",
    })


if __name__ == "__main__":
    main()
