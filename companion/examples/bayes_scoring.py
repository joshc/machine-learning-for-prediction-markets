"""Base rates, Bayes evidence, exact endpoint scores and calibration bins."""

from prediction_market_lab import bayes_update, brier_score, calibration_bins, log_loss
from prediction_market_lab.reporting import emit

from _support import rejected


def main() -> None:
    y, p = [0, 1, 0, 1], [0.1, 0.7, 0.4, 0.8]
    emit("Bayesian probability and single-YES proper scores", {
        "prior": 0.2, "likelihood_evidence_if_yes": 0.8, "likelihood_evidence_if_no": 0.1,
        "posterior": bayes_update(0.2, 0.8, 0.1),
        "brier_single_yes": brier_score(y, p), "natural_log_loss": log_loss(y, p),
        "correct_certainty_loss": log_loss([0, 1], [0, 1]),
        "wrong_certainty_loss": log_loss([1], [0]),
        "bins_left_closed_last_includes_one": calibration_bins(y, p, bins=4),
        "impossible_evidence": rejected(lambda: bayes_update(0.2, 0, 0)),
        "nonfinite_forecast": rejected(lambda: brier_score([1], [float("nan")])),
        "limitation": "invented observations; tiny bins do not establish calibration or tradeability",
    })


if __name__ == "__main__":
    main()
