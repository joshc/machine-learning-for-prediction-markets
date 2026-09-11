"""Single-YES Brier score and natural-log loss, without endpoint clipping."""

from dataclasses import dataclass
from math import inf, isfinite
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .validation import probability


def binary_arrays(y: Sequence[int], p: Sequence[float]) -> tuple[NDArray, NDArray]:
    outcomes = np.asarray(y, dtype=float)
    forecasts = np.asarray(p, dtype=float)
    if outcomes.ndim != 1 or forecasts.shape != outcomes.shape or not outcomes.size:
        raise ValueError("nonempty equal-length one-dimensional labels and forecasts required")
    if not np.all(np.isfinite(outcomes)) or not np.all(np.isin(outcomes, [0, 1])):
        raise ValueError("binary scores require labels exactly zero or one, not fractional payouts")
    if not np.all(np.isfinite(forecasts)) or np.any((forecasts < 0) | (forecasts > 1)):
        raise ValueError("forecasts must be finite probabilities")
    return outcomes, forecasts


def brier_score(y: Sequence[int], p: Sequence[float]) -> float:
    labels, forecasts = binary_arrays(y, p)
    return float(np.mean((forecasts - labels) ** 2))


def log_loss(y: Sequence[int], p: Sequence[float]) -> float:
    labels, forecasts = binary_arrays(y, p)
    assigned = np.where(labels == 1, forecasts, 1 - forecasts)
    if np.any(assigned == 0):
        return inf
    return float(-np.mean(np.log(assigned)))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_rate: float | None


def calibration_bins(y: Sequence[int], p: Sequence[float], bins: int = 5) -> list[CalibrationBin]:
    labels, forecasts = binary_arrays(y, p)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bin count must be a positive integer")
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum((forecasts * bins).astype(int), bins - 1)
    result = []
    for i in range(bins):
        mask = indices == i
        count = int(mask.sum())
        result.append(CalibrationBin(float(edges[i]), float(edges[i + 1]), count,
                                     float(forecasts[mask].mean()) if count else None,
                                     float(labels[mask].mean()) if count else None))
    return result


def bayes_update(prior: float, likelihood_yes: float, likelihood_no: float) -> float:
    p, yes, no = map(probability, (prior, likelihood_yes, likelihood_no))
    evidence = p * yes + (1 - p) * no
    if evidence == 0:
        raise ValueError("the observed evidence has zero probability under this model")
    return p * yes / evidence


def blend(forecasts: Sequence[Sequence[float]], weights: Sequence[float]) -> NDArray:
    array, w = np.asarray(forecasts, dtype=float), np.asarray(weights, dtype=float)
    if (array.ndim != 2 or not array.shape[1] or w.shape != (array.shape[0],)
            or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0):
        raise ValueError("nonnegative finite weights and equally sized forecast vectors required")
    for row in array:
        binary_arrays(np.zeros(len(row)), row)
    return np.average(array, axis=0, weights=w)


def regression_metrics(y: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    actual, predicted = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (actual.ndim != 1 or predicted.shape != actual.shape or not actual.size
            or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted))):
        raise ValueError("regression metrics require matching nonempty finite vectors")
    error = predicted - actual
    mae, mse = float(np.mean(np.abs(error))), float(np.mean(error ** 2))
    if not isfinite(mae) or not isfinite(mse):
        raise ValueError("regression errors exceed numeric range")
    return {"mae": mae, "mse": mse}
