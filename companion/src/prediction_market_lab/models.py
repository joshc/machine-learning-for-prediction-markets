"""Small CPU models with train-only preprocessing and held-out calibration."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import EventRow, TemporalSplit, latest_per_event
from .scoring import blend, brier_score, calibration_bins, log_loss


def feature_matrix(rows: Iterable[EventRow]) -> NDArray:
    observations = tuple(rows)
    if not observations or any(not row.features_available() for row in observations):
        raise ValueError("features must be nonempty and available at each decision")
    return np.asarray([[row.signal, row.horizon_hours, row.market_probability] for row in observations])


def _require_classes(rows: Iterable[EventRow]) -> None:
    if {row.outcome for row in rows} != {0, 1}:
        raise ValueError("this teaching fit requires both binary classes")


def _validate_split(split: TemporalSplit) -> None:
    cohorts = (split.train, split.validation, split.test)
    cutoffs = (split.train_cutoff, split.validation_cutoff, split.test_cutoff)
    if not cutoffs[0] < cutoffs[1] < cutoffs[2]:
        raise ValueError("invalid split cutoffs")
    seen_groups, seen_events = set(), set()
    for index, (rows, cutoff) in enumerate(zip(cohorts, cutoffs)):
        if not rows:
            raise ValueError("empty model cohort")
        groups, events = {row.group_id for row in rows}, {row.event_id for row in rows}
        if groups & seen_groups or events & seen_events:
            raise ValueError("event/group leakage across model cohorts")
        seen_groups |= groups
        seen_events |= events
        for row in rows:
            if (not row.features_available() or row.label_available_at > cutoff
                    or row.decision_at > cutoff
                    or (index and row.decision_at <= cutoffs[index - 1])):
                raise ValueError("point-in-time model cohort violation")


@dataclass(frozen=True)
class FittedForecaster:
    logistic: Pipeline
    boosted: GradientBoostingClassifier
    calibrator: IsotonicRegression
    reference_probability: float
    trained_event_ids: frozenset[str]
    calibrated_event_ids: frozenset[str]
    trained_group_ids: frozenset[str]
    calibrated_group_ids: frozenset[str]

    def predict(self, rows: Iterable[EventRow]) -> dict[str, NDArray]:
        observations = tuple(rows)
        x = feature_matrix(observations)
        logistic = self.logistic.predict_proba(x)[:, 1]
        boosted = self.boosted.predict_proba(x)[:, 1]
        calibrated = self.calibrator.predict(boosted)
        return {
            "reference_rate": np.full(len(x), self.reference_probability),
            "market_probability": np.asarray([row.market_probability for row in observations]),
            "logistic": logistic,
            "boosted": boosted,
            "boosted_calibrated": calibrated,
            "fixed_equal_ensemble": blend([logistic, calibrated], [0.5, 0.5]),
        }


def fit_forecaster(split: TemporalSplit, seed: int = 20260910) -> FittedForecaster:
    _validate_split(split)
    train, validation = latest_per_event(split.train), latest_per_event(split.validation)
    _require_classes(train)
    _require_classes(validation)
    x, y = feature_matrix(train), [row.outcome for row in train]
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=seed, max_iter=300))
    boosted = GradientBoostingClassifier(n_estimators=35, max_depth=2, learning_rate=0.05,
                                         min_samples_leaf=8, random_state=seed)
    logistic.fit(x, y)
    boosted.fit(x, y)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(boosted.predict_proba(feature_matrix(validation))[:, 1],
                   [row.outcome for row in validation])
    return FittedForecaster(logistic, boosted, calibrator, float(np.mean(y)),
                            frozenset(row.event_id for row in train),
                            frozenset(row.event_id for row in validation),
                            frozenset(row.group_id for row in split.train),
                            frozenset(row.group_id for row in split.validation))


def evaluate_forecaster(model: FittedForecaster, rows: Iterable[EventRow]) -> dict:
    events = latest_per_event(rows)
    if {row.event_id for row in events} & (model.trained_event_ids | model.calibrated_event_ids):
        raise ValueError("evaluation event was used for fitting or calibration")
    if {row.group_id for row in events} & (model.trained_group_ids | model.calibrated_group_ids):
        raise ValueError("evaluation dependence group was used for fitting or calibration")
    y = [row.outcome for row in events]
    predictions = model.predict(events)
    return {
        "evaluation_unit": "one latest predeclared snapshot per synthetic event",
        "events": len(events),
        "scores": {name: {"brier": brier_score(y, p), "natural_log_loss": log_loss(y, p)}
                   for name, p in predictions.items()},
        "logistic_calibration_bins": calibration_bins(y, predictions["logistic"], bins=4),
        "limitation": "synthetic scores; isotonic may emit exact endpoints with infinite test loss",
    }


def fit_text_and_neural(split: TemporalSplit, seed: int = 20260910) -> dict:
    _validate_split(split)
    train, test = latest_per_event(split.train), latest_per_event(split.test)
    _require_classes(train)
    y_train, y_test = [row.outcome for row in train], [row.outcome for row in test]
    text_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                               LogisticRegression(C=1, max_iter=300, random_state=seed))
    text_model.fit([row.text for row in train], y_train)
    text_p = text_model.predict_proba([row.text for row in test])[:, 1]
    neural = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(6,), solver="lbfgs",
                                                           alpha=2, max_iter=1500,
                                                           random_state=seed))
    neural.fit(feature_matrix(train), y_train)
    neural_p = neural.predict_proba(feature_matrix(test))[:, 1]
    return {
        "train_events": len(train), "test_events": len(test),
        "text_vocabulary_size": len(text_model[0].vocabulary_),
        "tfidf_logistic": {"brier": brier_score(y_test, text_p), "natural_log_loss": log_loss(y_test, text_p)},
        "six_hidden_unit_neural": {"brier": brier_score(y_test, neural_p),
                                  "natural_log_loss": log_loss(y_test, neural_p)},
        "limitation": "invented template text and numeric signals, no pretrained model or real news",
    }
