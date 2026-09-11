import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

from prediction_market_lab.data import generate_recurring, latest_per_event, load_recurring, teaching_split
from prediction_market_lab.lmsr import cost, prices, trade_cost
from prediction_market_lab.models import evaluate_forecaster, fit_forecaster, fit_text_and_neural
from prediction_market_lab.scoring import bayes_update, blend, brier_score, calibration_bins, log_loss, regression_metrics
from prediction_market_lab.validation import available, parse_utc

ROOT = Path(__file__).resolve().parents[1]


class ScoreTests(unittest.TestCase):
    def test_single_yes_brier_not_doubled(self):
        self.assertAlmostEqual(brier_score([0, 1, 0, 1], [0.1, 0.7, 0.4, 0.8]), 0.075)

    def test_log_loss_exact_endpoints(self):
        self.assertEqual(log_loss([0, 1], [0, 1]), 0)
        self.assertEqual(log_loss([1], [0]), float("inf"))
        self.assertEqual(log_loss([0], [1]), float("inf"))
        self.assertAlmostEqual(log_loss([1], [0.5]), np.log(2))
        self.assertAlmostEqual(log_loss([1], [1e-100]), -np.log(1e-100))

    def test_scores_reject_bad_labels_probabilities_shapes(self):
        for y, p in (([0.5], [0.5]), ([2], [0.5]), ([1], [float("nan")]),
                     ([1], [float("inf")]), ([1], [-0.1]), ([], []), ([0, 1], [0.5]),
                     ([[0]], [[0.5]])):
            for score in (brier_score, log_loss):
                with self.subTest(y=y, p=p, score=score), self.assertRaises(ValueError):
                    score(y, p)

    def test_calibration_edges_and_empty_bins(self):
        bins = calibration_bins([0, 1, 1], [0, 0.5, 1], bins=4)
        self.assertEqual([item.count for item in bins], [1, 0, 1, 1])
        self.assertIsNone(bins[1].observed_rate)
        with self.assertRaises(ValueError):
            calibration_bins([1], [0.5], bins=0)

    def test_bayes_reference_and_zero_evidence(self):
        self.assertAlmostEqual(bayes_update(0.2, 0.8, 0.1), 2 / 3)
        self.assertEqual(bayes_update(1, 0.5, 0.1), 1)
        for args in ((0.2, 0, 0), (float("nan"), 0.8, 0.1), (0.2, 1.1, 0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                bayes_update(*args)

    def test_fixed_ensemble_and_invalid_weights(self):
        np.testing.assert_allclose(blend([[0.2, 0.5], [0.4, 0.7]], [1, 1]), [0.3, 0.6])
        for weights in ((0, 0), (-1, 2), (float("nan"), 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                blend([[0.2], [0.3]], weights)

    def test_regression_requires_finite_inputs(self):
        self.assertEqual(regression_metrics([1, 3], [2, 2]), {"mae": 1.0, "mse": 1.0})
        for y, p in (([], []), ([1], [float("inf")]), ([1, 2], [2])):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                regression_metrics(y, p)


class LMSRTests(unittest.TestCase):
    def test_prices_and_trade_cost(self):
        np.testing.assert_allclose(prices([0, 0], 10), [0.5, 0.5])
        self.assertAlmostEqual(trade_cost([0, 0], [10, 0], 10), 6.20114506958)
        self.assertGreater(trade_cost([0, 0], [10, 0], 10), 5)
        self.assertAlmostEqual(sum(prices([10000, 9999], 10)), 1)
        self.assertTrue(np.isfinite(cost([10000, 9999], 10)))

    def test_translation_invariance(self):
        self.assertAlmostEqual(cost([10000, 9999], 10) - 10000, cost([0, -1], 10))
        self.assertAlmostEqual(trade_cost([10000, 9999], [1, 0], 10), trade_cost([0, -1], [1, 0], 10))

    def test_invalid_inputs(self):
        for q, B in (([0], 1), ([0, 0], 0), ([0, float("nan")], 10), ([0, 0], float("inf"))):
            with self.subTest(q=q, B=B), self.assertRaises(ValueError):
                cost(q, B)
        with self.assertRaises(ValueError):
            trade_cost([0, 0], [1], 10)


class PointInTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_recurring()
        cls.split = teaching_split(cls.rows)

    def test_stored_fixture_exactly_matches_generator(self):
        stored = load_recurring(ROOT / "data" / "fixtures" / "recurring_events.csv")
        self.assertEqual(stored, self.rows)
        self.assertEqual(len(stored), 360)
        self.assertEqual(len(latest_per_event(stored)), 180)
        self.assertTrue(all(row.synthetic for row in stored))

    def test_publication_and_observation_are_both_required(self):
        row = self.rows[0]
        self.assertTrue(row.features_available())
        self.assertFalse(available(row.published_at, row.decision_at + timedelta(seconds=1), row.decision_at))
        self.assertFalse(available(row.decision_at + timedelta(seconds=1), row.observed_at, row.decision_at))
        self.assertGreater(row.label_available_at, row.decision_at)
        with self.assertRaises(ValueError):
            parse_utc("2025-01-01T00:00:00")

    def test_groups_and_label_cutoffs(self):
        split = self.split
        group_sets = [{row.group_id for row in cohort} for cohort in (split.train, split.validation, split.test)]
        self.assertEqual(list(map(len, group_sets)), [89, 44, 45])
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        self.assertEqual(split.excluded_groups, ("toy-0089", "toy-0134"))
        for cohort, cutoff in ((split.train, split.train_cutoff),
                               (split.validation, split.validation_cutoff), (split.test, split.test_cutoff)):
            self.assertTrue(all(row.label_available_at <= cutoff for row in cohort))

    def test_future_label_group_is_purged_not_moved_into_validation(self):
        changed = tuple(replace(row, label_available_at=self.split.validation_cutoff)
                        if row.event_id == "toy-0000" else row for row in self.rows)
        split = teaching_split(changed)
        self.assertIn("toy-0000", split.excluded_groups)
        self.assertFalse(any(row.event_id == "toy-0000" for row in split.validation))

    def test_future_feature_is_rejected(self):
        late = replace(self.rows[0], observed_at=self.rows[0].decision_at + timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            teaching_split((late, *self.rows[1:]))

    def test_event_cannot_have_multiple_group_ids(self):
        bad = replace(self.rows[0], group_id="different")
        with self.assertRaisesRegex(ValueError, "dependence groups"):
            teaching_split((bad, *self.rows[1:]))

    def test_inconsistent_event_labels_rejected(self):
        bad = replace(self.rows[0], outcome=1 - self.rows[0].outcome)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            teaching_split((bad, *self.rows[1:]))

    def test_group_straddling_train_boundary_purged(self):
        changed = tuple(replace(row, group_id="linked-boundary")
                        if row.event_id in ("toy-0088", "toy-0090") else row for row in self.rows)
        split = teaching_split(changed)
        self.assertIn("linked-boundary", split.excluded_groups)
        self.assertFalse(any(row.group_id == "linked-boundary" for row in split.train + split.validation))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = teaching_split(generate_recurring())
        cls.model = fit_forecaster(cls.split)

    def test_train_and_validation_event_identity(self):
        self.assertEqual(len(self.model.trained_event_ids), 89)
        self.assertEqual(len(self.model.calibrated_event_ids), 44)
        self.assertFalse(self.model.trained_event_ids & self.model.calibrated_event_ids)
        self.assertEqual(evaluate_forecaster(self.model, self.split.test)["events"], 45)

    def test_preprocessing_fit_only_on_train(self):
        expected = np.mean([[row.signal, row.horizon_hours, row.market_probability]
                            for row in latest_per_event(self.split.train)], axis=0)
        np.testing.assert_allclose(self.model.logistic[0].mean_, expected)

    def test_changing_test_labels_cannot_change_fits(self):
        changed = replace(self.split, test=tuple(replace(row, outcome=1 - row.outcome) for row in self.split.test))
        model = fit_forecaster(changed)
        before = self.model.predict(self.split.test)
        after = model.predict(self.split.test)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])

    def test_cannot_pass_manually_leaked_split(self):
        bad = replace(self.split, test=self.split.train)
        with self.assertRaisesRegex(ValueError, "leakage"):
            fit_forecaster(bad)
        with self.assertRaisesRegex(ValueError, "evaluation event"):
            evaluate_forecaster(self.model, self.split.validation)

    def test_evaluation_rejects_new_events_in_fitted_dependence_groups(self):
        for fitted_row in (self.split.train[0], self.split.validation[0]):
            linked = replace(self.split.test[0], group_id=fitted_row.group_id)
            with self.subTest(group=linked.group_id), self.assertRaisesRegex(ValueError, "dependence group"):
                evaluate_forecaster(self.model, [linked])

    def test_bad_fitting_feature_or_label_clock_rejected(self):
        row = self.split.train[0]
        bad = replace(row, label_available_at=self.split.validation_cutoff)
        with self.assertRaisesRegex(ValueError, "point-in-time"):
            fit_forecaster(replace(self.split, train=(bad, *self.split.train[1:])))

    def test_tiny_text_and_neural_models_are_finite(self):
        result = fit_text_and_neural(self.split)
        for name in ("tfidf_logistic", "six_hidden_unit_neural"):
            self.assertTrue(np.isfinite(result[name]["brier"]))
            self.assertTrue(np.isfinite(result[name]["natural_log_loss"]))
        self.assertEqual(result["test_events"], 45)


if __name__ == "__main__":
    unittest.main()
