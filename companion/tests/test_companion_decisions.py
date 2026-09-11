import unittest
from dataclasses import replace
from decimal import Decimal

from prediction_market_lab import Contract, FeeSchedule, Level, OrderBook, PaperBroker, Side
from prediction_market_lab.constraints import BasketLeg, payoff_dominates, scan_basket
from prediction_market_lab.evidence import Evidence, assess_evidence, synthetic_evidence
from prediction_market_lab.operations import PaperInputs, paper_decision
from prediction_market_lab.policy import LoggedAction, ips_value, learn_tabular_policy, simulate_logged_policy
from prediction_market_lab.risk import Exposure, RiskBudget, fractional_kelly, scenario_stress, screen_buy_yes
from prediction_market_lab.validation import parse_utc

D = Decimal
NOW = parse_utc("2025-07-01T12:00:00Z")
FEES = FeeSchedule(D("0.01"))


def book(name="yes", ask="0.40", size=10):
    return OrderBook(name, name, NOW, (Level(D("0.38"), size),), (Level(D(ask), size),))


class RiskAndDecisionTests(unittest.TestCase):
    def test_cost_and_no_double_counted_spread(self):
        result = screen_buy_yes(0.55, book(), 10, FEES, RiskBudget("5.00", "5.00"), D("10.00"), D("0.00"))
        self.assertEqual(result.status, "PAPER_CANDIDATE_ONLY")
        self.assertEqual(result.executable_cost, D("4.10"))
        self.assertEqual(result.modeled_expected_profit, D("1.40"))

    def test_negative_edge_and_insufficient_cash_or_depth_no_trade(self):
        budget = RiskBudget("5.00", "5.00")
        for p, quote, cash, committed in ((0.40, book(), "10.00", "0.00"),
                                         (0.55, book(), "4.00", "0.00"),
                                         (0.55, book(), "10.00", "4.10"),
                                         (0.55, book(size=9), "10.00", "0.00")):
            with self.subTest(p=p, cash=cash, committed=committed):
                result = screen_buy_yes(p, quote, 10, FEES, budget, D(cash), D(committed))
                self.assertEqual(result.status, "NO_TRADE")

    def test_kelly_is_fraction_spent_with_clear_no_trade(self):
        self.assertAlmostEqual(fractional_kelly(0.55, 0.41, 0.25), 0.0593220338983)
        self.assertEqual(fractional_kelly(0.4, 0.41, 0.25), 0)
        for args in ((0.55, 0, 0.25), (0.55, 1, 0.25), (float("nan"), 0.41, 0.25),
                     (0.55, 0.41, -0.25)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fractional_kelly(*args)

    def test_joint_scenarios_capture_correlated_total_loss(self):
        positions = [Exposure("a", 10, D("4.10")), Exposure("b", 10, D("4.10"))]
        results = scenario_stress(positions, {"both_NO": {"a": D(0), "b": D(0)},
                                             "both_YES": {"a": D(1), "b": D(1)}})
        self.assertEqual(results, {"both_NO": D("-8.20"), "both_YES": D("11.80")})
        with self.assertRaises(ValueError):
            scenario_stress(positions, {"missing_b": {"a": D(0)}})

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            RiskBudget("-1.00", "5.00")
        with self.assertRaises(ValueError):
            RiskBudget("5.00", "5.00").permits(D("-1.00"), D("5.00"), D("0.00"))


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.yes = BasketLeg(Contract("yes", "event", "v1"), book(ask="0.44"),
                              {"yes": D(1), "no": D(0)})
        self.no = BasketLeg(Contract("no", "event", "v1"), book("no", ask="0.49"),
                             {"yes": D(0), "no": D(1)})

    def scan(self, legs=None, qty=10, capital="20.00", verified=True, fees=FEES):
        return scan_basket(legs or [self.yes, self.no], qty, fees, D(capital), NOW,
                           compatibility_verified=verified)

    def test_complement_cost_profit_and_leg_warning(self):
        result = self.scan()
        self.assertEqual(result.total_cost, D("9.50"))
        self.assertEqual(result.minimum_payout, D("10.00"))
        self.assertEqual(result.minimum_net_profit, D("0.50"))
        self.assertEqual(result.status, "PAPER_CANDIDATE_ONLY")
        self.assertIn("non-atomic", result.leg_risk)

    def test_false_matching_events_rules_and_states_rejected(self):
        for false_no in (
            replace(self.no, contract=Contract("no", "different-event", "v1")),
            replace(self.no, contract=Contract("no", "event", "different-rule")),
            replace(self.no, payouts={"yes": D(0), "maybe": D(1)}),
        ):
            with self.subTest(leg=false_no):
                self.assertEqual(self.scan([self.yes, false_no]).status, "REJECTED")
        self.assertEqual(self.scan(verified=False).status, "REJECTED")
        self.assertEqual(self.scan(verified="False").status, "REJECTED")

    def test_capital_depth_and_fees_kill_candidate(self):
        self.assertEqual(self.scan(capital="5.00").status, "NO_TRADE")
        self.assertEqual(self.scan(qty=11).status, "NO_TRADE")
        self.assertEqual(self.scan(fees=FeeSchedule(D("0.10"))).status, "NO_TRADE")

    def test_basket_funds_the_brokers_conservative_order_reservations(self):
        fees = FeeSchedule(D("0.001"))
        legs = [
            replace(leg, book=OrderBook(
                f"reserve-{leg.contract.contract_id}", leg.contract.contract_id, NOW,
                (Level(D("0.30"), 3),), (Level(D("0.3333"), 3),),
            ))
            for leg in (self.yes, self.no)
        ]
        unfunded = self.scan(legs, qty=3, fees=fees, capital="2.02")
        self.assertEqual(unfunded.status, "NO_TRADE")
        self.assertEqual(unfunded.total_cost, D("2.02"))
        self.assertEqual(unfunded.required_cash, D("2.10"))
        funded = self.scan(legs, qty=3, fees=fees, capital="2.10")
        self.assertEqual(funded.status, "PAPER_CANDIDATE_ONLY")
        broker = PaperBroker("2.10", fees)
        orders = []
        for leg in legs:
            broker.register(leg.contract)
            orders.append(broker.submit(leg.contract.contract_id, Side.BUY, 3, D("0.3333"), NOW))
        self.assertEqual(broker.reserved_cash, funded.required_cash)
        for order, leg in zip(orders, legs):
            broker.take(order, leg.book, NOW)
        self.assertEqual(broker.cash, D("0.08"))
        self.assertEqual(broker.initial_cash - broker.cash, funded.total_cost)

    def test_nonexhaustive_basket_has_no_positive_floor(self):
        yes = replace(self.yes, payouts={"yes": D(1), "no": D(0), "void": D(0)})
        no = replace(self.no, payouts={"yes": D(0), "no": D(1), "void": D(0)})
        result = self.scan([yes, no])
        self.assertEqual(result.minimum_payout, 0)
        self.assertEqual(result.status, "NO_TRADE")

    def test_subset_uses_payoffs_not_titles(self):
        broad = replace(self.yes, payouts={"low": D(0), "middle": D(1), "high": D(1)})
        narrow = replace(self.no, payouts={"low": D(0), "middle": D(0), "high": D(1)})
        self.assertTrue(payoff_dominates(broad, narrow, compatibility_verified=True))
        self.assertFalse(payoff_dominates(narrow, broad, compatibility_verified=True))
        with self.assertRaises(ValueError):
            payoff_dominates(broad, narrow, compatibility_verified=False)


class EvidenceAndOperationsTests(unittest.TestCase):
    def setUp(self):
        self.broker = PaperBroker("10.00", FEES)
        self.broker.register(Contract("yes", "event", "v1"))
        self.inputs = PaperInputs("yes", 0.55, book(), True, True, True)
        self.budget = RiskBudget("5.00", "5.00")

    def test_synthetic_never_establishes_live_edge(self):
        evidence = Evidence(True, ("synthetic provenance",), True, True, True, True, True, True)
        gate = assess_evidence(evidence)
        self.assertEqual(gate.status, "REJECT_LIVE_EDGE_INFERENCE")
        self.assertFalse(gate.eligible_for_research_review)
        self.assertFalse(gate.live_recommendation)
        self.assertFalse(synthetic_evidence().live_recommendation)

    def test_even_declared_complete_real_evidence_cannot_auto_promote(self):
        gate = assess_evidence(Evidence(False, ("claimed raw archive",), True, True, True, True, True, True))
        self.assertEqual(gate.status, "RESEARCH_REVIEW_REQUIRED")
        self.assertFalse(gate.live_recommendation)

    def test_missing_provenance_or_loose_evidence_flags(self):
        gate = assess_evidence(Evidence(False, (), True, True, True, True, True, True))
        self.assertFalse(gate.eligible_for_research_review)
        with self.assertRaises(ValueError):
            assess_evidence(Evidence(False, ("x",), "True", True, True, True, True, True))

    def test_fresh_inputs_only_are_paper_candidate(self):
        result = paper_decision(self.broker, self.inputs, self.budget, 10, NOW)
        self.assertEqual(result.status, "PAPER_CANDIDATE_ONLY")
        self.assertEqual(self.broker.fills, ())

    def test_missing_ambiguous_nonfinite_unregistered_and_stale_inputs_fail_closed(self):
        from datetime import timedelta
        variants = [
            replace(self.inputs, book=None),
            replace(self.inputs, probability=None),
            replace(self.inputs, probability=float("nan")),
            replace(self.inputs, probability=1.2),
            replace(self.inputs, probability={"not": "numeric"}),
            replace(self.inputs, rules_verified=False),
            replace(self.inputs, rules_verified="False"),
            replace(self.inputs, features_available=False),
            replace(self.inputs, fee_assumptions_explicit=False),
            replace(self.inputs, contract_id="no", book=book("no")),
            replace(self.inputs, book=replace(book(), observed_at=NOW - timedelta(seconds=6))),
        ]
        for inputs in variants:
            with self.subTest(inputs=inputs):
                self.assertEqual(paper_decision(self.broker, inputs, self.budget, 10, NOW).status, "NO_TRADE")
        self.assertEqual(self.broker.cash, D("10.00"))

    def test_pending_cash_counted_in_total_risk(self):
        self.broker.submit("yes", Side.BUY, 10, D("0.40"), NOW)
        result = paper_decision(self.broker, self.inputs, self.budget, 10, NOW)
        self.assertEqual(result.status, "NO_TRADE")

    def test_screen_uses_remaining_depth_without_consuming_it(self):
        broker = PaperBroker("100.00", FeeSchedule())
        broker.register(Contract("yes", "event", "v1"))
        quote = OrderBook(
            "finite-screen", "yes", NOW, (Level(D("0.38"), 30),),
            (Level(D("0.40"), 10), Level(D("0.60"), 20)),
        )
        first = broker.submit("yes", Side.BUY, 10, D("0.40"), NOW)
        broker.take(first, quote, NOW)
        inputs = replace(self.inputs, book=quote)
        budget = RiskBudget("100.00", "100.00")
        for _ in range(2):
            result = paper_decision(broker, inputs, budget, 20, NOW)
            self.assertEqual(result.status, "NO_TRADE")
            self.assertEqual(result.executable_cost, D("12.00"))
            self.assertEqual(result.modeled_expected_profit, D("-1.00"))
        self.assertEqual(len(broker.fills), 1)
        self.assertEqual(quote.asks[0].quantity, 10)
        second = broker.submit("yes", Side.BUY, 20, D("0.60"), NOW)
        broker.take(second, quote, NOW)
        self.assertEqual(broker.cash, D("84.00"))
        self.assertEqual(paper_decision(broker, inputs, budget, 1, NOW).status, "NO_TRADE")


class PolicyTests(unittest.TestCase):
    def test_deterministic_policy_and_adverse_shift(self):
        policy = learn_tabular_policy(simulate_logged_policy(800, seed=11))
        self.assertEqual(policy, (0, 1))
        favorable = ips_value(simulate_logged_policy(800, seed=12), policy)
        adverse = ips_value(simulate_logged_policy(800, seed=12, adverse_selection=True), policy)
        self.assertGreater(favorable["ips_reward_per_round"], 0)
        self.assertLess(adverse["ips_reward_per_round"], 0)
        self.assertEqual(simulate_logged_policy(10), simulate_logged_policy(10))

    def test_off_policy_support_rejected(self):
        log = [LoggedAction(0, 0, 0, 1), LoggedAction(1, 0, 0, 1)]
        with self.assertRaisesRegex(ValueError, "support"):
            ips_value(log, (1, 1))
        with self.assertRaisesRegex(ValueError, "support"):
            learn_tabular_policy(log)

    def test_invalid_propensity_and_nonfinite_reward(self):
        for reward, propensity in ((0, 0), (0, float("nan")), (float("inf"), 0.5)):
            with self.subTest(reward=reward, propensity=propensity), self.assertRaises(ValueError):
                LoggedAction(0, 1, reward, propensity)


if __name__ == "__main__":
    unittest.main()
