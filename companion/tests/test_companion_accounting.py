import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from random import Random

from prediction_market_lab import Contract, FeeSchedule, Level, OrderBook, PaperBroker, Settlement, Side
from prediction_market_lab.contracts import executable_limit, sweep_cost
from prediction_market_lab.validation import money, parse_utc, price, quantity

D = Decimal
NOW = parse_utc("2025-07-01T12:00:00Z")


def make_book(*, snapshot="book", bid="0.38", ask="0.40", size=100):
    return OrderBook(snapshot, "yes", NOW, (Level(D(bid), size),), (Level(D(ask), size),))


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.fees = FeeSchedule(D("0.01"))
        self.broker = PaperBroker("10.00", self.fees)
        self.broker.register(Contract("yes", "event", "v1"))

    def buy(self, qty=10, limit="0.40"):
        order = self.broker.submit("yes", Side.BUY, qty, D(limit), NOW)
        self.broker.take(order, make_book(), NOW)
        return order

    def test_canonical_binary_settlement_and_fee(self):
        self.buy()
        self.assertEqual(self.broker.cash, D("5.90"))
        self.assertEqual(self.broker.fills[0].fee, D("0.10"))
        self.broker.settle(Settlement("yes", D(1), NOW, "binary YES"))
        self.assertEqual(self.broker.realized_profit, D("5.90"))
        self.assertEqual(self.broker.inventory("yes"), 0)
        self.assertEqual(self.broker.reconcile()["cash"], D("15.90"))

    def test_binary_loss(self):
        self.buy()
        self.broker.settle(Settlement("yes", D(0), NOW, "binary NO"))
        self.assertEqual(self.broker.realized_profit, D("-4.10"))

    def test_exceptional_payout_not_binary_label(self):
        self.buy()
        event = Settlement("yes", D("0.5"), NOW, "explicit exceptional payout")
        self.assertIsNone(event.binary_label)
        self.broker.settle(event)
        self.assertEqual(self.broker.cash, D("10.90"))
        self.assertEqual(self.broker.realized_profit, D("0.90"))

    def test_sale_realizes_only_net_round_trip(self):
        self.buy()
        order = self.broker.submit("yes", Side.SELL, 10, D("0.46"), NOW)
        self.broker.take(order, make_book(snapshot="exit", bid="0.46", ask="0.48"), NOW)
        self.assertEqual(self.broker.realized_profit, D("0.40"))
        self.assertEqual(self.broker.cash, D("10.40"))

    def test_mark_changes_no_cash_or_realized_profit(self):
        self.buy()
        mark = self.broker.mark({"yes": make_book()}, NOW)
        self.assertEqual(mark.inventory_liquidation_value, D("3.70"))
        self.assertEqual(mark.marked_profit, D("-0.40"))
        self.assertEqual(mark.realized_profit, 0)
        self.assertEqual(self.broker.cash, D("5.90"))
        self.assertEqual(self.broker.inventory("yes"), 10)

    def test_fifo_partial_sale_and_settlement(self):
        self.buy()
        order = self.broker.submit("yes", Side.SELL, 3, D("0.46"), NOW)
        self.broker.synthetic_fill(order, 3, D("0.46"), NOW, evidence="test-supplied fill")
        self.assertEqual(self.broker.realized_profit, D("0.12"))
        self.assertEqual(self.broker.reconcile()["open_cost_basis"], D("2.87"))
        self.broker.settle(Settlement("yes", D(0), NOW, "NO"))
        self.assertEqual(self.broker.realized_profit, D("-2.75"))
        self.assertEqual(self.broker.cash, D("7.25"))

    def test_pending_orders_reserve_funded_cash(self):
        first = self.broker.submit("yes", Side.BUY, 20, D("0.40"), NOW)
        self.assertEqual(self.broker.reserved_cash, D("8.20"))
        with self.assertRaisesRegex(ValueError, "unreserved"):
            self.broker.submit("yes", Side.BUY, 5, D("0.40"), NOW)
        self.broker.cancel(first, NOW)
        self.assertEqual(self.broker.available_cash, D("10.00"))

    def test_partial_fills_release_only_remaining_reserve(self):
        order = self.broker.submit("yes", Side.BUY, 5, D("0.45"), NOW)
        depth = replace(make_book(), asks=(Level(D("0.40"), 2), Level(D("0.44"), 1)))
        fills = self.broker.take(order, depth, NOW)
        self.assertEqual([fill.quantity for fill in fills], [2, 1])
        self.assertEqual(self.broker.cash, D("8.73"))
        self.assertEqual(self.broker.reserved_cash, D("0.92"))
        self.assertEqual(self.broker.status(order).remaining, 2)
        self.assertEqual(self.broker.take(order, depth, NOW), ())
        self.broker.cancel(order, NOW)
        self.assertEqual(self.broker.reserved_cash, 0)

    def test_depth_shared_across_orders_not_infinite(self):
        a = self.broker.submit("yes", Side.BUY, 2, D("0.40"), NOW)
        b = self.broker.submit("yes", Side.BUY, 3, D("0.40"), NOW)
        depth = make_book(size=3)
        self.broker.take(a, depth, NOW)
        self.broker.take(b, replace(depth), NOW)
        self.assertEqual(self.broker.inventory("yes"), 3)
        self.assertEqual(self.broker.status(b).remaining, 2)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.broker.take(b, replace(depth, asks=(Level(D("0.40"), 9),)), NOW)

    def test_limit_does_not_sweep_worse_prices(self):
        order = self.broker.submit("yes", Side.BUY, 5, D("0.41"), NOW)
        depth = replace(make_book(), asks=(Level(D("0.40"), 2), Level(D("0.44"), 3)))
        self.broker.take(order, depth, NOW)
        self.assertEqual(self.broker.status(order).filled, 2)
        self.assertEqual(executable_limit(depth, Side.BUY, 5), D("0.44"))
        self.assertEqual(sweep_cost(depth, Side.BUY, 5, self.fees), D("2.17"))

    def test_touch_never_fills_resting_order(self):
        order = self.broker.submit("yes", Side.BUY, 5, D("0.40"), NOW)
        result = self.broker.observe_resting(order, make_book(), NOW)
        self.assertTrue(result.startswith("NO_INFERRED_FILL"))
        self.assertEqual(self.broker.status(order).filled, 0)
        self.assertEqual(self.broker.inventory("yes"), 0)
        self.broker.cancel(order, NOW)
        with self.assertRaisesRegex(ValueError, "closed order"):
            self.broker.observe_resting(order, make_book(), NOW)

    def test_synthetic_fill_enforces_limit_and_no_overfill(self):
        order = self.broker.submit("yes", Side.BUY, 3, D("0.40"), NOW)
        with self.assertRaisesRegex(ValueError, "limit"):
            self.broker.synthetic_fill(order, 1, D("0.41"), NOW, evidence="bad fill")
        with self.assertRaisesRegex(ValueError, "overfill"):
            self.broker.synthetic_fill(order, 4, D("0.40"), NOW, evidence="bad size")
        with self.assertRaisesRegex(ValueError, "explanation"):
            self.broker.synthetic_fill(order, 1, D("0.40"), NOW, evidence="")
        self.assertEqual(self.broker.cash, D("10.00"))
        self.assertEqual(self.broker.status(order).filled, 0)

    def test_sell_limit_is_bid_floor(self):
        self.buy(1)
        order = self.broker.submit("yes", Side.SELL, 1, D("0.46"), NOW)
        with self.assertRaisesRegex(ValueError, "limit"):
            self.broker.synthetic_fill(order, 1, D("0.45"), NOW, evidence="bad sell")
        self.assertEqual(self.broker.inventory("yes"), 1)

    def test_sell_units_reserved_no_naked_sale(self):
        with self.assertRaisesRegex(ValueError, "naked"):
            self.broker.submit("yes", Side.SELL, 1, D("0.40"), NOW)
        self.buy()
        sale = self.broker.submit("yes", Side.SELL, 8, D("0.40"), NOW)
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.broker.submit("yes", Side.SELL, 3, D("0.40"), NOW)
        self.broker.cancel(sale, NOW)
        self.broker.submit("yes", Side.SELL, 10, D("0.40"), NOW)

    def test_stale_future_and_backward_timestamps(self):
        order = self.broker.submit("yes", Side.BUY, 1, D("0.40"), NOW)
        for delta in (-6, 1):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "stale or future"):
                self.broker.take(order, replace(make_book(), observed_at=NOW + timedelta(seconds=delta)), NOW)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.broker.cancel(order, NOW - timedelta(seconds=1))
        self.assertEqual(self.broker.status(order).filled, 0)

    def test_settlement_cancels_pending_orders_and_is_final(self):
        self.buy(2)
        order = self.broker.submit("yes", Side.BUY, 3, D("0.40"), NOW)
        self.broker.settle(Settlement("yes", D(1), NOW, "YES"))
        self.assertTrue(self.broker.status(order).cancelled)
        self.assertEqual(self.broker.reserved_cash, 0)
        with self.assertRaises(ValueError):
            self.broker.settle(Settlement("yes", D(1), NOW, "duplicate"))
        with self.assertRaises(ValueError):
            self.broker.submit("yes", Side.BUY, 1, D("0.40"), NOW)

    def test_mark_rejects_missing_stale_insufficient_depth(self):
        self.buy()
        for books, now in (({}, NOW), ({"yes": make_book(size=9)}, NOW),
                           ({"yes": make_book()}, NOW + timedelta(seconds=6))):
            with self.subTest(books=books), self.assertRaises(ValueError):
                self.broker.mark(books, now)

    def test_mark_respects_already_consumed_bids(self):
        self.buy()
        order = self.broker.submit("yes", Side.SELL, 5, D("0.38"), NOW)
        small = make_book(snapshot="exit", size=5)
        self.broker.take(order, small, NOW)
        with self.assertRaisesRegex(ValueError, "bid depth"):
            self.broker.mark({"yes": small}, NOW)

    def test_rounding_policy_and_conservative_split_fill_reserve(self):
        schedule = FeeSchedule(D("0.001"))
        self.assertEqual(schedule.total(3, D("0.3333")), D("0.01"))
        self.assertEqual(sum(schedule.total(1, D("0.3333")) for _ in range(3)), D("0.03"))
        broker = PaperBroker("1.05", schedule)
        broker.register(Contract("yes", "event", "v1"))
        order = broker.submit("yes", Side.BUY, 3, D("0.3333"), NOW)
        self.assertEqual(broker.reserved_cash, D("1.05"))
        for _ in range(3):
            broker.synthetic_fill(order, 1, D("0.3333"), NOW, evidence="separate one-unit fill")
        self.assertEqual(broker.cash, D("0.03"))
        self.assertEqual(broker.reserved_cash, 0)
        broker.reconcile()

    def test_cent_basis_rounding_conserves_whole_lot(self):
        broker = PaperBroker("10.00", FeeSchedule())
        broker.register(Contract("yes", "event", "v1"))
        buy = broker.submit("yes", Side.BUY, 3, D("0.3333"), NOW)
        broker.synthetic_fill(buy, 3, D("0.3333"), NOW, evidence="one rounded aggregate lot")
        for _ in range(3):
            sell = broker.submit("yes", Side.SELL, 1, D("0.50"), NOW)
            broker.synthetic_fill(sell, 1, D("0.50"), NOW, evidence="one-unit liquidation")
        self.assertEqual(broker.realized_profit, D("0.50"))
        self.assertEqual(broker.reconcile()["open_cost_basis"], 0)

    def test_wrong_contract_and_metadata_change(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.broker.register(Contract("yes", "different-event", "v1"))
        order = self.broker.submit("yes", Side.BUY, 1, D("0.40"), NOW)
        with self.assertRaisesRegex(ValueError, "wrong book|unknown"):
            self.broker.take(order, replace(make_book(), contract_id="no"), NOW)

    def test_randomized_long_only_cash_and_inventory_conservation(self):
        rng = Random(19)
        for _ in range(60):
            side = Side.SELL if self.broker.inventory("yes") and rng.random() < 0.5 else Side.BUY
            unit = D(rng.randrange(10, 90)) / 100
            if side is Side.BUY and unit + D("0.01") > self.broker.available_cash:
                continue
            order = self.broker.submit("yes", side, 1, unit, NOW)
            self.broker.synthetic_fill(order, 1, unit, NOW, evidence="seeded conservation test")
            self.broker.reconcile()
        self.broker.settle(Settlement("yes", D("0.5"), NOW, "exceptional"))
        self.assertEqual(self.broker.cash - self.broker.initial_cash, self.broker.realized_profit)


class NumericAndBookValidationTests(unittest.TestCase):
    def test_money_and_prices_reject_floats_nonfinite_and_wrong_grids(self):
        for value in (0.4, True, "NaN", "Infinity", "0.001", "1e100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                money(value)
        for value in ("-0.01", "1.01", "0.12345"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                price(value)
        self.assertEqual(money("4.10"), D("4.10"))

    def test_whole_positive_units(self):
        for value in (0, -1, 0.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantity(value)

    def test_book_order_and_crossing(self):
        with self.assertRaises(ValueError):
            make_book(bid="0.5", ask="0.4")
        with self.assertRaises(ValueError):
            replace(make_book(), asks=(Level(D("0.5"), 2), Level(D("0.4"), 2)))
        with self.assertRaises(ValueError):
            replace(make_book(), bids=(Level(D("0.3"), 2), Level(D("0.3"), 2)))

    def test_executable_side_and_missing_depth(self):
        book = make_book(size=2)
        self.assertEqual(book.executable(Side.BUY)[0].price, D("0.40"))
        self.assertEqual(book.executable(Side.SELL)[0].price, D("0.38"))
        self.assertEqual(book.midpoint(), D("0.39"))
        with self.assertRaises(ValueError):
            sweep_cost(book, Side.BUY, 3, FeeSchedule())
        with self.assertRaises(ValueError):
            replace(book, bids=()).midpoint()

    def test_fee_and_currency_validation(self):
        with self.assertRaises(ValueError):
            FeeSchedule(D("-0.01"))
        with self.assertRaises(ValueError):
            Contract("yes", "event", "v1", "USDC")
        with self.assertRaises(ValueError):
            PaperBroker("-1.00", FeeSchedule())

    def test_net_negative_sell_rejected(self):
        with self.assertRaisesRegex(ValueError, "fees"):
            sweep_cost(make_book(), Side.SELL, 1, FeeSchedule(D("0.5")))


if __name__ == "__main__":
    unittest.main()
