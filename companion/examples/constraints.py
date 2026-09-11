"""Compatible complements, subset payoffs, false semantic matches and leg risk."""

from decimal import Decimal

from prediction_market_lab.constraints import BasketLeg, payoff_dominates, scan_basket
from prediction_market_lab.reporting import emit

from _support import FEES, NOW, book, contract


def main() -> None:
    yes = BasketLeg(contract("yes"), book("yes", ask="0.44", snapshot="yes-book"),
                    {"YES": Decimal(1), "NO": Decimal(0)})
    no = BasketLeg(contract("no"), book("no", ask="0.49", snapshot="no-book"),
                   {"YES": Decimal(0), "NO": Decimal(1)})
    false_no = BasketLeg(contract("other-no", rules="different-resolution-source"),
                         book("other-no", ask="0.49", snapshot="false-book"), no.payouts)
    broad = BasketLeg(contract("broad"), book("broad", snapshot="broad-book"),
                      {"low": Decimal(0), "middle": Decimal(1), "high": Decimal(1)})
    narrow = BasketLeg(contract("narrow"), book("narrow", snapshot="narrow-book"),
                       {"low": Decimal(0), "middle": Decimal(0), "high": Decimal(1)})
    emit("Explicit state-space consistency, never matching titles alone", {
        "complementary_basket": scan_basket([yes, no], 10, FEES, Decimal("20.00"), NOW,
                                           compatibility_verified=True),
        "capital_failure": scan_basket([yes, no], 10, FEES, Decimal("5.00"), NOW,
                                       compatibility_verified=True),
        "false_match": scan_basket([yes, false_no], 10, FEES, Decimal("20.00"), NOW,
                                   compatibility_verified=True),
        "subset_payoff_dominated_by_superset": payoff_dominates(broad, narrow, compatibility_verified=True),
        "assumption": "manually verified same event, rules, USD payout and exhaustive states; no exceptional state omitted",
    })


if __name__ == "__main__":
    main()
