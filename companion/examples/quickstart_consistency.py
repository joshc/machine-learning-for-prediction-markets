"""Three-state exhaustive partition, insufficient depth and explicit leg-risk loss."""

from decimal import Decimal

from prediction_market_lab import PaperBroker, Settlement, Side
from prediction_market_lab.constraints import BasketLeg, scan_basket
from prediction_market_lab.reporting import emit

from _support import FEES, NOW, book, contract


def main() -> None:
    states = ("low", "middle", "high")
    legs = [BasketLeg(contract(state), book(state, bid="0.25", ask="0.30", ask_size=4, snapshot=state),
                      {outcome: Decimal(int(state == outcome)) for outcome in states}) for state in states]
    candidate = scan_basket(legs, 4, FEES, Decimal("10.00"), NOW, compatibility_verified=True)
    too_large = scan_basket(legs, 5, FEES, Decimal("10.00"), NOW, compatibility_verified=True)
    broker = PaperBroker("10.00", FEES)
    broker.register(legs[0].contract)
    order = broker.submit("low", Side.BUY, 4, Decimal("0.30"), NOW)
    broker.take(order, legs[0].book, NOW)
    broker.settle(Settlement("low", Decimal(0), NOW, "invented middle state; other legs never filled"))
    emit("Consistency quickstart: cheap complete basket versus failed legs", {
        "verified_partition_candidate": candidate, "depth_failure": too_large,
        "only_first_leg_fills_then_loses_profit": broker.realized_profit,
        "required_manual_checks": ["same event and resolution source", "complete exceptional payout state space",
                                    "fresh executable depth", "fees", "funded capital", "non-atomic leg failure"],
    })


if __name__ == "__main__":
    main()
