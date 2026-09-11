"""Educational manual checklist only. This file cannot place any funded order."""

from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.reporting import emit

from _support import canonical_position


def main() -> None:
    broker = canonical_position()
    emit("Manual first-trade checklist: not cleared by these toy exercises", {
        "status": "DO_NOT_PROCEED_FROM_SYNTHETIC_EVIDENCE",
        "evidence_gate": synthetic_evidence(),
        "manual_checks_not_automatically_verified": [
            "eligibility, jurisdiction and current venue terms independently checked",
            "exact contract rules, exceptional payouts and collateral understood",
            "real point-in-time evidence independently reviewed after strategy search",
            "current fees, rounding, executable depth and maximum affordable loss checked",
            "open inventory, reserved cash and correlated loss reviewed",
            "a written manual decision, rejection conditions and outcome journal prepared",
        ],
        "invented_accounting_only": broker.reconcile(),
        "maximum_loss_if_this_binary_position_settles_NO": "4.10",
        "boundary": "no API keys, network requests, order submission, wallet signing or automatic promotion",
    })


if __name__ == "__main__":
    main()
