"""A contextual bandit can learn simulator assumptions rather than a real edge."""

from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.policy import LoggedAction, ips_value, learn_tabular_policy, simulate_logged_policy
from prediction_market_lab.reporting import emit

from _support import rejected


def main() -> None:
    policy = learn_tabular_policy(simulate_logged_policy(800, seed=11))
    emit("Wait/buy bandit under a deliberately simplified simulator", {
        "policy_actions_for_context_0_and_1": policy,
        "action_names": {"0": "wait", "1": "buy unit at all-in invented cost 0.51"},
        "held_out_same_simulator": ips_value(simulate_logged_policy(800, seed=12), policy),
        "held_out_adverse_selection_shift": ips_value(
            simulate_logged_policy(800, seed=12, adverse_selection=True), policy),
        "missing_support_rejected": rejected(lambda: ips_value(
            [LoggedAction(0, 0, 0, 1), LoggedAction(1, 0, 0, 1)], (1, 1))),
        "limitations": "one-step independent rewards, no funding/queue/latency; IPS needs known propensities and support, not just logged prices",
        "evidence_gate": synthetic_evidence(),
    })


if __name__ == "__main__":
    main()
