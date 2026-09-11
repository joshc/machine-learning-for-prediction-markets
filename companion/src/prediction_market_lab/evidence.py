"""A conservative evidence boundary: synthetic demonstrations cannot establish live edge."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    synthetic: bool
    provenance: tuple[str, ...]
    point_in_time_verified: bool
    execution_history_verified: bool
    costs_verified: bool
    untouched_evaluation: bool
    dependence_and_search_accounted_for: bool
    uncertainty_supports_positive_net_result: bool


@dataclass(frozen=True)
class EvidenceGate:
    status: str
    eligible_for_research_review: bool
    reasons: tuple[str, ...]
    live_recommendation: bool = False


def assess_evidence(evidence: Evidence) -> EvidenceGate:
    reasons = []
    if evidence.synthetic:
        reasons.append("synthetic data cannot establish empirical live edge")
    if not evidence.provenance or any(not item.strip() for item in evidence.provenance):
        reasons.append("auditable real-data provenance is missing")
    checks = {
        "point-in-time data not verified": evidence.point_in_time_verified,
        "executable depth/fill history not verified": evidence.execution_history_verified,
        "applicable costs not verified": evidence.costs_verified,
        "untouched evaluation missing": evidence.untouched_evaluation,
        "dependence and strategy search not accounted for": evidence.dependence_and_search_accounted_for,
        "net result not supported by justified uncertainty analysis": evidence.uncertainty_supports_positive_net_result,
    }
    if any(type(value) is not bool for value in (evidence.synthetic, *checks.values())):
        raise ValueError("evidence declarations must be explicit booleans")
    reasons.extend(reason for reason, verified in checks.items() if not verified)
    if reasons:
        return EvidenceGate("REJECT_LIVE_EDGE_INFERENCE", False, tuple(reasons))
    return EvidenceGate("RESEARCH_REVIEW_REQUIRED", True,
                        ("declarations require independent audit; no automatic live promotion",))


def synthetic_evidence() -> EvidenceGate:
    return assess_evidence(Evidence(True, ("original seeded teaching fixture",),
                                    True, False, False, False, False, False))
