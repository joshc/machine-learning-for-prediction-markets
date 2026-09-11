"""Tiny train-only TF-IDF/logistic and six-unit neural baselines, no paid models."""

from prediction_market_lab.data import teaching_split
from prediction_market_lab.evidence import synthetic_evidence
from prediction_market_lab.models import fit_text_and_neural
from prediction_market_lab.reporting import emit

from _support import recurring


def main() -> None:
    emit("Invented bulletin text and a small CPU neural model", {
        "models": fit_text_and_neural(teaching_split(recurring())),
        "negative_conclusion": "template text is not a test of news understanding or pretrained-model leakage",
        "evidence_gate": synthetic_evidence(),
    })


if __name__ == "__main__":
    main()
