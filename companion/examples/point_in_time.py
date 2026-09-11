"""Publication, observation and label clocks; group-purged chronological splits."""

from dataclasses import replace
from datetime import timedelta

from prediction_market_lab.data import teaching_split
from prediction_market_lab.reporting import emit
from prediction_market_lab.validation import available

from _support import recurring, rejected


def main() -> None:
    rows = recurring()
    split = teaching_split(rows)
    row = rows[0]
    late = replace(row, observed_at=row.decision_at + timedelta(seconds=1))
    emit("Point-in-time availability and shared-event grouping", {
        "fixture_rows": len(rows), "realized_events": len({r.event_id for r in rows}),
        "published_before_decision": row.published_at < row.decision_at,
        "observed_before_decision": row.observed_at < row.decision_at,
        "late_observation_available": available(late.published_at, late.observed_at, late.decision_at),
        "label_available_at_prediction": row.label_available_at <= row.decision_at,
        "events_per_cohort": {name: len({r.event_id for r in getattr(split, name)})
                              for name in ("train", "validation", "test")},
        "purged_label_or_boundary_groups": split.excluded_groups,
        "late_feature_rejected": rejected(lambda: teaching_split((late, *rows[1:]))),
        "limitation": "synthetic clocks are known by construction; current snapshots cannot recreate history",
    })


if __name__ == "__main__":
    main()
