"""Synthetic recurring-event fixtures and strict point-in-time grouped splits."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import exp, isfinite
from pathlib import Path
from random import Random
from typing import Iterable

from .validation import available, parse_utc, probability, utc

SEED = 20260910


@dataclass(frozen=True)
class EventRow:
    event_id: str
    group_id: str
    family_id: str
    event_time: datetime
    published_at: datetime
    observed_at: datetime
    decision_at: datetime
    label_available_at: datetime
    signal: float
    horizon_hours: float
    market_probability: float
    outcome: int
    text: str
    synthetic: bool = True

    def __post_init__(self) -> None:
        for name in ("event_time", "published_at", "observed_at", "decision_at", "label_available_at"):
            utc(getattr(self, name))
        if not all((self.event_id, self.group_id, self.family_id)):
            raise ValueError("event and dependence-group identifiers are required")
        if not isfinite(self.signal) or not isfinite(self.horizon_hours) or self.horizon_hours < 0:
            raise ValueError("features must be finite; horizon cannot be negative")
        probability(self.market_probability)
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise ValueError("outcome must be binary; keep exceptional payout records elsewhere")
        if self.label_available_at < self.event_time:
            raise ValueError("outcome label cannot precede this fixture's event time")
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic flag must be explicit boolean")

    def features_available(self, cutoff: datetime | None = None) -> bool:
        return available(self.published_at, self.observed_at, cutoff or self.decision_at)


@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[EventRow, ...]
    validation: tuple[EventRow, ...]
    test: tuple[EventRow, ...]
    excluded_groups: tuple[str, ...]
    train_cutoff: datetime
    validation_cutoff: datetime
    test_cutoff: datetime


def grouped_chronological_split(rows: Iterable[EventRow], train_cutoff: datetime,
                                validation_cutoff: datetime, test_cutoff: datetime) -> TemporalSplit:
    """Purge straddling groups; labels must be known at each cohort's fit/evaluation cutoff.

    Grouping is caller-specified dependence knowledge, not inferred from similar text.
    Validation labels are available before calibrator fitting at validation_cutoff.
    Test labels are only consumed for retrospective evaluation at test_cutoff.
    """
    cutoffs = tuple(map(utc, (train_cutoff, validation_cutoff, test_cutoff)))
    if not cutoffs[0] < cutoffs[1] < cutoffs[2]:
        raise ValueError("cutoffs must strictly increase")
    groups: dict[str, list[EventRow]] = {}
    identities: dict[str, str] = {}
    labels: dict[str, int] = {}
    for row in rows:
        if not row.features_available():
            raise ValueError(f"feature unavailable at decision: {row.event_id}")
        if row.event_id in identities and identities[row.event_id] != row.group_id:
            raise ValueError("one event cannot be split across dependence groups")
        if row.event_id in labels and labels[row.event_id] != row.outcome:
            raise ValueError("one event has inconsistent binary labels")
        identities[row.event_id], labels[row.event_id] = row.group_id, row.outcome
        groups.setdefault(row.group_id, []).append(row)
    cohorts: list[list[EventRow]] = [[], [], []]
    excluded = []
    for group_id, observations in sorted(groups.items()):
        earliest = min(row.decision_at for row in observations)
        latest = max(row.decision_at for row in observations)
        assigned = False
        for index, cutoff in enumerate(cutoffs):
            lower_ok = index == 0 or earliest > cutoffs[index - 1]
            if lower_ok and latest <= cutoff and all(row.label_available_at <= cutoff for row in observations):
                cohorts[index].extend(observations)
                assigned = True
                break
        if not assigned:
            excluded.append(group_id)
    cohorts = [sorted(cohort, key=lambda row: (row.decision_at, row.event_id)) for cohort in cohorts]
    if any(not cohort for cohort in cohorts):
        raise ValueError("each split needs at least one complete, available dependence group")
    return TemporalSplit(*(tuple(cohort) for cohort in cohorts), tuple(excluded), *cutoffs)


def latest_per_event(rows: Iterable[EventRow]) -> tuple[EventRow, ...]:
    """One latest decision per event; choice of horizon must be declared in advance."""
    latest: dict[str, EventRow] = {}
    for row in rows:
        if row.event_id not in latest or row.decision_at > latest[row.event_id].decision_at:
            latest[row.event_id] = row
    return tuple(sorted(latest.values(), key=lambda row: (row.decision_at, row.event_id)))


def generate_recurring(events: int = 180, seed: int = SEED) -> tuple[EventRow, ...]:
    """Original invented threshold-like process, two snapshots per realized event."""
    if isinstance(events, bool) or not isinstance(events, int) or events < 1:
        raise ValueError("positive integer event count required")
    rng = Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(events):
        event_at = start + timedelta(days=index, hours=18)
        latent = rng.uniform(-1.5, 1.5)
        true_p = 1 / (1 + exp(-0.9 * latent))
        outcome = int(rng.random() < true_p)
        for hours in (12, 3):
            decision = event_at - timedelta(hours=hours)
            signal = round(latent + rng.gauss(0, 0.8 if hours == 12 else 0.4), 6)
            market = round(min(0.95, max(0.05, 0.5 + 0.11 * signal + rng.gauss(0, 0.06))), 6)
            adjective = "improving" if signal > 0.3 else "weakening" if signal < -0.3 else "mixed"
            text = f"synthetic bulletin reports {adjective} conditions with {hours} hours remaining"
            rows.append(EventRow(f"toy-{index:04d}", f"toy-{index:04d}", "invented-recurring",
                                 event_at, decision - timedelta(minutes=10),
                                 decision - timedelta(minutes=2), decision,
                                 event_at + timedelta(hours=12), signal, float(hours), market,
                                 outcome, text, True))
    return tuple(rows)


def write_recurring(path: Path, events: int = 180, seed: int = SEED) -> None:
    rows = generate_recurring(events, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.isoformat()
            writer.writerow(record)


def load_recurring(path: Path) -> tuple[EventRow, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = []
    for record in records:
        for name in ("event_time", "published_at", "observed_at", "decision_at", "label_available_at"):
            record[name] = parse_utc(record[name])
        for name in ("signal", "horizon_hours", "market_probability"):
            record[name] = float(record[name])
        record["outcome"] = int(record["outcome"])
        if record["synthetic"] not in ("True", "False"):
            raise ValueError("invalid synthetic flag")
        record["synthetic"] = record["synthetic"] == "True"
        rows.append(EventRow(**record))
    return tuple(rows)


def teaching_split(rows: Iterable[EventRow]) -> TemporalSplit:
    return grouped_chronological_split(rows, parse_utc("2025-04-01T00:00:00Z"),
                                       parse_utc("2025-05-16T00:00:00Z"),
                                       parse_utc("2025-07-01T00:00:00Z"))
