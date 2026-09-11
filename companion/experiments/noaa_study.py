"""Explicit NOAA acquisition and offline final-vintage forecasting study."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocol.json"
DATA_PATH = ROOT / "data" / "real" / "noaa_co2_monthly_1975_2025.csv"
PROVENANCE_PATH = ROOT / "data" / "real" / "provenance.json"
RESULTS_DIR = ROOT / "experiments" / "results"
SOURCE_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
MAX_RESPONSE_BYTES = 2_000_000
TIMEOUT_SECONDS = 30
FIELDS = ("year", "month", "decimal_date", "average", "deseasonalized",
          "ndays", "sdev", "uncertainty")
PREDICTION_FIELDS = (
    "target_month", "previous_month", "forecast_at_utc", "label_available_at_utc",
    "label", "training_cutoff_month", "latest_admitted_target", "admitted_labels",
    "admitted_yes", "seasonal_labels", "seasonal_yes", "p_overall", "p_seasonal",
    "brier_overall", "brier_seasonal", "brier_difference",
    "log_loss_overall", "log_loss_seasonal", "log_loss_difference",
)


class StudyError(ValueError):
    """Invalid data or a violated frozen-study contract."""


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def month_index(year: int, month: int) -> int:
    date(year, month, 1)
    return year * 12 + month - 1


def parse_month(value: str) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise StudyError(f"Expected YYYY-MM, got {value!r}")
    try:
        return month_index(int(value[:4]), int(value[5:]))
    except ValueError as exc:
        raise StudyError(f"Invalid calendar month: {value}") from exc


def month_string(index: int) -> str:
    year, month_zero = divmod(index, 12)
    return f"{year:04d}-{month_zero + 1:02d}"


def start_utc(index: int) -> str:
    return month_string(index) + "-01T00:00:00Z"


@dataclass(frozen=True)
class Observation:
    index: int
    average: float
    tokens: tuple[str, ...]
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    target: int
    previous: int
    label: int

    def available_at(self, lag: int = 2) -> int:
        return self.target + lag


def parse_record(tokens: list[str], location: str) -> Observation:
    if len(tokens) != len(FIELDS):
        raise StudyError(f"{location}: expected exactly eight fields, got {len(tokens)}")
    if any(not re.fullmatch(r"-?\d+", tokens[i]) for i in (0, 1, 5)):
        raise StudyError(f"{location}: year, month, and ndays must be integers")
    try:
        year, month, ndays = int(tokens[0]), int(tokens[1]), int(tokens[5])
        index = month_index(year, month)
        numbers = [float(value) for value in tokens]
    except (ValueError, OverflowError) as exc:
        raise StudyError(f"{location}: invalid date or numeric field") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise StudyError(f"{location}: every numeric field must be finite")
    if not year <= numbers[2] < year + 1:
        raise StudyError(f"{location}: decimal_date outside its calendar year")
    if ndays > calendar.monthrange(year, month)[1]:
        raise StudyError(f"{location}: ndays exceeds days in the calendar month")
    reasons = []
    if numbers[3] <= 0:
        reasons.append("nonpositive_average")
    if numbers[4] <= 0:
        reasons.append("nonpositive_deseasonalized")
    if ndays <= 0:
        reasons.append("flagged_or_zero_ndays")
    if numbers[6] < 0:
        reasons.append("negative_sdev")
    if numbers[7] < 0:
        reasons.append("negative_uncertainty")
    return Observation(index, numbers[3], tuple(tokens), tuple(reasons))


def validate_order(rows: list[Observation]) -> None:
    if not rows:
        raise StudyError("No monthly observations")
    for left, right in zip(rows, rows[1:]):
        if right.index <= left.index:
            raise StudyError(f"Duplicate or out-of-order month: {month_string(right.index)}")


def parse_source(raw: bytes) -> tuple[list[Observation], str | None]:
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise StudyError("Source response empty or exceeds bounded response size")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise StudyError("Expected ASCII NOAA text") from exc
    rows = []
    creation = None
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.search(r"file\s+creation(?:\s+date)?\s*:\s*(.+)", line, re.I)
            if match:
                if creation is not None:
                    raise StudyError("Multiple source creation header strings")
                creation = match.group(1).strip()
            continue
        rows.append(parse_record(line.split(), f"source line {number}"))
    validate_order(rows)
    return rows, creation


def observations_csv(rows: list[Observation]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(row.tokens for row in rows)
    return stream.getvalue().encode("ascii")


def parse_csv(content: bytes) -> list[Observation]:
    try:
        reader = csv.reader(io.StringIO(content.decode("ascii"), newline=""))
        if tuple(next(reader, ())) != FIELDS:
            raise StudyError("Frozen CSV schema does not match exact eight-field schema")
        rows = [parse_record(tokens, f"CSV line {i}") for i, tokens in enumerate(reader, 2)]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise StudyError("Invalid frozen CSV") from exc
    validate_order(rows)
    return rows


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"Cannot read predeclared protocol: {path}") from exc
    if protocol.get("protocol_id") != "noaa-monthly-increase-v1" or protocol.get("protocol_version") != 1:
        raise StudyError("This implementation supports only noaa-monthly-increase-v1")
    if protocol["source_url"] != SOURCE_URL:
        raise StudyError("Unexpected source URL in protocol")
    # These are implementation semantics, not user-tunable hyperparameters.
    if (protocol["training"]["label_lag_months"] != 2
            or protocol["training"]["minimum_admitted_labels"] != 36
            or protocol["scoring"]["calibration_bins"] != 5
            or protocol["selected_months"] != {
                "start": "1975-01", "end": "2025-12", "expected_calendar_rows": 612}
            or protocol["evaluation_months"] != {
                "start": "2015-01", "end": "2025-12", "expected_calendar_months": 132}
            or protocol["sensitivity"]["affected_record_months_start"] != "2022-12"
            or protocol["sensitivity"]["affected_record_months_end"] != "2023-07"):
        raise StudyError("Changed study semantics require a new reviewed protocol and implementation")
    return protocol


def validate_extract(rows: list[Observation], protocol: dict) -> None:
    selection = protocol["selected_months"]
    expected = list(range(parse_month(selection["start"]), parse_month(selection["end"]) + 1))
    if [row.index for row in rows] != expected or len(rows) != selection["expected_calendar_rows"]:
        raise StudyError("Expected every calendar record January 1975 through December 2025, including flags")


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise StudyError("Unexpected source redirect; inspect the official URL rather than following")


def download_source() -> tuple[bytes, str]:
    request = Request(SOURCE_URL, headers={"User-Agent": "OriginalBookNOAAStudy/1.0",
                                         "Accept": "text/plain"})
    try:
        with build_opener(NoRedirects()).open(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200 or response.geturl() != SOURCE_URL:
                raise StudyError("Unexpected HTTP status or source URL")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                raise StudyError("Source Content-Length invalid or exceeds limit")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if not raw or len(raw) > MAX_RESPONSE_BYTES:
                raise StudyError("Source response empty or exceeds size limit")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise StudyError(f"Public NOAA acquisition failed: {exc}") from exc
    return raw, utc_now()


def acquire_data(protocol: dict, replace: bool = False,
                 data_path: Path = DATA_PATH, provenance_path: Path = PROVENANCE_PATH) -> dict:
    if (data_path.exists() or provenance_path.exists()) and not replace:
        raise StudyError("Extract already exists; --refresh-data --replace-data is required to replace it")
    raw, acquired = download_source()
    rows, creation = parse_source(raw)
    if rows[0].index != parse_month("1958-03") or rows[-1].index < parse_month("2025-12"):
        raise StudyError("Unexpected source historical range; manual review required")
    start = parse_month(protocol["selected_months"]["start"])
    end = parse_month(protocol["selected_months"]["end"])
    selected = [row for row in rows if start <= row.index <= end]
    validate_extract(selected, protocol)
    content = observations_csv(selected)
    invalid = [{"month": month_string(row.index), "reasons": list(row.invalid_reasons)}
               for row in selected if row.invalid_reasons]
    provenance = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "documentation_url": "https://gml.noaa.gov/ccgg/trends/data.html",
        "use_policy_url": "https://gml.noaa.gov/about/disclaimer.html",
        "attribution": "NOAA Global Monitoring Laboratory, Carbon Cycle and Greenhouse Gases group",
        "citation_ids": ["NOAA_CO2", "NOAA_DATA", "NOAA_POLICY"],
        "acquired_at_utc": acquired,
        "source_creation_string": creation,
        "source_creation_timezone": "Not specified by source; not converted or inferred",
        "raw_response_sha256": sha256(raw),
        "raw_response_bytes": len(raw),
        "raw_response_persisted": False,
        "raw_calendar_rows": len(rows),
        "raw_first_month": month_string(rows[0].index),
        "raw_last_month": month_string(rows[-1].index),
        "excluded_before_selection": sum(row.index < start for row in rows),
        "excluded_after_selection": sum(row.index > end for row in rows),
        "frozen_csv_sha256": sha256(content),
        "selected_calendar_rows": len(selected),
        "selected_valid_records": len(selected) - len(invalid),
        "selected_flagged_records": invalid,
        "selection": "Keep January 1975 through December 2025 inclusive; preserve original field tokens, including flagged records. No entire raw response persisted.",
        "rights_boundary": "Exclude early Scripps records (March 1958-April 1974) rather than assume NOAA government-work policy licenses third-party data. Conservative extract begins January 1975.",
        "rights": "NOAA government data under stated public-domain policy, subject to attribution, no endorsement, and labeling transformations. Third-party exceptions are not generalized away.",
        "transformation_notice": "Whitespace-to-CSV normalization and derived study outputs are original transformations, not official NOAA products.",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()),
        "csv_encoding": "ASCII",
        "csv_line_endings": "LF",
        "csv_columns_in_order": list(FIELDS),
        "csv_schema": {
            "year": "integer observation year",
            "month": "integer calendar month 1-12",
            "decimal_date": "finite source decimal year, not availability time",
            "average": "finite monthly mean CO2 ppm; nonpositive is unusable",
            "deseasonalized": "finite source adjusted CO2 ppm; retained, never a feature",
            "ndays": "integer contributing days; <=0 unusable, cannot exceed calendar length",
            "sdev": "finite daily standard deviation ppm; negative is a source flag",
            "uncertainty": "finite monthly-mean uncertainty ppm; negative is a source flag",
        },
        "location_discontinuity": "December 2022-July 4, 2023 measurements at Maunakea; sensitivity excludes full December 2022-July 2023 source months and dependent target pairs.",
        "vintage_caution": "Mutable current/final-vintage extract, including revisions, interpolation flags and center-of-month correction. No historical release archive.",
    }
    data_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(content)
    provenance_path.write_bytes(json_bytes(provenance))
    return provenance


def load_data(protocol: dict, data_path: Path = DATA_PATH,
              provenance_path: Path = PROVENANCE_PATH) -> tuple[list[Observation], dict]:
    try:
        content = data_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError("Missing/unreadable frozen data or provenance; explicit --refresh-data is required. No synthetic fallback.") from exc
    if sha256(content) != provenance.get("frozen_csv_sha256"):
        raise StudyError("Frozen CSV SHA-256 mismatch")
    if (provenance.get("source_url") != SOURCE_URL
            or provenance.get("protocol_id") != protocol["protocol_id"]
            or provenance.get("protocol_sha256") != sha256(PROTOCOL_PATH.read_bytes())
            or provenance.get("csv_columns_in_order") != list(FIELDS)):
        raise StudyError("Provenance does not match protocol or schema")
    rows = parse_csv(content)
    validate_extract(rows, protocol)
    invalid = [{"month": month_string(row.index), "reasons": list(row.invalid_reasons)}
               for row in rows if row.invalid_reasons]
    if (provenance.get("selected_calendar_rows") != len(rows)
            or provenance.get("selected_flagged_records") != invalid
            or provenance.get("selected_valid_records") != len(rows) - len(invalid)):
        raise StudyError("Provenance counts/flags do not match extracted observations")
    return rows, provenance


def build_events(rows: list[Observation], start: int, end: int,
                 excluded_records: frozenset[int] = frozenset()) -> tuple[list[Event], list[dict]]:
    validate_order(rows)
    by_month = {row.index: row for row in rows}
    events, exclusions = [], []
    for target in range(start, end + 1):
        reasons = []
        for role, index in (("target", target), ("previous", target - 1)):
            record = by_month.get(index)
            if record is None:
                reasons.append(f"{role}_missing")
            elif record.invalid_reasons:
                reasons.extend(f"{role}_{reason}" for reason in record.invalid_reasons)
            if index in excluded_records:
                reasons.append(f"{role}_alternate_location")
        if reasons:
            exclusions.append({"target_month": month_string(target), "reasons": reasons})
        else:
            events.append(Event(target, target - 1,
                                int(by_month[target].average > by_month[target - 1].average)))
    return events, exclusions


def forecast(events: list[Event], target: int, minimum: int = 36, lag: int = 2) -> dict:
    admitted = [event for event in events if event.available_at(lag) <= target]
    if len(admitted) < minimum:
        raise StudyError(f"{month_string(target)}: insufficient admitted labels: {len(admitted)} < {minimum}")
    if any(event.target >= target for event in admitted):
        raise StudyError("Future/current label admitted")
    seasonal = [event for event in admitted if event.target % 12 == target % 12]
    yes = sum(event.label for event in admitted)
    seasonal_yes = sum(event.label for event in seasonal)
    return {
        "training_cutoff_month": month_string(target - lag),
        "latest_admitted_target": month_string(max(event.target for event in admitted)),
        "admitted_labels": len(admitted),
        "admitted_yes": yes,
        "seasonal_labels": len(seasonal),
        "seasonal_yes": seasonal_yes,
        "p_overall": (yes + 1) / (len(admitted) + 2),
        "p_seasonal": (seasonal_yes + 1) / (len(seasonal) + 2),
    }


def predict(events: list[Event], start: int, end: int, protocol: dict) -> list[dict]:
    from prediction_market_lab.scoring import brier_score, log_loss

    predictions = []
    for event in events:
        if not start <= event.target <= end:
            continue
        result = {
            "target_month": month_string(event.target),
            "previous_month": month_string(event.previous),
            "forecast_at_utc": start_utc(event.target),
            "label_available_at_utc": start_utc(event.available_at()),
            "label": event.label,
            **forecast(events, event.target, protocol["training"]["minimum_admitted_labels"],
                       protocol["training"]["label_lag_months"]),
        }
        for name, score in (("brier", brier_score), ("log_loss", log_loss)):
            for model in ("overall", "seasonal"):
                result[f"{name}_{model}"] = score([event.label], [result[f"p_{model}"]])
            result[f"{name}_difference"] = result[f"{name}_seasonal"] - result[f"{name}_overall"]
        predictions.append(result)
    if not predictions:
        raise StudyError("No usable evaluation months")
    return predictions


def summarize(predictions: list[dict]) -> dict:
    from prediction_market_lab.scoring import brier_score, calibration_bins, log_loss

    if not predictions:
        raise StudyError("Cannot score an empty cohort")
    labels = [row["label"] for row in predictions]
    result = {
        "count": len(labels),
        "yes_count": sum(labels),
        "observed_yes_rate": sum(labels) / len(labels),
        "first_target": predictions[0]["target_month"],
        "last_target": predictions[-1]["target_month"],
        "year_counts": dict(sorted(Counter(row["target_month"][:4] for row in predictions).items())),
        "admitted_labels_range": [min(row["admitted_labels"] for row in predictions),
                                  max(row["admitted_labels"] for row in predictions)],
        "seasonal_labels_range": [min(row["seasonal_labels"] for row in predictions),
                                  max(row["seasonal_labels"] for row in predictions)],
        "models": {},
    }
    for model in ("overall", "seasonal"):
        probabilities = [row[f"p_{model}"] for row in predictions]
        result["models"][model] = {
            "brier": brier_score(labels, probabilities),
            "log_loss": log_loss(labels, probabilities),
            "probability_range": [min(probabilities), max(probabilities)],
            "mean_probability": sum(probabilities) / len(probabilities),
            "calibration_bins": [asdict(item) for item in calibration_bins(labels, probabilities, bins=5)],
        }
    result["paired_difference_seasonal_minus_overall"] = {
        score: sum(row[f"{score}_difference"] for row in predictions) / len(predictions)
        for score in ("brier", "log_loss")
    }
    return result


def predictions_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def evaluate(protocol: dict, rows: list[Observation], provenance: dict) -> dict[str, bytes]:
    start, end = parse_month(protocol["selected_months"]["start"]), parse_month(protocol["selected_months"]["end"])
    eval_start, eval_end = parse_month(protocol["evaluation_months"]["start"]), parse_month(protocol["evaluation_months"]["end"])
    location_start = parse_month(protocol["sensitivity"]["affected_record_months_start"])
    location_end = parse_month(protocol["sensitivity"]["affected_record_months_end"])
    blocked = frozenset(range(location_start, location_end + 1))
    events, exclusions = build_events(rows, start, end)
    sensitivity_events, sensitivity_exclusions = build_events(rows, start, end, blocked)
    primary = predict(events, eval_start, eval_end, protocol)
    sensitivity = predict(sensitivity_events, eval_start, eval_end, protocol)
    sensitivity_targets = {row["target_month"] for row in sensitivity}
    matched_primary = [row for row in primary if row["target_month"] in sensitivity_targets]
    eval_exclusions = lambda items: [
        item for item in items if eval_start <= parse_month(item["target_month"]) <= eval_end
    ]
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()),
        "frozen_csv_sha256": provenance["frozen_csv_sha256"],
        "source_acquired_at_utc": provenance["acquired_at_utc"],
        "study_status": "Retrospective FINAL-VINTAGE observational forecast scoring; not prediction-market backtest or profitability evidence.",
        "hypothetical_clock": protocol["training"]["availability"],
        "models": protocol["models"],
        "score_conventions": protocol["scoring"],
        "selected_calendar_rows": len(rows),
        "usable_primary_events_all_years": len(events),
        "initial_pre_evaluation_events": sum(event.target < eval_start for event in events),
        "initial_admitted_labels": primary[0]["admitted_labels"],
        "flagged_records": provenance["selected_flagged_records"],
        "primary": summarize(primary),
        "primary_exclusions_all_years": exclusions,
        "primary_evaluation_exclusions": eval_exclusions(exclusions),
        "sensitivity": summarize(sensitivity),
        "sensitivity_definition": protocol["sensitivity"],
        "sensitivity_exclusions_all_years": sensitivity_exclusions,
        "sensitivity_evaluation_exclusions": eval_exclusions(sensitivity_exclusions),
        "primary_on_sensitivity_cohort": summarize(matched_primary),
        "sensitivity_removed_primary_evaluation_months": [
            row["target_month"] for row in primary if row["target_month"] not in sensitivity_targets
        ],
        "limitations": protocol["interpretation"] + [
            "Only eleven evaluation years, dependent monthly labels and repeated seasonal structure; no inferential uncertainty claim.",
            "The immediately preceding mean is used solely to define the eventual outcome, not as an as-of predictor.",
            "Availability dates are invented index conventions; actual historical publication dates are unavailable.",
            "Final-vintage revisions or provider processing may change threshold labels; withholding future labels does not remove vintage leakage.",
            "Sensitivity changes both the evaluation cohort and later training histories; matched primary scores isolate the cohort change."
        ],
    }
    return {
        "predictions.csv": predictions_csv(primary),
        "sensitivity_predictions.csv": predictions_csv(sensitivity),
        "report.json": json_bytes(report),
    }


def input_hashes() -> dict[str, str]:
    from prediction_market_lab import scoring

    return {
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()),
        "data_sha256": sha256(DATA_PATH.read_bytes()),
        "provenance_sha256": sha256(PROVENANCE_PATH.read_bytes()),
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "shared_scoring_sha256": sha256(Path(scoring.__file__).read_bytes()),
    }


def verify_inputs(frozen: dict) -> None:
    if frozen.get("inputs") != input_hashes():
        raise StudyError("Frozen run input/code hashes changed; no evaluation or replacement performed. Inspect changes before explicit --freeze-results --replace-results.")


def run_offline(freeze: bool = False, replace: bool = False) -> dict:
    protocol = load_protocol()
    rows, provenance = load_data(protocol)
    manifest_path = RESULTS_DIR / "frozen_run.json"
    if freeze:
        if any(RESULTS_DIR.glob("*")) and not replace:
            raise StudyError("Frozen results already exist; --freeze-results --replace-results is required")
    else:
        try:
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudyError("No readable frozen evaluation; first run requires explicit --freeze-results") from exc
        verify_inputs(frozen)
    outputs = evaluate(protocol, rows, provenance)
    if freeze:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (RESULTS_DIR / name).write_bytes(content)
        frozen = {
            "frozen_at_utc": utc_now(),
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["protocol_version"],
            "inputs": input_hashes(),
            "outputs": {name: sha256(content) for name, content in outputs.items()},
            "run_policy": "One frozen evaluation plus its predeclared sensitivity; replacement requires explicit opt-in. Protocol is never rewritten by the CLI.",
        }
        manifest_path.write_bytes(json_bytes(frozen))
    else:
        expected = {name: sha256(content) for name, content in outputs.items()}
        if frozen.get("outputs") != expected:
            raise StudyError("Recomputed results differ from frozen output hashes")
        for name, content in outputs.items():
            try:
                stored = (RESULTS_DIR / name).read_bytes()
            except OSError as exc:
                raise StudyError(f"Missing frozen result: {name}") from exc
            if stored != content:
                raise StudyError(f"Stored frozen result differs from exact offline recomputation: {name}")
    return json.loads(outputs["report.json"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh-data", action="store_true", help="Explicitly fetch public NOAA source; acquire only, do not score")
    mode.add_argument("--freeze-results", action="store_true", help="Explicitly persist the evaluation and sensitivity")
    parser.add_argument("--replace-data", action="store_true", help="With --refresh-data, allow replacement of existing extract")
    parser.add_argument("--replace-results", action="store_true", help="With --freeze-results, allow replacement of frozen outputs")
    args = parser.parse_args(argv)
    if args.replace_data and not args.refresh_data:
        parser.error("--replace-data requires --refresh-data")
    if args.replace_results and not args.freeze_results:
        parser.error("--replace-results requires --freeze-results")
    try:
        if args.refresh_data:
            provenance = acquire_data(load_protocol(), replace=args.replace_data)
            print(f"Acquired {provenance['selected_calendar_rows']} NOAA calendar records at {provenance['acquired_at_utc']}")
            print(f"Frozen CSV SHA-256: {provenance['frozen_csv_sha256']}")
            print("Acquisition only; no scores computed and no frozen results replaced.")
        else:
            report = run_offline(freeze=args.freeze_results, replace=args.replace_results)
            print("FROZEN" if args.freeze_results else "VERIFIED OFFLINE (no files changed)")
            print("FINAL-VINTAGE retrospective teaching study; NOT a trading backtest.")
            for name in ("primary", "sensitivity"):
                section = report[name]
                print(f"{name}: n={section['count']}, YES={section['yes_count']}")
                for model in ("overall", "seasonal"):
                    scores = section["models"][model]
                    print(f"  {model}: Brier={scores['brier']:.9f}, log loss={scores['log_loss']:.9f}")
                delta = section["paired_difference_seasonal_minus_overall"]
                print(f"  seasonal minus overall: Brier={delta['brier']:.9f}, log loss={delta['log_loss']:.9f}")
    except (StudyError, ImportError, OSError) as exc:
        print(f"Study error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
