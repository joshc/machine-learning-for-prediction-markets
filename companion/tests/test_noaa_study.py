import contextlib
import io
import json
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

from experiments import noaa_study as study
from prediction_market_lab.scoring import brier_score, log_loss


def record(month, average="400.00", ndays="20", sdev="0.50", uncertainty="0.10"):
    year, month_number = month.split("-")
    decimal = f"{int(year) + (int(month_number) - 0.5) / 12:.4f}"
    return study.parse_record(
        [year, month_number, decimal, average, "400.00", ndays, sdev, uncertainty], "test"
    )


class NOAAParserTests(unittest.TestCase):
    def test_exact_fields_comments_and_original_creation_string(self):
        raw = (b"# comment\n# File Creation: Sat Sep  5 03:55:38 2026\n"
               b"1975 1 1975.0417 330.73 330.84 29 0.43 0.15\n")
        rows, creation = study.parse_source(raw)
        self.assertEqual(creation, "Sat Sep  5 03:55:38 2026")
        self.assertEqual(rows[0].average, 330.73)
        self.assertEqual(rows, study.parse_csv(study.observations_csv(rows)))
        for invalid in (raw.replace(b"0.15\n", b"0.15 extra\n"),
                        raw.replace(b"0.15\n", b"\n"), b"# only comments\n",
                        raw.replace(b"330.73", b"nan"),
                        raw.replace(b"0.43", b"inf"),
                        raw.replace(b"1975.0417", b"1976.0417"),
                        raw.replace(b" 29 ", b" 32 "),
                        raw.replace(b" 29 ", b" 29.0 ")):
            with self.subTest(invalid=invalid), self.assertRaises(study.StudyError):
                study.parse_source(invalid)

    def test_dates_duplicates_and_order(self):
        for invalid in ("1975-00", "1975-13", "0000-01", "1975-1", "1975/01"):
            with self.subTest(invalid=invalid), self.assertRaises(study.StudyError):
                study.parse_month(invalid)
        for rows in ([record("1975-01"), record("1975-01")],
                     [record("1975-02"), record("1975-01")]):
            with self.subTest(rows=rows), self.assertRaises(study.StudyError):
                study.validate_order(rows)
        self.assertEqual(study.month_string(study.parse_month("2023-12") + 1), "2024-01")
        with self.assertRaises(study.StudyError):
            record("2023-02", ndays="29")
        self.assertFalse(record("2024-02", ndays="29").invalid_reasons)

    def test_flagged_rows_are_preserved_but_not_usable(self):
        for kwargs in ({"average": "-99.99"}, {"ndays": "-1"}, {"ndays": "0"},
                       {"sdev": "-9.99"}, {"uncertainty": "-0.01"}):
            with self.subTest(kwargs=kwargs):
                row = record("2020-02", **kwargs)
                self.assertTrue(row.invalid_reasons)
                self.assertEqual([row], study.parse_csv(study.observations_csv([row])))
                rows = [record("2020-01"), row, record("2020-03")]
                events, excluded = study.build_events(rows, row.index, row.index + 1)
                self.assertEqual(events, [])
                self.assertEqual(len(excluded), 2)

    def test_gaps_do_not_shift_targets_and_ties_are_no(self):
        rows = [record("2020-01"), record("2020-03"), record("2020-04")]
        events, excluded = study.build_events(rows, study.parse_month("2020-02"),
                                             study.parse_month("2020-04"))
        self.assertEqual([study.month_string(event.target) for event in events], ["2020-04"])
        self.assertEqual(events[0].label, 0)
        self.assertEqual(excluded[0]["reasons"], ["target_missing"])
        self.assertEqual(excluded[1]["reasons"], ["previous_missing"])
        with self.assertRaises(study.StudyError):
            study.validate_extract(rows, study.load_protocol())

    def test_alternate_location_removes_both_pair_roles(self):
        indices = range(study.parse_month("2022-11"), study.parse_month("2023-10"))
        rows = [record(study.month_string(index)) for index in indices]
        blocked = frozenset(range(study.parse_month("2022-12"), study.parse_month("2023-08")))
        events, excluded = study.build_events(rows, study.parse_month("2022-12"),
                                             study.parse_month("2023-09"), blocked)
        self.assertEqual([study.month_string(event.target) for event in events], ["2023-09"])
        self.assertEqual(len(excluded), 9)
        self.assertEqual(excluded[-1]["target_month"], "2023-08")
        self.assertEqual(excluded[-1]["reasons"], ["previous_alternate_location"])


class NOAAClockTests(unittest.TestCase):
    def setUp(self):
        start = study.parse_month("2000-01")
        self.events = [study.Event(index, index - 1, int(index % 3 == 0))
                       for index in range(start, start + 60)]
        self.target = study.parse_month("2004-01")

    def test_exact_admission_boundary_and_no_future_labels(self):
        result = study.forecast(self.events, self.target)
        admitted = [event for event in self.events if event.target <= self.target - 2]
        self.assertEqual(result["admitted_labels"], 47)
        self.assertEqual(result["latest_admitted_target"], "2003-11")
        self.assertEqual(result["training_cutoff_month"], "2003-11")
        self.assertEqual(result["p_overall"], (sum(event.label for event in admitted) + 1) / 49)
        seasonal = [event for event in admitted if event.target % 12 == self.target % 12]
        self.assertEqual(result["seasonal_labels"], 4)
        self.assertEqual(result["p_seasonal"], (sum(event.label for event in seasonal) + 1) / 6)
        december = study.Event(study.parse_month("2003-12"), study.parse_month("2003-11"), 1)
        self.assertEqual(study.start_utc(december.available_at()), "2004-02-01T00:00:00Z")
        mutated = [replace(event, label=1 - event.label) if event.target >= self.target - 1
                   else event for event in self.events]
        self.assertEqual(result, study.forecast(mutated, self.target))
        self.assertEqual(result, study.forecast(admitted, self.target))

    def test_minimum_history_fails_explicitly(self):
        with self.assertRaisesRegex(study.StudyError, "insufficient admitted labels"):
            study.forecast(self.events, study.parse_month("2002-01"))

    def test_predictor_uses_labels_and_calendar_not_concentration(self):
        # Monotone positive transformations preserve labels but change every level.
        rows = [record(study.month_string(event.target), str(400 + i % 4))
                for i, event in enumerate(self.events)]
        changed = [replace(row, average=row.average * 2 + 100) for row in rows]
        start, end = rows[0].index, rows[-1].index
        original_events, _ = study.build_events(rows, start, end)
        transformed_events, _ = study.build_events(changed, start, end)
        self.assertEqual(original_events, transformed_events)
        self.assertEqual(study.forecast(original_events, self.target),
                         study.forecast(transformed_events, self.target))


class NOAAAcquisitionTests(unittest.TestCase):
    def test_existing_data_requires_opt_in_before_network(self):
        with patch.object(study, "download_source", side_effect=AssertionError("network")) as network:
            with self.assertRaisesRegex(study.StudyError, "already exists"):
                study.acquire_data(study.load_protocol())
            network.assert_not_called()

    def test_missing_data_has_no_synthetic_or_network_fallback(self):
        with patch.object(study, "download_source", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(study.StudyError, "No synthetic fallback"):
                study.load_data(study.load_protocol(), study.ROOT / "experiments" / "not-a-dataset.csv")

    def test_response_is_bounded_and_timeout_is_explicit(self):
        for payload, length in ((b"x" * (study.MAX_RESPONSE_BYTES + 1), None),
                                (b"", None), (b"x", "999999999"), (b"x", "bad")):
            with self.subTest(length=length, size=len(payload)):
                response = Mock(status=200, headers={"Content-Length": length})
                response.geturl.return_value = study.SOURCE_URL
                response.read.return_value = payload
                manager = Mock()
                manager.__enter__ = Mock(return_value=response)
                manager.__exit__ = Mock(return_value=False)
                opener = Mock()
                opener.open.return_value = manager
                with patch.object(study, "build_opener", return_value=opener):
                    with self.assertRaises(study.StudyError):
                        study.download_source()
                self.assertEqual(opener.open.call_args.kwargs["timeout"], study.TIMEOUT_SECONDS)
                if length is None:
                    response.read.assert_called_once_with(study.MAX_RESPONSE_BYTES + 1)

    def test_http_failure_redirects_and_flag_arguments_fail_closed(self):
        opener = Mock()
        opener.open.side_effect = URLError("test failure")
        with patch.object(study, "build_opener", return_value=opener):
            with self.assertRaisesRegex(study.StudyError, "acquisition failed"):
                study.download_source()
        with self.assertRaisesRegex(study.StudyError, "redirect"):
            study.NoRedirects().redirect_request(None, None, 302, "", {}, "https://elsewhere.test")
        for flags in (["--replace-data"], ["--replace-results"],
                      ["--refresh-data", "--freeze-results"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    study.main(flags)
                self.assertEqual(raised.exception.code, 2)


class NOAAFrozenStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = study.load_protocol()
        cls.rows, cls.provenance = study.load_data(cls.protocol)
        cls.report = json.loads((study.RESULTS_DIR / "report.json").read_text(encoding="utf-8"))

    def test_known_observational_hash_and_counts(self):
        self.assertEqual(study.sha256(study.DATA_PATH.read_bytes()),
                         "e3f26607a312e892d4a835d88db37d57c168039f048d2987a638b717c8ef51a4")
        self.assertEqual(len(self.rows), 612)
        self.assertEqual(sum(not row.invalid_reasons for row in self.rows), 610)
        self.assertEqual([study.month_string(row.index) for row in self.rows if row.invalid_reasons],
                         ["1975-12", "1984-04"])
        self.assertEqual(self.provenance["source_creation_string"], "Sat Sep  5 03:55:38 2026")
        self.assertEqual(self.report["usable_primary_events_all_years"], 607)
        self.assertEqual(self.report["initial_pre_evaluation_events"], 475)
        self.assertEqual(self.report["initial_admitted_labels"], 474)
        self.assertEqual(self.report["primary"]["count"], 132)
        self.assertEqual(self.report["sensitivity"]["count"], 123)
        self.assertEqual(self.report["primary_evaluation_exclusions"], [])

    def test_all_prediction_rows_match_core_scores_and_cutoffs(self):
        import csv

        rows = list(csv.DictReader(io.StringIO((study.RESULTS_DIR / "predictions.csv").read_text())))
        for row in rows:
            target = study.parse_month(row["target_month"])
            self.assertLessEqual(study.parse_month(row["latest_admitted_target"]), target - 2)
            self.assertEqual(row["label_available_at_utc"], study.start_utc(target + 2))
            for model in ("overall", "seasonal"):
                p, y = float(row[f"p_{model}"]), int(row["label"])
                self.assertTrue(0 < p < 1)
                self.assertEqual(float(row[f"brier_{model}"]), brier_score([y], [p]))
                self.assertEqual(float(row[f"log_loss_{model}"]), log_loss([y], [p]))
        self.assertEqual(len({row["target_month"] for row in rows}), 132)

    def test_frozen_protocol_precedes_actual_acquisition_and_evaluation(self):
        frozen = json.loads((study.RESULTS_DIR / "frozen_run.json").read_text())
        self.assertLess(self.protocol["predeclared_at_utc"], self.provenance["acquired_at_utc"])
        self.assertLessEqual(self.provenance["acquired_at_utc"], frozen["frozen_at_utc"])
        self.assertEqual(frozen["protocol_id"], self.protocol["protocol_id"])
        self.assertEqual(frozen["inputs"], study.input_hashes())

    @unittest.skipUnless(shutil.which("git"), "Git is needed to inspect checkout filters.")
    def test_git_filters_preserve_frozen_bytes(self):
        repository = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=study.ROOT, capture_output=True, text=True, check=False, timeout=10,
        )
        if repository.returncode:
            if "not a git repository" in repository.stderr.lower():
                self.skipTest("Source archive has no Git checkout.")
            self.fail(f"Cannot inspect Git checkout: {repository.stderr.strip()}")
        git_root = Path(repository.stdout.strip()).resolve()
        paths = [
            study.DATA_PATH,
            study.ROOT / "data" / "real" / "provenance.json",
            study.ROOT / "experiments" / "protocol.json",
            study.ROOT / "experiments" / "noaa_study.py",
            study.ROOT / "src" / "prediction_market_lab" / "scoring.py",
            *[study.RESULTS_DIR / name for name in (
                "predictions.csv", "sensitivity_predictions.csv", "report.json",
                "frozen_run.json",
            )],
        ]
        for path in paths:
            relative = path.resolve().relative_to(git_root).as_posix()
            with self.subTest(path=relative):
                unfiltered = subprocess.run(
                    ["git", "hash-object", "--no-filters", str(path)],
                    cwd=git_root, capture_output=True, text=True, check=True, timeout=10,
                ).stdout.strip()
                filtered = subprocess.run(
                    ["git", "hash-object", f"--path={relative}", str(path)],
                    cwd=git_root, capture_output=True, text=True, check=True, timeout=10,
                ).stdout.strip()
                self.assertEqual(filtered, unfiltered, "Git would change frozen file bytes.")

    def test_actual_offline_recomputation_never_downloads_or_writes(self):
        with patch.object(study, "download_source", side_effect=AssertionError("network")), \
             patch.object(study, "build_opener", side_effect=AssertionError("network")), \
             patch.object(Path, "write_bytes", side_effect=AssertionError("write")):
            self.assertEqual(study.run_offline(), self.report)
        result = subprocess.run(
            [sys.executable, str(study.ROOT / "experiments" / "noaa_study.py")],
            cwd=study.ROOT, text=True, capture_output=True, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VERIFIED OFFLINE", result.stdout)
        self.assertIn("primary: n=132", result.stdout)
        self.assertIn("sensitivity: n=123", result.stdout)

    def test_changed_input_or_output_fails_closed(self):
        original = Path.read_bytes
        for damaged in (study.DATA_PATH, study.RESULTS_DIR / "predictions.csv"):
            def changed(path, selected=damaged):
                content = original(path)
                return content + b"\n" if path == selected else content

            with self.subTest(path=damaged), patch.object(Path, "read_bytes", changed):
                with self.assertRaises(study.StudyError):
                    study.run_offline()
        with patch.object(study, "evaluate", side_effect=AssertionError("unexpected evaluation")):
            with self.assertRaisesRegex(study.StudyError, "already exist"):
                study.run_offline(freeze=True)

    def test_original_figure_matches_frozen_report(self):
        import xml.etree.ElementTree as ET
        from experiments.render_noaa_figure import ASSET_PATH, render_svg

        content = render_svg(self.report)
        self.assertEqual(ASSET_PATH.read_text(encoding="utf-8"), content)
        document = ET.fromstring(content)
        self.assertEqual(document.tag, "{http://www.w3.org/2000/svg}svg")
        for required in ("FINAL-VINTAGE", "Not a prediction-market backtest",
                         "n=132", "n=123", "no NOAA endorsement"):
            self.assertIn(required, content)


if __name__ == "__main__":
    unittest.main()
