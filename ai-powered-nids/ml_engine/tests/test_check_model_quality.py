import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import check_model_quality as gate  # noqa: E402


def metrics(recall: float, f1: float, fpr: float, threshold: float = 0.5) -> dict:
    return {"threshold": threshold, "recall": recall, "f1": f1, "false_positive_rate": fpr}


GOOD_REPORT = {
    "model_comparison": {
        "majority_baseline": metrics(0.0, 0.0, 0.0),
        "random_forest": metrics(0.97, 0.96, 0.01),
    },
    "threshold_analysis": {"deployed": metrics(0.95, 0.95, 0.0, threshold=0.7)},
}


class CheckReportTests(unittest.TestCase):
    def test_good_report_passes(self) -> None:
        self.assertEqual(gate.check_report(GOOD_REPORT), [])

    def test_reports_each_regression(self) -> None:
        report = copy.deepcopy(GOOD_REPORT)
        report["threshold_analysis"]["deployed"] = metrics(0.5, 0.6, 0.2, threshold=0.7)
        failures = gate.check_report(report)
        self.assertEqual(len(failures), 3)
        self.assertTrue(all("deployed threshold (0.7)" in f for f in failures))

    def test_must_beat_majority_baseline(self) -> None:
        report = copy.deepcopy(GOOD_REPORT)
        report["model_comparison"]["majority_baseline"] = metrics(1.0, 0.92, 1.0)
        failures = gate.check_report(report)
        self.assertEqual(len(failures), 1)
        self.assertIn("majority_baseline", failures[0])

    def test_thresholds_are_configurable(self) -> None:
        self.assertTrue(gate.check_report(GOOD_REPORT, min_recall=0.99))
        self.assertEqual(gate.check_report(GOOD_REPORT, max_fpr=0.01), [])

    def test_missing_model_or_metric_is_an_error(self) -> None:
        with self.assertRaises(gate.ReportError):
            gate.check_report(GOOD_REPORT, model="xgboost")
        report = copy.deepcopy(GOOD_REPORT)
        del report["threshold_analysis"]["deployed"]["recall"]
        with self.assertRaises(gate.ReportError):
            gate.check_report(report)


class MainTests(unittest.TestCase):
    def run_main(self, report: object) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                return gate.main(["--report", str(path)])

    def test_exit_codes(self) -> None:
        self.assertEqual(self.run_main(GOOD_REPORT), 0)
        bad = copy.deepcopy(GOOD_REPORT)
        bad["model_comparison"]["random_forest"]["recall"] = 0.1
        self.assertEqual(self.run_main(bad), 1)
        self.assertEqual(self.run_main(["not", "a", "report"]), 2)

    def test_missing_file_exits_2(self) -> None:
        with redirect_stderr(StringIO()):
            self.assertEqual(gate.main(["--report", "/nonexistent/report.json"]), 2)


if __name__ == "__main__":
    unittest.main()
