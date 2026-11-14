import contextlib
import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import real_time_detector as rtd  # noqa: E402
import score_logs  # noqa: E402
import train_model  # noqa: E402

SAMPLE_LOG = ML_ENGINE_DIR.parent / "zeek_logs" / "conn.log"

# Header and records in Zeek's default (non-JSON) ASCII conn.log format.
ZEEK_TSV = (
    "#separator \\x09\n"
    "#set_separator\t,\n"
    "#empty_field\t(empty)\n"
    "#unset_field\t-\n"
    "#path\tconn\n"
    "#open\t2026-06-10-21-30-00\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice"
    "\tduration\torig_bytes\tresp_bytes\tconn_state\tmissed_bytes\thistory"
    "\torig_pkts\tresp_pkts\n"
    "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring"
    "\tinterval\tcount\tcount\tstring\tcount\tstring\tcount\tcount\n"
    "1700000000.25\tCabc1\t192.168.1.20\t51544\t10.0.0.5\t443\ttcp\tssl"
    "\t1.23\t3200\t18400\tSF\t0\tShADadFf\t22\t31\n"
    "1700000001.5\tCabc2\t192.168.1.66\t40000\t10.0.0.8\t22\ttcp\t-"
    "\t-\t-\t-\tS0\t0\tS\t1\t0\n"
    "#close\t2026-06-10-22-00-00\n"
)


class BytesModel:
    """Deterministic stand-in: P(attack) grows with originator bytes."""

    def predict_proba(self, frame):
        score = float(np.clip(frame["orig_bytes"].iloc[0] / 100_000, 0.0, 1.0))
        return np.array([[1 - score, score]])


class ZeekParsingTests(unittest.TestCase):
    def test_tsv_records_use_declared_types_and_unset_marker(self) -> None:
        records = list(score_logs.iter_zeek_records(io.StringIO(ZEEK_TSV)))
        self.assertEqual(len(records), 2)
        first, second = records
        self.assertEqual(first["ts"], 1700000000.25)
        self.assertEqual(first["id.resp_p"], 443)
        self.assertEqual(first["orig_bytes"], 3200)
        self.assertEqual(first["history"], "ShADadFf")
        self.assertIsNone(second["service"])
        self.assertIsNone(second["duration"])

    def test_json_lines_and_bad_lines(self) -> None:
        lines = ['{"uid": "C1", "orig_bytes": 5}\n', "\n", "{broken\n", "plain text\n"]
        with self.assertLogs(score_logs.logger.name, level="WARNING"):
            with self.assertLogs(rtd.logger, level="WARNING"):
                records = list(score_logs.iter_zeek_records(lines))
        self.assertEqual(records, [{"uid": "C1", "orig_bytes": 5}, None, None])

    def test_tsv_record_with_wrong_field_count_is_skipped(self) -> None:
        text = "#fields\ta\tb\n1\t2\n1\t2\t3\n"
        with self.assertLogs(score_logs.logger.name, level="WARNING"):
            records = list(score_logs.iter_zeek_records(io.StringIO(text)))
        self.assertEqual(records, [{"a": "1", "b": "2"}, None])


class ScoringTests(unittest.TestCase):
    def test_score_records_counts_alerts_and_writes_documents(self) -> None:
        events = [
            {"id.orig_h": "192.168.1.5", "id.resp_p": 22, "orig_bytes": 95_000, "ts": 1},
            {"id.orig_h": "192.168.1.5", "id.resp_p": 80, "orig_bytes": 75_000, "ts": 2},
            {"id.orig_h": "192.168.1.9", "id.resp_p": 443, "orig_bytes": 1_000, "ts": 3},
            None,
        ]
        summary = score_logs.ScoreSummary()
        flows, alerts = io.StringIO(), io.StringIO()
        score_logs.score_records(
            events,
            BytesModel(),
            rtd.ZeekFeatureExtractor(list(train_model.FEATURE_COLUMNS)),
            threshold=0.7,
            summary=summary,
            flow_out=flows,
            alert_out=alerts,
        )

        result = summary.as_dict()
        self.assertEqual(result["flows_scored"], 3)
        self.assertEqual(result["records_skipped"], 1)
        self.assertEqual(result["alerts"], 2)
        self.assertEqual(result["severity"], {"critical": 1, "high": 1})
        self.assertEqual(result["top_alert_sources"], {"192.168.1.5": 2})
        self.assertEqual(result["top_alert_dst_ports"], {"22": 1, "80": 1})
        self.assertAlmostEqual(result["max_anomaly_score"], 0.95)

        flow_docs = [json.loads(line) for line in flows.getvalue().splitlines()]
        alert_docs = [json.loads(line) for line in alerts.getvalue().splitlines()]
        self.assertEqual([doc["prediction"] for doc in flow_docs], ["attack", "attack", "benign"])
        self.assertEqual(len(alert_docs), 2)
        self.assertTrue(flow_docs[0]["@timestamp"].startswith("1970-01-01T00:00:01"))


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.model_dir = root / "models"
        dataset = root / "flows.csv"
        train_model.generate_mock_cic_ids2017_dataset(dataset, normal_rows=150, attack_rows=100)
        train_model.main(["--dataset", str(dataset), "--model-dir", str(cls.model_dir)])
        cls.model = cls.model_dir / "random_forest_ids.pkl"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def run_cli(self, *args: str) -> tuple[int, dict | None, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = score_logs.main([*args, "--model", str(self.model)])
        output = stdout.getvalue()
        return code, json.loads(output) if output else None, stderr.getvalue()

    def test_scores_bundled_sample_log_and_flags_attacks(self) -> None:
        alerts_path = Path(self.temp_dir.name) / "alerts.jsonl"
        code, summary, _ = self.run_cli(str(SAMPLE_LOG), "--alerts", str(alerts_path))
        self.assertEqual(code, 0)
        self.assertEqual(summary["flows_scored"], 8)
        self.assertGreaterEqual(summary["alerts"], 1)
        alerts = [json.loads(line) for line in alerts_path.read_text().splitlines()]
        self.assertEqual(len(alerts), summary["alerts"])
        # The three bulk-transfer flows in the sample are the attack-like ones.
        self.assertIn("C100000007", {alert["uid"] for alert in alerts})

    def test_gzipped_tsv_and_fail_on_alert(self) -> None:
        path = Path(self.temp_dir.name) / "conn.00:00:00-01:00:00.log.gz"
        with gzip.open(path, "wt", encoding="utf-8") as file:
            file.write(ZEEK_TSV)
        output = Path(self.temp_dir.name) / "scored.jsonl"

        code, summary, _ = self.run_cli(str(path), "--output", str(output), "--threshold", "0")
        self.assertEqual(code, 0)
        self.assertEqual(summary["flows_scored"], 2)
        self.assertEqual(len(output.read_text().splitlines()), 2)

        code, summary, _ = self.run_cli(str(path), "--threshold", "0", "--fail-on-alert")
        self.assertEqual(code, 2)
        self.assertEqual(summary["alerts"], 2)

    def test_missing_log_and_missing_model_are_reported(self) -> None:
        code, summary, error = self.run_cli("/nonexistent/conn.log")
        self.assertEqual((code, summary), (1, None))
        self.assertIn("not found", error)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = score_logs.main([str(SAMPLE_LOG), "--model", "/nonexistent/model.pkl"])
        self.assertEqual(code, 1)
        self.assertIn("train_model.py", stderr.getvalue())

    def test_invalid_threshold_is_a_usage_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            score_logs.parse_args(["conn.log", "--threshold", "1.5"])


if __name__ == "__main__":
    unittest.main()
