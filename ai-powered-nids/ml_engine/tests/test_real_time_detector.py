import sys
import unittest
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import real_time_detector as rtd  # noqa: E402
from train_model import FEATURE_COLUMNS  # noqa: E402


class FeatureExtractorTests(unittest.TestCase):
    def test_transform_orders_columns_like_training_data(self) -> None:
        extractor = rtd.ZeekFeatureExtractor(list(FEATURE_COLUMNS))
        frame = extractor.transform(
            {
                "proto": "TCP",
                "service": "ssl",
                "history": "ShADad",
                "duration": "1.5",
                "orig_bytes": 120,
                "resp_bytes": "-",
            }
        )
        self.assertEqual(list(frame.columns), list(FEATURE_COLUMNS))
        row = frame.iloc[0]
        self.assertEqual(row["proto_tcp"], 1.0)
        self.assertEqual(row["service_http"], 1.0)
        self.assertEqual(row["history_len"], 6.0)
        self.assertEqual(row["duration"], 1.5)
        self.assertEqual(row["resp_bytes"], 0.0)

    def test_mock_events_have_every_feature_input(self) -> None:
        extractor = rtd.ZeekFeatureExtractor(list(FEATURE_COLUMNS))
        frame = extractor.transform(rtd.generate_mock_zeek_event())
        self.assertFalse(frame.isna().any().any())


class ParsingTests(unittest.TestCase):
    def test_parse_json_line_accepts_objects_only(self) -> None:
        self.assertEqual(rtd.parse_json_line('{"uid": "C1"}'), {"uid": "C1"})
        self.assertIsNone(rtd.parse_json_line("[1, 2]"))
        with self.assertLogs(rtd.logger, level="WARNING"):
            self.assertIsNone(rtd.parse_json_line("not json"))


class DocumentTests(unittest.TestCase):
    def test_alert_severity_thresholds(self) -> None:
        event = {"id.orig_h": "192.168.1.5", "id.resp_h": "10.0.0.9", "ts": "2026-01-01T00:00:00+00:00"}
        self.assertEqual(rtd.build_alert_document(event, 0.95)["severity"], "critical")
        self.assertEqual(rtd.build_alert_document(event, 0.75)["severity"], "high")

    def test_flow_document_keeps_iso_timestamps(self) -> None:
        doc = rtd.build_flow_document({"ts": "2026-01-01T00:00:00+00:00"}, "benign", 0.1)
        self.assertEqual(doc["@timestamp"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(doc["prediction"], "benign")


if __name__ == "__main__":
    unittest.main()
