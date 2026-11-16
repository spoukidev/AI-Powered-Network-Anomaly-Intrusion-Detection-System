"""Tests for alert deduplication in the real-time detector."""

import sys
import unittest
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import real_time_detector as rtd  # noqa: E402


def alert(src="10.0.0.5", dst="10.0.0.9", port=22, proto="tcp") -> dict:
    return {"src_ip": src, "dst_ip": dst, "dst_port": port, "proto": proto}


class AlertDeduplicatorTests(unittest.TestCase):
    def test_repeats_inside_window_are_suppressed_and_counted(self) -> None:
        dedup = rtd.AlertDeduplicator(60)
        first = alert()
        self.assertTrue(dedup.should_emit(first, 1000.0))
        self.assertEqual(first["suppressed_duplicates"], 0)
        for offset in (1, 10, 59.9):
            self.assertFalse(dedup.should_emit(alert(), 1000.0 + offset))
        later = alert()
        self.assertTrue(dedup.should_emit(later, 1060.0))
        self.assertEqual(later["suppressed_duplicates"], 3)
        self.assertFalse(dedup.should_emit(alert(), 1061.0))

    def test_different_flow_tuples_are_independent(self) -> None:
        dedup = rtd.AlertDeduplicator(60)
        self.assertTrue(dedup.should_emit(alert(), 0.0))
        self.assertTrue(dedup.should_emit(alert(port=23), 1.0))
        self.assertTrue(dedup.should_emit(alert(dst="10.0.0.10"), 1.0))
        self.assertTrue(dedup.should_emit(alert(proto="udp"), 1.0))
        self.assertFalse(dedup.should_emit(alert(port=23), 2.0))

    def test_zero_window_disables_deduplication(self) -> None:
        dedup = rtd.AlertDeduplicator(0)
        self.assertTrue(all(dedup.should_emit(alert(), 5.0) for _ in range(3)))

    def test_out_of_order_timestamps_do_not_suppress(self) -> None:
        dedup = rtd.AlertDeduplicator(60)
        self.assertTrue(dedup.should_emit(alert(), 1000.0))
        # An older event (replayed/out-of-order log) starts a fresh window.
        self.assertTrue(dedup.should_emit(alert(), 900.0))

    def test_memory_is_bounded(self) -> None:
        dedup = rtd.AlertDeduplicator(60, max_keys=10)
        for i in range(50):
            self.assertTrue(dedup.should_emit(alert(src=f"10.1.0.{i}"), 0.0))
        self.assertLessEqual(len(dedup._first_seen), 10)

    def test_expired_keys_are_pruned_first(self) -> None:
        dedup = rtd.AlertDeduplicator(60, max_keys=2)
        dedup.should_emit(alert(src="a"), 0.0)
        dedup.should_emit(alert(src="b"), 100.0)
        dedup.should_emit(alert(src="c"), 120.0)
        self.assertNotIn(dedup.key_for(alert(src="a")), dedup._first_seen)
        self.assertFalse(dedup.should_emit(alert(src="b"), 130.0))


class ParsingTests(unittest.TestCase):
    def test_parse_dedup_window(self) -> None:
        self.assertEqual(rtd.parse_dedup_window(None), 60.0)
        self.assertEqual(rtd.parse_dedup_window(" "), 60.0)
        self.assertEqual(rtd.parse_dedup_window("0"), 0.0)
        self.assertEqual(rtd.parse_dedup_window("300"), 300.0)
        for bad in ("-1", "abc", "nan", "inf"):
            with self.assertRaises(ValueError):
                rtd.parse_dedup_window(bad)

    def test_event_epoch(self) -> None:
        self.assertEqual(rtd.event_epoch({"ts": 1700000000.5}, 1.0), 1700000000.5)
        self.assertEqual(rtd.event_epoch({"ts": "1700000000"}, 1.0), 1700000000.0)
        self.assertEqual(rtd.event_epoch({}, 1.0), 1.0)
        self.assertEqual(rtd.event_epoch({"ts": "2024-01-01T00:00:00Z"}, 2.0), 1704067200.0)
        self.assertEqual(rtd.event_epoch({"ts": "2024-01-01T00:00:00+00:00"}, 2.0), 1704067200.0)
        self.assertEqual(rtd.event_epoch({"ts": "not a time"}, 2.0), 2.0)
        self.assertEqual(rtd.event_epoch({"ts": True}, 2.0), 2.0)
        self.assertEqual(rtd.event_epoch({"ts": float("nan")}, 3.0), 3.0)


if __name__ == "__main__":
    unittest.main()
