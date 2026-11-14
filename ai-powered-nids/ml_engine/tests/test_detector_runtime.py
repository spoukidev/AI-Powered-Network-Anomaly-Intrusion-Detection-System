"""Tests for the detector's runtime plumbing: tailing, config and Elasticsearch setup."""

import os
import pickle
import queue
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import real_time_detector as rtd  # noqa: E402
from train_model import FEATURE_COLUMNS  # noqa: E402


class Tailer:
    """Run ``tail_file`` in a daemon thread and collect what it yields."""

    def __init__(self, path: Path, **kwargs) -> None:
        self.lines: queue.Queue[str] = queue.Queue()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(path,), kwargs=kwargs, daemon=True)
        self.thread.start()
        # tail_file is a generator, so it only opens/seeks once iteration starts.
        # Give it a moment to reach its polling loop before the test writes.
        self.started.wait(5)
        time.sleep(0.1)

    def _run(self, path: Path, **kwargs) -> None:
        iterator = rtd.tail_file(path, poll_interval=0.01, **kwargs)
        self.started.set()
        for line in iterator:
            self.lines.put(line)

    def take(self, count: int, timeout: float = 5.0) -> list[str]:
        return [self.lines.get(timeout=timeout) for _ in range(count)]

    def nothing_yet(self, wait: float = 0.2) -> bool:
        time.sleep(wait)
        return self.lines.empty()


class TailFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "conn.log"
        rtd.shutdown_event.clear()
        self.tailers: list[Tailer] = []

    def tearDown(self) -> None:
        rtd.shutdown_event.set()
        for tailer in self.tailers:
            tailer.thread.join(5)
        rtd.shutdown_event.clear()
        self.temp_dir.cleanup()

    def tail(self, **kwargs) -> Tailer:
        tailer = Tailer(self.path, **kwargs)
        self.tailers.append(tailer)
        return tailer

    def append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text)

    def test_partial_lines_are_held_until_newline(self) -> None:
        tailer = self.tail()
        self.append('{"uid": "C1"')
        self.assertTrue(tailer.nothing_yet())
        self.append('}\n\n{"uid": "C2"}\n')
        self.assertEqual(tailer.take(2), ['{"uid": "C1"}', '{"uid": "C2"}'])

    def test_from_start_reads_existing_lines(self) -> None:
        self.path.write_text("a\nb\n", encoding="utf-8")
        self.assertEqual(self.tail(from_start=True).take(2), ["a", "b"])

    def test_default_skips_existing_lines(self) -> None:
        self.path.write_text("old\n", encoding="utf-8")
        tailer = self.tail()
        self.append("new\n")
        self.assertEqual(tailer.take(1), ["new"])

    def test_rotation_reopens_the_new_file(self) -> None:
        tailer = self.tail()
        self.append("before-rotation\n")
        self.assertEqual(tailer.take(1), ["before-rotation"])
        os.replace(self.path, self.path.with_suffix(".rotated"))
        self.path.write_text("after-rotation\n", encoding="utf-8")
        self.assertEqual(tailer.take(1), ["after-rotation"])

    def test_truncation_restarts_from_beginning(self) -> None:
        self.path.write_text("x" * 200 + "\n", encoding="utf-8")
        tailer = self.tail()
        self.path.write_text("short\n", encoding="utf-8")
        self.assertEqual(tailer.take(1), ["short"])

    def test_shutdown_stops_tailing(self) -> None:
        rtd.shutdown_event.set()
        self.assertEqual(list(rtd.tail_file(self.path, poll_interval=0.01)), [])


class ConfigTests(unittest.TestCase):
    def test_parse_threshold(self) -> None:
        self.assertEqual(rtd.parse_threshold(None), 0.70)
        self.assertEqual(rtd.parse_threshold(" "), 0.70)
        self.assertEqual(rtd.parse_threshold("0.85"), 0.85)
        self.assertEqual(rtd.parse_threshold("1"), 1.0)
        for bad in ("high", "1.5", "-0.1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rtd.parse_threshold(bad)

    def test_import_does_not_install_signal_handlers(self) -> None:
        self.assertIsNot(signal.getsignal(signal.SIGTERM), rtd.handle_shutdown)

    def test_boolean_ts_is_not_treated_as_epoch(self) -> None:
        with mock.patch.object(rtd, "now_iso", return_value="NOW"):
            self.assertEqual(rtd.event_timestamp({"ts": True}), "NOW")
            self.assertEqual(rtd.event_timestamp({"ts": ["bad"]}), "NOW")
            self.assertEqual(rtd.event_timestamp({}), "NOW")
        iso = "2026-01-01T00:00:00Z"
        self.assertEqual(rtd.event_timestamp({"ts": iso}), iso)


class FakeModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame):
        return np.array([[1 - self.probability, self.probability]] * len(frame))


class ModelLoadingTests(unittest.TestCase):
    def test_missing_model_file_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "train_model.py"):
            rtd.load_model_bundle(Path("/nonexistent/model.pkl"))

    def test_bundle_without_feature_columns_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            path.write_bytes(pickle.dumps({"model": object()}))
            with self.assertRaisesRegex(ValueError, "feature_columns"):
                rtd.load_model_bundle(path)

    def test_classify_event_uses_alert_threshold(self) -> None:
        extractor = rtd.ZeekFeatureExtractor(list(FEATURE_COLUMNS))
        with mock.patch.object(rtd, "ALERT_THRESHOLD", 0.7):
            self.assertEqual(rtd.classify_event({}, FakeModel(0.7), extractor), ("attack", 0.7))
            self.assertEqual(rtd.classify_event({}, FakeModel(0.69), extractor)[0], "benign")


class ElasticsearchSinkTests(unittest.TestCase):
    def test_create_indices_only_creates_missing_indices(self) -> None:
        with mock.patch.object(rtd, "Elasticsearch") as client_cls:
            client = client_cls.return_value
            client.indices.exists.side_effect = lambda index: index == rtd.FLOW_INDEX
            rtd.ElasticsearchSink("http://es:9200").create_indices()

        client.indices.create.assert_called_once()
        kwargs = client.indices.create.call_args.kwargs
        self.assertEqual(kwargs["index"], rtd.ALERT_INDEX)
        self.assertEqual(kwargs["mappings"]["properties"]["src_ip"], {"type": "ip"})


if __name__ == "__main__":
    unittest.main()
