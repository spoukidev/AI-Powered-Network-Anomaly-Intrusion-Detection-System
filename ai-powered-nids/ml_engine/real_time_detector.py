"""
Real-time Zeek conn.log detector.

The service tails JSON Zeek connection logs, extracts ML features, performs
Random Forest inference, and indexes flows/alerts into Elasticsearch.

For demo use, MOCK_LOG_PRODUCER=true generates synthetic Zeek-style events.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterator

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import ElasticsearchException


ELASTICSEARCH_URL: Final[str] = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
)
MODEL_PATH: Final[Path] = Path(
    os.getenv("MODEL_PATH", "models/random_forest_ids.pkl")
)
ZEEK_CONN_LOG: Final[Path] = Path(
    os.getenv("ZEEK_CONN_LOG", "../zeek_logs/conn.log")
)
ALERT_INDEX: Final[str] = os.getenv("ALERT_INDEX", "ids-alerts")
FLOW_INDEX: Final[str] = os.getenv("FLOW_INDEX", "ids-flows")
MOCK_LOG_PRODUCER: Final[bool] = os.getenv(
    "MOCK_LOG_PRODUCER",
    "true",
).lower() == "true"
ALERT_THRESHOLD: Final[float] = float(os.getenv("ALERT_THRESHOLD", "0.70"))

POLL_INTERVAL_SECONDS: Final[float] = 0.5
ELASTIC_RETRY_SECONDS: Final[int] = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | real_time_detector | %(message)s",
)
logger = logging.getLogger(__name__)

shutdown_event = threading.Event()


def handle_shutdown(signum: int, _frame: Any) -> None:
    """Handle container shutdown gracefully."""
    logger.info("Received signal %s. Shutting down detector.", signum)
    shutdown_event.set()


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


class ZeekFeatureExtractor:
    """Convert Zeek JSON conn.log records into the training feature schema."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    def transform(self, event: dict[str, Any]) -> pd.DataFrame:
        """Return a one-row DataFrame ordered exactly like training data."""
        proto = str(event.get("proto", "")).lower()
        service = str(event.get("service", "")).lower()
        history = str(event.get("history", ""))

        features = {
            "duration": self._to_float(event.get("duration")),
            "orig_bytes": self._to_float(event.get("orig_bytes")),
            "resp_bytes": self._to_float(event.get("resp_bytes")),
            "orig_pkts": self._to_float(event.get("orig_pkts")),
            "resp_pkts": self._to_float(event.get("resp_pkts")),
            "missed_bytes": self._to_float(event.get("missed_bytes")),
            "history_len": float(len(history)),
            "proto_tcp": 1.0 if proto == "tcp" else 0.0,
            "proto_udp": 1.0 if proto == "udp" else 0.0,
            "service_http": 1.0 if service in {"http", "ssl", "https"} else 0.0,
            "service_dns": 1.0 if service == "dns" else 0.0,
        }

        ordered = {column: features.get(column, 0.0) for column in self.feature_columns}
        return pd.DataFrame([ordered])

    @staticmethod
    def _to_float(value: Any) -> float:
        """Safely convert Zeek values, including '-' and missing fields."""
        try:
            if value in (None, "-", ""):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class ElasticsearchSink:
    """Small Elasticsearch wrapper with index initialization and retry logic."""

    def __init__(self, url: str) -> None:
        self.client = Elasticsearch(url, request_timeout=10)

    def wait_until_ready(self) -> None:
        """Block until Elasticsearch is reachable."""
        while not shutdown_event.is_set():
            try:
                if self.client.ping():
                    logger.info("Connected to Elasticsearch at %s", ELASTICSEARCH_URL)
                    return
            except ElasticsearchException as exc:
                logger.warning("Elasticsearch is not ready: %s", exc)

            logger.info(
                "Retrying Elasticsearch connection in %d seconds",
                ELASTIC_RETRY_SECONDS,
            )
            time.sleep(ELASTIC_RETRY_SECONDS)

        raise RuntimeError("Shutdown requested before Elasticsearch became ready")

    def create_indices(self) -> None:
        """Create minimal mappings for flow and alert indices."""
        flow_mapping = {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "uid": {"type": "keyword"},
                    "id.orig_h": {"type": "ip"},
                    "id.resp_h": {"type": "ip"},
                    "proto": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "anomaly_score": {"type": "float"},
                    "prediction": {"type": "keyword"},
                }
            }
        }

        alert_mapping = {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "severity": {"type": "keyword"},
                    "anomaly_score": {"type": "float"},
                    "src_ip": {"type": "ip"},
                    "dst_ip": {"type": "ip"},
                    "proto": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "reason": {"type": "text"},
                }
            }
        }

        for index_name, mapping in (
            (FLOW_INDEX, flow_mapping),
            (ALERT_INDEX, alert_mapping),
        ):
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=mapping)
                logger.info("Created Elasticsearch index: %s", index_name)

    def index_flow(self, document: dict[str, Any]) -> None:
        """Index every analyzed flow for Kibana dashboards."""
        self.client.index(index=FLOW_INDEX, document=document)

    def index_alert(self, document: dict[str, Any]) -> None:
        """Index high-risk flows into the alerts index."""
        self.client.index(index=ALERT_INDEX, document=document)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load the serialized model bundle produced by train_model.py."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py first."
        )

    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Invalid model bundle. Missing model or feature_columns.")

    logger.info("Loaded model bundle from %s", model_path)
    return bundle


def parse_json_line(line: str) -> dict[str, Any] | None:
    """Parse a Zeek JSON line and tolerate malformed input."""
    try:
        event = json.loads(line)
        if isinstance(event, dict):
            return event
    except json.JSONDecodeError:
        logger.warning("Skipping malformed JSON log line")
    return None


def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for Elasticsearch."""
    return datetime.now(timezone.utc).isoformat()


def classify_event(
    event: dict[str, Any],
    model: Any,
    extractor: ZeekFeatureExtractor,
) -> tuple[str, float]:
    """Run ML inference and return prediction label plus attack probability."""
    frame = extractor.transform(event)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][1])
    else:
        probability = float(model.predict(frame)[0])

    prediction = "attack" if probability >= ALERT_THRESHOLD else "benign"
    return prediction, probability


def build_flow_document(
    event: dict[str, Any],
    prediction: str,
    anomaly_score: float,
) -> dict[str, Any]:
    """Build a normalized flow document for Elasticsearch."""
    document = dict(event)
    document["@timestamp"] = event.get("ts") or now_iso()
    document["prediction"] = prediction
    document["anomaly_score"] = anomaly_score
    return document


def build_alert_document(
    event: dict[str, Any],
    anomaly_score: float,
) -> dict[str, Any]:
    """Build a SOC-friendly alert document."""
    src_ip = event.get("id.orig_h", "0.0.0.0")
    dst_ip = event.get("id.resp_h", "0.0.0.0")
    severity = "critical" if anomaly_score >= 0.90 else "high"

    return {
        "@timestamp": event.get("ts") or now_iso(),
        "severity": severity,
        "anomaly_score": anomaly_score,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": event.get("id.orig_p"),
        "dst_port": event.get("id.resp_p"),
        "proto": event.get("proto"),
        "service": event.get("service"),
        "uid": event.get("uid"),
        "reason": (
            "Random Forest model classified this network flow as anomalous "
            f"with probability {anomaly_score:.2f}."
        ),
        "raw_event": event,
    }


def tail_file(path: Path) -> Iterator[str]:
    """
    Tail a file forever and yield new lines.

    The function creates the file if missing, which makes local demos smooth.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8") as file:
        file.seek(0, os.SEEK_END)

        while not shutdown_event.is_set():
            line = file.readline()

            if line:
                yield line.strip()
            else:
                time.sleep(POLL_INTERVAL_SECONDS)


def generate_mock_zeek_event() -> dict[str, Any]:
    """Generate a Zeek-style conn.log JSON event for demo traffic."""
    is_attack = random.random() < 0.28

    if is_attack:
        orig_bytes = random.randint(25_000, 800_000)
        resp_bytes = random.randint(100, 8_000)
        orig_pkts = random.randint(80, 600)
        resp_pkts = random.randint(1, 60)
        missed_bytes = random.randint(1, 25)
        service = random.choice(["-", "http", "dns"])
        history = random.choice(["S", "SH", "S0", "REJ", "RSTOS0"])
    else:
        orig_bytes = random.randint(150, 8_000)
        resp_bytes = random.randint(300, 25_000)
        orig_pkts = random.randint(3, 60)
        resp_pkts = random.randint(3, 75)
        missed_bytes = random.choice([0, 0, 0, 1])
        service = random.choice(["http", "dns", "ssl", "-"])
        history = random.choice(["ShADadFf", "Dd", "ShADad", "SF"])

    proto = "udp" if service == "dns" else random.choice(["tcp", "tcp", "udp"])

    return {
        "ts": now_iso(),
        "uid": f"C{random.randint(10**8, 10**9 - 1)}",
        "id.orig_h": f"192.168.1.{random.randint(2, 254)}",
        "id.orig_p": random.randint(1024, 65535),
        "id.resp_h": f"10.0.0.{random.randint(2, 254)}",
        "id.resp_p": random.choice([22, 53, 80, 443, 8080]),
        "proto": proto,
        "service": service,
        "duration": round(random.uniform(0.02, 8.0), 4),
        "orig_bytes": orig_bytes,
        "resp_bytes": resp_bytes,
        "orig_pkts": orig_pkts,
        "resp_pkts": resp_pkts,
        "missed_bytes": missed_bytes,
        "history": history,
    }


def mock_log_producer(path: Path) -> None:
    """Continuously append mock Zeek JSON records to conn.log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting mock Zeek log producer at %s", path)

    while not shutdown_event.is_set():
        event = generate_mock_zeek_event()

        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")

        time.sleep(random.uniform(0.7, 2.0))


def run_detector() -> None:
    """Run the detector service loop."""
    bundle = load_model_bundle(MODEL_PATH)
    model = bundle["model"]
    extractor = ZeekFeatureExtractor(bundle["feature_columns"])

    sink = ElasticsearchSink(ELASTICSEARCH_URL)
    sink.wait_until_ready()
    sink.create_indices()

    if MOCK_LOG_PRODUCER:
        producer = threading.Thread(
            target=mock_log_producer,
            args=(ZEEK_CONN_LOG,),
            daemon=True,
        )
        producer.start()

    logger.info("Tailing Zeek log: %s", ZEEK_CONN_LOG)

    for line in tail_file(ZEEK_CONN_LOG):
        event = parse_json_line(line)

        if event is None:
            continue

        try:
            prediction, anomaly_score = classify_event(event, model, extractor)
            flow_document = build_flow_document(event, prediction, anomaly_score)
            sink.index_flow(flow_document)

            if prediction == "attack":
                alert_document = build_alert_document(event, anomaly_score)
                sink.index_alert(alert_document)
                logger.warning(
                    "ALERT src=%s dst=%s score=%.2f",
                    alert_document["src_ip"],
                    alert_document["dst_ip"],
                    anomaly_score,
                )
            else:
                logger.info("Benign flow scored %.2f", anomaly_score)

        except ElasticsearchException as exc:
            logger.error("Failed to index event in Elasticsearch: %s", exc)
        except Exception as exc:
            logger.exception("Unexpected detector error: %s", exc)


if __name__ == "__main__":
    run_detector()
