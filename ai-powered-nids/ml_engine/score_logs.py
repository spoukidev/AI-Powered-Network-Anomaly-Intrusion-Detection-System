"""
Score saved Zeek conn.log files offline, without Elasticsearch or Kibana.

The real-time detector only tails a live log and needs Elasticsearch. This
command runs the same feature extraction, model and alert logic over files
you already have (lab captures, a PCAP replayed through Zeek, archived
``conn.*.log.gz`` files) and writes the results as JSON lines plus a summary.

Both Zeek output formats are supported and detected automatically:

* JSON logs (``LogAscii::use_json=T``), one object per line.
* Zeek's default tab-separated logs with ``#fields`` / ``#types`` headers.

Usage:
    python score_logs.py ../zeek_logs/conn.log
    python score_logs.py conn.*.log.gz --alerts alerts.jsonl --threshold 0.8
    python score_logs.py conn.log --output scored.jsonl --fail-on-alert

Exit codes: 0 success, 1 usage/model error, 2 alerts found with --fail-on-alert.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Final, TextIO

import real_time_detector as rtd

logger = logging.getLogger("score_logs")

TOP_N: Final[int] = 5
_FLOAT_TYPES: Final[frozenset[str]] = frozenset({"time", "interval", "double"})
_INT_TYPES: Final[frozenset[str]] = frozenset({"count", "int", "port"})


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _convert(value: str, zeek_type: str, unset: str, empty: str) -> Any:
    """Convert one Zeek TSV field to a Python value using its declared type."""
    if value == unset:
        return None
    if value == empty:
        return ""
    try:
        if zeek_type in _FLOAT_TYPES:
            return float(value)
        if zeek_type in _INT_TYPES:
            return int(value)
    except ValueError:
        return value
    if zeek_type == "bool":
        return value == "T"
    return value


def _decode_separator(raw: str) -> str:
    """Decode the ``#separator \\x09`` header value."""
    return raw.encode("ascii").decode("unicode_escape") if raw.startswith("\\x") else raw


def iter_zeek_records(lines: Iterable[str]) -> Iterator[dict[str, Any] | None]:
    """Yield one dict per Zeek record, or None for a record that cannot be parsed.

    Handles JSON lines and Zeek TSV (headers may repeat, e.g. concatenated logs).
    """
    separator = "\t"
    unset = "-"
    empty = "(empty)"
    fields: list[str] | None = None
    types: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue

        if line.startswith("#"):
            directive, _, rest = line.partition(" " if line.startswith("#separator") else separator)
            if directive == "#separator":
                separator = _decode_separator(rest.strip())
            elif directive == "#unset_field":
                unset = rest
            elif directive == "#empty_field":
                empty = rest
            elif directive == "#fields":
                fields = rest.split(separator)
            elif directive == "#types":
                types = rest.split(separator)
            continue

        if line.lstrip().startswith("{"):
            yield rtd.parse_json_line(line)
            continue

        if fields is None:
            logger.warning("Skipping line that is neither JSON nor TSV with a #fields header")
            yield None
            continue

        values = line.split(separator)
        if len(values) != len(fields):
            logger.warning(
                "Skipping TSV record with %d values (expected %d)", len(values), len(fields)
            )
            yield None
            continue

        column_types = types if len(types) == len(fields) else ["string"] * len(fields)
        yield {
            name: _convert(value, zeek_type, unset, empty)
            for name, value, zeek_type in zip(fields, values, column_types, strict=True)
        }


@dataclass
class ScoreSummary:
    """Counters reported after a scoring run."""

    files: list[str] = field(default_factory=list)
    flows_scored: int = 0
    records_skipped: int = 0
    alerts: int = 0
    severity: Counter[str] = field(default_factory=Counter)
    top_alert_sources: Counter[str] = field(default_factory=Counter)
    top_alert_dst_ports: Counter[str] = field(default_factory=Counter)
    max_anomaly_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "flows_scored": self.flows_scored,
            "records_skipped": self.records_skipped,
            "alerts": self.alerts,
            "alert_rate": (self.alerts / self.flows_scored) if self.flows_scored else 0.0,
            "severity": dict(self.severity),
            "top_alert_sources": dict(self.top_alert_sources.most_common(TOP_N)),
            "top_alert_dst_ports": dict(self.top_alert_dst_ports.most_common(TOP_N)),
            "max_anomaly_score": round(self.max_anomaly_score, 4),
        }


def score_records(
    records: Iterable[dict[str, Any] | None],
    model: Any,
    extractor: rtd.ZeekFeatureExtractor,
    threshold: float,
    summary: ScoreSummary,
    flow_out: TextIO | None = None,
    alert_out: TextIO | None = None,
) -> None:
    """Score each record, update ``summary`` and stream documents to the outputs."""
    for event in records:
        if event is None:
            summary.records_skipped += 1
            continue

        prediction, score = rtd.classify_event(event, model, extractor, threshold)
        summary.flows_scored += 1
        summary.max_anomaly_score = max(summary.max_anomaly_score, score)

        if flow_out is not None:
            flow = rtd.build_flow_document(event, prediction, score)
            flow_out.write(json.dumps(flow, default=str) + "\n")

        if prediction == "attack":
            alert = rtd.build_alert_document(event, score)
            summary.alerts += 1
            summary.severity[alert["severity"]] += 1
            summary.top_alert_sources[str(alert["src_ip"])] += 1
            if alert["dst_port"] is not None:
                summary.top_alert_dst_ports[str(alert["dst_port"])] += 1
            if alert_out is not None:
                alert_out.write(json.dumps(alert, default=str) + "\n")


def score_files(
    paths: Iterable[Path],
    model_path: Path,
    threshold: float,
    flow_out: TextIO | None = None,
    alert_out: TextIO | None = None,
) -> ScoreSummary:
    """Load the model once and score every file in order."""
    bundle = rtd.load_model_bundle(model_path)
    extractor = rtd.ZeekFeatureExtractor(bundle["feature_columns"])
    summary = ScoreSummary()

    for path in paths:
        summary.files.append(str(path))
        with _open_text(path) as handle:
            score_records(
                iter_zeek_records(handle),
                bundle["model"],
                extractor,
                threshold,
                summary,
                flow_out=flow_out,
                alert_out=alert_out,
            )
    return summary


def _threshold(raw: str) -> float:
    try:
        return rtd.parse_threshold(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score saved Zeek conn.log files (JSON or TSV, optionally .gz) offline.",
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Zeek conn.log file(s) to score.")
    parser.add_argument(
        "--model",
        type=Path,
        default=rtd.MODEL_PATH,
        help=f"Model bundle from train_model.py (default: {rtd.MODEL_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=rtd.ALERT_THRESHOLD,
        help=f"Attack-probability threshold for alerts (default: {rtd.ALERT_THRESHOLD:.2f}).",
    )
    parser.add_argument("--output", type=Path, help="Write every scored flow as JSON lines.")
    parser.add_argument("--alerts", type=Path, help="Write alerts as JSON lines.")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with status 2 when any alert is raised (useful in scripts and CI).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    missing = [str(path) for path in args.logs if not path.is_file()]
    if missing:
        print(f"error: log file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    flow_out = args.output.open("w", encoding="utf-8") if args.output else None
    alert_out = args.alerts.open("w", encoding="utf-8") if args.alerts else None
    try:
        summary = score_files(args.logs, args.model, args.threshold, flow_out, alert_out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        for handle in (flow_out, alert_out):
            if handle is not None:
                handle.close()

    print(json.dumps(summary.as_dict(), indent=2))
    if args.fail_on_alert and summary.alerts:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
