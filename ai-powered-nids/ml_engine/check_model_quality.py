"""
Fail fast when the IDS model regresses: a model-quality gate for CI.

Reads the ``evaluation_report.json`` written by ``evaluate_model.py`` and checks
the production model against minimum detection quality at both the comparison
threshold (0.5) and the deployed alert threshold:

- recall (detection rate) at or above ``--min-recall``
- F1 at or above ``--min-f1``
- false-positive rate at or below ``--max-fpr``
- F1 beating the majority-class baseline by at least ``--min-baseline-margin``

Exits 0 when every check passes and 1 otherwise, listing each failure, so it can
be used directly as a CI step after ``evaluate_model.py``.

Usage:
    python evaluate_model.py --output-dir models
    python check_model_quality.py --report models/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

from evaluate_model import REPORT_JSON_NAME
from train_model import MODEL_DIR

DEFAULT_MODEL: Final[str] = "random_forest"
BASELINE_MODEL: Final[str] = "majority_baseline"
DEFAULT_MIN_RECALL: Final[float] = 0.90
DEFAULT_MIN_F1: Final[float] = 0.90
DEFAULT_MAX_FPR: Final[float] = 0.05
DEFAULT_MIN_BASELINE_MARGIN: Final[float] = 0.10


class ReportError(ValueError):
    """The evaluation report is missing data the gate needs."""


def _metric(metrics: dict[str, Any], name: str, where: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ReportError(f"{where}: missing numeric '{name}'")
    return float(value)


def check_report(
    report: dict[str, Any],
    model: str = DEFAULT_MODEL,
    min_recall: float = DEFAULT_MIN_RECALL,
    min_f1: float = DEFAULT_MIN_F1,
    max_fpr: float = DEFAULT_MAX_FPR,
    min_baseline_margin: float = DEFAULT_MIN_BASELINE_MARGIN,
) -> list[str]:
    """Return human-readable gate failures (empty when the model passes)."""
    comparison = report.get("model_comparison")
    if not isinstance(comparison, dict) or not isinstance(comparison.get(model), dict):
        raise ReportError(f"model_comparison has no entry for '{model}'")
    deployed = (report.get("threshold_analysis") or {}).get("deployed")
    if not isinstance(deployed, dict):
        raise ReportError("threshold_analysis.deployed is missing")

    failures: list[str] = []
    for label, metrics in (
        (f"{model} @ comparison threshold", comparison[model]),
        (f"{model} @ deployed threshold", deployed),
    ):
        threshold = metrics.get("threshold", "?")
        recall = _metric(metrics, "recall", label)
        f1 = _metric(metrics, "f1", label)
        fpr = _metric(metrics, "false_positive_rate", label)
        if recall < min_recall:
            failures.append(f"{label} ({threshold}): recall {recall:.3f} < {min_recall:.3f}")
        if f1 < min_f1:
            failures.append(f"{label} ({threshold}): F1 {f1:.3f} < {min_f1:.3f}")
        if fpr > max_fpr:
            failures.append(f"{label} ({threshold}): FPR {fpr:.3f} > {max_fpr:.3f}")

    if isinstance(comparison.get(BASELINE_MODEL), dict):
        model_f1 = _metric(comparison[model], "f1", model)
        baseline_f1 = _metric(comparison[BASELINE_MODEL], "f1", BASELINE_MODEL)
        if model_f1 - baseline_f1 < min_baseline_margin:
            failures.append(
                f"{model} F1 {model_f1:.3f} does not beat {BASELINE_MODEL} F1 "
                f"{baseline_f1:.3f} by {min_baseline_margin:.3f}"
            )
    return failures


def _probability(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {raw}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when evaluate_model.py metrics regress.")
    parser.add_argument(
        "--report",
        type=Path,
        default=MODEL_DIR / REPORT_JSON_NAME,
        help="evaluation_report.json from evaluate_model.py.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model entry to gate.")
    parser.add_argument("--min-recall", type=_probability, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--min-f1", type=_probability, default=DEFAULT_MIN_F1)
    parser.add_argument("--max-fpr", type=_probability, default=DEFAULT_MAX_FPR)
    parser.add_argument(
        "--min-baseline-margin", type=_probability, default=DEFAULT_MIN_BASELINE_MARGIN
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ReportError("report is not a JSON object")
        failures = check_report(
            report,
            model=args.model,
            min_recall=args.min_recall,
            min_f1=args.min_f1,
            max_fpr=args.max_fpr,
            min_baseline_margin=args.min_baseline_margin,
        )
    except (OSError, ValueError) as exc:
        print(f"model-quality gate: cannot read {args.report}: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("model-quality gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"model-quality gate passed for {args.model} ({args.report})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
