"""
Evaluate the IDS classifier against baselines and explain what drives it.

This implements the "Reproducible Baseline", "Detection Analysis" and global
"Explainable IDS" steps of RESEARCH_ROADMAP.md on the same stratified hold-out
split that ``train_model.py`` uses:

1. Baseline comparison: a majority-class dummy model, a scaled logistic
   regression and the production Random Forest, each with precision, recall,
   F1, ROC-AUC, average precision, false-positive/negative rates, confusion
   matrix, training time and per-flow inference cost.
2. Feature importance for the Random Forest: permutation importance on the
   test split (model-agnostic, with spread across repeats) next to the
   built-in impurity importance.
3. Alert-threshold analysis: precision/recall/F1/FPR across thresholds, the
   F1-optimal threshold, the threshold with the best recall inside a
   false-positive budget, and how the deployed ALERT_THRESHOLD performs.

Outputs ``evaluation_report.json`` (machine-readable) and
``evaluation_report.md`` (tables for a thesis or README) in --output-dir.

Usage:
    python evaluate_model.py
    python evaluate_model.py --dataset data/my_flows.csv --target-fpr 0.005
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_model import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_random_forest,
    load_dataset,
    sha256_file,
    split_dataset,
)

REPORT_JSON_NAME: Final[str] = "evaluation_report.json"
REPORT_MD_NAME: Final[str] = "evaluation_report.md"
DEFAULT_TARGET_FPR: Final[float] = 0.01
DEFAULT_DEPLOYED_THRESHOLD: Final[float] = 0.70
PERMUTATION_REPEATS: Final[int] = 10
THRESHOLD_GRID: Final[tuple[float, ...]] = tuple(round(t, 2) for t in np.arange(0.05, 1.0, 0.05))

logger = logging.getLogger("evaluate_model")


def candidate_models() -> dict[str, ClassifierMixin]:
    """Return the unfitted models to compare, simplest first."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": build_random_forest(),
    }


def detection_metrics(
    y_true: Sequence[int] | pd.Series,
    attack_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute IDS metrics for one model at one alert threshold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    predictions = (attack_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, predictions, labels=[0, 1]).ravel()

    has_both_classes = len(np.unique(y_true_arr)) == 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_arr, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_arr, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": (
            float(roc_auc_score(y_true_arr, attack_probabilities)) if has_both_classes else None
        ),
        "average_precision": (
            float(average_precision_score(y_true_arr, attack_probabilities))
            if has_both_classes
            else None
        ),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def attack_probabilities(model: ClassifierMixin, x: pd.DataFrame) -> np.ndarray:
    """Return P(attack) for each row, even when a model only saw one class."""
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return probabilities[:, classes.index(1)]


def compare_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict[str, dict[str, Any]], dict[str, ClassifierMixin]]:
    """Fit every candidate on the training split and score it on the test split."""
    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, ClassifierMixin] = {}

    for name, estimator in candidate_models().items():
        model = clone(estimator)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        probabilities = attack_probabilities(model, x_test)
        inference_seconds = time.perf_counter() - started

        metrics = detection_metrics(y_test, probabilities, threshold)
        metrics["fit_seconds"] = round(fit_seconds, 4)
        metrics["inference_microseconds_per_flow"] = round(
            inference_seconds / max(len(x_test), 1) * 1e6, 2
        )
        results[name] = metrics
        fitted[name] = model
        logger.info(
            "%-20s F1=%.3f recall=%.3f FPR=%.3f ROC-AUC=%s",
            name,
            metrics["f1"],
            metrics["recall"],
            metrics["false_positive_rate"],
            "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}",
        )

    return results, fitted


def feature_importance(
    model: ClassifierMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = PERMUTATION_REPEATS,
) -> list[dict[str, Any]]:
    """Rank features by permutation importance (F1 drop) on held-out data."""
    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    impurity = getattr(model, "feature_importances_", None)

    ranking = [
        {
            "feature": feature,
            "permutation_importance_mean": float(permutation.importances_mean[index]),
            "permutation_importance_std": float(permutation.importances_std[index]),
            "impurity_importance": None if impurity is None else float(impurity[index]),
        }
        for index, feature in enumerate(x_test.columns)
    ]
    ranking.sort(
        key=lambda row: (row["permutation_importance_mean"], row["impurity_importance"] or 0.0),
        reverse=True,
    )
    return ranking


def threshold_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
    target_fpr: float,
    deployed_threshold: float,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> dict[str, Any]:
    """Sweep alert thresholds and recommend operating points."""
    sweep = [detection_metrics(y_true, probabilities, t) for t in thresholds]
    for row in sweep:
        # AUC-style metrics do not depend on the threshold; keep the sweep compact.
        row.pop("roc_auc")
        row.pop("average_precision")

    # Ties go to the higher threshold, which raises fewer alerts.
    best_f1 = max(sweep, key=lambda row: (row["f1"], row["threshold"]))
    within_budget = [row for row in sweep if row["false_positive_rate"] <= target_fpr]
    best_recall_in_budget = (
        max(within_budget, key=lambda row: (row["recall"], row["threshold"]))
        if within_budget
        else None
    )

    deployed = detection_metrics(y_true, probabilities, deployed_threshold)
    deployed.pop("roc_auc")
    deployed.pop("average_precision")

    return {
        "target_false_positive_rate": target_fpr,
        "best_f1": best_f1,
        "best_recall_within_fpr_budget": best_recall_in_budget,
        "deployed": deployed,
        "sweep": sweep,
    }


def evaluate(
    dataset: pd.DataFrame,
    target_fpr: float = DEFAULT_TARGET_FPR,
    deployed_threshold: float = DEFAULT_DEPLOYED_THRESHOLD,
    permutation_repeats: int = PERMUTATION_REPEATS,
) -> dict[str, Any]:
    """Run the full evaluation on a validated dataset and return the report."""
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    comparison, fitted = compare_models(x_train, x_test, y_train, y_test)
    forest = fitted["random_forest"]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_protocol": {
            "split": "stratified_holdout",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "positive_class": 1,
            "comparison_threshold": 0.5,
            "training_rows": len(x_train),
            "test_rows": len(x_test),
            "class_distribution": {
                "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
                "test": {str(k): int(v) for k, v in y_test.value_counts().items()},
            },
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "model_comparison": comparison,
        "feature_importance": feature_importance(forest, x_test, y_test, permutation_repeats),
        "threshold_analysis": threshold_analysis(
            y_test,
            attack_probabilities(forest, x_test),
            target_fpr=target_fpr,
            deployed_threshold=deployed_threshold,
        ),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the evaluation report as Markdown tables."""
    protocol = report["evaluation_protocol"]
    lines = [
        "# IDS Model Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
    ]
    artifact = report.get("dataset_artifact")
    if artifact:
        lines += [f"Dataset: `{artifact['path']}` (sha256 `{artifact['sha256'][:12]}…`)", ""]
    lines += [
        f"Protocol: stratified hold-out, test size {protocol['test_size']}, "
        f"seed {protocol['random_state']}, {protocol['training_rows']} train / "
        f"{protocol['test_rows']} test rows. Class distribution (test): "
        f"{protocol['class_distribution']['test']}.",
        "",
        "## Model comparison (threshold 0.50)",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | Avg precision | FPR | FNR "
        "| Fit (s) | Inference (µs/flow) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["model_comparison"].items():
        lines.append(
            f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} "
            f"| {_fmt(m['roc_auc'])} | {_fmt(m['average_precision'])} "
            f"| {_fmt(m['false_positive_rate'])} | {_fmt(m['false_negative_rate'])} "
            f"| {m['fit_seconds']:.3f} | {m['inference_microseconds_per_flow']:.1f} |"
        )

    lines += [
        "",
        "## Random Forest feature importance",
        "",
        "Permutation importance is the mean drop in test F1 when the feature is shuffled "
        "(± standard deviation across repeats).",
        "",
        "| Rank | Feature | Permutation importance | Impurity importance |",
        "|---|---|---|---|",
    ]
    for rank, row in enumerate(report["feature_importance"], start=1):
        lines.append(
            f"| {rank} | {row['feature']} | {row['permutation_importance_mean']:.4f} "
            f"± {row['permutation_importance_std']:.4f} | {_fmt(row['impurity_importance'], 4)} |"
        )

    thresholds = report["threshold_analysis"]
    lines += [
        "",
        "## Alert threshold analysis (Random Forest)",
        "",
        "| Operating point | Threshold | Precision | Recall | F1 | FPR |",
        "|---|---|---|---|---|---|",
    ]
    budget = thresholds["target_false_positive_rate"]
    points = [
        ("Deployed ALERT_THRESHOLD", thresholds["deployed"]),
        ("Best F1", thresholds["best_f1"]),
        (f"Best recall with FPR ≤ {budget:g}", thresholds["best_recall_within_fpr_budget"]),
    ]
    for label, row in points:
        if row is None:
            lines.append(f"| {label} | none on the grid | | | | |")
            continue
        lines.append(
            f"| {label} | {row['threshold']:.2f} | {_fmt(row['precision'])} "
            f"| {_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['false_positive_rate'])} |"
        )

    lines += [
        "",
        "Results on the bundled synthetic demo data are a pipeline smoke test, not evidence "
        "of real-world detection performance. See the README's reproducibility section.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    md_path = output_dir / REPORT_MD_NAME
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _probability(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {raw}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IDS models, rank features and analyze alert thresholds.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Labeled flow CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=MODEL_DIR, help="Where to write the reports."
    )
    parser.add_argument(
        "--target-fpr",
        type=_probability,
        default=DEFAULT_TARGET_FPR,
        help="False-positive budget for the threshold recommendation (default: 0.01).",
    )
    parser.add_argument(
        "--alert-threshold",
        type=_probability,
        default=DEFAULT_DEPLOYED_THRESHOLD,
        help="Threshold the detector uses today, to report its performance (default: 0.70).",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=PERMUTATION_REPEATS,
        help="Shuffles per feature for permutation importance (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | evaluate_model | %(message)s",
    )
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)
    report = evaluate(
        dataset,
        target_fpr=args.target_fpr,
        deployed_threshold=args.alert_threshold,
        permutation_repeats=args.permutation_repeats,
    )
    report["dataset_artifact"] = {
        "path": str(args.dataset),
        "sha256": sha256_file(args.dataset),
        "bytes": args.dataset.stat().st_size,
    }
    json_path, md_path = write_report(report, args.output_dir)
    logger.info("Wrote %s and %s", json_path, md_path)
    return report


if __name__ == "__main__":
    main()
