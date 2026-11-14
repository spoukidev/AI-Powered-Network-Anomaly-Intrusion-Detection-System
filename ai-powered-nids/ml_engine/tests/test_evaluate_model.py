import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

import evaluate_model as ev  # noqa: E402
from train_model import FEATURE_COLUMNS, generate_mock_cic_ids2017_dataset  # noqa: E402


def noisy_dataset(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Overlapping classes where only orig_pkts carries signal, so models disagree."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=rows)
    data = {feature: rng.normal(size=rows) for feature in FEATURE_COLUMNS}
    data["orig_pkts"] = labels * 1.5 + rng.normal(size=rows)
    data["label"] = labels
    return pd.DataFrame(data)


class DetectionMetricsTests(unittest.TestCase):
    def test_counts_and_rates_at_threshold(self) -> None:
        y_true = [0, 0, 0, 1, 1]
        scores = np.array([0.1, 0.6, 0.2, 0.9, 0.4])
        metrics = ev.detection_metrics(y_true, scores, 0.5)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 2, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_auc_is_none_with_single_class(self) -> None:
        metrics = ev.detection_metrics([0, 0], np.array([0.2, 0.8]), 0.5)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])


class ThresholdAnalysisTests(unittest.TestCase):
    def test_recommends_thresholds_within_fpr_budget(self) -> None:
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.05, 0.2, 0.35, 0.25, 0.4, 0.7, 0.8, 0.95])
        analysis = ev.threshold_analysis(
            y_true, scores, target_fpr=0.0, deployed_threshold=0.7, thresholds=[0.3, 0.5, 0.65]
        )
        budget = analysis["best_recall_within_fpr_budget"]
        # 0.5 and 0.65 tie on recall with no false positives; prefer fewer alerts.
        self.assertEqual(budget["threshold"], 0.65)
        self.assertEqual(budget["false_positive_rate"], 0.0)
        # 0.3 catches every attack at the cost of one false positive (F1 0.89).
        self.assertEqual(analysis["best_f1"]["threshold"], 0.3)
        self.assertEqual(analysis["deployed"]["threshold"], 0.7)
        self.assertEqual(len(analysis["sweep"]), 3)

    def test_no_threshold_in_budget_returns_none(self) -> None:
        y_true = pd.Series([0, 1])
        scores = np.array([0.99, 0.99])
        analysis = ev.threshold_analysis(
            y_true, scores, target_fpr=0.0, deployed_threshold=0.7, thresholds=[0.5]
        )
        self.assertIsNone(analysis["best_recall_within_fpr_budget"])


class EvaluateTests(unittest.TestCase):
    def test_full_report_compares_baselines_and_ranks_signal_feature_first(self) -> None:
        report = ev.evaluate(noisy_dataset(), permutation_repeats=3)

        self.assertEqual(
            set(report["model_comparison"]),
            {"majority_baseline", "logistic_regression", "random_forest"},
        )
        baseline = report["model_comparison"]["majority_baseline"]
        forest = report["model_comparison"]["random_forest"]
        self.assertEqual(baseline["roc_auc"], 0.5)
        self.assertGreater(forest["roc_auc"], 0.7)
        self.assertGreater(forest["f1"], baseline["f1"])

        importance = report["feature_importance"]
        self.assertEqual(importance[0]["feature"], "orig_pkts")
        self.assertEqual({row["feature"] for row in importance}, set(FEATURE_COLUMNS))
        self.assertIn("best_f1", report["threshold_analysis"])
        json.dumps(report)  # must be serializable as-is

    def test_cli_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "flows.csv"
            output_dir = Path(temp_dir) / "reports"
            generate_mock_cic_ids2017_dataset(dataset, normal_rows=80, attack_rows=40)

            ev.main(
                [
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(output_dir),
                    "--permutation-repeats",
                    "2",
                    "--target-fpr",
                    "0.05",
                ]
            )

            report = json.loads((output_dir / ev.REPORT_JSON_NAME).read_text())
            markdown = (output_dir / ev.REPORT_MD_NAME).read_text()

        self.assertEqual(report["threshold_analysis"]["target_false_positive_rate"], 0.05)
        self.assertEqual(len(report["dataset_artifact"]["sha256"]), 64)
        self.assertIn("## Model comparison", markdown)
        self.assertIn("| random_forest |", markdown)
        self.assertIn("Best recall with FPR ≤ 0.05", markdown)

    def test_cli_rejects_out_of_range_fpr(self) -> None:
        with self.assertRaises(SystemExit):
            ev.parse_args(["--target-fpr", "2"])


if __name__ == "__main__":
    unittest.main()
