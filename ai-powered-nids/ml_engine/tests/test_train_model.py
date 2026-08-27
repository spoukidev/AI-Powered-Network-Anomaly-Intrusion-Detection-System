import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

from train_model import (  # noqa: E402
    FEATURE_COLUMNS,
    generate_mock_cic_ids2017_dataset,
    load_dataset,
    train_model,
)


class TrainingPipelineTests(unittest.TestCase):
    def test_invalid_labels_are_rejected_before_training(self) -> None:
        rows = []
        for index in range(8):
            row = {feature: float(index + 1) for feature in FEATURE_COLUMNS}
            row["label"] = 2 if index == 0 else index % 2
            rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "binary classes"):
                load_dataset(path)

    def test_training_exports_detection_relevant_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mock.csv"
            generate_mock_cic_ids2017_dataset(path, normal_rows=120, attack_rows=80)
            dataset = load_dataset(path)
            _, metrics = train_model(dataset)

        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertTrue(all(len(row) == 2 for row in metrics["confusion_matrix"]))
        for metric in (
            "roc_auc",
            "average_precision",
            "false_positive_rate",
            "false_negative_rate",
        ):
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

        protocol = metrics["evaluation_protocol"]
        self.assertEqual(protocol["split"], "stratified_holdout")
        self.assertEqual(protocol["positive_class"], 1)
        self.assertEqual(set(metrics["class_distribution"]["test"]), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
