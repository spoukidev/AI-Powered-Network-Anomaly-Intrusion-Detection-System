import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ENGINE_DIR))

from train_model import FEATURE_COLUMNS, validate_dataset  # noqa: E402


def make_row(label: int, value: float = 1.0) -> dict:
    row = {feature: value for feature in FEATURE_COLUMNS}
    row["label"] = label
    return row


class DatasetValidationTests(unittest.TestCase):
    def test_missing_feature_is_rejected_with_feature_name(self) -> None:
        rows = [make_row(index % 2) for index in range(8)]
        missing_feature = FEATURE_COLUMNS[0]
        dataset = pd.DataFrame(rows).drop(columns=[missing_feature])

        with self.assertRaisesRegex(ValueError, missing_feature):
            validate_dataset(dataset)

    def test_too_few_rows_in_minority_class_is_rejected(self) -> None:
        rows = [make_row(0, float(index + 1)) for index in range(6)]
        rows.extend(make_row(1, float(index + 10)) for index in range(3))

        with self.assertRaisesRegex(ValueError, "at least 4 usable rows"):
            validate_dataset(pd.DataFrame(rows))

    def test_non_finite_rows_are_removed_before_class_count_check(self) -> None:
        rows = [make_row(0, float(index + 1)) for index in range(4)]
        rows.extend(make_row(1, float(index + 10)) for index in range(4))
        rows[0][FEATURE_COLUMNS[0]] = np.inf

        with self.assertRaisesRegex(ValueError, "at least 4 usable rows"):
            validate_dataset(pd.DataFrame(rows))

    def test_non_numeric_feature_is_rejected_with_column_name(self) -> None:
        rows = [make_row(index % 2, float(index + 1)) for index in range(8)]
        invalid_feature = FEATURE_COLUMNS[2]
        rows[0][invalid_feature] = "not-a-number"

        with self.assertRaisesRegex(ValueError, invalid_feature):
            validate_dataset(pd.DataFrame(rows))

    def test_numeric_strings_are_normalized_to_numeric_features(self) -> None:
        rows = [make_row(index % 2, float(index + 1)) for index in range(8)]
        numeric_string_feature = FEATURE_COLUMNS[1]
        rows[0][numeric_string_feature] = "12.5"

        validated = validate_dataset(pd.DataFrame(rows))

        self.assertEqual(validated.loc[0, numeric_string_feature], 12.5)
        self.assertTrue(pd.api.types.is_numeric_dtype(validated[numeric_string_feature]))


if __name__ == "__main__":
    unittest.main()
