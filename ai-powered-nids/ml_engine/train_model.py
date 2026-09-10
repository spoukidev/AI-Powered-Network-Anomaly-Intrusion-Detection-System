"""
Train a Random Forest model for network anomaly and intrusion detection.

This module is self-contained for PFE/demo use:
- Generate a mock CIC-IDS2017-like flow dataset when no dataset exists.
- Train a supervised Random Forest classifier.
- Evaluate the model with standard classification metrics.
- Save a model bundle containing the estimator, schema, and metadata.

The trained model consumes features extracted from Zeek JSON conn.log records.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODEL_DIR: Final[Path] = BASE_DIR / "models"

DATASET_PATH: Final[Path] = DATA_DIR / "mock_cic_ids2017.csv"
MODEL_PATH: Final[Path] = MODEL_DIR / "random_forest_ids.pkl"
METRICS_PATH: Final[Path] = MODEL_DIR / "training_metrics.json"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.25

FEATURE_COLUMNS: Final[list[str]] = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "missed_bytes",
    "history_len",
    "proto_tcp",
    "proto_udp",
    "service_http",
    "service_dns",
]

LABEL_COLUMN: Final[str] = "label"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | train_model | %(message)s",
)
logger = logging.getLogger(__name__)


def generate_mock_cic_ids2017_dataset(
    output_path: Path,
    normal_rows: int = 4_000,
    attack_rows: int = 1_500,
) -> None:
    """
    Generate a realistic mock flow dataset inspired by CIC-IDS2017.

    The generated data is useful for an out-of-the-box demo and CI checks. It
    is not a replacement for real benchmark data during final model validation.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    normal = pd.DataFrame(
        {
            "duration": rng.gamma(shape=2.0, scale=1.2, size=normal_rows),
            "orig_bytes": rng.lognormal(mean=7.0, sigma=0.8, size=normal_rows),
            "resp_bytes": rng.lognormal(mean=7.4, sigma=0.9, size=normal_rows),
            "orig_pkts": rng.poisson(lam=18, size=normal_rows),
            "resp_pkts": rng.poisson(lam=20, size=normal_rows),
            "missed_bytes": rng.poisson(lam=0.2, size=normal_rows),
            "history_len": rng.integers(2, 8, size=normal_rows),
            "proto_tcp": rng.binomial(1, 0.78, size=normal_rows),
            "proto_udp": rng.binomial(1, 0.20, size=normal_rows),
            "service_http": rng.binomial(1, 0.35, size=normal_rows),
            "service_dns": rng.binomial(1, 0.18, size=normal_rows),
            "label": 0,
        }
    )

    attacks = pd.DataFrame(
        {
            "duration": rng.gamma(shape=1.4, scale=0.7, size=attack_rows),
            "orig_bytes": rng.lognormal(mean=9.2, sigma=1.1, size=attack_rows),
            "resp_bytes": rng.lognormal(mean=5.9, sigma=1.4, size=attack_rows),
            "orig_pkts": rng.poisson(lam=95, size=attack_rows),
            "resp_pkts": rng.poisson(lam=12, size=attack_rows),
            "missed_bytes": rng.poisson(lam=4.0, size=attack_rows),
            "history_len": rng.integers(1, 16, size=attack_rows),
            "proto_tcp": rng.binomial(1, 0.86, size=attack_rows),
            "proto_udp": rng.binomial(1, 0.12, size=attack_rows),
            "service_http": rng.binomial(1, 0.18, size=attack_rows),
            "service_dns": rng.binomial(1, 0.08, size=attack_rows),
            "label": 1,
        }
    )

    dataset = pd.concat([normal, attacks], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Generated mock dataset at %s with %d rows", output_path, len(dataset))


def validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, numeric features, and labels before splitting."""
    required_columns = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_columns)}")

    cleaned = dataset.copy()
    invalid_features: list[str] = []
    for feature in FEATURE_COLUMNS:
        try:
            cleaned[feature] = pd.to_numeric(cleaned[feature], errors="raise")
        except (TypeError, ValueError):
            invalid_features.append(feature)

    if invalid_features:
        raise ValueError(
            "Dataset features must contain numeric values; invalid columns: "
            f"{sorted(invalid_features)}"
        )

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if cleaned.empty:
        raise ValueError("Dataset contains no usable rows after removing NaN/inf values")

    try:
        labels = pd.to_numeric(cleaned[LABEL_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset labels must be numeric binary values 0 and 1") from exc

    unique_labels = set(labels.unique().tolist())
    if unique_labels != {0, 1}:
        raise ValueError(
            "Dataset labels must contain both binary classes 0 (benign) and 1 (attack); "
            f"found {sorted(unique_labels)}"
        )

    class_counts = labels.value_counts()
    if int(class_counts.min()) < 4:
        raise ValueError(
            "Each class must contain at least 4 usable rows for a stratified train/test split"
        )

    cleaned[LABEL_COLUMN] = labels.astype(int)
    return cleaned


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load training data and validate the required feature schema."""
    if not dataset_path.exists():
        generate_mock_cic_ids2017_dataset(dataset_path)

    dataset = validate_dataset(pd.read_csv(dataset_path))
    logger.info("Loaded dataset with shape %s", dataset.shape)
    return dataset


def train_model(dataset: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    """Train and evaluate the Random Forest IDS classifier."""
    dataset = validate_dataset(dataset)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[LABEL_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    logger.info("Training Random Forest classifier")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    attack_probabilities = model.predict_proba(x_test)[:, 1]
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    false_negative_rate = float(fn / (fn + tp)) if (fn + tp) else 0.0

    metrics = {
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": float(np.clip(roc_auc_score(y_test, attack_probabilities), 0.0, 1.0)),
        "average_precision": float(
            np.clip(average_precision_score(y_test, attack_probabilities), 0.0, 1.0)
        ),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_distribution": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().items()},
        },
        "evaluation_protocol": {
            "split": "stratified_holdout",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "positive_class": 1,
        },
    }

    logger.info(
        "Model evaluation:\n%s",
        classification_report(y_test, predictions, labels=[0, 1], zero_division=0),
    )
    logger.info(
        "ROC-AUC=%.4f AP=%.4f FPR=%.4f FNR=%.4f",
        metrics["roc_auc"],
        metrics["average_precision"],
        false_positive_rate,
        false_negative_rate,
    )
    return model, metrics


def save_model_bundle(model: RandomForestClassifier, metrics: dict) -> None:
    """Persist the estimator plus metadata required by inference services."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": {"benign": 0, "attack": 1},
        "model_type": "RandomForestClassifier",
        "trained_for": "AI-Powered Network Anomaly and Intrusion Detection System",
        "metrics": metrics,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info("Saved model bundle to %s", MODEL_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)


def main() -> None:
    """Run the complete model training workflow."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)
    model, metrics = train_model(dataset)
    save_model_bundle(model, metrics)


if __name__ == "__main__":
    main()
