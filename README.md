# AI-Powered Network Anomaly & Intrusion Detection System

[![CI](https://github.com/spoukidev/AI-Powered-Network-Anomaly-Intrusion-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/spoukidev/AI-Powered-Network-Anomaly-Intrusion-Detection-System/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A cybersecurity research project exploring machine-learning-based network intrusion detection using network telemetry, feature engineering, and supervised learning.

> **Research direction:** AI for cybersecurity, network intrusion detection, adversarial machine learning, explainable AI, and robust anomaly detection.

## Quick start

Requires Docker with Docker Compose.

```bash
git clone https://github.com/spoukidev/AI-Powered-Network-Anomaly-Intrusion-Detection-System.git
cd AI-Powered-Network-Anomaly-Intrusion-Detection-System/ai-powered-nids
docker compose up --build
```

This starts Elasticsearch (`http://localhost:9200`), Kibana (`http://localhost:5601`), the ML engine, a mock Zeek log producer and the real-time detector. The engine trains a Random Forest on the bundled synthetic CIC-IDS2017-like seed data, tails `zeek_logs/conn.log` and writes flows and alerts to the `ids-flows` and `ids-alerts` indices. The bundled data is synthetic, so demo alerts and metrics are pipeline smoke tests, not detection benchmarks.

Run the ML engine tests without Docker (Python 3.11):

```bash
cd ai-powered-nids/ml_engine
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

See [`ai-powered-nids/README.md`](./ai-powered-nids/README.md) for architecture, configuration and Kibana details.

## Project

The implementation lives in [`ai-powered-nids/`](./ai-powered-nids/).

This repository is being developed as a portfolio and research project for final-year cybersecurity work. The goal is not only to train a classifier, but to study the full detection pipeline: network data collection, preprocessing, feature extraction, model training, evaluation, and future robustness testing.

## Core Pipeline

```text
Network traffic / Zeek logs
        ↓
Preprocessing & feature engineering
        ↓
Machine-learning model
        ↓
Normal vs suspicious traffic classification
        ↓
Evaluation and security analysis
```

## Technologies

- Python
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Zeek network telemetry
- Docker / Docker Compose
- Linux

## Research Questions

The project is evolving around several questions relevant to modern AI-security research:

1. How accurately can supervised ML distinguish benign and malicious network behavior?
2. Which network features contribute most to detection quality?
3. How robust are trained models to distribution shift and adversarial manipulation?
4. Can explainability techniques make alerts more useful to security analysts?
5. How well does a model generalize across different traffic captures or datasets?

## Evaluation and Offline Scoring

- `python evaluate_model.py` compares a majority-class baseline, logistic regression and the Random Forest (precision, recall, F1, ROC-AUC, FPR/FNR, confusion matrices, cost). It also ranks features by permutation importance and recommends alert thresholds for a false-positive budget.
- `python score_logs.py conn.log` scores saved Zeek logs (JSON or tab-separated, optionally gzipped) without Elasticsearch.

See [`ai-powered-nids/README.md`](./ai-powered-nids/README.md) for options.

## Planned Research Extensions

- Add SHAP-based explanations for individual alerts.
- Evaluate class imbalance and probability calibration.
- Test cross-dataset generalization.
- Explore evasion attacks against ML-based IDS models.
- Investigate adversarial training and robust feature selection.
- Study concept drift in changing network environments.

## Responsible Use

This project is intended for defensive cybersecurity research and authorized laboratory environments. It is designed to improve detection and understanding of malicious network behavior, not to facilitate unauthorized access.

## Repository Structure

```text
.
├── README.md
├── ai-powered-nids/
│   ├── README.md
│   ├── docker-compose.yml
│   ├── ml_engine/
│   └── zeek_logs/
└── .gitignore
```

## Author

**Iyad Hamoudi**  
Cybersecurity student interested in AI security, network security, intrusion detection, and adversarial machine learning.
