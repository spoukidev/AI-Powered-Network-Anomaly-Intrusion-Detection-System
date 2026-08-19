# AI-Powered Network Anomaly & Intrusion Detection System

A cybersecurity research project exploring machine-learning-based network intrusion detection using network telemetry, feature engineering, and supervised learning.

> **Research direction:** AI for cybersecurity, network intrusion detection, adversarial machine learning, explainable AI, and robust anomaly detection.

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

## Planned Research Extensions

- Compare multiple baseline models and report precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Add feature-importance analysis and SHAP-based explanations.
- Evaluate class imbalance and threshold calibration.
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
