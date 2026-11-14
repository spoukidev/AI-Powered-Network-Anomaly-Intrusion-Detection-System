# Research Roadmap

This roadmap turns the AI-powered NIDS project from a software demonstration into a reproducible cybersecurity research project.

Status markers: **[done]** means a script in `ai-powered-nids/ml_engine` implements the
step. It still has to be run on real data to produce research evidence.

## Phase 1 — Reproducible Baseline

- Document the dataset or traffic-capture source.
- Record the train/validation/test split strategy.
- **[done]** Establish simple baselines before complex models (`evaluate_model.py`).
- **[done]** Report precision, recall, F1-score and confusion matrices in addition to accuracy.
- **[done]** Fix random seeds where applicable so experiments can be repeated.

## Phase 2 — Detection Analysis

- Measure per-class detection performance.
- **[done]** Study false positives and false negatives (FPR/FNR and threshold sweep).
- **[done]** Identify the features that contribute most strongly to predictions.
- **[done]** Compare model complexity, inference cost and detection quality.

## Phase 3 — Explainable IDS

- **[done]** Add global feature-importance analysis (permutation + impurity).
- Evaluate SHAP or another explanation method for individual alerts.
- Investigate whether explanations are stable across similar network events.

## Phase 4 — Robustness & Adversarial ML

- Define a threat model for an attacker attempting to evade the classifier.
- Identify which traffic-derived features can realistically be manipulated.
- Measure performance under controlled feature perturbations.
- Compare standard training with defensive approaches such as adversarial training or robust feature selection.

## Phase 5 — Generalization

- Evaluate the model on traffic captured in a different environment or dataset.
- Measure performance degradation under distribution shift.
- Explore concept-drift detection and retraining strategies.

## Research Principles

- Keep experiments reproducible.
- Separate measured results from assumptions.
- Document limitations and negative results.
- Avoid claiming zero-day detection without an evaluation that supports it.
- Perform all security testing only on owned or explicitly authorized data and systems.
