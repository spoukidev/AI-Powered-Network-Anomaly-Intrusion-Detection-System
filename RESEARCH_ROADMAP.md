# Research Roadmap

This roadmap turns the AI-powered NIDS project from a software demonstration into a reproducible cybersecurity research project.

## Phase 1 — Reproducible Baseline

- Document the dataset or traffic-capture source.
- Record the train/validation/test split strategy.
- Establish simple baselines before complex models.
- Report precision, recall, F1-score and confusion matrices in addition to accuracy.
- Fix random seeds where applicable so experiments can be repeated.

## Phase 2 — Detection Analysis

- Measure per-class detection performance.
- Study false positives and false negatives.
- Identify the features that contribute most strongly to predictions.
- Compare model complexity, inference cost and detection quality.

## Phase 3 — Explainable IDS

- Add global feature-importance analysis.
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
