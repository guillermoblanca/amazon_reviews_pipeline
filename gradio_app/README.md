---
title: Amazon Reviews Sentiment Analysis
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
short_description: Classify Amazon reviews as positive or negative using TF-IDF + Logistic Regression trained with PySpark MLlib
---

# Amazon Reviews — Sentiment Analysis

Sentiment classifier trained on ~21,000 Amazon reviews.

**Model:** TF-IDF + Logistic Regression (scikit-learn export of a PySpark MLlib pipeline)

**Metrics on held-out test set:**
- Accuracy: 94.32%
- F1-Score: 89.55%
- AUC-ROC: 97.74%

## How to use

1. **Single Review tab:** paste a review title and body → get sentiment + confidence
2. **Batch tab:** paste multiple reviews (one per line) → get a table of results
3. **About tab:** full explanation of the training pipeline and technical decisions

## Training pipeline

This model was trained with a full PySpark MLlib pipeline:
- Phase 1: Ingestion with schema validation
- Phase 2: Text cleaning, binary labeling (≥4★ = positive, ≤2★ = negative)
- Phase 3: TF-IDF + Logistic Regression with 3-fold Cross-Validation
- Phase 4: Evaluation with confusion matrix and ROC curve

The PySpark model is exported to scikit-learn format for lightweight serving here.
