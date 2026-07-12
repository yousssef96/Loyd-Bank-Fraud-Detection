# 🏦 Lloyd Bank — Loan Default Risk Prediction

An end-to-end machine learning system that predicts the probability of loan default (Charged Off vs. Fully Paid) from borrower and credit history data. The project covers the full lifecycle: data preprocessing, feature engineering, model experimentation, hyperparameter tuning, experiment tracking, model registration, and deployment as a containerized REST API with a Streamlit front-end.

---

## Overview

Lenders need to assess credit risk accurately before approving a loan. This project builds a binary classification pipeline that estimates the likelihood a borrower will default, using account-level and credit-bureau features (income, credit utilization, delinquency history, account age, etc.).

The final model is served through a FastAPI endpoint and can be queried directly or through a simple Streamlit UI designed for loan officers to run quick risk assessments.

---

## Project Architecture

```
Raw Data (Excel)
      │
      ▼
Data Loading & Cleaning ──────► src/data/
      │
      ▼
Feature Preprocessing (impute, log-transform, clip, scale, encode)
      │                                    ──────► src/features/
      ▼
Model Training & Hyperparameter Tuning
      │                                    ──────► src/models/
      ▼
MLflow Tracking & Model Registry
      │
      ▼
Model Serving (FastAPI + pyfunc)   ──────► src/serving/, src/api/
      │
      ▼
Streamlit UI                        ──────► streamlit_app.py
```

The entire preprocessing pipeline (imputation, log-transform, outlier clipping, scaling, encoding) is fitted as a **single scikit-learn `Pipeline`** alongside the model, logged as one MLflow artifact. This eliminates train/serve skew — there is no separate preprocessing script to keep in sync at inference time.

---

## Dataset

The dataset contains loan-level records with the following feature groups:

| Category | Examples |
|---|---|
| Loan details | `loan_amnt`, `term`, `int_rate`, `installment`, `purpose` |
| Borrower profile | `annual_inc`, `emp_length`, `home_ownership` |
| Credit history | `open_acc`, `total_acc`, `mort_acc`, `pub_rec_bankruptcies` |
| Account age/recency | `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_tl`, `mths_since_last_delinq` |
| Utilization & risk signals | `percent_bc_gt_75`, `num_tl_90g_dpd_24m`, `inq_last_12m` |

**Target:** `loan_status` — mapped to `1` (Charged Off) / `0` (Fully Paid).

---

## Modeling Approach

Three model families were compared as a baseline before tuning:

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight='balanced'`, tuned with Optuna |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` set from class ratio |

**Model selection: Logistic Regression** was chosen as the production model. While all three had comparable ROC-AUC, Logistic Regression gave the most stable precision/recall trade-off after threshold tuning, and its coefficients offer interpretability that matters for credit-risk decisions — a relevant consideration beyond raw predictive performance.

Key evaluation decisions:
- **Threshold tuning per model** rather than relying on the default 0.5 cutoff, since class-weighting affects predicted probabilities differently across model types (tree ensembles in particular cluster probabilities near the base rate, making the default threshold misleading).
- **PR-AUC reported alongside ROC-AUC**, since ROC-AUC can look deceptively strong under class imbalance.

## Preprocessing Pipeline

Built as a `ColumnTransformer` with per-feature-group branches:

- **Categorical columns** (`home_ownership`, `purpose`, `term`) → One-hot encoded (`drop='first'`, unseen categories ignored at inference).
- **Skewed numeric columns** (`total_bal_ex_mort`, `max_bal_bc`, `avg_cur_bal`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`) → Imputed → `log1p` transform → Winsorized to the 5th/95th percentile (bounds learned from training data only) → Scaled.
- **Other numeric columns** → Imputed → Scaled.
- **`emp_length`** mapped to a numeric 0–10 scale during data cleaning (preserves ordinal meaning rather than one-hot encoding it).

All imputation and outlier-clipping bounds are learned strictly on the training split and reused on the test split and at inference — no leakage.

## Hyperparameter Tuning & Experiment Tracking

- **Optuna** for Bayesian hyperparameter search (`C`, `penalty` for Logistic Regression), optimizing 5-fold cross-validated performance.
- **MLflow** for tracking every trial (nested runs under a parent search run), logging parameters, metrics, and the final model.
- Final model registered in the **MLflow Model Registry** for versioned, reproducible deployment.

---

## Project Structure

```
├── data/
│   └── raw/                      # Source Excel dataset
├── src/
│   ├── data/
│   │   ├── load_data.py          # Load raw Excel data
│   │   ├── preprocess.py         # Column cleaning, target mapping, emp_length mapping
│   │   └── split.py              # Stratified train/test split
│   ├── features/
│   │   └── build_preprocessor.py # ColumnTransformer: impute, log, clip, scale, encode
│   ├── models/
│   │   ├── evaluate_models.py    # Baseline comparison across model families
│   │   ├── tune_logreg.py        # Optuna + MLflow hyperparameter search
│   │   └── train_final_model.py  # Final model training, evaluation, MLflow registration
│   ├── serving/
│   │   ├── inference.py          # Loads registered pyfunc model, runs predictions
│   │   └── model/                # Downloaded MLflow model artifacts (git-ignored)
│   ├── api/
│   │   └── main.py               # FastAPI app exposing /predict and /health
│   └── utils/
│       └── logger.py             # Centralized logging
├── streamlit_app.py              # Loan officer-facing risk assessment UI
├── Dockerfile                    # FastAPI service image
├── Dockerfile.streamlit          # Streamlit UI image
├── docker-compose.yml            # Runs both services on a shared network
├── pyproject.toml / uv.lock      # Dependency management via uv
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Docker Desktop (for containerized deployment)
- A running MLflow tracking server (`mlflow server --host 127.0.0.1 --port 5000`)

### Local Setup

```bash
uv sync
```

### Run the training pipeline

```bash
uv run python -m src.models.train_final_model
```

This loads and preprocesses the data, tunes the model via Optuna, evaluates it on the held-out test set, and logs + registers the final pipeline in MLflow.

### Download the registered model for serving

```bash
uv run mlflow artifacts download \
  --artifact-uri "models:/Best_loan_status_calssifier/<version>" \
  --dst-path ./src/serving/model
```

### Run the API locally

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Run the Streamlit UI locally

```bash
uv run streamlit run streamlit_app.py
```

---

## Running with Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

The two services communicate over Docker Compose's internal network, with the Streamlit app calling the API by its service name rather than `localhost`.

---

## API Usage

**POST** `/predict`

```json
{
  "annual_inc": 98000,
  "emp_length": 3,
  "home_ownership": "MORTGAGE",
  "installment": 1000.59,
  "loan_amnt": 30000,
  "purpose": "debt_consolidation",
  "term": " 36 months",
  "int_rate": 0.1229,
  "avg_cur_bal": 16000,
  "inq_last_12m": 2,
  "max_bal_bc": 7600,
  "mo_sin_old_il_acct": 103,
  "mo_sin_old_rev_tl_op": 208,
  "mo_sin_rcnt_rev_tl_op": 8,
  "mo_sin_rcnt_tl": 8,
  "mort_acc": 3,
  "mths_since_last_delinq": 13,
  "num_bc_tl": 15,
  "num_il_tl": 8,
  "num_op_rev_tl": 25,
  "num_tl_90g_dpd_24m": 0,
  "num_tl_op_past_12m": 2,
  "open_acc": 28,
  "percent_bc_gt_75": 18.2,
  "pub_rec_bankruptcies": 1,
  "total_acc": 41,
  "total_bal_ex_mort": 50600
}
```

**Response**

```json
{
  "prediction": 0,
  "probability": 0.23
}
```

`prediction`: `1` = likely to default, `0` = not likely to default (based on a tuned decision threshold, not the default 0.5 cutoff).
`probability`: model's estimated probability of default.

`mths_since_last_delinq` can be omitted (`null`) for borrowers with no delinquency history — this is handled correctly by the pipeline's imputation step.

---

## Key Engineering Decisions

- **Single fitted pipeline over separate preprocessing artifacts** — avoids the classic train/serve skew problem where preprocessing logic drifts out of sync between training and serving code.
- **Threshold tuning over default 0.5 cutoff** — chosen per model based on the precision/recall trade-off, since imbalanced classification with class-weighting can produce very different probability distributions across model types.
- **`mlflow.pyfunc` with `pyfunc_predict_fn="predict_proba"`** for serving — keeps the serving code model-agnostic, so switching from Logistic Regression to a tree-based model later requires no changes to the API layer.
- **MLflow Model Registry** for versioned deployment — the Docker image bakes in a specific registered model version, downloaded explicitly rather than referencing MLflow's internal run storage paths directly.

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `feature-engine` · `XGBoost` · `Optuna` · `MLflow` · `FastAPI` · `Streamlit` · `Docker` · `uv`

---

## License

This project is for educational and portfolio purposes.
