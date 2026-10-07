# Adaptive Fraud Detection Platform

**Master's PFE project — Big Data and Cloud Computing (BDCC)**

This project was developed as a Master's end-of-studies project. It explores an adaptive fraud detection platform using the IEEE-CIS Fraud Detection dataset, machine learning, an API, and optional streaming services.

## What it does

- Builds transaction features from IEEE-CIS data.
- Scores transactions with an XGBoost model combined with a River online model.
- Serves predictions through a FastAPI REST API.
- Provides a Streamlit dashboard for transaction scoring, batch scoring, and model evaluation.
- Supports optional Kafka event streaming and Redis caching for SHAP explanations.

**Architecture:** IEEE-CIS data → feature engineering → XGBoost + River → FastAPI → dashboard / optional Kafka and Redis.

## Dataset and results

The models and metrics in this repository were generated from the IEEE-CIS training data supplied by the project owner. The original CSV files and generated Parquet files are excluded from GitHub because of their size and dataset terms. Obtain the data through the authorized Kaggle competition page and accept its rules before using it:

[IEEE-CIS Fraud Detection dataset on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

One chronological holdout run produced **ROC-AUC 0.898**, **PR-AUC 0.453**, and **fraud F1 0.469** at threshold `0.66`. These are research results for this run, not a guarantee of production performance. The current evaluation does not establish that adaptive learning improves detection.

## Requirements

- Python 3.13.5 (used for the included training run)
- Windows 11 PowerShell, Linux, or macOS
- Several GB of available RAM to rebuild features from the full IEEE-CIS dataset
- Redis and Kafka are optional for basic predictions

## Run on Windows

Run these commands from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.auth.init_admin
```

Create an administrator username and password when prompted.

Start the API in the first PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Start the dashboard in a second PowerShell window, from the same project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/serving/demo.py
```

Open the dashboard at [http://localhost:8501](http://localhost:8501). API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs); health status is at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

The API and dashboard can start using the model files in `data/models/`. The dashboard's sample held-out transaction screen also needs `data/processed/features.parquet`, which is intentionally excluded; create it with the data preparation steps below. The API can score raw transactions without that dashboard sample file.

## Build features and retrain

Place both authorized training files here:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Stop the API and dashboard before replacing the current model artifacts. From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_engineering
.\.venv\Scripts\python.exe -m src.models.train --trees 300 --online-rows 50000
```

Restart the API and dashboard after training. Check `/health`; `"synthetic": false` confirms the real-data artifacts are loaded. The training script uses a chronological 70/15/15 split and fits preprocessing on the training partition only.

## Optional Redis and Kafka

Kafka can run locally in Docker:

```powershell
docker compose up -d kafka
```

Redis is optional. The API can use a local Redis-compatible service such as Memurai; if port `6379` is already in use, do not start the Docker Redis service on the same port. Kafka requires the producer and consumer scripts to be started separately; starting the broker alone does not send or score events.

## Files intentionally excluded from GitHub

`.gitignore` excludes `.env`, virtual environments, CSV and Parquet data, local databases, logs, and online model state. Keep real credentials and dataset files on your machine. `.env.example` contains configuration placeholders only.

## Project structure

```text
src/             API, authentication, feature engineering, models, streaming
scripts/         Evaluation, demo generation, and test helpers
tests/           API and platform regression tests
data/models/     Trained model and preprocessing artifacts
data/raw/        Local IEEE-CIS CSV files (not committed)
data/processed/  Generated features (not committed)
```

This is an academic research prototype, not a production fraud prevention service.
