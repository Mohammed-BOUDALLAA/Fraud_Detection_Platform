# Adaptive Fraud Detection Platform — repaired PFE project

**Read this first:** the uploaded archive contained source code only. It did not contain IEEE-CIS CSVs or trained model artifacts. The supplied `data/processed/features.parquet` contains **3,000 explicitly synthetic demonstration transactions**, not IEEE-CIS observations. The bundled models and metrics were generated from this fixture. They demonstrate software integration only and must not be reported as IEEE-CIS thesis results.

The pipeline supports the actual IEEE-CIS training files. Follow “Build real IEEE-CIS artifacts” below to replace the demonstration dataset and models.

## Quick start — Windows 11 / PowerShell

Use **Python 3.12** (tested on 3.12.14). Open a terminal in this folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.auth.init_admin
.\.venv\Scripts\python.exe -m src.serving.api
```

Create your own administrator username/password when prompted; there are no bundled credentials. Optionally set a random `JWT_SECRET_KEY` in `.env` to preserve sessions across restarts. Without one, the API creates a temporary signing key at process startup.

In a **second terminal**, in the same folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/serving/demo.py
```

Open the dashboard at http://localhost:8501. Sign in using the administrator account. API documentation: http://127.0.0.1:8000/docs. Health: http://127.0.0.1:8000/health.

On Linux/macOS, create the environment with `python3.12 -m venv .venv`, activate using `source .venv/bin/activate`, and use `python` for these commands. Installation and execution were tested on Linux; Windows-specific execution has not been run here. `requirements.txt` pins the tested direct dependencies. `requirements-linux-lock.txt` captures all installed Linux dependencies and is not a Windows installation file.

Redis and Kafka are optional for interactive prediction. Models are already included; you do not need to train to try the demonstration.

## Build real IEEE-CIS artifacts

1. Obtain the official competition data through your authorized Kaggle account and accept its rules: https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Put **both** `train_transaction.csv` and `train_identity.csv` in `data/raw/`.
3. Stop the API/dashboard while replacing artifacts. Back up `data/` if retaining the demo or previous model matters.
4. Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_engineering
.\.venv\Scripts\python.exe -m src.models.train --trees 300 --online-rows 50000
```

5. Restart the API and dashboard. Check `/health`: `synthetic` must be `false`, and inspect `dataset_manifest.json` for provenance and row count.

The training files are joined by `TransactionID` with one-to-one validation. Chronological 70/15/15 partitions use `TransactionDT`; ties are kept together. Missing targets, duplicate transaction IDs, and unusable splits fail clearly. Category mappings and card statistics are fitted on training rows only, then frozen for validation/test/inference. Unknown categories use -1; missing numeric values use -999. Card counts/means are training-reference statistics, **not** a real-time rolling velocity calculation. Hour/day features are offsets from the dataset's reference time, not verified local calendar times.

The preprocessor preserves all numeric input features (including V columns) and encodes categorical fields. The 15-column demo schema is only a small fixture; actual IEEE-CIS training produces its own larger schema. This workflow reads and joins the full dataset in memory, so allow several GB of RAM and avoid other memory-heavy applications. Full-scale IEEE-CIS execution has not been verified without the missing data.

XGBoost early stopping uses validation only. River is warmed on the last `--online-rows` training observations (default 50,000) to bound Python training time. Set this argument to the full training-partition size for full warm-up. A fusion decision threshold is selected for validation F1, then frozen for held-out test evaluation. No validation/test rows are used to train River. Class weighting means scores should not be interpreted as calibrated fraud probabilities.

## Included artifacts

| Path | Purpose |
|---|---|
| `data/processed/features.parquet` | Complete synthetic demonstration feature matrix + `isFraud` |
| `data/processed/row_metadata.parquet` | Row-aligned IDs and chronological offsets |
| `data/processed/dataset_manifest.json` | Provenance, schema size and split boundaries |
| `data/models/preprocessor.pkl` | Training-fitted feature transformation |
| `data/models/feature_cols.pkl` | Exact ordered model schema |
| `data/models/xgb_model.json` | Portable XGBoost model used by the API |
| `data/models/xgb_model.pkl` | Compatibility artifact |
| `data/models/river_model.pkl` | Initial online model |
| `data/models/training_metrics.json` | Held-out metrics, threshold, model version |
| `data/demo_raw/` | Reproducible synthetic transaction/identity fixture |
| `src/auth/`, `src/users/` | Authentication, role checks, users and audit database |
| `src/serving/` | REST API, optional Redis cache, dashboard |
| `src/ingestion/`, `src/streaming/` | Kafka replay and API-forwarding consumer |
| `tests/` | Isolated integration and regression tests |
| `legacy/` | Original dashboard/notebook/manual scripts for historical reference only |

Only load trusted pickle/joblib artifacts. Keep their dependency versions aligned or retrain using the supplied scripts. The original notebook contains historical outputs from the uploaded project; those outputs are not validation of this repaired version.

## API usage

Log in with JSON:

```json
{"username":"your-admin","password":"your-password"}
```

Send this to `POST /auth/login`, then use the returned token in `Authorization: Bearer <token>` (or Swagger's Authorize button).

- `POST /predict`: `{"transaction_id":"unique-id","features":{...},"input_type":"processed"}`. Processed input must match the saved feature schema; `null` numeric values become -999. Omit transaction ID to generate one automatically.
- For a raw transaction use `"input_type":"raw"`; at least `TransactionAmt`, `TransactionDT`, and `card1` are required. Provide available identity/transaction fields; unseen categories and absent optional model fields are handled by the preprocessor.
- `POST /predict/batch`: JSON list of 1–500 prediction requests. Returns a list of predictions. Requests are schema-validated before scoring; database writes are per event, so a later ID conflict can leave earlier events recorded. Replay unchanged IDs safely.
- `POST /feedback`: `{"transaction_id":"unique-id","ground_truth":1}`. Administrator only, after a stored prediction and verified label. Feedback is applied once per event and persisted. Duplicate feedback is ignored; label corrections require a separately designed correction workflow.
- `GET /model/info`, `POST /model/save`, `/auth/users`, `/auth/audit-logs`: model management, administrator user management and audit history.

Prediction never trains the online model. Supplying a non-null `ground_truth` to `/predict` is rejected. This prevents accidental label leakage and repeated benchmark training. Same ID + same features + same model returns its stored decision. Reusing an ID with different input/model returns 409. Use new IDs for new scoring measurements.

TreeSHAP values are computed directly by XGBoost and explain **batch-model log-odds contributions**, not the fused score. Only explanations are cached, with the model version included in the cache key. Redis failures disable the cache and fall back to local computation with bounded connection/read timeouts and no retries.

## Adaptive behavior and scientific interpretation

Fusion starts with alpha=0.8 for XGBoost and 0.2 for River. Verified feedback feeds the **stored pre-learning classification error** to ADWIN. A detected change lowers alpha to 0.5. Subsequent feedback recovers alpha by 0.005 per event. This is a heuristic to evaluate, not proof that River is better during drift. Online state (River, ADWIN, alpha, counters and deduplication IDs) is saved atomically after feedback and at shutdown.

`training_metrics.json` reports frozen batch, River and fusion test results. It does **not** establish the effectiveness of adaptive fusion, realistic delayed labels, or production fraud accuracy. Use `scripts/evaluate_adaptation.py` for a separate predict-then-learn held-out experiment and record the label-delay assumption. Do not invent improved metrics or reuse synthetic results in the thesis.

## Optional Redis and Kafka

```powershell
docker compose up -d
```

API/dashboard run on the host. The Docker services bind only to localhost. Start the API, log in, then export its token:

```powershell
$env:FRAUD_API_TOKEN = "paste-your-token"
.\.venv\Scripts\python.exe -m src.streaming.consumer
```

In another terminal:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion.producer --rows 100 --delay 0.05
```

Use `consumer --learn` with an administrator token only for explicitly supervised replay; it submits the known dataset label after scoring. Kafka events use stable IDs; retries are idempotent. Offsets commit only after successful scoring/feedback. Bad messages or expired credentials stop processing without committing the failed event. This implementation deliberately favors review of failed messages over silently dropping them. No labels are artificially flipped by default.

Docker/Kafka services were not available for a live broker test in the repair environment; the forwarding/retry contract is tested with mocks. Redis outage fallback is tested; live cache service behavior remains to validate locally.

## Validation and reproducibility

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py
```

This copies data into a temporary folder and uses a temporary database; it does not alter your live model state. Reproduce bundled demonstration artifacts with:

```powershell
.\.venv\Scripts\python.exe scripts/make_demo.py
.\.venv\Scripts\python.exe -m src.models.train --trees 80
```

**These commands replace the current processed data/model artifacts with synthetic ones.** Stop the server and back up first.

Run `scripts/benchmark.py` with `FRAUD_API_TOKEN` set to measure your own API P50/P95/P99. It uses unique prediction IDs, checks HTTP responses and sends no training labels.

## Scope and limitations

This is a local master's PFE research platform. Run **one API worker**, because adaptive state is owned by that process. Do not run two API instances against the same online-state file. Checkpoints are atomic but not a distributed database transaction; if persistence fails, investigate and restart before retrying. There is no production-grade high availability, rate limiter, credential lockout, retention policy, data drift dashboard or independently validated calibration. SQLite prediction history and feedback IDs grow with usage. API health reports model readiness and cache availability; it does not claim Kafka is healthy. Retrain with services stopped and use new transaction IDs after a model change.

Architecture: IEEE-CIS CSVs → training-fitted features → XGBoost + River artifacts → authenticated FastAPI → Streamlit. Kafka forwards through that same FastAPI owner; explicit feedback updates River/ADWIN and persists state. Redis caches batch explanations only.
