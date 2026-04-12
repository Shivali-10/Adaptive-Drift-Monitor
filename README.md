# Adaptive Drift Monitor — Fraud Detection MLOps Pipeline

> A production-grade, end-to-end MLOps system that detects credit card fraud, monitors live data for statistical drift, and triggers automated model retraining — powered by FastAPI, Streamlit, Supabase, Docker, and GitHub Actions.

---

## What This Project Does

Most fraud detection projects train a model and stop there. In the real world, fraud patterns evolve — what looked like fraud in 2020 looks different in 2024. This project solves that by building a **self-monitoring, self-healing ML pipeline**:

- Streams live transaction data from a cloud database (Supabase)
- Compares it against a baseline using statistical drift tests (KS-Test)
- Alerts when significant drift is detected
- Automatically retrains the model and replaces **fraud_model.pkl**
- Displays everything in a real-time Streamlit dashboard

---

## Architecture Overview

```
GitHub Actions (weekly cron)
        |
        v
live_streamer.py  ──────────────>  Supabase PostgreSQL (Cloud DB)
                                          |
                                          v
                                   dashboard.py  (Streamlit UI)
                                          |
                            +────────────+────────────+
                            |                         |
                      Display charts           Drift detected?
                                                      |
                                               api.py (FastAPI)
                                                      |
                                            fraud_model.pkl
                                                      |
                                        Predict: FRAUD / NORMAL
```

---

## Project Structure

```
fraud_drift_project/
│
├── Phase1_EDA.ipynb                  # Exploratory Data Analysis
├── Phase2_model.ipynb                # Model training (Random Forest + SMOTE)
├── Phase3_Drift_Simulation.ipynb     # Artificially simulating drift across 4 weeks
├── Phase4_Drift_Detection.ipynb      # KS-Test drift detection logic
├── Phase5_Performance_Monitoring.ipynb  # Precision / Recall / F1 over time
├── Phase6_Retraining_Trigger.ipynb   # Auto-retraining when drift threshold breached
│
├── api.py                            # FastAPI backend — predictions & drift API
├── dashboard.py                      # Streamlit frontend — live charts & metrics
├── live_streamer.py                  # Pushes CSV data to Supabase in batches
│
├── fraud_model.pkl                   # Trained Random Forest model (serialized)
├── week1_baseline.csv                # Original baseline data
├── week2_drift.csv                   # Simulated week 2 drift
├── week3_drift.csv                   # Simulated week 3 drift
├── week4_drift.csv                   # Simulated week 4 drift (most drifted)
│
├── Dockerfile                        # Lightweight backend container (no Streamlit)
├── docker-compose.yml                # One-command deployment
├── requirements.txt                  # All Python dependencies
│
├── .github/
│   └── workflows/
│       └── streamer.yml              # GitHub Actions — auto-runs live_streamer.py weekly
│
└── shap_*.png                        # SHAP explainability charts
```

---

## API Endpoints (`api.py`)

The FastAPI backend exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Model status + Supabase connection check |
| GET | `/live-data` | Fetch latest rows from Supabase |
| POST | `/predict` | Predict FRAUD or NORMAL for a transaction |
| POST | `/detect-drift` | Run KS-Test between baseline and live data |
| POST | `/retrain` | Retrain model on baseline + return new AUC |
| POST | `/simulate` | Inject normal or drifted rows into Supabase |

Interactive docs available at: `http://localhost:8000/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest Classifier (scikit-learn) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Drift Detection | KS-Test (scipy.stats) |
| Explainability | SHAP |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Cloud Database | Supabase (PostgreSQL) |
| Containerization | Docker + Docker Compose |
| CI/CD Automation | GitHub Actions |
| Data Processing | Pandas, NumPy |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized API)
- A free [Supabase](https://supabase.com) account

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud_drift_project.git
cd fraud_drift_project
```

### 2. Set up environment variables
Create a `.env` file in the root directory:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the Streamlit dashboard (in a new terminal)
```bash
streamlit run dashboard.py
```

### 6. Stream data to Supabase (in a new terminal)
```bash
python live_streamer.py
```

---

## Docker Deployment (Backend Only)

The Dockerfile intentionally excludes Streamlit to keep the backend image lightweight.

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`

---

## GitHub Actions — Automated Live Streaming

The workflow at `.github/workflows/streamer.yml` automatically runs `live_streamer.py` every **Monday at midnight UTC**, pushing fresh data to Supabase so the dashboard always shows live activity — even when your local machine is off.

### Setup
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Add two secrets:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
3. The workflow will run automatically, or you can trigger it manually via **Actions → Run workflow**

---

## Notebooks Walkthrough

| Notebook | Purpose |
|---|---|
| `Phase1_EDA.ipynb` | Understand the dataset — class imbalance, feature distributions |
| `Phase2_model.ipynb` | Train Random Forest with SMOTE to handle imbalanced classes |
| `Phase3_Drift_Simulation.ipynb` | Simulate 4 weeks of evolving fraud patterns mathematically |
| `Phase4_Drift_Detection.ipynb` | Apply KS-Test to detect statistical differences between weeks |
| `Phase5_Performance_Monitoring.ipynb` | Track precision, recall, F1 as drift increases |
| `Phase6_Retraining_Trigger.ipynb` | Retrain automatically when drift score exceeds threshold |

---

## Key Design Decisions

**Defensive Data Fetching**
The dashboard caps all Supabase queries at `limit=10000` and all CSV reads at `nrows=10000`. This ensures the UI never crashes on large production datasets while still providing statistically representative drift samples.

**Separation of Concerns**
The FastAPI backend (`api.py`) handles all predictions and heavy computation. The Streamlit frontend (`dashboard.py`) is a pure inference and display layer — it never processes raw data directly.

**Containerized Backend**
The `Dockerfile` filters out `streamlit` from `requirements.txt`, creating a lean backend image focused solely on serving predictions via the REST API.

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- 284,807 transactions
- 492 fraudulent (0.17% — highly imbalanced)
- Features V1–V28 are PCA-transformed for anonymity

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

Built as a final-year MLOps portfolio project demonstrating end-to-end production ML practices including drift monitoring, automated retraining, containerization, and cloud-native data pipelines.
