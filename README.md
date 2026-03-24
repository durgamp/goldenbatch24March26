# Golden Batch Analytics

A pharmaceutical batch manufacturing analytics platform built with Python, Streamlit, FastAPI, and PostgreSQL. The application identifies **golden batches** (top-performing historical batches), builds multivariate statistical process control (MSPC) models, and monitors new batches in real time.

---

## Features

| # | Step | Description |
|---|------|-------------|
| 1 | **CPP Identification** | Ranks Critical Process Parameters using Random Forest, XGBoost, PCA, and LASSO regression |
| 2 | **Golden Batch Trajectory** | Aligns batch time-series to 100 points; defines golden profile from top-25% yield batches |
| 3 | **Batch Monitoring** | MSPC (T², SPE) and PLS-based monitoring with empirical 95/99% control limits |
| 4 | **Overview / Upload** | Drag-and-drop CSV upload with automatic column detection |

---

## Architecture

```
goldenbatchapp/
├── app.py                        # Streamlit dashboard (4-page app)
├── config.py                     # CPP variables, labels, targets, thresholds
├── run.py                        # Launcher: starts FastAPI + Streamlit together
├── requirements.txt
│
├── data/
│   ├── real_data_loader.py       # detect_columns(), parse_batch_csv()
│   └── synthetic_generator.py   # (unused – kept for reference)
│
├── analysis/
│   ├── cpp_identification.py     # Step 1: feature extraction + 4 ranking methods
│   ├── trajectory_analysis.py   # Step 2: alignment, golden profile
│   └── batch_monitoring.py      # Step 3: MSPCModel, PLSBatchModel
│
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── routers/
│   │   ├── step1.py              # /api/step1/* endpoints
│   │   └── step2.py              # /api/step2/* endpoints
│   └── services/
│       ├── file_processor.py     # CSV parsing service
│       └── data_validator.py     # Input validation
│
├── db/
│   ├── connection.py             # SQLAlchemy engine setup
│   ├── models.py                 # ORM table definitions
│   └── repository.py            # CRUD helpers
│
└── tests/
    └── locustfile.py             # Load testing (Locust)
```

---

## Input Data Format

The app expects a **long-format CSV** file — one row per time step per batch:

```csv
batch_id,time_index,temperature,pH,dissolved_o2,agitation,pressure,feed_rate,foam_level,conductivity,yield
BATCH_001,0,36.8,7.19,41.2,188,1.01,39.5,14.8,6.4,
BATCH_001,1,36.9,7.20,40.8,190,1.00,40.1,15.0,6.5,
...
BATCH_001,99,37.1,7.21,39.5,191,0.99,40.0,15.2,6.6,92.3
```

- `batch_id` — unique batch identifier
- `time_index` — integer time step (does not need to be evenly spaced)
- CPP columns — any subset of the 8 default CPPs (or your own named columns)
- `yield` — optional; required for golden batch identification

---

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ running locally (or update `DATABASE_URL` to point elsewhere)

### 1. Clone the repository

```bash
git clone https://github.com/durgamp/goldenbatch24March26.git
cd goldenbatch24March26
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example below into a new `.env` file in the project root. **Never commit `.env` to version control.**

```env
# .env  (create this file manually)

# PostgreSQL
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/goldenbatch

# API / Streamlit
API_HOST=127.0.0.1
API_PORT=8000
STREAMLIT_PORT=8501
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 5. Create the database

```bash
# Connect to PostgreSQL and create the database
psql -U postgres -c "CREATE DATABASE goldenbatch;"
```

### 6. Run the application

**Option A — Streamlit only (no database required):**

```bash
streamlit run app.py
```

**Option B — Full stack (Streamlit + FastAPI):**

```bash
python run.py
```

Open your browser:
- Streamlit dashboard: [http://localhost:8501](http://localhost:8501)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Configuration

Edit [config.py](config.py) to adjust model behaviour:

| Variable | Default | Description |
|---|---|---|
| `CPP_VARIABLES` | 8 fermentation CPPs | List of expected CPP column names |
| `N_ALIGNED_POINTS` | `100` | Time points after batch alignment |
| `YIELD_GOLDEN_PERCENTILE` | `0.75` | Top 25% yield → golden |
| `YIELD_GOOD_PERCENTILE` | `0.50` | Top 50% yield → good |
| `MSPC_LIMIT_95` / `99` | `95` / `99` | Empirical control limit percentiles |
| `MSPC_N_COMPONENTS` | `5` | PCA components for MSPC |
| `PLS_N_COMPONENTS` | `5` | PLS components for batch monitoring |

---

## Load Testing

```bash
locust -f tests/locustfile.py --host=http://localhost:8000
```

Open [http://localhost:8089](http://localhost:8089) to configure and run the load test.

---

## Key Design Decisions

- **No synthetic fallback** — app requires real uploaded data; no dummy data is injected
- **Session state pipeline** — all processed data lives in `st.session_state`; `_build_pipeline()` in `app.py` rebuilds everything from the uploaded CSV
- **Empirical control limits** — MSPC uses 95th/99th percentile (not chi²/F), which is more interpretable for practitioners
- **Future imputation** — unknown future time points are filled with training-set mean for real-time monitoring
- **Unfolded matrix layout** — `[var1_t1…var1_t100, var2_t1…var2_t100, …]` (variable-wise unfolding)

---

## License

MIT
