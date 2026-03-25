"""
Golden Batch Analytics – Streamlit Dashboard
=============================================
Frontend for the 3-step Golden Batch pipeline.
Calls the FastAPI backend (api/main.py) for all file-upload operations.

Run both servers with:   python run.py
Or separately:
    uvicorn api.main:app --port 8000
    streamlit run app.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import streamlit as st

from config import (
    CPP_VARIABLES, CPP_LABELS, CPP_TARGETS, QUALITY_COLORS,
    N_ALIGNED_POINTS, MSPC_N_COMPONENTS,
)
from data.real_data_loader import (
    detect_wide_columns, parse_wide_batch_summary,
    detect_pivot_columns, parse_pivot_file, merge_pivot_files,
)
from analysis.cpp_identification import (
    extract_batch_features, extract_rich_features, detect_outlier_batches,
    run_correlation_analysis,
    run_tree_importance, run_pca_analysis, run_lasso_analysis,
    compute_combined_ranking,
)
from analysis.trajectory_analysis import (
    align_batches, build_golden_profile, unfold_batches,
    dtw_distance_from_golden, compute_batch_conformance_score,
)
from analysis.batch_monitoring import fit_monitoring_models

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Golden Batch Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_get_bytes(endpoint: str) -> bytes | None:
    """GET request to API; return raw bytes or None on error."""
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        r.raise_for_status()
        return r.content
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API server. Make sure to start both servers with `python run.py`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return None


def _api_upload(endpoint: str, file_obj) -> dict | None:
    """POST a file to the API; return JSON response or None on error."""
    try:
        r = requests.post(
            f"{API_BASE}/{endpoint}",
            files={"file": (file_obj.name, file_obj.getvalue(), "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API server. Make sure to start both servers with `python run.py`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Upload failed: {detail or str(e)}")
        return None


def _api_post_json(endpoint: str, payload: dict) -> dict | None:
    """POST JSON to API; return response JSON or None on error."""
    try:
        r = requests.post(
            f"{API_BASE}/{endpoint}",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Confirm failed: {detail or str(e)}")
        return None




# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (real uploaded data, stored in session state)
# ─────────────────────────────────────────────────────────────────────────────

def _data_fingerprint(batch_df: pd.DataFrame, cpp_vars: list) -> str:
    """
    Compute a stable MD5 fingerprint of (batch_df content × cpp_vars).
    Used to skip `_build_pipeline` when the same data is re-uploaded.
    """
    h = hashlib.md5()
    try:
        h.update(pd.util.hash_pandas_object(batch_df, index=True).values.tobytes())
    except Exception:
        h.update(batch_df.to_csv(index=False).encode())
    h.update(str(sorted(cpp_vars)).encode())
    return h.hexdigest()


def _build_pipeline(batch_df, time_series, cpp_vars):
    """Align batches, build golden profile, extract features, fit models."""
    # ── Missing value strategy (non-exclusion) ────────────────────────────────
    # Real manufacturing data often contains legitimate missing points. We keep
    # ALL batches and use adaptive imputation per batch/CPP:
    #   - >=4 valid points: cubic spline + edge fill
    #   - >=2 valid points: linear interpolation + edge fill
    #   - 1 valid point: constant fill from that value
    #   - 0 valid points: global cross-batch median fallback
    # We track per-batch missingness and confidence rather than excluding data.
    imputed_counts  = {}   # { (batch_id, var): n_cells }
    imputation_methods = {}  # { (batch_id, var): method }
    batch_missing_summary = {}  # { batch_id: {max_missing_rate, mean_missing_rate, ...} }

    # Global fallback value per CPP for fully-missing series
    global_cpp_median = {}
    for v in cpp_vars:
        vals = []
        for _df in time_series.values():
            if v in _df.columns:
                arr = pd.to_numeric(_df[v], errors="coerce").to_numpy(dtype=float)
                vals.append(arr)
        if vals:
            all_vals = np.concatenate(vals)
            finite = all_vals[~np.isnan(all_vals)]
            global_cpp_median[v] = float(np.median(finite)) if len(finite) else 0.0
        else:
            global_cpp_median[v] = 0.0

    for bid, df in time_series.items():
        var_rates = []
        total_missing = 0
        total_cells = 0

        for v in cpp_vars:
            if v not in df.columns:
                continue

            s = pd.to_numeric(df[v], errors="coerce")
            n_total = int(len(s))
            n_nan = int(s.isna().sum())
            total_missing += n_nan
            total_cells += n_total

            rate = (n_nan / n_total) if n_total else 0.0
            var_rates.append(rate)

            if n_nan == 0:
                df[v] = s
                continue

            n_valid = n_total - n_nan
            method = ""

            if n_valid >= 4:
                # Prefer cubic for smooth process trajectories; fall back safely.
                try:
                    filled = s.interpolate(method="cubic", limit_direction="both")
                    method = "cubic"
                except Exception:
                    filled = s.interpolate(method="linear", limit_direction="both")
                    method = "linear-fallback"
            elif n_valid >= 2:
                filled = s.interpolate(method="linear", limit_direction="both")
                method = "linear"
            elif n_valid == 1:
                only_val = float(s.dropna().iloc[0])
                filled = s.fillna(only_val)
                method = "single-point-constant"
            else:
                fallback = float(global_cpp_median.get(v, 0.0))
                filled = s.fillna(fallback)
                method = "global-median"

            # Final safety for any remaining edge NaN.
            if filled.isna().any():
                fallback = float(global_cpp_median.get(v, 0.0))
                filled = filled.bfill().ffill().fillna(fallback)
                method = f"{method}+edge-fill"

            df[v] = filled
            imputed_counts[(bid, v)] = n_nan
            imputation_methods[(bid, v)] = method

        max_rate = float(max(var_rates)) if var_rates else 0.0
        mean_rate = float(np.mean(var_rates)) if var_rates else 0.0
        if max_rate <= 0.10:
            confidence = "High"
        elif max_rate <= 0.30:
            confidence = "Medium"
        else:
            confidence = "Low"

        batch_missing_summary[bid] = {
            "max_missing_rate": max_rate,
            "mean_missing_rate": mean_rate,
            "n_missing_cells": int(total_missing),
            "n_total_cells": int(total_cells),
            "confidence": confidence,
        }

    aligned = align_batches(time_series, variables=cpp_vars, n_points=N_ALIGNED_POINTS)

    # Try progressively lower golden percentiles when batch count is small;
    # record which percentile was actually used so the UI can warn the user.
    golden_pct_used = None
    for pct in [0.75, 0.67, 0.60, 0.50]:
        try:
            golden_profile = build_golden_profile(
                batch_df, aligned, variables=cpp_vars, yield_percentile=pct
            )
            golden_pct_used = pct
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            "Not enough batches to build a golden profile. "
            "Upload at least 5 batches with yield data."
        )

    # Build both feature sets: basic (mean+std) for backward compat,
    # rich (phase-aware + slope + AUC) for enhanced CPP ranking
    features      = extract_batch_features(time_series, variables=cpp_vars)
    rich_features = extract_rich_features(time_series,  variables=cpp_vars)
    yields = batch_df.set_index("batch_id")["yield"]

    # Detect outlier batches before modeling
    outlier_info = detect_outlier_batches(yields, features)

    X, batch_ids, _ = unfold_batches(aligned, variables=cpp_vars)

    # Impute any NaN in unfolded matrix with column means and track totals
    total_unfolded_nans = 0
    if np.isnan(X).any():
        total_unfolded_nans = int(np.isnan(X).sum())
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_means, nan_idx[1])

    y = np.array([yields[bid] for bid in batch_ids])
    yield_median = float(np.median(y))
    mspc, pls, mspc_meta = fit_monitoring_models(X, y, batch_ids, yield_median)

    pipeline_meta = {
        "golden_pct_used":    golden_pct_used,
        "imputed_ts_cells":   sum(imputed_counts.values()),
        "imputed_matrix_vals": total_unfolded_nans,
        "dropped_batches":    {},
        "batch_missing_summary": batch_missing_summary,
        "imputation_method_counts": pd.Series(list(imputation_methods.values())).value_counts().to_dict() if imputation_methods else {},
        "mspc_meta":          mspc_meta,
        "outlier_info":       outlier_info,
    }

    return aligned, golden_profile, features, yields, mspc, pls, X, batch_ids, pipeline_meta, rich_features

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        # ── Uploaded pipeline data ──────────────────────────────────────────
        "data_loaded":       False,
        "batch_df":          None,
        "time_series":       None,
        "aligned":           None,
        "golden_profile":    None,
        "features":          None,
        "yields":            None,
        "mspc_model":        None,
        "pls_model":         None,
        "X_unfolded":        None,
        "batch_ids_ordered": None,
        "active_cpp_vars":         None,   # CPP column names from uploaded data
        "data_format":             None,   # 'long' or 'wide'
        "batch_summary_features":  None,   # wide-format CPP features from File ①
        "rich_features":           None,   # phase-aware features (7/CPP)
        "target_yield":            None,   # user-specified target yield for Recommendations
        "pipeline_meta":           None,   # golden_pct_used, imputation counts, mspc_meta
        # ── CPP selection ───────────────────────────────────────────────────
        "validated_cpps":    CPP_VARIABLES[:],
        # ── Step 1 upload state (aggregate CPP format via API) ───────────────
        "s1_detection":      None,
        "s1_processed":      None,
        "s1_use_uploaded":   False,
        # ── Step 2 upload state (single trajectory via API) ──────────────────
        "s2_detection":      None,
        "s2_processed":      None,
        "s2_use_uploaded":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Shared chart helper
# ─────────────────────────────────────────────────────────────────────────────

def bar_chart(series: pd.Series, title: str, xlabel: str,
              highlight: list = None) -> go.Figure:
    colors = [
        "#E53935" if (highlight and v in highlight) else "#1976D2"
        for v in series.index
    ]
    fig = go.Figure(go.Bar(
        x=series.values, y=series.index, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in series.values], textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        height=max(350, 28 * len(series)),
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Golden Batch Analytics")
st.sidebar.caption("Pharma Batch Manufacturing | Fermentation Process")

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["Overview",
     "Step 1 – CPP Identification",
     "Step 2 – Golden Profile",
     "Step 3 – Batch Monitoring",
     "Recommendations",
     "Methodology & Models"],
)

# ── DB run history in sidebar ──────────────────────────────────────────────────
st.sidebar.divider()
with st.sidebar.expander("Run History (DB)", expanded=False):
    try:
        from db.connection import SessionLocal
        from db.repository import get_all_runs
        with SessionLocal() as _db:
            _runs = get_all_runs(_db)
        if _runs:
            for _r in _runs[:10]:
                _label = _r.run_name or _r.filename or f"Run #{_r.id}"
                st.sidebar.caption(
                    f"**#{_r.id}** {_label[:30]}  \n"
                    f"{_r.n_batches} batches · {_r.n_cpps} CPPs · "
                    f"{_r.created_at.strftime('%Y-%m-%d %H:%M') if _r.created_at else ''}"
                )
        else:
            st.sidebar.caption("No runs saved yet.")
    except Exception:
        st.sidebar.caption("DB unavailable.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Golden Batch Analytics")
    st.markdown(
        "End-to-end pipeline for identifying optimal operating parameters "
        "in pharma batch manufacturing and monitoring new batches against "
        "the **Golden Batch** reference profile."
    )

    # ── Data upload ────────────────────────────────────────────────────────────
    with st.expander(
        "Upload Historical Batch Data" if not st.session_state["data_loaded"]
        else "Historical Batch Data (loaded — expand to replace)",
        expanded=not st.session_state["data_loaded"],
    ):
        fmt = st.radio(
            "CSV format",
            ["Long-format time-series — two files (Steps 1, 2 & 3)",
             "Wide-format batch summary (Step 1 only)",
             "Pivot time-series — rows=timestamps, columns=batches (Steps 1, 2 & 3)"],
            horizontal=False,
            key="overview_fmt_radio",
        )
        is_wide  = fmt.startswith("Wide")
        is_pivot = fmt.startswith("Pivot")

        if is_wide:
            st.markdown("""
**Wide-format batch summary — one row per batch**

| Batch | CPP1 | CPP2 | … | CPP8 | Yield |
|-------|------|------|---|------|-------|
| B1    | 37.1 |  7.2 | … |  6.5 |  85.5 |
| B2    | 36.9 |  7.1 | … |  6.4 |  78.2 |

- `Batch` — unique identifier per batch (any column name)
- CPP columns — one aggregated value per parameter per batch
- `Yield` — optional; needed for CPP ranking in Step 1
- Steps 2 & 3 require long-format or pivot time-series data
""")
        elif is_pivot:
            st.markdown("""
**Pivot time-series — one file per CPP parameter**

Each file has rows = time steps and columns = batch IDs:

| Timestamp           | B1   | B2   | B3   | … | B29  |
|---------------------|------|------|------|---|------|
| 07-03-2026 00:00    | 37.1 | 36.9 | 37.0 | … | 37.3 |
| 07-03-2026 00:05    | 37.2 | 37.0 | 37.1 | … | 37.4 |

- Upload **one file per CPP** (e.g. temperature.csv, pH.csv, …)
- The CPP name is taken from the filename (editable below)
- Optionally provide yield values to enable full analysis
""")
        else:
            st.markdown("""
**Long-format time-series — two files required**

**File ①  Batch summary** (one row per batch):

| batch_id  | CPP1 | CPP2 | … | Yield |
|-----------|------|------|---|-------|
| BATCH_001 | 37.1 |  7.2 | … |  85.5 |
| BATCH_002 | 36.9 |  7.1 | … |  78.2 |

**File ②  Time-series** (rows = timestamps, columns = batches):

| Timestamp | BATCH_001 | BATCH_002 |
|-----------|-----------|-----------|
| 0         |      37.1 |      36.9 |
| 1         |      37.0 |      36.8 |

- (columns in File ②) − 1 must equal number of rows in File ①
- Batches are matched by position (left → right in File ② = top → bottom in File ①)
""")

        # ── Pivot format: multi-file upload ────────────────────────────────────
        if is_pivot:
            pivot_files = st.file_uploader(
                "Upload CPP files (one per parameter)",
                type=["csv"],
                accept_multiple_files=True,
                key="overview_pivot_uploader",
            )

            if pivot_files:
                st.markdown(f"**{len(pivot_files)} file(s) uploaded.** "
                            "Set the CPP name for each file below.")

                # Build per-file CPP name inputs + preview
                pivot_cpp_names = {}
                for f in pivot_files:
                    default_name = f.name.rsplit(".", 1)[0]   # strip .csv
                    col_a, col_b = st.columns([2, 3])
                    with col_a:
                        cpp_name = st.text_input(
                            f"CPP name for `{f.name}`",
                            value=default_name,
                            key=f"pivot_name_{f.name}",
                        )
                    pivot_cpp_names[f.name] = cpp_name

                st.markdown("**Yield data (optional)** — enter `batch_id: yield` pairs, "
                            "one per line (e.g. `B1: 85.5`)")
                yield_text = st.text_area(
                    "Yield values",
                    placeholder="B1: 85.5\nB2: 78.2\nB3: 91.0",
                    height=100,
                    key="pivot_yield_text",
                )

                pivot_load_btn = st.button(
                    "Parse & Build Pipeline", type="primary", key="pivot_load_btn"
                )

                if pivot_load_btn:
                    # Validate CPP names
                    names = list(pivot_cpp_names.values())
                    if len(names) != len(set(names)):
                        st.error("Duplicate CPP names — each file must have a unique name.")
                    else:
                        with st.spinner("Parsing pivot files and building pipeline…"):
                            try:
                                # Parse yield data
                                yield_data = {}
                                for line in yield_text.strip().splitlines():
                                    if ':' in line:
                                        k, v = line.split(':', 1)
                                        try:
                                            yield_data[k.strip()] = float(v.strip())
                                        except ValueError:
                                            pass

                                # Parse each pivot file
                                cpp_results = {}
                                for f in pivot_files:
                                    f.seek(0)
                                    raw = pd.read_csv(f)
                                    cpp_name = pivot_cpp_names[f.name]
                                    det = detect_pivot_columns(raw)
                                    for w in det["warnings"]:
                                        st.warning(f"[{f.name}] {w}")
                                    cpp_results[cpp_name] = parse_pivot_file(
                                        raw, det["time_col"], det["batch_cols"], cpp_name
                                    )

                                cpp_cols = list(cpp_results.keys())
                                b_df, ts = merge_pivot_files(
                                    cpp_results,
                                    yield_data if yield_data else None,
                                )

                                if not yield_data or b_df["yield"].isna().any():
                                    st.warning(
                                        "No yield data — quality labels and Steps 1 & 3 "
                                        "require yield values."
                                    )

                                _fp = _data_fingerprint(b_df, cpp_cols)
                                if (
                                    st.session_state.get("data_loaded")
                                    and st.session_state.get("_pipeline_fp") == _fp
                                ):
                                    st.info("Pipeline already built for this data — skipping recompute.")
                                    st.rerun()

                                (
                                    aln, gp, feat, ylds,
                                    mspc, pls, X_unf, bids, pmeta, rich_feat,
                                ) = _build_pipeline(b_df, ts, cpp_cols)

                                st.session_state.update({
                                    "data_loaded":       True,
                                    "data_format":       "long",
                                    "batch_df":          b_df,
                                    "time_series":       ts,
                                    "aligned":           aln,
                                    "golden_profile":    gp,
                                    "features":          feat,
                                    "rich_features":     rich_feat,
                                    "yields":            ylds,
                                    "mspc_model":        mspc,
                                    "pls_model":         pls,
                                    "X_unfolded":        X_unf,
                                    "batch_ids_ordered": bids,
                                    "active_cpp_vars":   cpp_cols,
                                    "validated_cpps":    cpp_cols[:],
                                    "pipeline_meta":     pmeta,
                                    "_pipeline_fp":      _fp,
                                })
                                try:
                                    from db.connection import SessionLocal
                                    from db.repository import save_batch_run, save_batches, save_golden_profile
                                    with SessionLocal() as _db:
                                        _run = save_batch_run(
                                            _db,
                                            filename=f"{len(pivot_files)} pivot file(s)",
                                            data_format="pivot",
                                            n_batches=len(b_df),
                                            n_cpps=len(cpp_cols),
                                            has_yield=bool(b_df["yield"].notna().any()),
                                            golden_pct_used=pmeta.get("golden_pct_used"),
                                            cpp_vars=cpp_cols,
                                            pipeline_meta=pmeta,
                                        )
                                        save_batches(_db, _run.id, b_df)
                                        if gp is not None:
                                            save_golden_profile(_db, _run.id, gp, cpp_cols)
                                        st.session_state["db_run_id"] = _run.id
                                except Exception as _db_exc:
                                    st.warning(f"Database save failed: {_db_exc}")
                                st.success(
                                    f"Loaded {len(b_df)} batches × {len(cpp_cols)} CPPs "
                                    f"from {len(pivot_files)} pivot file(s). Pipeline ready."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to build pipeline: {exc}")

        elif is_wide:
            # ── Wide format: single file ────────────────────────────────────────
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=["csv"],
                key="overview_uploader",
            )

            if uploaded_file is not None:
                raw_df = pd.read_csv(uploaded_file)
                det = detect_wide_columns(raw_df)
                for w in det["warnings"]:
                    st.warning(w)

                st.markdown(
                    f"**Detected:** `{len(raw_df)}` batches | "
                    f"`{len(det['cpp_cols'])}` CPP columns"
                )
                with st.expander("Preview (first 10 rows)"):
                    st.dataframe(raw_df.head(10), use_container_width=True)

                all_cols = list(raw_df.columns)
                with st.form("overview_confirm_form"):
                    batch_col = st.selectbox(
                        "Batch ID column", all_cols,
                        index=all_cols.index(det["batch_col"]) if det["batch_col"] in all_cols else 0,
                    )
                    yield_col_opts = ["(none)"] + all_cols
                    yield_default = (
                        yield_col_opts.index(det["yield_col"])
                        if det["yield_col"] and det["yield_col"] in yield_col_opts else 0
                    )
                    yield_col_sel = st.selectbox("Yield column (optional)", yield_col_opts, index=yield_default)
                    yield_col = None if yield_col_sel == "(none)" else yield_col_sel
                    skip = {c for c in [batch_col, yield_col] if c}
                    default_cpp = [c for c in det["cpp_cols"] if c not in skip]
                    cpp_col_str = st.text_area(
                        "CPP columns (one per line — remove any to exclude)",
                        value="\n".join(default_cpp),
                        height=140,
                    )
                    load_btn = st.form_submit_button("Load Data", type="primary")

                if load_btn:
                    cpp_cols = [c.strip() for c in cpp_col_str.strip().splitlines() if c.strip()]
                    missing = [c for c in cpp_cols if c not in raw_df.columns]
                    if missing:
                        st.error(f"Columns not found in file: {missing}")
                    elif len(cpp_cols) < 1:
                        st.error("Select at least 1 CPP column.")
                    elif yield_col and yield_col not in raw_df.columns:
                        st.error(f"Yield column '{yield_col}' not found.")
                    else:
                        with st.spinner("Parsing wide-format data…"):
                            try:
                                b_df, feat_df = parse_wide_batch_summary(
                                    raw_df, batch_col, cpp_cols, yield_col
                                )
                                ylds = b_df.set_index("batch_id")["yield"] if yield_col else None
                                if yield_col is None or b_df["yield"].isna().any():
                                    st.warning(
                                        "No yield data — CPP ranking in Step 1 requires "
                                        "a yield column."
                                    )
                                st.session_state.update({
                                    "data_loaded":       True,
                                    "data_format":       "wide",
                                    "batch_df":          b_df,
                                    "features":          feat_df,
                                    "yields":            ylds,
                                    "active_cpp_vars":   cpp_cols,
                                    "validated_cpps":    cpp_cols[:],
                                    "time_series":       None,
                                    "aligned":           None,
                                    "golden_profile":    None,
                                    "mspc_model":        None,
                                    "pls_model":         None,
                                    "X_unfolded":        None,
                                    "batch_ids_ordered": None,
                                })
                                try:
                                    from db.connection import SessionLocal
                                    from db.repository import save_batch_run, save_batches
                                    with SessionLocal() as _db:
                                        _run = save_batch_run(
                                            _db,
                                            filename=uploaded_file.name,
                                            data_format="wide",
                                            n_batches=len(b_df),
                                            n_cpps=len(cpp_cols),
                                            has_yield=bool(b_df["yield"].notna().any()),
                                            cpp_vars=cpp_cols,
                                        )
                                        save_batches(_db, _run.id, b_df)
                                        st.session_state["db_run_id"] = _run.id
                                except Exception as _db_exc:
                                    st.warning(f"Database save failed: {_db_exc}")
                                st.success(
                                    f"Loaded {len(b_df)} batches × {len(cpp_cols)} CPPs. "
                                    f"Step 1 (CPP Identification) is ready. "
                                    f"Upload long-format or pivot data for Steps 2 & 3."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to load data: {exc}")

        else:
            # ── Long format: two files (batch summary + time-series) ────────────
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                summary_file = st.file_uploader(
                    "① Batch summary (one row per batch)",
                    type=["csv"],
                    key="overview_summary_uploader",
                    help="One row per batch: batch_id, yield",
                )
            with col_f2:
                ts_file = st.file_uploader(
                    "② Time-series (rows=timestamps, columns=batches)",
                    type=["csv"],
                    key="overview_ts_uploader",
                    help="Pivot format: first column = timestamp, remaining columns = one per batch",
                )

            if summary_file and ts_file:
                sum_df  = pd.read_csv(summary_file)
                ts_raw  = pd.read_csv(ts_file)
                det_sum = detect_wide_columns(sum_df)
                det_ts  = detect_pivot_columns(ts_raw)

                for w in det_sum["warnings"]:
                    st.warning(f"[Summary] {w}")
                for w in det_ts["warnings"]:
                    st.warning(f"[Time-series] {w}")

                n_batches_sum = len(sum_df)
                n_batches_ts  = det_ts["n_batches"]

                if n_batches_sum != n_batches_ts:
                    st.error(
                        f"Batch count mismatch: File ① has {n_batches_sum} row(s) "
                        f"but File ② has {n_batches_ts} batch column(s). "
                        "Both files must contain the same number of batches."
                    )
                else:
                    st.markdown(
                        f"**Detected:** `{n_batches_sum}` batches | "
                        f"`{det_ts['n_timepoints']}` time steps | "
                        "batch columns matched by position"
                    )
                    with st.expander("Preview — Batch summary (first 10 rows)"):
                        st.dataframe(sum_df.head(10), use_container_width=True)
                    with st.expander("Preview — Time-series (first 10 rows)"):
                        st.dataframe(ts_raw.head(10), use_container_width=True)

                    sum_cols = list(sum_df.columns)
                    ts_cols  = list(ts_raw.columns)
                    with st.form("overview_confirm_form"):
                        st.markdown("**Batch summary columns**")
                        batch_col = st.selectbox(
                            "Batch ID column", sum_cols,
                            index=sum_cols.index(det_sum["batch_col"]) if det_sum["batch_col"] in sum_cols else 0,
                        )
                        yield_col_opts = ["(none)"] + sum_cols
                        yield_default = (
                            yield_col_opts.index(det_sum["yield_col"])
                            if det_sum["yield_col"] and det_sum["yield_col"] in yield_col_opts else 0
                        )
                        yield_col_sel = st.selectbox("Yield column (optional)", yield_col_opts, index=yield_default)
                        yield_col = None if yield_col_sel == "(none)" else yield_col_sel

                        st.markdown("**Time-series columns**")
                        time_col_ts = st.selectbox(
                            "Time index column (File ②)", ts_cols,
                            index=ts_cols.index(det_ts["time_col"]) if det_ts["time_col"] in ts_cols else 0,
                        )
                        cpp_name = st.text_input(
                            "CPP name (the single parameter measured in File ②)",
                            value="temperature",
                        )
                        load_btn = st.form_submit_button("Load Data & Build Pipeline", type="primary")

                    if load_btn:
                        cpp_name_clean = cpp_name.strip()
                        if not cpp_name_clean:
                            st.error("CPP name cannot be empty.")
                        elif yield_col and yield_col not in sum_df.columns:
                            st.error(f"Yield column '{yield_col}' not found in summary file.")
                        else:
                            with st.spinner("Parsing data and building pipeline…"):
                                try:
                                    # Batch IDs + yields from summary file
                                    b_df, _ = parse_wide_batch_summary(
                                        sum_df, batch_col, [], yield_col
                                    )
                                    batch_ids = list(b_df["batch_id"])

                                    # Extract all CPP columns from summary file for Step 1 ranking
                                    _skip_sum = {batch_col}
                                    if yield_col:
                                        _skip_sum.add(yield_col)
                                    sum_cpp_cols = [c for c in sum_df.columns if c not in _skip_sum]
                                    if sum_cpp_cols:
                                        sum_feat_df = sum_df[sum_cpp_cols].apply(
                                            pd.to_numeric, errors="coerce"
                                        ).copy()
                                        sum_feat_df.index = batch_ids
                                    else:
                                        sum_feat_df = None

                                    # Rename File ② batch columns → batch_ids by position
                                    rename_map = dict(zip(det_ts["batch_cols"], batch_ids))
                                    ts_renamed = ts_raw.rename(columns=rename_map)

                                    # Parse pivot time-series
                                    ts = parse_pivot_file(
                                        ts_renamed, time_col_ts, batch_ids, cpp_name_clean
                                    )
                                    cpp_cols = [cpp_name_clean]

                                    if yield_col is None or b_df["yield"].isna().any():
                                        st.warning(
                                            "No yield data — quality labels and Steps 1 & 3 "
                                            "require a yield column."
                                        )

                                    _fp = _data_fingerprint(b_df, cpp_cols)
                                    if (
                                        st.session_state.get("data_loaded")
                                        and st.session_state.get("_pipeline_fp") == _fp
                                    ):
                                        st.info("Pipeline already built for this data — skipping recompute.")
                                        st.rerun()

                                    (
                                        aln, gp, feat, ylds,
                                        mspc, pls, X_unf, bids_ord, pmeta, rich_feat,
                                    ) = _build_pipeline(b_df, ts, cpp_cols)

                                    st.session_state.update({
                                        "data_loaded":            True,
                                        "data_format":            "long",
                                        "batch_df":               b_df,
                                        "time_series":            ts,
                                        "aligned":                aln,
                                        "golden_profile":         gp,
                                        "features":               feat,
                                        "rich_features":          rich_feat,
                                        "yields":                 ylds,
                                        "mspc_model":             mspc,
                                        "pls_model":              pls,
                                        "X_unfolded":             X_unf,
                                        "batch_ids_ordered":      bids_ord,
                                        "active_cpp_vars":        cpp_cols,
                                        "validated_cpps":         cpp_cols[:],
                                        "batch_summary_features": sum_feat_df,
                                        "pipeline_meta":          pmeta,
                                        "_pipeline_fp":           _fp,
                                    })
                                    try:
                                        from db.connection import SessionLocal
                                        from db.repository import save_batch_run, save_batches, save_golden_profile
                                        with SessionLocal() as _db:
                                            _run = save_batch_run(
                                                _db,
                                                filename=f"{summary_file.name} + {ts_file.name}",
                                                data_format="long",
                                                n_batches=len(b_df),
                                                n_cpps=len(cpp_cols),
                                                has_yield=bool(b_df["yield"].notna().any()),
                                                golden_pct_used=pmeta.get("golden_pct_used"),
                                                cpp_vars=cpp_cols,
                                                pipeline_meta=pmeta,
                                            )
                                            save_batches(_db, _run.id, b_df)
                                            if gp is not None:
                                                save_golden_profile(_db, _run.id, gp, cpp_cols)
                                            st.session_state["db_run_id"] = _run.id
                                    except Exception as _db_exc:
                                        st.warning(f"Database save failed: {_db_exc}")
                                    n_sum_cpps = len(sum_cpp_cols) if sum_cpp_cols else 0
                                    st.success(
                                        f"Loaded {len(b_df)} batches | "
                                        f"Time-series: 1 CPP ({det_ts['n_timepoints']} time steps) | "
                                        f"Batch summary: {n_sum_cpps} CPP(s) for ranking. "
                                        "Pipeline ready."
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    import traceback
                                    st.error(f"Failed to build pipeline: {exc}")
                                    with st.expander("Error details"):
                                        st.code(traceback.format_exc())

        if st.session_state["data_loaded"]:
            if st.button("Clear loaded data", key="overview_clear"):
                for k in [
                    "data_loaded", "data_format", "batch_df", "time_series",
                    "aligned", "golden_profile", "features", "yields",
                    "mspc_model", "pls_model", "X_unfolded",
                    "batch_ids_ordered", "active_cpp_vars", "batch_summary_features",
                ]:
                    st.session_state[k] = None if k != "data_loaded" else False
                st.session_state["validated_cpps"] = CPP_VARIABLES[:]
                st.rerun()

    if not st.session_state["data_loaded"]:
        st.info("Upload historical batch data above to begin.")
        st.stop()

    # ── Pull from session state ────────────────────────────────────────────────
    batch_df    = st.session_state["batch_df"]
    time_series = st.session_state["time_series"]
    active_cpps = st.session_state["active_cpp_vars"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Batches",  len(batch_df))
    c2.metric("Golden Batches", int(batch_df["is_golden"].sum()))
    has_yield = batch_df["yield"].notna().all()
    c3.metric("Avg Yield", f"{batch_df['yield'].mean():.1f} %" if has_yield else "—")
    c4.metric("CPPs Monitored", len(active_cpps))

    # ── Pipeline quality warnings ──────────────────────────────────────────────
    pmeta = st.session_state.get("pipeline_meta") or {}
    if pmeta:
        _golden_pct = pmeta.get("golden_pct_used", 0.75)
        _mspc_meta  = pmeta.get("mspc_meta", {})
        _imp_ts     = pmeta.get("imputed_ts_cells", 0)
        _imp_mat    = pmeta.get("imputed_matrix_vals", 0)
        _miss_sum   = pmeta.get("batch_missing_summary", {})
        _imp_methods = pmeta.get("imputation_method_counts", {})

        if _golden_pct < 0.75:
            st.warning(
                f"⚠️ **Golden profile quality degraded.** With only {len(batch_df)} batches, "
                f"the golden profile could not be built from the top 25% of batches. "
                f"It was built from the top **{100 - int(_golden_pct*100)}%** instead "
                f"(yield ≥ {_golden_pct*100:.0f}th percentile). "
                "This means the 'golden' reference is closer to an average batch than a top-performing one. "
                "**Upload more batches** to improve the reference quality."
            )

        _mspc_label = _mspc_meta.get("threshold_label", "")
        _mspc_n     = _mspc_meta.get("n_good", 0)
        if _mspc_label and "75th" not in _mspc_label and "user target" not in _mspc_label:
            st.warning(
                f"⚠️ **MSPC trained on {_mspc_n} batches** at the {_mspc_label} threshold "
                "(fewer than ideal). Control limits may be too wide. "
                "More batches will improve monitoring precision."
            )
        else:
            st.info(
                f"✅ MSPC model trained on **{_mspc_n} quality batches** "
                f"({_mspc_meta.get('threshold_label', '')}). "
                "Control limits reflect genuine good-batch operating space."
            )

        if _miss_sum:
            _n_low = sum(1 for v in _miss_sum.values() if v.get("confidence") == "Low")
            _n_med = sum(1 for v in _miss_sum.values() if v.get("confidence") == "Medium")
            _n_high = sum(1 for v in _miss_sum.values() if v.get("confidence") == "High")
            _n_all = max(1, len(_miss_sum))
            st.info(
                "ℹ️ **No batches were excluded for missing data.** "
                f"Confidence tiers — High: {_n_high}/{_n_all}, Medium: {_n_med}/{_n_all}, Low: {_n_low}/{_n_all}."
            )
            if _n_low > 0:
                _low_items = [
                    (bid, vals.get("max_missing_rate", 0.0))
                    for bid, vals in _miss_sum.items()
                    if vals.get("confidence") == "Low"
                ]
                _low_items.sort(key=lambda x: x[1], reverse=True)
                preview = ", ".join(
                    f"`{bid}` ({rate*100:.0f}% max missing)" for bid, rate in _low_items[:8]
                )
                st.warning(
                    f"⚠️ **{_n_low} batch(es) have high missingness** and were heavily imputed. "
                    "They are retained as valid scenarios, but confidence is lower for those batches."
                    + (f"\n\nMost affected: {preview}" if preview else "")
                )

        if _imp_ts > 0 or _imp_mat > 0:
            total_cells = sum(
                len(df) * len([v for v in (st.session_state.get("active_cpp_vars") or []) if v in df.columns])
                for df in (st.session_state.get("time_series") or {}).values()
            ) or 1
            pct_imp = 100 * _imp_ts / total_cells
            method_msg = ""
            if _imp_methods:
                top_methods = sorted(_imp_methods.items(), key=lambda kv: kv[1], reverse=True)[:4]
                method_msg = " Methods used: " + ", ".join(f"{m} ({c})" for m, c in top_methods) + "."
            st.warning(
                f"⚠️ **Missing data handled:** {_imp_ts} time-series values ({pct_imp:.1f}% of CPP "
                "measurements) were imputed using adaptive methods (cubic, linear, constant, "
                "global-median fallback) to keep all batches in analysis."
                + method_msg
            )

    st.divider()
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Yield Distribution by Batch")
        if has_yield:
            fig = px.bar(
                batch_df.sort_values("yield", ascending=False).reset_index(drop=True),
                x="batch_id", y="yield", color="quality",
                color_discrete_map=QUALITY_COLORS,
                labels={"yield": "Yield (%)", "batch_id": "Batch"}, height=350,
            )
            fig.update_layout(xaxis_tickangle=45, plot_bgcolor="white", paper_bgcolor="white")
            fig.add_hline(y=batch_df["yield"].quantile(0.75), line_dash="dash",
                          line_color="gold", annotation_text="Golden threshold (75th pct)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No yield column — upload data with a yield column to see this chart.")

    with col_right:
        st.subheader("Yield Statistics")
        if has_yield:
            stat_df = batch_df.groupby("quality")["yield"].agg(["mean", "std", "count"])
            stat_df.columns = ["Mean %", "Std %", "Count"]
            stat_df = stat_df.reindex(
                [q for q in ["Golden", "Good", "Poor"] if q in stat_df.index]
            )
            st.dataframe(stat_df.style.format({"Mean %": "{:.1f}", "Std %": "{:.1f}"}),
                         use_container_width=True)
        else:
            st.info("No yield data available.")

        st.subheader("Batch Duration Distribution")
        fig2 = px.histogram(batch_df, x="duration", color="quality",
                            color_discrete_map=QUALITY_COLORS,
                            labels={"duration": "Batch Duration (steps)"},
                            barmode="overlay", opacity=0.7, height=220)
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    if time_series is not None:
        st.subheader("Raw CPP Trajectory Preview")
        if not active_cpps:
            st.info("No CPP columns detected in the loaded data.")
        else:
            var_pick = st.selectbox(
                "CPP", active_cpps,
                format_func=lambda v: CPP_LABELS.get(v, v),
            )
            quality_opts = [q for q in ["Golden", "Good", "Poor", "Unknown"]
                            if q in batch_df["quality"].unique()]
            quality_filter = st.multiselect("Show", quality_opts, default=quality_opts[:2])
            if var_pick is not None:
                fig3 = go.Figure()
                for bid in batch_df[batch_df["quality"].isin(quality_filter)]["batch_id"].tolist()[:30]:
                    q_rows = batch_df.loc[batch_df["batch_id"] == bid, "quality"]
                    if q_rows.empty:
                        continue
                    q = q_rows.values[0]
                    df_b = time_series.get(bid)
                    if df_b is None or var_pick not in df_b.columns:
                        continue
                    fig3.add_trace(go.Scatter(
                        x=df_b["time_index"], y=df_b[var_pick], mode="lines",
                        line=dict(color=QUALITY_COLORS.get(q, "#888888"), width=1),
                        opacity=0.5, showlegend=False,
                    ))
                target_val = CPP_TARGETS.get(var_pick)
                if target_val is not None:
                    fig3.add_hline(y=target_val, line_dash="dot", line_color="black",
                                   annotation_text=f"Target = {target_val}")
                fig3.update_layout(
                    title=f"Raw {CPP_LABELS.get(var_pick, var_pick)} Trajectories",
                    xaxis_title="Time Step (raw)", yaxis_title=CPP_LABELS.get(var_pick, var_pick),
                    height=380, plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(
            "Raw trajectory preview is not available for wide-format data. "
            "Upload long-format time-series data to see CPP trajectories."
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Step 1 – CPP Identification
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Step 1 – CPP Identification":
    st.title("Step 1 – Identifying Critical Process Parameters (CPPs)")
    st.markdown(
        "Four statistical techniques rank process variables by their influence on yield. "
        "Data comes from the historical batch time-series loaded on the Overview page. "
        "You can also upload a separate aggregate CPP summary file below."
    )

    if not st.session_state["data_loaded"]:
        st.warning("No data loaded. Please upload historical batch data on the Overview page.")
        st.stop()

    # ── Data Source ────────────────────────────────────────────────────────────
    _has_sum_features = st.session_state.get("batch_summary_features") is not None
    _loaded_label = (
        "Use loaded batch data (wide-format summary)"
        if st.session_state["data_format"] == "wide"
        else "Use loaded time-series features"
    )
    if _has_sum_features:
        _source_opts = ["Use batch summary (File ① – all CPPs)", _loaded_label]
    else:
        _source_opts = [_loaded_label]
    _source_opts.append("Upload aggregate CPP summary")

    data_source = st.radio(
        "Feature source",
        _source_opts,
        horizontal=True,
        key="s1_data_source",
    )
    st.session_state["s1_use_uploaded"] = (data_source == "Upload aggregate CPP summary")
    _use_batch_summary = _has_sum_features and (data_source == "Use batch summary (File ① – all CPPs)")

    # ── Upload panel ──────────────────────────────────────────────────────────
    if st.session_state["s1_use_uploaded"]:
        with st.container(border=True):
            st.subheader("Upload CPP Data")

            col_info, col_dl = st.columns([3, 1])
            with col_info:
                st.markdown("""
**Expected CSV format**

| CPP_Name | BATCH_001 | BATCH_002 | … |
|---|---|---|---|
| temperature | 37.2 | 36.8 | … |
| pH | 7.18 | 7.24 | … |
| yield | 87.2 | 82.1 | … |

- **Rows** = CPP names  (y-axis)
- **Columns** = Batch Serial Numbers  (x-axis)
- Include a **yield** row so the system can use it as the outcome variable
""")
            with col_dl:
                st.markdown("&nbsp;")
                template_bytes = _api_get_bytes("step1/template")
                if template_bytes:
                    st.download_button(
                        "Download Template",
                        data=template_bytes,
                        file_name="step1_cpp_template.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            uploaded = st.file_uploader(
                "Choose your CPP CSV file",
                type=["csv"],
                key="s1_file_uploader",
                help="Max 10 MB. First column = CPP names; remaining columns = batch IDs.",
            )

            # ── Auto-detection ─────────────────────────────────────────────
            if uploaded is not None:
                if st.button("Detect Columns", type="primary", key="s1_detect_btn"):
                    with st.spinner("Analysing file…"):
                        detection = _api_upload("step1/upload", uploaded)
                    if detection:
                        st.session_state["s1_detection"] = detection
                        st.session_state["s1_processed"] = None

            # ── Show detection results ─────────────────────────────────────
            if st.session_state["s1_detection"]:
                det = st.session_state["s1_detection"]
                st.divider()
                st.subheader("Detected Column Mapping")

                for w in det.get("warnings", []):
                    st.warning(w)

                st.table(pd.DataFrame({
                    "Field": [
                        "CPP label column",
                        "Batches detected",
                        "CPPs found",
                        "Yield row",
                    ],
                    "Detected value": [
                        det["cpp_column"],
                        str(det["n_batches"]),
                        ", ".join(det["detected_cpps"][:6])
                            + ("…" if len(det["detected_cpps"]) > 6 else ""),
                        det["yield_row_name"] if det["has_yield"] else "Not found",
                    ],
                }))

                with st.expander("Preview (first 5 rows)", expanded=True):
                    st.dataframe(pd.DataFrame(det["preview"]), use_container_width=True)

                # ── Confirmation form ──────────────────────────────────────
                st.subheader("Confirm or Correct the Column Mapping")
                st.caption("Only change if auto-detection got something wrong.")

                with st.form("s1_confirm_form"):
                    cpp_col = st.text_input(
                        "CPP label column name", value=det["cpp_column"],
                        help="Column that contains CPP names (e.g. 'CPP_Name')",
                    )
                    batch_cols_str = st.text_area(
                        "Batch columns (one per line)",
                        value="\n".join(det["batch_columns"]),
                        height=120,
                        help="Remove any columns you want to exclude.",
                    )
                    yield_row = st.text_input(
                        "Yield row name (blank if not present)",
                        value=det["yield_row_name"] or "",
                    )
                    confirmed = st.form_submit_button("Confirm & Load Data", type="primary")

                if confirmed:
                    batch_cols = [c.strip() for c in batch_cols_str.strip().splitlines() if c.strip()]
                    if len(batch_cols) < 2:
                        st.error("Need at least 2 batch columns.")
                    else:
                        payload = {
                            "cpp_column":     cpp_col,
                            "batch_columns":  batch_cols,
                            "yield_row_name": yield_row.strip() or None,
                        }
                        with st.spinner("Processing data…"):
                            processed = _api_post_json("step1/confirm", payload)
                        if processed:
                            st.session_state["s1_processed"] = processed
                            st.success(
                                f"Data loaded: {processed['n_batches']} batches × "
                                f"{processed['n_cpps']} CPPs. "
                                + ("Yield data included." if processed["yields"] else "No yield data.")
                            )

            if st.session_state["s1_processed"]:
                if st.button("Clear uploaded data", key="s1_clear"):
                    st.session_state["s1_detection"] = None
                    st.session_state["s1_processed"] = None
                    st.rerun()

    # ── Resolve active features / yields ──────────────────────────────────────
    if st.session_state["s1_use_uploaded"] and st.session_state["s1_processed"]:
        proc = st.session_state["s1_processed"]
        active_features = pd.DataFrame(proc["features"]).T
        active_features = active_features.apply(pd.to_numeric, errors="coerce")
        active_yields = (
            pd.Series(proc["yields"], name="yield") if proc["yields"] else None
        )
        data_note = f"Using **uploaded aggregate data** ({proc['n_batches']} batches, {proc['n_cpps']} CPPs)"
    elif _use_batch_summary:
        active_features = st.session_state["batch_summary_features"]
        active_yields   = st.session_state["yields"]
        if active_features is None or (hasattr(active_features, "empty") and active_features.empty):
            st.warning("Batch summary features not available. Please re-upload your data.")
            st.stop()
        n_b = len(st.session_state["batch_df"])
        n_c = len(active_features.columns)
        data_note = (
            f"Using **batch summary (File ①)** — {n_b} batches, {n_c} CPPs. "
            "All CPPs ranked by yield influence."
        )
    else:
        # Prefer rich features (phase-aware) over basic mean+std when available
        _rich = st.session_state.get("rich_features")
        if _rich is not None and not _rich.empty:
            active_features = _rich
            n_b = len(st.session_state["batch_df"])
            n_c = len(st.session_state["active_cpp_vars"])
            data_note = (
                f"Using **rich phase-aware features** ({n_b} batches, {n_c} CPPs × 7 features). "
                "Features include startup/reaction/harvest means, slope, and AUC — "
                "revealing *which phase* each CPP matters most in."
            )
        else:
            active_features = st.session_state["features"]
            n_b = len(st.session_state["batch_df"])
            n_c = len(st.session_state["active_cpp_vars"])
            fmt_label = "wide-format summary" if st.session_state["data_format"] == "wide" else "time-series"
            data_note = f"Using **loaded {fmt_label} data** ({n_b} batches, {n_c} CPPs)"
        active_yields = st.session_state["yields"]

    st.info(data_note)

    if active_yields is None:
        st.warning("No yield data found. Include a 'yield' row in your CSV to run CPP analysis.")
        st.stop()

    # ── Outlier detection warning ──────────────────────────────────────────────
    _pmeta = st.session_state.get("pipeline_meta") or {}
    _outlier_info = _pmeta.get("outlier_info", {})
    if _outlier_info.get("all_outliers"):
        _outs = _outlier_info["all_outliers"]
        st.warning(
            f"⚠️ **{len(_outs)} potential outlier batch(es) detected:** `{'`, `'.join(str(x) for x in _outs)}`.  \n"
            f"{_outlier_info.get('summary', '')}  \n"
            "These batches may distort CPP rankings and the golden profile. "
            "Review them carefully — if they represent equipment failures or data entry errors, "
            "consider excluding them before drawing conclusions."
        )

    # ── Target Yield Input ──────────────────────────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.subheader("🎯 Set Target Yield")
        st.markdown(
            "Define the yield % you want to reliably achieve. "
            "This drives the **Recommendations** page — the system will analyse which CPP ranges "
            "are associated with batches that hit this target."
        )
        _y_min = float(active_yields.min())
        _y_max = float(active_yields.max())
        _y_p75 = float(active_yields.quantile(0.75))
        _y_med = float(active_yields.median())
        _current_target = st.session_state.get("target_yield") or _y_p75
        _current_target = max(1.0, min(100.0, _current_target))

        tc1, tc2 = st.columns([2, 1])
        with tc1:
            target_yield_input = st.slider(
                "Target Yield (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(round(_current_target, 1)),
                step=0.5,
                key="s1_target_yield_slider",
                help="Batches at or above this yield will define the 'target-achieving' group for recommendations.",
            )
        with tc2:
            n_at_target = int((active_yields >= target_yield_input).sum())
            n_total_y = len(active_yields)
            pct_at = 100 * n_at_target / n_total_y
            st.metric("Batches achieving target", f"{n_at_target} / {n_total_y}")
            st.metric("Percentile", f"{pct_at:.0f}th")
            if n_at_target < 3:
                st.error("Need at least 3 target batches — lower the target.")
            else:
                st.success(f"Median yield of target batches: {active_yields[active_yields >= target_yield_input].median():.1f}%")

        st.session_state["target_yield"] = target_yield_input
        st.caption(
            f"Yield range in dataset: {_y_min:.1f}% – {_y_max:.1f}% | "
            f"Median: {_y_med:.1f}% | Top 25%: {_y_p75:.1f}%. "
            "Go to the **Recommendations** page (sidebar) to see the full analysis."
        )
    st.divider()

    common_idx = active_features.index.intersection(active_yields.index)
    if len(common_idx) < 5:
        st.error("Not enough matching batches between features and yield (need ≥ 5).")
        st.stop()
    active_features = active_features.loc[common_idx].copy()
    active_yields   = active_yields.loc[common_idx].copy()

    # Drop columns that are entirely NaN; fill remaining NaN with column mean
    active_features.dropna(axis=1, how="all", inplace=True)
    active_features.fillna(active_features.mean(), inplace=True)

    # Drop batches with NaN yield
    valid_yield_mask = active_yields.notna()
    if not valid_yield_mask.all():
        n_dropped = int((~valid_yield_mask).sum())
        st.warning(f"{n_dropped} batch(es) dropped due to missing yield values.")
        active_features = active_features.loc[valid_yield_mask]
        active_yields   = active_yields.loc[valid_yield_mask]

    if len(active_features) < 5:
        st.error("Too few valid batches after removing NaN values (need ≥ 5).")
        st.stop()
    if active_features.shape[1] < 1:
        st.error("No valid CPP columns remain after removing all-NaN columns.")
        st.stop()

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    tab_corr, tab_tree, tab_pca, tab_lasso, tab_val = st.tabs([
        "A. Correlation", "B. Tree Models", "C. PCA", "D. LASSO", "Engineer Validation"
    ])

    with tab_corr:
        st.subheader("Pearson & Spearman Correlation with Yield")
        st.markdown(
            "These charts answer: **which process parameters move together with yield?** "
            "A high correlation (|r| > 0.7) means that when this parameter changes, "
            "yield changes predictably — making it a strong CPP candidate. "
            "Both Pearson (linear) and Spearman (nonlinear/rank) are shown; "
            "agreement between them strengthens the evidence."
        )
        try:
            with st.spinner("Running correlation analysis…"):
                corr_res = run_correlation_analysis(active_features, active_yields)

            # Dynamic insight banner
            if len(corr_res["combined"]) > 0:
                top_feat = corr_res["combined"].index[0]
                top_score = float(corr_res["combined"].iloc[0])
                top_label = CPP_LABELS.get(top_feat.replace("_mean","").replace("_std",""), top_feat)
                strength = "strong" if top_score > 0.7 else ("moderate" if top_score > 0.4 else "weak")
                st.info(
                    f"**Key finding:** `{top_feat}` is the highest-ranked feature "
                    f"(combined score {top_score:.2f} — {strength} correlation with yield). "
                    f"Focus process control efforts on **{top_label}** first."
                )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    bar_chart(corr_res["pearson"].abs().sort_values(ascending=False),
                              "Pearson |r| vs Yield", "|r|"),
                    use_container_width=True,
                )
                st.caption(
                    "**Pearson r** measures linear correlation. "
                    "A bar reaching 0.7+ indicates a near-straight-line relationship "
                    "between this parameter and yield. Red points = control levers."
                )
            with c2:
                st.plotly_chart(
                    bar_chart(corr_res["spearman"].abs().sort_values(ascending=False),
                              "Spearman |ρ| vs Yield", "|ρ|"),
                    use_container_width=True,
                )
                st.caption(
                    "**Spearman ρ** captures monotone (not just linear) relationships. "
                    "If Spearman ranks a feature highly but Pearson does not, "
                    "the effect is nonlinear — e.g. yield drops sharply outside a narrow band."
                )
            st.plotly_chart(
                bar_chart(corr_res["combined"], "Combined Score (avg |r|, |ρ|)", "Score"),
                use_container_width=True,
            )
            st.caption(
                "**Combined score** averages both methods. "
                "Use this as the primary ranking for shortlisting CPPs. "
                "Parameters with `_std` suffix represent **variability** within a batch — "
                "high std scores mean process instability hurts yield more than the average level."
            )

            # ── Top-3 CPP vs Yield scatter plots ──────────────────────────────
            st.markdown("#### Top CPP vs Yield — Direct Evidence")
            st.markdown(
                "The scatter plots below show the **raw relationship** between each "
                "top-ranked parameter and yield. Each dot is one batch. "
                "A tight cluster around the trend line confirms the correlation is real, "
                "not a statistical artefact."
            )
            top3_feats = corr_res["combined"].head(3).index.tolist()
            y_arr = active_yields.values
            scatter_cols = st.columns(min(3, len(top3_feats)))
            for _si, _feat in enumerate(top3_feats):
                if _feat not in active_features.columns:
                    continue
                x_arr = active_features[_feat].values
                _valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
                x_v, y_v = x_arr[_valid], y_arr[_valid]
                from scipy.stats import pearsonr as _pearsonr
                _r, _p = _pearsonr(x_v, y_v) if len(x_v) >= 3 else (0, 1)
                # Regression line
                _m, _b = np.polyfit(x_v, y_v, 1) if len(x_v) >= 2 else (0, 0)
                x_line = np.linspace(x_v.min(), x_v.max(), 50)
                y_line = _m * x_line + _b
                _feat_label = CPP_LABELS.get(_feat.replace("_mean","").replace("_std",""), _feat)
                _pstr = f"p={'<0.001' if _p < 0.001 else f'{_p:.3f}'}"
                _fig_sc = go.Figure()
                _fig_sc.add_trace(go.Scatter(
                    x=x_v, y=y_v, mode="markers",
                    marker=dict(
                        size=8, color=y_v,
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="Yield", thickness=10),
                    ),
                    name="Batches",
                    hovertemplate=f"{_feat_label}: %{{x:.3f}}<br>Yield: %{{y:.2f}}<extra></extra>",
                ))
                _fig_sc.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color="steelblue", width=2, dash="dot"),
                    name=f"Trend (r={_r:+.2f})",
                    showlegend=True,
                ))
                _fig_sc.update_layout(
                    title=dict(text=f"{_feat_label}<br><sup>r={_r:+.2f}, {_pstr}</sup>", font_size=13),
                    xaxis_title=_feat,
                    yaxis_title="Yield",
                    height=300,
                    margin=dict(l=40, r=20, t=60, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                with scatter_cols[_si]:
                    st.plotly_chart(_fig_sc, use_container_width=True)
                    _sig = "✅ Statistically significant" if _p < 0.05 else "⚠️ Not significant (p≥0.05)"
                    st.caption(f"{_sig} | r={_r:+.2f}, {_pstr}")

            with st.expander("How to read these charts — decision guide"):
                st.markdown("""
| Combined score | Decision |
|---|---|
| **0.7 – 1.0** | Strong driver → **must control tightly** |
| **0.4 – 0.7** | Moderate → include in monitoring, investigate further |
| **0.0 – 0.4** | Weak → likely not a direct yield lever; deprioritise |

**`_std` vs `_mean` features:**
- `parameter_mean` = average level during the batch
- `parameter_std` = how much it fluctuated during the batch

Variability (`_std`) often ranks higher because *unstable* conditions hurt yield
even when the average value looks acceptable.

**Next step:** Compare these rankings with the Tree Models tab.
If both agree on the same top CPPs, you have high confidence they are truly critical.
""")
        except Exception as exc:
            st.error(f"Correlation analysis failed: {exc}")

    with tab_tree:
        st.subheader("Feature Importance – Random Forest & XGBoost")
        st.markdown(
            "Tree models learn which parameters they need to **split on most to predict yield accurately**. "
            "Unlike correlation, tree importance captures complex interactions and threshold effects "
            "(e.g. 'yield drops sharply only when temperature AND pH are both off-target together'). "
            "High importance = the model relies on this CPP to distinguish high-yield from low-yield batches."
        )
        try:
            with st.spinner("Training models…"):
                tree_res = run_tree_importance(active_features, active_yields)

            # Surface overfitting warning if sample size is small
            if tree_res.get("sample_warning"):
                st.warning(tree_res["sample_warning"])
            else:
                st.success(
                    f"✅ Adequate sample size for tree models "
                    f"(n={tree_res['n_samples']} batches, p={tree_res['n_features']} features). "
                    "Results are reasonably stable."
                )

            # Dynamic insight
            if len(tree_res["combined"]) > 0:
                top_tree = tree_res["combined"].index[0]
                top_tree_label = CPP_LABELS.get(top_tree.replace("_mean","").replace("_std",""), top_tree)
                top_tree_score = float(tree_res["combined"].iloc[0])
                st.info(
                    f"**Key finding:** Tree models agree that `{top_tree}` "
                    f"(normalised importance {top_tree_score:.3f}) is the most influential parameter. "
                    f"Controlling **{top_tree_label}** is the highest-leverage action to improve yield."
                )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    bar_chart(tree_res["random_forest"].head(16),
                              "Random Forest Feature Importance", "Importance"),
                    use_container_width=True,
                )
                st.caption(
                    "**Random Forest** averages importance across 300 trees, "
                    "making it robust to individual data quirks. "
                    "Long bars = consistently useful splits across many trees."
                )
            with c2:
                if tree_res["xgboost"] is not None:
                    st.plotly_chart(
                        bar_chart(tree_res["xgboost"].head(16),
                                  "XGBoost Feature Importance", "Importance"),
                        use_container_width=True,
                    )
                    st.caption(
                        "**XGBoost** uses gradient boosting — it focuses on the hardest-to-predict batches. "
                        "Features that rank high here are critical for getting *edge cases* right."
                    )
                else:
                    st.warning("XGBoost not installed. Run `pip install xgboost`.")
            st.plotly_chart(
                bar_chart(tree_res["combined"].head(16),
                          "Combined Tree Importance (normalised avg)", "Normalised Importance"),
                use_container_width=True,
            )
            st.caption(
                "**Combined importance** normalises and averages RF and XGBoost scores. "
                "Parameters appearing in the top 5 of *both* RF and XGBoost are the most reliable CPP candidates. "
                "A long tail of near-zero importance features can safely be deprioritised."
            )
            with st.expander("Comparing to the Correlation tab — what to look for"):
                st.markdown("""
**CPPs that rank highly in BOTH Correlation and Tree Models are your highest-confidence critical parameters.**

| Scenario | Interpretation |
|---|---|
| High correlation + high tree importance | Definite CPP — control tightly |
| High correlation, low tree importance | Simple linear effect; may be a proxy for another variable |
| Low correlation, high tree importance | Nonlinear or interaction effect — needs careful range control |
| Low in both | Unlikely to be a primary yield driver |
""")
        except Exception as exc:
            st.error(f"Tree importance failed: {exc}")

    with tab_pca:
        st.subheader("PCA – Loadings of the Yield-Relevant Component")
        st.markdown(
            "PCA compresses all CPP variation into a small number of 'directions' (principal components). "
            "**The component most correlated with yield tells us which combination of parameters "
            "collectively drives batch outcomes.** "
            "High-loading features on that component are the hidden levers behind yield variation."
        )
        try:
            with st.spinner("Running PCA…"):
                pca_res = run_pca_analysis(active_features, active_yields)

            imp_pc = pca_res["important_pc"]
            imp_corr = float(pca_res["yield_corr"][imp_pc])
            top_pca_feat = pca_res["cpp_importance"].index[0]
            top_pca_label = CPP_LABELS.get(top_pca_feat.replace("_mean","").replace("_std",""), top_pca_feat)

            st.info(
                f"**{imp_pc}** explains the most yield variation (correlation with yield: {imp_corr:.3f}). "
                f"The dominant feature on this component is `{top_pca_feat}` — "
                f"meaning **{top_pca_label}** is a core driver of the yield patterns seen across batches."
            )

            c1, c2 = st.columns([1, 2])
            with c1:
                ev = pca_res["explained_variance_ratio"]
                ev_df = pd.DataFrame({
                    "PC": [f"PC{i+1}" for i in range(len(ev))],
                    "Variance %": (ev * 100).round(1),
                    "Corr w/ Yield": pca_res["yield_corr"].values,
                })
                st.dataframe(ev_df.style.format(
                    {"Variance %": "{:.1f}", "Corr w/ Yield": "{:.3f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "**Variance %**: how much process variation this component captures. "
                    "**Corr w/ Yield**: how strongly this 'direction' of variation predicts yield. "
                    "The highlighted PC is used for CPP ranking."
                )
                st.success(
                    f"**{imp_pc}** has the highest yield correlation ({imp_corr:.3f}). "
                    "Its loadings (right chart) identify which CPPs are bundled into this yield driver."
                )
            with c2:
                st.plotly_chart(
                    bar_chart(pca_res["cpp_importance"],
                              f"Loadings of {imp_pc} (yield-relevant PC)", "|Loading|"),
                    use_container_width=True,
                )
                st.caption(
                    "**High loading = this feature moves strongly in the direction that drives yield.** "
                    "Features with loading > 0.3 are typically CPP candidates. "
                    "Both positive and negative loadings matter — "
                    "a negative loading means higher values of that feature push yield *lower*."
                )
            fig_heat = px.imshow(
                pca_res["loadings"], color_continuous_scale="RdBu_r", aspect="auto",
                title="PCA Loadings Heatmap — all components × all features",
                labels={"color": "Loading"},
            )
            fig_heat.update_layout(height=320)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "**Heatmap guide:** Red = strong positive loading (feature rises with this PC), "
                "Blue = strong negative loading, White ≈ 0 (feature irrelevant to this PC). "
                "Rows are PCs (process 'modes'), columns are CPP features. "
                "Look at the row for the yield-relevant PC to see which features define it."
            )
        except Exception as exc:
            st.error(f"PCA analysis failed: {exc}")

    with tab_lasso:
        st.subheader("LASSO Regression – Automatic Feature Selection")
        st.markdown(
            "LASSO (Least Absolute Shrinkage and Selection Operator) fits a regression model "
            "while actively **penalising unnecessary parameters**. "
            "It automatically sets the coefficient of irrelevant features to exactly zero, "
            "giving a definitive yes/no answer on whether a CPP should be controlled. "
            "This is the most conservative of the four methods."
        )
        try:
            with st.spinner("Running LASSO CV…"):
                lasso_res = run_lasso_analysis(active_features, active_yields)

            n_selected = len(lasso_res["selected"])
            n_total = len(active_features.columns)
            n_eliminated = n_total - n_selected
            st.info(
                f"Optimal regularisation (5-fold CV alpha = {lasso_res['best_alpha']:.4f}) | "
                f"**{n_selected} CPPs selected** as yield drivers | "
                f"**{n_eliminated} eliminated** (coefficient shrunk to zero — not significant)."
            )

            c1, c2 = st.columns(2)
            with c1:
                selected_coef = (
                    lasso_res["coefficients"][lasso_res["selected"]]
                    .sort_values(key=abs, ascending=False)
                )
                st.plotly_chart(
                    bar_chart(selected_coef.abs(), "LASSO |Coefficient| (selected only)", "|Coef|",
                              highlight=list(selected_coef.index)),
                    use_container_width=True,
                )
                st.caption(
                    "**Selected features only.** Larger bar = stronger marginal effect on yield "
                    "holding all other CPPs constant. "
                    "These are the parameters LASSO says are **statistically necessary** to predict yield."
                )
                if lasso_res["selected"]:
                    top_lasso = lasso_res["selected"][0]
                    coef_val = float(lasso_res["coefficients"][top_lasso])
                    direction = "increases" if coef_val > 0 else "decreases"
                    top_lasso_label = CPP_LABELS.get(top_lasso.replace("_mean","").replace("_std",""), top_lasso)
                    st.markdown(
                        f"> **Interpretation:** When `{top_lasso}` increases by 1 standard deviation, "
                        f"yield **{direction}** by ~{abs(coef_val):.2f} units (all else equal). "
                        f"Focus on **{top_lasso_label}** for the greatest yield leverage."
                    )
            with c2:
                fig_all = go.Figure(go.Bar(
                    x=lasso_res["coefficients"].index,
                    y=lasso_res["coefficients"].values,
                    marker_color=["#E53935" if v != 0 else "#90A4AE"
                                  for v in lasso_res["coefficients"].values],
                ))
                fig_all.update_layout(
                    title="All LASSO Coefficients — grey bars eliminated, red bars kept",
                    xaxis_tickangle=45, height=380,
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig_all, use_container_width=True)
                st.caption(
                    "**Grey bars = coefficient forced to zero** by LASSO — these CPPs do not "
                    "independently predict yield and can be de-prioritised. "
                    "**Red bars = retained** — these are the yield-relevant parameters. "
                    "Positive = higher value → higher yield; Negative = higher value → lower yield."
                )
        except Exception as exc:
            st.error(f"LASSO analysis failed: {exc}")

    with tab_val:
        st.subheader("Engineer Validation – Combined Ranking")
        st.markdown(
            "This tab aggregates all four methods into a single ranked list. "
            "**Your job as a process engineer:** review the ranking, apply domain knowledge, "
            "and confirm which CPPs to monitor in Steps 2 and 3. "
            "The algorithm narrows the field — you make the final call."
        )
        try:
            with st.spinner("Computing combined ranking…"):
                corr_v  = run_correlation_analysis(active_features, active_yields)
                tree_v  = run_tree_importance(active_features, active_yields)
                pca_v   = run_pca_analysis(active_features, active_yields)
                lasso_v = run_lasso_analysis(active_features, active_yields)
                ranking = compute_combined_ranking(corr_v, tree_v, pca_v, lasso_v)

            var_ranking = (
                ranking.groupby("base_variable")["avg_score"]
                .max().sort_values(ascending=False).reset_index()
            )
            var_ranking.columns = ["variable", "avg_score"]

            # Tier-based insight
            top3 = var_ranking.head(3)["variable"].tolist()
            top3_labels = [CPP_LABELS.get(v, v) for v in top3]
            cutoff_strong = float(var_ranking["avg_score"].quantile(0.75))
            strong_cpps = var_ranking[var_ranking["avg_score"] >= cutoff_strong]["variable"].tolist()
            strong_labels = [CPP_LABELS.get(v, v) for v in strong_cpps]
            st.success(
                f"**Top CPPs across all 4 methods:** {', '.join(top3_labels)}. "
                f"Parameters scoring above the 75th percentile ({cutoff_strong:.2f}): "
                f"**{', '.join(strong_labels)}** — these are your primary control targets."
            )

            st.plotly_chart(
                bar_chart(var_ranking.set_index("variable")["avg_score"],
                          "Combined CPP Importance (all 4 methods) — higher = more critical",
                          "Avg Normalised Score",
                          highlight=st.session_state["validated_cpps"]),
                use_container_width=True,
            )
            st.caption(
                "Score is normalised 0→1 across all methods and averaged. "
                "**Red bars** = currently validated CPPs (from previous save). "
                "The natural elbow in the chart separates strong from weak CPPs — "
                "include everything above the elbow in your final selection."
            )
            st.divider()
            st.subheader("Select Final CPPs")
            st.markdown(
                "Check the parameters you want to track in the Golden Profile and Batch Monitoring steps. "
                "**Recommendation:** select all CPPs with score above 0.40, "
                "plus any parameters you have strong process reason to include regardless of score."
            )

            available_vars = var_ranking["variable"].tolist()

            with st.form("cpp_validation_form"):
                selected = []
                cols = st.columns(4)
                for i, var in enumerate(available_vars):
                    label_text = CPP_LABELS.get(var, var)
                    row_score = var_ranking.loc[var_ranking["variable"] == var, "avg_score"]
                    score = float(row_score.values[0]) if len(row_score) > 0 else 0.0
                    default = var in st.session_state["validated_cpps"]
                    if cols[i % 4].checkbox(f"{label_text}  *(score: {score:.2f})*",
                                            value=default, key=f"cpp_{var}"):
                        selected.append(var)

                if st.form_submit_button("Save CPP Selection", type="primary"):
                    if len(selected) < 2:
                        st.error("Select at least 2 CPPs.")
                    else:
                        st.session_state["validated_cpps"] = selected
                        try:
                            from db.connection import SessionLocal
                            from db.repository import save_cpp_rankings
                            _run_id = st.session_state.get("db_run_id")
                            if _run_id:
                                _rank_df = ranking.rename(columns={"feature": "base_variable"})
                                with SessionLocal() as _db:
                                    save_cpp_rankings(_db, _run_id, _rank_df)
                        except Exception as _db_exc:
                            pass  # non-blocking
                        st.success(f"Saved {len(selected)} CPPs: {', '.join(selected)}")

            st.divider()
            st.dataframe(
                ranking[["rank", "feature", "corr_score", "tree_score",
                          "pca_score", "lasso_score", "avg_score"]].style.format(
                    {c: "{:.3f}" for c in ["corr_score", "tree_score",
                                            "pca_score", "lasso_score", "avg_score"]}
                ),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Combined ranking failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Step 2 – Golden Profile
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Step 2 – Golden Profile":
    st.title("Step 2 – CPP Trajectories & Golden Profile")
    st.markdown(
        "Compare batch trajectories against the Golden Profile "
        "(mean ± 2σ of the top-25 % yield batches). "
        "Upload a single new trajectory to compare it against the golden band."
    )

    if not st.session_state["data_loaded"]:
        st.warning("No data loaded. Please upload historical batch data on the Overview page.")
        st.stop()

    if st.session_state["data_format"] == "wide":
        st.info(
            "Step 2 requires long-format time-series data (one row per time step per batch). "
            "Go to the Overview page and upload a long-format CSV to access this step."
        )
        st.stop()

    # Pull from session state
    batch_df       = st.session_state["batch_df"]
    aligned        = st.session_state["aligned"]
    golden_profile = st.session_state["golden_profile"]
    time_series    = st.session_state["time_series"]
    validated_cpps = st.session_state["validated_cpps"]

    # ── Reconcile validated_cpps with golden_profile keys ─────────────────────
    # If the saved validated_cpps has no overlap with golden_profile (e.g. after
    # a Step-1 CPP edit or a fresh session), reset it to the full set of CPPs
    # that are actually present in the profile so every tab renders correctly.
    _gp_cpp_keys = golden_profile.get("variables", [])
    if not _gp_cpp_keys:
        _gp_cpp_keys = [k for k in golden_profile if isinstance(golden_profile.get(k), dict)]
    if validated_cpps and not any(v in golden_profile for v in validated_cpps):
        validated_cpps = _gp_cpp_keys
        st.session_state["validated_cpps"] = validated_cpps
    elif not validated_cpps:
        validated_cpps = _gp_cpp_keys
        st.session_state["validated_cpps"] = validated_cpps

    # ── Optional: upload a single trajectory to compare ───────────────────────
    temp_source = st.radio(
        "Single-trajectory comparison",
        ["No upload – browse historical data only", "Upload temperature trajectory"],
        horizontal=True,
        key="s2_data_source",
    )
    st.session_state["s2_use_uploaded"] = (temp_source == "Upload temperature trajectory")

    if st.session_state["s2_use_uploaded"]:
        with st.container(border=True):
            st.subheader("Upload Temperature Trajectory")

            col_info, col_dl = st.columns([3, 1])
            with col_info:
                st.markdown("""
**Expected CSV format**

| timestamp | temperature |
|---|---|
| 0 | 37.0 |
| 5 | 37.1 |
| 10 | 37.3 |

- **Column 1**: time / timestamp (numeric – minutes, seconds, or step index)
- **Column 2**: temperature reading (°C)
- Single batch trajectory – one file per batch
- Any column header names are accepted; the API auto-detects them
""")
            with col_dl:
                st.markdown("&nbsp;")
                tmpl = _api_get_bytes("step2/template")
                if tmpl:
                    st.download_button(
                        "Download Template",
                        data=tmpl,
                        file_name="step2_temperature_template.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            uploaded = st.file_uploader(
                "Choose temperature trajectory CSV",
                type=["csv"],
                key="s2_file_uploader",
                help="Max 10 MB. Two columns: time and temperature.",
            )

            if uploaded is not None:
                if st.button("Detect Columns", type="primary", key="s2_detect_btn"):
                    with st.spinner("Analysing file…"):
                        detection = _api_upload("step2/upload", uploaded)
                    if detection:
                        st.session_state["s2_detection"] = detection
                        st.session_state["s2_processed"] = None

            # ── Detection results ──────────────────────────────────────────
            if st.session_state["s2_detection"]:
                det = st.session_state["s2_detection"]
                st.divider()
                st.subheader("Detected Column Mapping")

                for w in det.get("warnings", []):
                    st.warning(w)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Time column",        det["time_column"])
                c2.metric("Temperature column", det["temp_column"])
                c3.metric("Data points",        det["n_rows"])
                c4.metric("Temp range (°C)",
                          f"{det['temp_range'][0]:.1f} – {det['temp_range'][1]:.1f}")

                col_a, col_b = st.columns(2)
                col_a.metric("Mean temperature", f"{det['temp_mean']:.2f} °C")
                col_b.metric("Temperature std (stability)", f"{det['temp_std']:.3f} °C")

                with st.expander("Data preview (first 10 rows)", expanded=True):
                    st.dataframe(pd.DataFrame(det["preview"]), use_container_width=True)

                # ── Confirmation form ──────────────────────────────────────
                st.subheader("Confirm Column Mapping")
                all_cols = det["all_columns"]

                with st.form("s2_confirm_form"):
                    time_col = st.selectbox(
                        "Time column", options=all_cols,
                        index=all_cols.index(det["time_column"]) if det["time_column"] in all_cols else 0,
                    )
                    temp_col = st.selectbox(
                        "Temperature column", options=all_cols,
                        index=all_cols.index(det["temp_column"]) if det["temp_column"] in all_cols else 1,
                    )
                    confirmed = st.form_submit_button("Confirm & Load Trajectory", type="primary")

                if confirmed:
                    if time_col == temp_col:
                        st.error("Time and temperature columns must be different.")
                    else:
                        payload = {"time_column": time_col, "temp_column": temp_col}
                        with st.spinner("Processing trajectory…"):
                            processed = _api_post_json("step2/confirm", payload)
                        if processed:
                            st.session_state["s2_processed"] = processed
                            stats = processed["stats"]
                            st.success(
                                f"Trajectory loaded: {processed['n_points']} points | "
                                f"Mean {stats['mean']:.2f} °C | "
                                f"Std {stats['std']:.3f} °C"
                            )

            if st.session_state["s2_processed"]:
                if st.button("Clear uploaded trajectory", key="s2_clear"):
                    st.session_state["s2_detection"] = None
                    st.session_state["s2_processed"] = None
                    st.rerun()

    # ── Resolve temperature data ───────────────────────────────────────────────
    if st.session_state["s2_use_uploaded"] and st.session_state["s2_processed"]:
        proc = st.session_state["s2_processed"]
        use_uploaded_temp = True
        data_note = (
            f"Single trajectory: **uploaded data** "
            f"({proc['n_points']} points, {proc['time_unit']})"
        )
    else:
        use_uploaded_temp = False
        data_note = "Showing historical trajectories from loaded data."

    st.info(data_note)

    tab_traj, tab_golden, tab_compare, tab_dtw, tab_phase = st.tabs([
        "Aligned Trajectories", "Golden Profile", "Upload vs Golden", "DTW Distances", "Phase Summary"
    ])

    with tab_traj:
        st.subheader("Aligned Historical Batch Trajectories")
        st.markdown(
            "All batches are stretched/compressed to a common 0–100% time axis so "
            "trajectories of different durations can be directly compared. "
            "**The golden band (amber shading) is your target operating corridor** — "
            "batches that stay inside it throughout their run tend to produce high yield. "
            "Batches that drift outside the band early are the ones worth investigating."
        )
        avail_cpps = [v for v in validated_cpps if v in golden_profile]
        # Fall back to all CPPs stored in the golden profile when validated_cpps
        # doesn't overlap (e.g. after a Step-1 CPP edit or a cross-session reload)
        if not avail_cpps:
            avail_cpps = [v for v in golden_profile.get("variables", []) if v in golden_profile]
        var_sel = st.selectbox(
            "CPP to display", avail_cpps if avail_cpps else [None],
            format_func=lambda v: CPP_LABELS.get(v, v) if v is not None else "— no CPPs available —",
            key="traj_var",
        )
        q_opts = [q for q in ["Golden", "Good", "Poor", "Unknown"]
                  if q in batch_df["quality"].unique()]
        quality_sel = st.multiselect("Quality tiers", q_opts,
                                     default=q_opts[:2], key="traj_q")
        if var_sel is None:
            st.warning(
                "No CPP trajectories are available. This can happen when the validated CPP list "
                "(Step 1) does not match the variables in the golden profile. "
                "Try re-uploading your data or resetting the CPP selection in Step 1."
            )
        else:
            fig = go.Figure()
            for bid in batch_df[batch_df["quality"].isin(quality_sel)]["batch_id"].tolist()[:40]:
                q = batch_df.loc[batch_df["batch_id"] == bid, "quality"].values[0]
                df_a = aligned[bid]
                if var_sel not in df_a.columns:
                    continue
                fig.add_trace(go.Scatter(
                    x=df_a["time_norm"] * 100, y=df_a[var_sel], mode="lines",
                    line=dict(color=QUALITY_COLORS.get(q, "#888888"), width=1),
                    opacity=0.4, showlegend=False,
                ))
            gp = golden_profile[var_sel]
            time_x = golden_profile["time_norm"] * 100
            fig.add_trace(go.Scatter(x=time_x, y=gp["upper"], mode="lines",
                                     line=dict(color="rgba(255,215,0,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=time_x, y=gp["lower"], mode="lines", fill="tonexty",
                                     fillcolor="rgba(255,215,0,0.25)",
                                     line=dict(color="rgba(255,215,0,0)"), name="Golden ±2σ"))
            fig.add_trace(go.Scatter(x=time_x, y=gp["mean"], mode="lines",
                                     line=dict(color="#FF8F00", width=3), name="Golden mean"))
            target_val = CPP_TARGETS.get(var_sel)
            if target_val is not None:
                fig.add_hline(y=target_val, line_dash="dot", line_color="grey",
                              annotation_text=f"Target = {target_val}")
            fig.update_layout(
                title=f"{CPP_LABELS.get(var_sel, var_sel)} – Aligned Trajectories with Golden Profile",
                xaxis_title="Normalised Batch Progress (%)", yaxis_title=CPP_LABELS.get(var_sel, var_sel),
                height=450, plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**How to read this chart:** Each thin line is one historical batch. "
            "Amber band = mean ± 2σ of golden (top-yield) batches — your target corridor. "
            "Orange line = golden mean trajectory. "
            "Dotted grey line = process setpoint. "
            "**Batches whose lines drift outside the amber band early in the run often finish as 'Poor' quality.** "
            "Use this to identify which phase of the batch is most prone to excursions."
        )
        with st.expander("What to look for — decision guide"):
            st.markdown(f"""
**Selecting batches to display:**
- Start with **Golden + Good** to see the 'normal' operating range
- Add **Poor** to see where failing batches diverge from the golden corridor

**Common patterns and what they mean:**
| Pattern | Likely cause | Action |
|---|---|---|
| Poor batches start inside band, drift out in first 30% | Early-stage instability | Tighten startup controls |
| Poor batches are consistently above/below band | Systematic setpoint offset | Recalibrate sensors or adjust setpoint |
| All batches scattered with no clear band | High process noise | Reduce variability source (equipment, raw material) |
| Band is very narrow | Tight, reproducible process | Good — maintain current controls |
""")


    with tab_golden:
        st.subheader("Golden Profile – All Validated CPPs")
        _gp_n   = golden_profile.get('n_golden', '?')
        _t_crit = golden_profile.get('t_crit', 2.0)
        _alpha  = golden_profile.get('alpha', 0.05)
        st.markdown(
            f"Built from **{_gp_n} golden batches** "
            f"(yield ≥ {batch_df['yield'].quantile(0.75):.1f}%). "
            f"Shaded band = **t-distribution {int((1-_alpha)*100)}% CI** combined with process spread "
            f"(t-crit = {_t_crit:.2f}, df = {_gp_n - 1}). "
            "With fewer golden batches the band widens automatically to reflect genuine uncertainty."
        )

        # ── Batch conformance scores ───────────────────────────────────────────
        if aligned:
            try:
                conformance_rows = []
                for bid, aln_df in aligned.items():
                    cs = compute_batch_conformance_score(aln_df, golden_profile, validated_cpps)
                    y_val = float(batch_df.loc[batch_df["batch_id"] == bid, "yield"].values[0]) \
                        if bid in batch_df["batch_id"].values else None
                    conformance_rows.append({
                        "Batch": bid,
                        "Overall Conformance (%)": round(cs["overall"], 1),
                        "Yield (%)": round(y_val, 1) if y_val is not None else None,
                        **{f"{CPP_LABELS.get(v,v)} conf.": round(cs["per_cpp"].get(v, 0), 1)
                           for v in validated_cpps[:4]},  # show top 4 CPPs inline
                    })
                conf_df = pd.DataFrame(conformance_rows).sort_values(
                    "Overall Conformance (%)", ascending=False
                )
                st.subheader("Batch Conformance to Golden Profile")
                st.caption(
                    "**Conformance = % of normalised-time points where a batch stays within the golden band.** "
                    "100% = perfectly golden at every time step. < 80% = investigate deviations. "
                    "Compare conformance against yield to validate that the golden profile is predictive."
                )
                # Colour-code: green ≥ 80, amber 60–80, red < 60
                def _colour_conf(val):
                    try:
                        v = float(val)
                        if v >= 80: return "background-color: #c8e6c9"
                        if v >= 60: return "background-color: #fff9c4"
                        return "background-color: #ffcdd2"
                    except Exception:
                        return ""
                st.dataframe(
                    conf_df.style.applymap(_colour_conf, subset=["Overall Conformance (%)"]),
                    use_container_width=True,
                )
                # Conformance vs Yield scatter + trendline
                _conf_merged = conf_df[["Overall Conformance (%)", "Yield (%)"]].dropna()
                if len(_conf_merged) >= 5:
                    from scipy.stats import pearsonr as _pr
                    _r, _p = _pr(_conf_merged["Overall Conformance (%)"], _conf_merged["Yield (%)"])
                    _pstr = f"p={'<0.001' if _p < 0.001 else f'{_p:.3f}'}"
                    if _p < 0.05:
                        st.success(
                            f"✅ **Conformance correlates with yield** (r = {_r:.2f}, {_pstr}). "
                            "Batches that stay within the golden band achieve higher yield — "
                            "validating that the golden profile is a meaningful reference."
                        )
                    else:
                        st.info(
                            f"Conformance–yield correlation: r = {_r:.2f}, {_pstr} (not significant). "
                            "This could mean the golden band is too wide, or that yield is driven "
                            "by factors not captured in the CPP trajectories."
                        )
                    # Scatter plot with trendline
                    _cx = _conf_merged["Overall Conformance (%)"].values
                    _cy = _conf_merged["Yield (%)"].values
                    _cm, _cb = np.polyfit(_cx, _cy, 1)
                    _cx_line = np.linspace(_cx.min(), _cx.max(), 50)
                    _fig_conf = go.Figure()
                    _fig_conf.add_trace(go.Scatter(
                        x=_cx, y=_cy, mode="markers",
                        marker=dict(size=10, color=_cy, colorscale="RdYlGn",
                                    showscale=True, colorbar=dict(title="Yield %", thickness=10)),
                        text=_conf_merged.index.astype(str) if hasattr(_conf_merged.index, 'astype') else None,
                        hovertemplate="Conformance: %{x:.1f}%<br>Yield: %{y:.2f}%<extra></extra>",
                        name="Batches",
                    ))
                    _fig_conf.add_trace(go.Scatter(
                        x=_cx_line, y=_cm * _cx_line + _cb,
                        mode="lines", line=dict(color="steelblue", width=2, dash="dot"),
                        name=f"Trend (r={_r:+.2f})", showlegend=True,
                    ))
                    _fig_conf.update_layout(
                        title=dict(
                            text=f"Conformance vs Yield — does staying in the golden band predict high yield?<br>"
                                 f"<sup>r={_r:+.2f}, {_pstr}</sup>",
                            font_size=13,
                        ),
                        xaxis_title="Overall Conformance to Golden Profile (%)",
                        yaxis_title="Final Yield (%)",
                        height=380, plot_bgcolor="white", paper_bgcolor="white",
                    )
                    st.plotly_chart(_fig_conf, use_container_width=True)
                    st.caption(
                        "Each dot = one historical batch. "
                        "Dots in the top-right = high conformance AND high yield (ideal). "
                        "Dots in the bottom-left = deviating batches with lower yield. "
                        "A strong upward trend confirms the golden profile is a valid process target."
                    )
            except Exception as _ce:
                st.caption(f"Conformance scores unavailable: {_ce}")
        valid_vars = [v for v in validated_cpps if v in golden_profile]
        if not valid_vars:
            valid_vars = [v for v in golden_profile.get("variables", []) if v in golden_profile]
        if not valid_vars:
            st.warning("No CPP profiles available to display. Please re-upload your data.")
        else:
            n_cols = 2
            n_rows = max(1, (len(valid_vars) + 1) // 2)
            fig = make_subplots(rows=n_rows, cols=n_cols,
                                subplot_titles=[CPP_LABELS.get(v, v) for v in valid_vars])
            time_x = golden_profile["time_norm"] * 100
            for idx, var in enumerate(valid_vars):
                r, c = divmod(idx, n_cols)
                gp = golden_profile[var]
                fig.add_trace(go.Scatter(x=time_x, y=gp["upper"], mode="lines",
                                         line=dict(color="rgba(255,215,0,0)"), showlegend=False),
                              row=r+1, col=c+1)
                fig.add_trace(go.Scatter(x=time_x, y=gp["lower"], mode="lines", fill="tonexty",
                                         fillcolor="rgba(255,215,0,0.3)",
                                         line=dict(color="rgba(255,215,0,0)"),
                                         name="±2σ", showlegend=(idx == 0)), row=r+1, col=c+1)
                fig.add_trace(go.Scatter(x=time_x, y=gp["mean"], mode="lines",
                                         line=dict(color="#FF8F00", width=2),
                                         name="Golden mean", showlegend=(idx == 0)), row=r+1, col=c+1)
            fig.update_layout(height=max(400, 220 * n_rows),
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            # ── Ideal CPP Range table ──────────────────────────────────────────
            st.divider()
            st.subheader("Ideal Operating Ranges (from Golden Batches)")
            st.markdown(
                "Derived from the golden batch trajectories. "
                "Stay within these bounds to maximise yield probability."
            )
            range_rows = []
            for var in valid_vars:
                gp_v = golden_profile[var]
                label = CPP_LABELS.get(var, var)
                n = len(gp_v["mean"])
                phases = {
                    "Startup (0–33%)":   slice(0, n // 3),
                    "Reaction (33–67%)": slice(n // 3, 2 * n // 3),
                    "Harvest (67–100%)": slice(2 * n // 3, n),
                }
                for phase_name, sl in phases.items():
                    range_rows.append({
                        "CPP": label,
                        "Phase": phase_name,
                        "Mean": round(float(np.mean(gp_v["mean"][sl])), 3),
                        "Lower (CI/σ band)": round(float(np.min(gp_v["lower"][sl])), 3),
                        "Upper (CI/σ band)": round(float(np.max(gp_v["upper"][sl])), 3),
                        "Std": round(float(np.mean(gp_v["std"][sl])), 3),
                    })

            range_df = pd.DataFrame(range_rows)
            st.dataframe(
                range_df.style.format({
                    "Mean": "{:.3f}", "Lower (CI/σ band)": "{:.3f}",
                    "Upper (CI/σ band)": "{:.3f}", "Std": "{:.3f}",
                }),
                use_container_width=True,
            )

            # Summary: overall range per CPP
            st.markdown("**Overall ideal range per CPP (all phases combined)**")
            overall_rows = []
            for var in valid_vars:
                gp_v = golden_profile[var]
                overall_rows.append({
                    "CPP": CPP_LABELS.get(var, var),
                    "Recommended Min": round(float(np.min(gp_v["lower"])), 3),
                    "Recommended Max": round(float(np.max(gp_v["upper"])), 3),
                    "Golden Mean":     round(float(np.mean(gp_v["mean"])), 3),
                    "Avg Variability (σ)": round(float(np.mean(gp_v["std"])), 3),
                })
            st.dataframe(
                pd.DataFrame(overall_rows).style.format({
                    "Recommended Min": "{:.3f}", "Recommended Max": "{:.3f}",
                    "Golden Mean": "{:.3f}", "Avg Variability (σ)": "{:.3f}",
                }),
                use_container_width=True,
            )

    with tab_compare:
        st.subheader("Uploaded Temperature vs Golden Profile")
        if not use_uploaded_temp:
            st.info(
                "Upload a temperature trajectory using the panel above to compare "
                "it against the Golden Profile."
            )
        else:
            proc = st.session_state["s2_processed"]
            t_up = proc["timestamps"]       # already 0–100 %
            T_up = proc["temperatures"]
            gp_t = golden_profile.get("temperature")
            time_x = golden_profile["time_norm"] * 100

            fig = go.Figure()
            if gp_t:
                fig.add_trace(go.Scatter(x=time_x, y=gp_t["upper"], mode="lines",
                                         line=dict(color="rgba(255,215,0,0)"), showlegend=False))
                fig.add_trace(go.Scatter(x=time_x, y=gp_t["lower"], mode="lines", fill="tonexty",
                                         fillcolor="rgba(255,215,0,0.3)",
                                         line=dict(color="rgba(255,215,0,0)"),
                                         name="Golden ±2σ band"))
                fig.add_trace(go.Scatter(x=time_x, y=gp_t["mean"], mode="lines",
                                         line=dict(color="#FF8F00", width=2.5),
                                         name="Golden mean"))

            fig.add_trace(go.Scatter(
                x=t_up, y=T_up, mode="lines+markers",
                line=dict(color="#1565C0", width=2.5),
                marker=dict(size=4), name="Uploaded trajectory",
            ))

            # Highlight out-of-band points
            if gp_t and len(t_up) > 0:
                t_norm_interp = np.array(t_up) / 100.0
                upper_interp = np.interp(t_norm_interp, golden_profile["time_norm"], gp_t["upper"])
                lower_interp = np.interp(t_norm_interp, golden_profile["time_norm"], gp_t["lower"])
                T_arr = np.array(T_up)
                outside = (T_arr > upper_interp) | (T_arr < lower_interp)
                if outside.any():
                    fig.add_trace(go.Scatter(
                        x=np.array(t_up)[outside], y=T_arr[outside],
                        mode="markers",
                        marker=dict(color="#E53935", size=9, symbol="x"),
                        name="Outside golden band",
                    ))
                    n_out = int(outside.sum())
                    pct_out = 100 * n_out / len(T_up)
                    st.warning(
                        f"{n_out} / {len(T_up)} points ({pct_out:.1f} %) "
                        "are **outside** the golden ±2σ band."
                    )
                else:
                    st.success("All temperature points are within the golden ±2σ band.")

            fig.add_hline(y=CPP_TARGETS.get("temperature", 37), line_dash="dot",
                          line_color="grey",
                          annotation_text=f"Target = {CPP_TARGETS.get('temperature', 37)} °C")
            fig.update_layout(
                title="Uploaded Temperature Trajectory vs Golden Profile",
                xaxis_title="Batch Progress (%)", yaxis_title="Temperature (°C)",
                height=480, plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            stats = proc["stats"]
            gp_mean_val = float(np.mean(gp_t["mean"])) if gp_t else None
            st.table(pd.DataFrame([
                {"Metric": "Mean temperature (°C)",
                 "Uploaded": f"{stats['mean']:.2f}",
                 "Golden mean": f"{gp_mean_val:.2f}" if gp_mean_val else "—"},
                {"Metric": "Temperature std (°C)",
                 "Uploaded": f"{stats['std']:.3f}",
                 "Golden mean": "reference"},
                {"Metric": "Min (°C)", "Uploaded": f"{stats['min']:.2f}", "Golden mean": "—"},
                {"Metric": "Max (°C)", "Uploaded": f"{stats['max']:.2f}", "Golden mean": "—"},
            ]))

    with tab_dtw:
        st.subheader("DTW Distance from Golden Profile")
        st.markdown(
            "**Dynamic Time Warping (DTW) distance** measures how far each batch's trajectory "
            "is from the golden mean trajectory, allowing for slight timing differences. "
            "A low DTW distance = the batch closely followed the golden path throughout its run. "
            "**Lower DTW → higher yield** is the expected pattern for a well-controlled process."
        )
        try:
            with st.spinner("Computing DTW distances…"):
                dtw_rows = []
                for bid in list(aligned.keys()):
                    dists = dtw_distance_from_golden(
                        aligned[bid], golden_profile,
                        [v for v in validated_cpps if v in golden_profile],
                    )
                    dists["batch_id"] = bid
                    dtw_rows.append(dists)
                dtw_df = pd.DataFrame(dtw_rows).set_index("batch_id")
                valid_cpp_cols = [v for v in validated_cpps if v in dtw_df.columns]
                dtw_df["total_dtw"] = dtw_df[valid_cpp_cols].sum(axis=1)
                dtw_df = dtw_df.join(batch_df.set_index("batch_id")[["yield", "quality"]])

            if batch_df["yield"].notna().any():
                fig_dtw = px.scatter(
                    dtw_df.reset_index(), x="total_dtw", y="yield",
                    color="quality", color_discrete_map=QUALITY_COLORS,
                    hover_data=["batch_id"],
                    title="Total DTW Distance vs Yield — lower distance = closer to golden path",
                    labels={"total_dtw": "Total DTW Distance", "yield": "Yield (%)"},
                    height=400,
                )
                fig_dtw.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_dtw, use_container_width=True)
                st.caption(
                    "**How to read:** Each dot is one batch. "
                    "Bottom-left = followed golden path AND achieved high yield (reference batches). "
                    "Top-right = deviated most from golden path AND had lower yield. "
                    "A clear downward trend confirms that following the golden trajectory predicts success."
                )
                try:
                    dtw_corr = float(dtw_df[["total_dtw", "yield"]].corr().loc["total_dtw", "yield"])
                    if dtw_corr < -0.5:
                        st.success(f"Strong negative DTW–yield correlation (r = {dtw_corr:.2f}): "
                                   "batches that follow the golden trajectory consistently achieve higher yield. "
                                   "The golden profile is a reliable production target.")
                    elif dtw_corr < -0.2:
                        st.info(f"Moderate DTW–yield correlation (r = {dtw_corr:.2f}): "
                                "trajectory similarity is a useful but not the only yield predictor. "
                                "Consider monitoring additional CPPs.")
                    else:
                        st.warning(f"Weak DTW–yield correlation (r = {dtw_corr:.2f}): "
                                   "the current CPP set may not fully explain yield variation. "
                                   "Review CPP selection in Step 1.")
                except Exception:
                    pass
            else:
                st.dataframe(dtw_df[["total_dtw"] + valid_cpp_cols].sort_values("total_dtw"),
                             use_container_width=True)
                st.caption("Batches sorted by total DTW distance — lower = closer to golden trajectory.")
        except Exception as exc:
            st.error(f"DTW computation failed: {exc}")

    with tab_phase:
        st.subheader("Phase-Based Summary Statistics")
        st.markdown(
            "Each batch is split into three phases (Startup 0–33%, Reaction 33–67%, Harvest 67–100%). "
            "This reveals **which phase of the batch is the most critical control window** — "
            "i.e. where golden batches consistently operate differently from poor batches."
        )
        from analysis.trajectory_analysis import compute_phase_summary
        try:
            with st.spinner("Computing…"):
                phase_vars = [v for v in validated_cpps
                              if v in aligned[list(aligned.keys())[0]].columns]
                if not phase_vars:
                    st.warning("No validated CPPs found in aligned data.")
                    st.stop()
                phase_df = compute_phase_summary(aligned, variables=phase_vars)
                phase_df = phase_df.join(batch_df.set_index("batch_id")[["yield", "quality"]],
                                         on="batch_id")

            phase_pick = st.selectbox("Phase", phase_df["phase"].unique())
            var_phase = st.selectbox("CPP", phase_vars,
                                     format_func=lambda v: CPP_LABELS.get(v, v), key="phase_var")
            col_mean = f"{var_phase}_mean"
            if col_mean in phase_df.columns:
                ph_sub = phase_df[phase_df["phase"] == phase_pick]
                golden_med = ph_sub[ph_sub["quality"] == "Golden"][col_mean].median() if "Golden" in ph_sub["quality"].values else None
                poor_med   = ph_sub[ph_sub["quality"] == "Poor"][col_mean].median()   if "Poor"   in ph_sub["quality"].values else None
                if golden_med is not None and poor_med is not None:
                    diff = golden_med - poor_med
                    direction = "higher" if diff > 0 else "lower"
                    st.info(
                        f"During **{phase_pick}**, golden batches have a **{direction}** "
                        f"mean {CPP_LABELS.get(var_phase, var_phase)} "
                        f"(median {golden_med:.2f}) vs poor batches "
                        f"(median {poor_med:.2f}, Δ = {abs(diff):.2f}). "
                        "This phase is a key control window for this parameter."
                    )
                fig_box = px.box(
                    ph_sub, x="quality", y=col_mean, color="quality",
                    color_discrete_map=QUALITY_COLORS, points="all",
                    title=f"{CPP_LABELS.get(var_phase, var_phase)} mean during {phase_pick} — by quality tier",
                    height=380,
                )
                fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption(
                    "**Box plot guide:** Box = 25th–75th percentile range. "
                    "Line inside = median. Dots = individual batches. "
                    "**Non-overlapping boxes between Golden and Poor tiers indicate "
                    f"that {CPP_LABELS.get(var_phase, var_phase)} during {phase_pick} "
                    "is a lever for improving yield.** "
                    "Try different phase + CPP combinations to find the most discriminating window."
                )
        except Exception as exc:
            st.error(f"Phase summary failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Step 3 – Batch Monitoring
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Step 3 – Batch Monitoring":
    st.title("Step 3 – Real-Time Batch Monitoring (MSPC + PLS)")
    st.markdown(
        "MSPC detects multivariate deviations (T² and SPE). "
        "PLS predicts final yield from partial batch data. "
        "Move the slider to simulate batch progress."
    )

    if not st.session_state["data_loaded"]:
        st.warning("No data loaded. Please upload historical batch data on the Overview page.")
        st.stop()

    if st.session_state["data_format"] == "wide":
        st.info(
            "Step 3 requires long-format time-series data (one row per time step per batch). "
            "Go to the Overview page and upload a long-format CSV to access this step."
        )
        st.stop()

    # Pull from session state
    batch_df          = st.session_state["batch_df"]
    aligned           = st.session_state["aligned"]
    golden_profile    = st.session_state["golden_profile"]
    mspc_model        = st.session_state["mspc_model"]
    pls_model         = st.session_state["pls_model"]
    batch_ids_ordered = st.session_state["batch_ids_ordered"]
    active_cpp_vars   = st.session_state["active_cpp_vars"]

    has_yield = batch_df["yield"].notna().all()

    if not has_yield:
        st.warning(
            "No yield data in the loaded dataset. "
            "PLS yield prediction and quality colour-coding are unavailable. "
            "MSPC monitoring is still available."
        )

    tab_mspc, tab_pls, tab_compare = st.tabs([
        "MSPC Monitoring", "PLS Yield Prediction", "Historical Comparison"
    ])

    st.sidebar.divider()
    st.sidebar.subheader("Select Batch to Monitor")
    selected_bid = st.sidebar.selectbox(
        "Batch",
        batch_ids_ordered,
        help="Pick any historical batch to replay through MSPC and PLS.",
    )

    from analysis.trajectory_analysis import future_impute_row

    test_aligned  = aligned[selected_bid]
    test_batch_id = selected_bid

    with st.spinner("Running monitoring…"):
        mspc_traj = mspc_model.monitor_realtime(test_aligned, variables=active_cpp_vars)
        pls_traj  = pls_model.predict_realtime(test_aligned, variables=active_cpp_vars) if has_yield else None

    with tab_mspc:
        st.subheader(f"MSPC – {test_batch_id}")
        st.markdown(
            "MSPC watches **all CPPs simultaneously** and fires an alarm the moment their combined "
            "behaviour falls outside what was seen in historical good batches. "
            "**T² (Hotelling)** detects whether the overall process state has shifted to an unusual region. "
            "**SPE (Q statistic)** detects whether new, unexpected patterns have appeared that the model "
            "has never seen before. Use the slider to replay the batch and see when alarms would have fired."
        )
        progress_pct = st.slider("Batch progress (%)", 5, 100, 100, 5, key="mspc_slider")
        t_show = max(1, int(progress_pct * N_ALIGNED_POINTS / 100))
        visible = mspc_traj[mspc_traj["time_step"] <= t_show]
        if visible.empty:
            st.info("No monitoring data available for the selected batch progress.")
            st.stop()
        last = visible.iloc[-1]

        if last["T2_breach_99"] or last["SPE_breach_99"]:
            alerts = []
            if last["T2_breach_99"]:  alerts.append("T² > 99% limit")
            if last["SPE_breach_99"]: alerts.append("SPE > 99% limit")
            st.error(
                f"🚨 ALARM at {progress_pct}% batch progress: " + " | ".join(alerts) +
                " — Immediate investigation required. Check the Contribution Plot below "
                "to identify which CPP is causing the deviation."
            )
        elif last["T2_breach_95"] or last["SPE_breach_95"]:
            st.warning(
                f"⚠️ WARNING at {progress_pct}% batch progress: Approaching control limit (95%). "
                "Monitor closely — if the trend continues, a full alarm is imminent. "
                "Check the Contribution Plot to identify the leading CPP."
            )
        else:
            st.success(
                f"✅ {progress_pct}% batch progress: Batch IN CONTROL — "
                "all CPPs operating within historical norms."
            )

        # ── Run rules: sustained deviation (3+ consecutive 95% breaches) ──────
        _RUN_N = 3  # consecutive breach threshold
        _t2_run = int(visible["T2_breach_95"].astype(int).rolling(_RUN_N).sum().max() or 0)
        _spe_run = int(visible["SPE_breach_95"].astype(int).rolling(_RUN_N).sum().max() or 0)
        if _t2_run >= _RUN_N or _spe_run >= _RUN_N:
            _run_msgs = []
            if _t2_run >= _RUN_N:
                _run_msgs.append(f"**T² sustained drift** ({_t2_run} consecutive steps above 95% limit)")
            if _spe_run >= _RUN_N:
                _run_msgs.append(f"**SPE sustained drift** ({_spe_run} consecutive steps above 95% limit)")
            st.warning(
                "📈 **Run Rule triggered — Sustained Deviation:** " + " | ".join(_run_msgs) + ". "
                "Even if the latest point is below the 99% alarm, a persistent trend above 95% "
                "indicates the process has drifted from its baseline. "
                "Investigate the root cause before the batch crosses into alarm territory."
            )

        fig_t2 = go.Figure()
        fig_t2.add_trace(go.Scatter(
            x=visible["time_step"], y=visible["T2"], mode="lines+markers", name="T²",
            line=dict(color="#1976D2", width=2),
            marker=dict(color=["#E53935" if b else "#1976D2"
                               for b in visible["T2_breach_95"]], size=5),
        ))
        fig_t2.add_hline(y=mspc_model.t2_limit_95, line_dash="dash",
                         line_color="orange", annotation_text="95% warning limit")
        fig_t2.add_hline(y=mspc_model.t2_limit_99, line_dash="dash",
                         line_color="red", annotation_text="99% action limit")
        fig_t2.update_layout(title="Hotelling's T² — Overall Process State Distance from Normal",
                             xaxis_title="Time Step",
                             yaxis_title="T²", height=300,
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_t2, use_container_width=True)
        st.caption(
            "**T² measures how far the current combination of all CPPs is from the 'normal' region** "
            "learned from historical good batches. "
            "Orange line = 95% warning limit (5% chance of false alarm). "
            "Red line = 99% action limit (1% chance of false alarm). "
            "Red dots = time steps where the 95% limit was exceeded."
        )

        fig_spe = go.Figure()
        fig_spe.add_trace(go.Scatter(
            x=visible["time_step"], y=visible["SPE"], mode="lines+markers", name="SPE",
            line=dict(color="#7B1FA2", width=2),
            marker=dict(color=["#E53935" if b else "#7B1FA2"
                               for b in visible["SPE_breach_95"]], size=5),
        ))
        fig_spe.add_hline(y=mspc_model.spe_limit_95, line_dash="dash",
                          line_color="orange", annotation_text="95% warning limit")
        fig_spe.add_hline(y=mspc_model.spe_limit_99, line_dash="dash",
                          line_color="red", annotation_text="99% action limit")
        fig_spe.update_layout(title="SPE (Q Statistic) — Unexplained Process Variation",
                              xaxis_title="Time Step",
                              yaxis_title="SPE", height=300,
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_spe, use_container_width=True)
        st.caption(
            "**SPE (Squared Prediction Error) measures process variation that the model cannot explain** — "
            "i.e. something new or unexpected is happening. "
            "High T² but normal SPE = process has shifted but in a known direction. "
            "High SPE = genuinely novel fault pattern. "
            "**Both alarms together = highest urgency.**"
        )

        with st.expander("T² vs SPE — what each alarm means and what to do"):
            st.markdown("""
| T² | SPE | Interpretation | Action |
|---|---|---|---|
| Normal | Normal | ✅ Process in control | Continue monitoring |
| **High** | Normal | Process has shifted in a known direction — likely a setpoint drift | Check CPP setpoints; consult Contribution Plot |
| Normal | **High** | Unexpected new pattern — possible sensor fault, raw material change, or equipment issue | Inspect equipment; check raw material lot |
| **High** | **High** | Severe multivariate deviation — multiple CPPs affected simultaneously | Escalate immediately; consider batch hold |
""")

        st.subheader(f"Contribution Plot at {progress_pct}% — Which CPP is Causing the Deviation?")
        st.markdown(
            "This chart breaks down the SPE alarm into individual CPP contributions. "
            "**The red bar is the top contributor** — the parameter most responsible for the current deviation. "
            "The chart below then shows where that CPP diverged from the golden profile."
        )
        row = future_impute_row(test_aligned, t_show,
                                mspc_model.X_train_mean_unscaled,
                                active_cpp_vars, N_ALIGNED_POINTS)
        contribs = mspc_model.spe_contributions(row, active_cpp_vars, N_ALIGNED_POINTS)
        fig_cb = go.Figure(go.Bar(
            x=contribs.values,
            y=[CPP_LABELS.get(v, v) for v in contribs.index],
            orientation="h",
            marker_color=["#E53935" if i == 0 else "#90A4AE" for i in range(len(contribs))],
            text=[f"{v:.1f}%" for v in contribs.values], textposition="outside",
        ))
        fig_cb.update_layout(title="SPE Contribution by CPP (%) — red bar = investigate this parameter first",
                             xaxis_title="Contribution (%)",
                             yaxis=dict(autorange="reversed"),
                             height=320, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cb, use_container_width=True)

        # Determine top contributing CPP (fallback to first CPP if contribs is empty)
        top_cpp = (
            contribs.index[0] if len(contribs) > 0
            else (active_cpp_vars[0] if active_cpp_vars else None)
        )
        top_cpp_label = CPP_LABELS.get(top_cpp, top_cpp) if top_cpp else "Unknown"
        if len(contribs) > 0:
            top_contrib_pct = float(contribs.iloc[0])
            st.info(
                f"**{top_cpp_label}** accounts for {top_contrib_pct:.1f}% of the current SPE deviation. "
                f"The chart below shows how this batch's {top_cpp_label} compares to the golden profile — "
                "look for the point where the red line exits the amber band."
            )
        else:
            st.info("No SPE contribution data available for this batch.")

        gp_top = golden_profile.get(top_cpp, {}) if top_cpp else {}
        time_x = golden_profile["time_norm"] * 100
        fig_top = go.Figure()
        if gp_top:
            fig_top.add_trace(go.Scatter(x=time_x, y=gp_top["upper"], mode="lines",
                                         line=dict(color="rgba(255,215,0,0)"), showlegend=False))
            fig_top.add_trace(go.Scatter(x=time_x, y=gp_top["lower"], mode="lines", fill="tonexty",
                                         fillcolor="rgba(255,215,0,0.25)",
                                         line=dict(color="rgba(255,215,0,0)"), name="Golden ±2σ"))
            fig_top.add_trace(go.Scatter(x=time_x, y=gp_top["mean"], mode="lines",
                                         line=dict(color="#FF8F00", width=2), name="Golden mean"))
        if top_cpp and top_cpp in test_aligned.columns:
            fig_top.add_trace(go.Scatter(
                x=test_aligned["time_norm"].values[:t_show] * 100,
                y=test_aligned[top_cpp].values[:t_show],
                mode="lines", line=dict(color="#E53935", width=2.5), name="Selected batch",
            ))
        fig_top.update_layout(
            title=f"{top_cpp_label}: {test_batch_id} trajectory vs Golden Profile",
            xaxis_title="Batch Progress (%)", yaxis_title=top_cpp_label,
            height=360, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.caption(
            f"Red line = selected batch's {top_cpp_label} up to {progress_pct}% progress. "
            "Amber band = golden ±2σ corridor. "
            "**When and how far the red line exits the amber band shows the root cause and severity of the deviation.** "
            "Use the slider above to step back in time and find the earliest point of excursion."
        )

    with tab_pls:
        st.subheader("PLS – Yield Prediction as Batch Progresses")
        st.markdown(
            "PLS (Partial Least Squares) learns the relationship between **how a batch behaves "
            "during its run** and the **final yield**. "
            "This enables early yield forecasting — you can see whether a batch is heading for "
            "golden territory or not, well before it finishes. "
            "**Use the slider to simulate what the model would have predicted at any point in the batch.**"
        )

        # ── LOO-CV model reliability ───────────────────────────────────────────
        if has_yield and pls_model is not None:
            try:
                with st.spinner("Computing leave-one-out cross-validation R²…"):
                    loocv = pls_model.loocv_r2()
                r2_loo   = loocv.get("r2_loocv")
                r2_train = loocv.get("r2_train")
                rmse_loo = loocv.get("rmse_loocv")
                if r2_loo is not None:
                    if r2_loo >= 0.70:
                        st.success(
                            f"✅ **PLS model is reliable** — LOO-CV R² = **{r2_loo:.2f}** "
                            f"(training R² = {r2_train:.2f}, RMSE = {rmse_loo:.1f}%). "
                            "The model generalises well to unseen batches. "
                            "Yield predictions are trustworthy."
                        )
                    elif r2_loo >= 0.40:
                        st.warning(
                            f"⚠️ **Moderate model reliability** — LOO-CV R² = **{r2_loo:.2f}** "
                            f"(training R² = {r2_train:.2f}, RMSE = {rmse_loo:.1f}%). "
                            "The model has some predictive power but is partially overfitting. "
                            "Use predictions as directional indicators, not precise forecasts. "
                            "More batches will improve reliability."
                        )
                    else:
                        st.error(
                            f"🔴 **Low model reliability** — LOO-CV R² = **{r2_loo:.2f}** "
                            f"(training R² = {r2_train:.2f}). "
                            "The model is memorising training data and does not generalise. "
                            "Yield predictions may be unreliable. "
                            f"Gap: training R² − LOO R² = {r2_train - r2_loo:.2f} (overfitting measure). "
                            "Do not use predictions for operational decisions without more historical batches."
                        )
                    with st.expander("What is LOO-CV R²?"):
                        st.markdown("""
**Leave-One-Out Cross-Validation R²** is the honest measure of model quality.

| Metric | What it measures |
|---|---|
| **Training R²** | How well the model fits the data it was trained on — always optimistic, can be misleadingly high |
| **LOO-CV R²** | How well the model predicts a batch it has **never seen** — the real test of generalisation |

**Interpreting LOO-CV R²:**
- ≥ 0.70 → Reliable — use predictions with confidence
- 0.40–0.70 → Moderate — use as directional indicator
- < 0.40 → Overfit — collect more batches before trusting predictions

A large gap between training R² and LOO-CV R² means the model has memorised the training data.
""")
                    # ── LOO-CV Actual vs Predicted scatter ─────────────────────
                    _loo_pairs = loocv.get("predictions", [])
                    if len(_loo_pairs) >= 3:
                        _loo_actuals, _loo_preds = zip(*_loo_pairs)
                    else:
                        _loo_actuals, _loo_preds = None, None
                    if _loo_preds is not None and len(_loo_preds) >= 3:
                        _la = np.array(_loo_actuals)
                        _lp = np.array(_loo_preds)
                        _all_vals = np.concatenate([_la, _lp])
                        _vmin, _vmax = float(_all_vals.min()), float(_all_vals.max())
                        _fig_loo = go.Figure()
                        _fig_loo.add_trace(go.Scatter(
                            x=_la, y=_lp, mode="markers",
                            marker=dict(size=9, color=_la, colorscale="RdYlGn",
                                        showscale=True, colorbar=dict(title="Actual Yield", thickness=10)),
                            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                            name="Batches",
                        ))
                        _fig_loo.add_trace(go.Scatter(
                            x=[_vmin, _vmax], y=[_vmin, _vmax],
                            mode="lines", line=dict(color="grey", dash="dot", width=1.5),
                            name="Perfect prediction", showlegend=True,
                        ))
                        _fig_loo.update_layout(
                            title=dict(
                                text=f"LOO-CV: Actual vs Predicted Yield  (R²={r2_loo:.2f})",
                                font_size=14,
                            ),
                            xaxis_title="Actual Yield", yaxis_title="LOO-CV Predicted Yield",
                            height=380, plot_bgcolor="white", paper_bgcolor="white",
                        )
                        st.plotly_chart(_fig_loo, use_container_width=True)
                        st.caption(
                            "Each dot = one historical batch left out of training. "
                            "Dots on the grey diagonal = perfect prediction. "
                            "Dots close to the diagonal confirm the model generalises. "
                            "Outliers far from the diagonal indicate batches the model struggles to predict."
                        )
            except Exception:
                pass

        if not has_yield or pls_traj is None:
            st.info("PLS yield prediction requires yield data in the loaded dataset.")
        else:
            progress_pls = st.slider("Batch progress (%)", 5, 100, 60, 5, key="pls_slider")
            golden_threshold = batch_df["yield"].quantile(0.75)
            fig_yield = go.Figure()
            fig_yield.add_trace(go.Scatter(
                x=pls_traj["time_step"], y=pls_traj["predicted_yield"],
                mode="lines", name="Predicted Yield", line=dict(color="#2E7D32", width=2.5),
            ))
            fig_yield.add_vline(x=int(progress_pls * N_ALIGNED_POINTS / 100),
                                line_dash="dash", line_color="black",
                                annotation_text=f"Now ({progress_pls}%)")
            fig_yield.add_hline(y=golden_threshold, line_dash="dot", line_color="gold",
                                annotation_text="Golden threshold")
            fig_yield.update_layout(
                title="PLS Yield Prediction — how the forecast evolves as more batch data arrives",
                xaxis_title="Time Step", yaxis_title="Predicted Yield (%)",
                height=380, plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_yield, use_container_width=True)
            st.caption(
                "Green line = predicted final yield at each point in the batch. "
                "Gold dashed line = golden threshold (top 25% of historical yields). "
                "Vertical dashed line = current batch progress (from slider). "
                "**If the green line is above the gold threshold, the batch is on track for golden yield.** "
                "Watch for the prediction stabilising — early in the batch it is noisy, "
                "later predictions are more reliable as more data is accumulated."
            )

            t_pls = max(1, int(progress_pls * N_ALIGNED_POINTS / 100))
            pred_row = pls_traj.loc[pls_traj["time_step"] == t_pls, "predicted_yield"]
            if len(pred_row):
                pred_val = float(pred_row.values[0])
                delta = pred_val - golden_threshold
                st.metric(
                    f"Predicted Yield at {progress_pls}% batch progress",
                    f"{pred_val:.1f} %",
                    delta=f"{delta:+.1f}% vs golden threshold ({golden_threshold:.1f}%)",
                    delta_color="normal",
                )
                if delta >= 0:
                    st.success(
                        f"At {progress_pls}% progress, the model predicts **{pred_val:.1f}%** yield — "
                        f"**{delta:.1f}% above the golden threshold**. "
                        "If process conditions remain stable, this batch should be a golden batch."
                    )
                else:
                    st.warning(
                        f"At {progress_pls}% progress, the model predicts **{pred_val:.1f}%** yield — "
                        f"**{abs(delta):.1f}% below the golden threshold**. "
                        "Consider reviewing MSPC alarms and the contributing CPPs to identify corrective actions."
                    )

            st.subheader("VIP Scores – Which CPPs Drive Yield?")
            st.markdown(
                "**VIP (Variable Importance in Projection)** scores quantify how much each CPP "
                "contributes to the PLS yield prediction model. "
                "VIP > 1.0 (red bars) = this parameter is above-average in its influence on yield — "
                "these are the highest-leverage parameters to tighten control on."
            )
            with st.spinner("Computing VIP scores…"):
                vip_full, vip_agg = pls_model.compute_vip_scores(active_cpp_vars, N_ALIGNED_POINTS)
            fig_vip = go.Figure(go.Bar(
                x=vip_agg["agg_vip"],
                y=[CPP_LABELS.get(v, v) for v in vip_agg["variable"]],
                orientation="h",
                marker_color=["#E53935" if v > 1.0 else "#90A4AE" for v in vip_agg["agg_vip"]],
                text=[f"{v:.2f}" for v in vip_agg["agg_vip"]], textposition="outside",
            ))
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black",
                              annotation_text="VIP = 1.0 significance threshold")
            fig_vip.update_layout(
                title="VIP Scores per CPP — red bars (VIP > 1.0) are significant yield drivers",
                xaxis_title="Aggregated VIP", yaxis=dict(autorange="reversed"),
                height=360, plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_vip, use_container_width=True)
            st.caption(
                "**VIP interpretation:** "
                "Red bars (VIP > 1.0) = parameters whose trajectory history is actively used by "
                "the model to discriminate high-yield from low-yield batches. "
                "Grey bars (VIP < 1.0) = minor contributors. "
                "**These VIP scores validate and complement the CPP rankings from Step 1** — "
                "agreement across both methods gives high confidence in your CPP selection."
            )

            # ── VIP Phase Heatmap ──────────────────────────────────────────────
            if vip_full is not None and not vip_full.empty and len(active_cpp_vars) >= 1:
                st.markdown("#### VIP Importance by Process Phase")
                st.markdown(
                    "The heatmap shows **when** during the batch each CPP matters most. "
                    "Darker red = higher VIP = more influence on yield in that phase. "
                    "This reveals whether a CPP is critical in startup, the reaction core, "
                    "or late-stage harvest — and tells you *exactly when* to tighten control."
                )
                J = N_ALIGNED_POINTS
                phase_boundaries = {
                    "Startup\n(0–33%)":   (0,      J // 3),
                    "Reaction\n(33–67%)": (J // 3, 2 * J // 3),
                    "Harvest\n(67–100%)": (2 * J // 3, J),
                }
                _heat_data = []
                for var in active_cpp_vars:
                    row = {}
                    for k, (_ph, (_s, _e)) in enumerate(phase_boundaries.items()):
                        # Pull VIP scores for this var's time slice
                        _start_idx = active_cpp_vars.index(var) * J + _s
                        _end_idx   = active_cpp_vars.index(var) * J + _e
                        _slice_vip = vip_full["vip_score"].iloc[_start_idx:_end_idx]
                        row[_ph] = float(np.sqrt(np.sum(_slice_vip.values ** 2)))
                    _heat_data.append({"variable": var, **row})
                _heat_df = pd.DataFrame(_heat_data).set_index("variable")
                _phases = list(phase_boundaries.keys())
                _cpp_labels_list = [CPP_LABELS.get(v, v) for v in _heat_df.index.tolist()]
                _fig_heat = go.Figure(go.Heatmap(
                    z=_heat_df.values,
                    x=_phases,
                    y=_cpp_labels_list,
                    colorscale="Reds",
                    text=np.round(_heat_df.values, 2).astype(str),
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(title="Agg VIP"),
                    hovertemplate="CPP: %{y}<br>Phase: %{x}<br>VIP: %{z:.2f}<extra></extra>",
                ))
                _fig_heat.update_layout(
                    title="VIP by CPP × Phase — where in the batch does each parameter matter most?",
                    xaxis_title="Process Phase",
                    yaxis_title="CPP",
                    height=max(300, 40 * len(active_cpp_vars) + 100),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(_fig_heat, use_container_width=True)
                st.caption(
                    "**How to use this:** Identify CPPs with high VIP in the Startup phase — "
                    "these need tight control from the very beginning. "
                    "High Harvest-phase VIP means the end of batch is the critical window. "
                    "CPPs with uniformly high VIP across all phases require sustained tight control."
                )

    with tab_compare:
        st.subheader("All Historical Batches – T² vs SPE")
        st.markdown(
            "This chart plots every historical batch as a single point based on its overall T² and SPE values. "
            "It provides a **fleet-level view** of process health — how normal or abnormal each batch was "
            "compared to the process baseline. "
            "**Golden batches should cluster in the bottom-left (low T², low SPE).** "
            "Use this to identify chronic outlier batches and understand whether your process control is consistent."
        )
        with st.spinner("Computing T² and SPE for all batches…"):
            hist_stats = []
            for bid in batch_ids_ordered:
                X_single = np.concatenate(
                    [aligned[bid][v].values for v in active_cpp_vars]
                ).reshape(1, -1)
                T2, SPE = mspc_model.compute_stats(X_single)
                _row = batch_df.loc[batch_df["batch_id"] == bid]
                y_val = float(_row["yield"].values[0]) if not _row.empty else np.nan
                q_val = _row["quality"].values[0] if not _row.empty else "Unknown"
                hist_stats.append({"batch_id": bid, "T2": T2, "SPE": SPE,
                                   "yield": y_val, "quality": q_val})
            hist_df = pd.DataFrame(hist_stats)

        fig_sc = go.Figure()
        for quality in hist_df["quality"].unique():
            sub = hist_df[hist_df["quality"] == quality]
            hover_text = sub["batch_id"] + (
                "<br>Yield: " + sub["yield"].round(1).astype(str) + "%"
                if has_yield else ""
            )
            fig_sc.add_trace(go.Scatter(
                x=sub["T2"], y=sub["SPE"], mode="markers",
                marker=dict(color=QUALITY_COLORS.get(quality, "#888888"), size=8, opacity=0.8),
                name=quality,
                text=hover_text,
                hoverinfo="text",
            ))
        fig_sc.add_vline(x=mspc_model.t2_limit_95, line_dash="dash",
                         line_color="orange", annotation_text="T² 95%")
        fig_sc.add_hline(y=mspc_model.spe_limit_95, line_dash="dash",
                         line_color="orange", annotation_text="SPE 95%")
        fig_sc.add_vline(x=mspc_model.t2_limit_99, line_dash="dash",
                         line_color="red", annotation_text="T² 99%")
        fig_sc.add_hline(y=mspc_model.spe_limit_99, line_dash="dash",
                         line_color="red", annotation_text="SPE 99%")
        fig_sc.update_layout(
            title="T² vs SPE – All Historical Batches (hover for batch ID and yield)",
            xaxis_title="T² (Overall Process Deviation)", yaxis_title="SPE (Unexplained Variation)",
            height=500, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "**Four quadrants — what each means:**  "
            "**Bottom-left (low T², low SPE):** Normal operation — these batches are your process baseline.  "
            "**Top-right (high T², high SPE):** Severe deviations — these batches had significant abnormal events.  "
            "**Top-left (low T², high SPE):** Novel, unexpected patterns — possible equipment or raw material issue.  "
            "**Bottom-right (high T², low SPE):** Known-direction shift — likely a setpoint drift.  "
            "Hover over any dot to see the batch ID and yield. "
            "**Golden batches should cluster in the bottom-left; if they do not, reconsider the CPP set.**"
        )
        with st.expander("Control limits reference"):
            st.dataframe(pd.DataFrame({
                "Statistic": ["T² (Hotelling) — Overall Deviation", "SPE (Q) — Unexplained Variation"],
                "95% Warning Limit": [f"{mspc_model.t2_limit_95:.2f}", f"{mspc_model.spe_limit_95:.2f}"],
                "99% Action Limit": [f"{mspc_model.t2_limit_99:.2f}", f"{mspc_model.spe_limit_99:.2f}"],
                "Meaning": [
                    "Derived from 95th/99th percentile of training batch T² values",
                    "Derived from 95th/99th percentile of training batch SPE values",
                ],
            }), use_container_width=True)
            st.markdown(
                "Limits are set **empirically** from the historical batch distribution "
                "(not from theoretical F/chi² distributions), making them directly interpretable: "
                "the 95% limit means 5% of normal batches will exceed it by random chance."
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Recommendations":
    from scipy import stats as scipy_stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    st.title("Recommendations – Ideal Operating Conditions for Target Yield")
    st.markdown(
        "This page uses four complementary data science techniques to translate your target yield "
        "into **specific, actionable CPP operating ranges**. "
        "Set the target yield in Step 1 (or use the input below), then review the three sections: "
        "**CPP Operating Windows**, **Temperature Trajectory**, and **Action Plan**."
    )

    if not st.session_state["data_loaded"]:
        st.warning("No data loaded. Please upload historical batch data on the Overview page.")
        st.stop()

    # ── Pull state ─────────────────────────────────────────────────────────────
    batch_df           = st.session_state["batch_df"]
    yields_s           = st.session_state["yields"]
    features_s         = st.session_state["features"]
    aligned            = st.session_state["aligned"]
    golden_profile     = st.session_state["golden_profile"]
    pls_model          = st.session_state["pls_model"]
    active_cpp_vars    = st.session_state["active_cpp_vars"] or []
    validated_cpps     = st.session_state["validated_cpps"] or []
    batch_sum_features = st.session_state["batch_summary_features"]

    # Resolve best feature set: batch summary > time-series features
    if batch_sum_features is not None:
        analysis_features = batch_sum_features.copy()
        feat_source = "batch summary (File ①)"
    elif features_s is not None:
        analysis_features = features_s.copy()
        feat_source = "time-series derived features"
    else:
        st.warning("No CPP feature data available. Load data with CPP columns.")
        st.stop()

    if yields_s is None:
        st.warning("No yield data available. Reload data with a yield column.")
        st.stop()

    # Align features and yields
    common_idx = analysis_features.index.intersection(yields_s.index)
    if len(common_idx) < 5:
        st.error("Not enough matching batches between features and yield (need ≥ 5).")
        st.stop()
    analysis_features = analysis_features.loc[common_idx].apply(pd.to_numeric, errors="coerce")
    analysis_features.dropna(axis=1, how="all", inplace=True)
    analysis_features.fillna(analysis_features.mean(), inplace=True)
    analysis_yields = yields_s.loc[common_idx].dropna()
    analysis_features = analysis_features.loc[analysis_yields.index]

    # ── Target yield sidebar input ─────────────────────────────────────────────
    _y_min = float(analysis_yields.min())
    _y_max = float(analysis_yields.max())
    _y_p75 = float(analysis_yields.quantile(0.75))
    _saved  = st.session_state.get("target_yield")
    _default_target = _saved if (_saved and 1.0 <= _saved <= 100.0) else _y_p75
    _default_target = max(1.0, min(100.0, _default_target))

    st.sidebar.divider()
    st.sidebar.subheader("Target Yield")
    target_yield = st.sidebar.number_input(
        "Target Yield (%)",
        min_value=1.0,
        max_value=100.0,
        value=float(round(_default_target, 1)),
        step=0.5,
        key="rec_target_yield",
        help="Minimum yield % to achieve. Set in Step 1 or adjust here.",
    )
    st.session_state["target_yield"] = target_yield

    # Segment batches
    target_mask   = analysis_yields >= target_yield
    n_target      = int(target_mask.sum())
    n_below       = int((~target_mask).sum())
    n_total       = len(analysis_yields)
    target_ids    = analysis_yields[target_mask].index.tolist()
    below_ids     = analysis_yields[~target_mask].index.tolist()

    if n_target < 3:
        st.error(
            f"Only {n_target} batch(es) achieve yield ≥ {target_yield:.1f}%. "
            "Lower the target (in Step 1 or the sidebar) so there are at least 3 reference batches."
        )
        st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Target Yield", f"{target_yield:.1f}%")
    m2.metric("Batches at Target", f"{n_target} / {n_total}",
              help="Batches with yield ≥ target")
    m3.metric("Target Batches Avg Yield",
              f"{analysis_yields[target_mask].mean():.1f}%")
    m4.metric("Below-Target Avg Yield",
              f"{analysis_yields[~target_mask].mean():.1f}%" if n_below > 0 else "—")
    gap = target_yield - (analysis_yields[~target_mask].mean() if n_below > 0 else target_yield)
    m5.metric("Avg Gap to Close", f"+{gap:.1f}%" if gap > 0 else "—",
              help="Average yield increase needed for below-target batches")

    # ── Statistical reliability banner ─────────────────────────────────────────
    # Approximate Mann-Whitney power for the group sizes we have.
    # With n_target and n_below samples, power to detect Cohen's d ≈ 0.8 (large):
    #   n_total ≥ 20 → reasonable;  n_total ≥ 10 → limited;  n_total < 10 → very low.
    _n_test = min(n_target, n_below) if n_below > 0 else n_target
    if _n_test >= 20:
        _power_msg = (
            f"✅ **Adequate sample size** ({n_target} target, {n_below} below-target batches). "
            "Statistical tests have reasonable power to detect meaningful CPP differences."
        )
        st.success(_power_msg)
    elif _n_test >= 10:
        st.warning(
            f"⚠️ **Limited statistical power** ({n_target} target, {n_below} below-target batches). "
            "With fewer than 20 batches per group, Mann-Whitney U tests have ~40–60% power to detect "
            "medium effects. Only trust CPPs that are **both** statistically significant (p < 0.05) "
            "**and** show large Cohen's d (> 0.8). Treat others as directional hints only."
        )
    else:
        st.error(
            f"🔴 **Very low statistical power** ({n_target} target, {n_below} below-target batches). "
            "With fewer than 10 batches per group, the Mann-Whitney U test has < 30% power and "
            "p-values are unreliable. The 'Significant' flags and priority rankings below are "
            "**indicative only** — treat them as hypotheses to test with more batches, not proven findings. "
            f"Consider lowering your target yield to include more reference batches."
        )
    st.info(
        f"Analysis based on **{feat_source}** | "
        f"{n_target} target-achieving batches define the recommended operating windows | "
        f"{n_below} below-target batches show what to avoid."
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # ALGORITHM ENGINE
    # Runs once; results shared across all three tabs
    # ══════════════════════════════════════════════════════════════════════════
    with st.spinner("Running recommendation algorithms…"):

        # ── 1. Conditional Percentile Analysis ────────────────────────────────
        # For each CPP feature: compute the distribution in target vs below batches.
        # Recommended range = P10–P90 of target-achieving batches.
        # Statistical test: Mann-Whitney U (non-parametric, no normality assumption).
        # Effect size: Cohen's d (standardised mean difference).
        # Bootstrap CI: 1000 re-samples of P10/P90 to show uncertainty in the bounds.
        _N_BOOT = 500   # bootstrap iterations (fast enough for interactive use)
        _RNG    = np.random.default_rng(42)

        def _bootstrap_percentile_ci(vals: np.ndarray, q: float, n_boot: int, rng) -> tuple:
            """Return (lower_ci, upper_ci) at 90% confidence for the q-th percentile."""
            if len(vals) < 3:
                return (float(np.percentile(vals, q)),) * 2
            boots = [float(np.percentile(rng.choice(vals, size=len(vals), replace=True), q))
                     for _ in range(n_boot)]
            return float(np.percentile(boots, 5)), float(np.percentile(boots, 95))

        cpp_rec_rows = []
        for col in analysis_features.columns:
            base = col.replace("_mean", "").replace("_std", "")
            t_vals = analysis_features.loc[target_ids, col].dropna()
            b_vals = analysis_features.loc[below_ids, col].dropna() if below_ids else pd.Series(dtype=float)
            all_vals = analysis_features[col].dropna()

            rec_lo  = float(t_vals.quantile(0.10))
            rec_hi  = float(t_vals.quantile(0.90))
            rec_mid = float(t_vals.median())

            # Bootstrap 90% CI on P10 and P90 (shows how uncertain the bounds are)
            t_arr = t_vals.values
            lo_ci_lo, lo_ci_hi = _bootstrap_percentile_ci(t_arr, 10, _N_BOOT, _RNG)
            hi_ci_lo, hi_ci_hi = _bootstrap_percentile_ci(t_arr, 90, _N_BOOT, _RNG)
            cur_mean = float(all_vals.mean())
            cur_std  = float(all_vals.std()) if all_vals.std() > 0 else 1.0
            delta    = rec_mid - cur_mean

            # Mann-Whitney U test for separation (minimum n=5 per group;
            # below that the test has < 20% power and p-values are meaningless)
            p_val = None
            if len(t_vals) >= 5 and len(b_vals) >= 5:
                _, p_val = scipy_stats.mannwhitneyu(t_vals, b_vals, alternative="two-sided")

            # Cohen's d effect size
            pooled_std = float(
                np.sqrt((t_vals.std()**2 + b_vals.std()**2) / 2)
            ) if len(b_vals) > 1 and b_vals.std() > 0 else cur_std
            cohens_d = abs(delta) / (pooled_std + 1e-9)

            significant = (p_val is not None and p_val < 0.05)
            direction   = "↑ Increase" if delta > 0 else "↓ Decrease"

            cpp_rec_rows.append({
                "feature":       col,
                "base":          base,
                "CPP":           CPP_LABELS.get(base, col),
                "Stat Type":     "variability (std)" if col.endswith("_std") else "mean level",
                "Current Avg":   round(cur_mean, 3),
                "Target Median": round(rec_mid, 3),
                "Rec. Min (P10)":  round(rec_lo, 3),
                "P10 CI low":      round(lo_ci_lo, 3),
                "P10 CI high":     round(lo_ci_hi, 3),
                "Rec. Max (P90)":  round(rec_hi, 3),
                "P90 CI low":      round(hi_ci_lo, 3),
                "P90 CI high":     round(hi_ci_hi, 3),
                "Range width":     round(rec_hi - rec_lo, 3),
                "CI uncertainty":  round((lo_ci_hi - lo_ci_lo + hi_ci_hi - hi_ci_lo) / 2, 3),
                "Delta":         round(delta, 3),
                "Delta (σ)":     round(delta / cur_std, 2),
                "Direction":     direction,
                "Cohen's d":     round(cohens_d, 3),
                "p-value":       round(p_val, 4) if p_val is not None else None,
                "Significant":   significant,
                "Priority":      round(cohens_d * (1.5 if significant else 1.0), 3),
            })

        rec_df = pd.DataFrame(cpp_rec_rows).sort_values("Priority", ascending=False).reset_index(drop=True)
        rec_df["Rank"] = range(1, len(rec_df) + 1)

        # ── 2. PLS coefficient direction (if available) ────────────────────────
        # PLS coefficients map X (CPP trajectories) → Y (yield).
        # Aggregation strategy per CPP:
        #   • Direction: sign of the weighted-mean coefficient (weights = |coef|),
        #     so time points with stronger leverage determine the direction.
        #     Raw sum fails when a CPP reverses sign across phases (net ≈ 0 despite
        #     large phase-specific effects).
        #   • Magnitude: mean of absolute values (= average leverage across time).
        pls_coef_per_cpp = {}        # signed weighted mean → direction
        pls_coef_mag_per_cpp = {}    # mean |coef| → magnitude
        if pls_model is not None and active_cpp_vars:
            try:
                coef_flat = pls_model.pls.coef_.flatten()
                J = N_ALIGNED_POINTS
                for k, cpp in enumerate(active_cpp_vars):
                    sl   = slice(k * J, (k + 1) * J)
                    seg  = coef_flat[sl]
                    abs_seg = np.abs(seg)
                    total_abs = abs_seg.sum()
                    if total_abs > 0:
                        # weighted mean preserves sign; large |coef| time steps dominate
                        pls_coef_per_cpp[cpp] = float(np.sum(seg * abs_seg) / total_abs)
                    else:
                        pls_coef_per_cpp[cpp] = 0.0
                    pls_coef_mag_per_cpp[cpp] = float(abs_seg.mean())
            except Exception:
                pls_coef_per_cpp = {}
                pls_coef_mag_per_cpp = {}

        # ── 3. Correlation-weighted priority ──────────────────────────────────
        # Pearson |r| between each CPP feature and yield — re-computed for ranking.
        corr_scores = {}
        for col in analysis_features.columns:
            try:
                r, _ = scipy_stats.pearsonr(analysis_features[col].values,
                                             analysis_yields.values)
                corr_scores[col] = abs(r)
            except Exception:
                corr_scores[col] = 0.0

        # ── 4. Temperature-specific trajectory analysis ────────────────────────
        # Identify temperature CPPs and compute recommended trajectory.
        temp_cpps = []
        if active_cpp_vars:
            for v in active_cpp_vars:
                if any(kw in v.lower() for kw in ["temp", "temperature", "tmp", "t_"]):
                    temp_cpps.append(v)
            if not temp_cpps:
                temp_cpps = active_cpp_vars[:1]   # fallback: first CPP

        traj_recs = {}   # {cpp_name: {target_mean, target_std, target_upper, target_lower, time_norm}}
        if aligned and temp_cpps:
            for cpp in temp_cpps:
                tgt_bids = [b for b in target_ids if b in aligned and cpp in aligned[b].columns]
                blw_bids = [b for b in below_ids  if b in aligned and cpp in aligned[b].columns]
                if len(tgt_bids) < 2:
                    continue
                time_norm_axis = aligned[tgt_bids[0]]["time_norm"].values
                tgt_mat = np.stack([aligned[b][cpp].values for b in tgt_bids])
                tgt_mean = tgt_mat.mean(axis=0)
                tgt_std  = tgt_mat.std(axis=0)
                blw_mean = None
                if blw_bids:
                    blw_mat  = np.stack([aligned[b][cpp].values for b in blw_bids])
                    blw_mean = blw_mat.mean(axis=0)
                traj_recs[cpp] = {
                    "time_norm":    time_norm_axis,
                    "target_mean":  tgt_mean,
                    "target_std":   tgt_std,
                    "target_upper": tgt_mean + tgt_std,
                    "target_lower": tgt_mean - tgt_std,
                    "below_mean":   blw_mean,
                    "n_target":     len(tgt_bids),
                    "n_below":      len(blw_bids),
                }

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY – three tabs
    # ══════════════════════════════════════════════════════════════════════════
    tab_win, tab_traj, tab_action = st.tabs([
        "CPP Operating Windows", "Temperature Profile", "Action Plan"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – CPP Operating Windows
    # ─────────────────────────────────────────────────────────────────────────
    with tab_win:
        st.subheader("Recommended CPP Operating Windows")
        st.markdown(
            "For each CPP, this shows the range observed in the **"
            f"{n_target} batches that achieved yield ≥ {target_yield:.1f}%**. "
            "Staying within the P10–P90 band of those batches gives the best probability of hitting your target. "
            "**Statistically significant CPPs** (p < 0.05) are the most reliable levers — "
            "the difference between target and below-target batches is unlikely to be random."
        )

        # Highlight most impactful
        sig_df = rec_df[rec_df["Significant"]].head(5)
        if not sig_df.empty:
            st.success(
                f"**Highest-impact CPPs (statistically significant, sorted by effect size):** "
                + " | ".join(
                    f"**{r['CPP']}** {r['Direction']} to [{r['Rec. Min (P10)']:.2f} – {r['Rec. Max (P90)']:.2f}]"
                    for _, r in sig_df.iterrows()
                )
            )

        # Visual: delta chart (target median – current avg)
        plot_df = rec_df[rec_df["Delta (σ)"].abs() > 0.05].copy()
        if not plot_df.empty:
            colors = ["#2E7D32" if d > 0 else "#C62828" for d in plot_df["Delta (σ)"]]
            fig_delta = go.Figure(go.Bar(
                x=plot_df["Delta (σ)"],
                y=plot_df["CPP"] + " [" + plot_df["Stat Type"] + "]",
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}σ" for v in plot_df["Delta (σ)"]],
                textposition="outside",
            ))
            fig_delta.add_vline(x=0, line_color="black", line_width=1)
            fig_delta.update_layout(
                title=f"Required shift per CPP to reach target yield ≥ {target_yield:.1f}% (in standard deviations)",
                xaxis_title="Delta from current average (σ units)",
                yaxis=dict(autorange="reversed"),
                height=max(350, 30 * len(plot_df)),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption(
                "Green bars = increase this CPP. Red bars = decrease this CPP. "
                "Length = magnitude of change needed in standard deviation units. "
                "CPPs with bars > 0.5σ are priority adjustment targets."
            )

        # Operating window range chart
        st.subheader("Recommended Ranges vs Current Average")
        range_plot_df = rec_df[["CPP", "Stat Type", "Current Avg",
                                  "Rec. Min (P10)", "Target Median", "Rec. Max (P90)",
                                  "Significant", "Cohen's d"]].copy()
        fig_range = go.Figure()
        for i, row in range_plot_df.iterrows():
            label = f"{row['CPP']} [{row['Stat Type']}]"
            color = "#1976D2" if row["Significant"] else "#90A4AE"
            # Recommended band
            fig_range.add_trace(go.Scatter(
                x=[row["Rec. Min (P10)"], row["Rec. Max (P90)"]],
                y=[label, label], mode="lines",
                line=dict(color=color, width=8), opacity=0.4,
                showlegend=False,
            ))
            # Target median
            fig_range.add_trace(go.Scatter(
                x=[row["Target Median"]], y=[label], mode="markers",
                marker=dict(color=color, size=12, symbol="diamond"),
                showlegend=False,
            ))
            # Current avg
            fig_range.add_trace(go.Scatter(
                x=[row["Current Avg"]], y=[label], mode="markers",
                marker=dict(color="#E53935", size=10, symbol="x"),
                showlegend=False,
            ))
        # Legend traces
        fig_range.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(color="#1976D2", size=12, symbol="diamond"),
            name="Target median (P10–P90 band)"))
        fig_range.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(color="#E53935", size=10, symbol="x"),
            name="Current average (all batches)"))
        fig_range.update_layout(
            title="CPP Operating Windows — blue band = target range, red × = current average",
            xaxis_title="CPP Value",
            height=max(400, 32 * len(range_plot_df)),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_range, use_container_width=True)
        st.caption(
            "**Blue/grey band** = P10–P90 range of target-achieving batches. "
            "**Blue diamond** = median of target-achieving batches. "
            "**Red ×** = current overall average. "
            "When the red × lies outside the band, that CPP needs adjustment. "
            "Solid-blue bands = statistically significant (p < 0.05); grey = not significant."
        )

        # Full table — include bootstrap CI columns
        st.subheader("Full Operating Window Table")
        st.caption(
            "**P10 CI / P90 CI**: 90% confidence interval around the recommended bound "
            f"(bootstrapped, {_N_BOOT} resamples). Wide CI = uncertain bound — the range is less reliable. "
            "Narrow CI = the bound is stable across the target-achieving batches."
        )
        display_cols = ["Rank", "CPP", "Stat Type", "Current Avg",
                        "Target Median",
                        "Rec. Min (P10)", "P10 CI low", "P10 CI high",
                        "Rec. Max (P90)", "P90 CI low", "P90 CI high",
                        "CI uncertainty",
                        "Direction", "Delta (σ)", "Cohen's d", "p-value", "Significant"]
        disp = rec_df[display_cols].copy()
        disp["Significant"] = disp["Significant"].map({True: "✅ Yes", False: "—"})
        st.dataframe(
            disp.style.format({
                "Current Avg": "{:.3f}", "Target Median": "{:.3f}",
                "Rec. Min (P10)": "{:.3f}", "P10 CI low": "{:.3f}", "P10 CI high": "{:.3f}",
                "Rec. Max (P90)": "{:.3f}", "P90 CI low": "{:.3f}", "P90 CI high": "{:.3f}",
                "CI uncertainty": "{:.3f}",
                "Delta (σ)": "{:+.2f}", "Cohen's d": "{:.3f}",
                "p-value": lambda x: f"{x:.4f}" if x is not None else "—",
            }).apply(
                lambda col: ["background-color: #e8f5e9" if v == "✅ Yes" else ""
                             for v in col],
                subset=["Significant"]
            ),
            use_container_width=True,
        )
        with st.expander("How these numbers are calculated"):
            st.markdown(f"""
**Method: Conditional Percentile Analysis + Mann-Whitney U Test + Bootstrap CI**

1. **Target batches**: {n_target} batches with yield ≥ {target_yield:.1f}% from the {feat_source}.
2. **Recommended range**: P10–P90 of each CPP's distribution among those target batches.
3. **Bootstrap CI**: The P10 and P90 bounds are re-computed {_N_BOOT} times on resampled target batches. The 5th–95th percentile of those {_N_BOOT} estimates forms the 90% CI. Wide CI = the bound is sensitive to which batches happened to achieve target — less reliable.
4. **CI uncertainty**: Average half-width of P10 and P90 CIs. Use this to compare CPPs: lower = more trustworthy range.
5. **Delta**: Target median minus current overall average — how much the CPP needs to shift.
6. **Delta (σ)**: Delta normalised by the overall standard deviation — comparisons across CPPs.
7. **Mann-Whitney U**: Non-parametric test for whether target vs below-target distributions differ (requires n ≥ 5 per group).
8. **Cohen's d**: Effect size — how many standard deviations separate the two groups. Values > 0.5 = medium, > 0.8 = large effect.
9. **Priority**: Cohen's d × 1.5 (if statistically significant) — overall importance score for ranking.
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Temperature Profile
    # ─────────────────────────────────────────────────────────────────────────
    with tab_traj:
        st.subheader("Ideal Temperature Trajectory for Target Yield")
        st.markdown(
            "The recommended temperature profile is derived from the **mean ± 1σ trajectory of "
            f"batches that achieved yield ≥ {target_yield:.1f}%**. "
            "This is a data-driven operating setpoint — the amber band is the tolerance window "
            "at each time point along the batch. "
            "Staying within this band throughout the batch maximises the probability of hitting target yield."
        )

        if not traj_recs:
            if not aligned:
                st.info(
                    "Temperature trajectory analysis requires time-series data. "
                    "Upload long-format or pivot-format data (Steps 2 & 3 compatible) on the Overview page."
                )
            else:
                st.info("No temperature-related CPP found in the active CPP list.")
        else:
            # CPP selector (only temperature-related ones shown)
            cpp_pick = st.selectbox(
                "CPP to profile",
                list(traj_recs.keys()),
                format_func=lambda v: CPP_LABELS.get(v, v),
                key="rec_traj_cpp",
            )
            tr = traj_recs[cpp_pick]
            time_x = tr["time_norm"] * 100
            cpp_label = CPP_LABELS.get(cpp_pick, cpp_pick)

            # ── Phase-by-phase recipe summary (plain language) ─────────────────
            phases = {
                "Startup (0–33%)":   (0.00, 0.33),
                "Reaction (33–67%)": (0.33, 0.67),
                "Harvest (67–100%)": (0.67, 1.00),
            }
            phase_rows = []
            for phase_name, (t_lo, t_hi) in phases.items():
                t_norm = tr["time_norm"]
                mask = (t_norm >= t_lo) & (t_norm <= t_hi)
                if not mask.any():
                    continue
                tgt_phase = tr["target_mean"][mask]
                tgt_std_p = tr["target_std"][mask]
                below_phase_mean = tr["below_mean"][mask].mean() if tr["below_mean"] is not None else None
                phase_rows.append({
                    "Phase":              phase_name,
                    "Recommended Mean":   round(float(tgt_phase.mean()), 3),
                    "Lower Bound (−1σ)":  round(float((tgt_phase - tgt_std_p).mean()), 3),
                    "Upper Bound (+1σ)":  round(float((tgt_phase + tgt_std_p).mean()), 3),
                    "Below-Target Mean":  round(float(below_phase_mean), 3) if below_phase_mean is not None else None,
                    "Separation":         round(float(tgt_phase.mean() - below_phase_mean), 3) if below_phase_mean is not None else None,
                })

            # Plain-language recipe callout
            if phase_rows:
                recipe_lines = []
                for pr in phase_rows:
                    lb = pr["Lower Bound (−1σ)"]
                    ub = pr["Upper Bound (+1σ)"]
                    mn = pr["Recommended Mean"]
                    sep = pr["Separation"]
                    sep_str = ""
                    if sep is not None and abs(sep) >= 0.01:
                        direction_word = "higher" if sep > 0 else "lower"
                        sep_str = f" *(target batches run {abs(sep):.2f} {direction_word} than below-target)*"
                    recipe_lines.append(
                        f"- **{pr['Phase']}:** maintain **{cpp_label}** at "
                        f"**{mn:.2f}** (range {lb:.2f} – {ub:.2f}){sep_str}"
                    )
                # Identify the most critical phase
                _sorted_phases = sorted(
                    [pr for pr in phase_rows if pr["Separation"] is not None],
                    key=lambda x: abs(x["Separation"]), reverse=True
                )
                _critical_note = ""
                if _sorted_phases:
                    _cp = _sorted_phases[0]
                    _critical_note = (
                        f"\n\n> **Most critical phase: {_cp['Phase']}** — "
                        f"target batches differ from below-target batches by "
                        f"**{abs(_cp['Separation']):.2f} units** here. "
                        "Tightest control is needed during this window."
                    )
                st.markdown("#### Recommended Temperature Recipe")
                st.success(
                    f"**Recipe for {cpp_label} to achieve yield ≥ {target_yield:.1f}%**\n\n"
                    + "\n".join(recipe_lines)
                    + _critical_note
                )

            fig_traj = go.Figure()

            # Individual below-target batch trajectories (faint, grey)
            if tr["below_mean"] is not None and aligned:
                blw_bids = [b for b in below_ids if b in aligned and cpp_pick in aligned[b].columns]
                for _i, _bid in enumerate(blw_bids):
                    _traj = aligned[_bid][cpp_pick].values
                    fig_traj.add_trace(go.Scatter(
                        x=time_x, y=_traj, mode="lines",
                        line=dict(color="rgba(183,28,28,0.18)", width=1),
                        showlegend=(_i == 0),
                        name="Below-target batches",
                        hovertemplate=f"Batch {_bid}<br>%{{y:.2f}}<extra></extra>",
                    ))
                # Below-target mean
                fig_traj.add_trace(go.Scatter(
                    x=time_x, y=tr["below_mean"], mode="lines",
                    line=dict(color="#B71C1C", width=2, dash="dot"),
                    name=f"Below-target mean (n={tr['n_below']})",
                ))

            # Individual target batch trajectories (faint, green)
            if aligned:
                tgt_bids = [b for b in target_ids if b in aligned and cpp_pick in aligned[b].columns]
                for _i, _bid in enumerate(tgt_bids):
                    _traj = aligned[_bid][cpp_pick].values
                    fig_traj.add_trace(go.Scatter(
                        x=time_x, y=_traj, mode="lines",
                        line=dict(color="rgba(21,101,192,0.15)", width=1),
                        showlegend=(_i == 0),
                        name="Target batches (individual)",
                        hovertemplate=f"Batch {_bid}<br>%{{y:.2f}}<extra></extra>",
                    ))

            # Golden profile (if available)
            if golden_profile and cpp_pick in golden_profile:
                gp = golden_profile[cpp_pick]
                gp_time = golden_profile["time_norm"] * 100
                fig_traj.add_trace(go.Scatter(
                    x=gp_time, y=gp["upper"], mode="lines",
                    line=dict(color="rgba(255,193,7,0)"), showlegend=False))
                fig_traj.add_trace(go.Scatter(
                    x=gp_time, y=gp["lower"], mode="lines", fill="tonexty",
                    fillcolor="rgba(255,193,7,0.15)",
                    line=dict(color="rgba(255,193,7,0)"),
                    name="Historical golden ±2σ",
                ))
                fig_traj.add_trace(go.Scatter(
                    x=gp_time, y=gp["mean"], mode="lines",
                    line=dict(color="#FF8F00", width=2, dash="dash"),
                    name="Historical golden mean",
                ))

            # Recommended band (target-achieving mean ± 1σ)
            fig_traj.add_trace(go.Scatter(
                x=time_x, y=tr["target_upper"], mode="lines",
                line=dict(color="rgba(30,136,229,0)"), showlegend=False))
            fig_traj.add_trace(go.Scatter(
                x=time_x, y=tr["target_lower"], mode="lines", fill="tonexty",
                fillcolor="rgba(30,136,229,0.25)",
                line=dict(color="rgba(30,136,229,0)"),
                name=f"Recommended ±1σ band (yield ≥ {target_yield:.1f}%)",
            ))
            fig_traj.add_trace(go.Scatter(
                x=time_x, y=tr["target_mean"], mode="lines",
                line=dict(color="#1565C0", width=3),
                name=f"Recommended setpoint (n={tr['n_target']} batches)",
            ))

            # Process setpoint from config
            target_val = CPP_TARGETS.get(cpp_pick)
            if target_val is not None:
                fig_traj.add_hline(y=target_val, line_dash="dot", line_color="grey",
                                   annotation_text=f"Process setpoint = {target_val}")

            # Phase boundary lines
            for _pct in [33, 67]:
                fig_traj.add_vline(x=_pct, line_dash="dash", line_color="rgba(128,128,128,0.4)",
                                   annotation_text=["Startup|Reaction", "Reaction|Harvest"][_pct // 33 - 1],
                                   annotation_font_size=10)

            fig_traj.update_layout(
                title=f"Recommended {cpp_label} Recipe for Yield ≥ {target_yield:.1f}%",
                xaxis_title="Normalised Batch Progress (%)",
                yaxis_title=cpp_label,
                height=520, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            st.plotly_chart(fig_traj, use_container_width=True)
            st.caption(
                "**Dark blue line** = recommended setpoint trajectory. "
                "**Blue band** = ±1σ tolerance — stay within this band to hit target yield. "
                "**Faint blue lines** = individual target-achieving batches. "
                "**Faint red lines + red dashed** = individual and mean of below-target batches (what to avoid). "
                "**Amber** = historical golden profile ±2σ (top-25% yield batches). "
                "**Grey dashed verticals** = phase boundaries."
            )

            # ── Phase-by-phase recipe table ────────────────────────────────────
            st.subheader("Phase-by-Phase Setpoints")
            st.markdown(
                "Programme these setpoints into your process control system for each batch phase. "
                "The **Separation** column quantifies the evidence: how different is the target group "
                "from below-target batches in that phase — larger = more critical to control."
            )
            if phase_rows:
                phase_tbl = pd.DataFrame(phase_rows)
                phase_tbl_display = phase_tbl.copy()
                phase_tbl_display["Below-Target Mean"] = phase_tbl_display["Below-Target Mean"].apply(
                    lambda x: f"{x:.3f}" if x is not None else "—"
                )
                phase_tbl_display["Separation"] = phase_tbl_display["Separation"].apply(
                    lambda x: f"{x:+.3f}" if x is not None else "—"
                )

                def _highlight_sep(row):
                    try:
                        v = float(str(row["Separation"]).replace("+",""))
                        if abs(v) > 1.0:
                            return ["background-color: #fff3cd"] * len(row)
                    except Exception:
                        pass
                    return [""] * len(row)

                st.dataframe(
                    phase_tbl_display.style.apply(_highlight_sep, axis=1),
                    use_container_width=True,
                )
                st.caption(
                    "**Yellow rows** = phases where separation from below-target batches is > 1 unit — "
                    "highest priority for tight process control. "
                    "Separation sign: positive = target batches run higher; negative = they run lower."
                )

                # ── Downloadable recipe CSV ────────────────────────────────────
                st.markdown("#### Download Recipe")
                st.markdown(
                    "Export the full time-point-by-time-point recipe as a CSV file "
                    "to upload into your process control system or SCADA historian."
                )
                _recipe_df = pd.DataFrame({
                    "time_pct":          (tr["time_norm"] * 100).round(1),
                    f"{cpp_pick}_setpoint": tr["target_mean"].round(4),
                    f"{cpp_pick}_lower_1sigma": tr["target_lower"].round(4),
                    f"{cpp_pick}_upper_1sigma": tr["target_upper"].round(4),
                })
                _csv_bytes = _recipe_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {cpp_label} recipe CSV ({len(_recipe_df)} time points)",
                    data=_csv_bytes,
                    file_name=f"recipe_{cpp_pick}_yield_ge_{target_yield:.0f}pct.csv",
                    mime="text/csv",
                )

            # ── PLS confirmation ───────────────────────────────────────────────
            if cpp_pick in pls_coef_per_cpp:
                coef_val = pls_coef_per_cpp[cpp_pick]
                direction = "increase" if coef_val > 0 else "decrease"
                coef_mag = pls_coef_mag_per_cpp.get(cpp_pick, abs(coef_val))
                st.info(
                    f"**PLS model confirms:** {cpp_label} has a "
                    f"{'positive' if coef_val > 0 else 'negative'} coefficient "
                    f"(weighted Σcoef = {coef_val:+.4f}, magnitude = {coef_mag:.4f}). "
                    f"The model learned that **{direction}ing** {cpp_label} improves yield — "
                    "consistent with the recipe recommendation above."
                )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – Action Plan
    # ─────────────────────────────────────────────────────────────────────────
    with tab_action:
        st.subheader("Prioritised Action Plan")
        st.markdown(
            "The table below ranks each CPP by **how much it needs to change** × "
            "**how reliably that change predicts yield**. "
            "Tackle the top items first — they give the highest probability of closing the gap "
            f"to target yield ≥ {target_yield:.1f}%."
        )

        # ── Yield Gap Closure Opportunity Banner ──────────────────────────────
        if n_below > 0:
            _below_avg   = float(analysis_yields[~target_mask].mean())
            _target_avg  = float(analysis_yields[target_mask].mean())
            _gap_abs     = _target_avg - _below_avg
            _pct_improve = (_gap_abs / max(_below_avg, 1e-6)) * 100
            st.markdown("#### Yield Improvement Opportunity")
            _gc1, _gc2, _gc3 = st.columns(3)
            _gc1.metric(
                "Below-Target Avg Yield", f"{_below_avg:.1f}%",
                help=f"{n_below} batches below the {target_yield:.1f}% target"
            )
            _gc2.metric(
                "Target-Group Avg Yield", f"{_target_avg:.1f}%",
                help=f"{n_target} batches that achieved ≥ {target_yield:.1f}% yield"
            )
            _gc3.metric(
                "Uplift Opportunity", f"+{_gap_abs:.1f}%",
                delta=f"+{_pct_improve:.0f}% relative improvement",
                help="Average gain if below-target batches matched the target group",
            )
            st.info(
                f"**Historical evidence:** Your {n_below} below-target batch(es) average "
                f"**{_below_avg:.1f}%** yield. Your {n_target} target batch(es) average "
                f"**{_target_avg:.1f}%**. Closing the gap requires an uplift of "
                f"**+{_gap_abs:.1f}% absolute** ({_pct_improve:.0f}% relative). "
                "The CPP changes ranked below are your highest-leverage levers to achieve this. "
                "Addressing all Critical and High priority items is estimated to close "
                "the majority of this gap based on historical process behaviour."
            )
            st.divider()

        # Enrich rec_df with correlation scores
        rec_df["Corr |r|"] = rec_df["feature"].map(corr_scores).fillna(0).round(3)

        # PLS confirmation flag
        def _pls_confirm(row):
            cpp = row["base"]
            if cpp not in pls_coef_per_cpp:
                return "—"
            coef = pls_coef_per_cpp[cpp]
            model_dir = "↑" if coef > 0 else "↓"
            data_dir  = "↑" if row["Delta"] > 0 else "↓"
            match = "✅" if model_dir == data_dir else "⚠️"
            return f"{match} {model_dir} (coef={coef:+.3f})"

        rec_df["PLS Direction"] = rec_df.apply(_pls_confirm, axis=1)

        # Priority tier
        def _tier(priority):
            if priority >= 0.8:   return "🔴 Critical"
            if priority >= 0.5:   return "🟠 High"
            if priority >= 0.25:  return "🟡 Medium"
            return "⚪ Low"

        rec_df["Priority Tier"] = rec_df["Priority"].apply(_tier)

        # Action statement
        def _action(row):
            if abs(row["Delta (σ)"]) < 0.1:
                return f"Maintain {row['CPP']} at current level."
            verb = "Increase" if row["Delta"] > 0 else "Decrease"
            return (
                f"{verb} **{row['CPP']}** [{row['Stat Type']}] "
                f"from current avg {row['Current Avg']:.3f} → "
                f"target {row['Target Median']:.3f} "
                f"(range: {row['Rec. Min (P10)']:.3f} – {row['Rec. Max (P90)']:.3f})."
            )

        rec_df["Action"] = rec_df.apply(_action, axis=1)

        # Display action plan
        action_display = rec_df[["Rank", "Priority Tier", "CPP", "Stat Type",
                                   "Action", "Corr |r|", "Cohen's d",
                                   "p-value", "PLS Direction"]].copy()

        st.dataframe(
            action_display.style.format({
                "Corr |r|": "{:.3f}",
                "Cohen's d": "{:.3f}",
                "p-value": lambda x: f"{x:.4f}" if x is not None else "—",
            }),
            use_container_width=True,
            height=min(600, 40 * len(action_display) + 60),
        )

        st.divider()

        # Narrative summary — top 5 critical/high priority actions
        critical = rec_df[rec_df["Priority Tier"].str.contains("Critical|High")].head(5)
        if not critical.empty:
            st.subheader("Executive Summary — Top Recommendations")
            for _, row in critical.iterrows():
                icon = "🔴" if "Critical" in row["Priority Tier"] else "🟠"
                sig_str = " *(statistically significant)*" if row["Significant"] else ""
                pls_str = f" PLS model confirms direction ({row['PLS Direction']})." if row["PLS Direction"] != "—" else ""
                cohens_d_val = row["Cohen's d"]
                corr_r_val = row["Corr |r|"]
                st.markdown(
                    f"{icon} **{row['Rank']}. {row['Action']}**  \n"
                    f"Effect size: Cohen's d = {cohens_d_val:.2f}{sig_str}. "
                    f"Yield correlation: |r| = {corr_r_val:.2f}.{pls_str}"
                )

        with st.expander("Algorithm methodology"):
            st.markdown(f"""
**Four algorithms were run and combined:**

| Algorithm | What it measures | How it's used |
|---|---|---|
| **Conditional Percentile Analysis** | Distribution of each CPP in target-achieving batches | Defines recommended range (P10–P90) and delta direction |
| **Mann-Whitney U test** | Statistical separation between target vs below-target groups | Identifies CPPs that are reliably different (p < 0.05) |
| **Cohen's d effect size** | Practical magnitude of the CPP difference | Weights the priority score (> 0.8 = large, clinically meaningful effect) |
| **PLS Coefficient Direction** | How the predictive model uses each CPP trajectory | Validates that the recommended direction aligns with the model |

**Priority score** = Cohen's d × 1.5 (if statistically significant, else × 1.0).
CPPs with *both* high Cohen's d and p < 0.05 are the most reliable levers.

**Limitations to keep in mind:**
- Recommendations are based on historical patterns; causation requires process understanding.
- With fewer than ~20 target batches, statistical tests have limited power.
- CPPs not measured or uploaded will not appear in the analysis.
""")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Methodology & Models
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Methodology & Models":
    st.title("Methodology & Models — Technical Reference")
    st.markdown(
        "Complete in-depth reference for every statistical and machine-learning model "
        "used in this application — the problem each solves, the mathematics, its "
        "limitations, and what else could be tried."
    )

    (
        m_corr, m_tree, m_pca, m_lasso,
        m_dtw, m_golden, m_mspc, m_pls,
        m_cpa, m_boot,
    ) = st.tabs([
        "A · Correlation",
        "B · Random Forest & XGBoost",
        "C · PCA",
        "D · LASSO",
        "E · DTW",
        "F · Golden Profile",
        "G · MSPC (T² & SPE)",
        "H · Batch PLS",
        "I · Percentile Analysis",
        "J · Bootstrap CI",
    ])

    # ── A · Correlation ──────────────────────────────────────────────────────
    with m_corr:
        st.header("A · Pearson & Spearman Correlation")
        st.markdown("""
### The Problem
We have N batches. For each batch we know the final yield and the value of each Critical
Process Parameter (CPP). The first question is simple: *which CPPs move up and down
together with yield?*

---
### Why This Model Is Useful Here
Correlation requires no model training, no hyperparameters, and no assumptions about
distribution shape (for Spearman). It is the most transparent analysis available and
serves as the **primary sanity check** against which tree models and LASSO are compared.
In a pharma regulatory context, being able to explain every ranked CPP with a simple
scatter plot is essential.

---
### How It Works

**Pearson r** — measures linear association:

$$r = \\frac{\\sum_{i=1}^{N}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i-\\bar{x})^2 \\cdot \\sum(y_i-\\bar{y})^2}}$$

- $x_i$ = CPP value for batch $i$, $y_i$ = yield for batch $i$
- $r \\in [-1, 1]$ — $|r|=1$ is perfect linear correlation, $r=0$ is no linear association

**Spearman ρ** — replaces raw values with their ranks, then applies Pearson. This
captures *monotonic* (not just linear) relationships and is robust to outliers:

$$\\rho = 1 - \\frac{6 \\sum d_i^2}{N(N^2 - 1)}$$

where $d_i$ = rank difference between $x_i$ and $y_i$.

**Combined score** used in this app:

$$\\text{combined}_j = \\frac{|r_j| + |\\rho_j|}{2}$$

**Statistical significance** — two-sided p-value from t-distribution (df = N−2):

$$t = r\\sqrt{\\frac{N-2}{1-r^2}}, \\quad p = 2 \\cdot P(T_{N-2} > |t|)$$

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Only pairwise relationships | Misses interactions — temperature matters only when pH is also high |
| Sensitive to outlier batches (Pearson) | One extreme batch can substantially inflate or deflate $r$ |
| Multicollinearity blind | Two correlated CPPs both appear important; the true driver is ambiguous |
| Whole-batch mean loses phase information | A CPP critical only during the reaction phase shows a weaker overall correlation |
| Correlation ≠ causation | A correlated CPP may be a proxy for an unmeasured root cause |

---
### What Else Could Be Tried
- **Partial correlation** — measures CPP–yield association after removing the effect of all other CPPs; eliminates multicollinearity bias
- **Distance correlation (dCor)** — detects *any* statistical dependence, not just monotonic
- **Mutual information** — fully non-parametric; captures U-shaped or threshold relationships
- **Phase-specific correlation** — correlate yield with the CPP during each phase separately to pinpoint *when* the CPP matters most
""")

    # ── B · Random Forest & XGBoost ──────────────────────────────────────────
    with m_tree:
        st.header("B · Random Forest & XGBoost")
        st.markdown("""
### The Problem
Correlation detects only pairwise linear/monotonic effects. In reality CPPs interact:
temperature may matter only when dissolved oxygen is simultaneously below 30%.
Tree ensembles discover these threshold interactions automatically.

---
### Why These Models Are Useful Here
- **Interaction detection** — trees split on one variable then another, capturing joint effects
- **Non-linear thresholds** — a CPP may be fine up to 37 °C then catastrophic above it; a tree finds this naturally
- **Robustness to scale** — no normalisation needed
- **Feature importance** — directly comparable across all CPPs after normalisation

---
### How Random Forest Works

1. Draw B bootstrap samples from the N-batch dataset
2. For each sample, grow a regression tree — at each node, consider a random subset of
   features (√p) and split on the feature/threshold minimising MSE of child nodes
3. Predict = average of all B trees

**Impurity-based feature importance** for feature $j$:

$$\\text{Imp}_j = \\frac{1}{B}\\sum_{b=1}^{B}\\sum_{\\substack{t \\in T_b \\\\ \\text{split on }j}} \\frac{N_t}{N}\\Big(\\text{MSE}_t - \\frac{N_{tL}}{N_t}\\text{MSE}_{tL} - \\frac{N_{tR}}{N_t}\\text{MSE}_{tR}\\Big)$$

$N_t$ = samples at node $t$; subscripts $L$, $R$ = left and right children.

### How XGBoost Works

XGBoost builds trees *sequentially*, each correcting the residuals of the previous:

$$\\hat{y}_i^{(m)} = \\hat{y}_i^{(m-1)} + \\eta \\cdot f_m(x_i)$$

The objective at step $m$ adds an L2 regularisation term on leaf weights:

$$\\mathcal{L}^{(m)} = \\sum_i l(y_i, \\hat{y}_i^{(m)}) + \\gamma T + \\frac{\\lambda}{2}\\sum_j w_j^2$$

$T$ = number of leaves, $\\eta$ = learning rate, $w_j$ = leaf weight. Regularisation
limits overfitting — critical on small pharma datasets.

**Adaptive complexity in this app** (small-sample guard):

| n vs p | RF max_depth | XGB max_depth | Trees |
|---|---|---|---|
| n ≥ 3p | 6 | 4 | 300 |
| 2p ≤ n < 3p | 3 | 2 | 200 |
| n < 2p | 2 | 1 | 100 |

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| **Overfitting on small n** | With N=20 batches and p=56 features (rich), importances are highly unstable |
| No extrapolation | Trees cannot predict outside the training range of CPP values |
| Cardinality bias | Continuous CPPs with many split points are artificially inflated vs low-variance ones |
| No direction information | Importance tells you a CPP matters; it does not tell you whether to increase or decrease it |
| No confidence intervals | You cannot tell whether the #1 ranked CPP is reliably different from #2 |

---
### What Else Could Be Tried
- **Permutation importance** (not impurity-based) — shuffle each CPP and measure prediction degradation; unbiased for cardinality
- **SHAP values** — game-theory decomposition of each prediction; shows direction + magnitude per CPP per batch
- **BART (Bayesian Additive Regression Trees)** — provides full posterior over feature importance; ideal for small n
- **Gradient boosting with early stopping** — use a held-out validation fold to stop before overfitting rather than fixing depth
""")

    # ── C · PCA ──────────────────────────────────────────────────────────────
    with m_pca:
        st.header("C · Principal Component Analysis (PCA)")
        st.markdown("""
### The Problem
CPPs are correlated with each other — temperature and dissolved O₂ often move together.
Analysing them individually ignores this multicollinearity. We need to find the
*underlying axes of variation* in the process and ask which axes correlate with yield.

---
### Why PCA Is Useful Here
PCA compresses correlated CPPs into uncorrelated principal components. By correlating
each component's scores with yield, we identify the dominant mode of process variation
most linked to yield — even when that mode spans several correlated CPPs simultaneously.

---
### How It Works

Given standardised feature matrix $\\mathbf{X} \\in \\mathbb{R}^{N \\times p}$ (zero mean, unit variance):

$$\\mathbf{X} = \\mathbf{T}\\mathbf{P}^T + \\mathbf{E}$$

$\\mathbf{T}$ = scores (batch positions in PC space),
$\\mathbf{P}$ = loadings (CPP weights per PC),
$\\mathbf{E}$ = residual.

**Eigendecomposition** of the covariance matrix $\\mathbf{C} = \\frac{1}{N-1}\\mathbf{X}^T\\mathbf{X}$:

$$\\mathbf{C} = \\mathbf{P}\\mathbf{\\Lambda}\\mathbf{P}^T, \\quad \\lambda_1 \\geq \\lambda_2 \\geq \\ldots$$

**Explained variance ratio**:

$$\\text{EVR}_a = \\frac{\\lambda_a}{\\sum_{j=1}^p \\lambda_j}$$

**Adaptive component selection in this app** — minimum $A$ such that:

$$\\sum_{a=1}^{A}\\text{EVR}_a \\geq 0.85$$

This replaces the previous fixed 5-component approach, avoiding retaining noise
components on small datasets.

**Yield-relevant PC** — the component with highest $|r|$ with yield:

$$r_a = \\text{Pearson}\\big(\\mathbf{t}_a,\\; \\mathbf{y}\\big)$$

CPP importance = absolute loading on that PC: $|P_{j,a^*}|$ for each feature $j$.

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| PCA maximises variance, not yield correlation | The yield-relevant PC may explain only 5% of total variance |
| Linear only | Non-linear CPP structures are not captured |
| Post-hoc component selection | Searching over A components for the most yield-correlated one inflates the correlation estimate (multiple comparisons) |
| Loadings are hard to interpret | A CPP can have a large loading on a component for reasons unrelated to yield |
| Unstable on n < p | With fewer batches than features, covariance matrix is rank-deficient |

---
### What Else Could Be Tried
- **Supervised PLS** (Step 3 of this app) — directly maximises covariance with yield rather than total variance; more principled
- **Sparse PCA** — constrains loadings to be non-zero for only a few CPPs; easier to interpret
- **Kernel PCA** — non-linear extension using RBF or polynomial kernels
- **Factor Analysis** — models latent variables with explicit noise terms; better theoretical grounding for separating signal from measurement noise
""")

    # ── D · LASSO ─────────────────────────────────────────────────────────────
    with m_lasso:
        st.header("D · LASSO Regression")
        st.markdown("""
### The Problem
We want to predict yield from CPP features AND simultaneously identify which CPPs are
truly relevant. Ordinary least squares overfits badly when p approaches N, which is
common in batch manufacturing.

---
### Why LASSO Is Useful Here
LASSO is the most statistically appropriate method for CPP identification on small
datasets. Its L1 penalty drives irrelevant CPP coefficients **exactly to zero** —
automatic feature selection. The regularisation strength is chosen by cross-validation.
This app gives LASSO the highest weight (45%) in the reliability-weighted ranking
when sample sizes are small, precisely because it is the only method that
intrinsically guards against overfitting.

---
### How It Works

LASSO solves a penalised least squares problem:

$$\\hat{\\beta} = \\arg\\min_{\\beta}\\left\\{\\frac{1}{N}\\sum_{i=1}^N\\left(y_i - \\beta_0 - \\sum_j \\beta_j x_{ij}\\right)^2 + \\alpha\\sum_j|\\beta_j|\\right\\}$$

The L1 penalty $\\alpha\\sum_j|\\beta_j|$ creates a diamond-shaped constraint region.
The optimum touches the corners of this diamond — exactly where some $\\beta_j = 0$.
This is why LASSO performs feature selection while Ridge (L2) does not.

**Cross-validated alpha selection** (LassoCV):

For each $\\alpha$ on a grid:
1. Split data into $k = \\min(5, N)$ folds
2. Fit on k−1 folds, predict on held-out fold
3. Select $\\alpha^* = \\arg\\min_\\alpha \\text{CV-MSE}(\\alpha)$

**Coefficient interpretation** (after standardisation):
- $\\beta_j > 0$ → increasing this CPP is associated with higher yield
- $\\beta_j < 0$ → increasing this CPP is associated with lower yield
- $\\beta_j = 0$ → CPP eliminated — not predictive given the others
- $|\\beta_j|$ → relative importance magnitude

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Selects only one from a correlated group | If temperature and pH are correlated, LASSO zeros out one arbitrarily |
| Biased coefficient estimates | L1 shrinkage understates the true coefficient magnitudes |
| Linear model | Cannot capture non-linear or interaction effects |
| CV instability at very small n | With n=8 batches, each CV fold has 1–2 samples; the alpha selection is noisy |

---
### What Else Could Be Tried
- **Elastic Net** — L1 + L2 combined; selects groups of correlated CPPs together (unlike LASSO which picks one)
- **Adaptive LASSO** — weights the L1 penalty by inverse initial estimates; removes coefficient bias
- **Bayesian LASSO (Spike-and-Slab prior)** — posterior probability that each CPP is truly relevant; full uncertainty quantification
- **Stability Selection** — run LASSO on many bootstrap samples; report the fraction of runs each CPP is selected; far more robust than a single fit
""")

    # ── E · DTW ───────────────────────────────────────────────────────────────
    with m_dtw:
        st.header("E · Dynamic Time Warping (DTW)")
        st.markdown("""
### The Problem
Two batches can have identical overall mean temperature but very different temperature
*profiles* — one heats quickly then plateaus, the other rises slowly throughout. Euclidean
distance between aligned trajectories treats time mismatches as full deviations.
We need a distance metric robust to slight phase timing differences.

---
### Why DTW Is Useful Here
DTW finds the optimal alignment between two time series, stretching or compressing the
time axis locally. A batch that follows the golden temperature profile but with a
5-minute phase delay has a small DTW distance (similar shape) but a large Euclidean
distance (every point is offset in time). DTW reveals *process similarity* rather
than *timing coincidence*.

---
### How It Works

Given batch series $\\mathbf{s} = (s_1,\\ldots,s_n)$ and golden mean $\\mathbf{g} = (g_1,\\ldots,g_m)$:

**Element cost matrix**: $C_{ij} = (s_i - g_j)^2$

**Accumulated cost via dynamic programming**:

$$D_{ij} = C_{ij} + \\min\\begin{cases}D_{i-1,j} \\\\ D_{i,j-1} \\\\ D_{i-1,j-1}\\end{cases}$$

with boundary conditions $D_{0,0}=0$, $D_{i,0}=D_{0,j}=\\infty$ for $i,j>0$.

**DTW distance**:

$$\\text{DTW}(\\mathbf{s},\\mathbf{g}) = \\sqrt{D_{n,m}}$$

The optimal warping path traces back through $D$ from $(n,m)$ to $(1,1)$, revealing
exactly which time points are matched — and *where* the batch deviates.

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| O(n·m) complexity | Manageable for 100-point aligned series; would be slow for raw 10,000-point series |
| No phase constraint by default | Can match startup data to harvest data — biologically meaningless |
| Sensitive to amplitude differences | Large amplitude deviations dominate even with perfect shape alignment |
| Scalar output loses trajectory shape | One number does not tell you *when* the deviation occurred |

---
### What Else Could Be Tried
- **Sakoe-Chiba band constraint** — restrict warping to $|i-j| \\leq w$; prevents biologically unreasonable alignments
- **Derivative DTW (DDTW)** — applies DTW to the first derivative; focuses on shape changes rather than absolute values; more robust to offset
- **Soft-DTW** — differentiable version; can be used as a loss function for learning golden profiles via gradient descent
- **Shape-based distance (SBD)** — normalises for scale and shift before computing distance; scale-invariant
""")

    # ── F · Golden Profile ────────────────────────────────────────────────────
    with m_golden:
        st.header("F · Golden Batch Profile")
        st.markdown("""
### The Problem
We need a *reference trajectory* for each CPP representing ideal operating conditions.
New batches are compared against this reference in real time to detect deviations early
and provide CPP setpoint guidance to operators.

---
### Why This Approach Is Used
The golden profile is the most interpretable reference: the mean ± band of the
top-performing batches. It translates directly into setpoint recommendations, requires
no complex model, and is directly visualisable by process engineers and regulators.

---
### How It Works

**Step 1 — Golden batch selection**:

$$\\mathcal{G} = \\{b : y_b \\geq Q_{0.75}(\\mathbf{y})\\}$$

Batches at or above the 75th yield percentile. Falls back to 67%, 60%, 50% if fewer
than 3 qualify.

**Step 2 — Consistency filter** (new in this app):

For each CPP and golden batch $b$, compute z-score vs the golden group at each time $t$:

$$z_{b,t} = \\frac{x_{b,t} - \\mu_t^\\mathcal{G}}{\\sigma_t^\\mathcal{G}}$$

A batch is excluded if $> 30\\%$ of its time points have $|z_{b,t}| > 2.5$ — it is a
non-replicable outlier even among the golden group.

**Step 3 — Profile and confidence band** (size $G$ retained batches):

$$\\mu_t = \\frac{1}{G}\\sum_{b \\in \\mathcal{G}} x_{b,t}, \\quad \\sigma_t = \\text{std}_{b \\in \\mathcal{G}}(x_{b,t})$$

**t-distribution 95% CI half-width** (reflects uncertainty in estimating $\\mu_t$):

$$\\text{CI}_t = t_{0.025,\\, G-1} \\cdot \\frac{\\sigma_t}{\\sqrt{G}}$$

**Process spread** (expected variability in a new golden batch):

$$\\text{Spread}_t = 2\\sigma_t$$

**Final band** — wider of the two at each time point:

$$\\text{Band}_t = \\max(\\text{CI}_t,\\; \\text{Spread}_t)$$

With $G = 5$ batches, $t_{0.025,4} = 2.78$ vs $t_{0.025,\\infty} = 1.96$ — the band
widens automatically to reflect genuine uncertainty rather than false precision.

**Batch conformance score** (new in this app):

$$\\text{Conf}_{b,j} = \\frac{1}{T}\\sum_{t=1}^T \\mathbf{1}\\!\\left[\\mu_{j,t} - \\text{Band}_{j,t} \\leq x_{b,j,t} \\leq \\mu_{j,t} + \\text{Band}_{j,t}\\right] \\times 100\\%$$

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| **Circular by construction** | The golden profile IS the high-yield batches; it cannot independently validate that those conditions cause high yield |
| Assumes stationarity | Golden batches from 2 years ago may not reflect current raw material or equipment state |
| Unimodal assumption | If two equally good operating modes exist (36°C and 38°C), the mean (37°C) is a region no golden batch operated in |
| Normalised time alignment | Assumes phases take the same *relative* duration across batches |

---
### What Else Could Be Tried
- **GMM clustering on trajectories** — find 2–3 natural clusters of high-yield batches; one golden profile per cluster
- **Functional Data Analysis (FDA)** — treats each trajectory as a mathematical function; computes functional mean/covariance
- **DTW barycenter averaging** — true average shape of a set of time series without assuming equal phase timing
- **Gaussian Process regression** — Bayesian golden profile with principled uncertainty bands that account for both measurement noise and model uncertainty
""")

    # ── G · MSPC ──────────────────────────────────────────────────────────────
    with m_mspc:
        st.header("G · Multivariate Statistical Process Control (MSPC)")
        st.markdown("""
### The Problem
Traditional univariate SPC monitors one CPP at a time. But a batch can be within
specification on every individual CPP yet still be abnormal — because the *combination*
of values is unusual (e.g. both temperature and pH are high simultaneously, which never
occurred in golden batches). We need multivariate monitoring.

---
### Why MSPC With PCA Is Used Here
PCA compresses the 800-variable unfolded matrix (8 CPPs × 100 time points) into a small
number of components. MSPC then monitors two complementary statistics:
- **T²**: distance from the centre of the normal operating cloud (in PC space)
- **SPE/Q**: pattern not seen in training data at all (residual from the PC subspace)

Together they detect both *known-type* deviations and *completely new* anomalies.

---
### How It Works

**Training** on good batches (yield ≥ 75th percentile):

1. Standardise: $\\tilde{\\mathbf{X}} = (\\mathbf{X} - \\boldsymbol{\\mu}) / \\boldsymbol{\\sigma}$
2. Adaptive PCA (≥ 90% variance): $\\mathbf{T} = \\tilde{\\mathbf{X}}\\mathbf{P}$
3. Eigenvalues $\\lambda_a$ = variance per PC

**Hotelling's T² statistic** for new batch $\\mathbf{x}$:

$$T^2 = \\sum_{a=1}^{A}\\frac{t_a^2}{\\lambda_a}, \\quad t_a = \\tilde{\\mathbf{x}} \\cdot \\mathbf{p}_a$$

$T^2$ measures how far the batch is from the process centre, *scaled by* how spread
out the training data is in each PC direction.

**Squared Prediction Error (SPE)**:

$$\\text{SPE} = \\|\\tilde{\\mathbf{x}} - \\hat{\\tilde{\\mathbf{x}}}\\|^2 = \\|\\tilde{\\mathbf{x}} - \\mathbf{T}\\mathbf{P}^T\\|^2$$

SPE measures how well the PCA model can *reconstruct* the batch. High SPE = the
batch has a pattern of variation the model has never seen.

**Control limits** (empirical from training data):

$$\\text{UCL}_{95} = Q_{95}(T^2_{\\text{train}}), \\quad \\text{UCL}_{99} = Q_{99}(T^2_{\\text{train}})$$

**SPE contribution per CPP** — identifies which CPP caused an alarm:

$$\\text{Contribution}_{j,\\%} = \\frac{\\sum_t e_{j,t}^2}{\\sum_{j'}\\sum_t e_{j',t}^2} \\times 100\\%$$

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Empirical control limits | Based on the 95th percentile of training T²/SPE — not a true statistical distribution |
| Training data quality | If any "good" training batches were actually abnormal, the normal region is distorted |
| Linear PCA | Non-linear process dynamics are not captured |
| T² and SPE can conflict | A batch can be in-control by T² but out-of-control by SPE; requires engineering judgement |
| Future imputation | Unknown future time points filled with training mean — may understate alarm severity |

---
### What Else Could Be Tried
- **Theoretical F/chi² limits** — analytically derived; more principled but require multivariate normality of scores
- **Kernel PCA-based MSPC** — handles non-linear process dynamics
- **Recursive PCA** — model updates as new good batches arrive; avoids stale reference data
- **One-class SVM** — defines the normal operating boundary non-linearly
- **Isolation Forest** — anomaly detection that does not require a multivariate normal distribution for the normal operating region
""")

    # ── H · Batch PLS ─────────────────────────────────────────────────────────
    with m_pls:
        st.header("H · Batch Partial Least Squares (B-PLS)")
        st.markdown("""
### The Problem
MSPC tells us *when* a batch deviates from the normal operating region. But it cannot
forecast the final yield. We need a supervised model that links the entire batch
trajectory (all CPPs, all time points) to the final yield — and can make this
prediction progressively as the batch runs in real time.

---
### Why B-PLS Is Used Here
PLS was designed specifically for the case of many correlated predictors and small N —
exactly the situation in batch manufacturing (800 X-variables, 20–100 batches). It handles
multicollinearity by finding latent variables (LVs) that simultaneously explain variance
in X *and* covariance with Y. This makes it far more stable than multiple linear regression.

---
### How It Works

**Unfolded matrix**: each batch becomes one row of $\\mathbf{X} \\in \\mathbb{R}^{N \\times KJ}$
(K = CPPs, J = 100 time points).

**PLS objective** — find weight vectors $\\mathbf{w}_a$ maximising:

$$\\text{cov}(\\mathbf{X}\\mathbf{w}_a,\\; \\mathbf{y}\\mathbf{q}_a)^2$$

subject to $\\|\\mathbf{w}_a\\| = 1$ and orthogonality of successive LVs.

**NIPALS algorithm** (iterative for each LV $a$):

$$\\mathbf{t}_a = \\mathbf{X}\\mathbf{w}_a$$
$$\\mathbf{p}_a = \\frac{\\mathbf{X}^T\\mathbf{t}_a}{\\mathbf{t}_a^T\\mathbf{t}_a}, \\quad b_a = \\frac{\\mathbf{t}_a^T\\mathbf{y}}{\\mathbf{t}_a^T\\mathbf{t}_a}$$
$$\\mathbf{X} \\leftarrow \\mathbf{X} - \\mathbf{t}_a\\mathbf{p}_a^T, \\quad \\mathbf{y} \\leftarrow \\mathbf{y} - b_a\\mathbf{t}_a$$

**Yield prediction** for a new batch:

$$\\hat{y} = \\mathbf{x}^T\\mathbf{B}_{\\text{PLS}}, \\quad \\mathbf{B}_{\\text{PLS}} = \\mathbf{W}(\\mathbf{P}^T\\mathbf{W})^{-1}\\mathbf{B}\\mathbf{Q}^T$$

**Real-time prediction** — at batch progress $t < J$, fill future time points with
training column means (conservative but unbiased future imputation).

**VIP scores** (Variable Importance in Projection):

$$\\text{VIP}_j = \\sqrt{KJ \\cdot \\frac{\\sum_{a=1}^{A}\\text{SS}_a\\left(\\frac{w_{ja}}{\\|\\mathbf{w}_a\\|}\\right)^2}{\\sum_{a=1}^{A}\\text{SS}_a}}$$

$\\text{SS}_a$ = variance in $\\mathbf{y}$ explained by LV $a$. VIP > 1.0 means
above-average influence on yield.

**LOO-CV R²** (honest generalisation performance):

For each batch $i$: train on all others, predict $\\hat{y}_i$.

$$R^2_{\\text{LOO}} = 1 - \\frac{\\sum_i(y_i - \\hat{y}_i)^2}{\\sum_i(y_i - \\bar{y})^2}$$

$R^2_{\\text{train}}$ is always optimistic. $R^2_{\\text{LOO}}$ is the real test.

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Number of LVs is a hyperparameter | Too few = underfitting; too many = overfitting; currently fixed at 5 |
| Trained on all batches (good and bad) | Failure modes distort the model near average yield values |
| Linear X-Y relationship assumed | Yield response surface may be non-linear |
| Future imputation with training mean | Early predictions are overly stable — they "know" the future is average |
| LOO-CV is pessimistic at small n | Each fold model is trained on n−1 samples, weakening it relative to the full model |

---
### What Else Could Be Tried
- **N-way PLS (Tucker3/NPLS)** — treats data as a 3-way array (batches × CPPs × time); theoretically sounder than unfolding
- **Gaussian Process regression** — full Bayesian treatment with principled uncertainty on yield predictions
- **LSTM/GRU networks** — can learn temporal patterns without unfolding; requires far more batches (>200) than typical pharma datasets
- **Online PLS with recursive updates** — model updates as new completed batches arrive rather than requiring a full refit
""")

    # ── I · Conditional Percentile Analysis ───────────────────────────────────
    with m_cpa:
        st.header("I · Conditional Percentile Analysis + Mann-Whitney U + Cohen's d")
        st.markdown("""
### The Problem
Given a target yield, which CPP values are *consistently associated* with achieving it?
We need to answer three questions simultaneously:
1. What CPP level is typical of target-achieving batches? → recommended operating range
2. Is the difference between groups statistically reliable?
3. How practically large is that difference?

---
### Why This Combination Is Used Here
- **Conditional Percentile Analysis** — gives the operating range in original units; directly actionable for operators
- **Mann-Whitney U** — tests significance without assuming normality; essential for small, potentially non-normal batch data
- **Cohen's d** — quantifies *practical* importance; a p < 0.05 with d = 0.1 is statistically significant but operationally useless

---
### How It Works

**Batch segmentation**:

$$\\mathcal{T} = \\{b : y_b \\geq y^*\\}, \\quad \\mathcal{B} = \\{b : y_b < y^*\\}$$

**Recommended range** for CPP $j$ (from target-achieving batches):

$$[Q_{0.10}(\\mathcal{T}_j),\\; Q_{0.90}(\\mathcal{T}_j)]$$

**Delta** — required shift from current process average:

$$\\Delta_j = \\text{Median}_{\\mathcal{T}}(x_j) - \\text{Mean}_{\\text{all}}(x_j), \\quad \\Delta_j^\\sigma = \\frac{\\Delta_j}{\\text{Std}_{\\text{all}}(x_j)}$$

**Mann-Whitney U** (requires $\\geq 5$ batches per group in this app):

$$U = \\sum_{b \\in \\mathcal{T}}\\sum_{b' \\in \\mathcal{B}}\\mathbf{1}[x_{j,b} > x_{j,b'}]$$

$U$ counts how often a target-achieving batch has a higher CPP value than a below-target batch. P-value from normal approximation to $U$.

**Cohen's d** (effect size):

$$d = \\frac{\\bar{x}_{j,\\mathcal{T}} - \\bar{x}_{j,\\mathcal{B}}}{s_{\\text{pooled}}}, \\quad s_{\\text{pooled}} = \\sqrt{\\frac{s_{\\mathcal{T}}^2 + s_{\\mathcal{B}}^2}{2}}$$

| d | Practical meaning |
|---|---|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| ≥ 0.8 | Large — operationally meaningful |

**Priority score**:

$$\\text{Priority}_j = d_j \\times \\begin{cases}1.5 & p_j < 0.05 \\\\ 1.0 & \\text{otherwise}\\end{cases}$$

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Segmentation threshold is arbitrary | Moving $y^*$ changes which CPPs appear significant |
| CPPs analysed independently | Interactions (temperature × pH) are not captured |
| Low power at small n | With 5 target and 8 below-target batches, power to detect medium effects is ~35% |
| P10–P90 are uncertain at small n | Mitigated by bootstrap CI; but fundamentally requires more data |
| Association ≠ causation | A CPP may correlate with yield as a proxy for an unmeasured root cause |

---
### What Else Could Be Tried
- **ROC analysis** — find the optimal CPP threshold that best discriminates target from below-target batches
- **Logistic regression with binary yield** — jointly estimates all CPP effects while controlling for each other
- **CART decision tree on 2-class yield** — finds the multi-CPP combination (not individual CPPs) that best separates the groups
- **Causal inference (propensity score matching)** — if batches have confounders (different raw material), match them to isolate the true CPP effect
""")

    # ── J · Bootstrap CI ──────────────────────────────────────────────────────
    with m_boot:
        st.header("J · Bootstrap Confidence Intervals")
        st.markdown("""
### The Problem
The recommended CPP ranges (P10–P90) are estimated from a small set of target-achieving
batches. With only 5–10 batches, these estimates are highly variable — but without
bootstrap they were presented as precise point estimates. We need to communicate
the uncertainty in those bounds.

---
### Why Bootstrap Is Used Here
Bootstrap is the most generally applicable method for confidence intervals when:
- The sample size is small
- The distribution of the statistic is unknown (P10/P90 are order statistics with complex sampling distributions)
- No parametric assumptions are warranted

---
### How It Works

For the P10 of CPP $j$ among target batches $\\mathcal{T}$ (size $n_T$):

1. For $b = 1, \\ldots, B = 500$:
   - Draw resample $\\mathcal{T}^*_b$ of size $n_T$ from $\\mathcal{T}$ **with replacement**
   - Compute $\\hat{P}_{10,b} = Q_{0.10}(\\mathcal{T}^*_b)$

2. Bootstrap distribution $\\{\\hat{P}_{10,b}\\}$ approximates the sampling distribution of $\\hat{P}_{10}$

3. **90% percentile CI**:

$$\\text{CI}_{90\\%} = \\left[Q_{0.05}\\big(\\{\\hat{P}_{10,b}\\}\\big),\\; Q_{0.95}\\big(\\{\\hat{P}_{10,b}\\}\\big)\\right]$$

**CI uncertainty** — average half-width of P10 and P90 CIs:

$$\\text{Uncertainty}_j = \\frac{(\\text{CI}_{P10,\\text{hi}} - \\text{CI}_{P10,\\text{lo}}) + (\\text{CI}_{P90,\\text{hi}} - \\text{CI}_{P90,\\text{lo}})}{2}$$

**Practical interpretation**:
- CI width < 10% of recommended range → bound is well-constrained; act on it
- CI width 10–30% → directional guidance is sound; treat exact limits as indicative
- CI width > 30% → collect more target-achieving batches before acting on this CPP's range

---
### Shortcomings

| Issue | Practical Effect |
|---|---|
| Assumes i.i.d. batches | Batches from the same campaign or raw material lot are not independent |
| Percentile CI coverage issues at small n | The 90% CI may not actually contain the true P10 90% of the time with n < 10 |
| B = 500 may be insufficient for extreme percentiles | P10 and P90 are the hardest percentiles to estimate stably |

---
### What Else Could Be Tried
- **BCa bootstrap (Bias-Corrected and Accelerated)** — corrects for both bias and skewness; better coverage than the simple percentile method
- **Bayesian credible intervals with weakly informative prior** — formally incorporates prior process knowledge about plausible CPP ranges
- **Conformal prediction intervals** — distribution-free coverage guarantees for regression-type problems, even with small n

---

### Complete Model Summary

| Model | Step | Primary purpose | Reliable at small n? | Captures interactions? |
|---|---|---|---|---|
| Pearson / Spearman Correlation | 1 | CPP–yield association | ✅ Yes | ❌ No |
| Random Forest | 1 | Non-linear CPP importance | ⚠️ With depth guard | ✅ Yes |
| XGBoost | 1 | Boosted CPP importance | ⚠️ With depth guard | ✅ Yes |
| PCA | 1 | Latent process axes | ⚠️ Moderate | ❌ No |
| LASSO | 1 | CPP selection + direction | ✅ Best at small n | ❌ No |
| Linear Interpolation Alignment | 2 | Time normalisation | ✅ | N/A |
| DTW | 2 | Trajectory shape similarity | ✅ | N/A |
| Golden Profile (t-CI bands) | 2 | Reference trajectory | ✅ Widens for small n | N/A |
| Conformance Score | 2 | Batch adherence to golden | ✅ | N/A |
| MSPC (T² / SPE) | 3 | Multivariate deviation detection | ✅ | Implicitly via PCA |
| Batch PLS + LOO-CV | 3 | Yield forecasting from trajectories | ⚠️ LOO-CV required | ✅ Via latent variables |
| Conditional Percentile Analysis | Rec | CPP operating windows | ⚠️ Low power at n < 20 | ❌ No |
| Mann-Whitney U + Cohen's d | Rec | Significance & practical effect size | ⚠️ Needs n ≥ 5/group | ❌ No |
| Bootstrap CI | Rec | Uncertainty on P10/P90 bounds | ✅ Honest uncertainty | N/A |
""")

