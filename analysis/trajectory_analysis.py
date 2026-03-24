"""
Step 2 – Time-Series Trajectory Analysis
=========================================
Handles the core challenge: batches have DIFFERENT durations, so raw time
series cannot be compared directly.

Pipeline
--------
1. Align batches  – linear interpolation to a fixed 100-point time axis
2. Build golden profile – mean ± k·std of the top-yield batches
3. Unfold batches  – create [N × (K·J)] matrix for MSPC/PLS
4. DTW distance    – measure how far any batch is from the golden trajectory

All functions are pure (no Streamlit side-effects); visualisation is done
in app.py via Plotly.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from config import CPP_VARIABLES, N_ALIGNED_POINTS, YIELD_GOLDEN_PERCENTILE


# ── Step 2A – Batch Alignment ─────────────────────────────────────────────────

def align_batches(
    time_series: dict,
    variables: list = None,
    n_points: int = N_ALIGNED_POINTS,
) -> dict:
    """
    Align all batches to a common normalised time axis [0 … n_points-1].

    Method: Linear interpolation of each CPP from its native time axis
    onto a uniform grid of n_points.

    Why this works
    --------------
    Batch A at t=50/120 and Batch B at t=50/95 represent different
    process stages. By mapping each batch onto normalised time 0–1 we
    ensure t=50/100 always means "halfway through the batch" regardless
    of actual batch duration.

    Returns
    -------
    dict { batch_id → pd.DataFrame with columns [time_norm, CPP1, CPP2, …] }
    """
    if variables is None:
        variables = CPP_VARIABLES

    aligned = {}
    time_norm = np.linspace(0, 1, n_points)     # 0.00, 0.01, …, 1.00

    for batch_id, df in time_series.items():
        raw_time = df['time_index'].values.astype(float)
        t_min, t_max = raw_time.min(), raw_time.max()
        t_range = t_max - t_min
        # Guard: if all timestamps are identical, treat as evenly-spaced
        if t_range == 0:
            t_raw_norm = np.linspace(0, 1, len(raw_time))
        else:
            t_raw_norm = (raw_time - t_min) / t_range

        interp_data = {'time_norm': time_norm}
        for var in variables:
            if var not in df.columns:
                # Fill missing CPP with zeros rather than crashing
                interp_data[var] = np.zeros(n_points)
                continue
            col_vals = pd.to_numeric(df[var], errors='coerce').values.astype(float)
            # Replace NaN with local mean so interp1d gets finite inputs
            nan_mask = np.isnan(col_vals)
            if nan_mask.all():
                col_vals = np.zeros_like(col_vals)
            elif nan_mask.any():
                col_mean = np.nanmean(col_vals)
                col_vals = np.where(nan_mask, col_mean, col_vals)
            f = interp1d(t_raw_norm, col_vals, kind='linear', fill_value='extrapolate')
            interp_data[var] = f(time_norm)

        aligned[batch_id] = pd.DataFrame(interp_data)

    return aligned


# ── Step 2B – Trajectory Representation ──────────────────────────────────────

def build_golden_profile(
    batch_df: pd.DataFrame,
    aligned: dict,
    variables: list = None,
    yield_percentile: float = YIELD_GOLDEN_PERCENTILE,
    k_sigma: float = 2.0,
    consistency_filter: bool = True,
    consistency_cv_threshold: float = 0.5,
) -> dict:
    """
    Compute the "Golden Trajectory" as the mean ± confidence band of the best batches.

    Improvements over the previous fixed-±2σ approach
    --------------------------------------------------
    1. Confidence bands use a t-distribution interval that widens automatically
       when fewer golden batches are available.  With n=5 batches the band is
       wider than with n=30 — reflecting genuine uncertainty, not false precision.
       Formula: mean ± t(α=0.05, df=n-1) × std / sqrt(n)   [standard error CI]
       Combined with the process-spread term: ± k_sigma × std to catch outlier
       single-batch deviations within the golden set.
       Final band = mean ± max(CI_half_width, k_sigma × std)

    2. Consistency filter: optionally removes "lucky" golden batches whose CPP
       profiles are themselves highly variable (CV > threshold).  A batch with
       90% yield but wildly unstable temperature is not a replicable reference.

    3. Reports 'band_source': 'ci' or 'sigma' at each time point so the UI can
       indicate where the band is CI-driven vs spread-driven.

    Parameters
    ----------
    consistency_filter       : bool   remove golden batches with high process CV
    consistency_cv_threshold : float  max allowed CV (std/|mean|) across golden
                                      batches at any time point (per CPP)
    """
    from scipy.stats import t as t_dist

    if variables is None:
        variables = CPP_VARIABLES

    threshold  = batch_df['yield'].quantile(yield_percentile)
    golden_ids = batch_df.loc[batch_df['yield'] >= threshold, 'batch_id'].tolist()
    golden_ids = [b for b in golden_ids if b in aligned]

    if len(golden_ids) < 3:
        raise ValueError(f"Too few golden batches ({len(golden_ids)}). Lower yield_percentile.")

    # ── Consistency filter ────────────────────────────────────────────────────
    # Remove golden batches that are themselves highly erratic across the golden
    # group — they represent "lucky" non-replicable results.
    if consistency_filter and len(golden_ids) >= 5:
        excluded = set()
        for var in variables:
            matrix = np.stack([aligned[bid][var].values for bid in golden_ids])  # [G × T]
            var_mean = np.abs(matrix.mean(axis=0))
            var_std  = matrix.std(axis=0)
            # CV at each time point; ignore near-zero mean points
            safe_mean = np.where(var_mean < 1e-6, 1e-6, var_mean)
            cv = var_std / safe_mean
            # For each batch, check if it is a consistent outlier from the golden group
            batch_z = np.abs(matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
            # Flag batch if it has > 30% of time points with z > 2.5
            for i, bid in enumerate(golden_ids):
                if (batch_z[i] > 2.5).mean() > 0.30:
                    excluded.add(bid)
        filtered_ids = [b for b in golden_ids if b not in excluded]
        if len(filtered_ids) >= 3:
            golden_ids = filtered_ids  # only use filtered if we still have ≥ 3

    n_golden  = len(golden_ids)
    time_norm = aligned[golden_ids[0]]['time_norm'].values
    alpha     = 0.05   # 95% confidence

    # t critical value for the CI; clip to ≤ k_sigma to avoid extreme widening
    if n_golden >= 3:
        t_crit = float(t_dist.ppf(1 - alpha / 2, df=n_golden - 1))
    else:
        t_crit = k_sigma  # fallback

    profile = {
        'batch_ids':   golden_ids,
        'time_norm':   time_norm,
        'variables':   variables,
        'n_golden':    n_golden,
        't_crit':      t_crit,
        'alpha':       alpha,
    }

    for var in variables:
        matrix = np.stack([aligned[bid][var].values for bid in golden_ids])  # [G × T]
        mu     = matrix.mean(axis=0)
        sigma  = matrix.std(axis=0)
        se     = sigma / np.sqrt(n_golden)  # standard error of the mean

        # CI half-width: covers uncertainty in estimating the mean trajectory
        ci_half = t_crit * se
        # Process spread: covers expected variability in a new golden batch
        spread  = k_sigma * sigma
        # Use the wider of the two at each time point
        half_width = np.maximum(ci_half, spread)

        profile[var] = {
            'mean':        mu,
            'std':         sigma,
            'upper':       mu + half_width,
            'lower':       mu - half_width,
            'ci_upper':    mu + ci_half,   # CI-only band (for display)
            'ci_lower':    mu - ci_half,
            'half_width':  half_width,
        }

    return profile


def compute_batch_conformance_score(
    batch_aligned: pd.DataFrame,
    golden_profile: dict,
    variables: list = None,
) -> dict:
    """
    Quantify how closely a batch follows the golden trajectory.

    Conformance score per CPP = % of normalised-time points where the batch
    value falls within the golden ±half_width band.

    Overall conformance = weighted average across CPPs (equal weights).

    Interpretation
    --------------
    Score 100% → batch is perfectly golden at every time point for that CPP.
    Score < 80% → significant deviation — investigate which phase caused it.
    Score < 60% → the batch is operating outside the golden zone for most of
                  the run; expect below-target yield.

    Returns
    -------
    dict {
      'overall':     float  (0–100)
      'per_cpp':     dict { var → score (0–100) }
      'phase_scores': dict { var → { phase_name → score } }
      'deviation_profile': dict { var → np.array of signed z-scores per time point }
    }
    """
    if variables is None:
        variables = golden_profile.get('variables', CPP_VARIABLES)

    per_cpp        = {}
    phase_scores   = {}
    deviation_prof = {}

    time_norm = golden_profile['time_norm']
    phases = {
        'Startup':  (0.00, 0.33),
        'Reaction': (0.33, 0.67),
        'Harvest':  (0.67, 1.00),
    }

    for var in variables:
        if var not in batch_aligned.columns or var not in golden_profile:
            continue
        actual    = batch_aligned[var].values
        gp        = golden_profile[var]
        mu        = gp['mean']
        hw        = gp['half_width']
        sigma     = np.where(gp['std'] < 1e-9, 1e-9, gp['std'])

        within    = (actual >= mu - hw) & (actual <= mu + hw)
        per_cpp[var] = float(within.mean() * 100.0)

        # Signed z-score profile: positive = above golden mean
        deviation_prof[var] = (actual - mu) / sigma

        # Phase-level conformance
        phase_scores[var] = {}
        for pname, (tlo, thi) in phases.items():
            mask = (time_norm >= tlo) & (time_norm <= thi)
            if mask.any():
                phase_scores[var][pname] = float(within[mask].mean() * 100.0)

    overall = float(np.mean(list(per_cpp.values()))) if per_cpp else 0.0

    return {
        'overall':            overall,
        'per_cpp':            per_cpp,
        'phase_scores':       phase_scores,
        'deviation_profile':  deviation_prof,
    }


def compute_phase_summary(
    aligned: dict,
    variables: list = None,
    phases: dict = None,
) -> pd.DataFrame:
    """
    Summarise each variable per batch phase (heating / reaction / cooling).

    Default phases split normalised time into thirds:
      Phase 1 (Startup):   t = 0.00 – 0.33
      Phase 2 (Reaction):  t = 0.33 – 0.67
      Phase 3 (Harvest):   t = 0.67 – 1.00

    Returns
    -------
    pd.DataFrame with columns:
      batch_id, phase, <var>_mean, <var>_std, <var>_max, <var>_auc
    """
    if variables is None:
        variables = CPP_VARIABLES
    if phases is None:
        phases = {
            'Startup  (0–33%)':  (0.00, 0.33),
            'Reaction (33–67%)': (0.33, 0.67),
            'Harvest  (67–100%)': (0.67, 1.00),
        }

    rows = []
    for batch_id, df in aligned.items():
        t = df['time_norm'].values
        for phase_name, (t_lo, t_hi) in phases.items():
            mask = (t >= t_lo) & (t <= t_hi)
            row = {'batch_id': batch_id, 'phase': phase_name}
            for var in variables:
                v = df.loc[mask, var].values
                row[f'{var}_mean'] = v.mean()
                row[f'{var}_std']  = v.std()
                row[f'{var}_max']  = v.max()
                row[f'{var}_auc']  = np.trapezoid(v) if hasattr(np, 'trapezoid') else np.trapz(v)   # area under curve
            rows.append(row)

    return pd.DataFrame(rows)


# ── Step 2C – Batch Unfolding for MSPC / PLS ─────────────────────────────────

def unfold_batches(
    aligned: dict,
    variables: list = None,
    n_points: int = N_ALIGNED_POINTS,
) -> tuple:
    """
    Unfold aligned batch data into a 2-D matrix suitable for MSPC / PLS.

    Unfolding layout (batch-wise):
        Row i  = one complete batch history
        Columns: [var1@t1, var1@t2, …, var1@tJ,
                  var2@t1, …, var2@tJ,
                  …
                  varK@t1, …, varK@tJ]

    Shape: [N_batches × (K_vars × n_points)]

    Returns
    -------
    X         : np.ndarray  [N × (K·J)]
    batch_ids : list of batch_ids in row order
    col_names : list of column names  (e.g. 'temperature_t001')
    """
    if variables is None:
        variables = CPP_VARIABLES

    batch_ids = list(aligned.keys())
    col_names = [f'{v}_t{t+1:03d}' for v in variables for t in range(n_points)]

    rows = []
    for bid in batch_ids:
        df = aligned[bid]
        row = np.concatenate([df[v].values for v in variables])
        rows.append(row)

    X = np.stack(rows)   # [N × (K·J)]
    return X, batch_ids, col_names


# ── DTW distance (simple implementation, no external deps) ────────────────────

def dtw_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two univariate time series.

    Uses the standard O(n·m) DP algorithm.
    For two aligned series of equal length (n = m = 100) this is fast.

    Returns the square-root of the accumulated cost (Euclidean DTW).
    """
    n, m = len(series1), len(series2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost     = (series1[i - 1] - series2[j - 1]) ** 2
            D[i, j]  = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(np.sqrt(D[n, m]))


def dtw_distance_from_golden(
    batch_df: pd.DataFrame,
    golden_profile: dict,
    variables: list = None,
) -> dict:
    """
    Compute DTW distance between a batch and the golden profile for each variable.

    batch_df : aligned batch DataFrame (output of align_batches for one batch)
    golden_profile : output of build_golden_profile

    Returns dict { variable → dtw_distance }
    """
    if variables is None:
        variables = golden_profile.get('variables', CPP_VARIABLES)

    distances = {}
    for var in variables:
        current = batch_df[var].values
        golden_mean = golden_profile[var]['mean']
        distances[var] = dtw_distance(current, golden_mean)

    return distances


def future_impute_row(
    batch_partial: pd.DataFrame,
    current_t: int,
    X_train_mean: np.ndarray,
    variables: list,
    n_points: int = N_ALIGNED_POINTS,
) -> np.ndarray:
    """
    Build a complete unfolded row for a PARTIAL batch at time step current_t.

    Past time steps  (0 … current_t-1)  → use actual measurements
    Future time steps (current_t … J-1) → fill with training mean (imputation)

    This enables real-time MSPC monitoring before the batch ends.

    Parameters
    ----------
    batch_partial : aligned batch DataFrame (all time steps present, we slice)
    current_t     : how many time steps of data are available (1-indexed)
    X_train_mean  : shape [(K·J)], training mean of the unfolded matrix (scaled)
    variables     : list of variable names (same order used in unfold_batches)
    n_points      : total time steps after alignment

    Returns
    -------
    row : np.ndarray shape [1 × (K·J)]
    """
    row = X_train_mean.copy()   # start from training mean (= imputed future)

    for k, var in enumerate(variables):
        start_col = k * n_points
        # Overwrite past columns with real measurements
        actual = batch_partial[var].values[:current_t]
        row[start_col: start_col + current_t] = actual

    return row.reshape(1, -1)
