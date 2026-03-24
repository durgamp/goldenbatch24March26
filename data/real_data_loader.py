"""
Real Batch Data Loader
======================
Supports two CSV formats:

Format A – Long-format time-series (one row per time step per batch)
---------------------------------------------------------------------
  batch_id    : string / categorical  – identifies each batch
  time_index  : integer / float       – time step within the batch
  CPP columns : numeric               – process parameters
  yield       : numeric, optional     – constant within a batch

  Enables: Step 1 (CPP identification) + Step 2 (trajectories) + Step 3 (monitoring)

  Example:
    batch_id, time_index, temperature, pH, …, yield
    B001,  0, 37.1, 7.2, …, 85.5
    B001,  1, 37.0, 7.2, …, 85.5
    B002,  0, 36.9, 7.2, …, 78.2

Format B – Wide-format batch summary (one row per batch)
---------------------------------------------------------
  Batch (or batch_id) : string  – unique per row
  CPP1 … CPPN         : numeric – one aggregated value per CPP per batch
  Yield               : numeric, optional

  Enables: Step 1 (CPP identification) only — no time-series for Steps 2 & 3

  Example:
    Batch, CPP1, CPP2, CPP3, …, CPP8, Yield
    B1,    37.1,  7.2, 80.0, …,  6.5,  85.5
    B2,    36.9,  7.1, 79.5, …,  6.4,  78.2
"""

import numpy as np
import pandas as pd

# Column name candidates for auto-detection (lowercase keys)
_BATCH_CANDIDATES = {
    'batch_id', 'batch', 'batchid', 'batch_name',
    'lot', 'lot_id', 'run', 'run_id',
}
_TIME_CANDIDATES = {
    'time_index', 'time', 'timestamp', 'step', 't',
    'time_step', 'timestep', 'elapsed', 'index',
}
_YIELD_CANDIDATES = {
    'yield', 'yield_pct', 'yield_%', 'product_yield',
    'batch_yield', 'output_yield',
}


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect batch_id, time_index, yield, and CPP columns.

    Returns
    -------
    dict with keys:
      batch_col : str
      time_col  : str
      yield_col : str or None
      cpp_cols  : list[str]
      warnings  : list[str]
    """
    cols = list(df.columns)
    cols_lower = {c.lower().strip(): c for c in cols}
    warnings = []

    # ── batch_id ─────────────────────────────────────────────────────────────
    batch_col = None
    for cand in _BATCH_CANDIDATES:
        if cand in cols_lower:
            batch_col = cols_lower[cand]
            break
    if batch_col is None:
        # Fallback: first column whose unique-count is well below row count
        for col in cols:
            if df[col].nunique() < len(df) * 0.6:
                batch_col = col
                break
    if batch_col is None:
        batch_col = cols[0]
        warnings.append(
            f"Could not confidently detect the batch ID column. "
            f"Defaulting to '{batch_col}'. Please verify."
        )

    # ── time_index ────────────────────────────────────────────────────────────
    time_col = None
    for cand in _TIME_CANDIDATES:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break
    if time_col is None:
        for col in cols:
            if col == batch_col:
                continue
            if pd.api.types.is_integer_dtype(df[col]):
                time_col = col
                break
    if time_col is None:
        remaining = [c for c in cols if c != batch_col]
        if remaining:
            time_col = remaining[0]
            warnings.append(
                f"Could not detect the time column. "
                f"Defaulting to '{time_col}'. Please verify."
            )

    # ── yield ─────────────────────────────────────────────────────────────────
    yield_col = None
    for cand in _YIELD_CANDIDATES:
        if cand in cols_lower:
            yield_col = cols_lower[cand]
            break
    if yield_col is None:
        # Fallback: any column whose name contains 'yield'
        for col_key, col_orig in cols_lower.items():
            if 'yield' in col_key and col_orig not in {batch_col, time_col}:
                yield_col = col_orig
                break

    # ── CPP columns ───────────────────────────────────────────────────────────
    skip = {c for c in [batch_col, time_col, yield_col] if c}
    cpp_cols = [
        c for c in cols
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not cpp_cols:
        warnings.append(
            "No CPP columns detected. Ensure CPP columns contain numeric values."
        )

    return {
        'batch_col': batch_col,
        'time_col':  time_col,
        'yield_col': yield_col,
        'cpp_cols':  cpp_cols,
        'warnings':  warnings,
    }


def parse_batch_csv(
    df: pd.DataFrame,
    batch_col: str,
    time_col: str,
    cpp_cols: list,
    yield_col: str = None,
) -> tuple:
    """
    Parse a long-format batch DataFrame into (batch_df, time_series).

    Parameters
    ----------
    df         : raw DataFrame (one row per time step per batch)
    batch_col  : column identifying batches
    time_col   : column identifying time steps within a batch
    cpp_cols   : list of CPP column names
    yield_col  : optional column with yield values (constant per batch)

    Returns
    -------
    batch_df    : pd.DataFrame  columns [batch_id, yield, duration, quality, is_golden]
    time_series : dict  { batch_id -> DataFrame[batch_id, time_index, cpp1, …] }
    """
    records = []
    time_series = {}

    for batch_id, group in df.groupby(batch_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)

        # Build time-series DataFrame with standardised 'time_index' column name
        ts_df = group[[time_col] + cpp_cols].copy()
        ts_df = ts_df.rename(columns={time_col: 'time_index'})
        ts_df.insert(0, 'batch_id', str(batch_id))

        # Yield: constant per batch → take mean to handle repeated values
        if yield_col and yield_col in group.columns:
            yield_val = float(group[yield_col].dropna().mean())
        else:
            yield_val = np.nan

        time_series[str(batch_id)] = ts_df
        records.append({
            'batch_id': str(batch_id),
            'yield':    round(yield_val, 2) if not np.isnan(yield_val) else np.nan,
            'duration': len(group),
        })

    batch_df = pd.DataFrame(records)

    # Quality labelling (only when yield is fully available)
    if batch_df['yield'].notna().all() and len(batch_df) >= 4:
        q75 = batch_df['yield'].quantile(0.75)
        q50 = batch_df['yield'].quantile(0.50)
        batch_df['quality']   = batch_df['yield'].apply(
            lambda y: 'Golden' if y >= q75 else ('Good' if y >= q50 else 'Poor')
        )
        batch_df['is_golden'] = batch_df['yield'] >= q75
    else:
        batch_df['quality']   = 'Unknown'
        batch_df['is_golden'] = False

    return batch_df, time_series


# ── Format B: Wide-format batch summary ───────────────────────────────────────

def detect_wide_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect batch_id, yield, and CPP columns in wide-format batch summary.

    One row per batch; no time column expected.

    Returns
    -------
    dict with keys:
      batch_col : str
      yield_col : str or None
      cpp_cols  : list[str]
      warnings  : list[str]
    """
    cols = list(df.columns)
    cols_lower = {c.lower().strip(): c for c in cols}
    warnings = []

    # ── batch_id ──────────────────────────────────────────────────────────────
    batch_col = None
    for cand in _BATCH_CANDIDATES:
        if cand in cols_lower:
            batch_col = cols_lower[cand]
            break
    if batch_col is None:
        # Prefer first non-numeric column as batch identifier
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                batch_col = col
                break
    if batch_col is None:
        batch_col = cols[0]
        warnings.append(
            f"Could not confidently detect the batch ID column. "
            f"Defaulting to '{batch_col}'. Please verify."
        )

    # ── yield ─────────────────────────────────────────────────────────────────
    yield_col = None
    for cand in _YIELD_CANDIDATES:
        if cand in cols_lower:
            yield_col = cols_lower[cand]
            break
    if yield_col is None:
        for col_key, col_orig in cols_lower.items():
            if 'yield' in col_key and col_orig != batch_col:
                yield_col = col_orig
                break

    # ── CPP columns ───────────────────────────────────────────────────────────
    skip = {c for c in [batch_col, yield_col] if c}
    cpp_cols = [
        c for c in cols
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not cpp_cols:
        warnings.append(
            "No CPP columns detected. Ensure CPP columns contain numeric values."
        )

    return {
        'batch_col': batch_col,
        'yield_col': yield_col,
        'cpp_cols':  cpp_cols,
        'warnings':  warnings,
    }


def parse_wide_batch_summary(
    df: pd.DataFrame,
    batch_col: str,
    cpp_cols: list,
    yield_col: str = None,
) -> tuple:
    """
    Parse a wide-format batch summary into (batch_df, features_df).

    Parameters
    ----------
    df         : raw DataFrame (one row per batch)
    batch_col  : column identifying batches
    cpp_cols   : list of CPP column names
    yield_col  : optional column with yield values

    Returns
    -------
    batch_df    : pd.DataFrame  columns [batch_id, yield, duration, quality, is_golden]
    features_df : pd.DataFrame  indexed by batch_id, columns = cpp_cols
    """
    records = []
    features_records = {}

    for _, row in df.iterrows():
        batch_id = str(row[batch_col])
        if yield_col and yield_col in row.index and pd.notna(row[yield_col]):
            yield_val = float(row[yield_col])
        else:
            yield_val = np.nan

        features_records[batch_id] = {
            c: (float(pd.to_numeric(row[c], errors='coerce'))
                if pd.notna(row[c]) else np.nan)
            for c in cpp_cols
        }
        records.append({
            'batch_id': batch_id,
            'yield':    round(yield_val, 2) if not np.isnan(yield_val) else np.nan,
            'duration': 1,   # no time-series
        })

    batch_df = pd.DataFrame(records)
    features_df = pd.DataFrame.from_dict(features_records, orient='index')
    features_df.index.name = 'batch_id'

    # Quality labelling
    if batch_df['yield'].notna().all() and len(batch_df) >= 4:
        q75 = batch_df['yield'].quantile(0.75)
        q50 = batch_df['yield'].quantile(0.50)
        batch_df['quality']   = batch_df['yield'].apply(
            lambda y: 'Golden' if y >= q75 else ('Good' if y >= q50 else 'Poor')
        )
        batch_df['is_golden'] = batch_df['yield'] >= q75
    else:
        batch_df['quality']   = 'Unknown'
        batch_df['is_golden'] = False

    return batch_df, features_df


# ── Format C: Pivot time-series (rows=timestamps, columns=batches) ────────────

def detect_pivot_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect the timestamp column and batch columns in a pivot time-series file.

    Format: rows = time steps, columns = [Timestamp, B1, B2, …, BN]
    One file represents ONE CPP parameter across all batches.

    Returns
    -------
    dict with keys:
      time_col   : str
      batch_cols : list[str]
      n_batches  : int
      n_timepoints : int
      warnings   : list[str]
    """
    cols = list(df.columns)
    warnings = []

    # Time column: first non-numeric column, or first column if all numeric
    time_col = None
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            time_col = col
            break
    if time_col is None:
        time_col = cols[0]
        warnings.append(
            f"All columns appear numeric. Defaulting to '{time_col}' as the time column. "
            "Please verify."
        )

    batch_cols = [c for c in cols if c != time_col]
    if not batch_cols:
        warnings.append("No batch columns detected after removing the time column.")

    return {
        'time_col':     time_col,
        'batch_cols':   batch_cols,
        'n_batches':    len(batch_cols),
        'n_timepoints': len(df),
        'warnings':     warnings,
    }


def parse_pivot_file(
    df: pd.DataFrame,
    time_col: str,
    batch_cols: list,
    cpp_name: str,
) -> dict:
    """
    Parse one pivot file into per-batch DataFrames for a single CPP.

    Parameters
    ----------
    df         : raw DataFrame (rows = time steps, cols = time + batches)
    time_col   : column name for timestamps
    batch_cols : column names corresponding to batch IDs
    cpp_name   : name of the CPP this file represents (e.g. 'temperature')

    Returns
    -------
    dict  { batch_id -> DataFrame[batch_id, time_index, <cpp_name>] }
    """
    result = {}
    for batch_id in batch_cols:
        ts = df[[time_col, batch_id]].copy()
        ts = ts.rename(columns={batch_id: cpp_name})
        # Normalise time to integer index (0, 1, 2, …) for pipeline compatibility
        ts['time_index'] = np.arange(len(ts))
        ts.insert(0, 'batch_id', str(batch_id))
        result[str(batch_id)] = ts[['batch_id', 'time_index', cpp_name]].copy()
    return result


def merge_pivot_files(
    cpp_results: dict,
    yield_data: dict = None,
) -> tuple:
    """
    Merge multiple CPP pivot results into (batch_df, time_series).

    Parameters
    ----------
    cpp_results : dict  { cpp_name -> dict { batch_id -> DataFrame } }
                  Output from multiple calls to parse_pivot_file.
    yield_data  : dict  { batch_id -> float }  or None

    Returns
    -------
    batch_df    : pd.DataFrame  [batch_id, yield, duration, quality, is_golden]
    time_series : dict  { batch_id -> DataFrame[batch_id, time_index, cpp1, cpp2, …] }
    """
    # Collect all batch IDs that appear in every CPP file
    all_batches: set = set()
    for cpp_batches in cpp_results.values():
        all_batches.update(cpp_batches.keys())

    time_series = {}
    cpp_names = list(cpp_results.keys())

    for batch_id in sorted(all_batches):
        merged = None
        for cpp_name in cpp_names:
            batches = cpp_results[cpp_name]
            if batch_id not in batches:
                continue
            sub = batches[batch_id][['time_index', cpp_name]]
            if merged is None:
                merged = batches[batch_id][['batch_id', 'time_index', cpp_name]].copy()
            else:
                merged = merged.merge(sub, on='time_index', how='outer')

        if merged is None:
            continue
        merged = merged.sort_values('time_index').reset_index(drop=True)
        merged['batch_id'] = str(batch_id)
        time_series[str(batch_id)] = merged

    # Build batch_df
    records = []
    for batch_id, ts in time_series.items():
        yield_val = (
            float(yield_data[batch_id])
            if yield_data and batch_id in yield_data else np.nan
        )
        records.append({
            'batch_id': batch_id,
            'yield':    round(yield_val, 2) if not np.isnan(yield_val) else np.nan,
            'duration': len(ts),
        })

    batch_df = pd.DataFrame(records)

    # Quality labelling
    if batch_df['yield'].notna().all() and len(batch_df) >= 4:
        q75 = batch_df['yield'].quantile(0.75)
        q50 = batch_df['yield'].quantile(0.50)
        batch_df['quality']   = batch_df['yield'].apply(
            lambda y: 'Golden' if y >= q75 else ('Good' if y >= q50 else 'Poor')
        )
        batch_df['is_golden'] = batch_df['yield'] >= q75
    else:
        batch_df['quality']   = 'Unknown'
        batch_df['is_golden'] = False

    return batch_df, time_series


# ── Format A variant: Long time-series without batch_id column ────────────────

def detect_ts_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect time_index and CPP columns in a time-series file that has
    no batch_id column (rows = all time steps for all batches, stacked).

    Returns
    -------
    dict with keys:
      time_col : str
      cpp_cols : list[str]
      warnings : list[str]
    """
    cols = list(df.columns)
    cols_lower = {c.lower().strip(): c for c in cols}
    warnings = []

    time_col = None
    for cand in _TIME_CANDIDATES:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break
    if time_col is None:
        for col in cols:
            if pd.api.types.is_integer_dtype(df[col]):
                time_col = col
                break
    if time_col is None:
        time_col = cols[0]
        warnings.append(
            f"Could not detect the time column. Defaulting to '{time_col}'. Please verify."
        )

    cpp_cols = [
        c for c in cols
        if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not cpp_cols:
        warnings.append("No CPP columns detected. Ensure CPP columns contain numeric values.")

    return {
        'time_col': time_col,
        'cpp_cols': cpp_cols,
        'warnings': warnings,
    }


def parse_timeseries_no_batchid(
    df: pd.DataFrame,
    time_col: str,
    cpp_cols: list,
    batch_ids: list,
) -> dict:
    """
    Parse a long-format time-series that has no batch_id column.

    Batches are assumed to be stacked sequentially (top-to-bottom) in the
    same order as `batch_ids`.  All batches have equal time steps:
    n_steps = len(df) // len(batch_ids).  Remainder rows are dropped.

    Parameters
    ----------
    df         : raw DataFrame without a batch_id column
    time_col   : column name for the time / step index
    cpp_cols   : list of CPP column names
    batch_ids  : ordered list of batch identifiers (from the summary file)

    Returns
    -------
    dict  { batch_id -> DataFrame[batch_id, time_index, cpp1, …] }
    """
    n_batches = len(batch_ids)
    n_steps   = len(df) // n_batches

    time_series = {}
    for i, batch_id in enumerate(batch_ids):
        chunk = df.iloc[i * n_steps : (i + 1) * n_steps].copy()
        ts_df = chunk[[time_col] + cpp_cols].copy()
        ts_df = ts_df.rename(columns={time_col: 'time_index'})
        ts_df['time_index'] = np.arange(len(ts_df))
        ts_df.insert(0, 'batch_id', str(batch_id))
        time_series[str(batch_id)] = ts_df.reset_index(drop=True)

    return time_series
