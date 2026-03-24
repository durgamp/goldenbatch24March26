"""
Step 1 – Critical Process Parameter (CPP) Identification
=========================================================
Four statistical techniques are applied to rank CPPs by their influence on yield.
A process engineer then reviews all four rankings and validates the final CPP list.

Techniques
----------
A. Pearson + Spearman Correlation
B. Tree-based Feature Importance (Random Forest + XGBoost)
C. PCA – loadings of the first principal component correlated with yield
D. LASSO Regression – coefficient shrinkage to zero for irrelevant CPPs

Input feature matrix
--------------------
Two feature sets are available:

  Basic (2 features/CPP):
    • mean  – average level across the batch
    • std   – variability (process stability indicator)

  Rich (7 features/CPP) – used when time-series data is available:
    • mean              – whole-batch average
    • std               – whole-batch variability
    • startup_mean      – average during startup phase   (t = 0–33%)
    • reaction_mean     – average during reaction phase  (t = 33–67%)
    • harvest_mean      – average during harvest phase   (t = 67–100%)
    • slope             – linear trend coefficient (normalised by CPP std)
    • auc_norm          – area under curve normalised by batch length

  Phase-specific features tell the models WHERE in the batch each CPP matters,
  which the whole-batch mean completely misses.  A CPP might be critical only
  during the reaction phase — the model cannot discover this from the mean alone.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import CPP_VARIABLES


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_batch_features(time_series: dict, variables: list = None) -> pd.DataFrame:
    """
    Basic feature matrix: mean + std per CPP per batch.
    Returns pd.DataFrame  shape [N_batches × (2 × K_variables)]
    """
    if variables is None:
        variables = CPP_VARIABLES

    rows = {}
    for batch_id, df in time_series.items():
        row = {}
        for var in variables:
            if var not in df.columns:
                continue
            vals = df[var].dropna().values
            row[f'{var}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
            row[f'{var}_std']  = float(np.std(vals))  if len(vals) else np.nan
        rows[batch_id] = row

    return pd.DataFrame(rows).T   # [N × 2K]


def extract_rich_features(time_series: dict, variables: list = None) -> pd.DataFrame:
    """
    Rich phase-aware feature matrix: 7 features per CPP per batch.

    Features
    --------
    For each CPP variable:
      {var}_mean          whole-batch mean
      {var}_std           whole-batch std  (process stability)
      {var}_startup_mean  mean during startup  phase (t = 0.00–0.33)
      {var}_reaction_mean mean during reaction phase (t = 0.33–0.67)
      {var}_harvest_mean  mean during harvest  phase (t = 0.67–1.00)
      {var}_slope         linear trend coefficient, normalised by {var}_std
                          → positive = CPP rising through the batch
                          → large |slope| = non-stationary behaviour
      {var}_auc_norm      area under curve / batch_length
                          → differs from mean only when the trajectory is
                          skewed (peaks or valleys); highlights non-flat profiles

    Why this is better than mean+std
    ---------------------------------
    A CPP might be critical only in ONE phase (e.g. temperature must be stable
    during the reaction phase but is less critical at startup).  The whole-batch
    mean obscures this.  LASSO and correlation on phase-specific features will
    directly reveal which phase matters most.

    Returns
    -------
    pd.DataFrame  shape [N_batches × (7 × K_variables)]
    Index = batch_id
    """
    if variables is None:
        variables = CPP_VARIABLES

    rows = {}
    for batch_id, df in time_series.items():
        if 'time_index' not in df.columns:
            continue
        row = {}
        t_raw = df['time_index'].values.astype(float)
        t_range = t_raw.max() - t_raw.min()
        if t_range == 0:
            continue
        t_norm = (t_raw - t_raw.min()) / t_range  # normalise to [0, 1]
        n = len(t_norm)

        for var in variables:
            if var not in df.columns:
                continue
            v = df[var].interpolate(method='cubic').bfill().ffill().values.astype(float)

            # Whole-batch stats
            whole_mean = float(np.nanmean(v))
            whole_std  = float(np.nanstd(v)) if np.nanstd(v) > 0 else 1.0

            # Phase stats
            startup_mask  = t_norm <= 0.33
            reaction_mask = (t_norm > 0.33) & (t_norm <= 0.67)
            harvest_mask  = t_norm > 0.67

            startup_mean  = float(np.nanmean(v[startup_mask]))  if startup_mask.any()  else whole_mean
            reaction_mean = float(np.nanmean(v[reaction_mask])) if reaction_mask.any() else whole_mean
            harvest_mean  = float(np.nanmean(v[harvest_mask]))  if harvest_mask.any()  else whole_mean

            # Normalised linear slope (trend per unit normalised time / std)
            if n >= 3:
                slope_raw, _, _, _, _ = linregress(t_norm, v)
                slope_norm = float(slope_raw / whole_std)
            else:
                slope_norm = 0.0

            # Normalised AUC (trapezoid area / batch_length, same units as mean)
            _trapfn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
            auc_norm = float(_trapfn(v, t_norm)) if n >= 2 else whole_mean

            row[f'{var}_mean']          = whole_mean
            row[f'{var}_std']           = whole_std
            row[f'{var}_startup_mean']  = startup_mean
            row[f'{var}_reaction_mean'] = reaction_mean
            row[f'{var}_harvest_mean']  = harvest_mean
            row[f'{var}_slope']         = slope_norm
            row[f'{var}_auc_norm']      = auc_norm

        rows[batch_id] = row

    return pd.DataFrame(rows).T   # [N × 7K]


# ── Outlier detection ─────────────────────────────────────────────────────────

def detect_outlier_batches(
    yields: pd.Series,
    features: pd.DataFrame = None,
    iqr_multiplier: float = 2.5,
) -> dict:
    """
    Detect outlier batches that could distort CPP rankings and the golden profile.

    Two checks:
      1. Yield outliers   – yield < Q1 - k*IQR  or  yield > Q3 + k*IQR
      2. Process outliers – batches whose CPP feature vector is unusually far
                            from the process centre (Mahalanobis distance,
                            approximated by high z-scores on multiple features).

    Parameters
    ----------
    yields          : pd.Series  batch_id → yield value
    features        : pd.DataFrame  [N × p]  optional CPP feature matrix
    iqr_multiplier  : float  threshold multiplier (2.5 = less aggressive than 3)

    Returns
    -------
    dict with keys:
      'yield_outliers'   : list of batch_ids with extreme yield
      'process_outliers' : list of batch_ids with extreme CPP profiles
      'all_outliers'     : union of both lists
      'yield_flags'      : pd.Series  batch_id → 'high' / 'low' / None
      'summary'          : str  human-readable description
    """
    q1  = yields.quantile(0.25)
    q3  = yields.quantile(0.75)
    iqr = q3 - q1

    yield_low  = yields[yields < q1 - iqr_multiplier * iqr].index.tolist()
    yield_high = yields[yields > q3 + iqr_multiplier * iqr].index.tolist()
    yield_flags = pd.Series(index=yields.index, dtype=object)
    for bid in yield_low:
        yield_flags[bid] = 'low'
    for bid in yield_high:
        yield_flags[bid] = 'high'

    process_outliers = []
    if features is not None and len(features) >= 5:
        # Z-score based: flag batch if it has ≥ 3 features with |z| > 3
        scaler = StandardScaler()
        Z = scaler.fit_transform(features.fillna(features.mean()))
        n_extreme = (np.abs(Z) > 3).sum(axis=1)
        extreme_idx = np.where(n_extreme >= 3)[0]
        process_outliers = [features.index[i] for i in extreme_idx]

    all_outliers = list(set(yield_low + yield_high + process_outliers))

    lines = []
    if yield_low:
        lines.append(f"{len(yield_low)} very-low-yield batch(es): {yield_low}")
    if yield_high:
        lines.append(f"{len(yield_high)} unusually-high-yield batch(es): {yield_high}")
    if process_outliers:
        lines.append(f"{len(process_outliers)} process outlier(s) (≥3 extreme CPP features): {process_outliers}")
    summary = "; ".join(lines) if lines else "No outliers detected."

    return {
        'yield_outliers':   yield_low + yield_high,
        'process_outliers': process_outliers,
        'all_outliers':     all_outliers,
        'yield_flags':      yield_flags,
        'summary':          summary,
    }


# ── Method A – Correlation analysis ──────────────────────────────────────────

def run_correlation_analysis(features: pd.DataFrame, yields: pd.Series) -> dict:
    """
    Compute Pearson and Spearman correlation of each feature with yield.
    Also computes p-values and flags significant correlations.

    Returns dict with keys:
      'pearson'    : pd.Series  (feature → r value)
      'spearman'   : pd.Series  (feature → ρ value)
      'p_values'   : pd.Series  (feature → Pearson p-value)
      'combined'   : pd.Series  (average of |r| and |ρ|, used for ranking)
      'n_significant': int
    """
    pearson_r  = {}
    spearman_r = {}
    p_values   = {}

    y = yields.loc[features.index].values

    for col in features.columns:
        x = features[col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            pearson_r[col]  = 0.0
            spearman_r[col] = 0.0
            p_values[col]   = 1.0
            continue
        r,  p = stats.pearsonr(x[mask], y[mask])
        rho, _ = stats.spearmanr(x[mask], y[mask])
        pearson_r[col]  = r
        spearman_r[col] = rho
        p_values[col]   = p

    pearson_s  = pd.Series(pearson_r)
    spearman_s = pd.Series(spearman_r)
    p_val_s    = pd.Series(p_values)
    combined   = (pearson_s.abs() + spearman_s.abs()) / 2.0
    n_sig      = int((p_val_s < 0.05).sum())

    return {
        'pearson':       pearson_s,
        'spearman':      spearman_s,
        'p_values':      p_val_s,
        'combined':      combined.sort_values(ascending=False),
        'n_significant': n_sig,
    }


# ── Method B – Tree-based feature importance ──────────────────────────────────

def run_tree_importance(features: pd.DataFrame, yields: pd.Series,
                        random_state: int = 42) -> dict:
    """
    Fit Random Forest and (optionally) XGBoost on the feature matrix.

    Small-sample guard
    ------------------
    N ≥ 3×P  → full model  (max_depth 6 / 4)
    N ≥ 2×P  → shallow     (max_depth 3 / 2)
    N <  2×P → very shallow (max_depth 2 / 1)

    Returns dict with keys:
      'random_forest'  : pd.Series sorted descending
      'xgboost'        : pd.Series sorted descending (or None)
      'combined'       : average of normalised importances
      'sample_warning' : str or None
      'n_samples'      : int
      'n_features'     : int
    """
    X = features.values
    y = yields.loc[features.index].values
    N, P = X.shape

    if N >= 3 * P:
        rf_depth, xgb_depth, n_trees = 6, 4, 300
        sample_warning = None
    elif N >= 2 * P:
        rf_depth, xgb_depth, n_trees = 3, 2, 200
        sample_warning = (
            f"⚠️ Moderate overfitting risk (n={N} batches, p={P} features). "
            "Tree depth reduced. Use correlation and LASSO as primary ranking."
        )
    else:
        rf_depth, xgb_depth, n_trees = 2, 1, 100
        sample_warning = (
            f"🔴 High overfitting risk (n={N} batches, p={P} features). "
            f"Need ≥ {2*P} batches for reliable tree importance. "
            "Correlation and LASSO are more trustworthy at this sample size."
        )

    rf = RandomForestRegressor(
        n_estimators=n_trees, max_depth=rf_depth,
        min_samples_leaf=max(2, N // 10),
        random_state=random_state,
    )
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=features.columns)

    xgb_imp = None
    if HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=n_trees, max_depth=xgb_depth, learning_rate=0.05,
            subsample=min(0.8, max(0.5, N / (N + 10))),
            colsample_bytree=0.8, random_state=random_state,
            verbosity=0,
        )
        model.fit(X, y)
        xgb_imp = pd.Series(model.feature_importances_, index=features.columns)

    rf_norm = rf_imp / rf_imp.sum()
    if xgb_imp is not None:
        xgb_norm = xgb_imp / xgb_imp.sum()
        combined = ((rf_norm + xgb_norm) / 2.0).sort_values(ascending=False)
    else:
        combined = rf_norm.sort_values(ascending=False)

    return {
        'random_forest':  rf_imp.sort_values(ascending=False),
        'xgboost':        xgb_imp.sort_values(ascending=False) if xgb_imp is not None else None,
        'combined':       combined,
        'sample_warning': sample_warning,
        'n_samples':      N,
        'n_features':     P,
    }


# ── Method C – PCA loadings ───────────────────────────────────────────────────

def run_pca_analysis(features: pd.DataFrame, yields: pd.Series,
                     n_components: int = 5,
                     variance_threshold: float = 0.85) -> dict:
    """
    Apply PCA to the (scaled) feature matrix.

    Improvement over previous version
    ----------------------------------
    Components are selected by EXPLAINED VARIANCE (default: first components
    that together explain ≥ 85% of variance), not a fixed count.
    This avoids retaining noise-only components on small datasets.

    Strategy:
      1. Fit PCA.
      2. Auto-select number of components at variance_threshold.
      3. Correlate each selected PC's scores with yield.
      4. Report absolute loadings of the PC most correlated with yield.

    Returns dict with keys:
      'explained_variance_ratio' : array
      'n_components_selected'    : int  (components needed for 85% variance)
      'loadings'                 : pd.DataFrame
      'yield_corr'               : pd.Series  (PC → correlation with yield)
      'important_pc'             : str
      'cpp_importance'           : pd.Series
    """
    X = StandardScaler().fit_transform(features.values)
    y = yields.loc[features.index].values

    # Fit full PCA first to determine how many components are needed
    max_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca_full = PCA(n_components=max_comp)
    pca_full.fit(X)

    # Select minimum components that explain ≥ variance_threshold
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_sel  = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_sel  = max(1, min(n_sel, max_comp))

    pca = PCA(n_components=n_sel)
    T   = pca.fit_transform(X)

    yield_corr = {}
    for i in range(n_sel):
        r, _ = stats.pearsonr(T[:, i], y)
        yield_corr[f'PC{i+1}'] = r
    yield_corr_s = pd.Series(yield_corr)

    important_pc_idx  = int(yield_corr_s.abs().argmax())
    important_pc_name = f'PC{important_pc_idx + 1}'

    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f'PC{i+1}' for i in range(n_sel)],
        columns=features.columns,
    )

    cpp_importance = loadings_df.loc[important_pc_name].abs().sort_values(ascending=False)

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'n_components_selected':    n_sel,
        'loadings':                 loadings_df,
        'yield_corr':               yield_corr_s,
        'important_pc':             important_pc_name,
        'cpp_importance':           cpp_importance,
    }


# ── Method D – LASSO regression ───────────────────────────────────────────────

def run_lasso_analysis(features: pd.DataFrame, yields: pd.Series,
                       cv: int = 5, random_state: int = 42) -> dict:
    """
    Fit LASSO with cross-validated alpha selection.

    LASSO is the most reliable method on small n because:
      • It performs automatic feature selection (drives irrelevant CPPs to zero)
      • CV-alpha selection prevents overfitting to a single regularisation level
      • Coefficients are directly interpretable: sign = direction, size = impact

    Returns dict with keys:
      'best_alpha'    : float
      'coefficients'  : pd.Series (feature → coefficient, signed)
      'selected'      : list of feature names with non-zero coefficient
      'cpp_importance': pd.Series (|coefficient|, sorted descending)
      'r2_train'      : float  (R² on training data — informational only)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    y = yields.loc[features.index].values

    # Clamp CV folds to min(5, n_samples) to avoid errors on very small datasets
    n_cv = min(cv, len(y))
    lasso = LassoCV(cv=n_cv, max_iter=10000, random_state=random_state)
    lasso.fit(X_scaled, y)

    coef = pd.Series(lasso.coef_, index=features.columns)
    cpp_importance = coef.abs().sort_values(ascending=False)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(y, lasso.predict(X_scaled)))

    return {
        'best_alpha':    lasso.alpha_,
        'coefficients':  coef,
        'selected':      list(coef[coef != 0].index),
        'cpp_importance': cpp_importance,
        'r2_train':      r2,
    }


# ── Combined ranking ──────────────────────────────────────────────────────────

def compute_combined_ranking(
    corr_result:  dict,
    tree_result:  dict,
    pca_result:   dict,
    lasso_result: dict,
) -> pd.DataFrame:
    """
    Combine importance scores from all four methods using RELIABILITY-WEIGHTED
    averaging.

    Weighting rationale
    -------------------
    On small n (n < 2p): tree importance is heavily overfit; LASSO and
    correlation are more trustworthy because they either regularise (LASSO)
    or have no model complexity (correlation).

    Reliability tiers based on n_samples / n_features ratio (r):
      r ≥ 3  (large)   : Corr=0.25, Tree=0.30, PCA=0.20, LASSO=0.25  (equal weight)
      r ≥ 2  (medium)  : Corr=0.30, Tree=0.20, PCA=0.15, LASSO=0.35
      r <  2 (small)   : Corr=0.35, Tree=0.10, PCA=0.10, LASSO=0.45

    LASSO gets the highest weight on small n because it is the only method
    that intrinsically guards against overfitting (regularisation path + CV).

    Returns a DataFrame with columns:
      feature, corr_score, tree_score, pca_score, lasso_score,
      weighted_score, rank, base_variable, stat,
      method_agreement  (number of methods where feature is in top 50%)
    """
    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    corr_imp  = _norm(corr_result['combined'])
    tree_imp  = _norm(tree_result['combined'])
    pca_imp   = _norm(pca_result['cpp_importance'])
    lasso_imp = _norm(lasso_result['cpp_importance'])

    # Reliability weights
    n, p = tree_result.get('n_samples', 30), tree_result.get('n_features', 16)
    r = n / max(p, 1)
    if r >= 3:
        w_corr, w_tree, w_pca, w_lasso = 0.25, 0.30, 0.20, 0.25
    elif r >= 2:
        w_corr, w_tree, w_pca, w_lasso = 0.30, 0.20, 0.15, 0.35
    else:
        w_corr, w_tree, w_pca, w_lasso = 0.35, 0.10, 0.10, 0.45

    all_features = (
        corr_imp.index.union(tree_imp.index)
        .union(pca_imp.index).union(lasso_imp.index)
    )

    df = pd.DataFrame(index=all_features)
    df['corr_score']  = corr_imp.reindex(all_features, fill_value=0)
    df['tree_score']  = tree_imp.reindex(all_features, fill_value=0)
    df['pca_score']   = pca_imp.reindex(all_features, fill_value=0)
    df['lasso_score'] = lasso_imp.reindex(all_features, fill_value=0)
    df['weighted_score'] = (
        w_corr  * df['corr_score'] +
        w_tree  * df['tree_score'] +
        w_pca   * df['pca_score']  +
        w_lasso * df['lasso_score']
    )

    # Method agreement: count how many methods rank this feature in the top 50%
    n_feat = len(df)
    thresh = 0.5
    df['method_agreement'] = (
        (df['corr_score']  >= thresh).astype(int) +
        (df['tree_score']  >= thresh).astype(int) +
        (df['pca_score']   >= thresh).astype(int) +
        (df['lasso_score'] >= thresh).astype(int)
    )

    df = df.sort_values('weighted_score', ascending=False).reset_index()
    df.rename(columns={'index': 'feature'}, inplace=True)
    df['rank'] = range(1, len(df) + 1)
    df['w_corr']  = round(w_corr, 2)
    df['w_tree']  = round(w_tree, 2)
    df['w_pca']   = round(w_pca, 2)
    df['w_lasso'] = round(w_lasso, 2)

    df['base_variable'] = df['feature'].str.replace(
        r'_(mean|std|startup_mean|reaction_mean|harvest_mean|slope|auc_norm)$',
        '', regex=True,
    )
    df['stat'] = df['feature'].apply(
        lambda f: (
            'startup mean'  if '_startup_mean' in f else
            'reaction mean' if '_reaction_mean' in f else
            'harvest mean'  if '_harvest_mean' in f else
            'slope'         if '_slope' in f else
            'AUC'           if '_auc_norm' in f else
            'std'           if f.endswith('_std') else
            'mean'
        )
    )

    # Alias for backward-compatibility with any code referencing avg_score
    df['avg_score'] = df['weighted_score']

    return df
