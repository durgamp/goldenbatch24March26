"""
Step 3 – Batch Monitoring: MSPC + PLS
=======================================

A. MSPC – Multivariate Statistical Process Control
   • Fit PCA on the unfolded matrix of GOOD historical batches
   • Compute Hotelling's T² and SPE (Q) statistics
   • Empirical control limits at 95th / 99th percentile of training data
   • Real-time monitoring using future imputation for partial batches

B. PLS – Partial Least Squares (Batch PLS)
   • Supervised counterpart: links X (trajectories) to Y (yield)
   • Predict final yield from partial batch data (with future imputation)
   • VIP scores identify which CPP×time combinations drive yield
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from config import (
    CPP_VARIABLES, N_ALIGNED_POINTS,
    MSPC_N_COMPONENTS, MSPC_LIMIT_95, MSPC_LIMIT_99,
    PLS_N_COMPONENTS,
)
from analysis.trajectory_analysis import future_impute_row


# ── A. MSPC Model ─────────────────────────────────────────────────────────────

class MSPCModel:
    """
    Multivariate SPC model based on PCA of the unfolded batch matrix.

    Training data should be GOOD batches only (yield ≥ median).
    The model learns the "normal operating region" and flags batches
    that deviate from it.

    Statistics
    ----------
    T² (Hotelling)
        Measures how far a batch is from the centre of the normal
        operating region in the principal-component subspace.
        T² = Σ (t_a² / λ_a)  where λ_a = variance of a-th PC score

    SPE / Q (Squared Prediction Error)
        Measures how well the model CAN describe the batch.
        High SPE means "the batch has something the model never saw before."
        SPE = ‖x – x̂‖²  where x̂ = PCA reconstruction

    Control limits
    --------------
    Empirical: 95th / 99th percentile of training-data statistics.
    (More interpretable than asymptotic F / chi² limits for small N.)
    """

    def __init__(self, n_components: int = MSPC_N_COMPONENTS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n_components)
        self.t2_limit_95  = None
        self.t2_limit_99  = None
        self.spe_limit_95 = None
        self.spe_limit_99 = None
        self.X_train_mean_unscaled = None   # for future imputation (original scale)
        self.X_train_mean_scaled   = None   # for future imputation (scaled)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray,
            variance_threshold: float = 0.90) -> 'MSPCModel':
        """
        Fit on the unfolded matrix of good batches.
        X shape: [N_good × (K·J)]

        Adaptive component selection
        ----------------------------
        Previous version used a fixed n_components (default 5).  This under-
        or over-compressed the data depending on how many CPPs and time points
        are in the matrix.

        New behaviour: fit a full PCA, then keep the minimum number of
        components that explain ≥ variance_threshold of the total variance
        (default 90%).  This ensures:
          • Small datasets → fewer components (less noise retained)
          • Rich datasets  → more components (more signal captured)
        Result is stored as self.n_components_used for inspection.
        """
        self.X_train_mean_unscaled = X.mean(axis=0)
        X_scaled = self.scaler.fit_transform(X)
        self.X_train_mean_scaled = X_scaled.mean(axis=0)

        # Determine max feasible components
        max_comp = min(X_scaled.shape[0] - 1, X_scaled.shape[1], self.n_components)

        # Fit full PCA up to max_comp to find the variance curve
        pca_full = PCA(n_components=max_comp)
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        # Minimum components to reach variance_threshold
        n_sel = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_sel = max(1, min(n_sel, max_comp))
        self.n_components_used     = n_sel
        self.explained_variance_at = float(cumvar[n_sel - 1])

        self.pca = PCA(n_components=n_sel)
        T = self.pca.fit_transform(X_scaled)                 # scores [N × A]
        eigenvalues = self.pca.explained_variance_           # variance per PC

        # T² for training batches
        T2_train = np.sum((T ** 2) / eigenvalues, axis=1)

        # SPE for training batches
        X_recon = self.pca.inverse_transform(T)
        SPE_train = np.sum((X_scaled - X_recon) ** 2, axis=1)

        self.t2_limit_95  = np.percentile(T2_train,  MSPC_LIMIT_95)
        self.t2_limit_99  = np.percentile(T2_train,  MSPC_LIMIT_99)
        self.spe_limit_95 = np.percentile(SPE_train, MSPC_LIMIT_95)
        self.spe_limit_99 = np.percentile(SPE_train, MSPC_LIMIT_99)

        return self

    # ── compute T² and SPE for one row ────────────────────────────────────────

    def compute_stats(self, X_row: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        X_row : np.ndarray  shape [1 × (K·J)]

        Returns
        -------
        (T2, SPE)  both scalars
        """
        X_scaled = self.scaler.transform(X_row)
        T = self.pca.transform(X_scaled)
        eigenvalues = self.pca.explained_variance_

        T2 = float(np.sum((T ** 2) / eigenvalues))

        X_recon = self.pca.inverse_transform(T)
        SPE = float(np.sum((X_scaled - X_recon) ** 2))

        return T2, SPE

    # ── SPE contributions per original variable ────────────────────────────────

    def spe_contributions(
        self,
        X_row: np.ndarray,
        variables: list = None,
        n_points: int = N_ALIGNED_POINTS,
    ) -> pd.Series:
        """
        Decompose SPE into per-variable contributions.

        For each CPP, sum the squared residuals across all time columns
        belonging to that variable.  Normalise so contributions sum to 100 %.

        Returns pd.Series { variable → contribution_pct }
        """
        if variables is None:
            variables = CPP_VARIABLES

        X_scaled = self.scaler.transform(X_row)
        T = self.pca.transform(X_scaled)
        X_recon = self.pca.inverse_transform(T)
        residuals_sq = (X_scaled - X_recon)[0] ** 2   # [K·J]

        contribs = {}
        for k, var in enumerate(variables):
            start = k * n_points
            end   = start + n_points
            contribs[var] = residuals_sq[start:end].sum()

        s = pd.Series(contribs)
        total = s.sum()
        if total == 0:
            return s.sort_values(ascending=False)
        return (s / total * 100.0).sort_values(ascending=False)

    # ── Real-time monitoring trajectory ──────────────────────────────────────

    def monitor_realtime(
        self,
        batch_aligned: pd.DataFrame,
        variables: list = None,
        n_points: int = N_ALIGNED_POINTS,
    ) -> pd.DataFrame:
        """
        Compute T² and SPE at every time step of an ongoing (or completed) batch.

        At each time step t (1 … n_points):
          - Past data (0 … t-1):   use actual measurements
          - Future data (t … J-1): fill with training mean (future imputation)

        Returns
        -------
        pd.DataFrame with columns:
          time_step, T2, SPE, T2_breach_95, T2_breach_99,
          SPE_breach_95, SPE_breach_99
        """
        if variables is None:
            variables = CPP_VARIABLES

        results = []
        for t in range(1, n_points + 1):
            row = future_impute_row(
                batch_partial=batch_aligned,
                current_t=t,
                X_train_mean=self.X_train_mean_unscaled,
                variables=variables,
                n_points=n_points,
            )
            T2, SPE = self.compute_stats(row)
            results.append({
                'time_step':      t,
                'T2':             T2,
                'SPE':            SPE,
                'T2_breach_95':   T2  > self.t2_limit_95,
                'T2_breach_99':   T2  > self.t2_limit_99,
                'SPE_breach_95':  SPE > self.spe_limit_95,
                'SPE_breach_99':  SPE > self.spe_limit_99,
            })

        return pd.DataFrame(results)


# ── B. Batch PLS Model ────────────────────────────────────────────────────────

class PLSBatchModel:
    """
    Supervised Batch PLS: links unfolded trajectory matrix to batch yield.

    Unlike MSPC (which only detects WHEN something goes wrong), PLS:
      • Predicts final yield from partial batch data (early warning)
      • VIP scores identify WHICH CPP-time combinations drive yield
    """

    def __init__(self, n_components: int = PLS_N_COMPONENTS):
        self.n_components = n_components
        self.scaler_x = StandardScaler()
        self.pls      = PLSRegression(n_components=n_components, scale=False)
        self.X_train_mean_unscaled = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PLSBatchModel':
        """
        X : [N × (K·J)]  unfolded batch matrix
        y : [N]          yield values
        """
        self.X_train_mean_unscaled = X.mean(axis=0)
        self.X_train = X.copy()   # kept for LOO-CV
        self.y_train = y.copy()
        X_scaled = self.scaler_x.fit_transform(X)
        self.pls.fit(X_scaled, y.reshape(-1, 1))
        return self

    def loocv_r2(self) -> dict:
        """
        Leave-One-Out Cross-Validation for the PLS yield prediction.

        Why this matters
        ----------------
        Without CV, the PLS R² is measured on the same data it was trained on
        — always optimistic, sometimes near 1.0 even for random data.  LOO-CV
        gives an HONEST estimate of how well the model predicts unseen batches.

        A low LOO-CV R² (e.g. < 0.5) means the model is memorising the
        training data, not learning a generalizable pattern.  Users should
        treat yield predictions with appropriate skepticism in that case.

        Returns
        -------
        dict {
          'r2_loocv'  : float  (–∞ … 1.0; < 0 means worse than predicting mean)
          'rmse_loocv': float  (same units as yield)
          'r2_train'  : float  (training R² — always optimistic)
          'predictions': list of (actual, predicted) tuples
        }
        """
        X, y = self.X_train, self.y_train
        n = len(y)
        if n < 4:
            return {'r2_loocv': None, 'rmse_loocv': None,
                    'r2_train': None, 'predictions': []}

        y_pred_loo = np.zeros(n)
        for i in range(n):
            idx_train = [j for j in range(n) if j != i]
            X_tr = X[idx_train]
            y_tr = y[idx_train]
            X_te = X[[i]]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            n_comp = min(self.n_components, len(idx_train) - 1, X_tr.shape[1])
            if n_comp < 1:
                y_pred_loo[i] = float(np.mean(y_tr))
                continue

            pls_loo = PLSRegression(n_components=n_comp, scale=False)
            pls_loo.fit(X_tr_s, y_tr.reshape(-1, 1))
            y_pred_loo[i] = float(pls_loo.predict(X_te_s)[0, 0])

        ss_res = float(np.sum((y - y_pred_loo) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_loo = 1.0 - ss_res / (ss_tot + 1e-9)
        rmse   = float(np.sqrt(np.mean((y - y_pred_loo) ** 2)))

        # Training R² for comparison
        X_scaled_tr = self.scaler_x.transform(X)
        y_pred_tr   = self.pls.predict(X_scaled_tr).flatten()
        ss_res_tr   = float(np.sum((y - y_pred_tr) ** 2))
        r2_train    = 1.0 - ss_res_tr / (ss_tot + 1e-9)

        return {
            'r2_loocv':   round(float(r2_loo), 3),
            'rmse_loocv': round(rmse, 2),
            'r2_train':   round(float(r2_train), 3),
            'predictions': list(zip(y.tolist(), y_pred_loo.tolist())),
        }

    def predict(self, X_row: np.ndarray) -> float:
        """Predict yield for a single (complete or imputed) unfolded row."""
        X_scaled = self.scaler_x.transform(X_row)
        return float(self.pls.predict(X_scaled)[0, 0])

    def predict_realtime(
        self,
        batch_aligned: pd.DataFrame,
        variables: list = None,
        n_points: int = N_ALIGNED_POINTS,
    ) -> pd.DataFrame:
        """
        Predict yield at every time step of an ongoing batch
        using future imputation for unknown future values.

        Returns pd.DataFrame with columns: time_step, predicted_yield
        """
        if variables is None:
            variables = CPP_VARIABLES

        results = []
        for t in range(1, n_points + 1):
            row = future_impute_row(
                batch_partial=batch_aligned,
                current_t=t,
                X_train_mean=self.X_train_mean_unscaled,
                variables=variables,
                n_points=n_points,
            )
            y_hat = self.predict(row)
            results.append({'time_step': t, 'predicted_yield': y_hat})

        return pd.DataFrame(results)

    # ── VIP scores ────────────────────────────────────────────────────────────

    def compute_vip_scores(
        self,
        variables: list = None,
        n_points: int = N_ALIGNED_POINTS,
    ) -> pd.DataFrame:
        """
        Compute Variable Importance in Projection (VIP) scores.

        VIP > 1.0 → variable (CPP × time) significantly influences yield.
        Aggregate per CPP by summing VIP² then taking the square root
        to get a single importance score per variable.

        VIP formula (Wold 1993):
          VIP_i = sqrt( K × Σ_a [ SS_a × (w_ia / ‖w_a‖)² ] / Σ_a SS_a )

        where:
          K     = number of X-variables (K·J)
          SS_a  = variance in y explained by the a-th latent variable
          w_ia  = a-th column of X weights for feature i

        Returns
        -------
        pd.DataFrame with columns:
          feature, vip_score        (full K·J features)
        And a separate summary:
          variable, agg_vip         (aggregated per CPP)
        """
        if variables is None:
            variables = CPP_VARIABLES

        T = self.pls.x_scores_     # [N × A]
        W = self.pls.x_weights_    # [K·J × A]
        Q = self.pls.y_loadings_   # [1 × A]

        n_features  = W.shape[0]
        n_components = W.shape[1]

        # SS_a = variance in y explained by component a
        SS = np.diag(T.T @ T @ Q.T @ Q).flatten()     # [A]
        total_SS = SS.sum()

        vip = np.zeros(n_features)
        if total_SS > 0:
            for i in range(n_features):
                weight = np.array(
                    [(W[i, a] / (np.linalg.norm(W[:, a]) + 1e-10)) ** 2
                     for a in range(n_components)]
                )
                vip[i] = np.sqrt(n_features * (SS @ weight) / total_SS)

        # Full feature VIP
        col_names = [f'{v}_t{t+1:03d}' for v in variables for t in range(n_points)]
        vip_df = pd.DataFrame({'feature': col_names, 'vip_score': vip})

        # Aggregate VIP per CPP variable (sum VIP² then sqrt → aggregate importance)
        agg = {}
        for k, var in enumerate(variables):
            start = k * n_points
            end   = start + n_points
            agg[var] = float(np.sqrt(np.sum(vip[start:end] ** 2)))

        agg_df = (
            pd.DataFrame.from_dict(agg, orient='index', columns=['agg_vip'])
            .reset_index()
            .rename(columns={'index': 'variable'})
            .sort_values('agg_vip', ascending=False)
        )

        return vip_df, agg_df


# ── Convenience: fit both models in one call ──────────────────────────────────

def fit_monitoring_models(
    X_all: np.ndarray,
    y_all: np.ndarray,
    batch_ids_all: list,
    yield_median: float,
    yield_quality_threshold: float = None,
) -> tuple:
    """
    Fit MSPC on quality batches and PLS on all batches.

    MSPC training strategy
    ----------------------
    The MSPC model defines the "normal operating region". Training it on
    yield ≥ median means 50 % of all batches are considered normal by
    definition — too permissive. Instead we use the 75th percentile (top
    quartile) so the model captures the operating space of genuinely good
    batches. If fewer than 3 batches meet the quality threshold we fall
    back to the 60th, then 50th percentile, and warn the caller via the
    returned metadata dict.

    Parameters
    ----------
    yield_quality_threshold : float, optional
        If provided, use this absolute yield value as the MSPC training
        threshold (e.g. user's target yield). Otherwise use 75th percentile.

    Returns (mspc_model, pls_model, mspc_meta)
    mspc_meta : dict with keys 'n_good', 'threshold_used', 'threshold_label'
    """
    mspc_meta = {}

    if yield_quality_threshold is not None:
        # Use user-specified threshold; fall back if too few batches
        thresholds = [(yield_quality_threshold, f"user target ({yield_quality_threshold:.1f}%)")]
    else:
        thresholds = []

    # Add percentile-based fallbacks
    for pct, label in [(0.75, "75th percentile"), (0.60, "60th percentile"), (0.50, "median")]:
        thresholds.append((float(np.percentile(y_all, pct * 100)), label))

    for thresh, label in thresholds:
        good_mask = y_all >= thresh
        if good_mask.sum() >= 3:
            X_good = X_all[good_mask]
            mspc_meta = {
                "n_good": int(good_mask.sum()),
                "threshold_used": float(thresh),
                "threshold_label": label,
            }
            break
    else:
        # Absolute fallback: use everything
        X_good = X_all
        mspc_meta = {
            "n_good": len(X_all),
            "threshold_used": float(y_all.min()),
            "threshold_label": "all batches (fallback)",
        }

    mspc = MSPCModel().fit(X_good)
    pls  = PLSBatchModel().fit(X_all, y_all)

    return mspc, pls, mspc_meta
