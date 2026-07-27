"""
  1. Stationarity      ADF + KPSS
  2. Redundancy        Spearman correlation  
  3. Multicollinearity VIF elimination (linear model only)
  4. Importance        permutation importance on out-of-sample folds
"""

import warnings
 
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
 
 
# ============================================================
 
# >> 1  STATIONARITY
 
# ============================================================
 
_LEVEL_PATTERNS = (
    # rolling means of levels
    "rmean",
    # raw lags of level series (knife price, competitor ratios, player count)
    "knife_mean_lag",
    "knife_rmean",
    "units_rmean",
    "players_rmean",
    "ratio_min_lag",
    "ratio_med_lag",
    "ratio_min_rmean",
    "ratio_med_rmean",
    "ratio_spread_lag",
    # roi levels (mean-reverting but borderline — diff for OLS safety)
    "roi_mean_lag",
    "roi_rmean",
    "roi_range_lag",
    "roi_x_players",       # interaction of two level-ish series
)
 
 
_DROP_PATTERNS = (
    # raw OHLC of the byproduct — no lag applied, non-stationary levels
    "knife_low", "knife_high", "knife_open", "knife_close",
)
 
 
def _is_level(col: str) -> bool:
    return any(col.startswith(p) or p in col for p in _LEVEL_PATTERNS)
 
 
def _should_drop(col: str) -> bool:
    return any(col == p or col.startswith(p) for p in _DROP_PATTERNS)
 
 
def _adf_kpss(series: pd.Series, min_obs: int = 20):
    """Run ADF + KPSS on a single clean series. Returns (adf_p, kpss_p)."""
    s = series.dropna()
    if len(s) < min_obs:
        return np.nan, np.nan
    try:
        adf_p = adfuller(s, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan
    try:
        # KPSS reports p-values from a lookup table and warns whenever the
        # statistic falls outside it. That happens for most strongly
        # stationary or strongly trending series, so it fires constantly and
        # means only "p is beyond 0.01/0.10", which _verdict already treats as
        # a bound. Suppressed here specifically — not globally, because
        # convergence warnings elsewhere are worth seeing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    return adf_p, kpss_p
 
def _verdict(adf_p, kpss_p, adf_thresh=0.05, kpss_thresh=0.05) -> str:
    if pd.isna(adf_p) or pd.isna(kpss_p):
        return "SKIP"
    adf_ok  = adf_p  < adf_thresh    # rejects unit root  → good
    kpss_ok = kpss_p > kpss_thresh   # fails to reject stationarity → good
    if adf_ok and kpss_ok:
        return "STATIONARY"
    if not adf_ok and not kpss_ok:
        return "NON-STATIONARY"
    return f"INCONCLUSIVE (ADF {'ok' if adf_ok else 'weak'}, KPSS {'ok' if kpss_ok else 'fails'})"
 
 
def test_stationarity(X: pd.DataFrame) -> pd.DataFrame:
    """Run ADF + KPSS on each feature's raw values and record a verdict."""
    records = []
    for col in X.columns:
        raw = X[col]
        adf_p, kpss_p = _adf_kpss(raw)
        v = _verdict(adf_p, kpss_p)
 
        if v == "STATIONARY":
            action = "USE RAW"
        elif _should_drop(col):
            action = "DROP"
        elif _is_level(col) or v == "NON-STATIONARY":
            action = "DIFF"
        else:
            action = "REVIEW"          # inconclusive on a non-level feature
 
        records.append(dict(
            feature  = col,
            verdict  = v,
            adf_p    = round(adf_p,  4) if not pd.isna(adf_p)  else np.nan,
            kpss_p   = round(kpss_p, 4) if not pd.isna(kpss_p) else np.nan,
            action   = action,
        ))
 
    return pd.DataFrame(records).set_index("feature")
 
 
def apply_transforms(
    X: pd.DataFrame,
    stat: pd.DataFrame,
    mode: str = "xgboost",   # "ols" | "xgboost"
) -> tuple[pd.DataFrame, dict]:
    """Apply the transforms from test_stationarity().
 
    mode="ols"     -> diff every flagged column (linear model needs it)
    mode="xgboost" -> diff only genuine non-stationary levels; trees tolerate
                      the inconclusive ones.
    """
    assert mode in ("ols", "xgboost")
    Xt  = X.copy()
    log = {}
 
    for col, row in stat.iterrows():
        if col not in Xt.columns:
            continue
 
        action = row["action"]
 
        if action == "DROP":
            Xt.drop(columns=[col], inplace=True)
            log[col] = "DROPPED"
 
        elif action == "DIFF":
            Xt[f"{col}_d1"] = Xt[col].diff()
            Xt.drop(columns=[col], inplace=True)
            log[col] = f"→ {col}_d1"
 
        elif action == "REVIEW" and mode == "ols":
            # OLS: be conservative, diff inconclusive features too
            Xt[f"{col}_d1"] = Xt[col].diff()
            Xt.drop(columns=[col], inplace=True)
            log[col] = f"→ {col}_d1 (inconclusive, ols-safe)"
 
        else:
            log[col] = "RAW"
 
    return Xt, log
 
 
def stationarity_report(stat: pd.DataFrame) -> None:
    groups = {
        "✅  stationary as-is":    stat[stat.action == "USE RAW"],
        "⚙️   will be differenced": stat[stat.action == "DIFF"],
        "🗑️   will be dropped":     stat[stat.action == "DROP"],
        "🔍  review (inconclusive)":stat[stat.action == "REVIEW"],
    }
    sep = "─" * 64
    print(f"\n{sep}\n  STATIONARITY REPORT  ({len(stat)} features)\n{sep}")
    for label, df in groups.items():
        print(f"\n{label}  ({len(df)})")
        if len(df):
            print(df[["verdict", "adf_p", "kpss_p"]].to_string())
    print(f"\n{sep}")
 
 
# ============================================================
 
# >> 2  SPEARMAN REDUNDANCY
 
# ============================================================
 
DOMAIN_PRIORITY = {
    # Price return structure (Section A/B2)
    "logret_lag1":        1,
    "logret_lag2":        1,
    "body_ratio_lag1":    2,
    "close_position_lag1":2,
    "wick_asymmetry":     3,
    "intraweek_range":    3,
 
    # Rolling return stats (Section B3) — std > mean when both present
    "logret_rmean4":      2,
    "logret_rmean13":     2,
    "logret_rstd4":       1,   # volatility is primary
    "logret_rstd13":      1,
    "vol_4w":             1,
    "vol_13w":            1,
    "price_vol_ratio":    2,   # derived from vol_4w / vol_13w
 
    # Momentum (Section B4)
    "momentum_4w":        1,
    "momentum_13w":       1,
    "price_roc_4w":       2,   # near-duplicate of momentum_4w in log space
 
    # Win rate (Section B6)
    "price_win_rate4":    1,
    "price_win_rate8":    2,
 
    # Units (Section C)
    "units_sum_lag1":     1,
    "units_sum_lag2":     1,
    "units_sum_lag3":     2,
    "units_sum_lag4":     2,
    "units_rmean4":       1,
    "units_rmean13":      2,   # long mean dominated by short mean
    "units_sum_ratio":    2,
    "units_roc_1w":       1,
    "units_roc_4w":       2,
 
    # Knife (Section D)
    "knife_mean_lag1":    1,
    "knife_rmean4":       1,
    "knife_rmean13":      2,
    "knife_rstd4":        1,
    "knife_rstd13":       2,
    "knife_roc_4w":       1,
    "knife_macd":         1,
    "knife_macd_hist":    2,   # derived from macd
    "knife_vol_4w":       1,
    "knife_vol_13w":      2,
    "knife_vol_ratio":    3,
 
    # ROI (Section E)
    "roi_mean_lag1":      1,
    "roi_mean_lag2":      2,
    "roi_rmean4":         1,
    "roi_rmean13":        2,
    "roi_roc_1w":         1,
    "roi_roc_4w":         2,
    "roi_mom_cross_4_13": 2,
    "roi_cross_up":       1,
    "roi_cross_down":     1,
    "roi_range_lag1":     2,
 
    # Players (Section F)
    "players_rmean4":     1,
    "players_rmean13":    2,
    "players_delta":      1,
    "players_roc_4w":     1,
    "players_roc_8w":     2,
    "players_mom_cross":  3,
 
    # Competitor ratios (Section G)
    "ratio_min_lag1":     1,
    "ratio_med_lag1":     1,
    "ratio_min_rmean4":   1,
    "ratio_med_rmean4":   1,
    "ratio_min_rmean13":  2,
    "ratio_med_rmean13":  2,
    "ratio_min_roc_1w":   1,
    "ratio_med_roc_1w":   1,
    "ratio_spread_lag1":  2,
 
    # Interactions (Section H)
    "price_vol_divergence": 2,
    "roi_x_players":        2,
    "knife_vs_price_mom":   1,
    "demand_supply_ratio":  1,
    "ratio_x_units_mom":    2,
 
    # Calendar (Section I)
    "is_major_week":  1,
    "is_steam_sale":  1,
    "month_sin":      1,
    "month_cos":      1,
}
 
 
_DEFAULT_PRIORITY = 5
 
 
def _priority(col: str) -> int:
    return DOMAIN_PRIORITY.get(col, _DEFAULT_PRIORITY)
 
 
def spearman_vs_target(
    X: pd.DataFrame,
    y: pd.Series,
    near_zero_thresh: float = 0.02,
) -> pd.DataFrame:
    """|Spearman rho| of each feature vs the target, sorted, with a near-zero flag."""
    records = []
    for col in X.columns:
        mask  = X[col].notna() & y.notna()
        rho, p = spearmanr(X.loc[mask, col], y.loc[mask])
        records.append(dict(
            feature   = col,
            rho       = round(rho, 4),
            abs_rho   = round(abs(rho), 4),
            p_value   = round(p, 4),
            near_zero = abs(rho) < near_zero_thresh,
            priority  = _priority(col),
        ))
    return (
        pd.DataFrame(records)
        .set_index("feature")
        .sort_values("abs_rho", ascending=False)
    )
 
 
def spearman_inter_feature(
    X: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Return (corr_matrix, pairs), where pairs are the (a, b, rho) above threshold."""
    corr_matrix, _ = spearmanr(X, nan_policy="omit")
    corr_df = pd.DataFrame(
        np.abs(corr_matrix),
        index=X.columns,
        columns=X.columns,
    )
 
    pairs = []
    cols  = X.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr_df.iloc[i, j]
            if rho > threshold:
                pairs.append((cols[i], cols[j], round(rho, 4)))
 
    pairs.sort(key=lambda x: -x[2])
    return corr_df, pairs
 
 
def resolve_redundant_pairs(
    pairs: list[tuple],
    target_corr: pd.DataFrame,
    near_zero_thresh: float = 0.02,
) -> dict:
    """For each redundant pair, pick one to drop; returns {dropped: reason}.
 
    Rule: drop the near-zero-signal feature, else the lower domain priority,
    else the weaker target correlation.
    """
    to_drop = {}
 
    for feat_a, feat_b, pair_rho in pairs:
        if feat_a in to_drop or feat_b in to_drop:
            continue  # already resolved upstream
 
        rho_a = target_corr.loc[feat_a, "abs_rho"] if feat_a in target_corr.index else 0
        rho_b = target_corr.loc[feat_b, "abs_rho"] if feat_b in target_corr.index else 0
        nz_a  = rho_a < near_zero_thresh
        nz_b  = rho_b < near_zero_thresh
        pr_a  = _priority(feat_a)
        pr_b  = _priority(feat_b)
 
        reason_prefix = f"pair rho={pair_rho} with {feat_b if True else feat_a}"
 
        if nz_a and not nz_b:
            to_drop[feat_a] = f"near-zero target signal; {reason_prefix}"
        elif nz_b and not nz_a:
            to_drop[feat_b] = f"near-zero target signal; {reason_prefix}"
        elif nz_a and nz_b:
            drop = feat_a if pr_a >= pr_b else feat_b
            to_drop[drop] = f"both near-zero; lower domain priority; {reason_prefix}"
        elif pr_a != pr_b:
            drop = feat_a if pr_a > pr_b else feat_b
            keep = feat_b if drop == feat_a else feat_a
            to_drop[drop] = f"lower domain priority (p={_priority(drop)}) vs {keep} (p={_priority(keep)}); pair rho={pair_rho}"
        else:
            # Same priority → drop the one with lower target signal
            drop = feat_a if rho_a <= rho_b else feat_b
            keep = feat_b if drop == feat_a else feat_a
            to_drop[drop] = f"equal priority; lower target rho ({rho_a:.3f} vs {rho_b:.3f}); pair rho={pair_rho}"
 
    return to_drop
 
 
def spearman_report(
    target_corr : pd.DataFrame,
    pairs       : list[tuple],
    to_drop     : dict,
) -> None:
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  SPEARMAN REPORT")
    print(sep)
 
    print(f"\n── Feature–target correlation (top 20 by |rho|) ──")
    print(target_corr.head(20)[["rho", "abs_rho", "p_value", "near_zero"]].to_string())
 
    nz = target_corr[target_corr["near_zero"]]
    print(f"\n── Near-zero signal features (|rho| < 0.02):  {len(nz)} ──")
    if len(nz):
        print(nz[["rho", "p_value"]].to_string())
 
    print(f"\n── Redundant pairs (|rho| > 0.85):  {len(pairs)} ──")
    for a, b, r in pairs:
        marker = "  DROP " + (a if a in to_drop else b) if (a in to_drop or b in to_drop) else ""
        print(f"  {a:40s}  {b:40s}  rho={r:.3f}{marker}")
 
    print(f"\n── Features to drop:  {len(to_drop)} ──")
    for feat, reason in sorted(to_drop.items()):
        print(f"  {feat:40s}  {reason}")
 
    print(f"\n{sep}")
    print(f"  Input features  : {len(target_corr)}")
    print(f"  Dropping        : {len(to_drop)}")
    print(f"  Surviving       : {len(target_corr) - len(to_drop)}")
    print(sep)
 
 
# ============================================================
 
# >> 3  VARIANCE INFLATION FACTOR
 
# ============================================================
 
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF for every feature in X (constant added internally), sorted descending."""
    X_c    = add_constant(X, has_constant="add")
    vif_df = pd.DataFrame({
        "feature": X_c.columns,
        "VIF"    : [
            variance_inflation_factor(X_c.values, i)
            for i in range(X_c.shape[1])
        ],
    }).set_index("feature")
 
    # Drop the constant row — not a real feature
    vif_df = vif_df.drop(index="const", errors="ignore")
    return vif_df.sort_values("VIF", ascending=False)
 
 
def iterative_vif_elimination(
    X         : pd.DataFrame,
    threshold : float = 10.0,
    verbose   : bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the highest-VIF feature repeatedly until all are below threshold.
 
    Returns (X_vif, vif_log, final_vif): the pruned matrix, a record of each
    removal, and the VIF table of the features that survived.
    """
    X_work   = X.copy()
    log_rows = []
    step     = 0
 
    if verbose:
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  ITERATIVE VIF ELIMINATION  (threshold = {threshold})")
        print(f"  Starting features: {X_work.shape[1]}")
        print(sep)
 
    while True:
        vif_df  = compute_vif(X_work)
        max_vif = vif_df["VIF"].iloc[0]
        worst   = vif_df.index[0]
 
        if max_vif <= threshold:
            if verbose:
                print(f"\n  Converged — all features below VIF {threshold}")
                print(f"  Features remaining: {X_work.shape[1]}")
            break
 
        step += 1
        log_rows.append(dict(
            step             = step,
            dropped_feature  = worst,
            vif_at_removal   = round(max_vif, 2),
            features_remaining = X_work.shape[1] - 1,
        ))
 
        if verbose:
            print(f"  Step {step:>2d} | drop: {worst:<45s} VIF={max_vif:.2f} "
                  f"| remaining: {X_work.shape[1] - 1}")
 
        X_work = X_work.drop(columns=[worst])
 
    # ── Final VIF table ───────────────────────────────────────────────────
    final_vif = compute_vif(X_work)
 
    if verbose:
        print(f"\n── Final VIF table ──")
        print(final_vif.round(2).to_string())
        print(f"\n── Removed features ({len(log_rows)}) ──")
        for row in log_rows:
            print(f"  step {row['step']:>2d} | {row['dropped_feature']:<45s} "
                  f"VIF={row['vif_at_removal']:.2f}")
 
    vif_log = pd.DataFrame(log_rows)
 
    return X_work, vif_log, final_vif
 
 
# ============================================================
 
# >> 4  PERMUTATION IMPORTANCE
 
# ============================================================
 
def permutation_importance_wf(
    models      : list,
    X_oos       : pd.DataFrame,   # full OOS feature matrix (one row per fold)
    y_oos       : pd.Series,      # true targets aligned to X_oos
    recent_folds: int  = 100,     # use only the most recent N models
    n_repeats   : int  = 10,      # permutation repeats per feature
    random_state: int  = 42,
) -> pd.DataFrame:
    """Permutation importance (drop in Spearman rho) averaged over recent folds.
 
    Positive = feature helps out of sample; negative = it hurts. Measured on
    OOS predictions, not training data. Sorted by mean_importance descending.
    """
    rng = np.random.default_rng(random_state)
 
    # Use the most recent folds — they reflect the current feature regime
    models_subset = models[-recent_folds:]
    n_models      = len(models_subset)
 
    # Align X_oos rows to the models subset
    # models[-recent_folds:] correspond to the last recent_folds test dates
    X_sub = X_oos.iloc[-recent_folds:].copy()
    y_sub = y_oos.iloc[-recent_folds:].copy()
 
    features = X_oos.columns.tolist()
 
    # ── Baseline: average predictions from all models in subset ──────────
    baseline_preds = np.zeros(len(X_sub))
    for i, model in enumerate(models_subset):
        baseline_preds += model.predict(X_sub.values)
    baseline_preds /= n_models
    baseline_rho, _ = spearmanr(y_sub.values, baseline_preds)
 
    # ── Per-feature permutation ───────────────────────────────────────────
    records = []
 
    for feat in features:
        feat_idx    = features.index(feat)
        repeat_rhos = []
 
        for _ in range(n_repeats):
            X_perm          = X_sub.values.copy()
            perm_idx        = rng.permutation(len(X_sub))
            X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
 
            perm_preds = np.zeros(len(X_sub))
            for model in models_subset:
                perm_preds += model.predict(X_perm)
            perm_preds /= n_models
 
            perm_rho, _ = spearmanr(y_sub.values, perm_preds)
            repeat_rhos.append(perm_rho)
 
        repeat_rhos  = np.array(repeat_rhos)
        importance   = baseline_rho - repeat_rhos   # drop when permuted
 
        records.append(dict(
            feature          = feat,
            baseline_rho     = round(baseline_rho, 4),
            mean_importance  = round(importance.mean(), 5),
            std_importance   = round(importance.std(),  5),
            min_importance   = round(importance.min(),  5),
            max_importance   = round(importance.max(),  5),
        ))
 
    return (
        pd.DataFrame(records)
        .set_index("feature")
        .sort_values("mean_importance", ascending=False)
    )
 
 
def permutation_report(perm_imp: pd.DataFrame, top_n: int = 20) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  PERMUTATION IMPORTANCE  (baseline rho = "
          f"{perm_imp['baseline_rho'].iloc[0]:.4f})")
    print(sep)
 
    pos = perm_imp[perm_imp["mean_importance"] > 0]
    neg = perm_imp[perm_imp["mean_importance"] <= 0]
 
    print(f"\n── Helpful features (permuting hurts):  {len(pos)} ──")
    print(pos.head(top_n)[
        ["mean_importance", "std_importance", "min_importance", "max_importance"]
    ].to_string())
 
    print(f"\n── Neutral / harmful features (permuting does not hurt):  {len(neg)} ──")
    print(neg[
        ["mean_importance", "std_importance", "min_importance", "max_importance"]
    ].to_string())
 
    print(f"\n── Stability check (std / |mean| ratio — lower is more stable) ──")
    stable = perm_imp[perm_imp["mean_importance"].abs() > 1e-5].copy()
    stable["stability_ratio"] = (
        stable["std_importance"] / stable["mean_importance"].abs()
    ).round(2)
    print(stable["stability_ratio"].sort_values().head(15).to_string())
    print(sep)
 
 
# ============================================================
# >> 5  WALD TEST  (linear model joint significance)
# ============================================================
 
# ============================================================
# >> 5  WALD TEST  (linear model joint significance)
# ============================================================
 
def wald_test(model, features=None):
    """Test whether coefficients are jointly zero (the "does the restricted
    model lose anything?" test).
 
    Beats reading individual p-values one at a time: correlated features can
    each look insignificant alone yet be jointly significant, because the
    single-coefficient tests each blame the others for the shared variance.
 
    model    : fitted statsmodels result (what _fit_mle returns). Works whether
               its coefficients are named (DataFrame X) or unnamed (numpy X).
    features : which coefficients to restrict to zero.
                 None            -> all of them except the intercept
                 list of names   -> those coefficients (needs a named model)
                 list of ints    -> those coefficient positions
    Returns a dict: chi2, df, p_value, significant.
    """
    params = model.params
    has_names = hasattr(params, "index")
    names = list(params.index) if has_names else [f"x{i}" for i in range(len(params))]
    k = len(params)
 
    # Resolve `features` to a list of integer positions. Doing this by position
    # rather than by name means the test works even when statsmodels labels the
    # coefficients x0, x1, ... (which happens when the model was fit on a numpy
    # array), where a string constraint would fail to parse.
    if features is None:
        # Everything except an intercept, detected by name if present,
        # otherwise assumed to be the first column (statsmodels convention).
        if has_names:
            positions = [i for i, n in enumerate(names)
                         if n not in ("const", "Intercept")]
        else:
            positions = list(range(1, k))   # skip the constant at column 0
        label = "all coefficients"
    else:
        positions = []
        for f in features:
            if isinstance(f, str):
                if f not in names:
                    raise KeyError(f"not in model: {f}")
                positions.append(names.index(f))
            else:
                positions.append(int(f))
        label = f"{len(positions)} coefficient(s)"
 
    # Restriction matrix R: one row per tested coefficient, a single 1 in that
    # coefficient's column. R @ beta = 0 says "these coefficients are zero".
    R = np.zeros((len(positions), k))
    for row, col in enumerate(positions):
        R[row, col] = 1.0
 
    result = model.wald_test(R, use_f=False, scalar=True)
 
    stat = float(np.squeeze(result.statistic))
    p = float(np.squeeze(result.pvalue))
    dof = len(positions)
    tested = [names[c] for c in positions]
 
    sep = "─" * 60
    print(sep)
    print(f"  WALD TEST — H0: {label} = 0")
    print(sep)
    print(f"  restricted : {dof} coefficient(s) set to zero")
    print(f"  chi2({dof}) : {stat:.4f}")
    print(f"  p-value    : {p:.4g}")
    verdict = ("reject H0 — the full model is significantly better"
               if p < 0.05 else
               "cannot reject H0 — dropping these loses nothing detectable")
    print(f"  verdict    : {verdict}")
    print(sep)
 
    return {"tested": tested, "df": dof, "chi2": stat,
            "p_value": p, "significant": p < 0.05}
 
 
def wald_nested_test(model, feature_names, borderline):
    """Compare a restricted model against the full model by testing whether a
    named group of borderline coefficients is JOINTLY zero.
 
    The setup for "should I keep these economically-motivated but statistically
    marginal variables?":
 
        full model       = every feature in the fit
        restricted model = full minus `borderline`
        H0               = all borderline coefficients are zero at once
 
    If the test rejects H0, the borderline group adds real signal on top of the
    significant variables and the FULL model is justified — even though each
    borderline variable looked weak on its own. Individually marginal
    coefficients (p ~ 0.05-0.15) can be jointly significant, because the
    single-coefficient p-values each blame the others for shared variance.
    If the test cannot reject, the group is jointly noise and the RESTRICTED
    model is preferred on parsimony.
 
    model         : fitted statsmodels result. Because the walk-forward fits on
                    a numpy array, its coefficients are unnamed (x0, x1, ...);
                    that is exactly why `feature_names` must be passed in.
    feature_names : the feature columns in the order the model saw them (i.e.
                    X_vif.columns). A constant is assumed at position 0, so
                    feature i sits at coefficient position i+1.
    borderline    : the feature names to test as a group. Names not present in
                    the fitted model are skipped with a note (they may have been
                    removed earlier by correlation or VIF pruning).
 
    Returns a dict: tested, missing, df, chi2, p_value, keep_full.
    """
    feature_names = list(feature_names)
 
    # feature i -> coefficient position i+1 (const at 0)
    name_to_pos = {name: i + 1 for i, name in enumerate(feature_names)}
 
    present = [b for b in borderline if b in name_to_pos]
    missing = [b for b in borderline if b not in name_to_pos]
 
    sep = "─" * 64
    print(sep)
    print("  NESTED WALD TEST — restricted vs full model")
    print(f"  H0: the {len(present)} borderline coefficient(s) are jointly zero")
    print(sep)
 
    if missing:
        print(f"  not in model (pruned earlier, skipped): {missing}")
    if not present:
        print("  none of the borderline features are in the model — nothing to test")
        print(sep)
        return {"tested": [], "missing": missing, "df": 0,
                "chi2": float("nan"), "p_value": float("nan"), "keep_full": False}
 
    k = len(model.params)
    positions = [name_to_pos[b] for b in present]
 
    R = np.zeros((len(positions), k))
    for row, col in enumerate(positions):
        R[row, col] = 1.0
 
    result = model.wald_test(R, use_f=False, scalar=True)
    stat = float(np.squeeze(result.statistic))
    p = float(np.squeeze(result.pvalue))
    dof = len(positions)
    keep_full = p < 0.05
 
    print(f"  borderline group ({dof}): {present}")
    print(f"  chi2({dof}) : {stat:.4f}")
    print(f"  p-value    : {p:.4g}")
    if keep_full:
        print("  verdict    : reject H0 — the group adds joint signal; KEEP the "
              "full model")
    else:
        print("  verdict    : cannot reject H0 — the group is jointly "
              "insignificant; the RESTRICTED model is preferred")
    print(sep)
 
    return {"tested": present, "missing": missing, "df": dof,
            "chi2": stat, "p_value": p, "keep_full": keep_full}
 
# ============================================================
# >> 6  SIGNIFICANCE
# ============================================================
 
def binomial_winrate_test(n_trades, win_rate, p_null=0.5):
    """One-sided binomial test that a directional hit rate beats a coin flip.
 
    A win rate of 55% over 80 trades is not evidence of anything; this is how
    you find that out.
    """
    correct = int(round(n_trades * win_rate))
    r = stats.binomtest(correct, n_trades, p=p_null, alternative="greater")
    ci = r.proportion_ci(0.95)
 
    print(f"  {correct}/{n_trades} correct  ({correct / n_trades:.1%})")
    print(f"  p-value : {r.pvalue:.4f}")
    print(f"  CI 95%  : [{ci.low:.3f}, {ci.high:.3f}]")
 
    return {"n": n_trades, "correct": correct, "p_value": r.pvalue,
            "ci_low": ci.low, "ci_high": ci.high,
            "significant": r.pvalue < 0.05}