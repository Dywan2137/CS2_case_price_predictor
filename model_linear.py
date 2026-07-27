"""
Linear regression model shouldnt even be used because of the non-linear nature of the data, but here we are, did that anyway.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
from config import MIN_TRAIN_WEEKS, STEP
 
 
def _fit_mle(
    X_train : np.ndarray,
    y_train : np.ndarray,
    mode    : str = "mle",
    alpha   : float = 1.0,
) -> object:
    """
    Fit a linear model via MLE.
 
    mode='mle'   → pure OLS/MLE via statsmodels (no penalty)
    mode='ridge' → Ridge MLE via statsmodels OLS on
                   augmented system (equivalent to MAP
                   estimate under Gaussian prior)
    """
    X_c = sm.add_constant(X_train, has_constant="add")
 
    if mode == "mle":
        model = sm.OLS(y_train, X_c).fit(method="pinv")
 
    elif mode == "ridge":
        # Ridge via augmented data: append sqrt(alpha)*I rows
        # This is algebraically equivalent to L2-penalized MLE
        n_feat = X_c.shape[1]
        X_aug  = np.vstack([X_c,  np.sqrt(alpha) * np.eye(n_feat)])
        y_aug  = np.concatenate([y_train, np.zeros(n_feat)])
        model  = sm.OLS(y_aug, X_aug).fit(method="pinv")
 
    else:
        raise ValueError("mode must be 'mle' or 'ridge'")
 
    return model
 
 
def walk_forward_mle(
    X               : pd.DataFrame,
    y               : pd.Series,
    min_train_weeks : int   = 104,
    val_weeks       : int   = 26,
    step            : int   = 1,
    window          : str   = "rolling",
    train_weeks     : int   = 104,
    mode            : str   = "mle",
    ridge_alpha     : float = 1.0,
) -> tuple[pd.DataFrame, list]:
    """
    Walk-forward validation for MLE linear regression.
 
    window = "rolling"   -> train on the most recent `train_weeks` only, so the
                            coefficients re-fit when the market changes regime.
    window = "expanding" -> train on all history from the start.
 
    Returns
    -------
    results_df : DataFrame indexed by test date with:
                   y_true, y_pred, fold, train_size,
                   log_likelihood, aic, bic, r2_train
    models     : list of fitted statsmodels results objects
    """
    if window not in ("rolling", "expanding"):
        raise ValueError("window must be 'rolling' or 'expanding'")
 
    idx   = X.index
    n     = len(idx)
    folds = range(min_train_weeks, n, step)
 
    records = []
    models  = []
 
    for fold_num, test_pos in enumerate(folds):
        val_start = test_pos - val_weeks
        train_end = val_start
 
        # Rolling starts the window `train_weeks` back; expanding starts at 0.
        train_start = max(0, train_end - train_weeks) if window == "rolling" else 0
 
        if train_end < (min_train_weeks - val_weeks):
            continue
 
        X_train = X.iloc[train_start:train_end].values
        y_train = y.iloc[train_start:train_end].values
        X_test  = X.iloc[[test_pos]].values
        y_test  = y.iloc[test_pos]
 
        # Guard: skip if training matrix is rank-deficient
        # (can happen on very early folds with many features)
        if X_train.shape[0] < X_train.shape[1] + 2:
            continue
 
        model  = _fit_mle(X_train, y_train, mode=mode, alpha=ridge_alpha)
        X_test_c = np.concatenate([[1.0], X_test.flatten()])
 
        # Ridge augmented model has extra rows — predict only on test
        # Use params directly to avoid shape mismatch
        y_pred   = float(X_test_c @ model.params[:len(X_test_c)])
 
        # MLE diagnostics — only valid for pure MLE mode
        # For ridge, log-likelihood is approximate
        ll  = float(model.llf)   if hasattr(model, "llf")  else np.nan
        aic = float(model.aic)   if hasattr(model, "aic")  else np.nan
        bic = float(model.bic)   if hasattr(model, "bic")  else np.nan
        r2t = float(model.rsquared) if hasattr(model, "rsquared") else np.nan
 
        records.append(dict(
            date           = idx[test_pos],
            y_true         = y_test,
            y_pred         = y_pred,
            fold           = fold_num,
            train_size     = len(X_train),
            log_likelihood = round(ll,  4),
            aic            = round(aic, 4),
            bic            = round(bic, 4),
            r2_train       = round(r2t, 4),
        ))
        models.append(model)
 
        if fold_num % 50 == 0:
            print(f"  fold {fold_num:>4d} | test={idx[test_pos].date()} "
                  f"| train={len(X_train):>3d} "
                  f"| LL={ll:>10.2f} "
                  f"| AIC={aic:>10.2f}")
 
    results_df = pd.DataFrame(records).set_index("date")
    return results_df, models
 
 
def mle_metrics(results: pd.DataFrame) -> pd.DataFrame:
    def _metrics(df, label):
        y_t = df["y_true"].values
        y_p = df["y_pred"].values
        nonzero  = np.abs(y_t) > 1e-8
        residuals = y_p - y_t
        rho, _   = spearmanr(y_t, y_p)
        return dict(
            period   = label,
            n_folds  = len(df),
            MAE      = round(mean_absolute_error(y_t, y_p),              5),
            RMSE     = round(np.sqrt(mean_squared_error(y_t, y_p)),      5),
            MAPE     = round(np.mean(np.abs(residuals[nonzero]
                             / y_t[nonzero])) * 100,                     4),
            R2       = round(r2_score(y_t, y_p),                         4),
            Dir_Acc  = round(np.mean(np.sign(y_t) == np.sign(y_p)),      4),
            Bias     = round(residuals.mean(),                            6),
            Spearman = round(rho,                                         4),
        )
 
    rows = [_metrics(results, "full OOS")]
    results["year"] = results.index.year
    for yr, grp in results.groupby("year"):
        rows.append(_metrics(grp, str(yr)))
    results.drop(columns=["year"], inplace=True)
 
    return pd.DataFrame(rows).set_index("period")
 
 
def mle_report(results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  WALK-FORWARD MLE LINEAR REGRESSION RESULTS")
    print(sep)
 
    print(f"\n── OOS prediction metrics ──")
    print(metrics.to_string())
 
    print(f"\n── MLE diagnostics per fold (summary) ──")
    print(f"  Log-likelihood : mean={results['log_likelihood'].mean():.2f}  "
          f"std={results['log_likelihood'].std():.2f}")
    print(f"  AIC            : mean={results['aic'].mean():.2f}  "
          f"std={results['aic'].std():.2f}")
    print(f"  BIC            : mean={results['bic'].mean():.2f}  "
          f"std={results['bic'].std():.2f}")
 
    print(f"\n── Train size distribution ──")
    ts = results["train_size"]
    print(f"  first={ts.iloc[0]}  last={ts.iloc[-1]}  mean={ts.mean():.0f}")
    print(sep)
 
 
def compare_models(
    mle_results  : pd.DataFrame,
    xgb_results  : pd.DataFrame,
    mle_metrics  : pd.DataFrame,
    xgb_metrics  : pd.DataFrame,
) -> None:
    """Side-by-side OOS metric comparison between MLE and XGBoost."""
 
    # Normalise column names to uppercase for both — handles the
    # walk_forward_metrics (lowercase) vs mle_metrics (uppercase) mismatch
    mle_m = mle_metrics.copy()
    xgb_m = xgb_metrics.copy()
    mle_m.columns = [c.upper() for c in mle_m.columns]
    xgb_m.columns = [c.upper() for c in xgb_m.columns]
 
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  MODEL COMPARISON — MLE LINEAR vs XGBoost  (full OOS)")
    print(sep)
 
    compare_metrics = [
        ("MAE",      "lower"),   # lower is better
        ("RMSE",     "lower"),
        ("DIR_ACC",  "higher"),  # higher is better
        ("BIAS",     "zero"),    # closer to zero is better
    ]
 
    for metric, direction in compare_metrics:
        if metric not in mle_m.columns or metric not in xgb_m.columns:
            print(f"  {metric:<12}: not available in both models — skipping")
            continue
 
        m_val = mle_m.loc["full OOS", metric]
        x_val = xgb_m.loc["full OOS", metric]
 
        if direction == "lower":
            winner = "MLE" if m_val < x_val else "XGB"
        elif direction == "higher":
            winner = "MLE" if m_val > x_val else "XGB"
        else:  # "zero" — the least biased model wins
            winner = "MLE" if abs(m_val) < abs(x_val) else "XGB"
 
        print(f"  {metric:<12}: MLE={m_val:.5f}   XGB={x_val:.5f}   → {winner}")
 
    # ── Annual breakdown comparison ───────────────────────────────────────
    print(f"\n── Annual Dir_Acc comparison ──")
    print(f"  {'Year':<8} {'MLE Dir_Acc':>12} {'XGB Dir_Acc':>12} {'Better':>8}")
 
    mle_yr = mle_m.drop(index="full OOS", errors="ignore")
    xgb_yr = xgb_m.drop(index="full OOS", errors="ignore")
    shared_years = mle_yr.index.intersection(xgb_yr.index)
 
    for yr in shared_years:
        if "DIR_ACC" not in mle_yr.columns or "DIR_ACC" not in xgb_yr.columns:
            break
        m = mle_yr.loc[yr, "DIR_ACC"]
        x = xgb_yr.loc[yr, "DIR_ACC"]
        better = "MLE" if m > x else "XGB" if x > m else "TIE"
        print(f"  {yr:<8} {m:>12.4f} {x:>12.4f} {better:>8}")
 
    print(sep)
 
 
def mle_coefficient_table(
    mle_models : list,
    feature_names : list,
) -> pd.DataFrame:
    """
    Extract coefficients from the last (largest sample) MLE fold.
    Returns a DataFrame sorted by |t-statistic| descending.
    """
    last_model = mle_models[-1]
    params     = last_model.params[1:]   # drop constant
    bse        = last_model.bse[1:]
    tvalues    = last_model.tvalues[1:]
    pvalues    = last_model.pvalues[1:]
    conf       = last_model.conf_int()[1:]
 
    coef_df = pd.DataFrame({
        "feature"  : feature_names,
        "coef"     : params,
        "std_err"  : bse,
        "t_stat"   : tvalues,
        "p_value"  : pvalues,
        "ci_low"   : conf[:, 0],
        "ci_high"  : conf[:, 1],
        "signif"   : ["***" if p < 0.001 else
                      "**"  if p < 0.01  else
                      "*"   if p < 0.05  else
                      "."   if p < 0.10  else ""
                      for p in pvalues],
    }).set_index("feature")
 
    return coef_df.reindex(
        coef_df["t_stat"].abs().sort_values(ascending=False).index
    )
