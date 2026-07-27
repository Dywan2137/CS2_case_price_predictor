"""XGBoost under rolling-window walk-forward validation.

Per fold k:
    [----------- train -----------][-- val --][test]
    t=0 ..................... t=k-v  t=k-v..k   t=k

The validation window is used only for early stopping and never enters the
test fold.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
 
from config import MIN_TRAIN_WEEKS, STEP, VAL_WEEKS, XGB_PARAMS
 
 
def walk_forward_xgb(
    X            : pd.DataFrame,
    y            : pd.Series,
    min_train_weeks : int = 104,
    val_weeks       : int = 26,
    step            : int = 1,
    window          : str = "rolling",
    train_weeks     : int = 104,
    model_params : dict = None,
) -> tuple[pd.DataFrame, list]:
    """
    Walk-forward validation for XGBoost.
 
    window = "rolling"   -> train on the most recent `train_weeks` only, so the
                            model re-weights when the market changes regime.
    window = "expanding" -> train on all history from the start (can overfit the
                            early regime once the market shifts).
 
    Returns
    -------
    results_df : DataFrame indexed by test date with columns:
                   y_true, y_pred, fold, train_size, val_size
    models     : list of fitted XGBRegressor objects (one per fold)
                 — used later for permutation importance averaging
    """
    if model_params is None:
        model_params = {}
    if window not in ("rolling", "expanding"):
        raise ValueError("window must be 'rolling' or 'expanding'")
 
    idx    = X.index
    n      = len(idx)
    folds  = range(min_train_weeks, n, step)
 
    records = []
    models  = []
 
    for fold_num, test_pos in enumerate(folds):
        test_idx  = idx[test_pos]
        val_start = test_pos - val_weeks
        train_end = val_start          # exclusive
 
        # Rolling: start the training window `train_weeks` back from the
        # validation block. Expanding: always start at 0.
        train_start = max(0, val_start - train_weeks) if window == "rolling" else 0
 
        # Need at least min_train_weeks - val_weeks of actual training data
        if train_end < (min_train_weeks - val_weeks):
            continue
 
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_val   = X.iloc[val_start:test_pos]
        y_val   = y.iloc[val_start:test_pos]
        X_test  = X.iloc[[test_pos]]
        y_test  = y.iloc[[test_pos]]
 
        model = XGBRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set        = [(X_val, y_val)],
            verbose         = False,
        )
 
        y_pred = model.predict(X_test)[0]
 
        records.append(dict(
            date       = test_idx,
            y_true     = y_test.values[0],
            y_pred     = y_pred,
            fold       = fold_num,
            train_size = len(X_train),
            val_size   = len(X_val),
            best_iter  = model.best_iteration,
        ))
        models.append(model)
 
        if fold_num % 50 == 0:
            print(f"  fold {fold_num:>4d} | test={test_idx.date()} "
                  f"| train={len(X_train):>3d} | best_iter={model.best_iteration:>4d}")
 
    results_df = pd.DataFrame(records).set_index("date")
    return results_df, models
 
 
def walk_forward_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OOS metrics on the full results and on annual slices.
    Directional accuracy = fraction of weeks where sign(pred) == sign(true).
    """
    def _metrics(df, label):
        y_t = df["y_true"].values
        y_p = df["y_pred"].values
        rmse    = np.sqrt(mean_squared_error(y_t, y_p))
        mae     = mean_absolute_error(y_t, y_p)
        corr, _ = spearmanr(y_t, y_p)
        dir_acc = np.mean(np.sign(y_t) == np.sign(y_p))
        bias    = np.mean(y_p - y_t)   # mean signed error; matches MLE's Bias
        ic_mean = corr   # information coefficient
        return dict(
            period   = label,
            n_folds  = len(df),
            rmse     = round(rmse,    5),
            mae      = round(mae,     5),
            spearman = round(corr,    4),
            dir_acc  = round(dir_acc, 4),
            bias     = round(bias,    6),
        )
 
    rows = [_metrics(results, "full OOS")]
 
    # Annual breakdown
    results["year"] = results.index.year
    for yr, grp in results.groupby("year"):
        rows.append(_metrics(grp, str(yr)))
    results.drop(columns=["year"], inplace=True)
 
    return pd.DataFrame(rows).set_index("period")
 
 
def walk_forward_report(results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  WALK-FORWARD RESULTS")
    print(sep)
    print(f"\n── OOS metrics ──")
    print(metrics.to_string())
    print(f"\n── Early stopping (best_iter) distribution ──")
    bi = results["best_iter"]
    print(f"  mean={bi.mean():.0f}  median={bi.median():.0f}  "
          f"min={bi.min()}  max={bi.max()}")
    print(f"\n── Train size distribution ──")
    ts = results["train_size"]
    print(f"  first fold={ts.iloc[0]}  last fold={ts.iloc[-1]}  "
          f"mean={ts.mean():.0f}")
    print(sep)
