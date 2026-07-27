
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
 
 
def walk_forward_regressor(df, final_features,
                            target="log_return",
                            min_train=104,
                            val_size=13,
                            window="rolling",
                            train_weeks=104,
                            xgb_params=None):
    """
    Walk-forward regressor producing out-of-fold predictions for the backtest.
 
    window = "rolling"   -> train on the most recent `train_weeks` only.
    window = "expanding" -> train on all history from the start.
 
    Returns an out-of-fold DataFrame with:
      - log_return  : actual return
      - pred        : predicted log_return
      - pred_sign   : sign of prediction  (+1 / -1)
      - pred_abs    : abs(pred) — conviction proxy
    No future data touches the model at any point.
    """
    if xgb_params is None:
        xgb_params = dict(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=3,
            learning_rate=0.015,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=8,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=42,
        )
 
    if window not in ("rolling", "expanding"):
        raise ValueError("window must be 'rolling' or 'expanding'")
 
    df    = df.dropna(subset=[target] + final_features).copy()
    n     = len(df)
    start = min_train
 
    records = []
 
    while start + val_size <= n:
        # Rolling trains on the most recent train_weeks; expanding from 0.
        train_start = max(0, start - train_weeks) if window == "rolling" else 0
        train = df.iloc[train_start:start]
        val   = df.iloc[start : start + val_size]
 
        X_tr, y_tr = train[final_features], train[target]
        X_va, y_va = val[final_features],   val[target]
 
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False)
 
        preds = model.predict(X_va)
 
        for date, actual, pred in zip(val.index, y_va.values, preds):
            records.append({
                "date"      : date,
                "log_return": actual,
                "pred"      : pred,
                "pred_sign" : np.sign(pred),
                "pred_abs"  : abs(pred),
            })
 
        start += val_size
 
    oof = (pd.DataFrame(records)
             .set_index("date")
             .sort_index())
 
    # Quick diagnostic
    dir_acc = np.mean(np.sign(oof["pred"]) == np.sign(oof["log_return"]))
    r2      = r2_score(oof["log_return"], oof["pred"])
    rmse    = np.sqrt(mean_squared_error(oof["log_return"], oof["pred"]))
 
    print(f"\n── Regressor OOF diagnostics ────────────────────────────")
    print(f"  Weeks predicted : {len(oof)}")
    print(f"  RMSE            : {rmse:.5f}")
    print(f"  R²              : {r2:.4f}")
    print(f"  Dir accuracy    : {dir_acc:.3f}")
    print(f"  Pred range      : [{oof['pred'].min():.4f}, {oof['pred'].max():.4f}]")
    print(f"  Pred std        : {oof['pred'].std():.5f}  "
          f"(actual std: {oof['log_return'].std():.5f})")
 
    return oof
 