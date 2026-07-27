"""Evaluation and backtesting.

Two different claims are checked here, and they are not the same claim:
  - Is the model more accurate than a naive forecast?   (metrics)
  - Does trading on it make money after costs?          (backtest)
A model can win the first and lose the second.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
 
from config import PERIODS_PER_YEAR, TRANSACTION_COST
 
 
# ============================================================
 
# ACCURACY  (log-return space, then price space)
 
# ============================================================
 
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute all metrics for one prediction series."""
    residuals = y_pred - y_true
    nonzero   = np.abs(y_true) > 1e-8   # mask near-zero actuals for MAPE
 
    mae     = mean_absolute_error(y_true, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mape    = mean_absolute_percentage_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    bias    = residuals.mean()
    rho, _  = spearmanr(y_true, y_pred)
 
    return dict(
        predictor = label,
        MAE       = round(mae,     5),
        RMSE      = round(rmse,    5),
        MAPE      = round(mape,    3),
        R2        = round(r2,      4),
        Dir_Acc   = round(dir_acc, 4),
        Bias      = round(bias,    6),
        Spearman  = round(rho,     4),
    )
 
 
def evaluate_vs_naive(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model predictions against the naive random-walk benchmark.
 
    Parameters
    ----------
    results : walk-forward results DataFrame with columns y_true, y_pred,
              indexed by date (output of walk_forward_xgb)
 
    Returns
    -------
    metrics_df : wide DataFrame — one row per predictor, one col per metric
    breakdown  : annual breakdown for both predictors
    """
    df = results[["y_true", "y_pred"]].copy().sort_index()
 
    # Naive: previous week's actual return (random walk)
    df["y_naive"] = df["y_true"].shift(1)
    df = df.dropna()   # first row loses naive due to shift
 
    y_true  = df["y_true"].values
    y_model = df["y_pred"].values
    y_naive = df["y_naive"].values
 
    # ── Full-period metrics ───────────────────────────────────────────────
    rows = [
        compute_metrics(y_true, y_model, "Model"),
        compute_metrics(y_true, y_naive, "Naive (RW)"),
    ]
    metrics_df = pd.DataFrame(rows).set_index("predictor")
 
    # ── Annual breakdown ──────────────────────────────────────────────────
    df["year"] = df.index.year
    annual_rows = []
    for yr, grp in df.groupby("year"):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        yn = grp["y_naive"].values
        r  = compute_metrics(yt, yp, "Model");    r["year"] = yr; annual_rows.append(r)
        r  = compute_metrics(yt, yn, "Naive");    r["year"] = yr; annual_rows.append(r)
 
    breakdown = (
        pd.DataFrame(annual_rows)
        .set_index(["year", "predictor"])
        .sort_index()
    )
 
    return metrics_df, breakdown
 
 
def evaluation_report(metrics_df: pd.DataFrame, breakdown: pd.DataFrame) -> None:
    sep = "─" * 72
 
    print(f"\n{sep}")
    print(f"  FULL OOS EVALUATION — MODEL vs NAIVE RANDOM WALK")
    print(sep)
    print(metrics_df.to_string())
 
    print(f"\n{sep}")
    print(f"  ANNUAL BREAKDOWN")
    print(sep)
 
    years = breakdown.index.get_level_values("year").unique()
    for yr in years:
        print(f"\n  {yr}")
        print(
            breakdown.loc[yr][["MAE","RMSE","MAPE","R2","Dir_Acc","Bias","Spearman"]]
            .to_string()
        )
 
    # ── Skill score summary: how often does model beat naive? ────────────
    print(f"\n{sep}")
    print(f"  SKILL SCORES  (model metric / naive metric — <1 is better for errors)")
    print(sep)
    m = metrics_df.loc["Model"]
    n = metrics_df.loc["Naive (RW)"]
    print(f"  MAE  skill : {m['MAE']  / n['MAE']  :.3f}  ({'better' if m['MAE']  < n['MAE']  else 'worse'})")
    print(f"  RMSE skill : {m['RMSE'] / n['RMSE'] :.3f}  ({'better' if m['RMSE'] < n['RMSE'] else 'worse'})")
    print(f"  MAPE skill : {m['MAPE'] / n['MAPE'] :.3f}  ({'better' if m['MAPE'] < n['MAPE'] else 'worse'})")
    print(f"  Dir acc    : {m['Dir_Acc']:.4f} vs {n['Dir_Acc']:.4f}  "
          f"({'better' if m['Dir_Acc'] > n['Dir_Acc'] else 'worse'})")
    print(f"  Bias (model): {m['Bias']:+.6f}  "
          f"({'low' if abs(m['Bias']) < abs(n['Bias']) else 'higher than naive'})")
    print(sep)
 
 
def backtransform_and_evaluate(
    results : pd.DataFrame,
    closes  : pd.Series,
) -> tuple:
 
    df = results[["y_true", "y_pred"]].copy().sort_index()
    df["y_naive"]     = df["y_true"].shift(1)
    df                = df.dropna()
    prior_close       = closes.shift(1).reindex(df.index)
    df["prior_close"] = prior_close.values
 
    df["price_true"]  = df["prior_close"] * np.exp(df["y_true"])
    df["price_model"] = df["prior_close"] * np.exp(df["y_pred"])
    df["price_naive"] = df["prior_close"] * np.exp(df["y_naive"])
 
    # Sanity check
    actual_close = closes.reindex(df.index)
    recon_error  = (df["price_true"] - actual_close).abs()
    print(f"  Reconstruction — mean error: {recon_error.mean():.6f}  "
          f"max: {recon_error.max():.6f}  "
          f"({'OK' if recon_error.max() < 0.01 else 'CHECK'})")
 
    def _mape(true, pred):
        mask = np.abs(true) > 1e-8
        return np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
 
    def _direction_mask(price_true: np.ndarray, price_model: np.ndarray) -> np.ndarray:
        """
        Correct direction definition:
        Did the model predict the price would move in the right direction
        compared to the PREVIOUS week's actual price?
 
        direction_true  = sign(price_true[t]  - price_true[t-1])
        direction_pred  = sign(price_model[t] - price_true[t-1])
 
        Both use price_true[t-1] as the baseline — the model is judged on
        whether it correctly predicted up or down from where we actually were,
        not whether the predicted level was above or below the actual level.
        """
        true_direction  = np.sign(price_true[1:]  - price_true[:-1])
        pred_direction  = np.sign(price_model[1:] - price_true[:-1])
 
        # Pad first entry with False — no prior week available
        match = np.concatenate([[False], true_direction == pred_direction])
        return match
 
    def _price_metrics(true, pred, log_true, log_pred, label):
        residuals = pred - true
 
        # Direction: did prediction move the right way vs prior actual price?
        true_dir = np.sign(true[1:]  - true[:-1])
        pred_dir = np.sign(pred[1:]  - true[:-1])   # baseline = prior ACTUAL
        dir_acc  = np.mean(true_dir == pred_dir)
 
        return dict(
            predictor  = label,
            MAE_price  = round(mean_absolute_error(true, pred),         4),
            RMSE_price = round(np.sqrt(mean_squared_error(true, pred)), 4),
            MAPE_price = round(_mape(true, pred),                       4),
            R2_price   = round(r2_score(true, pred),                    4),
            Dir_Acc    = round(dir_acc,                                  4),
            Bias_price = round(residuals.mean(),                         4),
        )
 
    pt = df["price_true"].values
    pm = df["price_model"].values
    pn = df["price_naive"].values
    lt = df["y_true"].values
    lp = df["y_pred"].values
    ln = df["y_naive"].values
 
    metrics_price = pd.DataFrame([
        _price_metrics(pt, pm, lt, lp, "Model"),
        _price_metrics(pt, pn, lt, ln, "Naive (RW)"),
    ]).set_index("predictor")
 
    # Annual breakdown
    df["year"]  = df.index.year
    annual_rows = []
    for yr, grp in df.groupby("year"):
        t   = grp["price_true"].values
        m   = grp["price_model"].values
        n   = grp["price_naive"].values
        lt_ = grp["y_true"].values
        lp_ = grp["y_pred"].values
        ln_ = grp["y_naive"].values
        r = _price_metrics(t, m, lt_, lp_, "Model"); r["year"] = yr; annual_rows.append(r)
        r = _price_metrics(t, n, lt_, ln_, "Naive"); r["year"] = yr; annual_rows.append(r)
 
    breakdown_price = (
        pd.DataFrame(annual_rows)
        .set_index(["year", "predictor"])
        .sort_index()
    )
 
    return metrics_price, breakdown_price, df
 
 
def price_evaluation_report(
    metrics_price   : pd.DataFrame,
    breakdown_price : pd.DataFrame,
) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  PRICE-LEVEL EVALUATION — MODEL vs NAIVE RANDOM WALK")
    print(sep)
    print(metrics_price.to_string())
 
    print(f"\n{sep}")
    print(f"  ANNUAL PRICE-LEVEL BREAKDOWN")
    print(sep)
    for yr in breakdown_price.index.get_level_values("year").unique():
        print(f"\n  {yr}")
        print(breakdown_price.loc[yr].to_string())
 
    print(f"\n{sep}")
    print(f"  SKILL SCORES ON PRICE LEVELS")
    print(sep)
    m = metrics_price.loc["Model"]
    n = metrics_price.loc["Naive (RW)"]
    for metric in ["MAE_price", "RMSE_price", "MAPE_price"]:
        ratio = m[metric] / n[metric]
        print(f"  {metric:<12}: {ratio:.3f}  "
              f"({'better' if ratio < 1 else 'worse'})  "
              f"model={m[metric]:.4f}  naive={n[metric]:.4f}")
    print(f"  Dir_Acc     : model={m['Dir_Acc']:.4f}  "
          f"naive={n['Dir_Acc']:.4f}  "
          f"({'better' if m['Dir_Acc'] > n['Dir_Acc'] else 'worse'})")
    print(f"  Bias_price  : model={m['Bias_price']:+.4f}  "
          f"naive={n['Bias_price']:+.4f}")
    print(sep)
 
 
def magnitude_analysis(results, label="model"):
    """Per-year std and range of predicted vs actual log returns.
 
    The question this answers: does the model actually predict the *size* of
    weekly moves, or does it play safe by hugging the mean? A model that only
    ever predicts small returns can still score a decent direction accuracy
    while being useless for anything that depends on magnitude.
 
    The tell is the attenuation ratio, std(pred) / std(true):
        ~1.0  predictions are as volatile as reality
        <1.0  predictions are compressed toward the mean (the usual outcome)
        ~0.0  predictions are essentially flat
 
    Works unchanged for XGBoost or the linear model, since both return a
    date-indexed frame with `y_true` and `y_pred` columns.
 
    Returns
    -------
    A DataFrame with one row per calendar year plus a final "ALL" row, columns:
        n, std_true, std_pred, attenuation,
        range_true, range_pred, min_true, max_true, min_pred, max_pred
    """
    df = results[["y_true", "y_pred"]].dropna().sort_index()
 
    def row(g):
        t, p = g["y_true"].values, g["y_pred"].values
        std_t, std_p = t.std(), p.std()
        return pd.Series({
            "n":           len(g),
            "std_true":    std_t,
            "std_pred":    std_p,
            # guard the ratio: a constant-true year would divide by zero
            "attenuation": std_p / std_t if std_t > 1e-12 else np.nan,
            "range_true":  t.max() - t.min(),
            "range_pred":  p.max() - p.min(),
            "min_true":    t.min(), "max_true": t.max(),
            "min_pred":    p.min(), "max_pred": p.max(),
        })
 
    # Per-year rows, then one aggregated row over the whole sample.
    by_year = df.groupby(df.index.year).apply(row)
    by_year.index.name = "year"
 
    overall = row(df).to_frame().T
    overall.index = ["ALL"]
 
    table = pd.concat([by_year, overall])
    table["n"] = table["n"].astype(int)
    return table
 
 
def magnitude_report(table, label="model"):
    """Print the per-year magnitude table in the house style."""
    sep = "─" * 76
    print(sep)
    print(f"  MAGNITUDE ANALYSIS — {label}")
    print(f"  attenuation = std(pred)/std(true);  <1 means predictions hug the mean")
    print(sep)
    print(f"  {'year':>5} {'n':>4} {'std_true':>9} {'std_pred':>9} "
          f"{'atten':>6} {'rng_true':>9} {'rng_pred':>9}")
    print(f"  {'─'*5} {'─'*4} {'─'*9} {'─'*9} {'─'*6} {'─'*9} {'─'*9}")
    for idx, r in table.iterrows():
        marker = "  <<<" if idx == "ALL" else ""
        print(f"  {str(idx):>5} {int(r['n']):>4} "
              f"{r['std_true']:>9.5f} {r['std_pred']:>9.5f} "
              f"{r['attenuation']:>6.3f} "
              f"{r['range_true']:>9.4f} {r['range_pred']:>9.4f}{marker}")
    print(sep)
 
 
# ============================================================
 
# SIGNALS  (continuous prediction -> position in {-1, 0, 1})
 
# ============================================================
 
def build_regressor_signals(oof, df_clean,
                              mag_threshold_pct=0.60,
                              vol_col="vol_4w"):
    """
    Build trading signals from regressor OOF predictions.
 
    mag_threshold_pct : percentile of |pred| above which
                        we consider the model "confident"
    """
    returns  = oof["log_return"]
    pred     = oof["pred"]
    pred_abs = oof["pred_abs"]
 
    # Conviction threshold — top 40% of predicted magnitudes
    mag_thresh = pred_abs.quantile(mag_threshold_pct)
 
    signals = {}
 
    # ── E: long/short on predicted sign (same as classifier) ──
    signals["E_reg_long_short"] = np.sign(pred).rename("signal")
 
    # ── F: long only on positive prediction ───────────────────
    signals["F_reg_long_only"] = pd.Series(
        np.where(pred > 0, 1, 0), index=pred.index
    )
 
    # ── G: high-conviction long/short ─────────────────────────
    # Only trade when |pred| is in top (1-mag_threshold_pct)%
    signals["G_conv_long_short"] = pd.Series(
        np.where(pred_abs >= mag_thresh, np.sign(pred), 0),
        index=pred.index
    )
 
    # ── H: high-conviction long only ──────────────────────────
    signals["H_conv_long_only"] = pd.Series(
        np.where((pred > 0) & (pred_abs >= mag_thresh), 1, 0),
        index=pred.index
    )
 
    # ── I: scaled position by predicted magnitude ─────────────
    # Normalise pred to [-1, +1] range using rolling percentile
    # so position size reflects relative conviction
    pred_norm = pred / (pred_abs.rolling(52, min_periods=13).quantile(0.95) + 1e-9)
    pred_norm = pred_norm.clip(-1, 1)
    signals["I_scaled_position"] = pred_norm
 
    # ── J: vol-adjusted filter ────────────────────────────────
    # Skip high-volatility weeks regardless of model signal
    # (model is least reliable when vol is highest)
    vol = df_clean[vol_col].reindex(oof.index)
    vol_ok = vol < vol.quantile(0.70)    # trade only calmer 70% of weeks
 
    signals["J_conv_vol_filter"] = pd.Series(
        np.where(
            (pred > 0) & (pred_abs >= mag_thresh) & vol_ok, 1,
            np.where(
                (pred < 0) & (pred_abs >= mag_thresh) & vol_ok, -1, 0
            )
        ),
        index=pred.index
    )
    # Add this to build_regressor_signals()
    vol    = df_clean[vol_col].reindex(oof.index)
    vol_ok = vol < vol.quantile(0.70)
 
    signals["K_best_combined"] = pd.Series(
        np.where(
            (pred > 0) & (pred_abs >= mag_thresh) & vol_ok, 1, 0
        ),
        index=pred.index
    )
    # Summary of signal coverage
    print(f"\n── Signal coverage ──────────────────────────────────────")
    print(f"  Magnitude threshold (p{mag_threshold_pct*100:.0f}): {mag_thresh:.5f}")
    print(f"  {'Signal':<25} {'Long':>6} {'Short':>6} {'Flat':>6}")
    print(f"  {'-'*45}")
    for k, sig in signals.items():
        n_long  = (sig ==  1).sum()
        n_short = (sig == -1).sum()
        n_flat  = (sig ==  0).sum()
        print(f"  {k:<25} {n_long:>6} {n_short:>6} {n_flat:>6}")
 
    return signals, returns
 
 
def regime_aware_signal(oof, df_clean, mag_threshold_pct=0.70):
    """
    Only activate the model during stable regimes.
    Fall back to always-long during detected structural breaks.
    
    Regime detection: if rolling 13-week volatility is in the
    top 25% historically, the market is in a stress regime —
    sit flat rather than applying the model filter.
    """
    pred     = oof["pred"]
    pred_abs = oof["pred_abs"]
    returns  = oof["log_return"]
    
    mag_thresh = pred_abs.quantile(mag_threshold_pct)
    
    # Rolling vol of actual returns (use realized, not predicted)
    realized_vol = returns.rolling(13).std()
    high_vol_regime = realized_vol > realized_vol.quantile(0.75)
    
    signal = pd.Series(0, index=oof.index)
    
    # In normal regimes: apply H filter
    normal = ~high_vol_regime
    signal[normal & (pred > 0) & (pred_abs >= mag_thresh)] = 1
    
    # In stress regimes: go flat (don't fight the model OR go long)
    # Alternatively: always_long in stress (uncomment below)
    # signal[high_vol_regime] = 1
    
    return signal
 
 
def naive_signals(returns):
    """Three naive baselines to beat."""
    signals = {}
    
    # 1. Always long — buy and hold
    signals["always_long"] = pd.Series(1, index=returns.index)
    
    # 2. Momentum — go long if last week was up
    signals["lag1_momentum"] = np.sign(returns.shift(1)).fillna(0)
    
    # 3. 4-week momentum — long if 4-week return positive
    signals["mom_4w"] = np.sign(
        returns.shift(1).rolling(4).sum()
    ).fillna(0)
    
    return signals
 
 
# ============================================================
 
# BACKTEST
 
# ============================================================
 
def backtest_strategy(returns, signal,
                       transaction_cost=0.002,
                       name="Strategy"):
    signal         = signal.reindex(returns.index).fillna(0)
    pos_changes    = signal.diff().abs().fillna(0)
    costs          = pos_changes * transaction_cost
    strat_ret      = signal.shift(1) * returns - costs
    cum_ret        = (1 + strat_ret).cumprod()
    cum_bh         = (1 + returns).cumprod()
    n_years        = len(returns) / 52
    total_ret      = cum_ret.iloc[-1] - 1
    ann_ret        = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol        = strat_ret.std() * np.sqrt(52)
    sharpe         = ann_ret / (ann_vol + 1e-9)
    rolling_max    = cum_ret.cummax()
    max_dd         = ((cum_ret - rolling_max) / rolling_max).min()
    active         = signal.shift(1) != 0
    win_rate       = (strat_ret[active] > 0).mean() if active.sum() > 0 else 0
    gains          = strat_ret[strat_ret > 0].sum()
    losses         = strat_ret[strat_ret < 0].abs().sum()
    pf             = gains / (losses + 1e-9)
    n_trades       = pos_changes[pos_changes > 0].count()
 
    print(f"\n── {name} ──────────────────────────────────────")
    print(f"  Total return    : {total_ret:+.1%}")
    print(f"  B&H return      : {cum_bh.iloc[-1]-1:+.1%}")
    print(f"  Ann. return     : {ann_ret:+.1%}")
    print(f"  Ann. volatility : {ann_vol:.1%}")
    print(f"  Sharpe ratio    : {sharpe:.3f}")
    print(f"  Max drawdown    : {max_dd:.1%}")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Trades          : {n_trades}")
 
    return dict(name=name, total_ret=total_ret, ann_ret=ann_ret,
                sharpe=sharpe, max_dd=max_dd, win_rate=win_rate,
                profit_factor=pf, n_trades=n_trades,
                cum_returns=cum_ret, strat_returns=strat_ret)
 
 
def run_full_comparison(oof, df_clean, final_features,
                         mag_threshold_pct=0.60,
                         transaction_cost=0.002):
    """
    Build all regressor signals, all naive baselines,
    run backtests, and plot equity curves + summary table.
    """
    signals, returns = build_regressor_signals(
        oof, df_clean,
        mag_threshold_pct=mag_threshold_pct
    )
 
    # ── Naive baselines ───────────────────────────────────────
    naive_signals = {
        "naive_always_long"   : pd.Series(1,  index=returns.index),
        "naive_lag1_mom"      : np.sign(returns.shift(1)).fillna(0),
        "naive_mom_4w"        : np.sign(
            returns.shift(1).rolling(4).sum()
        ).fillna(0),
    }
    signals.update(naive_signals)
 
    # ── Run all backtests ─────────────────────────────────────
    results = {}
    for key, sig in signals.items():
        results[key] = backtest_strategy(
            returns, sig,
            transaction_cost=transaction_cost,
            name=key
        )
 
    # ── Summary table ─────────────────────────────────────────
    summary = pd.DataFrame([{
        "strategy"     : k,
        "total_ret"    : f"{v['total_ret']:+.1%}",
        "sharpe"       : f"{v['sharpe']:.3f}",
        "max_dd"       : f"{v['max_dd']:.1%}",
        "win_rate"     : f"{v['win_rate']:.1%}",
        "profit_factor": f"{v['profit_factor']:.2f}",
        "n_trades"     : v["n_trades"],
        "is_model"     : not k.startswith("naive"),
    } for k, v in results.items()])
 
    print(f"\n── Strategy comparison summary ───────────────────────────")
    print(summary.drop(columns="is_model").to_string(index=False))
 
    model_strats  = summary[summary["is_model"]]
    best_key      = model_strats.loc[
        model_strats["sharpe"].astype(float).idxmax(), "strategy"
    ]
    best_naive    = "naive_always_long"   # hardest baseline to beat
 
    best_sharpe   = results[best_key]["sharpe"]
    naive_sharpe  = results[best_naive]["sharpe"]
    beats          = best_sharpe > naive_sharpe
 
    print(f"\n── Verdict ───────────────────────────────────────────────")
    print(f"  Best model strategy : {best_key}")
    print(f"  Model Sharpe        : {best_sharpe:.3f}")
    print(f"  Naive Sharpe        : {naive_sharpe:.3f}  (always long)")
    print(f"  Beats naive?        : {'YES ✓' if beats else 'NO ✗'}")
 
    # ── Equity curve plot ─────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
 
    palette = {
        "E_reg_long_short"  : ("#534AB7", "-",  2.0),
        "F_reg_long_only"   : ("#1D9E75", "-",  2.0),
        "G_conv_long_short" : ("#E24B4A", "-",  2.0),
        "H_conv_long_only"  : ("#EF9F27", "-",  2.5),
        "I_scaled_position" : ("#D4537E", "-",  2.0),
        "J_conv_vol_filter" : ("#0F6E56", "-",  2.0),
        "naive_always_long" : ("#888780", "--", 1.5),
        "naive_lag1_mom"    : ("#B4B2A9", "--", 1.0),
        "naive_mom_4w"      : ("#D3D1C7", "--", 1.0),
    }
 
    for key, res in results.items():
        c, ls, lw = palette.get(key, ("gray", "-", 1))
        axes[0].plot(res["cum_returns"], label=key,
                      color=c, linestyle=ls, linewidth=lw)
 
    axes[0].axhline(1, color="black", linewidth=0.5, linestyle=":")
    axes[0].set_title("Equity curves — regressor strategies vs naive")
    axes[0].set_ylabel("Portfolio value (start=1.0)")
    axes[0].legend(fontsize=8, ncol=2)
 
    # Drawdown of best model vs always-long
    for key, color in [(best_key, "#534AB7"), ("naive_always_long", "#888780")]:
        cr  = results[key]["cum_returns"]
        dd  = (cr - cr.cummax()) / cr.cummax()
        axes[1].fill_between(dd.index, dd, 0,
                               alpha=0.45, color=color, label=key)
 
    axes[1].set_title("Drawdown — best model vs always-long")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
 
    return results, summary
