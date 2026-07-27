
import config as cfg
import data
import evaluate
import features
import model_linear
import model_xgb
import plots
import stats_tests
import walkforward_oof
from helpers import save_table
 
 
def main():
    # 1. data
    print("\n=== 1. DATA ===")
    daily = data.load_daily(cfg.DATASET)
    weekly = data.resample_weekly(daily)
 
    # 2. features
    print("\n=== 2. FEATURES ===")
    X, y, closes, feats = features.build_matrix(weekly)
 
    # 3. stationarity
    print("\n=== 3. STATIONARITY ===")
    stat = stats_tests.test_stationarity(X)
    stats_tests.stationarity_report(stat)
 
    # Two matrices from here on. The linear model needs stationary inputs for
    # its inference to mean anything; XGBoost splits don't care, and
    # differencing everything would throw away level information it can use.
    X_ols, _ = stats_tests.apply_transforms(X, stat, mode="ols")
    X_xgb, _ = stats_tests.apply_transforms(X, stat, mode="xgboost")
    X_ols, X_xgb = X_ols.dropna(), X_xgb.dropna()
 
    # ---------------------------------------------------- 4. redundancy prune
    print("\n=== 4. SPEARMAN PRUNING ===")
    y_xgb = y.loc[X_xgb.index]
 
    vs_target = stats_tests.spearman_vs_target(X_xgb, y_xgb)
    _, pairs = stats_tests.spearman_inter_feature(X_xgb, threshold=cfg.SPEARMAN_THRESHOLD)
    to_drop = stats_tests.resolve_redundant_pairs(pairs, vs_target)
    stats_tests.spearman_report(vs_target, pairs, to_drop)
 
    X_spear = X_xgb.drop(columns=list(to_drop), errors="ignore")
 
    # 5. permutation importance
    # A first walk-forward pass, used only to score features. Importance is
    # measured on out-of-sample predictions — measuring it on training data
    # would reward memorisation.
    print("\n=== 5. PERMUTATION IMPORTANCE ===")
    wf0, models0 = model_xgb.walk_forward_xgb(
        X_spear, y_xgb.loc[X_spear.index],
        window=cfg.WINDOW, train_weeks=cfg.TRAIN_WEEKS,
        model_params=cfg.XGB_PARAMS,
    )
    perm = stats_tests.permutation_importance_wf(
        models=models0,
        X_oos=X_spear.loc[wf0.index],
        y_oos=y_xgb.loc[wf0.index],
        recent_folds=cfg.PERM_FOLDS,
        random_state=cfg.SEED,
    )
    stats_tests.permutation_report(perm)
 
    keep = perm[perm["mean_importance"] > 0].index.tolist()
    X_final = X_spear[keep]
    print(f"Features: {X_xgb.shape[1]} -> {X_spear.shape[1]} -> {len(keep)}")
 
    # 6. xgboost
    print("\n=== 6. XGBOOST ===")
    xgb_results, xgb_models = model_xgb.walk_forward_xgb(
        X_final, y.loc[X_final.index],
        window=cfg.WINDOW, train_weeks=cfg.TRAIN_WEEKS,
        model_params=cfg.XGB_PARAMS,
    )
    xgb_met = model_xgb.walk_forward_metrics(xgb_results)
    model_xgb.walk_forward_report(xgb_results, xgb_met)
 
    # 7. linear model
    print("\n=== 7. LINEAR MODEL ===")
    shared = X_ols.columns.intersection(X_final.columns)
    X_vif, vif_log, _ = stats_tests.iterative_vif_elimination(
        X_ols[shared], threshold=cfg.VIF_THRESHOLD
    )
    print(f"VIF: {len(shared)} -> {X_vif.shape[1]} features")
 
    mle_results, mle_models = model_linear.walk_forward_mle(
        X_vif, y.loc[X_vif.index],
        window=cfg.WINDOW, train_weeks=cfg.TRAIN_WEEKS,
    )
    mle_met = model_linear.mle_metrics(mle_results)
    model_linear.mle_report(mle_results, mle_met)
    model_linear.compare_models(mle_results, xgb_results, mle_met, xgb_met)
 
    # Wald tests on the last fold's fitted model.
    if mle_models:
        # (a) overall: are the coefficients jointly significant vs intercept-only?
        stats_tests.wald_test(mle_models[-1])
        # (b) nested: do the borderline features add joint signal on top of the
        #     significant ones? Restricted (significant-only) vs full model.
        #     Feature order must match what the model saw = X_vif.columns.
        stats_tests.wald_nested_test(
            mle_models[-1],
            feature_names=list(X_vif.columns),
            borderline=cfg.BORDERLINE_FEATURES,
        )
 
    # 8. evaluation
    print("\n=== 8. EVALUATION ===")
    metrics_df, breakdown = evaluate.evaluate_vs_naive(xgb_results)
    evaluate.evaluation_report(metrics_df, breakdown)
 
    price_met, price_break, prices = evaluate.backtransform_and_evaluate(
        results=xgb_results, closes=closes
    )
    evaluate.price_evaluation_report(price_met, price_break)
 
    # Magnitude / attenuation: do predictions capture the size of moves, or
    # just hug the mean? Per-year std and range for each model.
    mag_xgb = evaluate.magnitude_analysis(xgb_results, label="XGBoost")
    evaluate.magnitude_report(mag_xgb, label="XGBoost")
    mag_mle = evaluate.magnitude_analysis(mle_results, label="Linear (MLE)")
    evaluate.magnitude_report(mag_mle, label="Linear (MLE)")
 
    # 9. backtest

    print("\n=== 9. BACKTEST ===")
    extra_cols = [c for c in ("log_return", "Close", "vol_4w", "vol_13w")
                  if c in feats.columns and c not in X_final.columns]
    df_clean = X_final.join(feats.loc[X_final.index, extra_cols])
    oof = walkforward_oof.walk_forward_regressor(
        df_clean, list(X_final.columns),
        window=cfg.WINDOW, train_weeks=cfg.TRAIN_WEEKS,
    )
    comparison = evaluate.run_full_comparison(
        oof, df_clean, list(X_final.columns), transaction_cost=cfg.TRANSACTION_COST
    )
 
    # 10. save
    print("\n=== 10. SAVING ===")
    for name, table in [
        ("walk_forward_xgb", xgb_results),
        ("walk_forward_mle", mle_results),
        ("permutation_importance", perm),
        ("stationarity", stat),
        ("vif_log", vif_log),
        ("evaluation_metrics", metrics_df),
        ("evaluation_annual", breakdown),
        ("price_predictions", prices),
        ("magnitude_xgb", mag_xgb),
        ("magnitude_mle", mag_mle),
    ]:
        save_table(table, name, cfg.OUTPUT_DIR)
 
    # 11. plots
    print("\n=== 11. PLOTS ===")

    mle_price_met, mle_price_break, mle_prices = evaluate.backtransform_and_evaluate(
        results=mle_results, closes=closes
    )
 
    import os
    fig_dir = os.path.join(cfg.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
 
    plots.plot_walkforward_yearly(
        prices, price_met, price_break,
        save_dir=os.path.join(fig_dir, "xgb_yearly"),
    )
    plots.plot_mle_yearly(
        mle_prices, mle_price_met, mle_price_break,
        save_dir=os.path.join(fig_dir, "mle_yearly"),
    )
    plots.plot_residuals_comparison(
        mle_results, xgb_results,
        save_path=os.path.join(fig_dir, "residuals_comparison.png"),
    )
    plots.mle_residual_diagnostics(
        mle_results, mle_models, list(X_vif.columns),
        save_path=os.path.join(fig_dir, "mle_diagnostics.png"),
    )
    plots.xgb_diagnostics(
        xgb_results, xgb_models, list(X_final.columns),
        save_path=os.path.join(fig_dir, "xgb_diagnostics.png"),
    )
 
    print("\nDone.")
    return locals()
 
 
if __name__ == "__main__":
    main()