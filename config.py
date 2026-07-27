"""
All settings
"""

# data
DATASET = "data/operation_bravo_case_data.csv"
 
END_DATE = "2025-10-22"        # data cut-off
DROP_POOL_DATE = "2025-12-17"  # rare drop pool removed, drops become 0 after
 
# target
TARGET = "log_return"
 

EXCLUDE = {
    TARGET,
    # raw OHLC — the target is derived from Close
    "Open", "High", "Low", "Close",
    # raw weekly aggregations — superseded by their lagged versions
    "units_sum", "units_std", "units_min", "units_max",
    "knife_mean", "knife_low", "knife_high", "knife_open", "knife_close",
    "roi_mean", "roi_std", "roi_min", "roi_max",
    "players_mean", "players_std", "players_max",
    "comp_close", "comp_mean", "comp_open", "comp_std", "n_active_cases",
    # unlagged OHLC derivatives — contemporaneous, therefore leakage
    "intraweek_dir", "weekly_range", "body_ratio",
    "upper_wick", "lower_wick", "close_position",
    # intermediate computations
    "log_close",
    "ratio_to_min_mean", "ratio_to_min_max", "ratio_to_min_min", "ratio_to_min_std",
    "ratio_to_med_mean", "ratio_to_med_std", "comp_spread_mean", "comp_spread_max",
    # metadata
    "n_days",
}
 
# selection
SPEARMAN_THRESHOLD = 0.85  #
VIF_THRESHOLD = 7.0       # linear model only
PERM_FOLDS = 100           # folds used for permutation importance
 
# walk-forward
# Rolling window trains on a fixed number of weeks, then rolls forward.
# Expanding on the other hand keeps all history, which in a regime changing market leeds to model overfit.
# When testing the expanding window the model prediction was useless, so rolling is the baseline for now,
#untill i figire out what to do with the expanding window.

WINDOW = "rolling"     # "rolling" or "expanding"
TRAIN_WEEKS = 104      # rolling window length (2 years)
MIN_TRAIN_WEEKS = 104  # history required before the first test fold
VAL_WEEKS = 26         # 6 months, early stopping only, never in the test fold
STEP = 1               # test every week, then metrics are agregated later for all years
 
# models
SEED = 42
 
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "min_child_weight": 5,
    "early_stopping_rounds": 75,
    "eval_metric": "rmse",
    "random_state": SEED,
    "verbosity": 0,
}
 
# backtest
TRANSACTION_COST = 0.002  # float market fee, didnt use steam because of the 15% market fee which eats all the profits.
PERIODS_PER_YEAR = 52
CONVICTION_QUANTILE = 0.70
 
# output
OUTPUT_DIR = "outputs"