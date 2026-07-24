"""
All settings in one place.
"""

# data
DATASET = "data/operation_bravo_case_chart_data.csv"

END_DATE = "2025-10-22"        # data quality cut-off
DROP_POOL_DATE = "2025-12-17"  # rare drop pool removed; drops become 0 after

# target
TARGET = "log_return"

# Columns that do not enter the feature matrix.
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
SPEARMAN_THRESHOLD = 0.85  # features correlating above this are redundant
VIF_THRESHOLD = 10.0       # linear model only
PERM_FOLDS = 100           # folds used for permutation importance

# walk-forward
MIN_TRAIN_WEEKS = 104  # 2 years before the first test fold
VAL_WEEKS = 26         # 6 months for early stopping
STEP = 1               # test every week

# models
SEED = 42

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
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
TRANSACTION_COST = 0.002  # floatDB market fee 
PERIODS_PER_YEAR = 52
CONVICTION_QUANTILE = 0.70

# output
OUTPUT_DIR = "outputs"
