"""
Weekly feature engineering.

This is the part that took the longest to get somewhat right (disclaimer it still aint right). 
Still the features are good enough to grate horseradish not to actually predict anything.
For now those are the features that are used in the linear model and XGBoost model.
this will change however, as the XGBoost performace is questionable and linear model wasnt even a good aproach to begin with.


Every predictor is lagged by at least one week to avoid data leackage.
The eventual leackage is what tests/test_leakage.py checks.
"""

import numpy as np
import pandas as pd

from config import EXCLUDE, TARGET, DATASET
from data import resample_weekly, load_daily, add_price_ratios


def engineer_weekly(df):
    d = df.copy().sort_index()

    # ── Target ────────────────────────────────────────────────────────────
    d["log_return"] = np.log(d["Close"] / d["Close"].shift(1))
    d["log_close"]  = np.log(d["Close"])

    # ======================================================
    # SECTION A — OHLC CANDLESTICK SIGNALS
    # Kept: body_ratio, close_position, wick_asymmetry,
    #       intraweek_range (all direct price structure signals)
    # Dropped: upper_wick_lag1, lower_wick_lag1 individually
    #          (subsumed by wick_asymmetry which captures the
    #          imbalance more directly), upper/lower_wick_rmean4
    #          (second-order rolling mean of a candlestick ratio
    #          adds marginal signal), intraweek_gap (weekly
    #          open-to-prior-close gap is mostly noise)
    # ======================================================
    hl = d["High"] - d["Low"] + 1e-9

    body_ratio_raw = (d["Close"] - d["Open"]) / hl
    lower_wick_raw = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / hl
    upper_wick_raw = (d["High"] - d[["Open", "Close"]].max(axis=1)) / hl
    close_pos_raw  = (d["Close"] - d["Low"]) / hl
    intraweek_range_raw = (d["High"] - d["Low"]) / d["Close"]

    d["body_ratio_lag1"]     = body_ratio_raw.shift(1)
    d["close_position_lag1"] = close_pos_raw.shift(1)
    d["wick_asymmetry"]      = lower_wick_raw.shift(1) - upper_wick_raw.shift(1)
    d["intraweek_range"]     = intraweek_range_raw.shift(1)

    # ======================================================
    # SECTION B — KNIFE (BYPRODUCT) CANDLESTICK SIGNALS
    # Kept: body, close_position, wick_asymmetry, range
    #       (mirrors item OHLC logic for the byproduct)
    # Dropped: knife_upper/lower_wick_lag1 individually
    #          (subsumed by knife_wick_asym), knife_gap_lag1
    #          (noise at weekly freq), rolling wick/body means
    #          (second-order signals)
    # ======================================================
    k_hl = d["knife_high"] - d["knife_low"] + 1e-9

    knife_body_raw      = (d["knife_close"] - d["knife_open"]) / k_hl
    knife_lower_raw     = (d[["knife_open", "knife_close"]].min(axis=1) - d["knife_low"]) / k_hl
    knife_upper_raw     = (d["knife_high"] - d[["knife_open", "knife_close"]].max(axis=1)) / k_hl
    knife_close_pos_raw = (d["knife_close"] - d["knife_low"]) / k_hl
    knife_range_raw     = (d["knife_high"] - d["knife_low"]) / (d["knife_close"] + 1e-9)

    d["knife_body_lag1"]      = knife_body_raw.shift(1)
    d["knife_close_pos_lag1"] = knife_close_pos_raw.shift(1)
    d["knife_wick_asym_lag1"] = knife_lower_raw.shift(1) - knife_upper_raw.shift(1)
    d["knife_range_lag1"]     = knife_range_raw.shift(1)

    # ======================================================
    # SECTION B2 — PRICE LOG-RETURN AR LAGS
    # Kept: lag1, lag2 (standard AR structure, PACF-confirmed)
    #       lag6, lag7 only if your PACF showed significance
    #       there — remove them here if PACF did not spike
    # NOTE: comment out logret_lag6/7 if PACF was flat at
    #       those lags — do not let AIC override PACF
    # ======================================================
    for lag in [1, 2]:                        # core AR lags
        d[f"logret_lag{lag}"] = d["log_return"].shift(lag)

    # Uncomment below ONLY if PACF showed significance at lag 6/7
    # for lag in [6, 7]:
    #     d[f"logret_lag{lag}"] = d["log_return"].shift(lag)

    # ======================================================
    # SECTION B3 — ROLLING STATS ON PRICE RETURN
    # Kept: mean (trend baseline), std (volatility regime)
    #       for both short (4w) and long (13w) windows
    # Dropped: rmin, rmax, rmed — these are dominated by the
    #          mean and std and add collinearity without
    #          clear marginal economic content
    # ======================================================
    base = d["log_return"].shift(1)

    for w in [4, 13]:
        d[f"logret_rmean{w}"] = base.rolling(w).mean()
        d[f"logret_rstd{w}"]  = base.rolling(w).std()

    # ======================================================
    # SECTION B4 — MOMENTUM
    # Kept: 4w and 13w momentum (medium/long horizon)
    #       price_roc_4w (rate of change confirmation)
    # Dropped: momentum_2w (too short, noisy), momentum_8w
    #          (redundant between 4w and 13w),
    #          price_roc_2w/8w/13w (duplicates of momentum
    #          at same horizons — momentum_Xw = price_roc_Xw
    #          in log-return space)
    # ======================================================
    d["momentum_4w"]  = base.rolling(4).sum()
    d["momentum_13w"] = base.rolling(13).sum()
    d["price_roc_4w"] = d["log_close"].shift(1) - d["log_close"].shift(5)

    # ======================================================
    # SECTION B5 — VOLATILITY
    # Kept: vol_4w, vol_13w, price_vol_ratio
    #       (volatility clustering is a core stylized fact;
    #        ratio captures regime shifts in vol)
    # Dropped: vol_8w (redundant between 4w and 13w)
    # ======================================================
    d["vol_4w"]          = base.rolling(4).std()
    d["vol_13w"]         = base.rolling(13).std()
    d["price_vol_ratio"] = d["vol_4w"] / (d["vol_13w"] + 1e-9)

    # ======================================================
    # SECTION B6 — PRICE DIRECTION-SPECIFIC
    # Kept: price_win_rate4, price_win_rate8
    #       (fraction of up-weeks is interpretable and clean)
    # Dropped: price_down/up_semi (redundant given vol + win
    #          rate), price_semi_ratio (noisy ratio of noisy
    #          semi-deviations), price_win_rate13 (duplicates
    #          momentum_13w direction at same horizon),
    #          price_skew, price_kurt (too noisy at weekly
    #          freq with 8-13 obs windows),
    #          price_gl_ratio (duplicates win_rate + vol)
    # ======================================================
    for w in [4, 8]:
        d[f"price_win_rate{w}"] = base.rolling(w).apply(
            lambda x: (x > 0).mean(), raw=True
        )

    # ======================================================
    # SECTION C — UNITS SOLD
    # Kept: direct lags 1–4 (AR structure on volume),
    #       rolling mean at 4w and 13w (demand baseline),
    #       units_sum_ratio (current vs recent avg — surge),
    #       roc_1w and roc_4w (volume momentum),
    #       demand_supply_ratio (supply/demand framework)
    # Dropped: lag8, lag10 (too distant, no clear mechanism),
    #          units_max/min/std lags (redundant with sum lags
    #          and rolling stats), units_rsum (collinear with
    #          rmean), units_rmax/rmin/rstd (dominated by mean
    #          and sum), units_expanding_mean (stale baseline
    #          from 2013 contaminates recent signal),
    #          units_norm_mom_4w (marginal over roc+rstd),
    #          units_surge_4w (overlaps units_sum_ratio),
    #          units_mom_cross_4_13 (over-engineered ratio)
    # ======================================================
    for lag in [1, 2, 3, 4]:
        d[f"units_sum_lag{lag}"] = d["units_sum"].shift(lag)

    for w in [4, 13]:
        d[f"units_rmean{w}"] = d["units_sum"].shift(1).rolling(w).mean()

    d["units_sum_ratio"] = d["units_sum"].shift(1) / (d["units_rmean4"] + 1e-9)

    units_log = np.log(d["units_sum"].shift(1).replace(0, np.nan))
    d["units_roc_1w"] = units_log - units_log.shift(1)
    d["units_roc_4w"] = units_log - units_log.shift(4)

    # demand_supply_ratio computed in Section H (needs units_rmean13)

    # ======================================================
    # SECTION D — KNIFE (BYPRODUCT) VALUE
    # Kept: knife_mean_lag1 (direct byproduct level),
    #       rolling mean/std at 4w and 13w (trend + vol),
    #       roc_4w (momentum), knife_macd / macd_hist
    #       (trend signal), knife_vs_price_mom (relative
    #       momentum between byproduct and item — unique
    #       economic signal), knife_vol_4w/13w/ratio
    # Dropped: knife_mean lags 2–10 (distant lags with no
    #          clear mechanism — lag1 suffices),
    #          knife_low/high/open/close lags (levels are
    #          non-stationary; OHLC structure captured by
    #          candlestick section above), knife_rmin/rmax
    #          (dominated by rmean/rstd), knife_roc_1w/2w/8w
    #          (roc_4w is the primary horizon),
    #          knife_mom_cross_4_13 (redundant with macd),
    #          knife_bb_width (indirect signal),
    #          knife_ewma_vol (duplicates knife_vol series),
    #          knife_down/up_semi, knife_semi_ratio
    #          (redundant given vol + win_rate),
    #          knife_win_rate (captured by macd direction),
    #          knife_skew (noisy at short windows),
    #          knife_gl_ratio (duplicates win_rate + vol)
    # ======================================================
    d["knife_mean_lag1"] = d["knife_mean"].shift(1)

    knife_log = np.log(d["knife_mean"].shift(1).replace(0, np.nan))

    for w in [4, 13]:
        d[f"knife_rmean{w}"] = d["knife_mean"].shift(1).rolling(w).mean()
        d[f"knife_rstd{w}"]  = d["knife_mean"].shift(1).rolling(w).std()

    d["knife_roc_4w"] = knife_log - knife_log.shift(4)

    ema4_k  = knife_log.ewm(span=4,  adjust=False).mean()
    ema13_k = knife_log.ewm(span=13, adjust=False).mean()
    d["knife_macd"]      = ema4_k - ema13_k
    d["knife_macd_hist"] = d["knife_macd"] - d["knife_macd"].ewm(span=4, adjust=False).mean()

    for w in [4, 13]:
        d[f"knife_vol_{w}w"] = knife_log.rolling(w).std()

    d["knife_vol_ratio"] = d["knife_vol_4w"] / (d["knife_vol_13w"] + 1e-9)

    # knife_vs_price_mom computed in Section H (needs price_roc_4w)

    # ======================================================
    # SECTION E — ROI
    # Kept: roi_mean lags 1–2 (ROI is a direct demand driver;
    #       recent profitability is most relevant),
    #       rolling mean at 4w and 13w,
    #       roc_1w and roc_4w (momentum in profitability),
    #       roi_mom_cross_4_13 (regime shift signal),
    #       roi_cross_up / roi_cross_down (clean binary
    #       regime flags around 26w median),
    #       roi_range_lag1 (weekly spread of ROI)
    # Dropped: roi_mean lags 3–10 (no clear mechanism
    #          beyond lag 2), roi_min/max lags (levels;
    #          range lag1 captures the spread cleanly),
    #          roi_rstd/rmin/rmax (rstd and range_lag1 cover
    #          dispersion; rmin/rmax add noise), roi_roc_2w
    #          (redundant between 1w and 4w), roi_vol series
    #          (volatility of ROI changes is a third-order
    #          signal with no clear price channel),
    #          roi_vol_ratio (same), roi_down/up_semi
    #          (redundant), roi_win_rate (replaced by cleaner
    #          roi_cross_up/down regime flags), roi_skew
    #          (noisy, indirect)
    # ======================================================
    for lag in [1, 2]:
        d[f"roi_mean_lag{lag}"] = d["roi_mean"].shift(lag)

    d["roi_range_lag1"] = (d["roi_max"] - d["roi_min"]).shift(1)

    for w in [4, 13]:
        d[f"roi_rmean{w}"] = d["roi_mean"].shift(1).rolling(w).mean()

    d["roi_roc_1w"] = d["roi_mean"].shift(1) - d["roi_mean"].shift(2)
    d["roi_roc_4w"] = d["roi_mean"].shift(1) - d["roi_mean"].shift(5)

    d["roi_mom_cross_4_13"] = (
        d["roi_mean"].shift(1).rolling(4).mean()
        - d["roi_mean"].shift(1).rolling(13).mean()
    )

    roi_threshold = d["roi_mean"].shift(1).rolling(26).median()
    d["roi_cross_up"] = (
        (d["roi_mean"].shift(1) > roi_threshold)
        & (d["roi_mean"].shift(2) <= roi_threshold.shift(1))
    ).astype(int)
    d["roi_cross_down"] = (
        (d["roi_mean"].shift(1) < roi_threshold)
        & (d["roi_mean"].shift(2) >= roi_threshold.shift(1))
    ).astype(int)

    # ======================================================
    # SECTION F — PLAYERS
    # Kept: players_delta (week-over-week change),
    #       players_roc_4w and roc_8w (demand expansion),
    #       players_mom_cross (short vs long trend),
    #       players_rmean4 and rmean13 (participation baseline)
    # Dropped: players_rstd (vol of player count has no
    #          clear price transmission), players_vol series
    #          and vol_ratio (same reason), players_win_rate
    #          (captured by roc direction), players_skew
    #          (noisy, indirect)
    # ======================================================
    player     = d["players_mean"].shift(1)
    player_log = np.log(player.replace(0, np.nan))

    for w in [4, 13]:
        d[f"players_rmean{w}"] = player.rolling(w).mean()

    d["players_delta"]     = d["players_mean"].shift(1) - d["players_mean"].shift(2)
    d["players_roc_4w"]    = player_log - player_log.shift(4)
    d["players_roc_8w"]    = player_log - player_log.shift(8)
    d["players_mom_cross"] = player.rolling(4).mean() / (player.rolling(13).mean() + 1e-9) - 1

    # ======================================================
    # SECTION G — COMPETITOR PRICING RATIOS
    # Kept: ratio_min_lag1, ratio_med_lag1 (current
    #       competitive position), rolling mean at 4w and 13w,
    #       roc_1w for both (how fast relative price changes),
    #       ratio_spread_lag1 (positioning spread between
    #       the two competitor benchmarks)
    # Dropped: ratio lags 2–4 (lag1 is primary; further lags
    #          add collinearity), rolling rstd (marginal given
    #          rmean and roc), roc_2w/4w (roc_1w is the
    #          primary signal for a ratio series), vol series
    #          (third-order signal), win_rate (replaced by
    #          roc direction which is continuous and cleaner)
    # ======================================================
    _rmin = d["ratio_to_min_mean"].shift(1)
    _rmed = d["ratio_to_med_mean"].shift(1)

    d["ratio_min_lag1"] = _rmin
    d["ratio_med_lag1"] = _rmed

    for w in [4, 13]:
        d[f"ratio_min_rmean{w}"] = _rmin.rolling(w).mean()
        d[f"ratio_med_rmean{w}"] = _rmed.rolling(w).mean()

    d["ratio_min_roc_1w"]  = _rmin - _rmin.shift(1)
    d["ratio_med_roc_1w"]  = _rmed - _rmed.shift(1)
    d["ratio_spread_lag1"] = _rmin - _rmed

    # ======================================================
    # SECTION H — CROSS-VARIABLE INTERACTION FEATURES
    # Kept: all four interactions — these are unique economic
    #       signals that no single feature captures alone
    # ======================================================

    # Price × Volume: return confirmation by trading activity
    d["price_vol_divergence"] = (
        d["log_return"].shift(1) * np.log1p(d["units_sum"].shift(1))
    )

    # ROI × participation: profitable + growing market = demand setup
    d["roi_x_players"] = (
        d["roi_mean"].shift(1) * np.log1p(d["players_mean"].shift(1))
    )

    # Byproduct vs item relative momentum (uses knife_roc_4w and price_roc_4w)
    d["knife_vs_price_mom"] = d["knife_roc_4w"] - d["price_roc_4w"]

    # Demand absorption: current week vs 13w average supply
    d["demand_supply_ratio"] = (
        d["units_sum"].shift(1)
        / (d["units_sum"].shift(1).rolling(13).mean() + 1e-9)
    )

    # Competitive squeeze while volume rises = dangerous pricing setup
    d["ratio_x_units_mom"] = d["ratio_min_roc_1w"] * d["units_roc_1w"]

    # ======================================================
    # SECTION I — CALENDAR & EVENT FLAGS
    # Kept: all — CS Majors and Steam Sales are documented
    #       demand events; month sin/cos encodes seasonality
    #       correctly without dummy proliferation
    # ======================================================
    d["month_sin"] = np.sin(2 * np.pi * d.index.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d.index.month / 12)

    cs_majors = [
        ("2013-11-28", "2013-11-30"),
        ("2014-03-13", "2014-03-16"),
        ("2014-07-11", "2014-07-13"),
        ("2014-11-20", "2014-11-23"),
        ("2015-03-12", "2015-03-15"),
        ("2015-08-22", "2015-08-23"),
        ("2015-10-28", "2015-11-01"),
        ("2016-03-29", "2016-04-03"),
        ("2016-07-05", "2016-07-10"),
        ("2017-01-22", "2017-01-29"),
        ("2017-07-16", "2017-07-23"),
        ("2018-01-12", "2018-01-28"),
        ("2018-09-05", "2018-09-23"),
        ("2019-02-14", "2019-03-03"),
        ("2019-08-23", "2019-09-08"),
        ("2021-10-26", "2021-11-07"),
        ("2022-05-09", "2022-05-22"),
        ("2022-10-31", "2022-11-13"),
        ("2023-05-08", "2023-05-21"),
        ("2024-03-17", "2024-03-31"),
        ("2024-11-30", "2024-12-15"),
        ("2025-06-03", "2025-06-22"),
        ("2025-11-24", "2025-12-14"),
    ]
    all_major_dates = []
    for start, end in cs_majors:
        all_major_dates.extend(pd.date_range(start, end, freq="D"))
    major_weeks = pd.DatetimeIndex(all_major_dates).to_period("W").to_timestamp("W")
    d["is_major_week"] = d.index.isin(major_weeks).astype(int)

    steam_sales = [
        ("2013-11-27", "2013-12-03"),
        ("2014-06-19", "2014-06-30"),
        ("2014-12-18", "2015-01-02"),
        ("2015-06-11", "2015-06-21"),
        ("2015-12-22", "2016-01-04"),
        ("2016-06-23", "2016-07-04"),
        ("2016-11-23", "2016-11-29"),
        ("2016-12-22", "2017-01-02"),
        ("2017-06-22", "2017-07-05"),
        ("2017-11-22", "2017-11-28"),
        ("2017-12-21", "2018-01-04"),
        ("2018-06-21", "2018-07-05"),
        ("2018-11-21", "2018-11-27"),
        ("2018-12-20", "2019-01-03"),
        ("2019-06-25", "2019-07-09"),
        ("2019-11-27", "2019-12-03"),
        ("2019-12-19", "2020-01-02"),
        ("2020-06-25", "2020-07-09"),
        ("2020-11-25", "2020-12-01"),
        ("2020-12-22", "2021-01-05"),
        ("2021-06-24", "2021-07-08"),
        ("2021-11-24", "2021-11-30"),
        ("2021-12-22", "2022-01-05"),
        ("2022-06-23", "2022-07-07"),
        ("2022-11-23", "2022-11-29"),
        ("2022-12-22", "2023-01-05"),
        ("2023-06-29", "2023-07-13"),
        ("2023-11-21", "2023-11-28"),
        ("2023-12-21", "2024-01-04"),
        ("2024-06-27", "2024-07-11"),
        ("2024-11-27", "2024-12-03"),
        ("2024-12-19", "2025-01-02"),
        ("2025-06-26", "2025-07-10"),
        ("2025-11-26", "2025-12-02"),
        ("2025-12-18", "2026-01-01"),
    ]
    all_sale_dates = []
    for start, end in steam_sales:
        all_sale_dates.extend(pd.date_range(start, end, freq="D"))
    sale_weeks = pd.DatetimeIndex(all_sale_dates).to_period("W").to_timestamp("W")
    d["is_steam_sale"] = d.index.isin(sale_weeks).astype(int)

    print(f"is_major_week fires : {d['is_major_week'].sum()} weeks")
    print(f"is_steam_sale fires : {d['is_steam_sale'].sum()} weeks")

    return d


def build_matrix(weekly):
    """Weekly bars -> (X, y, closes), with excluded columns removed."""
    feats = engineer_weekly(weekly)

    cols = [c for c in feats.columns if c not in EXCLUDE]
    feats = feats.dropna(subset=cols + [TARGET])

    X, y, closes = feats[cols], feats[TARGET], feats["Close"]
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    return X, y, closes, feats



