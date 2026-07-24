"""
Loading the raw data, adding additional variables and agregating them to weekly bars.
"""

import pandas as pd

from config import DATASET, END_DATE, DROP_POOL_DATE


# loading

def add_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    creates ratios of the target case price to the minimum and median competitor prices.
    it's a better aproach than using the raw competitor prices, which are correlated and have a lot of noise.

    Adds:
        price_to_min_comp  : Price / min(competitor prices) that day
        price_to_med_comp  : Price / median(competitor prices) that day
        comp_price_spread  : (max - min) / min  → normalised market fragmentation
    """
    df = df.copy()

    competitor_cols = [c for c in df.columns if c.endswith("_Price")]

    print(f"Competitors found ({len(competitor_cols)}): {competitor_cols}")

    comp_prices    = df[competitor_cols]
    daily_comp_min = comp_prices.min(axis=1)
    daily_comp_med = comp_prices.median(axis=1)
    daily_comp_max = comp_prices.max(axis=1)

    eps = 1e-9
    df["price_to_min_comp"]  = df["Price"] / daily_comp_min.clip(lower=eps)
    df["price_to_med_comp"]  = df["Price"] / daily_comp_med.clip(lower=eps)
    df["comp_price_spread"]  = (daily_comp_max - daily_comp_min) / daily_comp_min.clip(lower=eps)

    return df


def load_daily(path=DATASET):
    """Read the CSV, apply the two domain rules, index by date."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= END_DATE]

    # After this date the rare drop pool was removed, so affected cases
    # structurally stop dropping.
    df.loc[df["Date"] > DROP_POOL_DATE, "estimated_drops"] = 0

    df = df.set_index("Date").sort_index()
    df = add_price_ratios(df)

    # Raw competitor prices are dropped once the ratios are derived.
    df = df.drop(columns=[c for c in df.columns
                          if c.endswith("_Price") and c != "Price"])

    print(f"Daily: {len(df)} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def resample_weekly(df):
    """Daily observations -> weekly OHLC bars plus summary statistics."""
    price = df["Price"]

    w = pd.DataFrame({
        "Open":  price.resample("W").first(),
        "High":  price.resample("W").max(),
        "Low":   price.resample("W").min(),
        "Close": price.resample("W").last(),
    })

    w["units_sum"] = df["Units Sold"].resample("W").sum()
    w["units_std"] = df["Units Sold"].resample("W").std().fillna(0)
    w["units_min"] = df["Units Sold"].resample("W").min()
    w["units_max"] = df["Units Sold"].resample("W").max()

    w["knife_mean"]  = df["Mean_Knife_Value"].resample("W").mean()
    w["knife_low"]   = df["Mean_Knife_Value"].resample("W").min()
    w["knife_high"]  = df["Mean_Knife_Value"].resample("W").max()
    w["knife_open"]  = df["Mean_Knife_Value"].resample("W").first()
    w["knife_close"] = df["Mean_Knife_Value"].resample("W").last()

    w["roi_mean"] = df["ROI"].resample("W").mean()
    w["roi_std"]  = df["ROI"].resample("W").std().fillna(0)
    w["roi_min"]  = df["ROI"].resample("W").min()
    w["roi_max"]  = df["ROI"].resample("W").max()

    w["players_mean"] = df["Avg. Players"].resample("W").mean()
    w["players_max"]  = df["Avg. Players"].resample("W").max()

    w["ratio_to_min_mean"] = df["price_to_min_comp"].resample("W").mean()
    w["ratio_to_min_min"]  = df["price_to_min_comp"].resample("W").min()
    w["ratio_to_min_max"]  = df["price_to_min_comp"].resample("W").max()

    w["ratio_to_med_mean"] = df["price_to_med_comp"].resample("W").mean()
    w["ratio_to_med_std"]  = df["price_to_med_comp"].resample("W").std().fillna(0)

    w["n_days"] = price.resample("W").count()
    w.index.name = "Week"
    w = w.dropna()

    print(f"Weekly: {len(w)} bars  {w.index[0].date()} -> {w.index[-1].date()}")
    return w
