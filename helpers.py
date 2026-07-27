
import os

import numpy as np

# color palette
MLE_COLOR = "#9b2393"  
XGB_COLOR = "#3a56d4"   
ACCENT    = "#3a56d4"
NAIVE_C   = "#e07b39"   
GREEN     = "#2a9d63"   
RED       = "#d94f3d"   
GOLD      = "#c9880a"   
DARK      = "#1a1d2e"   
MUTED     = "#6b7194"   
PANEL     = "#f7f8fc"   
CARD      = "#f0f2f8"   
BORDER    = "#d0d4e4"   
BG        = "#ffffff"   


# metrics
def direction_mask(price_true, price_model):
    """
    Both directions are measured against the previous week's actual price:

    So the model is judged on whether it said up or down from where the market
    actually was, not on whether its level landed above or below the outturn.
    """
    price_true = np.asarray(price_true, dtype=float)
    price_model = np.asarray(price_model, dtype=float)

    true_dir = np.sign(price_true[1:] - price_true[:-1])
    pred_dir = np.sign(price_model[1:] - price_true[:-1])
    return np.concatenate([[False], true_dir == pred_dir])


def pct_error(price_true, price_model):
    """
    Signed percentage error, guarded against near-zero prices.
    """
    price_true = np.asarray(price_true, dtype=float)
    price_model = np.asarray(price_model, dtype=float)

    denom = np.where(np.abs(price_true) > 1e-8, price_true, np.nan)
    return (price_model - price_true) / denom * 100.0


# ---------------------------------------------------------------- io
def save_table(df, name, outdir="outputs"):
    
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.csv")
    df.to_csv(path)
    print(f"  saved {path}  ({len(df)} rows)")
    return path
