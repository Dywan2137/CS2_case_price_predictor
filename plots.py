"""All charts.

Split out from the analysis so that results can be produced without a display,
and so that changing a colour never risks touching a number.

The diagnostic suites live here rather than in stats_tests.py: they run
Breusch-Pagan, Durbin-Watson, Ljung-Box and Jarque-Bera, but they are 90%
figure code, and stats_tests.py is more useful without matplotlib in it.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import jarque_bera, probplot, spearmanr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from helpers import (
    ACCENT, BG, BORDER, CARD, DARK, GOLD, GREEN, MLE_COLOR, MUTED,
    NAIVE_C, PANEL, RED, XGB_COLOR,
)
from helpers import direction_mask as _direction_mask
from helpers import pct_error as _pct_error
from model_linear import mle_coefficient_table


# ============================================================
# WALK-FORWARD RESULT PANELS
# ============================================================

def plot_walkforward_yearly(
    df_prices     : pd.DataFrame,
    metrics_price : pd.DataFrame,
    breakdown_price: pd.DataFrame,
    figsize       : tuple = (18, 14),
    save_dir      : str   = ".",
    dpi           : int   = 180,
) -> None:
    """
    One publication-quality plot per year of the walk-forward OOS period.
    White background, paper-ready.

    Parameters
    ----------
    df_prices       : price-level predictions from backtransform_and_evaluate()
    metrics_price   : full-period metrics (for global reference lines)
    breakdown_price : annual metrics from price_evaluation_report()
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    df    = df_prices.sort_index().dropna(
                subset=["price_true", "price_model", "price_naive"])
    years = sorted(df.index.year.unique())

    # Global MAE/MAPE for reference lines across all panels
    global_mae  = metrics_price.loc["Model",      "MAE_price"]
    global_mape = metrics_price.loc["Model",      "MAPE_price"]
    naive_mae   = metrics_price.loc["Naive (RW)", "MAE_price"]
    naive_mape  = metrics_price.loc["Naive (RW)", "MAPE_price"]

    for yr in years:
        yr_df = df[df.index.year == yr].copy()
        if len(yr_df) < 2:
            print(f"  Skipping {yr} — fewer than 2 observations")
            continue

        dates = yr_df.index
        pt    = yr_df["price_true"].values
        pm    = yr_df["price_model"].values
        pn    = yr_df["price_naive"].values
        lt    = yr_df["y_true"].values
        lp    = yr_df["y_pred"].values

        ok_model = _direction_mask(pt, pm)
        abs_err   = np.abs(pm - pt)
        pct_err   = _pct_error(pt, pm)
        naive_abs = np.abs(pn - pt)

        # ── Annual metrics ────────────────────────────────────────────────
        try:
            m_yr = breakdown_price.loc[(yr, "Model")]
            n_yr = breakdown_price.loc[(yr, "Naive")]
        except KeyError:
            print(f"  No breakdown metrics for {yr} — skipping")
            continue

        mae_yr   = m_yr["MAE_price"]
        rmse_yr  = m_yr["RMSE_price"]
        mape_yr  = m_yr["MAPE_price"]
        r2_yr    = m_yr["R2_price"]
        dir_yr   = m_yr["Dir_Acc"] * 100
        bias_yr  = m_yr["Bias_price"]
        rho_yr, _= spearmanr(lt, lp)

        # ── Layout ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize, facecolor=BG)
        gs  = GridSpec(
            3, 3,
            figure = fig,
            hspace = 0.52, wspace = 0.38,
            left   = 0.06, right  = 0.97,
            top    = 0.91, bottom = 0.07,
        )
        ax_main  = fig.add_subplot(gs[0, :])
        ax_err   = fig.add_subplot(gs[1, :2])
        ax_pct   = fig.add_subplot(gs[2, :2])
        ax_table = fig.add_subplot(gs[1:, 2])

        # ── Panel 1: Price chart ──────────────────────────────────────────
        ax_main.fill_between(
            dates, pt, pm,
            where=pm >= pt, interpolate=True,
            color=GREEN, alpha=0.12, label="_nolegend_",
        )
        ax_main.fill_between(
            dates, pt, pm,
            where=pm < pt, interpolate=True,
            color=RED, alpha=0.12, label="_nolegend_",
        )

        ax_main.plot(dates, pn,
                     color=NAIVE_C, lw=1.4, ls=":",
                     zorder=3, alpha=0.85, label="Naive (random walk)")
        ax_main.plot(dates, pt,
                     color=DARK, lw=2.2,
                     zorder=5, label="Actual price")
        ax_main.plot(dates, pm,
                     color=ACCENT, lw=1.9, ls="--",
                     zorder=4, label="Model prediction")

        # ax_main.scatter(dates[ok_model],  pm[ok_model],
        #                 color=GREEN, s=40, zorder=6,
        #                 marker="o", edgecolors="none", label="Direction ✓")
        # ax_main.scatter(dates[~ok_model], pm[~ok_model],
        #                 color=RED, s=55, zorder=6,
        #                 marker="x", linewidths=1.4, label="Direction ✗")

        ax_main.set_title(
            f"Walk-Forward Validation {yr} — Actual vs Predicted Weekly Close Price",
            fontsize=13, fontweight="bold", color=DARK, pad=10,
        )
        ax_main.set_ylabel("Price  ($)", fontsize=11, color=DARK)
        # ax_main.legend(fontsize=9, facecolor=BG, edgecolor=BORDER,
        #                labelcolor=DARK, ncol=5, loc="upper left")
        ax_main.legend(fontsize=9, facecolor=BG, edgecolor=BORDER,
               labelcolor=DARK, ncol=3, loc="upper left")

        kpi = (f"MAE ${mae_yr:.4f}   RMSE ${rmse_yr:.4f}   "
               f"MAPE {mape_yr:.2f}%   R² {r2_yr:+.4f}   "
               f"Dir Acc {dir_yr:.1f}%   Spearman ρ {rho_yr:.4f}   "
               f"Bias {bias_yr:+.4f}")
        ax_main.text(
            0.01, 0.04, kpi, transform=ax_main.transAxes,
            fontsize=8.5, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=CARD, edgecolor=BORDER, alpha=0.90),
        )

        # Monthly shading bands for intra-year readability
        months = sorted(yr_df.index.month.unique())
        for i, mo in enumerate(months):
            mo_dates = yr_df.index[yr_df.index.month == mo]
            if len(mo_dates) == 0:
                continue
            if i % 2 == 0:
                for ax in [ax_main, ax_err, ax_pct]:
                    ax.axvspan(mo_dates[0], mo_dates[-1],
                               color=BORDER, alpha=0.13, zorder=0)

        # ── Panel 2: Absolute error ───────────────────────────────────────
        bar_colors = [GREEN if ok else RED for ok in ok_model]
        ax_err.bar(dates, abs_err,
                   color=bar_colors, alpha=0.72,
                   width=3.5, edgecolor="none", label="Model |error|")
        ax_err.plot(dates, naive_abs,
                    color=NAIVE_C, lw=1.4, ls=":",
                    zorder=3, alpha=0.85, label="Naive |error|")
        ax_err.axhline(mae_yr, color=GOLD, lw=1.5, ls="--",
                       label=f"Model MAE  ${mae_yr:.4f}")
        ax_err.axhline(n_yr["MAE_price"], color=NAIVE_C, lw=1.0, ls="-.",
                       label=f"Naive MAE  ${n_yr['MAE_price']:.4f}", alpha=0.75)
        # Global reference
        ax_err.axhline(global_mae, color=MUTED, lw=0.8, ls=":",
                       label=f"Global MAE ${global_mae:.4f}", alpha=0.5)

        ax_err.set_title(
            f"{yr} — Absolute Price Error per Week  (green = direction correct)",
            fontsize=11, fontweight="bold", color=DARK,
        )
        ax_err.set_ylabel("$ Error", fontsize=10, color=DARK)
        ax_err.legend(fontsize=8, facecolor=BG, edgecolor=BORDER,
                      labelcolor=DARK, ncol=2)

        # ── Panel 3: Percentage error ─────────────────────────────────────
        ax_pct.plot(dates, pct_err, color=ACCENT, lw=1.8, zorder=3)
        ax_pct.fill_between(dates, pct_err, color=ACCENT, alpha=0.12)
        ax_pct.axhline(mape_yr, color=GOLD, lw=1.5, ls="--",
                       label=f"Model MAPE  {mape_yr:.2f}%")
        ax_pct.axhline(n_yr["MAPE_price"], color=NAIVE_C, lw=1.0, ls="-.",
                       label=f"Naive MAPE  {n_yr['MAPE_price']:.2f}%", alpha=0.75)
        ax_pct.axhline(global_mape, color=MUTED, lw=0.8, ls=":",
                       label=f"Global MAPE {global_mape:.2f}%", alpha=0.5)
        ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax_pct.set_title(f"{yr} — Percentage Price Error per Week",
                         fontsize=11, fontweight="bold", color=DARK)
        ax_pct.set_ylabel("% Error", fontsize=10, color=DARK)
        ax_pct.legend(fontsize=8, facecolor=BG, edgecolor=BORDER,
                      labelcolor=DARK, ncol=2)

        # ── X-axis: monthly ticks, readable within a single year ──────────
        for ax in [ax_main, ax_err, ax_pct]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=0, ha="center", fontsize=9)
            ax.set_facecolor(PANEL)

        # ── Panel 4: Metrics table — annual vs naive ──────────────────────
        ax_table.set_facecolor(CARD)
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(0, 1)
        ax_table.axis("off")

        ax_table.text(0.5, 0.975, f"{yr}  —  Model vs Naive",
                      transform=ax_table.transAxes,
                      fontsize=11, fontweight="bold",
                      color=DARK, ha="center", va="top")

        # Column headers
        for x, label, color in [
            (0.06, "Metric", MUTED),
            (0.60, "Model",  ACCENT),
            (0.88, "Naive",  NAIVE_C),
        ]:
            ax_table.text(x, 0.925, label,
                          transform=ax_table.transAxes,
                          fontsize=8.5, fontweight="bold",
                          color=color,
                          ha="center" if x > 0.1 else "left",
                          va="top")

        ax_table.plot([0, 1], [0.905, 0.905],
                  color=BORDER, lw=0.8,
                  transform=ax_table.transAxes,
                  clip_on=False)

        table_rows = [
            ("MAE ($)",    f"{mae_yr:.4f}",        f"{n_yr['MAE_price']:.4f}",   True),
            ("RMSE ($)",   f"{rmse_yr:.4f}",       f"{n_yr['RMSE_price']:.4f}",  True),
            ("MAPE (%)",   f"{mape_yr:.2f}",       f"{n_yr['MAPE_price']:.2f}",  True),
            ("R²",         f"{r2_yr:+.4f}",        f"{n_yr['R2_price']:+.4f}",   False),
            ("Dir Acc",    f"{dir_yr:.1f}%",       f"{n_yr['Dir_Acc']*100:.1f}%",False),
            ("Bias ($)",   f"{bias_yr:+.4f}",      f"{n_yr['Bias_price']:+.4f}", False),
            ("Spearman ρ", f"{rho_yr:.4f}",        "—",                          False),
            ("Weeks",      f"{len(yr_df)}",        f"{len(yr_df)}",              False),
            ("Global MAE", f"${global_mae:.4f}",   f"${naive_mae:.4f}",          True),
            ("Global MAPE",f"{global_mape:.2f}%",  f"{naive_mape:.2f}%",         True),
        ]

        row_h   = 0.075
        start_y = 0.88

        for i, (lbl, val_m, val_n, lower_better) in enumerate(table_rows):
            y   = start_y - i * row_h
            bg  = PANEL if i % 2 == 0 else CARD
            ax_table.add_patch(plt.Rectangle(
                (0.0, y - 0.034), 1.0, row_h,
                transform=ax_table.transAxes,
                facecolor=bg, edgecolor="none", clip_on=False,
            ))
            ax_table.text(0.06, y, lbl,
                          transform=ax_table.transAxes,
                          fontsize=8.5, color=DARK, va="center")

            # Colour model value vs naive
            try:
                mv = float(val_m.replace("%","").replace("+","").replace("$",""))
                nv = float(val_n.replace("%","").replace("+","")
                               .replace("$","").replace("—","nan"))
                if lower_better:
                    mc = GREEN if mv < nv else RED
                else:
                    mc = GREEN if mv > nv else RED
            except Exception:
                mc = GOLD

            ax_table.text(0.60, y, val_m,
                          transform=ax_table.transAxes,
                          fontsize=8.5, fontweight="bold",
                          color=mc, ha="center", va="center")
            ax_table.text(0.88, y, val_n,
                          transform=ax_table.transAxes,
                          fontsize=8.5, color=MUTED,
                          ha="center", va="center")

        ax_table.add_patch(plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            transform=ax_table.transAxes,
            facecolor="none", edgecolor=BORDER, lw=1.0, clip_on=False,
        ))

        # ── Figure title ──────────────────────────────────────────────────
        fig.suptitle(
            f"XGBoost Walk-Forward Validation  —  {yr}  |  Weekly Price Prediction",
            fontsize=13, fontweight="bold", color=DARK, y=0.975,
        )
        fig.patch.set_facecolor(BG)

        path = f"{save_dir}/walkforward_{yr}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
        print(f"  Saved → {path}")

    print(f"\nAll {len(years)} yearly plots saved to '{save_dir}/'")


def plot_mle_yearly(
    df_prices_mle   : pd.DataFrame,
    metrics_price_mle : pd.DataFrame,
    breakdown_price_mle : pd.DataFrame,
    figsize         : tuple = (18, 14),
    save_dir        : str   = "mle_yearly",
    dpi             : int   = 180,
) -> None:
    """
    One publication-quality plot per year for the MLE walk-forward results.
    Mirrors plot_walkforward_yearly exactly but labelled for Linear Regression.

    Parameters
    ----------
    df_prices_mle       : output of backtransform_and_evaluate(mle_results, ...)
    metrics_price_mle   : full-period price metrics
    breakdown_price_mle : annual price metrics
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    df    = df_prices_mle.sort_index().dropna(
                subset=["price_true", "price_model", "price_naive"])
    years = sorted(df.index.year.unique())

    global_mae  = metrics_price_mle.loc["Model",      "MAE_price"]
    global_mape = metrics_price_mle.loc["Model",      "MAPE_price"]
    naive_mae   = metrics_price_mle.loc["Naive (RW)", "MAE_price"]
    naive_mape  = metrics_price_mle.loc["Naive (RW)", "MAPE_price"]

    for yr in years:
        yr_df = df[df.index.year == yr].copy()
        if len(yr_df) < 2:
            print(f"  Skipping {yr} — fewer than 2 observations")
            continue

        dates = yr_df.index
        pt    = yr_df["price_true"].values
        pm    = yr_df["price_model"].values
        pn    = yr_df["price_naive"].values
        lt    = yr_df["y_true"].values
        lp    = yr_df["y_pred"].values

        ok_model = _direction_mask(pt, pm)
        abs_err   = np.abs(pm - pt)
        pct_err   = _pct_error(pt, pm)
        naive_abs = np.abs(pn - pt)

        # ── Annual metrics ────────────────────────────────────────────────
        try:
            m_yr = breakdown_price_mle.loc[(yr, "Model")]
            n_yr = breakdown_price_mle.loc[(yr, "Naive")]
        except KeyError:
            print(f"  No breakdown metrics for {yr} — skipping")
            continue

        mae_yr   = m_yr["MAE_price"]
        rmse_yr  = m_yr["RMSE_price"]
        mape_yr  = m_yr["MAPE_price"]
        r2_yr    = m_yr["R2_price"]
        dir_yr   = m_yr["Dir_Acc"] * 100
        bias_yr  = m_yr["Bias_price"]
        rho_yr, _= spearmanr(lt, lp)

        # ── Layout ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize, facecolor=BG)
        gs  = GridSpec(
            3, 3,
            figure = fig,
            hspace = 0.52, wspace = 0.38,
            left   = 0.06, right  = 0.97,
            top    = 0.91, bottom = 0.07,
        )
        ax_main  = fig.add_subplot(gs[0, :])
        ax_err   = fig.add_subplot(gs[1, :2])
        ax_pct   = fig.add_subplot(gs[2, :2])
        ax_table = fig.add_subplot(gs[1:, 2])

        # ── Panel 1: Price chart ──────────────────────────────────────────
        ax_main.fill_between(
            dates, pt, pm,
            where=pm >= pt, interpolate=True,
            color=GREEN, alpha=0.12, label="_nolegend_",
        )
        ax_main.fill_between(
            dates, pt, pm,
            where=pm < pt, interpolate=True,
            color=RED, alpha=0.12, label="_nolegend_",
        )

        ax_main.plot(dates, pn,
                     color=NAIVE_C, lw=1.4, ls=":",
                     zorder=3, alpha=0.85, label="Naive (random walk)")
        ax_main.plot(dates, pt,
                     color=DARK, lw=2.2,
                     zorder=5, label="Actual price")

        # MLE uses a different line style/colour to distinguish from XGBoost
        MLE_COLOR = "#9b2393"   # purple — visually distinct from XGB blue
        ax_main.plot(dates, pm,
                     color=MLE_COLOR, lw=1.9, ls="--",
                     zorder=4, label="MLE prediction")

        # ax_main.scatter(dates[ok_model],  pm[ok_model],
        #                 color=GREEN, s=40, zorder=6,
        #                 marker="o", edgecolors="none", label="Direction ✓")
        # ax_main.scatter(dates[~ok_model], pm[~ok_model],
        #                 color=RED, s=55, zorder=6,
        #                 marker="x", linewidths=1.4, label="Direction ✗")

        ax_main.set_title(
            f"MLE Walk-Forward {yr} — Actual vs Predicted Weekly Close Price",
            fontsize=13, fontweight="bold", color=DARK, pad=10,
        )
        ax_main.set_ylabel("Price  ($)", fontsize=11, color=DARK)
        # ax_main.legend(fontsize=9, facecolor=BG, edgecolor=BORDER,
        #                labelcolor=DARK, ncol=5, loc="upper left")
        ax_main.legend(fontsize=9, facecolor=BG, edgecolor=BORDER,
               labelcolor=DARK, ncol=3, loc="upper left")

        kpi = (f"MAE ${mae_yr:.4f}   RMSE ${rmse_yr:.4f}   "
               f"MAPE {mape_yr:.2f}%   R² {r2_yr:+.4f}   "
               f"Dir Acc {dir_yr:.1f}%   Spearman ρ {rho_yr:.4f}   "
               f"Bias {bias_yr:+.4f}")
        ax_main.text(
            0.01, 0.04, kpi, transform=ax_main.transAxes,
            fontsize=8.5, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=CARD, edgecolor=BORDER, alpha=0.90),
        )

        # Monthly shading bands
        months = sorted(yr_df.index.month.unique())
        for i, mo in enumerate(months):
            mo_dates = yr_df.index[yr_df.index.month == mo]
            if len(mo_dates) == 0:
                continue
            if i % 2 == 0:
                for ax in [ax_main, ax_err, ax_pct]:
                    ax.axvspan(mo_dates[0], mo_dates[-1],
                               color=BORDER, alpha=0.13, zorder=0)

        # ── Panel 2: Absolute error ───────────────────────────────────────
        bar_colors = [GREEN if ok else RED for ok in ok_model]
        ax_err.bar(dates, abs_err,
                   color=bar_colors, alpha=0.72,
                   width=3.5, edgecolor="none", label="MLE |error|")
        ax_err.plot(dates, naive_abs,
                    color=NAIVE_C, lw=1.4, ls=":",
                    zorder=3, alpha=0.85, label="Naive |error|")
        ax_err.axhline(mae_yr, color=MLE_COLOR, lw=1.5, ls="--",
                       label=f"MLE MAE  ${mae_yr:.4f}")
        ax_err.axhline(n_yr["MAE_price"], color=NAIVE_C, lw=1.0, ls="-.",
                       label=f"Naive MAE  ${n_yr['MAE_price']:.4f}", alpha=0.75)
        ax_err.axhline(global_mae, color=MUTED, lw=0.8, ls=":",
                       label=f"Global MAE ${global_mae:.4f}", alpha=0.5)

        ax_err.set_title(
            f"{yr} — Absolute Price Error per Week  (green = direction correct)",
            fontsize=11, fontweight="bold", color=DARK,
        )
        ax_err.set_ylabel("$ Error", fontsize=10, color=DARK)
        ax_err.legend(fontsize=8, facecolor=BG, edgecolor=BORDER,
                      labelcolor=DARK, ncol=2)

        # ── Panel 3: Percentage error ─────────────────────────────────────
        ax_pct.plot(dates, pct_err, color=MLE_COLOR, lw=1.8, zorder=3)
        ax_pct.fill_between(dates, pct_err, color=MLE_COLOR, alpha=0.10)
        ax_pct.axhline(mape_yr, color=MLE_COLOR, lw=1.5, ls="--",
                       label=f"MLE MAPE  {mape_yr:.2f}%")
        ax_pct.axhline(n_yr["MAPE_price"], color=NAIVE_C, lw=1.0, ls="-.",
                       label=f"Naive MAPE  {n_yr['MAPE_price']:.2f}%", alpha=0.75)
        ax_pct.axhline(global_mape, color=MUTED, lw=0.8, ls=":",
                       label=f"Global MAPE {global_mape:.2f}%", alpha=0.5)
        ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax_pct.set_title(f"{yr} — Percentage Price Error per Week",
                         fontsize=11, fontweight="bold", color=DARK)
        ax_pct.set_ylabel("% Error", fontsize=10, color=DARK)
        ax_pct.legend(fontsize=8, facecolor=BG, edgecolor=BORDER,
                      labelcolor=DARK, ncol=2)

        # ── X-axis formatting ─────────────────────────────────────────────
        for ax in [ax_main, ax_err, ax_pct]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=0, ha="center", fontsize=9)
            ax.set_facecolor(PANEL)

        # ── Panel 4: Metrics table ────────────────────────────────────────
        ax_table.set_facecolor(CARD)
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(0, 1)
        ax_table.axis("off")

        ax_table.text(0.5, 0.975, f"{yr}  —  MLE vs Naive",
                      transform=ax_table.transAxes,
                      fontsize=11, fontweight="bold",
                      color=DARK, ha="center", va="top")

        for x, label, color in [
            (0.06, "Metric",  MUTED),
            (0.60, "MLE",     MLE_COLOR),
            (0.88, "Naive",   NAIVE_C),
        ]:
            ax_table.text(x, 0.925, label,
                          transform=ax_table.transAxes,
                          fontsize=8.5, fontweight="bold", color=color,
                          ha="center" if x > 0.1 else "left", va="top")

        ax_table.plot([0, 1], [0.905, 0.905],
                      color=BORDER, lw=0.8,
                      transform=ax_table.transAxes, clip_on=False)

        table_rows = [
            ("MAE ($)",    f"{mae_yr:.4f}",        f"{n_yr['MAE_price']:.4f}",    True),
            ("RMSE ($)",   f"{rmse_yr:.4f}",       f"{n_yr['RMSE_price']:.4f}",   True),
            ("MAPE (%)",   f"{mape_yr:.2f}",       f"{n_yr['MAPE_price']:.2f}",   True),
            ("R²",         f"{r2_yr:+.4f}",        f"{n_yr['R2_price']:+.4f}",    False),
            ("Dir Acc",    f"{dir_yr:.1f}%",       f"{n_yr['Dir_Acc']*100:.1f}%", False),
            ("Bias ($)",   f"{bias_yr:+.4f}",      f"{n_yr['Bias_price']:+.4f}",  False),
            ("Spearman ρ", f"{rho_yr:.4f}",        "—",                           False),
            ("Weeks",      f"{len(yr_df)}",        f"{len(yr_df)}",               False),
            ("Global MAE", f"${global_mae:.4f}",   f"${naive_mae:.4f}",           True),
            ("Global MAPE",f"{global_mape:.2f}%",  f"{naive_mape:.2f}%",          True),
        ]

        row_h   = 0.075
        start_y = 0.88

        for i, (lbl, val_m, val_n, lower_better) in enumerate(table_rows):
            y   = start_y - i * row_h
            bg  = PANEL if i % 2 == 0 else CARD
            ax_table.add_patch(plt.Rectangle(
                (0.0, y - 0.034), 1.0, row_h,
                transform=ax_table.transAxes,
                facecolor=bg, edgecolor="none", clip_on=False,
            ))
            ax_table.text(0.06, y, lbl,
                          transform=ax_table.transAxes,
                          fontsize=8.5, color=DARK, va="center")

            try:
                mv = float(val_m.replace("%","").replace("+","").replace("$",""))
                nv = float(val_n.replace("%","").replace("+","")
                               .replace("$","").replace("—","nan"))
                if lower_better:
                    mc = GREEN if mv < nv else RED
                else:
                    mc = GREEN if mv > nv else RED
            except Exception:
                mc = GOLD

            ax_table.text(0.60, y, val_m,
                          transform=ax_table.transAxes,
                          fontsize=8.5, fontweight="bold",
                          color=mc, ha="center", va="center")
            ax_table.text(0.88, y, val_n,
                          transform=ax_table.transAxes,
                          fontsize=8.5, color=MUTED,
                          ha="center", va="center")

        ax_table.add_patch(plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            transform=ax_table.transAxes,
            facecolor="none", edgecolor=BORDER, lw=1.0, clip_on=False,
        ))

        fig.suptitle(
            f"MLE Linear Regression Walk-Forward — {yr}  |  Weekly Price Prediction",
            fontsize=13, fontweight="bold", color=DARK, y=0.975,
        )
        fig.patch.set_facecolor(BG)

        path = f"{save_dir}/mle_walkforward_{yr}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
        print(f"  Saved → {path}")

    print(f"\nAll {len(years)} MLE yearly plots saved to '{save_dir}/'")


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

def mle_residual_diagnostics(
    mle_results   : pd.DataFrame,
    mle_models    : list,
    feature_names : list,
    save_path     : str = "mle_diagnostics.png",
    dpi           : int = 180,
) -> dict:
    """
    Full residual diagnostic suite for the MLE model.
    Returns dict of test statistics for thesis reporting.
    """
    residuals = (mle_results["y_pred"] - mle_results["y_true"]).values
    fitted    = mle_results["y_pred"].values
    dates     = mle_results.index

    # ── Formal tests ─────────────────────────────────────────────────────
    # Breusch-Pagan heteroscedasticity
    last_model   = mle_models[-1]
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(
        last_model.resid, last_model.model.exog
    )

    # Durbin-Watson autocorrelation
    dw_stat = durbin_watson(residuals)

    # Ljung-Box (lags 1, 4, 8, 13)
    lb_result = acorr_ljungbox(residuals, lags=[1, 4, 8, 13], return_df=True)

    # Jarque-Bera normality
    jb_stat, jb_p = jarque_bera(residuals)

    tests = {
        "Breusch-Pagan LM stat"  : round(bp_lm,   4),
        "Breusch-Pagan p-value"  : round(bp_p,    4),
        "Durbin-Watson"          : round(dw_stat,  4),
        "Jarque-Bera stat"       : round(jb_stat,  4),
        "Jarque-Bera p-value"    : round(jb_p,     4),
    }

    # ── Print test summary ────────────────────────────────────────────────
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  MLE RESIDUAL DIAGNOSTICS")
    print(sep)
    print(f"\n  Heteroscedasticity — Breusch-Pagan")
    print(f"    LM statistic : {bp_lm:.4f}")
    print(f"    p-value      : {bp_p:.4f}  "
          f"({'REJECT homoscedasticity' if bp_p < 0.05 else 'fail to reject — OK'})")
    print(f"\n  Autocorrelation — Durbin-Watson")
    print(f"    DW statistic : {dw_stat:.4f}  "
          f"({'near 2 — no autocorrelation' if 1.5 < dw_stat < 2.5 else 'POSSIBLE AUTOCORRELATION'})")
    print(f"\n  Autocorrelation — Ljung-Box")
    print(lb_result.to_string())
    print(f"\n  Normality — Jarque-Bera")
    print(f"    JB statistic : {jb_stat:.4f}")
    print(f"    p-value      : {jb_p:.4f}  "
          f"({'REJECT normality' if jb_p < 0.05 else 'fail to reject — OK'})")
    print(sep)

    # ── Coefficient table ─────────────────────────────────────────────────
    coef_df = mle_coefficient_table(mle_models, feature_names)
    print(f"\n  COEFFICIENT TABLE (last fold, n={mle_models[-1].nobs:.0f})")
    print(f"  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")
    print(sep)
    print(coef_df[["coef","std_err","t_stat","p_value","signif",
                   "ci_low","ci_high"]].round(5).to_string())
    print(sep)

    # ── Diagnostic plots ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.07)

    ax_rv  = fig.add_subplot(gs[0, 0])   # residual vs fitted
    ax_time= fig.add_subplot(gs[0, 1])   # residuals over time
    ax_qq  = fig.add_subplot(gs[0, 2])   # QQ plot
    ax_acf = fig.add_subplot(gs[1, 0])   # ACF of residuals
    ax_his = fig.add_subplot(gs[1, 1])   # residual histogram
    ax_r2  = fig.add_subplot(gs[1, 2])   # rolling R² over folds
    ax_aic = fig.add_subplot(gs[2, 0])   # AIC over folds
    ax_cof = fig.add_subplot(gs[2, 1:])  # coefficient plot

    for ax in fig.axes:
        ax.set_facecolor(PANEL)

    # 1. Residual vs fitted
    ax_rv.scatter(fitted, residuals, color=MLE_COLOR,
                  alpha=0.4, s=18, edgecolors="none")
    ax_rv.axhline(0, color=DARK, lw=1.0, ls="--")
    ax_rv.set_xlabel("Fitted values", fontsize=9, color=DARK)
    ax_rv.set_ylabel("Residuals",     fontsize=9, color=DARK)
    ax_rv.set_title("Residuals vs Fitted\n(heteroscedasticity check)",
                    fontsize=9, fontweight="bold", color=DARK)
    ax_rv.text(0.05, 0.93,
               f"BP p={bp_p:.3f} {'⚠' if bp_p < 0.05 else '✓'}",
               transform=ax_rv.transAxes, fontsize=8,
               color=RED if bp_p < 0.05 else GREEN)

    # 2. Residuals over time
    ax_time.plot(dates, residuals, color=MLE_COLOR, lw=1.2, alpha=0.8)
    ax_time.axhline(0, color=DARK, lw=0.8, ls="--")
    ax_time.fill_between(dates, residuals, color=MLE_COLOR, alpha=0.12)
    # Shade known regime breaks
    for yr, label in [("2019-01-01", "FTP"), ("2020-03-01", "COVID")]:
        ax_time.axvline(pd.Timestamp(yr), color=RED,
                        lw=1.0, ls=":", alpha=0.7)
        ax_time.text(pd.Timestamp(yr), ax_time.get_ylim()[0],
                     label, fontsize=7, color=RED, va="bottom")
    ax_time.set_title("OOS Residuals Over Time",
                      fontsize=9, fontweight="bold", color=DARK)
    ax_time.set_ylabel("Residual", fontsize=9, color=DARK)

    # 3. QQ plot
    (osm, osr), (slope, intercept, r) = probplot(residuals, dist="norm")
    ax_qq.scatter(osm, osr, color=MLE_COLOR, alpha=0.5, s=14, edgecolors="none")
    ax_qq.plot(osm, slope * np.array(osm) + intercept,
               color=DARK, lw=1.2, ls="--")
    ax_qq.set_title(f"QQ Plot — Normality\nJB p={jb_p:.4f} "
                    f"{'⚠' if jb_p < 0.05 else '✓'}",
                    fontsize=9, fontweight="bold", color=DARK)
    ax_qq.set_xlabel("Theoretical quantiles", fontsize=9, color=DARK)
    ax_qq.set_ylabel("Sample quantiles",      fontsize=9, color=DARK)

    # 4. ACF
    plot_acf(residuals, ax=ax_acf, lags=26, color=MLE_COLOR,
             title="", zero=False, alpha=0.05)
    ax_acf.set_title(f"ACF of Residuals\nDW={dw_stat:.3f} "
                     f"{'⚠' if not (1.5 < dw_stat < 2.5) else '✓'}",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_acf.set_xlabel("Lag (weeks)", fontsize=9, color=DARK)
    ax_acf.axhline(0, color=DARK, lw=0.8)

    # 5. Residual histogram
    ax_his.hist(residuals, bins=30, color=MLE_COLOR,
                alpha=0.7, edgecolor="white", linewidth=0.5)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax_his.plot(xr,
                stats.norm.pdf(xr, residuals.mean(), residuals.std())
                * len(residuals) * (residuals.max()-residuals.min()) / 30,
                color=DARK, lw=1.5, ls="--", label="Normal fit")
    ax_his.set_title("Residual Distribution",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_his.set_xlabel("Residual", fontsize=9, color=DARK)
    ax_his.legend(fontsize=8)

    # 6. Rolling R² over folds
    r2_series = mle_results["r2_train"] if "r2_train" in mle_results.columns else None
    if r2_series is not None:
        ax_r2.plot(mle_results.index, r2_series,
                   color=MLE_COLOR, lw=1.4)
        ax_r2.axhline(r2_series.mean(), color=GOLD, lw=1.0,
                      ls="--", label=f"mean={r2_series.mean():.3f}")
        ax_r2.set_title("Train R² per Fold",
                        fontsize=9, fontweight="bold", color=DARK)
        ax_r2.set_ylabel("R²", fontsize=9, color=DARK)
        ax_r2.legend(fontsize=8)

    # 7. AIC over folds
    if "aic" in mle_results.columns:
        ax_aic.plot(mle_results.index, mle_results["aic"],
                    color=MLE_COLOR, lw=1.4)
        ax_aic.set_title("AIC per Fold  (lower = better fit)",
                         fontsize=9, fontweight="bold", color=DARK)
        ax_aic.set_ylabel("AIC", fontsize=9, color=DARK)

    # 8. Coefficient bar chart (top 20 by |t-stat|)
    top20 = coef_df.head(20)
    colors = [GREEN if c > 0 else RED for c in top20["coef"]]
    ax_cof.barh(range(len(top20)), top20["coef"],
                xerr=top20["std_err"] * 1.96,
                color=colors, alpha=0.75, edgecolor="none",
                error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    ax_cof.set_yticks(range(len(top20)))
    ax_cof.set_yticklabels(top20.index, fontsize=8)
    ax_cof.axvline(0, color=DARK, lw=0.8, ls="--")
    ax_cof.set_title("Top 20 Coefficients  (± 95% CI, sorted by |t-stat|)",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_cof.set_xlabel("Coefficient value", fontsize=9, color=DARK)

    fig.suptitle("MLE Linear Regression — Post-Estimation Diagnostics",
                 fontsize=13, fontweight="bold", color=DARK, y=0.975)
    fig.patch.set_facecolor(BG)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    print(f"Saved → {save_path}")
    return tests


def xgb_diagnostics(
    wf_results  : pd.DataFrame,
    wf_models   : list,
    feature_names : list,
    save_path   : str = "xgb_diagnostics.png",
    dpi         : int = 180,
) -> None:
    """
    XGBoost post-estimation diagnostics — no distributional
    assumptions to test, so we focus on importance stability,
    residual patterns, and early stopping behaviour.
    """
    residuals = (wf_results["y_pred"] - wf_results["y_true"]).values
    fitted    = wf_results["y_pred"].values
    dates     = wf_results.index

    # ── Average gain importance across all folds ──────────────────────────
    importance_records = []
    for model in wf_models:
        scores = model.get_booster().get_score(importance_type="gain")
        importance_records.append(scores)

    imp_df = pd.DataFrame(importance_records).fillna(0)
    mean_imp = imp_df.mean().sort_values(ascending=False)

    # ── Ljung-Box on XGB residuals ────────────────────────────────────────
    lb_xgb = acorr_ljungbox(residuals, lags=[1, 4, 8, 13], return_df=True)
    dw_xgb = durbin_watson(residuals)

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  XGBOOST RESIDUAL DIAGNOSTICS")
    print(sep)
    print(f"\n  Autocorrelation — Durbin-Watson: {dw_xgb:.4f}  "
          f"({'⚠ possible autocorrelation' if not (1.5 < dw_xgb < 2.5) else '✓ OK'})")
    print(f"\n  Ljung-Box:")
    print(lb_xgb.to_string())
    print(sep)

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.07)

    ax_imp   = fig.add_subplot(gs[0, :2])  # feature importance bar
    ax_biter = fig.add_subplot(gs[0, 2])   # best_iter histogram
    ax_rv    = fig.add_subplot(gs[1, 0])   # residual vs fitted
    ax_time  = fig.add_subplot(gs[1, 1])   # residuals over time
    ax_acf   = fig.add_subplot(gs[1, 2])   # ACF of residuals
    ax_rstd  = fig.add_subplot(gs[2, :2])  # rolling residual std
    ax_his   = fig.add_subplot(gs[2, 2])   # residual histogram

    for ax in fig.axes:
        ax.set_facecolor(PANEL)

    # 1. Feature importance (top 20)
    top20_imp = mean_imp.head(20)
    ax_imp.barh(range(len(top20_imp)), top20_imp.values,
                color=XGB_COLOR, alpha=0.75, edgecolor="none")
    ax_imp.set_yticks(range(len(top20_imp)))
    ax_imp.set_yticklabels(top20_imp.index, fontsize=8)
    ax_imp.set_title("Top 20 Features — Average Gain Importance (all folds)",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_imp.set_xlabel("Mean gain", fontsize=9, color=DARK)

    # 2. best_iter histogram
    best_iters = [m.best_iteration for m in wf_models]
    ax_biter.hist(best_iters, bins=30, color=XGB_COLOR,
                  alpha=0.75, edgecolor="white", linewidth=0.5)
    ax_biter.axvline(np.mean(best_iters), color=GOLD, lw=1.5,
                     ls="--", label=f"mean={np.mean(best_iters):.0f}")
    ax_biter.set_title("Early Stopping — best_iter Distribution",
                       fontsize=9, fontweight="bold", color=DARK)
    ax_biter.set_xlabel("best_iter", fontsize=9, color=DARK)
    ax_biter.legend(fontsize=8)

    # 3. Residual vs fitted
    ax_rv.scatter(fitted, residuals, color=XGB_COLOR,
                  alpha=0.4, s=18, edgecolors="none")
    ax_rv.axhline(0, color=DARK, lw=1.0, ls="--")
    ax_rv.set_xlabel("Fitted values", fontsize=9, color=DARK)
    ax_rv.set_ylabel("Residuals",     fontsize=9, color=DARK)
    ax_rv.set_title("Residuals vs Fitted\n(heteroscedasticity — descriptive)",
                    fontsize=9, fontweight="bold", color=DARK)

    # 4. Residuals over time
    ax_time.plot(dates, residuals, color=XGB_COLOR, lw=1.2, alpha=0.8)
    ax_time.axhline(0, color=DARK, lw=0.8, ls="--")
    ax_time.fill_between(dates, residuals, color=XGB_COLOR, alpha=0.10)
    for yr, label in [("2019-01-01", "FTP"), ("2020-03-01", "COVID")]:
        ax_time.axvline(pd.Timestamp(yr), color=RED,
                        lw=1.0, ls=":", alpha=0.7)
        ax_time.text(pd.Timestamp(yr), 0, label,
                     fontsize=7, color=RED, va="bottom")
    ax_time.set_title("OOS Residuals Over Time",
                      fontsize=9, fontweight="bold", color=DARK)
    ax_time.set_ylabel("Residual", fontsize=9, color=DARK)

    # 5. ACF
    plot_acf(residuals, ax=ax_acf, lags=26, color=XGB_COLOR,
             title="", zero=False, alpha=0.05)
    ax_acf.set_title(f"ACF of Residuals\nDW={dw_xgb:.3f} "
                     f"{'⚠' if not (1.5 < dw_xgb < 2.5) else '✓'}",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_acf.set_xlabel("Lag (weeks)", fontsize=9, color=DARK)

    # 6. Rolling residual std (12-week window) — heteroscedasticity proxy
    resid_s   = pd.Series(residuals, index=dates)
    roll_std  = resid_s.rolling(12).std()
    ax_rstd.plot(dates, roll_std, color=XGB_COLOR, lw=1.4)
    ax_rstd.axhline(roll_std.mean(), color=GOLD, lw=1.0, ls="--",
                    label=f"mean={roll_std.mean():.4f}")
    ax_rstd.set_title("Rolling 12-Week Residual Std  (variance clustering check)",
                      fontsize=9, fontweight="bold", color=DARK)
    ax_rstd.set_ylabel("Std of residuals", fontsize=9, color=DARK)
    ax_rstd.legend(fontsize=8)
    for yr, label in [("2019-01-01", "FTP"), ("2020-03-01", "COVID")]:
        ax_rstd.axvline(pd.Timestamp(yr), color=RED,
                        lw=1.0, ls=":", alpha=0.7)

    # 7. Residual histogram
    ax_his.hist(residuals, bins=30, color=XGB_COLOR,
                alpha=0.75, edgecolor="white", linewidth=0.5)
    ax_his.set_title("Residual Distribution",
                     fontsize=9, fontweight="bold", color=DARK)
    ax_his.set_xlabel("Residual", fontsize=9, color=DARK)

    fig.suptitle("XGBoost — Post-Estimation Diagnostics",
                 fontsize=13, fontweight="bold", color=DARK, y=0.975)
    fig.patch.set_facecolor(BG)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    print(f"Saved → {save_path}")


def plot_residuals_comparison(
    mle_results : pd.DataFrame,
    xgb_results : pd.DataFrame,
    save_path   : str = "residuals_comparison.png",
    dpi         : int = 180,
) -> None:
    """
    Both models' OOS residuals on one chart.
    Shared x-axis makes regime clustering visible.
    """
    # NOTE: the notebook read the global `wf_results_final` here and ignored
    # its own `xgb_results` argument. It looked correct only because the caller
    # happened to pass that same object.
    mle_res = (mle_results["y_pred"] - mle_results["y_true"])
    xgb_res = (xgb_results["y_pred"] - xgb_results["y_true"])

    shared = mle_res.index.intersection(xgb_res.index)
    mle_res = mle_res.loc[shared]
    xgb_res = xgb_res.loc[shared]

    fig, axes = plt.subplots(2, 1, figsize=(18, 8),
                             sharex=True, facecolor=BG)
    fig.subplots_adjust(hspace=0.12, left=0.06, right=0.97,
                        top=0.92, bottom=0.08)

    for ax, res, color, label in [
        (axes[0], mle_res, MLE_COLOR, "MLE Linear Regression"),
        (axes[1], xgb_res, XGB_COLOR, "XGBoost"),
    ]:
        ax.set_facecolor(PANEL)
        ax.plot(shared, res.values, color=color, lw=1.2, alpha=0.85)
        ax.fill_between(shared, res.values, color=color, alpha=0.10)
        ax.axhline(0, color=DARK, lw=0.8, ls="--")
        ax.axhline(res.std(),  color=GOLD, lw=0.8, ls=":", alpha=0.7)
        ax.axhline(-res.std(), color=GOLD, lw=0.8, ls=":", alpha=0.7,
                   label=f"±1σ = {res.std():.4f}")
        for yr, lbl in [("2019-01-01", "FTP"), ("2020-03-01", "COVID")]:
            ax.axvline(pd.Timestamp(yr), color=RED, lw=1.0, ls=":", alpha=0.7)
            ax.text(pd.Timestamp(yr), res.max() * 0.85, lbl,
                    fontsize=7.5, color=RED, va="top")
        ax.set_ylabel("Residual", fontsize=10, color=DARK)
        ax.set_title(f"{label} — OOS Residuals",
                     fontsize=10, fontweight="bold", color=DARK)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER)

    fig.suptitle("OOS Residuals Comparison — MLE vs XGBoost",
                 fontsize=13, fontweight="bold", color=DARK, y=0.975)
    fig.patch.set_facecolor(BG)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    print(f"Saved → {save_path}")
