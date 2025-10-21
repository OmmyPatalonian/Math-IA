

import argparse, os, math, random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def logistic(x, L, k, x0):
    # standard logistic curve equation
    return L / (1.0 + np.exp(-k * (x - x0)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def bin_stats(df, wpm_col, acc_col, bin_width=10):
    x = df[wpm_col].to_numpy(float)
    y = df[acc_col].to_numpy(float)

    # Define bins covering data range
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    # Expand edges to full bin coverage
    start = bin_width * math.floor(xmin / bin_width)
    end   = bin_width * math.ceil(xmax / bin_width)
    edges = np.arange(start, end + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.digitize(x, edges) - 1  # bin indices in [0, len(edges)-2]

    medians, p90s, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if np.any(mask):
            vals = y[mask]
            medians.append(np.nanmedian(vals))
            p90s.append(np.nanpercentile(vals, 90))
            counts.append(int(np.sum(mask)))
        else:
            medians.append(np.nan)
            p90s.append(np.nan)
            counts.append(0)

    medians = np.array(medians, float)
    p90s = np.array(p90s, float)
    counts = np.array(counts, int)
    return centers, medians, p90s, counts, edges



def fit_logistic_binned(xc, ym, L=100.0) -> Tuple[float, float]:
    """
    ### fit logistic curve to binned data using grid search + local optimization
    returns best k and x0 parameters for given L
    """
    ### remove nan values before fitting
    mask = ~np.isnan(xc) & ~np.isnan(ym)
    x = np.asarray(xc)[mask]
    y = np.asarray(ym)[mask]
    if len(x) < 3:
        # Not enough points; default mild slope & center
        return 0.03, float(np.nanmedian(x) if len(x) else 50.0)

    # Reasonable search ranges
    k_grid = np.linspace(0.005, 0.15, 40)  # slope
    x0_grid = np.linspace(np.nanmin(x) - 50, np.nanmax(x) + 50, 60)

    best = (None, None, np.inf)
    for k in k_grid:
        pred = None
        for x0 in x0_grid:
            pred = logistic(x, L, k, x0)
            sse = np.sum((y - pred) ** 2)
            if sse < best[2]:
                best = (k, x0, sse)

    k, x0, _ = best

  
    rng = np.random.default_rng(42)
    T0 = 1.0
    for it in range(2000):
        k_new = abs(k + rng.normal(0, 0.002))
        x0_new = x0 + rng.normal(0, 2.0)
        pred = logistic(x, L, k_new, x0_new)
        sse_new = np.sum((y - pred) ** 2)
        pred_old = logistic(x, L, k, x0)
        sse_old = np.sum((y - pred_old) ** 2)
        if sse_new < sse_old or rng.random() < math.exp(-(sse_new - sse_old) / max(T0,1e-8)):
            k, x0 = k_new, x0_new
        # cool down
        T0 *= 0.9995
        if T0 < 1e-3:
            T0 = 1e-3

    return float(k), float(x0)

def bootstrap_ci_band(df, wpm_col, acc_col, k, x0, L, bin_width, B=500, x_grid=None, seed=123):
    rng = np.random.default_rng(seed)
    n = len(df)
    if x_grid is None:
        x_grid = np.linspace(df[wpm_col].min(), df[wpm_col].max(), 200)

    preds = np.zeros((B, len(x_grid)), dtype=float)
    ### bootstrap resampling to get confidence intervals
    for b in range(B):
      
        idx = rng.integers(0, n, size=n)
        samp = df.iloc[idx]
    
        xc, ym, _, _, _ = bin_stats(samp, wpm_col, acc_col, bin_width)
        kk, xx0 = fit_logistic_binned(xc, ym, L=L)
        preds[b, :] = logistic(x_grid, L, kk, xx0)

    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    return x_grid, lo, hi


def plot_residuals_vs_fitted(xc, ym, L, k, x0, outpath):
    # Pred at bin centers
    yhat = logistic(xc, L, k, x0)
    resid = ym - yhat
    r = rmse(ym, yhat)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(yhat, resid)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted accuracy (Â)")
    plt.ylabel("Residual (median - Â)")
    plt.title(f"Residuals vs Fitted (RMSE = {r:.2f})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_logistic_with_ci(df, wpm_col, acc_col, xc, ym, L, k, x0, bin_width, B, outpath):
    
    x_grid = np.linspace(df[wpm_col].min(), df[wpm_col].max(), 400)
    y_fit = logistic(x_grid, L, k, x0)
    # Bootstrap band
    xg, lo, hi = bootstrap_ci_band(df, wpm_col, acc_col, k, x0, L, bin_width, B=B, x_grid=x_grid)
    fig = plt.figure(figsize=(7, 5))
    # CI band
    plt.fill_between(xg, lo, hi, alpha=0.25, label="95% bootstrap CI")
    plt.plot(x_grid, y_fit, label="Fitted logistic")
    # Binned medians
    plt.scatter(xc, ym, label="Bin medians")
    plt.ylim(0, 100)
    plt.xlabel("WPM (x)")
    plt.ylabel("Accuracy A(x)")
    plt.title("Accuracy vs WPM with 95% Bootstrap CI")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_counts_with_overlays(edges, counts, medians, p90s, outpath):
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (edges[1] - edges[0]) * 0.9

    fig = plt.figure(figsize=(8, 5))
    # Bars: counts per bin
    plt.bar(centers, counts, width=width, alpha=0.7, label="Count per bin")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(centers, medians, marker="o", label="Median accuracy")
    ax2.plot(centers, p90s, marker="s", linestyle="--", label="P90 accuracy")
    ax.set_xlabel("WPM bins (center)")
    ax.set_ylabel("Count")
    ax2.set_ylabel("Accuracy (%)")
    ax.set_title("Counts per Bin with Median and P90 Overlays")

    # Build a combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with typing data")
    ap.add_argument("--wpm-col", default="wpm", help="Column name for WPM")
    ap.add_argument("--acc-col", default="accuracy", help="Column name for Accuracy (0–100)")
    ap.add_argument("--bin-width", type=float, default=10.0, help="Bin width in WPM")
    ap.add_argument("--boot", type=int, default=500, help="Bootstrap iterations for CI")
    ap.add_argument("--outdir", default="ia_plots", help="Output directory for figures")
    ap.add_argument("--L", type=float, default=100.0, help="Upper bound (L) for accuracy")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[[args.wpm_col, args.acc_col]].dropna()
    df[args.acc_col] = df[args.acc_col].clip(lower=0, upper=args.L)

    #bin stats
    xc, med, p90, counts, edges = bin_stats(df, args.wpm_col, args.acc_col, bin_width=args.bin_width)
    k, x0 = fit_logistic_binned(xc, med, L=args.L)


    plot_residuals_vs_fitted(xc, med, args.L, k, x0, os.path.join(args.outdir, "residuals_vs_fitted.png"))

    # logistic with 95% bootstrap CI and medians
    plot_logistic_with_ci(df, args.wpm_col, args.acc_col, xc, med, args.L, k, x0, args.bin_width, args.boot,
                          os.path.join(args.outdir, "logistic_with_bootstrap_CI.png"))

    #counts-per-bin bars + median + P90 overlays
    plot_counts_with_overlays(edges, counts, med, p90,
                              os.path.join(args.outdir, "counts_per_bin_with_median_p90.png"))


    print(f"Fitted logistic parameters: L={args.L:.3f}, k={k:.5f}, x0={x0:.3f}")
    # RMSE on binned medians
    yhat = logistic(xc, args.L, k, x0)
    r = rmse(med, yhat)
    print(f"RMSE on binned medians: {r:.3f}")
    print(f"Saved figures to: {args.outdir}")

if __name__ == "__main__":
    main()
