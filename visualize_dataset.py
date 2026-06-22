"""
Visualize a saved toy training dataset (.pt file).

Loads the dict written by `generate_dataset.py` (keys: c, X, y) and produces a set
of diagnostic plots of the NOMINAL dataset:

  1.  x-space 2D density per class                                 → ds_x_2d.{pdf,png}
  2.  x-space 1D marginals per class (x₁, x₂)                      → ds_x_1d.{pdf,png}
  3.  y-space 2D density per class                                 → ds_y_2d.{pdf,png}
  4.  y-space 1D marginals per class (y₁, y₂)                      → ds_y_1d.{pdf,png}
  5.  y-space binned in x₁ — sees the x-conditioning of p(y|x,c)   → ds_y_xbinned.{pdf,png}
  6.  stacked histograms of the total mixture (x and y)            → ds_stacked_x.{pdf,png},
                                                                      ds_stacked_y.{pdf,png}
  7.  class balance / sample counts                                → printed to stdout

Usage:
    python visualize_dataset.py data/dataset.pt
    python visualize_dataset.py data/dataset.pt --out-dir validation/dataset
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch


CLASS_COLORS = {0: "dodgerblue", 1: "darkorange"}
CLASS_LABELS = {0: "Class A", 1: "Class B"}


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved {name}")
    plt.close(fig)


def _class_idx(c):
    """one-hot [N, 2]  →  int [N] in {0, 1}."""
    arr = c.cpu().numpy() if torch.is_tensor(c) else c
    if arr.ndim == 1:
        return arr.astype(int)
    return arr[:, 1].astype(int)


def _contour2d(ax, arr, bins=60, range2d=None, smooth=1.2, cmap=None, levels=7, alpha=0.6):
    """Filled smoothed 2D histogram. Returns (xc, yc, h_smoothed)."""
    h, xe, ye = np.histogram2d(arr[:, 0], arr[:, 1], bins=bins,
                                range=list(range2d) if range2d is not None else None,
                                density=True)
    h = gaussian_filter(h, sigma=smooth)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    if h.max() > 0:
        lvls = np.linspace(h.max() * 0.04, h.max() * 0.95, levels)
        ax.contourf(xc, yc, h.T, levels=lvls, cmap=cmap, alpha=alpha)
    return xc, yc, h


def _auto_range_2d(*arrays_2d, pct=(0.1, 99.9), pad_frac=0.15):
    """Per-dim percentile range for 2D plots — returns ((x_lo, x_hi), (y_lo, y_hi)).

    Default (0.1, 99.9) + 15% pad sits comfortably outside the outer (4% × max
    density) contour so the smoothed cluster envelope is never clipped.
    """
    out = []
    for d in range(2):
        vals = np.concatenate([np.asarray(a)[:, d] for a in arrays_2d if a is not None])
        lo, hi = np.percentile(vals, pct)
        pad = pad_frac * (hi - lo + 1e-6)
        out.append((float(lo - pad), float(hi + pad)))
    return tuple(out)


def _auto_range_1d(*arrays_1d, pct=(0.1, 99.9), pad_frac=0.10):
    """Single-axis percentile range from any number of 1D arrays."""
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays_1d if a is not None])
    lo, hi = np.percentile(vals, pct)
    pad = pad_frac * (hi - lo + 1e-6)
    return float(lo - pad), float(hi + pad)


def _y_range(y):
    lo = float(y.min())
    hi = float(y.max())
    pad = 0.05 * (hi - lo + 1e-6)
    return ((lo - pad, hi + pad),) * 2


def plot_x_2d(c_idx, X, out_dir):
    """2D distributions of x per class (filled contours)."""
    range_2d = _auto_range_2d(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cl in [0, 1]:
        mask = c_idx == cl
        ax = axes[cl]
        cmap = "Blues" if cl == 0 else "Oranges"
        _contour2d(ax, X[mask], bins=80, range2d=range_2d, smooth=1.2, cmap=cmap, alpha=0.6)
        ax.set_xlim(*range_2d[0]); ax.set_ylim(*range_2d[1]); ax.set_aspect("equal")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"{CLASS_LABELS[cl]} — p(x | c)")
        ax.legend(handles=[Patch(facecolor=CLASS_COLORS[cl], alpha=0.6, label="x (nominal)")],
                  fontsize=9)
    plt.suptitle("Dataset — x distribution per class", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "ds_x_2d")


def plot_x_1d(c_idx, X, out_dir):
    # Per-dim percentile range so each marginal zooms to its actual support.
    bins_dim = {dim: np.linspace(*_auto_range_1d(X[:, dim]), 60) for dim in (0, 1)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for cl in [0, 1]:
        mask = c_idx == cl
        for dim in [0, 1]:
            ax = axes[cl, dim]
            bins = bins_dim[dim]
            ax.hist(X[mask, dim], bins=bins, density=True, histtype="stepfilled",
                    alpha=0.4, color=CLASS_COLORS[cl], label="x (nominal)")
            ax.set_title(f"{CLASS_LABELS[cl]} — x{'₁' if dim == 0 else '₂'}")
            ax.set_xlabel(f"x{'₁' if dim == 0 else '₂'}")
            ax.set_ylabel("density")
            ax.set_xlim(bins[0], bins[-1])
            ax.legend(fontsize=9)
    plt.suptitle("Dataset — x 1D marginals", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "ds_x_1d")


def plot_y_2d(c_idx, y, out_dir):
    range_y = _y_range(y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cl in [0, 1]:
        mask = c_idx == cl
        ax = axes[cl]
        cmap = "Blues" if cl == 0 else "Oranges"
        _contour2d(ax, y[mask], bins=80, range2d=range_y, smooth=1.5, cmap=cmap, alpha=0.6)
        ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
        ax.set_xlabel("y₁"); ax.set_ylabel("y₂")
        ax.set_title(f"{CLASS_LABELS[cl]} — p(y | c)")
        ax.legend(handles=[Patch(facecolor=CLASS_COLORS[cl], alpha=0.6, label="y (nominal)")],
                  fontsize=9)
    plt.suptitle("Dataset — y distribution per class", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "ds_y_2d")


def plot_y_1d(c_idx, y, out_dir):
    range_y = _y_range(y)
    bins = np.linspace(range_y[0][0], range_y[0][1], 60)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for cl in [0, 1]:
        mask = c_idx == cl
        for dim in [0, 1]:
            ax = axes[cl, dim]
            ax.hist(y[mask, dim], bins=bins, density=True, histtype="stepfilled",
                    alpha=0.4, color=CLASS_COLORS[cl], label="y (nominal)")
            ax.set_title(f"{CLASS_LABELS[cl]} — y{'₁' if dim == 0 else '₂'}")
            ax.set_xlabel(f"y{'₁' if dim == 0 else '₂'}")
            ax.set_ylabel("density")
            ax.legend(fontsize=9)
    plt.suptitle("Dataset — y 1D marginals", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "ds_y_1d")


def plot_y_xbinned(c_idx, X, y, out_dir):
    """For each class, slice events by x₁ band (with |x₂|<0.5) and plot y density.
    Shows how p(y|x,c) varies with x₁."""
    x1_edges = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    n_bins = len(x1_edges) - 1
    range_y = _y_range(y)
    x2_central = np.abs(X[:, 1]) < 0.5

    fig, axes = plt.subplots(2, n_bins, figsize=(4 * n_bins, 8))
    for cl in [0, 1]:
        for xb in range(n_bins):
            xl, xr = x1_edges[xb], x1_edges[xb + 1]
            mask = ((X[:, 0] >= xl) & (X[:, 0] < xr) & x2_central & (c_idx == cl))
            ax = axes[cl, xb]
            if mask.sum() > 50:
                cmap = "Blues" if cl == 0 else "Oranges"
                _contour2d(ax, y[mask], bins=40, range2d=range_y, smooth=1.5,
                           cmap=cmap, alpha=0.7)
            ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
            ax.set_xlabel("y₁", fontsize=8); ax.set_ylabel("y₂", fontsize=8)
            ax.set_title(f"{CLASS_LABELS[cl]}  x₁∈[{xl:.1f},{xr:.1f}), |x₂|<0.5", fontsize=9)
    plt.suptitle("Dataset — p(y | x₁, c)  (|x₂|<0.5 slices)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "ds_y_xbinned")


def plot_stacked_1d(c_idx, arr, dim_names, bin_range, out_dir, fname, title):
    """Stacked 1D histograms of the total mixture decomposed into class A + class B.

    arr: [N, D] nominal samples (e.g. X or y).
    bin_range: either a single (lo, hi) tuple shared across dims, or None to auto-zoom
        per dim from the data percentiles. Each axis of `arr` gets its own subplot.
    Shows counts (not density) so the stacking is meaningful as event yield.
    """
    n_dim = arr.shape[1]
    # Per-dim bins: auto from percentiles if no fixed range given.
    if bin_range is None:
        bins_dim = {dim: np.linspace(*_auto_range_1d(arr[:, dim]), 60) for dim in range(n_dim)}
    else:
        shared = np.linspace(bin_range[0], bin_range[1], 60)
        bins_dim = {dim: shared for dim in range(n_dim)}
    fig, axes = plt.subplots(1, n_dim, figsize=(6 * n_dim, 5), squeeze=False)
    axes = axes[0]
    for dim in range(n_dim):
        ax = axes[dim]
        bins = bins_dim[dim]
        arrs = [arr[c_idx == 0, dim], arr[c_idx == 1, dim]]
        labels = [f"{CLASS_LABELS[0]} ({len(arrs[0])})",
                  f"{CLASS_LABELS[1]} ({len(arrs[1])})"]
        ax.hist(arrs, bins=bins, stacked=True, histtype="stepfilled",
                color=[CLASS_COLORS[0], CLASS_COLORS[1]],
                alpha=0.75, edgecolor="white", linewidth=0.4,
                label=labels)
        ax.set_xlabel(dim_names[dim])
        ax.set_ylabel("Events / bin")
        ax.set_xlim(bins[0], bins[-1])
        ax.legend(fontsize=9, loc="best")
        ax.set_title(f"Stacked — {dim_names[dim]}")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, fname)


def main():
    parser = argparse.ArgumentParser(description="Visualize a toy training dataset (.pt).")
    parser.add_argument("dataset", help="Path to the dataset .pt file (e.g. data/dataset.pt)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: validation/<dataset-basename>/)")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(args.dataset)
    out_dir = args.out_dir or os.path.join(
        "validation", os.path.splitext(os.path.basename(args.dataset))[0]
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {args.dataset}")
    print(f"Output:  {out_dir}/")

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    expected = {"c", "X", "y"}
    missing = expected - set(data.keys())
    if missing:
        raise KeyError(f"dataset missing required keys: {missing}")

    c = data["c"]
    X = data["X"].cpu().numpy() if torch.is_tensor(data["X"]) else np.asarray(data["X"])
    y = data["y"].cpu().numpy() if torch.is_tensor(data["y"]) else np.asarray(data["y"])

    c_idx = _class_idx(c)
    n_total = len(c_idx); n_A = int((c_idx == 0).sum()); n_B = int((c_idx == 1).sum())

    print("\n--- Dataset summary ---")
    print(f"N total      = {n_total}")
    print(f"N class A    = {n_A}  ({100 * n_A / n_total:.2f}%)")
    print(f"N class B    = {n_B}  ({100 * n_B / n_total:.2f}%)")
    print(f"X shape      = {X.shape}      range x₁ [{X[:, 0].min():+.3f}, {X[:, 0].max():+.3f}]"
          f"   x₂ [{X[:, 1].min():+.3f}, {X[:, 1].max():+.3f}]")
    print(f"y shape      = {y.shape}      range y₁ [{y[:, 0].min():+.3f}, {y[:, 0].max():+.3f}]"
          f"   y₂ [{y[:, 1].min():+.3f}, {y[:, 1].max():+.3f}]")

    for cl in [0, 1]:
        mask = c_idx == cl
        print(f"  {CLASS_LABELS[cl]}: ⟨x⟩ = ({X[mask, 0].mean():+.3f}, {X[mask, 1].mean():+.3f}),"
              f"  σ(x) = ({X[mask, 0].std():.3f}, {X[mask, 1].std():.3f})"
              f"  ⟨y⟩ = ({y[mask, 0].mean():+.3f}, {y[mask, 1].mean():+.3f})")
    print()

    plot_x_2d(c_idx, X, out_dir)
    plot_x_1d(c_idx, X, out_dir)
    plot_y_2d(c_idx, y, out_dir)
    plot_y_1d(c_idx, y, out_dir)
    plot_y_xbinned(c_idx, X, y, out_dir)

    # Stacked total-mixture histograms (counts, decomposed by class).
    plot_stacked_1d(c_idx, X, ("x₁", "x₂"), None,
                    out_dir, "ds_stacked_x", "Dataset — stacked x distribution (A+B)")
    y_lo, y_hi = float(y.min()), float(y.max())
    pad = 0.05 * (y_hi - y_lo + 1e-6)
    plot_stacked_1d(c_idx, y, ("y₁", "y₂"), (y_lo - pad, y_hi + pad),
                    out_dir, "ds_stacked_y", "Dataset — stacked y distribution (A+B)")

    print(f"\nDone. All plots in {out_dir}/")


if __name__ == "__main__":
    main()
