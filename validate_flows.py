"""
Validate trained base and systematic flow models against the generator truth.

Base flows (--cfg configs/base_flows.yaml):
  - kin model:   p(x | c)      — 2D kinematics, compare model samples vs generator per class
  - score model: p(y | x, c)   — compare model samples vs generator, x-binned

Systematic residuals (--sys-cfg configs/systematics.yaml):
  - residual kin: scans `kin_systematics` list, evaluates the residual at each m-vector
    against the matching `generator_*`.
  - residual score: scans `score_systematics`, samples the residual score model
    p(y | x, c, ν) at each m-vector conditioned on the variation generator's (c, x), and
    compares to that generator's truth y. Skipped automatically if the score residual is
    identity (`residual_score_model.num_nuisances == 0` or no `score_systematics`).

Usage:
    python validate_flows.py -c configs/base_flows.yaml
    python validate_flows.py -c configs/base_flows.yaml --sys-cfg configs/systematics.yaml
    python validate_flows.py -c configs/base_flows.yaml --out-dir validation/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml
import zuko

from generator import ParametricLikelihoodDataset
from residual_flow import SystematicCorrectedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(p, d):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(d, p))


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved {name}")
    plt.close(fig)


def _ratio_grid(nrows, ncols, figsize, main_h=3, ratio_h=1):
    """Figure with interleaved main+ratio rows. Returns (fig, main[nrows,ncols], ratio[nrows,ncols])."""
    heights = []
    for _ in range(nrows):
        heights.extend([main_h, ratio_h])
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(nrows * 2, ncols, figure=fig, height_ratios=heights, hspace=0.05)
    main_ax  = np.empty((nrows, ncols), dtype=object)
    ratio_ax = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            main_ax[r, c]  = fig.add_subplot(gs[r * 2,     c])
            ratio_ax[r, c] = fig.add_subplot(gs[r * 2 + 1, c], sharex=main_ax[r, c])
            main_ax[r, c].tick_params(labelbottom=False)
    return fig, main_ax, ratio_ax


def _draw_ratio(ax, bins, arr_model, arr_truth, color="k"):
    """Model/Truth ratio with Poisson errors. Both arrays are raw 1D samples."""
    h_m, _ = np.histogram(arr_model, bins=bins)
    h_t, _ = np.histogram(arr_truth, bins=bins)
    n_m = max(len(arr_model), 1)
    n_t = max(len(arr_truth), 1)
    bc   = 0.5 * (bins[:-1] + bins[1:])
    safe = np.where(h_t > 0, h_t / n_t, np.nan)
    ratio = (h_m / n_m) / safe
    err   = ratio / np.sqrt(np.where(h_m > 0, h_m, 1))
    ax.errorbar(bc, ratio, yerr=err, fmt=".", color=color, ms=3, lw=1)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.fill_between(bc, 0.9, 1.1, alpha=0.1, color="gray")
    ax.set_ylim(0.5, 1.5)
    ax.set_ylabel("M/T", fontsize=7)
    ax.tick_params(labelsize=7)


def _auto_range_2d(*arrays_2d, pct=(0.5, 99.5), pad_frac=0.05):
    """Per-axis percentile range from one or more [N, 2] arrays.
    Returns ((lo0, hi0), (lo1, hi1)) with a small symmetric pad."""
    out = []
    for d in range(2):
        vals = np.concatenate([np.asarray(a)[:, d] for a in arrays_2d if a is not None])
        lo, hi = np.percentile(vals, pct)
        pad = pad_frac * (hi - lo + 1e-6)
        out.append((float(lo - pad), float(hi + pad)))
    return tuple(out)


def _auto_range_1d(*arrays_1d, pct=(0.5, 99.5), pad_frac=0.05):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays_1d if a is not None])
    lo, hi = np.percentile(vals, pct)
    pad = pad_frac * (hi - lo + 1e-6)
    return float(lo - pad), float(hi + pad)


def _contour2d(ax, arr, bins=60, range2d=((-4, 4), (-4, 4)), smooth=1.5, filled=False,
               own_levels=False, n_own=6, **kwargs):
    """Smoothed 2D histogram contour from raw samples [N, 2].

    own_levels=True recomputes the contour levels from THIS sample's own peak
    (0.04–0.95 × max) instead of using levels passed in kwargs. Use it for an
    overlaid model/reference so a difference in peak height vs the (filled) truth
    doesn't render the overlay as spuriously broad — only the contour shapes are
    compared, not absolute density."""
    h, xe, ye = np.histogram2d(arr[:, 0], arr[:, 1], bins=bins,
                                range=list(range2d), density=True)
    h = gaussian_filter(h, sigma=smooth)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    if own_levels and h.max() > 0:
        kwargs["levels"] = np.linspace(h.max() * 0.04, h.max() * 0.95, n_own)
    return (ax.contourf if filled else ax.contour)(xc, yc, h.T, **kwargs)


def _make_gen(gcfg, device):
    kwargs = dict(device=device)
    for key in ("center_A", "center_B", "sigma_A", "sigma_B", "shift_dir"):
        if key in gcfg:
            kwargs[key] = tuple(gcfg[key])
    for key in ("variation_shift", "variation_rot", "variation_squeeze",
                "shift_scale", "rot_scale", "squeeze_scale",
                "y_shift_scale", "y_squeeze_scale", "distortion_strength",
                "distortion_shift_scale", "distortion_squeeze_scale"):
        if key in gcfg:
            kwargs[key] = float(gcfg[key])
    if "sigmoid_y" in gcfg:
        kwargs["sigmoid_y"] = bool(gcfg["sigmoid_y"])
    return ParametricLikelihoodDataset(**kwargs)


def _class_idx(c):
    """Integer class array [N] from one-hot float [N, K] tensor.
    Column 0 == 1  →  class 0;  column 1 == 1  →  class 1."""
    return c.cpu().numpy()[:, 1].astype(int)


def _gen_nominal(gen, n, device):
    """Returns c [N, 2] float one-hot, X [N, 2], y [N, 2] from nominal generator."""
    with torch.no_grad():
        c, X, y, _, _ = gen.generate_batch(n, distorsion=False)
    return c.float().to(device), X.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Section 1: Kin model  p(x | c)   — 2D
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_kin(cfg, device, out_dir, n=100_000):
    print("Validating kin model …")
    gen = _make_gen(cfg["generator"], device)
    c_true, X_true, _ = _gen_nominal(gen, n, device)

    kin_cfg = cfg["kin_flow"]
    model = zuko.flows.NSF(
        features=kin_cfg["features"], context=kin_cfg["context"],
        bins=kin_cfg["bins"], transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["kin_model"], map_location=device))
    model.eval()

    # c_true is already one-hot [N, 2]
    x_model_t = model(c_true).rsample()
    x_model   = x_model_t.cpu().numpy()         # [N, 2]
    x_true_np = X_true.cpu().numpy()            # [N, 2]
    c_np      = _class_idx(c_true)

    colors = {0: "dodgerblue", 1: "darkorange"}
    labels = {0: "Class A", 1: "Class B"}

    # --- 1a: 2D scatter / contours per class ---
    # Per-axis percentile-zoom so we don't waste 80 % of the canvas on whitespace.
    range_2d = _auto_range_2d(x_true_np, x_model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cl in [0, 1]:
        mask = c_np == cl
        ax = axes[cl]
        h_t, xe, ye = np.histogram2d(
            x_true_np[mask, 0], x_true_np[mask, 1],
            bins=60, range=list(range_2d), density=True,
        )
        h_t = gaussian_filter(h_t, sigma=1.2)
        if h_t.max() > 0:
            lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 7)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            ax.contourf(xc, yc, h_t.T, levels=lvls,
                        cmap="Blues" if cl == 0 else "Oranges", alpha=0.6)
            _contour2d(ax, x_model[mask], bins=60, range2d=range_2d, smooth=1.2,
                       filled=False, colors=["k"], linewidths=1.5,
                       linestyles="--", levels=lvls)
        ax.legend(handles=[
            Patch(facecolor=colors[cl], alpha=0.6, label="Truth"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="Model"),
        ], fontsize=9)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"Kin model — {labels[cl]}  p(x | c)")
        ax.set_xlim(*range_2d[0]); ax.set_ylim(*range_2d[1]); ax.set_aspect("equal")
    plt.suptitle("Kin model 2D validation  p(x | c)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_kin_2d")

    # --- 1b: 1D marginals (x₁ and x₂) per class with ratio ---
    # Per-dim percentile-zoom so each axis fits the actual support.
    bins_dim = {
        dim: np.linspace(*_auto_range_1d(x_true_np[:, dim], x_model[:, dim]), 60)
        for dim in (0, 1)
    }
    fig, axes_m, axes_r = _ratio_grid(2, 2, figsize=(12, 10))
    for cl in [0, 1]:
        mask = c_np == cl
        for dim in [0, 1]:
            bins = bins_dim[dim]
            ax = axes_m[cl, dim]
            ax.hist(x_true_np[mask, dim], bins=bins, density=True,
                    histtype="stepfilled", alpha=0.35, color=colors[cl], label="Truth")
            ax.hist(x_model[mask, dim],   bins=bins, density=True,
                    histtype="step", lw=2, color=colors[cl], label="Model")
            ax.set_title(f"{labels[cl]} — x{'₁' if dim == 0 else '₂'}")
            ax.legend(fontsize=9)
            ax.set_xlim(bins[0], bins[-1])
            _draw_ratio(axes_r[cl, dim], bins,
                        x_model[mask, dim], x_true_np[mask, dim], color=colors[cl])
            axes_r[cl, dim].set_xlabel(f"x{'₁' if dim == 0 else '₂'}")
    plt.suptitle("Kin model 1D marginals  p(x | c)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_kin_1d_marginals")


# ---------------------------------------------------------------------------
# Section 2: Score model  p(y | x, c)   — context = c_onehot(2) + 2D x
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_score(cfg, device, out_dir, n=100_000):
    print("Validating score model …")
    gen = _make_gen(cfg["generator"], device)
    c_true, X_true, y_true = _gen_nominal(gen, n, device)

    score_cfg = cfg["score_flow"]
    model = zuko.flows.NSF(
        features=score_cfg["features"], context=score_cfg["context"],
        bins=score_cfg["bins"], transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["score_model"], map_location=device))
    model.eval()

    # context = [c_onehot, X] → shape [N, 2 + 2] = [N, 4] in v12
    context  = torch.cat([c_true, X_true], dim=-1)
    y_model  = model(context).rsample().cpu().numpy()

    y_true_np = y_true.cpu().numpy()
    c_np      = _class_idx(c_true)
    x_np      = X_true.cpu().numpy()             # [N, 2]

    colors = {0: "dodgerblue", 1: "darkorange"}
    labels = {0: "Class A", 1: "Class B"}

    # In v12 y is in (0, 1)² (sigmoid). Percentile-based per-axis zoom so a single
    # outlier sample from the model can't blow the range out to (-4, 1) like before.
    range_y = _auto_range_2d(y_true_np, y_model)

    # --- 2a: Global 2D contours per class (truth filled + model dashed lines) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cl in [0, 1]:
        mask = c_np == cl
        ax = axes[cl]
        h_t, xe, ye = np.histogram2d(y_true_np[mask, 0], y_true_np[mask, 1],
                                      bins=60, range=list(range_y), density=True)
        h_t = gaussian_filter(h_t, sigma=1.5)
        if h_t.max() > 0:
            lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 7)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            ax.contourf(xc, yc, h_t.T, levels=lvls,
                        cmap="Blues" if cl == 0 else "Oranges", alpha=0.6)
            _contour2d(ax, y_model[mask], bins=60, range2d=range_y, smooth=1.5,
                       filled=False, colors=["k"], linewidths=1.5,
                       linestyles="--", levels=lvls)
        ax.legend(handles=[
            Patch(facecolor=colors[cl], alpha=0.6, label="Truth"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="Model"),
        ], fontsize=9)
        ax.set_xlabel("y₁"); ax.set_ylabel("y₂")
        ax.set_title(f"{labels[cl]} — p(y | c)")
        ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
    plt.suptitle("Score model — 2D contours  p(y | c)  (marginalised over x)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_score_2d_global")

    # --- 2a-bis: 2D contours binned on x₁ (at central x₂ ≈ 0) and on x₂ (at central x₁ ≈ 0) ---
    x1_edges = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    n_x1bins = len(x1_edges) - 1
    x2_central = (np.abs(x_np[:, 1]) < 0.5)
    fig, axes = plt.subplots(2, n_x1bins, figsize=(4 * n_x1bins, 8))
    for cl in [0, 1]:
        for xb in range(n_x1bins):
            xl, xr = x1_edges[xb], x1_edges[xb + 1]
            mask = ((x_np[:, 0] >= xl) & (x_np[:, 0] < xr)
                    & (c_np == cl) & x2_central)
            ax = axes[cl, xb]
            if mask.sum() > 100:
                h_t, xe, ye = np.histogram2d(y_true_np[mask, 0], y_true_np[mask, 1],
                                              bins=40, range=list(range_y), density=True)
                h_t = gaussian_filter(h_t, sigma=1.5)
                if h_t.max() > 0:
                    lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 6)
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    yc = 0.5 * (ye[:-1] + ye[1:])
                    ax.contourf(xc, yc, h_t.T, levels=lvls,
                                cmap="Blues" if cl == 0 else "Oranges", alpha=0.6)
                    _contour2d(ax, y_model[mask], bins=40, range2d=range_y, smooth=1.5,
                               filled=False, colors=["k"], linewidths=1.2,
                               linestyles="--", levels=lvls)
            ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
            ax.set_xlabel("y₁", fontsize=8); ax.set_ylabel("y₂", fontsize=8)
            ax.set_title(f"{labels[cl]}  x₁∈[{xl:.1f},{xr:.1f}), |x₂|<0.5", fontsize=9)
            if cl == 0 and xb == 0:
                ax.legend(handles=[
                    Patch(facecolor=colors[cl], alpha=0.6, label="Truth"),
                    Line2D([0], [0], color="k", lw=1.2, ls="--", label="Model"),
                ], fontsize=8)
    plt.suptitle("Score model — 2D contours per x₁-bin (|x₂|<0.5)  p(y | x, c)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_score_2d_contours_x1binned")

    # --- 2b: 1D marginals per class ---
    bins_1d = {dim: np.linspace(range_y[dim][0], range_y[dim][1], 60) for dim in (0, 1)}
    fig, axes_m, axes_r = _ratio_grid(2, 2, figsize=(12, 10))
    for cl in [0, 1]:
        mask = c_np == cl
        for dim in [0, 1]:
            bins = bins_1d[dim]
            ax = axes_m[cl, dim]
            ax.hist(y_true_np[mask, dim], bins=bins, density=True,
                    histtype="stepfilled", alpha=0.35, color=colors[cl], label="Truth")
            ax.hist(y_model[mask, dim],  bins=bins, density=True,
                    histtype="step", lw=2, color=colors[cl], label="Model")
            ax.set_title(f"{labels[cl]} — dim {dim}")
            ax.legend(fontsize=9)
            ax.set_xlim(bins[0], bins[-1])
            dname = f"y{'₁' if dim == 0 else '₂'}"
            _draw_ratio(axes_r[cl, dim], bins,
                        y_model[mask, dim], y_true_np[mask, dim], color=colors[cl])
            axes_r[cl, dim].set_xlabel(dname)
    plt.suptitle("Score model — 1D marginals", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_score_1d_marginals")

    # --- 2c: X-binned 1D conditional slices (bin on x₁; collapse x₂) ---
    x_edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    n_xbins = len(x_edges) - 1
    bins_xbin = {dim: np.linspace(range_y[dim][0], range_y[dim][1], 31) for dim in (0, 1)}
    fig, axes_m, axes_r = _ratio_grid(4, n_xbins, figsize=(4 * n_xbins, 20))
    row_labels = ["Class A — y₁", "Class A — y₂", "Class B — y₁", "Class B — y₂"]

    for xb in range(n_xbins):
        xl, xr = x_edges[xb], x_edges[xb + 1]
        x_mask = (x_np[:, 0] >= xl) & (x_np[:, 0] < xr)
        for cl in [0, 1]:
            for dim in [0, 1]:
                row = cl * 2 + dim
                bins = bins_xbin[dim]
                ax = axes_m[row, xb]
                mask = x_mask & (c_np == cl)
                if mask.sum() > 50:
                    ax.hist(y_true_np[mask, dim], bins=bins, density=True,
                            histtype="stepfilled", alpha=0.35, color=colors[cl], label="Truth")
                    ax.hist(y_model[mask, dim],   bins=bins, density=True,
                            histtype="step", lw=1.5, color=colors[cl], label="Model")
                    ax.set_xlim(bins[0], bins[-1])
                    _draw_ratio(axes_r[row, xb], bins,
                                y_model[mask, dim], y_true_np[mask, dim], color=colors[cl])
                if xb == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8)
                ax.set_title(f"x₁∈[{xl:.1f},{xr:.1f})", fontsize=8)
                if row == 0 and xb == 0:
                    ax.legend(fontsize=7)
                dname = f"y{'₁' if dim == 0 else '₂'}"
                axes_r[row, xb].set_xlabel(dname, fontsize=7)

    plt.suptitle("Score model — conditional slices  p(y | x₁, c)  (marginalised over x₂)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_score_x1_binned")

    # --- 2d: log-prob diagnostics ---
    lp_true  = model(context).log_prob(y_true).cpu().numpy()
    lp_model = model(context).log_prob(
        torch.tensor(y_model, device=device, dtype=torch.float32)
    ).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lo, hi = np.percentile(lp_true, [1, 99])
    bins_lp = np.linspace(lo, hi, 50)
    for cl in [0, 1]:
        mask = c_np == cl
        axes[0].hist(lp_true[mask],  bins=bins_lp, density=True, histtype="stepfilled",
                     alpha=0.35, color=colors[cl], label=f"Truth {labels[cl]}")
        axes[0].hist(lp_model[mask], bins=bins_lp, density=True, histtype="step",
                     lw=2, color=colors[cl], label=f"Model {labels[cl]}")
    axes[0].set_xlabel("log p(y | x, c)")
    axes[0].set_title("Log-prob distribution")
    axes[0].legend(fontsize=8)

    for cl in [0, 1]:
        mask = c_np == cl
        lp = lp_true[mask]
        u  = (lp - lp.min()) / (lp.max() - lp.min() + 1e-9)
        axes[1].hist(u, bins=20, density=True, histtype="step",
                     lw=2, color=colors[cl], label=labels[cl])
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Normalised log-prob rank")
    axes[1].set_title("Uniformity check (PIT approx.)")
    axes[1].legend(fontsize=8)

    plt.suptitle("Score model — log-prob diagnostics", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_score_logprob")


# ---------------------------------------------------------------------------
# Section 3: Systematic residual models — multi-nuisance, 2D x
# ---------------------------------------------------------------------------

def _resolve_kin_systematics(sys_cfg):
    """Resolve the v12 `kin_systematics` list, or fall back to the v11
    `generator_kin_up`/`generator_kin_down` pattern.
    Returns a list of dicts: [{'name': ..., 'gen_cfg': ..., 'm': tensor[K]}, ...]"""
    out = []
    if "kin_systematics" in sys_cfg:
        for entry in sys_cfg["kin_systematics"]:
            name = entry["generator"]
            m    = list(entry["m"])
            gen_cfg = sys_cfg[name]
            out.append({"name": name, "gen_cfg": gen_cfg, "m": m})
        return out
    if "generator_kin_up" in sys_cfg:
        out.append({"name": "generator_kin_up",   "gen_cfg": sys_cfg["generator_kin_up"],   "m": [1.0]})
        out.append({"name": "generator_kin_down", "gen_cfg": sys_cfg["generator_kin_down"], "m": [-1.0]})
    return out


def _resolve_score_systematics(sys_cfg):
    """Resolve the v14 `score_systematics` list (generators carrying the p(y|x,ν)
    systematic, labelled by their m-vector). Returns [{'name','gen_cfg','m'}] (empty
    if absent — e.g. v12/v13 where the score residual is identity)."""
    out = []
    for entry in sys_cfg.get("score_systematics", []):
        name = entry["generator"]
        out.append({"name": name, "gen_cfg": sys_cfg[name], "m": list(entry["m"])})
    return out


@torch.no_grad()
def validate_systematics(base_cfg, sys_cfg, device, out_dir, n=80_000):
    print("Validating systematic residual models …")

    gen_nom = _make_gen(sys_cfg["generator"], device)
    c_nom, X_nom, _ = _gen_nominal(gen_nom, n, device)

    # --- Load base kin + residual kin model ---
    kin_cfg = base_cfg["kin_flow"]
    kin_base = zuko.flows.NSF(
        features=kin_cfg["features"], context=kin_cfg["context"],
        bins=kin_cfg["bins"], transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)
    kin_base.load_state_dict(torch.load(base_cfg["paths"]["kin_model"], map_location=device))

    res_k = sys_cfg["residual_kin_model"]
    if res_k["num_nuisances"] == 0:
        print("  residual_kin_model.num_nuisances == 0 → skipping kin residual validation.")
        return

    residual_kin = SystematicCorrectedModel(
        kin_base, features_dim=res_k["features_dim"], context_dim=res_k["context_dim"],
        num_nuisances=res_k["num_nuisances"], num_residual_layers=res_k["num_residual_layers"],
        hidden_features=res_k["hidden_features"], type=res_k["type"],
    ).to(device)
    residual_kin.load_state_dict(
        torch.load(sys_cfg["paths"]["kin_residual_model"], map_location=device), strict=False
    )
    residual_kin.eval()

    K = res_k["num_nuisances"]
    syst_entries = _resolve_kin_systematics(sys_cfg)
    if not syst_entries:
        print("  no systematic generators found in config → skipping kin residual validation.")
        return

    colors = {0: "dodgerblue", 1: "darkorange"}
    labels = {0: "Class A", 1: "Class B"}
    c_np = _class_idx(c_nom)

    def _kin_sample(model, c_arr, m_vec):
        """m_vec: list/tuple of length K → batch tensor [N, K]."""
        m = torch.tensor(m_vec, device=device, dtype=torch.float32)
        m = m.unsqueeze(0).expand(c_arr.shape[0], -1)
        return model.sample(c_arr.shape[0], c_arr, m).squeeze(0).cpu().numpy()

    # --- 3a: per-systematic 2D contours (Truth-up/down vs Model-m_up/m_down) ---
    # m = 0 baseline
    x_m0 = _kin_sample(residual_kin, c_nom, [0.0] * K)
    x_nom_np = X_nom.cpu().numpy()

    # Pre-generate the variation samples and the model samples at each m so we can
    # (a) compute a single percentile-zoom range that contains every contour, and
    # (b) avoid regenerating these batches in the later loops.
    variation_samples = []  # list of dicts: {ent, c_v_np, x_v_np, x_mv}
    for ent in syst_entries:
        gen_v = _make_gen(ent["gen_cfg"], device)
        c_v, X_v, _ = _gen_nominal(gen_v, n, device)
        variation_samples.append({
            "ent":    ent,
            "c_v_np": _class_idx(c_v),
            "x_v_np": X_v.cpu().numpy(),
            "x_mv":   _kin_sample(residual_kin, c_nom, ent["m"]),
        })

    range_2d = _auto_range_2d(
        x_nom_np, x_m0,
        *[s["x_v_np"] for s in variation_samples],
        *[s["x_mv"]   for s in variation_samples],
    )

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    for cl in [0, 1]:
        mask = c_np == cl
        ax = axes[cl]
        h_t, xe, ye = np.histogram2d(x_nom_np[mask, 0], x_nom_np[mask, 1],
                                      bins=60, range=list(range_2d), density=True)
        h_t = gaussian_filter(h_t, sigma=1.2)
        if h_t.max() > 0:
            lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 6)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            ax.contourf(xc, yc, h_t.T, levels=lvls,
                        cmap="Blues" if cl == 0 else "Oranges", alpha=0.6)
            _contour2d(ax, x_m0[mask], bins=60, range2d=range_2d, smooth=1.2,
                       filled=False, colors=["k"], linewidths=1.5,
                       linestyles="--", levels=lvls)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"{labels[cl]} — central (m=0)")
        ax.set_xlim(*range_2d[0]); ax.set_ylim(*range_2d[1]); ax.set_aspect("equal")
        ax.legend(handles=[
            Patch(facecolor=colors[cl], alpha=0.6, label="Gen nominal"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="Model m=0"),
        ], fontsize=9)
    plt.suptitle("Residual kin — central (m = 0)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_sys_kin_central")

    # one figure per systematic entry (reuse pre-computed samples)
    for s in variation_samples:
        ent    = s["ent"]
        c_v_np = s["c_v_np"]
        x_v_np = s["x_v_np"]
        x_mv   = s["x_mv"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for cl in [0, 1]:
            mask_n = c_np == cl
            mask_v = c_v_np == cl

            # Left column: truth (variation) vs nominal (Gen reference)
            ax = axes[cl, 0]
            h_t, xe, ye = np.histogram2d(x_v_np[mask_v, 0], x_v_np[mask_v, 1],
                                          bins=60, range=list(range_2d), density=True)
            h_t = gaussian_filter(h_t, sigma=1.2)
            if h_t.max() > 0:
                lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 6)
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])
                ax.contourf(xc, yc, h_t.T, levels=lvls,
                            cmap="Greens", alpha=0.6)
                _contour2d(ax, x_nom_np[mask_n], bins=60, range2d=range_2d, smooth=1.2,
                           filled=False, colors=[colors[cl]], linewidths=1.0,
                           linestyles=":", levels=lvls)
            ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
            ax.set_title(f"{labels[cl]} — Truth: {ent['name']}")
            ax.set_xlim(*range_2d[0]); ax.set_ylim(*range_2d[1]); ax.set_aspect("equal")
            ax.legend(handles=[
                Patch(facecolor="green", alpha=0.6, label=f"Gen {ent['name']}"),
                Line2D([0], [0], color=colors[cl], lw=1.0, ls=":", label="Gen nominal (ref)"),
            ], fontsize=8)

            # Right column: Model at m_vec vs truth (variation)
            ax = axes[cl, 1]
            if h_t.max() > 0:
                ax.contourf(xc, yc, h_t.T, levels=lvls,
                            cmap="Greens", alpha=0.6)
                _contour2d(ax, x_mv[mask_n], bins=60, range2d=range_2d, smooth=1.2,
                           filled=False, colors=["k"], linewidths=1.5,
                           linestyles="--", levels=lvls)
            ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
            ax.set_title(f"{labels[cl]} — Model m={ent['m']}")
            ax.set_xlim(*range_2d[0]); ax.set_ylim(*range_2d[1]); ax.set_aspect("equal")
            ax.legend(handles=[
                Patch(facecolor="green", alpha=0.6, label=f"Gen {ent['name']}"),
                Line2D([0], [0], color="k", lw=1.5, ls="--", label=f"Model m={ent['m']}"),
            ], fontsize=8)

        plt.suptitle(f"Residual kin — {ent['name']}   (m = {ent['m']})", fontsize=13)
        plt.tight_layout()
        _save(fig, out_dir, f"val_sys_kin_{ent['name']}")

    # --- 3b: 1D marginal ratios per systematic ---
    # Per-dim percentile range computed from nominal + every variation + every model
    # m-point, so the histograms zoom to the actual support.
    bins_dim = {
        dim: np.linspace(*_auto_range_1d(
            x_nom_np[:, dim],
            *[s["x_v_np"][:, dim] for s in variation_samples],
            *[s["x_mv"][:, dim]   for s in variation_samples],
        ), 60)
        for dim in (0, 1)
    }
    fig, axes_m, axes_r = _ratio_grid(2 * len(syst_entries), 2, figsize=(12, 5 * len(syst_entries)))
    for i, s in enumerate(variation_samples):
        ent    = s["ent"]
        c_v_np = s["c_v_np"]
        x_v_np = s["x_v_np"]
        x_mv   = s["x_mv"]
        for cl in [0, 1]:
            mask_n = c_np == cl
            mask_v = c_v_np == cl
            row = 2 * i + cl
            for dim in [0, 1]:
                bins = bins_dim[dim]
                ax = axes_m[row, dim]
                ax.hist(x_v_np[mask_v, dim], bins=bins, density=True,
                        histtype="stepfilled", alpha=0.35, color="green",
                        label=f"Gen {ent['name']}")
                ax.hist(x_mv[mask_n, dim],   bins=bins, density=True,
                        histtype="step", lw=2, color="k",
                        label=f"Model m={ent['m']}")
                ax.hist(x_nom_np[mask_n, dim], bins=bins, density=True,
                        histtype="step", lw=1, ls="--", color=colors[cl],
                        label="Gen nominal (ref)")
                ax.set_title(f"{labels[cl]} — x{'₁' if dim == 0 else '₂'} — {ent['name']}", fontsize=9)
                ax.legend(fontsize=7)
                ax.set_xlim(bins[0], bins[-1])
                _draw_ratio(axes_r[row, dim], bins,
                            x_mv[mask_n, dim], x_v_np[mask_v, dim], color="k")
                axes_r[row, dim].set_xlabel(f"x{'₁' if dim == 0 else '₂'}", fontsize=8)
    plt.suptitle("Residual kin — 1D marginal closure tests", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_sys_kin_1d_marginals")

    # --- 3c: scan ⟨x_dim⟩ and σ(x_dim) vs each m_k axis ---
    # Mean tracks shift-type nuisances; std tracks second-moment / squeeze-type
    # nuisances (centroid-preserving, so the mean panel is flat there).
    m_vals = np.linspace(-1.5, 1.5, 13)
    fig_m, axes_m = plt.subplots(K, 2, figsize=(12, 4 * K), squeeze=False)
    fig_s, axes_s = plt.subplots(K, 2, figsize=(12, 4 * K), squeeze=False)
    for k in range(K):
        mean_x = {0: {0: [], 1: []}, 1: {0: [], 1: []}}  # [class][dim] = list of means
        std_x  = {0: {0: [], 1: []}, 1: {0: [], 1: []}}  # [class][dim] = list of stds
        for mv in m_vals:
            mvec = [0.0] * K
            mvec[k] = float(mv)
            xs = _kin_sample(residual_kin, c_nom, mvec)
            for cl in [0, 1]:
                m_cls = c_np == cl
                for dim in [0, 1]:
                    mean_x[cl][dim].append(xs[m_cls, dim].mean())
                    std_x[cl][dim].append(xs[m_cls, dim].std())
        for dim in [0, 1]:
            ax = axes_m[k, dim]
            for cl in [0, 1]:
                ax.plot(m_vals, mean_x[cl][dim], marker="o",
                        color=colors[cl], label=labels[cl])
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(f"m[{k}]")
            ax.set_ylabel(f"⟨x{'₁' if dim == 0 else '₂'}⟩")
            ax.set_title(f"Residual kin — ⟨x{'₁' if dim == 0 else '₂'}⟩ vs m[{k}]")
            ax.legend(fontsize=9)

            ax = axes_s[k, dim]
            for cl in [0, 1]:
                ax.plot(m_vals, std_x[cl][dim], marker="o",
                        color=colors[cl], label=labels[cl])
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(f"m[{k}]")
            ax.set_ylabel(f"σ(x{'₁' if dim == 0 else '₂'})")
            ax.set_title(f"Residual kin — σ(x{'₁' if dim == 0 else '₂'}) vs m[{k}]")
            ax.legend(fontsize=9)
    fig_m.tight_layout()
    _save(fig_m, out_dir, "val_sys_mean_x_vs_m")
    fig_s.tight_layout()
    _save(fig_s, out_dir, "val_sys_std_x_vs_m")


# ---------------------------------------------------------------------------
# Section 4: Systematic residual SCORE model  p(y | x, c, ν)   — v14
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_score_systematics(base_cfg, sys_cfg, device, out_dir, n=80_000):
    """Validate the residual SCORE model p(y | x, c, ν) against generator truth.

    For each `score_systematics` entry the residual score model is sampled at the
    matching m-vector, conditioned on the variation generator's (c, x) — i.e. the
    SAME x-domain the residual was trained on (x is sampled with the systematic) — and
    compared to that generator's nominal-conditional truth y (distorsion=False).
    Skips cleanly for v12/v13 (no score residual)."""
    print("Validating systematic residual SCORE model …")
    res_s = sys_cfg.get("residual_score_model")
    if res_s is None or int(res_s.get("num_nuisances", 0)) == 0:
        print("  residual_score_model absent or num_nuisances == 0 → skipping score residual validation.")
        return
    syst_entries = _resolve_score_systematics(sys_cfg)
    if not syst_entries:
        print("  no `score_systematics` in config → skipping score residual validation.")
        return
    score_residual_path = sys_cfg["paths"].get("score_residual_model")
    if not score_residual_path or not os.path.isfile(score_residual_path):
        print(f"  score residual weights not found ({score_residual_path}) → skipping.")
        return

    # --- base score flow + residual ---
    score_cfg = base_cfg["score_flow"]
    score_base = zuko.flows.NSF(
        features=score_cfg["features"], context=score_cfg["context"],
        bins=score_cfg["bins"], transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)
    score_base.load_state_dict(torch.load(base_cfg["paths"]["score_model"], map_location=device))
    residual_score = SystematicCorrectedModel(
        score_base, features_dim=res_s["features_dim"], context_dim=res_s["context_dim"],
        num_nuisances=res_s["num_nuisances"], num_residual_layers=res_s["num_residual_layers"],
        hidden_features=res_s["hidden_features"], type=res_s["type"],
    ).to(device)
    residual_score.load_state_dict(
        torch.load(score_residual_path, map_location=device), strict=False
    )
    residual_score.eval()
    K = int(res_s["num_nuisances"])

    colors = {0: "dodgerblue", 1: "darkorange"}
    labels = {0: "Class A", 1: "Class B"}

    def _score_sample(context, m_vec):
        """Sample the residual score model at m_vec, conditioned on `context`=[c, x]."""
        m = torch.tensor(m_vec, device=device, dtype=torch.float32)
        m = m.unsqueeze(0).expand(context.shape[0], -1)
        return residual_score.sample(context.shape[0], context, m).squeeze(0).cpu().numpy()

    # central (m = 0) closure: gen nominal score vs model at m = 0
    gen_nom = _make_gen(sys_cfg["generator"], device)
    c_nom, X_nom, y_nom = _gen_nominal(gen_nom, n, device)
    ctx_nom = torch.cat([c_nom, X_nom], dim=-1)
    c_np = _class_idx(c_nom)
    y_nom_np = y_nom.cpu().numpy()
    y_m0 = _score_sample(ctx_nom, [0.0] * K)

    # Truth + model at the SAME systematically-varied x — the domain that actually
    # occurs at ν and on which the residual was TRAINED (x is sampled with the
    # systematic in _train_score). Validating here (rather than at nominal x) keeps the
    # train/eval x-domain consistent and avoids testing the base flow out of
    # distribution. Truth (distorsion=False) is the nominal-conditional p(y|x,ν) the
    # score residual must reproduce; model is sampled at the same context.
    variation_samples = []
    for ent in syst_entries:
        gen_v = _make_gen(ent["gen_cfg"], device)
        c_v, X_v, y_v = _gen_nominal(gen_v, n, device)         # x varied by the systematic
        ctx_v = torch.cat([c_v, X_v], dim=-1)
        variation_samples.append({
            "ent": ent, "c_v_np": _class_idx(c_v),
            "y_v_np": y_v.cpu().numpy(),
            "y_mv": _score_sample(ctx_v, ent["m"]),
        })

    range_y = _auto_range_2d(
        y_nom_np, y_m0,
        *[s["y_v_np"] for s in variation_samples],
        *[s["y_mv"] for s in variation_samples],
    )

    # --- 4a: central (m = 0) closure ---
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    for cl in [0, 1]:
        mask = c_np == cl
        ax = axes[cl]
        h_t, xe, ye = np.histogram2d(y_nom_np[mask, 0], y_nom_np[mask, 1],
                                      bins=60, range=list(range_y), density=True)
        h_t = gaussian_filter(h_t, sigma=1.5)
        if h_t.max() > 0:
            lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 6)
            xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
            ax.contourf(xc, yc, h_t.T, levels=lvls,
                        cmap="Blues" if cl == 0 else "Oranges", alpha=0.6)
            _contour2d(ax, y_m0[mask], bins=60, range2d=range_y, smooth=1.5,
                       filled=False, colors=["k"], linewidths=1.5, linestyles="--", own_levels=True)
        ax.set_xlabel("y₁"); ax.set_ylabel("y₂")
        ax.set_title(f"{labels[cl]} — central (m = 0)")
        ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
        ax.legend(handles=[
            Patch(facecolor=colors[cl], alpha=0.6, label="Gen nominal"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="Model m=0"),
        ], fontsize=9)
    plt.suptitle("Residual score — central (m = 0)  p(y | x, c)", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_sys_score_central")

    # --- 4b: per-systematic 2D contours (truth variation vs model at m) ---
    for s in variation_samples:
        ent = s["ent"]; c_v_np = s["c_v_np"]; y_v_np = s["y_v_np"]; y_mv = s["y_mv"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for cl in [0, 1]:
            mask_n = c_np == cl; mask_v = c_v_np == cl
            ax = axes[cl]
            h_t, xe, ye = np.histogram2d(y_v_np[mask_v, 0], y_v_np[mask_v, 1],
                                          bins=60, range=list(range_y), density=True)
            h_t = gaussian_filter(h_t, sigma=1.5)
            if h_t.max() > 0:
                lvls = np.linspace(h_t.max() * 0.04, h_t.max() * 0.95, 6)
                xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
                ax.contourf(xc, yc, h_t.T, levels=lvls, cmap="Greens", alpha=0.6)
                # y_mv was sampled at ctx_v=(c_v, X_v) -> mask by the SAME classes (mask_v),
                # NOT the independent nominal draw (mask_n), or the classes get scrambled.
                _contour2d(ax, y_mv[mask_v], bins=60, range2d=range_y, smooth=1.5,
                           filled=False, colors=["k"], linewidths=1.5, linestyles="--", own_levels=True)
                _contour2d(ax, y_nom_np[mask_n], bins=60, range2d=range_y, smooth=1.5,
                           filled=False, colors=[colors[cl]], linewidths=1.0, linestyles=":", own_levels=True)
            ax.set_xlabel("y₁"); ax.set_ylabel("y₂")
            ax.set_title(f"{labels[cl]} — {ent['name']}  m={ent['m']}")
            ax.set_xlim(*range_y[0]); ax.set_ylim(*range_y[1]); ax.set_aspect("equal")
            ax.legend(handles=[
                Patch(facecolor="green", alpha=0.6, label=f"Gen {ent['name']}"),
                Line2D([0], [0], color="k", lw=1.5, ls="--", label=f"Model m={ent['m']}"),
                Line2D([0], [0], color=colors[cl], lw=1.0, ls=":", label="Gen nominal (ref)"),
            ], fontsize=8)
        plt.suptitle(f"Residual score — {ent['name']}   (m = {ent['m']})", fontsize=13)
        plt.tight_layout()
        _save(fig, out_dir, f"val_sys_score_{ent['name']}")

    # --- 4c: 1D marginal closure per systematic ---
    bins_dim = {dim: np.linspace(range_y[dim][0], range_y[dim][1], 60) for dim in (0, 1)}
    fig, axes_m, axes_r = _ratio_grid(2 * len(syst_entries), 2, figsize=(12, 5 * len(syst_entries)))
    for i, s in enumerate(variation_samples):
        ent = s["ent"]; c_v_np = s["c_v_np"]; y_v_np = s["y_v_np"]; y_mv = s["y_mv"]
        for cl in [0, 1]:
            mask_n = c_np == cl; mask_v = c_v_np == cl
            row = 2 * i + cl
            for dim in [0, 1]:
                bins = bins_dim[dim]; ax = axes_m[row, dim]
                ax.hist(y_v_np[mask_v, dim], bins=bins, density=True, histtype="stepfilled",
                        alpha=0.35, color="green", label=f"Gen {ent['name']}")
                ax.hist(y_mv[mask_v, dim], bins=bins, density=True, histtype="step", lw=2,
                        color="k", label=f"Model m={ent['m']}")
                ax.hist(y_nom_np[mask_n, dim], bins=bins, density=True, histtype="step", lw=1,
                        ls="--", color=colors[cl], label="Gen nominal (ref)")
                ax.set_title(f"{labels[cl]} — y{'₁' if dim == 0 else '₂'} — {ent['name']}", fontsize=9)
                ax.legend(fontsize=7); ax.set_xlim(bins[0], bins[-1])
                _draw_ratio(axes_r[row, dim], bins, y_mv[mask_v, dim], y_v_np[mask_v, dim], color="k")
                axes_r[row, dim].set_xlabel(f"y{'₁' if dim == 0 else '₂'}", fontsize=8)
    plt.suptitle("Residual score — 1D marginal closure tests", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir, "val_sys_score_1d_marginals")

    # --- 4d: scan ⟨y⟩ and σ(y) vs each m_k (at fixed nominal context) ---
    # Holding x fixed isolates the explicit p(y|x,ν) response: mean tracks the
    # shift-type score systematic, std tracks the squeeze-type one.
    m_vals = np.linspace(-1.5, 1.5, 13)
    fig_m, axes_mu = plt.subplots(K, 2, figsize=(12, 4 * K), squeeze=False)
    fig_s, axes_sd = plt.subplots(K, 2, figsize=(12, 4 * K), squeeze=False)
    for k in range(K):
        mean_y = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
        std_y  = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
        for mv in m_vals:
            mvec = [0.0] * K; mvec[k] = float(mv)
            ys = _score_sample(ctx_nom, mvec)
            for cl in [0, 1]:
                m_cls = c_np == cl
                for dim in [0, 1]:
                    mean_y[cl][dim].append(ys[m_cls, dim].mean())
                    std_y[cl][dim].append(ys[m_cls, dim].std())
        for dim in [0, 1]:
            ax = axes_mu[k, dim]
            for cl in [0, 1]:
                ax.plot(m_vals, mean_y[cl][dim], marker="o", color=colors[cl], label=labels[cl])
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(f"m[{k}]"); ax.set_ylabel(f"⟨y{'₁' if dim == 0 else '₂'}⟩")
            ax.set_title(f"Residual score — ⟨y{'₁' if dim == 0 else '₂'}⟩ vs m[{k}]")
            ax.legend(fontsize=9)
            ax = axes_sd[k, dim]
            for cl in [0, 1]:
                ax.plot(m_vals, std_y[cl][dim], marker="o", color=colors[cl], label=labels[cl])
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel(f"m[{k}]"); ax.set_ylabel(f"σ(y{'₁' if dim == 0 else '₂'})")
            ax.set_title(f"Residual score — σ(y{'₁' if dim == 0 else '₂'}) vs m[{k}]")
            ax.legend(fontsize=9)
    fig_m.tight_layout(); _save(fig_m, out_dir, "val_sys_score_mean_y_vs_m")
    fig_s.tight_layout(); _save(fig_s, out_dir, "val_sys_score_std_y_vs_m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate base and systematic flow models.")
    parser.add_argument("-c", "--cfg", required=True, help="Base flows YAML config")
    parser.add_argument("--sys-cfg", default=None, help="Systematics YAML config (optional)")
    parser.add_argument("--out-dir", default="validation", help="Output directory")
    parser.add_argument("--n", type=int, default=100_000, help="Samples for comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for k, v in list(base_cfg["paths"].items()):
        if isinstance(v, str):
            base_cfg["paths"][k] = _resolve(v, cfg_dir)

    device = base_cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}  |  Output: {args.out_dir}/")

    validate_kin(base_cfg, device, args.out_dir, n=args.n)
    validate_score(base_cfg, device, args.out_dir, n=args.n)

    if args.sys_cfg:
        sys_cfg_path = os.path.abspath(args.sys_cfg)
        with open(sys_cfg_path) as f:
            sys_cfg = yaml.safe_load(f)
        sys_dir = os.path.dirname(sys_cfg_path)
        sys_cfg["_cfg_dir"] = sys_dir
        for k, v in list(sys_cfg["paths"].items()):
            if isinstance(v, str):
                sys_cfg["paths"][k] = _resolve(v, sys_dir)
        validate_systematics(base_cfg, sys_cfg, device, args.out_dir, n=args.n)
        validate_score_systematics(base_cfg, sys_cfg, device, args.out_dir, n=args.n)

    print(f"\nDone. All plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
