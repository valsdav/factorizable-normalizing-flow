"""
Showcase the transformation learnt by the residual kin / score models — paper figures.

The residual is a *factorizable* affine correction applied per feature dimension:

        x_obs  --[residual stack]-->  x_nom        (forward)
        y = x * exp(s(x,c,ν)) + t(x,c,ν)

where the per-dimension log-scale ``s`` and shift ``t`` are polynomials in the
nuisance vector ``ν`` (= ``m``).  This script visualises that map two ways:

  1. RESPONSE CURVES  (``*_response.pdf``)
     At a handful of representative phase-space points, the locally-felt
     *scale* (diagonal Jacobian ∂T_d/∂x_d ≡ exp s_d) and *shift* (net
     displacement Δ_d = T(x)_d − x_d) are scanned as each nuisance ν_k is swept.
     These are the two ingredients of the affine deformation, shown vs ν.

  2. DISPLACEMENT VECTOR FIELD  (``*_field_*.pdf``)
     Over a 2D grid spanning the phase space, arrows show the displacement
     Δx = T(x, ν) − x at a fixed ν, overlaid on the nominal density.  This
     exposes the spatial structure of the learnt deformation.

Both the kin residual  p(x|c,ν)  and the score residual  p(y|x,c,ν)  are handled.
The "scale" and "shift" are computed from the FULL residual stack (all
permutation + polynomial layers composed), so they are exact for any number of
residual layers.

Usage:
    python visualize_residual_transform.py -c configs/base_flows.yaml \
        --sys-cfg configs/systematics.yaml --out-dir validation_transform/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml
import zuko

from residual_flow import SystematicCorrectedModel
from validate_flows import (_resolve, _save, _make_gen, _gen_nominal, _class_idx,
                            _resolve_kin_systematics, _resolve_score_systematics)


# ---------------------------------------------------------------------------
# Model construction (mirrors validate_flows.py)
# ---------------------------------------------------------------------------

def _build_base(flow_cfg, weights_path, device):
    model = zuko.flows.NSF(
        features=flow_cfg["features"], context=flow_cfg["context"],
        bins=flow_cfg["bins"], transforms=flow_cfg["transforms"],
        hidden_features=tuple(flow_cfg["hidden_features"]),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def _build_residual(base, res_cfg, weights_path, device):
    # Honour cross-term / damping config so the architecture AND the s,t response match
    # the trained checkpoint (cross_term_pairs sizes a buffer that must match; the
    # dampings are not in the state_dict but scale the response we plot).
    extra = {}
    pairs = res_cfg.get("cross_term_pairs")
    if pairs:
        extra["cross_term_pairs"] = [tuple(int(i) for i in pr) for pr in pairs]
    if "cross_term_damping" in res_cfg:
        extra["cross_term_damping"] = float(res_cfg["cross_term_damping"])
    if "quadratic_damping" in res_cfg:
        extra["quadratic_damping"] = float(res_cfg["quadratic_damping"])
    residual = SystematicCorrectedModel(
        base, features_dim=res_cfg["features_dim"], context_dim=res_cfg["context_dim"],
        num_nuisances=res_cfg["num_nuisances"], num_residual_layers=res_cfg["num_residual_layers"],
        hidden_features=res_cfg["hidden_features"], type=res_cfg["type"], **extra,
    ).to(device)
    residual.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    residual.eval()
    return residual


# ---------------------------------------------------------------------------
# Core: apply ONLY the residual stack (no base flow) and read off scale + shift
# ---------------------------------------------------------------------------

@torch.no_grad()
def residual_map(residual, x, context, m, direction="forward"):
    """Apply only the residual transform stack (not the frozen base flow).

    forward : x_obs -> x_nom   (the literal  x*exp(s)+t  parameterisation)
    inverse : x_nom -> x_obs   (the systematic deformation of the nominal density)
    """
    xc = x
    if direction == "forward":
        for tr in residual.transforms:
            xc, _ = tr(xc, context=context, m=m)
    else:
        for tr in reversed(residual.transforms):
            xc, _ = tr.inverse(xc, context=context, m=m)
    return xc


@torch.no_grad()
def scale_and_shift(residual, x, context, m, direction="forward", eps=1e-3):
    """Local scale (diagonal Jacobian) and net shift of the composed residual map.

    Returns (scale [B, D], shift [B, D], y [B, D]) where
        y      = T(x)
        shift  = y - x                         (additive displacement)
        scale  = ∂T_d/∂x_d  via finite diff    (multiplicative factor ≡ exp s_d)
    Both are exact at m=0 (scale→1, shift→0) by the identity initialisation.
    """
    y = residual_map(residual, x, context, m, direction)
    shift = y - x
    D = x.shape[1]
    scale = torch.empty_like(x)
    for d in range(D):
        xp = x.clone()
        xp[:, d] += eps
        yp = residual_map(residual, xp, context, m, direction)
        scale[:, d] = (yp[:, d] - y[:, d]) / eps
    return scale, shift, y


def _m_batch(m_list, B, device):
    m = torch.tensor(m_list, dtype=torch.float32, device=device)
    return m.unsqueeze(0).expand(B, -1).contiguous()


def _onehot(cl, B, device):
    """Class one-hot [B, 2];  cl=0 -> [1,0],  cl=1 -> [0,1]."""
    oh = torch.zeros(B, 2, device=device)
    oh[:, cl] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Figure 1: scale & shift response curves vs ν at representative points
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_response(residual, *, model_tag, feat_syms, var_sym, points, ctx_builder, K,
                  out_dir, device, direction="forward", m_range=(-2.0, 2.0), n_m=41,
                  title_extra="", display_tag=None, dims=None):
    """Scan local scale and net shift vs each nuisance, per representative point.

    points       : list of (label, feature_point[D])  — where in phase space to probe.
    ctx_builder  : fn(cl, B) -> context tensor [B, ctx_dim]  for class cl.
    feat_syms    : per-dim axis symbols, e.g. ('x₁','x₂') or ('y₁','y₂').
    var_sym      : base observable symbol ('x' or 'y') used in legend coordinates.
    dims         : feature dims to show as rows (default all); e.g. [0] shows only the
                   first feature (y₁ / x₁) → a single row of plots.
    """
    D = len(feat_syms)
    show_dims = list(range(D)) if dims is None else [d for d in dims if 0 <= d < D]
    nrow = len(show_dims)
    m_vals = np.linspace(*m_range, n_m)
    m_t = torch.tensor(m_vals, dtype=torch.float32, device=device)

    cls_styles = {0: ("Class A", "-"), 1: ("Class B", "--")}
    pt_colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(points)))

    # rows = selected feature dims, columns = [scale, shift] per nuisance
    fig, axes = plt.subplots(nrow, 2 * K, figsize=(5 * K, 3.4 * nrow), squeeze=False)

    for k in range(K):                                   # nuisance index
        for pidx, (plabel, pvec) in enumerate(points):
            for cl, (clname, ls) in cls_styles.items():
                B = n_m
                x = torch.tensor(pvec, dtype=torch.float32, device=device)
                x = x.unsqueeze(0).expand(B, -1).contiguous()
                ctx = ctx_builder(cl, B)
                m = torch.zeros(B, K, device=device)
                m[:, k] = m_t
                scale, shift, _ = scale_and_shift(residual, x, ctx, m, direction)
                scale = scale.cpu().numpy()
                shift = shift.cpu().numpy()
                for ridx, d in enumerate(show_dims):
                    axes[ridx, 2 * k].plot(
                        m_vals, scale[:, d], ls=ls, color=pt_colors[pidx], lw=1.8)
                    axes[ridx, 2 * k + 1].plot(
                        m_vals, shift[:, d], ls=ls, color=pt_colors[pidx], lw=1.8)

    for ridx, d in enumerate(show_dims):
        for k in range(K):
            a_s = axes[ridx, 2 * k]
            a_t = axes[ridx, 2 * k + 1]
            for a in (a_s, a_t):
                a.axvline(0, color="k", lw=0.6, ls=":")
                a.set_xlabel(f"ν[{k}]")
                a.grid(alpha=0.2)
            a_s.axhline(1.0, color="gray", lw=0.6, ls="--")
            a_t.axhline(0.0, color="gray", lw=0.6, ls="--")
            a_s.set_ylabel(f"scale  exp(s) on {feat_syms[d]}")
            a_t.set_ylabel(f"shift  Δ{feat_syms[d]}")
            a_s.set_title(f"{feat_syms[d]} scale — ν[{k}]", fontsize=9)
            a_t.set_title(f"{feat_syms[d]} shift — ν[{k}]", fontsize=9)

    # legend: colour = phase-space point (coordinates), linestyle = class
    handles = [Line2D([0], [0], color=pt_colors[i], lw=2,
                      label=f"{var_sym}=({v[0]:g}, {v[1]:g})")
               for i, (_, v) in enumerate(points)]
    handles += [Line2D([0], [0], color="k", lw=2, ls="-", label="Class A"),
                Line2D([0], [0], color="k", lw=2, ls="--", label="Class B")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Residual {display_tag or model_tag} — scale & shift response vs ν "
                 f"({direction} map){title_extra}", fontsize=13, y=1.06)
    fig.tight_layout()
    _save(fig, out_dir, f"residual_{model_tag}_response")


def _density_contour(ax, samp, range2d, *, color, bins=60, smooth=1.5, n_levels=6,
                     alpha=1.0, lw=0.8, linestyles="solid", zorder=1):
    """Smoothed density of samples [N,2] as (unfilled) contour LINES in `color`."""
    (gx_lo, gx_hi), (gy_lo, gy_hi) = range2d
    h, xe, ye = np.histogram2d(samp[:, 0], samp[:, 1], bins=bins,
                               range=[[gx_lo, gx_hi], [gy_lo, gy_hi]], density=True)
    h = gaussian_filter(h, sigma=smooth)
    if h.max() <= 0:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = np.linspace(h.max() * 0.05, h.max() * 0.95, n_levels)
    ax.contour(xc, yc, h.T, levels=levels, colors=color, linewidths=lw,
               alpha=alpha, linestyles=linestyles, zorder=zorder)


@torch.no_grad()
def _model_densities(residual, ctx, m_vec, device, n_samp):
    """Model nominal (m=0) and distorted (m=ν) obs-space samples at fixed context.

    nominal  = base-flow samples (residual is identity at m=0).
    distorted = those nominal points pushed through the residual INVERSE (nom→obs),
                i.e. the density the systematic ν produces — exactly the inverse of
                the displacement field drawn as arrows.
    """
    ctx_n = ctx if ctx.shape[0] == n_samp else ctx[:1].expand(n_samp, -1).contiguous()
    nom = residual.base_model(ctx_n).rsample()                   # [n_samp, D]
    m = _m_batch(list(m_vec), n_samp, device)
    dist = residual_map(residual, nom, ctx_n, m, direction="inverse")
    return nom.cpu().numpy(), dist.cpu().numpy()


# --- generator-truth distorted density at a nuisance point -------------------

def _match_template(entries, mvec):
    """Find the systematics-list entry whose m is the unit template along mvec.

    mvec has one non-zero component (the panel's single active nuisance). Returns
    (entry, k) for the matching ±unit template, scaled later by |mvec[k]|, or None.
    """
    mvec = np.asarray(mvec, dtype=float)
    nz = np.nonzero(mvec)[0]
    if len(nz) != 1:
        return None
    k = int(nz[0])
    for e in entries:
        em = np.asarray(e["m"], dtype=float)
        enz = np.nonzero(em)[0]
        if len(enz) == 1 and enz[0] == k and em[k] * mvec[k] > 0:
            return e, k
    return None


def _scale_gen_cfg(gen_cfg, factor):
    """Copy a generator config with its variation_* magnitudes scaled by `factor`."""
    cfg = dict(gen_cfg)
    for key in ("variation_shift", "variation_squeeze", "variation_rot"):
        if key in cfg:
            cfg[key] = float(cfg[key]) * factor
    return cfg


def _make_kin_truth_fn(sys_cfg, device, n_truth):
    """truth(mvec) -> {cl: x-samples} from the matching generator template, or None."""
    entries = _resolve_kin_systematics(sys_cfg)

    @torch.no_grad()
    def truth(mvec):
        match = _match_template(entries, mvec)
        if match is None:
            return None
        e, k = match
        cfg = _scale_gen_cfg(e["gen_cfg"], abs(mvec[k]) / abs(e["m"][k]))
        gen = _make_gen(cfg, device)
        c, X, _ = _gen_nominal(gen, n_truth, device)
        ci = _class_idx(c)
        Xn = X.cpu().numpy()
        return {0: Xn[ci == 0], 1: Xn[ci == 1]}

    return truth


def _make_score_truth_fn(sys_cfg, device, n_truth, x_ctx):
    """truth(mvec) -> {cl: y-samples} of the generator's p(y | x_ctx, ν), or None.

    Samples the pre-distortion bivariate Gaussian (get_base_dist_params) directly at
    the fixed x_ctx — the conditional density the score residual must reproduce."""
    entries = _resolve_score_systematics(sys_cfg)

    @torch.no_grad()
    def truth(mvec):
        match = _match_template(entries, mvec)
        if match is None:
            return None
        e, k = match
        cfg = _scale_gen_cfg(e["gen_cfg"], abs(mvec[k]) / abs(e["m"][k]))
        gen = _make_gen(cfg, device)
        x_rep = torch.tensor(x_ctx, dtype=torch.float32, device=device)
        x_rep = x_rep.unsqueeze(0).expand(n_truth, -1)
        out = {}
        for cl in (0, 1):
            c_rep = torch.full((n_truth, 1), cl, dtype=torch.long, device=device)
            mu, L = gen.get_base_dist_params(x_rep, c_rep)
            eps = torch.randn(n_truth, 2, device=device)
            y = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
            if getattr(gen, "sigmoid_y", False):
                y = torch.sigmoid(y)
            out[cl] = y.cpu().numpy()
        return out

    return truth


# ---------------------------------------------------------------------------
# Figure 2: displacement vector field over phase space, at fixed ν
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_vector_field(residual, *, model_tag, axis_syms, var_sym, grid_range, ctx_builder, K,
                      out_dir, device, direction="forward", truth_fn=None,
                      m_value=1.0, ngrid=20, n_density=30_000, title_extra=""):
    """Quiver of Δ = T(x,ν) − x over a 2D grid, one panel per (nuisance, class).

    Each panel overlays three densities (contour lines) at that conditioning:
      - model distorted ρ(·|ν)        → class colour (dodgerblue / darkorange), solid
      - generator-truth ρ_true(·|ν)   → black solid   (if truth_fn supplies it)
      - nominal ρ(·|0)                → grey dashed
    Produces one figure for ν=+m_value and one for ν=−m_value.
    """
    (gx_lo, gx_hi), (gy_lo, gy_hi) = grid_range
    gx = np.linspace(gx_lo, gx_hi, ngrid)
    gy = np.linspace(gy_lo, gy_hi, ngrid)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)        # [G, 2]
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    G = grid_t.shape[0]
    rng2d = ((gx_lo, gx_hi), (gy_lo, gy_hi))

    for sign in (+1.0, -1.0):
        mv = sign * m_value
        fig, axes = plt.subplots(K, 2, figsize=(11, 5 * K), squeeze=False)
        for k in range(K):
            mvec = [0.0] * K
            mvec[k] = mv
            truth_dict = truth_fn(mvec) if truth_fn is not None else None
            for cl in (0, 1):
                ax = axes[k, cl]
                # density overlays (contour lines): model distorted = class colour,
                # truth distorted = black, nominal = grey dashed
                ctx_d = ctx_builder(cl, n_density)
                nom_np, dist_np = _model_densities(residual, ctx_d, mvec, device, n_density)
                dist_color = "dodgerblue" if cl == 0 else "darkorange"
                _density_contour(ax, dist_np, rng2d, color=dist_color,
                                 lw=1.3, alpha=0.9, zorder=1)
                if truth_dict is not None and len(truth_dict[cl]) > 100:
                    _density_contour(ax, truth_dict[cl], rng2d, color="black",
                                     lw=1.0, alpha=0.7, zorder=1)
                _density_contour(ax, nom_np, rng2d, color="0.5", lw=0.9,
                                 alpha=0.85, linestyles="dashed", zorder=1)
                # displacement field
                ctx = ctx_builder(cl, G)
                m = torch.zeros(G, K, device=device)
                m[:, k] = mv
                _, shift, _ = scale_and_shift(residual, grid_t, ctx, m, direction)
                dxy = shift.cpu().numpy()
                mag = np.hypot(dxy[:, 0], dxy[:, 1])
                q = ax.quiver(grid[:, 0], grid[:, 1], dxy[:, 0], dxy[:, 1], mag,
                              angles="xy", scale_units="xy", scale=1.0,
                              cmap="viridis", width=0.004, zorder=2)
                fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label=f"|Δ{var_sym}|")
                ax.set_xlim(gx_lo, gx_hi)
                ax.set_ylim(gy_lo, gy_hi)
                ax.set_xlabel(axis_syms[0])
                ax.set_ylabel(axis_syms[1])
                ax.set_title(f"Class {'A' if cl == 0 else 'B'} — ν[{k}]={mv:+.1f}",
                             fontsize=10)
        handles = [Line2D([0], [0], color="dodgerblue", lw=1.6,
                          label=f"Model ρ({var_sym}|ν)  [A blue / B orange]"),
                   Line2D([0], [0], color="black", lw=1.4,
                          label=f"Truth ρ({var_sym}|ν)  (generator)"),
                   Line2D([0], [0], color="0.5", lw=1.2, ls="--",
                          label=f"Nominal ρ({var_sym}|0)")]
        fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(f"Displacement field   Δ{var_sym} = T({var_sym}, ν) − {var_sym}"
                     f"    ({direction}){title_extra}", fontsize=13, y=1.02)
        fig.tight_layout()
        tag = "p" if sign > 0 else "m"
        _save(fig, out_dir, f"residual_{model_tag}_field_nu{tag}{m_value:g}")


# ---------------------------------------------------------------------------
# Figure 3: nuisance cross-term — response over the (ν_i, ν_j) plane
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_cross_term(residual, *, model_tag, feat_syms, var_sym, probe, ctx_builder, K,
                    out_dir, device, direction="forward", nu_pair=(0, 1),
                    nu_syms=(r"\nu_\mathrm{shift}", r"\nu_\mathrm{squeeze}"),
                    nu_range=(-2.0, 2.0), ngrid=41, dims=None, classes=(0, 1),
                    title_extra=""):
    """Expose the nuisance CROSS-TERM by mapping the response over the (ν_i, ν_j) plane.

    At a fixed probe point the affine parameters log-scale ``s`` and shift ``t`` (both
    additive polynomials in ν, plus the cross net) are evaluated on a 2D ν grid and
    decomposed as

        full(ν_i, ν_j)  =  additive[ single_i + single_j ]  +  cross

    where  cross = full − additive  isolates the bilinear  C·ν_i·ν_j  term — ≈0 for an
    additive residual, a sign-flipping hyperbolic pattern when the cross net is active.
    One figure per class; needs K≥2 (a nuisance pair to cross)."""
    if K < 2:
        print("  cross-term plot needs ≥2 nuisances → skipping.")
        return
    D = len(feat_syms)
    show_dims = list(range(D)) if dims is None else [d for d in dims if 0 <= d < D]
    i, j = nu_pair
    gv = np.linspace(*nu_range, ngrid)
    GI, GJ = np.meshgrid(gv, gv)                    # GI=ν_i (cols/x), GJ=ν_j (rows/y)
    G = GI.size
    ic = int(np.argmin(np.abs(gv)))                # index of ν≈0
    ext = [gv[0], gv[-1], gv[0], gv[-1]]
    x = torch.tensor(probe, dtype=torch.float32, device=device).unsqueeze(0)
    x = x.expand(G, -1).contiguous()

    for cl in classes:
        ctx = ctx_builder(cl, G)
        m = torch.zeros(G, K, device=device)
        m[:, i] = torch.tensor(GI.ravel(), dtype=torch.float32, device=device)
        m[:, j] = torch.tensor(GJ.ravel(), dtype=torch.float32, device=device)
        scale, shift, _ = scale_and_shift(residual, x, ctx, m, direction)
        s_all = torch.log(scale.clamp_min(1e-6)).cpu().numpy().reshape(ngrid, ngrid, D)
        t_all = shift.cpu().numpy().reshape(ngrid, ngrid, D)

        for d in show_dims:
            rows = [(r"log-scale  $s$", s_all[:, :, d]),
                    (rf"shift  $\Delta {var_sym}_{d + 1}$", t_all[:, :, d])]
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
            for ri, (rlabel, arr) in enumerate(rows):
                add = arr[ic:ic + 1, :] + arr[:, ic:ic + 1] - arr[ic, ic]  # single_i + single_j − origin
                cross = arr - add
                fa = max(abs(arr.min()), abs(arr.max()), 1e-9)
                ca = max(abs(cross.min()), abs(cross.max()), 1e-9)
                panels = [("full", arr, fa),
                          ("additive  ($i$+$j$)", add, fa),
                          ("cross-term  (full − additive)", cross, ca)]
                for ci, (clabel, data, amp) in enumerate(panels):
                    ax = axes[ri, ci]
                    im = ax.imshow(data, origin="lower", extent=ext, aspect="auto",
                                   cmap="RdBu_r", vmin=-amp, vmax=amp)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.axhline(0, color="k", lw=0.5, ls=":")
                    ax.axvline(0, color="k", lw=0.5, ls=":")
                    ax.set_xlabel(f"${nu_syms[0]}$")
                    ax.set_ylabel(f"${nu_syms[1]}$")
                    ax.set_title(f"{rlabel} — {clabel}", fontsize=9)
            fig.suptitle(f"Residual {model_tag} — nuisance cross-term on {feat_syms[d]}  "
                         f"(class {'A' if cl == 0 else 'B'}, {direction} map){title_extra}\n"
                         f"probe {var_sym}=({probe[0]:g}, {probe[1]:g})", fontsize=12)
            fig.tight_layout()
            dsym = feat_syms[d].translate(str.maketrans("₁₂₃", "123"))
            cltag = "A" if cl == 0 else "B"
            _save(fig, out_dir, f"residual_{model_tag}_crossterm_{dsym}_class{cltag}")


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_kin(base_cfg, sys_cfg, device, out_dir, direction, m_value, n, response_dims=None,
            crossterm=True, crossterm_probe=(1.0, 0.5)):
    res_k = sys_cfg.get("residual_kin_model")
    if res_k is None or int(res_k.get("num_nuisances", 0)) == 0:
        print("  kin residual absent / num_nuisances==0 → skipping.")
        return
    kin_base = _build_base(base_cfg["kin_flow"], base_cfg["paths"]["kin_model"], device)
    residual = _build_residual(kin_base, res_k, sys_cfg["paths"]["kin_residual_model"], device)
    K = int(res_k["num_nuisances"])
    print(f"Visualising residual KIN  (K={K} nuisances, {direction}) …")

    # grid extent from the generator nominal support
    gen = _make_gen(sys_cfg["generator"], device)
    _, X_nom, _ = _gen_nominal(gen, n, device)
    x_np = X_nom.cpu().numpy()
    lo0, hi0 = np.percentile(x_np[:, 0], [0.5, 99.5])
    lo1, hi1 = np.percentile(x_np[:, 1], [0.5, 99.5])
    grid_range = ((float(lo0), float(hi0)), (float(lo1), float(hi1)))

    # representative probe points in x-space
    points = [
        ("centre",  [0.0,  0.0]),
        ("+x₁",     [1.0,  0.0]),
        ("−x₁",     [-1.0, 0.0]),
        ("+x₂",     [0.0,  0.8]),
        ("−x₂",     [0.0, -0.8]),
    ]
    ctx_builder = lambda cl, B: _onehot(cl, B, device)   # kin context = class one-hot

    plot_response(residual, model_tag="kin", feat_syms=("x₁", "x₂"), var_sym="x",
                  points=points, ctx_builder=ctx_builder, K=K, out_dir=out_dir,
                  device=device, direction=direction, display_tag="kin",
                  dims=response_dims)
    truth_fn = _make_kin_truth_fn(sys_cfg, device, n_truth=min(n, 40_000))
    plot_vector_field(residual, model_tag="kin", axis_syms=("x₁", "x₂"), var_sym="x",
                      grid_range=grid_range, ctx_builder=ctx_builder, K=K,
                      out_dir=out_dir, device=device, truth_fn=truth_fn,
                      direction=direction, m_value=m_value)
    if crossterm:
        plot_cross_term(residual, model_tag="kin", feat_syms=("x₁", "x₂"), var_sym="x",
                        probe=list(crossterm_probe), ctx_builder=ctx_builder, K=K,
                        out_dir=out_dir, device=device, direction=direction, dims=response_dims)


def run_score(base_cfg, sys_cfg, device, out_dir, direction, m_value, n, x_ctxs,
              response_dims=None, crossterm=True, crossterm_probe=(0.0, 0.0)):
    res_s = sys_cfg.get("residual_score_model")
    if res_s is None or int(res_s.get("num_nuisances", 0)) == 0:
        print("  score residual absent / num_nuisances==0 → skipping.")
        return
    score_path = sys_cfg["paths"].get("score_residual_model")
    if not score_path or not os.path.isfile(score_path):
        print(f"  score residual weights not found ({score_path}) → skipping.")
        return
    score_base = _build_base(base_cfg["score_flow"], base_cfg["paths"]["score_model"], device)
    residual = _build_residual(score_base, res_s, score_path, device)
    K = int(res_s["num_nuisances"])

    # representative probe points in y-space (pre-sigmoid score)
    points = [
        ("centre",  [0.0,  0.0]),
        ("+y₁",     [1.0,  0.0]),
        ("−y₁",     [-1.0, 0.0]),
        ("+y₂",     [0.0,  1.0]),
        ("−y₂",     [0.0, -1.0]),
    ]

    # The score residual is conditional on x, so loop over representative x-contexts:
    # the generator's score-shift is ∝ tanh(x₁) (zero at x₁=0), so probing several x
    # exposes the x-dependence of the deformation.
    for xi, x_ctx in enumerate(x_ctxs):
        suffix = "" if len(x_ctxs) == 1 else f"_x{xi}"
        xtag = f"x=({x_ctx[0]:g},{x_ctx[1]:g})"
        print(f"Visualising residual SCORE (K={K} nuisances, {direction}) at {xtag} …")
        x_ctx_t = torch.tensor(x_ctx, dtype=torch.float32, device=device)

        def ctx_builder(cl, B, _x=x_ctx_t):
            oh = _onehot(cl, B, device)
            xc = _x.unsqueeze(0).expand(B, -1)
            return torch.cat([oh, xc], dim=-1)

        # grid extent from the model's CONDITIONAL score density at this x_ctx (both
        # classes), so the figure zooms to p(y | x_ctx) rather than the x-marginal.
        with torch.no_grad():
            ys = torch.cat([residual.base_model(ctx_builder(cl, n)).rsample()
                            for cl in (0, 1)], dim=0).cpu().numpy()
        lo0, hi0 = np.percentile(ys[:, 0], [0.5, 99.5])
        lo1, hi1 = np.percentile(ys[:, 1], [0.5, 99.5])
        grid_range = ((float(lo0), float(hi0)), (float(lo1), float(hi1)))

        truth_fn = _make_score_truth_fn(sys_cfg, device, min(n, 40_000), x_ctx)
        plot_response(residual, model_tag=f"score{suffix}", feat_syms=("y₁", "y₂"),
                      var_sym="y", points=points, ctx_builder=ctx_builder, K=K,
                      out_dir=out_dir, device=device, direction=direction,
                      title_extra=f"  [{xtag}]", display_tag="score",
                      dims=response_dims)
        plot_vector_field(residual, model_tag=f"score{suffix}", axis_syms=("y₁", "y₂"),
                          var_sym="y", grid_range=grid_range, ctx_builder=ctx_builder, K=K,
                          out_dir=out_dir, device=device, truth_fn=truth_fn,
                          direction=direction, m_value=m_value, title_extra=f"  [{xtag}]")
        if crossterm:
            plot_cross_term(residual, model_tag=f"score{suffix}", feat_syms=("y₁", "y₂"),
                            var_sym="y", probe=list(crossterm_probe), ctx_builder=ctx_builder,
                            K=K, out_dir=out_dir, device=device, direction=direction,
                            dims=response_dims, title_extra=f"  [{xtag}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualise the learnt residual transform.")
    p.add_argument("-c", "--cfg", required=True, help="Base flows YAML config")
    p.add_argument("--sys-cfg", required=True, help="Systematics YAML config")
    p.add_argument("--out-dir", default="validation_transform", help="Output directory")
    p.add_argument("--direction", choices=["forward", "inverse"], default="forward",
                   help="forward: literal x*exp(s)+t (obs->nom); "
                        "inverse: systematic deformation (nom->obs)")
    p.add_argument("--m-value", type=float, default=1.0,
                   help="|ν| at which to draw the vector field")
    p.add_argument("--n", type=int, default=80_000, help="Nominal samples for backdrop")
    p.add_argument("--which", choices=["both", "kin", "score"], default="both")
    p.add_argument("--score-x", default="0,0;1.5,0",
                   help="';'-separated x-context points for the score residual "
                        "(it is conditional on x), e.g. '0,0;1.5,0'")
    p.add_argument("--response-dims", default=None,
                   help="Feature dims to show as rows in the response figure, comma-separated "
                        "(0 = y₁/x₁, 1 = y₂/x₂). E.g. '0' → only y₁ as a single row. "
                        "Default: all dims. The vector field is unaffected.")
    p.add_argument("--no-crossterm", action="store_true",
                   help="Skip the nuisance cross-term decomposition figures "
                        "(full = additive + cross over the ν_shift–ν_squeeze plane).")
    p.add_argument("--crossterm-y", default="0,0",
                   help="Fixed y point (pre-sigmoid score) at which the SCORE cross-term "
                        "is evaluated, 'y1,y2' (default '0,0').")
    p.add_argument("--crossterm-x", default="1.0,0.5",
                   help="Fixed x point at which the KIN cross-term is evaluated, "
                        "'x1,x2' (default '1.0,0.5').")
    args = p.parse_args()

    def _parse_point(s):
        a, b = s.split(",")
        return [float(a), float(b)]
    crossterm_x = _parse_point(args.crossterm_x)
    crossterm_y = _parse_point(args.crossterm_y)

    response_dims = None
    if args.response_dims:
        response_dims = [int(t) for t in args.response_dims.split(",") if t.strip() != ""]

    score_x_ctxs = []
    for tok in args.score_x.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(",")
        score_x_ctxs.append((float(a), float(b)))

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(cfg_path)
    for k, v in list(base_cfg["paths"].items()):
        if isinstance(v, str):
            base_cfg["paths"][k] = _resolve(v, cfg_dir)

    sys_path = os.path.abspath(args.sys_cfg)
    with open(sys_path) as f:
        sys_cfg = yaml.safe_load(f)
    sys_dir = os.path.dirname(sys_path)
    for k, v in list(sys_cfg["paths"].items()):
        if isinstance(v, str):
            sys_cfg["paths"][k] = _resolve(v, sys_dir)

    device = base_cfg.get("runtime", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}  |  Output: {args.out_dir}/  |  direction: {args.direction}")

    crossterm = not args.no_crossterm
    if args.which in ("both", "kin"):
        run_kin(base_cfg, sys_cfg, device, args.out_dir, args.direction, args.m_value, args.n,
                response_dims=response_dims, crossterm=crossterm, crossterm_probe=crossterm_x)
    if args.which in ("both", "score"):
        run_score(base_cfg, sys_cfg, device, args.out_dir, args.direction, args.m_value,
                  args.n, score_x_ctxs, response_dims=response_dims, crossterm=crossterm,
                  crossterm_probe=crossterm_y)

    print(f"\nDone. Plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
