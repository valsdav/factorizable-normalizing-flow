import math
import torch
import numpy as np


class ParametricLikelihoodDataset:
    """v12 / v13 — 2D kinematics, sigmoid score, x-only systematics.

    Design (see plan file):
      • x  : 2D, two Gaussian clusters with distinct centroids and covariances.
      • y  : 2D pre-sigmoid Gaussian with x-dependent mean/cov, then sigmoid → (0,1)².
      • Data distortion (`apply_distortion_map`): class-dependent rotation + shear on
        pre-sigmoid y. Antisymmetric in x_1 (odd parity), so it is orthogonal to any
        even-in-x systematic — but here we have no y-space systematics, so the
        orthogonality argument is structural, not statistical.
      • Systematics: act on x only. Each ν has a 0 default so a given config can
        activate any subset.
            ν_shift   : anti-correlated centroid shift along `shift_dir`
            ν_rot     : opposite-sign rotation of each class around its own centroid
            ν_squeeze : per-class axis-anti-correlated squeeze about each centroid,
                        x → μ + diag(e^{±α}, e^{∓α})·(x − μ) with α = ν · squeeze_scale.
                        Class A stretches x₁ / shrinks x₂; class B is the mirror.
                        Volume-preserving (det = 1), centroid-preserving, first-order
                        on the diagonal of Σ → high Fisher information at ν = 0.

    v13 configs activate [ν_shift, ν_squeeze] and zero ν_rot; v12 configs are
    backward-compatible (ν_squeeze stays at 0 by default).

    Dropped vs v11: variation_y, variation_norm, x_shift, rotation_angle.
    """

    def __init__(self,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 # x-space cluster geometry
                 center_A=(-0.5, 0.0),
                 center_B=(+0.5, 0.0),
                 sigma_A=(0.9, 0.6),
                 sigma_B=(0.6, 0.4),
                 # systematics on x
                 variation_shift=0.0,
                 variation_rot=0.0,
                 variation_squeeze=0.0,
                 shift_scale=0.3,
                 shift_dir=(1.0, 0.0),
                 rot_scale=0.3,
                 squeeze_scale=0.2,
                 # score-space systematics (v14): explicit nuisance dependence of p(y|x).
                 # Driven by the SAME nuisances that move x, so x pins them; default 0
                 # reproduces v12/v13 exactly. ν_shift -> linear score-mean shift (Δm*=∞),
                 # ν_squeeze -> exp score-spread scaling (Δm* = 1/y_squeeze_scale).
                 y_shift_scale=0.0,
                 y_squeeze_scale=0.0,
                 # distortion strength (y-space, pre-sigmoid)
                 distortion_strength=0.0,
                 # nuisance dependence of the distortion MAP (v15): gives the data↔MC
                 # transfer T a genuine ν-dependence that ONLY residual_transfer (step-2)
                 # can absorb — residual_score is trained on undistorted templates so it
                 # cannot represent it. This breaks the score/transfer degeneracy and
                 # gives step-2 profiling a real effect. Driven by the SAME nuisances
                 # (ν_shift -> rotation gain, ν_squeeze -> shear gain). Default 0
                 # reproduces the ν-independent distortion of v12–v14 exactly.
                 distortion_shift_scale=0.0,
                 distortion_squeeze_scale=0.0,
                 # output transform
                 sigmoid_y=True,
                 ):
        self.device = device

        # Cluster geometry → tensors on device
        self.center_A = torch.tensor(center_A, device=device, dtype=torch.float32)
        self.center_B = torch.tensor(center_B, device=device, dtype=torch.float32)
        self.sigma_A = torch.tensor(sigma_A, device=device, dtype=torch.float32)
        self.sigma_B = torch.tensor(sigma_B, device=device, dtype=torch.float32)

        # Systematics
        self.variation_shift = float(variation_shift)
        self.variation_rot = float(variation_rot)
        self.variation_squeeze = float(variation_squeeze)
        self.shift_scale = float(shift_scale)
        d = torch.tensor(shift_dir, device=device, dtype=torch.float32)
        self.shift_dir = d / (torch.linalg.norm(d) + 1e-12)
        self.rot_scale = float(rot_scale)
        self.squeeze_scale = float(squeeze_scale)
        self.y_shift_scale = float(y_shift_scale)
        self.y_squeeze_scale = float(y_squeeze_scale)

        # Distortion
        self.distortion_strength = float(distortion_strength)
        self.distortion_shift_scale = float(distortion_shift_scale)
        self.distortion_squeeze_scale = float(distortion_squeeze_scale)

        # Output transform
        self.sigmoid_y = bool(sigmoid_y)

        self.pi = torch.tensor(np.pi, device=device)

    # ------------------------------------------------------------------ x

    def _rotate_about(self, x, centroid, angle):
        """Rotate the rows of x (shape [N, 2]) about `centroid` by scalar `angle`."""
        cos_t = torch.cos(torch.tensor(angle, device=x.device, dtype=x.dtype))
        sin_t = torch.sin(torch.tensor(angle, device=x.device, dtype=x.dtype))
        R = torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t,  cos_t]),
        ])
        return (x - centroid) @ R.T + centroid

    def _squeeze_about(self, x, centroid, ax1, ax2):
        """Scale x − centroid by diag(ax1, ax2). Volume-preserving iff ax1*ax2 == 1."""
        s = torch.tensor([ax1, ax2], device=x.device, dtype=x.dtype)
        return centroid + (x - centroid) * s

    def sample_x_conditional(self, c, n_samples,
                             variation_shift=None, variation_rot=None,
                             variation_squeeze=None):
        """Sample 2D x conditional on class c.

        c: long tensor of shape [n_samples, 1] with values in {0, 1}.
        Returns x of shape [n_samples, 2].
        """
        if variation_shift is None:
            variation_shift = self.variation_shift
        if variation_rot is None:
            variation_rot = self.variation_rot
        if variation_squeeze is None:
            variation_squeeze = self.variation_squeeze

        x = torch.zeros((n_samples, 2), device=self.device)
        mA = (c == 0).flatten()
        mB = (c == 1).flatten()

        # 1. draw from nominal clusters (diagonal cov via per-axis sigmas)
        if mA.any():
            x[mA] = torch.randn(int(mA.sum()), 2, device=self.device) * self.sigma_A + self.center_A
        if mB.any():
            x[mB] = torch.randn(int(mB.sum()), 2, device=self.device) * self.sigma_B + self.center_B

        # 2. ν_rot: opposite-sign rotation about each class's own centroid
        if variation_rot != 0.0:
            if mA.any():
                x[mA] = self._rotate_about(x[mA], self.center_A,
                                           -variation_rot * self.rot_scale)
            if mB.any():
                x[mB] = self._rotate_about(x[mB], self.center_B,
                                           +variation_rot * self.rot_scale)

        # 3. ν_squeeze: per-class axis-anti-correlated squeeze about each centroid.
        #    Class A stretches x₁ / shrinks x₂; class B is the mirror.
        #    Uses diag(e^{±α}, e^{∓α}) so det = 1 (no Jacobian leak into density).
        if variation_squeeze != 0.0:
            a = variation_squeeze * self.squeeze_scale
            ep, em = math.exp(+a), math.exp(-a)
            if mA.any():
                x[mA] = self._squeeze_about(x[mA], self.center_A, ep, em)
            if mB.any():
                x[mB] = self._squeeze_about(x[mB], self.center_B, em, ep)

        # 4. ν_shift: anti-correlated centroid shift along shift_dir (last, since
        #    it moves the centroid that 2/3 are defined about).
        if variation_shift != 0.0:
            delta = variation_shift * self.shift_scale * self.shift_dir
            if mA.any():
                x[mA] = x[mA] - delta
            if mB.any():
                x[mB] = x[mB] + delta

        return x

    # ------------------------------------------------------------------ p(y|x,c)

    def get_base_dist_params(self, x, c, variation_shift=None, variation_squeeze=None):
        """Return (mu, L) for the pre-sigmoid bivariate Gaussian p(y_raw | x, c).

        x: [B, 2]; c: [B, 1].

        Score-space systematics (v14): when ``y_shift_scale`` / ``y_squeeze_scale`` are
        non-zero, p(y|x,c) gains an EXPLICIT dependence on the nuisances (beyond the
        x-mediated one), so the means become class-dependent. ``variation_shift`` /
        ``variation_squeeze`` default to the generator's own values; pass them explicitly
        to build templates at a given nuisance point.
            ν_shift   -> class-anti-correlated LINEAR shift of the score mean   (Δm* = ∞)
            ν_squeeze -> class-anti-correlated EXPONENTIAL scaling of the score
                         spread (det = 1 per class)                            (Δm* = 1/y_squeeze_scale)
        """
        if variation_shift is None:
            variation_shift = self.variation_shift
        if variation_squeeze is None:
            variation_squeeze = self.variation_squeeze

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        # Means — 2D-x generalisation of the v11 1D dependence
        mu_0 = torch.sin(1.5 * x1) + 0.3 * x2
        mu_1 = 0.3 * x1**2 - 1.2 + 0.5 * torch.sin(x2)
        mu = torch.cat([mu_0, mu_1], dim=1)  # [B, 2]

        # Per-class sign (A: -1, B: +1) for the anti-correlated score systematics.
        sign = torch.where(c == 0, -1.0, 1.0).to(mu.dtype)  # [B, 1]

        # ν_shift score systematic: linear, anti-correlated, x-dependent mean shift.
        if variation_shift != 0.0 and self.y_shift_scale != 0.0:
            d = variation_shift * self.y_shift_scale
            g = torch.tanh(x1)  # smooth, bounded, non-trivial x-dependence
            mu = mu + torch.cat([sign * d * g, -sign * d * g], dim=1)

        # Diagonal sigmas + x-dependent correlation
        sigma_0 = torch.nn.functional.softplus(0.4 * x1 + 0.1)
        sigma_1 = torch.nn.functional.softplus(-0.2 * x1 + 0.4)

        # ν_squeeze score systematic: exp, anti-correlated per-class spread scaling
        # (det of the diag scaling = 1, so it reshapes the score covariance without
        # inflating it — directly analogous to the x-space squeeze).
        if variation_squeeze != 0.0 and self.y_squeeze_scale != 0.0:
            beta = variation_squeeze * self.y_squeeze_scale
            sigma_0 = sigma_0 * torch.exp(+sign * beta)
            sigma_1 = sigma_1 * torch.exp(-sign * beta)

        rho = 0.8 * torch.tanh(0.5 * (x1 + x2))

        var_0 = sigma_0**2
        var_1 = sigma_1**2
        cov_01 = rho * sigma_0 * sigma_1

        row0 = torch.stack([var_0, cov_01], dim=1)
        row1 = torch.stack([cov_01, var_1], dim=1)
        Sigma = torch.stack([row0, row1], dim=1).squeeze(-1)
        L = torch.linalg.cholesky(Sigma)
        return mu, L

    # ------------------------------------------------------------------ distortion

    def apply_distortion_map(self, y, x, c, variation_shift=None, variation_squeeze=None):
        """Pre-sigmoid distortion applied to the bivariate-Gaussian y.

        Class-dependent rotation (opposite signs) with x-dependent magnitude that is
        odd in x_1, plus a small class-dependent shear.

        The MAP itself carries a small nuisance dependence when
        ``distortion_shift_scale`` / ``distortion_squeeze_scale`` are set (v15): the
        rotation gain is ``1 + distortion_shift_scale·ν_shift`` and the shear gain is
        ``1 + distortion_squeeze_scale·ν_squeeze``. This changes the data↔MC transfer
        at a FIXED y, so it is a genuine ν-dependence of T that residual_transfer
        (step-2) must profile — distinct from the score systematic (which moves the
        pre-distortion density p(y|x,ν) and is owned by residual_score). ``variation_*``
        default to the generator's own values. Scales = 0 reproduce the
        ν-independent distortion exactly.
        """
        if variation_shift is None:
            variation_shift = self.variation_shift
        if variation_squeeze is None:
            variation_squeeze = self.variation_squeeze

        sign = torch.zeros_like(c, dtype=y.dtype, device=y.device)  # [B, 1]
        sign[c == 0] = -1.0
        sign[c == 1] = +1.0

        # radial decay of the rotation magnitude, with a ν_shift-dependent gain
        r = torch.linalg.norm(x, dim=1, keepdim=True)  # [B, 1]
        base_strength = 2.0 * self.distortion_strength
        rot_gain = 1.0 + self.distortion_shift_scale * variation_shift
        theta_mag = rot_gain * base_strength / (0.7 * r + 0.5)

        # antisymmetric in x_1, class-dependent overall sign
        x1 = x[:, 0:1]
        theta = sign * theta_mag * torch.tanh(x1 / 0.5)  # [B, 1]

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.stack([
            torch.cat([cos_t, -sin_t], dim=1),
            torch.cat([sin_t,  cos_t], dim=1),
        ], dim=1)  # [B, 2, 2]

        y_rot = torch.bmm(R, y.unsqueeze(-1)).squeeze(-1)

        # class-dependent shear mixing y_0 and y_1 (sign-symmetric to keep balance),
        # with a ν_squeeze-dependent gain on the shear coefficient.
        shear_coef = 0.10 * (1.0 + self.distortion_squeeze_scale * variation_squeeze)
        shear = torch.zeros_like(y_rot)
        shear[:, 0] = sign.squeeze(-1) * ( shear_coef * y_rot[:, 0] - shear_coef * y_rot[:, 1])
        shear[:, 1] = sign.squeeze(-1) * (-shear_coef * y_rot[:, 0] + shear_coef * y_rot[:, 1])
        y_dist = y_rot + shear

        return y_dist

    # ------------------------------------------------------------------ batch

    def generate_batch(self, batch_size=1024, distorsion=False, c=None):
        """Generate a batch.

        X always reflects the generator's `variation_shift` / `variation_rot` settings —
        same semantics as v11 (where `X` reflected `variation_x`). The `distorsion` flag
        only governs whether the pre-sigmoid y-distortion is applied.

        If `c` is provided ([batch_size, 1] long tensor in {0, 1}), it is used instead of
        drawing fresh class labels — this lets a caller synchronise the class assignment
        across two generators (e.g. nominal + data) so the per-event correspondence
        survives downstream.

        Returns (c_onehot, X, y, y_data_distorted_or_None, X_data_distorted_or_None).
        Shapes:
          c_onehot              [B, 2]
          X (kinematics)        [B, 2]   — reflects ν_shift / ν_rot of this generator
          y (score)             [B, 2]   in (0,1) if sigmoid_y; from p(y|X,c)
          y_data_distorted      [B, 2]   in (0,1); pre-sigmoid distortion applied   (None if not requested)
          X_data_distorted      [B, 2]   == X in v12 (kept for tuple compat)        (None if not requested)
        """
        if c is None:
            # 50/50 class draw (no normalisation systematic in v12)
            c = (torch.rand((batch_size, 1), device=self.device) > 0.5).long()
        else:
            c = c.to(self.device).long()
            assert c.shape == (batch_size, 1), f"c must have shape ({batch_size}, 1), got {tuple(c.shape)}"

        X = self.sample_x_conditional(c, batch_size,
                                      variation_shift=self.variation_shift,
                                      variation_rot=self.variation_rot,
                                      variation_squeeze=self.variation_squeeze)

        mu, L = self.get_base_dist_params(X, c,
                                          variation_shift=self.variation_shift,
                                          variation_squeeze=self.variation_squeeze)
        eps = torch.randn(batch_size, 2, 1, device=self.device)
        y_raw = mu + torch.bmm(L, eps).squeeze(-1)  # pre-sigmoid

        if distorsion:
            y_dist_raw = self.apply_distortion_map(y_raw, X, c,
                                                   variation_shift=self.variation_shift,
                                                   variation_squeeze=self.variation_squeeze)
            y_dist = torch.sigmoid(y_dist_raw) if self.sigmoid_y else y_dist_raw
            X_d = X  # no separate kinematic data-shift in v12; X already reflects systematics
        else:
            y_dist = None
            X_d = None

        y_nom = torch.sigmoid(y_raw) if self.sigmoid_y else y_raw

        c_onehot = torch.nn.functional.one_hot(c.flatten(), 2)
        return c_onehot, X, y_nom, y_dist, X_d


# =====================================================================
# Previous versions (kept for lineage — see Paper_plots.ipynb history)
# =====================================================================
#
# ---- v11 (1D x, class-agnostic vortex + rotation_angle patch) -------
#
#     def __init__(self, device=..., x_separation=2.0, variation_x=0.0,
#                  variation_y=0.0, variation_norm=0.0,
#                  distortion_strength=0.0, x_shift=[0.], rotation_angle=0.0):
#         self.center_0 = -x_separation / 2.0
#         self.center_1 = +x_separation / 2.0
#         self.rotation_angle = float(rotation_angle)
#
#     def sample_x_conditional(self, c, n_samples, variation_x=0.):
#         x = torch.zeros((n_samples, 1), device=self.device)
#         mask_0 = (c == 0).flatten();  mask_1 = (c == 1).flatten()
#         if mask_0.any():
#             raw_x = torch.randn(mask_0.sum(), 1, device=self.device) * 0.9 + self.center_0
#             x[mask_0] = raw_x * (1 + variation_x)
#         if mask_1.any():
#             raw_x = torch.randn(mask_1.sum(), 1, device=self.device) * 0.7 + self.center_1
#             x[mask_1] = raw_x * (1.0 - variation_x)
#         return x
#
#     def get_base_dist_params(self, x, c, variation_y=0.):
#         sign = torch.zeros((c.shape[0], 1), device=c.device); sign += c
#         sign[sign == 0] += -1.
#         mu_0 = torch.sin(x * 1.5) - sign * variation_y
#         mu_1 = 0.3 * x**2 - 1.2 + sign * variation_y
#         mu = torch.cat([mu_0, mu_1], dim=1)
#         sigma_0 = torch.nn.functional.softplus(0.4 * x + 0.1)
#         sigma_1 = torch.nn.functional.softplus(-0.2 * x + 0.4)
#         rho = torch.tanh(x) * 0.8
#         # … Cholesky as in v12 …
#
#     def apply_distortion_map(self, y, x, c):  # v11 — class-agnostic
#         y_dist = y.clone()
#         y_dist = y_dist + 0.15 * y_dist[:, 0:1] - 0.15 * y_dist[:, 1:2]
#         base_strength = 2.0 * self.distortion_strength
#         decay_rate = 0.7
#         theta_mag = base_strength / (decay_rate * torch.abs(x) + 0.5)
#         theta_vortex = theta_mag * torch.tanh(x / 0.5) + self.rotation_angle
#         cos_t, sin_t = torch.cos(theta_vortex), torch.sin(theta_vortex)
#         R_vortex = torch.stack([torch.cat([cos_t, -sin_t], dim=1),
#                                 torch.cat([sin_t,  cos_t], dim=1)], dim=1)
#         y_final = torch.bmm(R_vortex, y_dist.unsqueeze(-1)).squeeze(-1)
#         y_final = y_final * (1 - 0.04 * x)
#         return y_final
#
# ---- v10, v9, v8, v6, v5 distortion variants are kept in the git history
#      of this file (commit 4c01444 onwards) and not duplicated here.
