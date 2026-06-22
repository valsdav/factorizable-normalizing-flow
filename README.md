# Factorizable Normalizing Flow

Reference implementation accompanying the paper *Factorizable Normalizing Flow* (citation
below). This repository contains the **factorizable-flow** building block on a controllable
2-class toy: a **frozen pretrained base flow** plus a **trainable residual correction** that
absorbs systematic effects parameterized by continuous nuisance parameters **ν**.

> Scope: this release covers the density models — the base flows and their residual systematic
> ("input") models. The downstream profile-likelihood fitting pipeline is not included here.

---

## The idea

A systematic uncertainty deforms a probability density as a nuisance ν is varied. Instead of
re-training a flow per ν, we freeze a base flow trained at the nominal point (ν = 0) and learn a
**residual transform** that is the identity at ν = 0 and morphs the density as ν moves:

```
        x_obs  ──[ residual R(·; ν) ]──►  x_nom  ──[ frozen base flow ]──►  z ~ N(0, I)
```

The residual acts **per feature dimension** as an affine map whose log-scale `s` and shift `t`
are **low-order polynomials in ν**:

```
        R(x; ν):   x  ↦  x · exp(s(x, c, ν))  +  t(x, c, ν)
```

with `s, t` built from per-nuisance linear + (damped) quadratic terms and optional pairwise
cross-terms (`IndependentPolynomialResidualTransform` in `residual_flow.py`). Because only the
residual depends on ν — and it factors into an interpretable scale × shift — the systematic
response is **factorized** out of the base density and is cheap to evaluate and visualize.

Two density sectors are modelled, each as a frozen base NSF + residual:

| Sector | Density | Base context | Residual learns |
|--------|---------|--------------|-----------------|
| Kinematic | `p(x \| c)` | class one-hot | the ν-deformation of the 2D kinematics |
| Score | `p(y \| x, c)` | class one-hot + x | the ν-deformation of the 2D score, conditional on x |

The toy `generator.py` injects known systematics — an anti-correlated centroid **shift** (`ν_shift`)
and a volume-preserving **squeeze** (`ν_squeeze`) in x-space, plus matched location/scale
systematics in score-space — so the learned residual response can be validated against ground truth.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, [Zuko](https://github.com/probabilists/zuko) (normalizing flows), NumPy,
SciPy, Matplotlib, PyYAML. A GPU is optional; everything in this toy runs on CPU in minutes (the
code falls back to CPU automatically if CUDA is unavailable).

---

## Repository contents

```
factorizable-flow/
├── residual_flow.py                 # the method: SystematicCorrectedModel,
│                                     #   IndependentPolynomialResidualTransform, PermutationLayer
├── generator.py                     # ParametricLikelihoodDataset — the 2-class toy + truth
├── utils.py                         # LR scheduler + checkpoint helper
│
├── train_base_flows.py              # train the frozen base flows  p(x|c), p(y|x,c)
├── train_systematics.py             # train the residual systematic models on top of them
│
├── validate_flows.py                # validate base + residual models vs generator truth
├── visualize_residual_transform.py  # showcase the residual scale/shift response + displacement field
│
├── generate_dataset.py              # (optional) materialize a toy dataset to disk
├── visualize_dataset.py             # (optional) inspect a saved dataset
│
├── configs/
│   ├── base_flows.yaml              # base-flow architecture + training
│   ├── systematics.yaml             # residual models + systematic generators
│   └── dataset.yaml                 # toy dataset generation (optional path)
└── models/                          # pretrained checkpoints (committed)
    ├── score_density.pt             # base  p(y|x,c)
    ├── kin_density.pt               # base  p(x|c)
    ├── score_density_residuals.pt   # residual score model
    └── kin_density_residuals.pt     # residual kin model
```

---

## Quickstart A — reproduce the figures from the pretrained models

The four checkpoints in `models/` are shipped, so the figures reproduce with no training:

```bash
# Base- and residual-model closure vs the generator truth
python validate_flows.py -c configs/base_flows.yaml \
    --sys-cfg configs/systematics.yaml --out-dir validation/

# The method showcase: residual scale/shift response curves + displacement vector field
python visualize_residual_transform.py -c configs/base_flows.yaml \
    --sys-cfg configs/systematics.yaml --out-dir validation_transform/
```

`visualize_residual_transform.py` produces, for both the kin and score residuals:
- **`residual_*_response`** — the local **scale** `exp(s)` and **shift** `Δ` at representative
  phase-space points as each nuisance ν is swept (the factorized response);
- **`residual_*_field_nu{p,m}*`** — the displacement field `Δ = R(·;ν) − ·` over the phase space at
  fixed ν, overlaid on the model (and generator-truth) densities.

Useful flags: `--direction forward|inverse` (correction vs deformation), `--m-value` (|ν| for the
field), `--which kin|score|both`, `--score-x "0,0;1.5,0"` (the score residual is conditional on x).

---

## Quickstart B — train everything from scratch

```bash
# 1. (optional) materialize + inspect a toy dataset
python generate_dataset.py -c configs/dataset.yaml
python visualize_dataset.py data/dataset.pt --out-dir dataset_plots/

# 2. train the frozen base flows  p(x|c)  and  p(y|x,c)   (data generated on the fly)
python train_base_flows.py -c configs/base_flows.yaml -s train_score,train_kin

# 3. train the residual systematic models on top of the frozen bases
python train_systematics.py -c configs/systematics.yaml -s train_kin,train_score

# 4. validate / visualize as in Quickstart A
```

Steps 2–3 sample the generator directly (no saved dataset needed); `generate_dataset.py` is only
for inspecting the toy. Each `-s` step is independent and may be run separately.

---

## Configuration

All architecture, training, generator, and path settings live in the YAML files under `configs/`.
Paths are resolved relative to the config file. Key sections:

- `generator` (and the `generator_*` variations) — toy geometry and the injected systematics.
- `score_flow` / `kin_flow` — base NSF architecture (features, context, spline bins, transforms).
- `residual_score_model` / `residual_kin_model` — residual size (`num_nuisances`,
  `num_residual_layers`, `hidden_features`) and `type` (`flow` for an NSF base).
- `kin_systematics` / `score_systematics` — the ±1σ generator templates the residual is trained
  against, each labelled by its nuisance vector `m`.

---

## Citation

```bibtex
@article{factorizable_normalizing_flow,
  title  = {Factorizable Normalizing Flow},
  author = {Valsecchi, Davide and others},
  year   = {2026},
  note   = {TODO: fill in once published}
}
```

## License

See [LICENSE](LICENSE) (to be finalized before publishing).
