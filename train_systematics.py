"""
Train residual (systematic-corrected) score and kinematic flow models.
Extracted from Systematics_inputs.ipynb (Cells 4-14).

Each base flow is frozen; only the SystematicCorrectedModel residual layers are trained.
The up/down generators use nuisance labels m=+1 / m=-1.

Steps:
  train_score  -- train residual score model
  train_kin    -- train residual kin model
"""
import argparse
import os
from typing import Any, Dict

import torch
import yaml
import zuko

from generator import ParametricLikelihoodDataset
from residual_flow import SystematicCorrectedModel
from utils import LinearWarmupCosineDecay


def _resolve_path(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _make_generator(c: Dict, device: str) -> ParametricLikelihoodDataset:
    kwargs = dict(device=device)
    for key in ("center_A", "center_B", "sigma_A", "sigma_B", "shift_dir"):
        if key in c:
            kwargs[key] = tuple(c[key])
    for key in ("variation_shift", "variation_rot", "variation_squeeze",
                "shift_scale", "rot_scale", "squeeze_scale",
                "y_shift_scale", "y_squeeze_scale", "distortion_strength",
                "distortion_shift_scale", "distortion_squeeze_scale"):
        if key in c:
            kwargs[key] = float(c[key])
    if "sigmoid_y" in c:
        kwargs["sigmoid_y"] = bool(c["sigmoid_y"])
    return ParametricLikelihoodDataset(**kwargs)


def _resolve_systematics_spec(cfg: Dict[str, Any], key: str) -> list:
    """Return a list of (generator_dict, m_vector_list) tuples.

    Accepts either:
      • Legacy two-generator schema (v11): `generator_kin_up` / `generator_kin_down`
        labelled m=+1 / m=-1 — used when no list is given.
      • Multi-nuisance list (v12+) under `cfg[key]`, each entry:
            { generator: <subkey of cfg>, m: [list of nuisance values] }
    """
    if key in cfg:
        spec = []
        for entry in cfg[key]:
            gen_dict = cfg[entry["generator"]]
            m_vec = [float(v) for v in entry["m"]]
            spec.append((gen_dict, m_vec))
        return spec
    # legacy fallback
    if key == "kin_systematics":
        return [(cfg["generator_kin_up"], [+1.0]),
                (cfg["generator_kin_down"], [-1.0])]
    if key == "score_systematics":
        return [(cfg["generator_score_up"], [+1.0]),
                (cfg["generator_score_down"], [-1.0])]
    raise KeyError(key)


def _train_kin(cfg: Dict[str, Any], device: str) -> None:
    spec = _resolve_systematics_spec(cfg, "kin_systematics")
    generators = [(_make_generator(gd, device), m) for gd, m in spec]
    n_nuis = len(spec[0][1])
    if any(len(m) != n_nuis for _, m in spec):
        raise ValueError("All m-vectors in kin_systematics must have the same length.")

    kin_cfg = cfg["kin_flow"]
    base_model = zuko.flows.NSF(
        features=kin_cfg["features"],
        context=kin_cfg["context"],
        bins=kin_cfg["bins"],
        transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)
    base_model.load_state_dict(torch.load(cfg["paths"]["kin_base_model"], map_location=device))

    res_cfg = cfg["residual_kin_model"]
    if int(res_cfg["num_nuisances"]) != n_nuis:
        raise ValueError(
            f"residual_kin_model.num_nuisances={res_cfg['num_nuisances']} disagrees with "
            f"kin_systematics m-vector length={n_nuis}"
        )
    model = SystematicCorrectedModel(
        base_model,
        features_dim=res_cfg["features_dim"],
        context_dim=res_cfg["context_dim"],
        num_nuisances=res_cfg["num_nuisances"],
        num_residual_layers=res_cfg["num_residual_layers"],
        hidden_features=res_cfg["hidden_features"],
        type=res_cfg["type"],
    ).to(device)

    train_cfg = cfg["training"]["kin"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    lr = float(train_cfg["lr"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = nepochs * steps_per_epoch
    scheduler = LinearWarmupCosineDecay(optimizer, steps_per_epoch, total_steps)

    chunk = batch_size // len(generators)
    global_step = 0
    print(f"Training residual kin model ({len(generators)} systematic generators, "
          f"{n_nuis} nuisances): {nepochs} epochs x {steps_per_epoch} steps")
    for epoch in range(nepochs):
        model.train()
        train_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            cs, Xs, ms = [], [], []
            for gen, m_vec in generators:
                c_i, X_i, _, _, _ = gen.generate_batch(chunk, distorsion=False)
                cs.append(c_i)
                Xs.append(X_i)
                m_i = torch.tensor(m_vec, device=device, dtype=torch.float32)
                ms.append(m_i.unsqueeze(0).expand(chunk, -1))
            c = torch.cat(cs, dim=0)
            X = torch.cat(Xs, dim=0)
            m = torch.cat(ms, dim=0)
            _, log_prob = model(X, c.to(torch.float32), m)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            global_step += 1
            if step % 500 == 0:
                print(f"  epoch={epoch} step={step}/{steps_per_epoch} loss={loss.item():.4f}")
        avg = train_loss / steps_per_epoch
        print(f"  epoch={epoch} avg_loss={avg:.4f}")

    out_path = cfg["paths"]["kin_residual_model"]
    torch.save(model.state_dict(), out_path)
    print(f"Saved residual kin model to {out_path}")


def _train_score(cfg: Dict[str, Any], device: str) -> None:
    res_cfg = cfg["residual_score_model"]
    if int(res_cfg["num_nuisances"]) == 0:
        print("residual_score_model.num_nuisances=0 — nothing to train; skipping.")
        return

    spec = _resolve_systematics_spec(cfg, "score_systematics")
    generators = [(_make_generator(gd, device), m) for gd, m in spec]
    n_nuis = len(spec[0][1])
    if any(len(m) != n_nuis for _, m in spec):
        raise ValueError("All m-vectors in score_systematics must have the same length.")
    if int(res_cfg["num_nuisances"]) != n_nuis:
        raise ValueError(
            f"residual_score_model.num_nuisances={res_cfg['num_nuisances']} disagrees with "
            f"score_systematics m-vector length={n_nuis}"
        )

    score_cfg = cfg["score_flow"]
    base_model = zuko.flows.NSF(
        features=score_cfg["features"],
        context=score_cfg["context"],
        bins=score_cfg["bins"],
        transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)
    base_model.load_state_dict(torch.load(cfg["paths"]["score_base_model"], map_location=device))

    model = SystematicCorrectedModel(
        base_model,
        features_dim=res_cfg["features_dim"],
        context_dim=res_cfg["context_dim"],
        num_nuisances=res_cfg["num_nuisances"],
        num_residual_layers=res_cfg["num_residual_layers"],
        hidden_features=res_cfg["hidden_features"],
        type=res_cfg["type"],
    ).to(device)

    train_cfg = cfg["training"]["score"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    lr = float(train_cfg["lr"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = nepochs * steps_per_epoch
    scheduler = LinearWarmupCosineDecay(optimizer, steps_per_epoch, total_steps)

    chunk = batch_size // len(generators)
    global_step = 0
    print(f"Training residual score model ({len(generators)} systematic generators, "
          f"{n_nuis} nuisances): {nepochs} epochs x {steps_per_epoch} steps")
    for epoch in range(nepochs):
        model.train()
        train_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            cs, Xs, ys, ms = [], [], [], []
            for gen, m_vec in generators:
                c_i, X_i, y_i, _, _ = gen.generate_batch(chunk, distorsion=False)
                cs.append(c_i)
                Xs.append(X_i)
                ys.append(y_i)
                m_i = torch.tensor(m_vec, device=device, dtype=torch.float32)
                ms.append(m_i.unsqueeze(0).expand(chunk, -1))
            c = torch.cat(cs, dim=0)
            X = torch.cat(Xs, dim=0)
            y = torch.cat(ys, dim=0)
            m = torch.cat(ms, dim=0)
            context = torch.cat([c.to(torch.float32), X], dim=-1)
            _, log_prob = model(y, context, m)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            global_step += 1
            if step % 500 == 0:
                print(f"  epoch={epoch} step={step}/{steps_per_epoch} loss={loss.item():.4f}")
        avg = train_loss / steps_per_epoch
        print(f"  epoch={epoch} avg_loss={avg:.4f}")

    out_path = cfg["paths"]["score_residual_model"]
    torch.save(model.state_dict(), out_path)
    print(f"Saved residual score model to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train systematic residual flows from YAML config.")
    parser.add_argument("-c", "--cfg", type=str, required=True)
    parser.add_argument("-s", "--steps", type=str, required=True,
                        help="Comma-separated: train_score,train_kin")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.cfg)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = os.path.dirname(cfg_path)
    cfg.setdefault("runtime", {})
    requested_device = cfg["runtime"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        requested_device = "cpu"

    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_path(value, cfg_dir)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if "train_score" in steps:
        _train_score(cfg, requested_device)
    if "train_kin" in steps:
        _train_kin(cfg, requested_device)


if __name__ == "__main__":
    main()
