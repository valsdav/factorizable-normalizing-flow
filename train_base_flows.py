"""
Train base score and kinematic normalizing flows (no nuisance parameters).
Extracted from ToyDataset_inputs.ipynb (Cells 14-35).

Steps:
  train_score  -- train conditional score flow p(y|x,c)
  train_kin    -- train kinematic flow p(x|c)
"""
import argparse
import os
from typing import Any, Dict

import torch
import yaml
import zuko

from generator import ParametricLikelihoodDataset
from utils import LinearWarmupCosineDecay


def _resolve_path(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _make_generator(gen_cfg: Dict[str, Any], device: str) -> ParametricLikelihoodDataset:
    kwargs: Dict[str, Any] = dict(device=device)
    for key in ("center_A", "center_B", "sigma_A", "sigma_B", "shift_dir"):
        if key in gen_cfg:
            kwargs[key] = tuple(gen_cfg[key])
    for key in ("variation_shift", "variation_rot", "variation_squeeze",
                "shift_scale", "rot_scale", "squeeze_scale",
                "y_shift_scale", "y_squeeze_scale", "distortion_strength",
                "distortion_shift_scale", "distortion_squeeze_scale"):
        if key in gen_cfg:
            kwargs[key] = float(gen_cfg[key])
    if "sigmoid_y" in gen_cfg:
        kwargs["sigmoid_y"] = bool(gen_cfg["sigmoid_y"])
    return ParametricLikelihoodDataset(**kwargs)


def _train_score(cfg: Dict[str, Any], device: str) -> None:
    gen_cfg = cfg["generator"]
    gen = _make_generator(gen_cfg, device)

    score_cfg = cfg["score_flow"]
    model = zuko.flows.NSF(
        features=score_cfg["features"],
        context=score_cfg["context"],
        bins=score_cfg["bins"],
        transforms=score_cfg["transforms"],
        hidden_features=tuple(score_cfg["hidden_features"]),
    ).to(device)

    train_cfg = cfg["training"]["score"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    lr = float(train_cfg["lr"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = nepochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    scheduler = LinearWarmupCosineDecay(optimizer, warmup_steps, total_steps)

    print(f"Training score flow: {nepochs} epochs x {steps_per_epoch} steps, batch={batch_size}")
    for epoch in range(nepochs):
        model.train()
        train_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            c, X, y, _, _ = gen.generate_batch(batch_size, distorsion=False)
            context = torch.cat([c.to(torch.float32), X], dim=-1)
            log_prob = model(context).log_prob(y)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            if step % 500 == 0:
                print(f"  epoch={epoch} step={step}/{steps_per_epoch} loss={loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(10):
                c, X, y, _, _ = gen.generate_batch(batch_size, distorsion=False)
                context = torch.cat([c.to(torch.float32), X], dim=-1)
                val_loss += -model(context).log_prob(y).mean().item()
        val_loss /= 10
        print(f"  epoch={epoch} train_loss={train_loss/steps_per_epoch:.4f} val_loss={val_loss:.4f}")

    out_path = cfg["paths"]["score_model"]
    torch.save(model.state_dict(), out_path)
    print(f"Saved score model to {out_path}")


def _train_kin(cfg: Dict[str, Any], device: str) -> None:
    gen_cfg = cfg["generator"]
    gen = _make_generator(gen_cfg, device)

    kin_cfg = cfg["kin_flow"]
    model = zuko.flows.NSF(
        features=kin_cfg["features"],
        context=kin_cfg["context"],
        bins=kin_cfg["bins"],
        transforms=kin_cfg["transforms"],
        hidden_features=tuple(kin_cfg["hidden_features"]),
    ).to(device)

    train_cfg = cfg["training"]["kin"]
    batch_size = int(train_cfg["batch_size"])
    nepochs = int(train_cfg["nepochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    lr = float(train_cfg["lr"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = nepochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    scheduler = LinearWarmupCosineDecay(optimizer, warmup_steps, total_steps)

    print(f"Training kin flow: {nepochs} epochs x {steps_per_epoch} steps, batch={batch_size}")
    for epoch in range(nepochs):
        model.train()
        train_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            c, X, _, _, _ = gen.generate_batch(batch_size, distorsion=False)
            log_prob = model(c.to(torch.float32)).log_prob(X)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            if step % 500 == 0:
                print(f"  epoch={epoch} step={step}/{steps_per_epoch} loss={loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(10):
                c, X, _, _, _ = gen.generate_batch(batch_size, distorsion=False)
                val_loss += -model(c.to(torch.float32)).log_prob(X).mean().item()
        val_loss /= 10
        print(f"  epoch={epoch} train_loss={train_loss/steps_per_epoch:.4f} val_loss={val_loss:.4f}")

    out_path = cfg["paths"]["kin_model"]
    torch.save(model.state_dict(), out_path)
    print(f"Saved kin model to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train base score/kin flows from YAML config.")
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
