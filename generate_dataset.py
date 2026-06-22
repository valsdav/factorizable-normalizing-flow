"""
Generate the nominal toy dataset and save it to disk.

Samples the `generator` config section (no systematic pulls, no distortion) and
writes a dict with keys c, X, y. The base and residual training scripts generate
their own batches on the fly, so a saved dataset is only needed for inspection
(e.g. with visualize_dataset.py).
"""
import argparse
import os

import torch
import yaml

from generator import ParametricLikelihoodDataset


def _resolve_path(path: str, cfg_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cfg_dir, path))


def _make_gen(gcfg: dict, device: str) -> ParametricLikelihoodDataset:
    """Build a v12 ParametricLikelihoodDataset from a generator-section dict.

    All geometry / nuisance keys are optional — defaults match the v12 baseline.
    """
    kwargs = dict(device=device)
    for key in ("center_A", "center_B", "sigma_A", "sigma_B", "shift_dir"):
        if key in gcfg:
            kwargs[key] = tuple(gcfg[key])
    for key in ("variation_shift", "variation_rot", "variation_squeeze",
                "shift_scale", "rot_scale", "squeeze_scale",
                "y_shift_scale", "y_squeeze_scale",
                "distortion_strength",
                "distortion_shift_scale", "distortion_squeeze_scale"):
        if key in gcfg:
            kwargs[key] = float(gcfg[key])
    if "sigmoid_y" in gcfg:
        kwargs["sigmoid_y"] = bool(gcfg["sigmoid_y"])
    return ParametricLikelihoodDataset(**kwargs)


def _run_generate(cfg: dict, device: str) -> None:
    gen_nom = _make_gen(cfg["generator"], device)
    n_events = int(cfg["n_events"])

    print(f"Generating {n_events} nominal events "
          f"(variation_shift={gen_nom.variation_shift}, "
          f"variation_squeeze={gen_nom.variation_squeeze}) ...")
    c_true, X_true, y_true, _, _ = gen_nom.generate_batch(n_events, distorsion=False)

    out_path = cfg["paths"]["dataset"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Saving dataset to {out_path}")
    torch.save({"c": c_true, "X": X_true, "y": y_true}, out_path)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy dataset from YAML config.")
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the RNG before generation (for reproducibility). "
                             "Default: leave the global RNG state.")
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seeded RNG with {args.seed}")

    for key, value in list(cfg["paths"].items()):
        if isinstance(value, str):
            cfg["paths"][key] = _resolve_path(value, cfg_dir)

    _run_generate(cfg, requested_device)


if __name__ == "__main__":
    main()
