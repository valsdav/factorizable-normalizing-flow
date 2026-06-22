from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import math
# Define the warmup + cosine decay function
def LinearWarmupCosineDecay(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        # If before or within warmup, do linear warmup (inclusive of step==warmup_steps -> 1.0)
        if current_step <= warmup_steps:
            # Linear warmup from 0 -> 1 over warmup_steps
            return float(current_step) / float(max(1, warmup_steps))

        # After warmup, for steps up to total_steps use cosine decay from 1 -> 0.
        # For any step >= total_steps we must not let the lr go back up: clamp to 0.0
        if current_step >= total_steps:
            return 1e-8

        # Cosine decay in (warmup_steps, total_steps)
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine goes from 1 -> 0 as progress goes 0 -> 1
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def load_state_dict_checked(module, state_dict, label="model", error=True):
    """load_state_dict(strict=False) that surfaces silent architecture mismatches.

    `strict=False` tolerates missing/unexpected keys, which has repeatedly hidden
    residual mismatches (num_residual_layers / hidden_features / num_nuisances not
    matching the config the weights were trained with) — leaving a residual at
    identity init. This raises (error=True) or warns (error=False) when any model
    parameter is left unloaded or any checkpoint tensor is dropped. (Tensor-size
    mismatches already raise inside load_state_dict.) Returns the load result."""
    res = module.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        msg = (
            f"{label}: state_dict key mismatch — "
            f"{len(res.missing_keys)} model params NOT loaded (left at init: "
            f"{list(res.missing_keys)[:3]}{'…' if len(res.missing_keys) > 3 else ''}); "
            f"{len(res.unexpected_keys)} checkpoint tensors dropped "
            f"({list(res.unexpected_keys)[:3]}{'…' if len(res.unexpected_keys) > 3 else ''}). "
            "The residual architecture (num_residual_layers / hidden_features / "
            "num_nuisances) likely differs from the config the weights were trained with."
        )
        if error:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
    return res