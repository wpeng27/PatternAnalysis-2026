"""Atomic checkpoint files and epoch-boundary random states for reproducible resume.

Resume is supported at completed epoch boundaries on the same software/device
configuration. Mid-epoch restart and cross-platform bitwise identity are not promised.
AI-assisted implementation.
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


class WarmupCosine:
    """Linear warmup followed by cosine decay, indexed by successful updates.

    Inspired by ConvNeXt's schedule, implemented without a precomputed array:
    https://github.com/facebookresearch/ConvNeXt/blob/main/utils.py
    Adaptations: nonzero start, no duplicated peak at the warmup boundary,
    final planned update reaches min_lr, skipped AMP updates do not advance.
    The planned horizon is immutable across epoch-boundary resume.
    """

    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=1e-6, start_factor=0.01):
        if (not isinstance(total_steps, int) or not isinstance(warmup_steps, int)
                or warmup_steps < 0 or total_steps - warmup_steps < 2):
            raise ValueError("Cosine schedule requires nonnegative warmup and at least two decay steps")
        self.optimizer = optimizer
        self.settings = {"name": "WarmupCosine", "total_steps": total_steps, "warmup_steps": warmup_steps,
                         "min_lr": min_lr, "start_factor": start_factor,
                         "base_lrs": [float(g["lr"]) for g in optimizer.param_groups],
                         "step_unit": "successful optimizer update"}
        if (not math.isfinite(start_factor) or not 0 < start_factor <= 1
                or not math.isfinite(min_lr) or min_lr < 0 or not self.settings["base_lrs"]
                or any(not math.isfinite(lr) or lr <= 0 or lr < min_lr for lr in self.settings["base_lrs"])):
            raise ValueError("Invalid cosine learning-rate bounds")
        self.completed_updates = 0
        self._set_rates()

    def rates_at(self, completed_updates):
        """LR for the next update. The final rate is held after the horizon."""
        s = self.settings
        index = min(completed_updates, s["total_steps"] - 1)
        if index < s["warmup_steps"]:
            factor = s["start_factor"] + (1 - s["start_factor"]) * index / s["warmup_steps"]
            return [base * factor for base in s["base_lrs"]]
        progress = (index - s["warmup_steps"]) / (s["total_steps"] - s["warmup_steps"] - 1)
        return [s["min_lr"] + 0.5 * (base - s["min_lr"]) * (1 + math.cos(math.pi * progress))
                for base in s["base_lrs"]]

    def _set_rates(self):
        for group, lr in zip(self.optimizer.param_groups, self.rates_at(self.completed_updates)):
            group["lr"] = lr

    def step(self):
        if self.completed_updates >= self.settings["total_steps"]:
            raise ValueError("Cosine schedule horizon exhausted")
        self.completed_updates += 1
        self._set_rates()

    def state_dict(self):
        return {"settings": dict(self.settings), "completed_updates": self.completed_updates,
                "last_lrs": self.rates_at(self.completed_updates)}

    def load_state_dict(self, state):
        if state.get("settings") != self.settings:
            raise ValueError("Cosine schedule settings/horizon differ on resume")
        updates = state.get("completed_updates")
        if not isinstance(updates, int) or not 0 <= updates <= self.settings["total_steps"]:
            raise ValueError("Invalid cosine scheduler update count")
        expected = self.rates_at(updates)
        if state.get("last_lrs") != expected or [g["lr"] for g in self.optimizer.param_groups] != expected:
            raise ValueError("Cosine scheduler/optimizer learning rates disagree")
        self.completed_updates = updates
        self._set_rates()


def verify_cosine_checkpoint(checkpoint, config):
    """Verify saved scheduler settings, update count, optimizer LR and history."""
    settings = config["scheduler_settings"]
    if checkpoint["config"]["scheduler_settings"] != settings:
        raise ValueError("Cosine checkpoint/config settings mismatch")
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    # Scalar dummy parameters reconstruct the schedule without the full model.
    optimizer = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.zeros(()))], "lr": lr}
                                 for lr in settings["base_lrs"]])
    scheduler = WarmupCosine(optimizer, settings["total_steps"], settings["warmup_steps"],
                             settings["min_lr"], settings["start_factor"])
    if settings != scheduler.settings or len(groups) != len(optimizer.param_groups):
        raise ValueError("Invalid cosine settings or group count")
    for target, saved in zip(optimizer.param_groups, groups):
        target["lr"] = saved["lr"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    updates = 0
    for record in checkpoint["history"]:
        if record["learning_rate"] != scheduler.rates_at(updates)[0]:
            raise ValueError("Cosine history initial learning rate mismatch")
        updates += record["train"]["optimizer_steps"]
        if (record["next_learning_rate"] != scheduler.rates_at(updates)[0]
                or record.get("scheduler_updates") != updates):
            raise ValueError("Cosine history update/rate mismatch")
    if updates != scheduler.completed_updates:
        raise ValueError("Cosine update count differs from successful optimizer steps")
    return dict(settings)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_random_state(loaders):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": state[0], "keys": state[1].tolist(), "position": state[2],
                  "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generators": {name: loader.generator.get_state() for name, loader in loaders.items()},
    }


def restore_random_state(state, loaders):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.array(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    for name, loader in loaders.items():
        loader.generator.set_state(state["loader_generators"][name].cpu())


def update_selection(state, loss, epoch, patience, min_delta=0.0):
    """Strict loss decrease wins. Ties retain the earlier checkpoint."""
    improved = state["best_loss"] is None or loss < state["best_loss"] - min_delta
    if improved:
        state.update(best_loss=float(loss), best_epoch=epoch, bad_epochs=0)
    else:
        state["bad_epochs"] += 1
    state["stopped"] = state["bad_epochs"] >= patience
    return improved
