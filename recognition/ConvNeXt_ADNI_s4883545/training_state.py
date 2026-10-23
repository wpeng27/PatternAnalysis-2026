"""Atomic checkpoint files and epoch-boundary random states for reproducible resume.

Resume is supported at completed epoch boundaries on the same software/device
configuration. Mid-epoch restart and cross-platform bitwise identity are not promised.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


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
