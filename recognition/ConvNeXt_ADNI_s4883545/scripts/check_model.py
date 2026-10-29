"""Verify the complete ConvNeXt on synthetic grayscale images, without ADNI data.

Saves measured checks and a verification-only checkpoint, not an experiment or
a checkpoint accepted by train.py --resume. Raises on any failed check.
AI-assisted implementation.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import source_path

import argparse
from datetime import datetime, timezone
import hashlib

import torch
from torch.nn import functional as F

from dataset import CLASS_TO_INDEX
from modules import build_model
from train import PREPROCESSING, seed_everything
from training_state import save_checkpoint, write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_equal(left, right):
    """Exact comparison of nested optimizer/scaler state, across device storage."""
    if isinstance(left, torch.Tensor):
        require(torch.equal(left.cpu(), right.cpu()), "Reloaded state tensor differs")
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), "Reloaded state keys differ")
        for key in left:
            check_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        require(len(left) == len(right), "Reloaded state lengths differ")
        for a, b in zip(left, right):
            check_equal(a, b)
    else:
        require(left == right, "Reloaded state value differs")


def verify(output, device_name="cuda", amp=False):
    require(not output.exists(), "Output already exists. Choose a new verification directory.")
    device = torch.device(device_name)
    require(device.type in {"cpu", "cuda"}, "Use cpu or cuda")
    require(not amp or device.type == "cuda", "AMP verification requires CUDA")
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    torch.set_num_threads(4)
    seed_everything(42)
    output.mkdir(parents=True)
    report = {"state": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_kind": "Synthetic grayscale in [0, 1], repeated into three channels",
              "adni_data_used": False, "seed": 42, "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "precision": "CUDA float16 autocast with GradScaler" if amp else "float32",
              "torch": str(torch.__version__), "checks": {}}
    write_json(output / "checks.json", report)
    try:
        model = build_model("convnext_tiny").to(device).train()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report.update(total_parameters=total, trainable_parameters=trainable,
                      frozen_parameters=total - trainable)
        images = torch.rand(2, 1, 224, 224, device=device).repeat(1, 3, 1, 1).requires_grad_()
        labels = torch.tensor([CLASS_TO_INDEX["NC"], CLASS_TO_INDEX["AD"]], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        require(tuple(logits.shape) == (2, 2), "Expected two logits per image")
        require(torch.isfinite(logits).all().item(), "Non-finite logits")
        require(torch.isfinite(loss).item(), "Non-finite cross-entropy loss")
        report.update(input_shape=list(images.shape), output_shape=list(logits.shape), loss=loss.item())
        report["checks"].update(two_logits_per_image=True, finite_logits=True, finite_loss=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Input gradients are scaled in AMP, but must still be finite.
        require(images.grad is not None and torch.isfinite(images.grad).all().item(), "Invalid input gradient")
        nonzero_gradients = 0
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None, f"Missing gradient: {name}")
            require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}")
            nonzero_gradients += int(torch.count_nonzero(parameter.grad).item() > 0)
        require(nonzero_gradients > 0, "All parameter gradients are zero")
        report["checks"].update(finite_input_gradient=True, all_parameter_gradients_present_and_finite=True)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        require(scaler.get_scale() >= scale_before, "AMP skipped the optimizer update")
        changed = []
        for name, parameter in model.named_parameters():
            require(torch.isfinite(parameter).all().item(), f"Non-finite updated parameter: {name}")
            if not torch.equal(before[name], parameter.detach().cpu()):
                changed.append(name)
        require(changed, "Optimizer changed no parameters")
        report.update(parameter_tensors=len(before), nonzero_gradient_tensors=nonzero_gradients,
                      changed_parameter_tensors=len(changed), gradient_norm_before_clipping=gradient_norm.item(),
                      scaler_initial_scale=scale_before, scaler_final_scale=scaler.get_scale())
        report["checks"].update(optimizer_update_applied=True, updated_parameters_finite=True)
        del before
        config = {"model": "convnext_tiny", "class_to_index": CLASS_TO_INDEX,
                  "preprocessing": PREPROCESSING, "seed": 42, "pretrained": False,
                  "total_parameters": total, "trainable_parameters": trainable,
                  "architecture": str(model), "drop_path_rate": 0.0,
                  "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
                  "source_sha256": hashlib.sha256(source_path("modules.py").read_bytes()).hexdigest()}
        checkpoint_path = output / "verification.pt"
        save_checkpoint(checkpoint_path, {
            "purpose": "Synthetic model verification only", "model_name": "convnext_tiny",
            "optimizer_steps": 1, "config": config, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
        })
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        restored = build_model(saved["model_name"]).to(device)
        restored.load_state_dict(saved["model_state_dict"], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-4)
        restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
        restored_scaler = torch.amp.GradScaler("cuda", enabled=amp)
        restored_scaler.load_state_dict(saved["scaler_state_dict"])
        check_equal(model.state_dict(), restored.state_dict())
        check_equal(optimizer.state_dict(), restored_optimizer.state_dict())
        check_equal(scaler.state_dict(), restored_scaler.state_dict())
        check_equal(config, saved["config"])
        model.eval()
        restored.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
            expected = model(images.detach())
            actual = restored(images.detach())
        require(torch.isfinite(actual).all().item(), "Non-finite reloaded output")
        require(torch.equal(expected, actual), "Reloaded predictions differ on the same device")
        report.update(reload_max_absolute_logit_error=(expected - actual).abs().max().item(),
                      checkpoint=str(checkpoint_path.resolve()), checkpoint_bytes=checkpoint_path.stat().st_size,
                      class_to_index=CLASS_TO_INDEX, source_sha256=config["source_sha256"])
        report["checks"].update(checkpoint_saved_and_strictly_loaded=True, model_state_exact=True,
                                optimizer_state_exact=True, scaler_state_exact=True, config_exact=True,
                                reloaded_eval_logits_exact=True)
        report["state"] = "passed"
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "checks.json", report)
    return report


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory for checks.json and verification.pt")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.device, args.amp), indent=2))


if __name__ == "__main__":
    main()
