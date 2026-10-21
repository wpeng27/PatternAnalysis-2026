"""Training entry-point scaffold with a runnable GPU and dependency check.

The setup check uses random tensors. It is not an ADNI training experiment.
"""

import argparse
import json
import platform


def check_environment() -> dict:
    """Verify imports and an actual CUDA forward/backward computation."""
    import matplotlib
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.")
    torch.manual_seed(42)
    layer = torch.nn.Conv2d(3, 8, kernel_size=3).to("cuda")
    inputs = torch.randn(2, 3, 224, 224, device="cuda")
    loss = layer(inputs).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("CUDA check produced a non-finite loss or gradient.")
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "pillow": PIL.__version__,
        "numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "pytorch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "cuda_forward_backward": "passed", "model_training": "not performed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()
    if args.check_environment:
        print(json.dumps(check_environment(), indent=2))
    else:
        parser.error("Training is not implemented yet. Use --check-environment for the setup check.")


if __name__ == "__main__":
    main()
