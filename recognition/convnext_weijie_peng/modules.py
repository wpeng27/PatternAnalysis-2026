"""Model entry point for Weijie Peng's COMP3710 project.

Planned architectures: a small CNN baseline and ConvNeXt-Tiny, both initially
trained from scratch. Implementations belong here in the next milestone.
"""

MODEL_NAMES = ("small_cnn", "convnext_tiny")
NUM_CLASSES = 2
INPUT_CHANNELS = 3
IMAGE_SIZE = 224


def build_model(name: str):
    """Reserved model factory. Never silently substitute an unimplemented model."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
    raise NotImplementedError("Model implementation is the next project milestone.")
