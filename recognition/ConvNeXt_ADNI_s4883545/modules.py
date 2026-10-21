"""Small CNN baseline. AI-assisted implementation for Weijie Peng's COMP3710 project."""

from torch import nn

MODEL_NAMES = ("small_cnn", "convnext_tiny")
NUM_CLASSES = 2
INPUT_CHANNELS = 3
IMAGE_SIZE = 224


class SmallCNN(nn.Module):
    """Three convolution blocks and a two-class classifier, trained from scratch."""

    def __init__(self):
        super().__init__()
        blocks = []
        channels = INPUT_CHANNELS
        for output_channels in (16, 32, 64):
            blocks.extend([
                nn.Conv2d(channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ])
            channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            # Three pools reduce 224 to 28. Fixed pooling yields a 4x4 feature map.
            nn.AvgPool2d(kernel_size=7), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, images):
        # CrossEntropyLoss expects raw logits, not softmax probabilities.
        return self.classifier(self.features(images))


def build_model(name: str) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
    raise NotImplementedError("ConvNeXt implementation is a later project milestone.")
