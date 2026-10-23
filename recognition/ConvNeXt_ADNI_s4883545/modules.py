"""ADNI model components. AI-assisted implementation for Weijie Peng's COMP3710 project."""

import math
import torch
from torch import nn
from torchvision.ops import StochasticDepth

MODEL_NAMES = ("small_cnn", "convnext_tiny")
NUM_CLASSES = 2
INPUT_CHANNELS = 3
IMAGE_SIZE = 224


class ConvNeXtBlock(nn.Module):
    """Shape-preserving ConvNeXt V1 block: NCHW -> NCHW.

    Adapted from Liu et al., A ConvNet for the 2020s (CVPR 2022),
    sections 2.3-2.6 and 3.1: https://arxiv.org/abs/2201.03545
    Official Block: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Copyright (c) Meta Platforms, Inc. and affiliates. MIT notice in LICENSE-ConvNeXt.txt.
    Sources checked 2026-10-23.

    Adaptations: native nn.LayerNorm replaces the official channels-last wrapper.
    torchvision StochasticDepth(mode="row") replaces timm DropPath, retaining
    one residual-branch mask per sample and survival-probability scaling in train
    mode, with no dropping in eval mode. No new timm dependency is required.
    https://docs.pytorch.org/vision/0.24/generated/torchvision.ops.stochastic_depth.html
    Layer scale is always learnable here, including when initialised to zero
    (an identity block initially). The official implementation disables it at zero.

    dim is the feature-channel count, not necessarily the three image channels.
    Spatial downsampling and full-model weight initialisation belong to
    ConvNeXtTiny. This standalone block uses PyTorch's default Conv/Linear init.
    drop_path is in [0, 1], where 1 drops the entire branch during training.
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not math.isfinite(drop_path) or not 0 <= drop_path <= 1:
            raise ValueError("drop_path must be finite and in [0, 1]")
        if not math.isfinite(layer_scale_init_value) or layer_scale_init_value < 0:
            raise ValueError("layer_scale_init_value must be finite and nonnegative")
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale_init_value)))
        self.drop_path = StochasticDepth(float(drop_path), mode="row")

    def forward(self, features):
        residual = features
        # NHWC makes LayerNorm and Linear operate only on channels at each pixel.
        branch = self.dwconv(features).permute(0, 2, 3, 1)
        branch = self.pwconv2(self.act(self.pwconv1(self.norm(branch))))
        branch = (branch * self.gamma).permute(0, 3, 1, 2)
        return residual + self.drop_path(branch)


class ChannelLayerNorm(nn.LayerNorm):
    """Normalize channels independently at each NCHW spatial position.

    Adaptation of the official ConvNeXt channels-first LayerNorm: permute to
    NHWC and use native LayerNorm, then restore NCHW. Spatial axes are not
    normalized. Inherits the standard learnable weight and bias.
    """

    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-6)

    def forward(self, features):
        return super().forward(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny V1 with a two-class ADNI head, initialized from scratch.

    Adapted from Liu et al. (CVPR 2022), section 3.1 and the official ConvNeXt:
    https://arxiv.org/abs/2201.03545
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Copyright (c) Meta Platforms, Inc. and affiliates. See LICENSE-ConvNeXt.txt.

    The stem is Conv4/stride4 -> LN. Each later stage begins with a separate
    LN -> Conv2/stride2 downsampler. Blocks preserve shape within their stage.
    Spatial global average pooling -> LN -> Linear produces raw logits in
    class order [NC, AD]. There is no softmax before CrossEntropyLoss.

    Adaptations: two outputs instead of 1000, native ChannelLayerNorm, and an
    explicit AdaptiveAvgPool2d(1) equivalent to the official spatial mean.
    Conv/Linear weights use trunc_normal_(std=0.02) and zero biases, LN uses
    unit weights and zero biases, and layer scale starts at 1e-6 by default.
    No pretrained weights are fetched. Seed the caller before construction.
    Stochastic-depth probabilities increase linearly from zero to
    drop_path_rate across all 18 blocks (default zero, as in official code).
    """

    depths = (3, 3, 9, 3)
    dims = (96, 192, 384, 768)

    def __init__(self, drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        if not math.isfinite(drop_path_rate) or not 0 <= drop_path_rate <= 1:
            raise ValueError("drop_path_rate must be finite and in [0, 1]")
        if not math.isfinite(layer_scale_init_value) or layer_scale_init_value < 0:
            raise ValueError("layer_scale_init_value must be finite and nonnegative")
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(INPUT_CHANNELS, self.dims[0], kernel_size=4, stride=4),
                          ChannelLayerNorm(self.dims[0]))
        ])
        for previous, current in zip(self.dims[:-1], self.dims[1:]):
            self.downsample_layers.append(nn.Sequential(
                ChannelLayerNorm(previous),
                nn.Conv2d(previous, current, kernel_size=2, stride=2),
            ))
        rates = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for depth, width in zip(self.depths, self.dims):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(width, rates[offset + index], layer_scale_init_value)
                for index in range(depth)
            ]))
            offset += depth
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], NUM_CLASSES)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images):
        features = images
        for downsample, stage in zip(self.downsample_layers, self.stages):
            features = stage(downsample(features))
        return self.norm(self.avgpool(features).flatten(1))

    def forward(self, images):
        return self.head(self.forward_features(images))


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


def build_model(name: str, drop_path_rate: float = 0.0) -> nn.Module:
    if name == "small_cnn":
        if drop_path_rate != 0:
            raise ValueError("Stochastic depth is only supported by convnext_tiny")
        return SmallCNN()
    if name == "convnext_tiny":
        return ConvNeXtTiny(drop_path_rate=drop_path_rate)
    raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
