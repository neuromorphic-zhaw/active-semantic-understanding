"""Lite R-ASPP models used by all ADE20K-Object experiments."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

from .config import N_CLASSES, PAPER_CONFIG


class LRASPPHead(nn.Module):
    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        intermediate_channels: int = 128,
    ):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, intermediate_channels, 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, intermediate_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(intermediate_channels, num_classes, 1)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        low, high = features["low"], features["high"]
        output = self.cbr(high) * self.scale(high)
        output = F.interpolate(
            output, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.low_classifier(low) + self.high_classifier(output)


def _find_first_conv(
    module: nn.Module,
) -> tuple[nn.Module, str, nn.Conv2d] | None:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        result = _find_first_conv(child)
        if result is not None:
            return result
    return None


def _replace_first_conv(backbone: nn.Module, input_channels: int) -> nn.Conv2d:
    found = _find_first_conv(backbone)
    if found is None:
        raise RuntimeError("Could not locate MobileNetV3's first convolution")
    parent, name, convolution = found
    if convolution.in_channels == input_channels:
        return convolution
    replacement = nn.Conv2d(
        input_channels,
        convolution.out_channels,
        kernel_size=convolution.kernel_size,
        stride=convolution.stride,
        padding=convolution.padding,
        dilation=convolution.dilation,
        groups=convolution.groups,
        bias=False,
        padding_mode=convolution.padding_mode,
    )
    with torch.no_grad():
        weights = convolution.weight
        original_channels = weights.shape[1]
        if input_channels <= original_channels:
            adapted = weights[:, :input_channels].clone()
        else:
            repetitions, remainder = divmod(input_channels, original_channels)
            pieces = [weights.repeat(1, repetitions, 1, 1)]
            if remainder:
                pieces.append(weights[:, :remainder].clone())
            adapted = torch.cat(pieces, dim=1)
            adapted *= (original_channels / float(input_channels)) ** 0.5
        replacement.weight.copy_(adapted)
    if isinstance(parent, nn.Sequential) and name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)
    return replacement


def _infer_channels(
    backbone: nn.Module,
    input_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, int]:
    with torch.no_grad():
        sample = torch.zeros(
            1, input_channels, 128, 128, device=device, dtype=dtype
        )
        features = backbone(sample)
    return features["low"].shape[1], features["high"].shape[1]


def _copy_pretrained_head(destination: LRASPPHead, source: nn.Module) -> None:
    with torch.no_grad():
        destination.cbr[0].weight.copy_(source.cbr[0].weight)
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(destination.cbr[1], name).copy_(getattr(source.cbr[1], name))
        destination.scale[1].weight.copy_(source.scale[1].weight)

    def adapt_classifier(target: nn.Conv2d, original: nn.Conv2d) -> None:
        with torch.no_grad():
            count = target.weight.shape[0]
            if count <= original.weight.shape[0]:
                values = original.weight[:count].clone()
            else:
                repetitions, remainder = divmod(count, original.weight.shape[0])
                pieces = [original.weight.repeat(repetitions, 1, 1, 1)]
                if remainder:
                    pieces.append(original.weight[:remainder].clone())
                values = torch.cat(pieces, dim=0)
            target.weight.copy_(values)

    adapt_classifier(destination.low_classifier, source.low_classifier)
    adapt_classifier(destination.high_classifier, source.high_classifier)


class LRASPPSingleHead(nn.Module):
    """MobileNetV3-Large backbone with a 150-class Lite R-ASPP decoder."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = N_CLASSES,
        *,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        intermediate_channels: int = 128,
    ):
        super().__init__()
        if pretrained:
            base = lraspp_mobilenet_v3_large(
                weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
                num_classes=21,
            )
        else:
            base = lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=21,
            )
        self.backbone = base.backbone
        self._pretrained_head = base.classifier
        self._first_conv: nn.Conv2d | None = None
        if input_channels != 3:
            self._first_conv = _replace_first_conv(
                self.backbone, input_channels
            )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if self._first_conv is not None:
                for parameter in self._first_conv.parameters():
                    parameter.requires_grad = True
        self._input_channels = input_channels
        self._num_classes = num_classes
        self._intermediate_channels = intermediate_channels
        self._head: LRASPPHead | None = None
        self._head_built = False

    def build_head(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._head_built:
            return
        low_channels, high_channels = _infer_channels(
            self.backbone, self._input_channels, device, dtype
        )
        self._head = LRASPPHead(
            low_channels,
            high_channels,
            self._num_classes,
            self._intermediate_channels,
        ).to(device=device, dtype=dtype)
        _copy_pretrained_head(self._head, self._pretrained_head)
        self._head_built = True

    def encode(self, image: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        self.build_head(image.device, image.dtype)
        return self.backbone(image)

    def decode(
        self, features: OrderedDict[str, torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        if self._head is None:
            raise RuntimeError("Decoder head has not been initialized")
        logits = self._head(features)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image), image.shape[-2:])


class ContextFoveaModel(nn.Module):
    """Foveal Lite R-ASPP conditioned by a compact global RGB embedding."""

    def __init__(
        self,
        fovea_size: int = PAPER_CONFIG.fovea_size,
        context_embedding_size: int = 128,
        *,
        pretrained: bool = True,
    ):
        super().__init__()
        self.base = LRASPPSingleHead(pretrained=pretrained)
        with torch.no_grad():
            sample = torch.zeros(1, 3, fovea_size, fovea_size)
            self.base.build_head(sample.device, sample.dtype)
        low_channels, high_channels = _infer_channels(
            self.base.backbone, 3, torch.device("cpu"), torch.float32
        )
        self.context_net = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, context_embedding_size),
            nn.ReLU(inplace=True),
        )
        self.low_bias = nn.Linear(context_embedding_size, low_channels)
        self.high_bias = nn.Linear(context_embedding_size, high_channels)

    def forward(self, fovea: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self.base.encode(fovea)
        embedding = self.context_net(context)
        conditioned = OrderedDict(
            low=features["low"]
            + self.low_bias(embedding).unsqueeze(-1).unsqueeze(-1),
            high=features["high"]
            + self.high_bias(embedding).unsqueeze(-1).unsqueeze(-1),
        )
        return self.base.decode(conditioned, fovea.shape[-2:])


def build_model(
    model_type: str,
    *,
    pretrained: bool,
    fovea_size: int = PAPER_CONFIG.fovea_size,
) -> nn.Module:
    if model_type in {"baseline", "fovea_only"}:
        return LRASPPSingleHead(pretrained=pretrained)
    if model_type in {"fovea_context", "full_object"}:
        return ContextFoveaModel(fovea_size=fovea_size, pretrained=pretrained)
    raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(
    model_type: str,
    checkpoint: str | Path,
    device: torch.device,
    *,
    fovea_size: int = PAPER_CONFIG.fovea_size,
) -> nn.Module:
    model = build_model(model_type, pretrained=False, fovea_size=fovea_size)
    model.to(device)
    if isinstance(model, LRASPPSingleHead):
        model.build_head(device, torch.float32)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model
