"""Deterministic Gaussian-weighted semantic canvas."""

from __future__ import annotations

import torch


def gaussian_write_mask(
    size: int,
    *,
    sigma: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    sigma = sigma if sigma is not None else size / 4.0
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    center = (size - 1) / 2.0
    return torch.exp(
        -((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma * sigma)
    ).float()


class SemanticCanvas:
    def __init__(
        self,
        num_classes: int,
        height: int,
        width: int,
        device: torch.device,
    ):
        self.logit_sum = torch.zeros(
            num_classes, height, width, device=device
        )
        self.weight_sum = torch.zeros(1, height, width, device=device)

    def write(
        self,
        logits: torch.Tensor,
        center_x: int,
        center_y: int,
        weight: torch.Tensor,
    ) -> None:
        _, height, width = self.logit_sum.shape
        crop_height, crop_width = logits.shape[-2:]
        x0, y0 = center_x - crop_width // 2, center_y - crop_height // 2
        x1, y1 = x0 + crop_width, y0 + crop_height
        source_x0, source_y0 = max(0, x0), max(0, y0)
        source_x1, source_y1 = min(width, x1), min(height, y1)
        if source_x1 <= source_x0 or source_y1 <= source_y0:
            return
        patch_x0, patch_y0 = source_x0 - x0, source_y0 - y0
        patch_x1 = patch_x0 + source_x1 - source_x0
        patch_y1 = patch_y0 + source_y1 - source_y0
        local_weight = weight[patch_y0:patch_y1, patch_x0:patch_x1]
        self.logit_sum[:, source_y0:source_y1, source_x0:source_x1] += (
            logits[:, patch_y0:patch_y1, patch_x0:patch_x1]
            * local_weight.unsqueeze(0)
        )
        self.weight_sum[:, source_y0:source_y1, source_x0:source_x1] += (
            local_weight.unsqueeze(0)
        )

    @property
    def logits(self) -> torch.Tensor:
        return self.logit_sum / self.weight_sum.clamp_min(1e-6)

    @property
    def coverage(self) -> torch.Tensor:
        return self.weight_sum[0] > 1e-4
