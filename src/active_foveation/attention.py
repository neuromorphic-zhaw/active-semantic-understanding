"""Itti-Koch saliency, winner-take-all selection, and inhibition of return.

The saliency operator is a cleaned PyTorch adaptation of pySaliencyMap. It uses
intensity, red-green/blue-yellow color opponency, four Gabor orientations, and
center-surround differences over a Gaussian pyramid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GABOR_0 = (
    (1.85212e-06, 1.28181e-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181e-05, 1.85212e-06),
    (2.80209e-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209e-05),
    (0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076),
    (0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940),
    (0.000921261, 0.006375831, -0.174308068, -0.067914552, 1.0, -0.067914552, -0.174308068, 0.006375831, 0.000921261),
    (0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940),
    (0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076),
    (2.80209e-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209e-05),
    (1.85212e-06, 1.28181e-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181e-05, 1.85212e-06),
)

GABOR_45 = (
    (4.04180e-06, 2.25320e-05, -0.000279806, -0.001028923, 3.79931e-05, 0.000744712, 0.000132863, -9.04408e-06, -1.01551e-06),
    (2.25320e-05, 0.000925120, 0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408e-06),
    (-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863),
    (-0.001028923, -0.013561362, 0.052928748, 0.460162150, 0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712),
    (3.79931e-05, -0.022947700, -0.139178011, 0.249959607, 1.0, 0.249959607, -0.139178011, -0.022947700, 3.79931e-05),
    (0.000744712, 0.003899160, -0.108372072, -0.302454279, 0.249959607, 0.460162150, 0.052928748, -0.013561362, -0.001028923),
    (0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806),
    (-9.04408e-06, 0.000288732, 0.003516954, 0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.000925120, 2.25320e-05),
    (-1.01551e-06, -9.04408e-06, 0.000132863, 0.000744712, 3.79931e-05, -0.001028923, -0.000279806, 2.25320e-05, 4.04180e-06),
)

GABOR_90 = (
    (1.85212e-06, 2.80209e-05, 0.000195076, 0.000624940, 0.000921261, 0.000624940, 0.000195076, 2.80209e-05, 1.85212e-06),
    (1.28181e-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181e-05),
    (-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433),
    (-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537),
    (0.002010422, 0.030415784, 0.211749204, 0.678352526, 1.0, 0.678352526, 0.211749204, 0.030415784, 0.002010422),
    (-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537),
    (-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433),
    (1.28181e-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181e-05),
    (1.85212e-06, 2.80209e-05, 0.000195076, 0.000624940, 0.000921261, 0.000624940, 0.000195076, 2.80209e-05, 1.85212e-06),
)

GABOR_135 = (
    (-1.01551e-06, -9.04408e-06, 0.000132863, 0.000744712, 3.79931e-05, -0.001028923, -0.000279806, 2.25320e-05, 4.04180e-06),
    (-9.04408e-06, 0.000288732, 0.003516954, 0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.000925120, 2.25320e-05),
    (0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806),
    (0.000744712, 0.000389916, -0.108372072, -0.302454279, 0.249959607, 0.460162150, 0.052928748, -0.013561362, -0.001028923),
    (3.79931e-05, -0.022947700, -0.139178011, 0.249959607, 1.0, 0.249959607, -0.139178011, -0.022947700, 3.79931e-05),
    (-0.001028923, -0.013561362, 0.052928748, 0.460162150, 0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712),
    (-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863),
    (2.25320e-05, 0.000925120, 0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408e-06),
    (4.04180e-06, 2.25320e-05, -0.000279806, -0.001028923, 3.79931e-05, 0.000744712, 0.000132863, -9.04408e-06, -1.01551e-06),
)


class IttiKochSaliency(nn.Module):
    """Static Itti-Koch saliency without motion or post-filtering."""

    def __init__(self, output_size: tuple[int, int], levels: int = 3):
        super().__init__()
        if levels < 2:
            raise ValueError("Itti-Koch saliency needs at least two pyramid levels")
        self.output_size = output_size
        self.levels = levels
        gaussian = torch.tensor(
            (
                (1, 4, 6, 4, 1),
                (4, 16, 24, 16, 4),
                (6, 24, 36, 24, 6),
                (4, 16, 24, 16, 4),
                (1, 4, 6, 4, 1),
            ),
            dtype=torch.float32,
        ) / 256.0
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, 5, 5))
        gabors = torch.tensor(
            (GABOR_0, GABOR_45, GABOR_90, GABOR_135),
            dtype=torch.float32,
        )
        self.register_buffer("gabor_kernels", gabors.unsqueeze(1))

    def gaussian_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        pyramid = [image]
        current = image
        for _ in range(1, self.levels):
            current = F.pad(current, (2, 2, 2, 2), mode="replicate")
            current = F.conv2d(current, self.gaussian_kernel, stride=2)
            pyramid.append(current)
        return pyramid

    @staticmethod
    def center_surround(pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        maps = []
        for center_index, center in enumerate(pyramid):
            for surround in pyramid[center_index + 1 :]:
                resized = F.interpolate(
                    surround,
                    size=center.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                maps.append((center - resized).abs())
        return maps

    @staticmethod
    def range_normalize(image: torch.Tensor) -> torch.Tensor:
        minimum, maximum = image.min(), image.max()
        span = maximum - minimum
        return (image - minimum) / span if bool(span > 0) else image - minimum

    @staticmethod
    def average_local_maximum(image: torch.Tensor, step: int = 16) -> float:
        image = image.squeeze()
        height, width = image.shape
        maxima = []
        for y in range(0, height - step, step):
            for x in range(0, width - step, step):
                maxima.append(image[y : y + step, x : x + step].max())
        if not maxima:
            return 0.0
        return float(torch.stack(maxima).mean().item())

    def normalize_feature(self, image: torch.Tensor) -> torch.Tensor:
        normalized = self.range_normalize(image)
        local_mean = self.average_local_maximum(normalized)
        return normalized * (1.0 - local_mean) ** 2

    def conspicuity(self, maps: list[torch.Tensor]) -> torch.Tensor:
        normalized = []
        for feature in maps:
            feature = self.normalize_feature(feature)
            normalized.append(
                F.interpolate(
                    feature,
                    size=self.output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.stack(normalized).sum(0)

    def one_image(self, rgb: torch.Tensor) -> torch.Tensor:
        red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        intensity = 0.299 * red + 0.587 * green + 0.114 * blue

        intensity_maps = self.center_surround(
            self.gaussian_pyramid(intensity)
        )
        maximum = torch.maximum(torch.maximum(red, green), blue).clamp_min(1e-4)
        red_green = ((red - green) / maximum).clamp_min(0.0)
        blue_yellow = ((blue - torch.minimum(red, green)) / maximum).clamp_min(0.0)
        color_maps_rg = self.center_surround(
            self.gaussian_pyramid(red_green)
        )
        color_maps_by = self.center_surround(
            self.gaussian_pyramid(blue_yellow)
        )

        orientation_groups: list[list[torch.Tensor]] = []
        intensity_pyramid = self.gaussian_pyramid(intensity)
        for kernel in self.gabor_kernels:
            filtered = []
            for level in intensity_pyramid:
                filtered.append(
                    F.conv2d(
                        F.pad(level, (4, 4, 4, 4), mode="replicate"),
                        kernel.unsqueeze(0),
                    )
                )
            orientation_groups.append(self.center_surround(filtered))

        intensity_conspicuity = self.conspicuity(intensity_maps)
        color_conspicuity = self.conspicuity(
            color_maps_rg
        ) + self.conspicuity(color_maps_by)
        orientation_conspicuity = torch.zeros_like(intensity_conspicuity)
        for group in orientation_groups:
            orientation_conspicuity += self.normalize_feature(
                self.conspicuity(group)
            )
        combined = (
            0.30 * intensity_conspicuity
            + 0.30 * color_conspicuity
            + 0.20 * orientation_conspicuity
        )
        return self.range_normalize(combined)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected RGB input with shape Bx3xHxW")
        image = rgb.float()
        if bool(image.max() > 1.0):
            image = image / 255.0
        return torch.cat(
            [self.one_image(image[index : index + 1]) for index in range(len(image))],
            dim=0,
        )


class WinnerTakeAllIOR:
    """Deterministic peak selection with multiplicative Gaussian IoR."""

    def __init__(
        self,
        saliency: torch.Tensor,
        *,
        margin: int,
        radius: float,
    ):
        if saliency.ndim != 2:
            raise ValueError("saliency must be a 2D tensor")
        self.saliency = saliency.float()
        self.ior = torch.ones_like(self.saliency)
        self.margin = margin
        self.radius = float(radius)
        height, width = saliency.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=saliency.device),
            torch.arange(width, device=saliency.device),
            indexing="ij",
        )
        self.xx, self.yy = xx.float(), yy.float()

    def next(self) -> tuple[int, int]:
        score = (self.saliency * self.ior).clone()
        if self.margin:
            score[: self.margin] = -torch.inf
            score[-self.margin :] = -torch.inf
            score[:, : self.margin] = -torch.inf
            score[:, -self.margin :] = -torch.inf
        index = int(score.argmax().item())
        width = score.shape[1]
        y, x = divmod(index, width)
        distance_sq = (self.xx - x) ** 2 + (self.yy - y) ** 2
        suppression = 1.0 - torch.exp(
            -distance_sq / (2.0 * self.radius * self.radius)
        )
        self.ior *= suppression
        return x, y

    def take(self, count: int) -> list[tuple[int, int]]:
        return [self.next() for _ in range(count)]
