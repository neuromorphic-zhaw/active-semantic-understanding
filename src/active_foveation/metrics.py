"""Metrics used in the ADE20K-Object paper tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import IGNORE_INDEX, N_CLASSES, OBJECT_CLASS_IDS


@torch.no_grad()
def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = N_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    prediction, target = prediction[valid], target[valid]
    if prediction.numel() == 0:
        return torch.zeros(
            num_classes, num_classes, dtype=torch.int64, device=prediction.device
        )
    indices = (target * num_classes + prediction).long()
    return torch.bincount(
        indices, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def per_class_iou(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.float()
    true_positive = torch.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1) - true_positive
    return torch.where(
        denominator > 0,
        true_positive / denominator,
        torch.full_like(denominator, torch.nan),
    )


def object_miou(
    matrix: torch.Tensor, class_ids: tuple[int, ...] = OBJECT_CLASS_IDS
) -> float:
    values = per_class_iou(matrix)[list(class_ids)]
    valid = ~torch.isnan(values)
    return float(values[valid].mean().item()) if bool(valid.any()) else 0.0


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_ids: tuple[int, ...] = OBJECT_CLASS_IDS,
) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    probabilities = torch.softmax(logits, dim=1)
    valid_float = valid.float()
    losses = []
    for class_id in class_ids:
        ground_truth = ((target == class_id) & valid).float()
        if not bool(ground_truth.any()):
            continue
        prediction = probabilities[:, class_id] * valid_float
        intersection = (prediction * ground_truth).sum()
        losses.append(
            1.0
            - (2.0 * intersection + 1.0)
            / (prediction.sum() + ground_truth.sum() + 1.0)
        )
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


@dataclass
class CropMetricAccumulator:
    device: torch.device
    matrix: torch.Tensor = field(init=False)
    pixel_correct: int = 0
    pixel_total: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    object_total: int = 0
    target_iou_sum: float = 0.0

    def __post_init__(self) -> None:
        self.matrix = torch.zeros(
            N_CLASSES, N_CLASSES, dtype=torch.int64, device=self.device
        )

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> None:
        prediction = logits.argmax(1)
        valid = target != IGNORE_INDEX
        self.pixel_correct += int((prediction[valid] == target[valid]).sum().item())
        self.pixel_total += int(valid.sum().item())
        self.matrix += confusion_matrix(prediction, target)
        for index in range(target.shape[0]):
            class_id = int(class_ids[index])
            object_pixels = target[index] == class_id
            if not bool(object_pixels.any()):
                continue
            scores = logits[index, :, object_pixels].mean(1)
            top3 = scores.topk(3).indices
            self.top1_correct += int(int(top3[0]) == class_id)
            self.top3_correct += int(bool((top3 == class_id).any()))
            self.object_total += 1
            predicted_object = prediction[index] == class_id
            intersection = int((predicted_object & object_pixels).sum().item())
            union = int((predicted_object | object_pixels).sum().item())
            self.target_iou_sum += intersection / max(1, union)

    def summary(self) -> dict[str, float | int]:
        return {
            "object_class_miou": object_miou(self.matrix),
            "pixel_accuracy": self.pixel_correct / max(1, self.pixel_total),
            "top1_object_accuracy": self.top1_correct / max(1, self.object_total),
            "top3_object_accuracy": self.top3_correct / max(1, self.object_total),
            "mean_target_object_iou": self.target_iou_sum
            / max(1, self.object_total),
            "samples": self.object_total,
            "valid_pixels": self.pixel_total,
        }


def discovery_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_fraction: float,
    class_ids: tuple[int, ...] = OBJECT_CLASS_IDS,
) -> tuple[int, int]:
    found = 0
    total = 0
    for class_id in class_ids:
        ground_truth = target == class_id
        count = int(ground_truth.sum().item())
        if count == 0:
            continue
        total += 1
        hits = int(((prediction == class_id) & ground_truth).sum().item())
        found += int(hits / count >= min_fraction)
    return found, total
