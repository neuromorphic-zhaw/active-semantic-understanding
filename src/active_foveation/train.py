"""Train the four ADE20K-Object models reported in the paper."""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import IGNORE_INDEX, PAPER_CONFIG
from .data import CropDataset, FullImageDataset, normalize_image
from .metrics import (
    CropMetricAccumulator,
    confusion_matrix,
    object_miou,
    soft_dice_loss,
)
from .models import ContextFoveaModel, LRASPPSingleHead, build_model

PAPER_TRAINING = {
    "baseline": {"epochs": 15, "batch_size": 12},
    "fovea_only": {"epochs": 30, "batch_size": 64},
    "fovea_context": {"epochs": 30, "batch_size": 64},
    "full_object": {"epochs": 30, "batch_size": 64},
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def forward_crop_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device)
    target = batch["mask"].to(device)
    if isinstance(model, ContextFoveaModel):
        context = batch["context"].to(device)
        logits = model(normalize_image(image), normalize_image(context))
    else:
        logits = model(normalize_image(image))
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    return logits, target, batch["class_id"].to(device)


@torch.no_grad()
def evaluate_crops(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    dice_weight: float,
) -> dict[str, float | int]:
    model.eval()
    accumulator = CropMetricAccumulator(device)
    loss_sum = 0.0
    samples = 0
    for batch in tqdm(loader, desc="validate", leave=False):
        with amp_context(device, amp):
            logits, target, class_ids = forward_crop_model(model, batch, device)
            loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
            loss = loss + dice_weight * soft_dice_loss(logits, target)
        accumulator.update(logits.float(), target, class_ids)
        count = target.shape[0]
        loss_sum += float(loss.item()) * count
        samples += count
    return {"loss": loss_sum / max(1, samples), **accumulator.summary()}


@torch.no_grad()
def evaluate_baseline(
    model: LRASPPSingleHead,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    dice_weight: float,
) -> dict[str, float | int]:
    model.eval()
    matrix = torch.zeros(
        150, 150, dtype=torch.int64, device=device
    )
    correct = total = samples = 0
    loss_sum = 0.0
    for batch in tqdm(loader, desc="validate", leave=False):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        with amp_context(device, amp):
            logits = model(normalize_image(image))
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
            loss = loss + dice_weight * soft_dice_loss(logits, target)
        prediction = logits.argmax(1)
        valid = target != IGNORE_INDEX
        correct += int((prediction[valid] == target[valid]).sum().item())
        total += int(valid.sum().item())
        matrix += confusion_matrix(prediction, target)
        count = target.shape[0]
        loss_sum += float(loss.item()) * count
        samples += count
    return {
        "loss": loss_sum / max(1, samples),
        "pixel_accuracy": correct / max(1, total),
        "object_class_miou": object_miou(matrix),
        "images": samples,
        "valid_pixels": total,
    }


def repeat_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    pin_memory = args.device.type == "cuda"
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": pin_memory,
    }
    if args.model == "baseline":
        if not args.ade_root:
            raise ValueError("--ade-root is required for baseline training")
        train_dataset = FullImageDataset(
            args.ade_root,
            args.prepared_root,
            "training",
            PAPER_CONFIG.image_size,
            args.train_limit,
        )
        validation_dataset = FullImageDataset(
            args.ade_root,
            args.prepared_root,
            "validation",
            PAPER_CONFIG.image_size,
            args.validation_limit,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **common)
        validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
        return train_loader, validation_loader, len(train_loader)

    needs_context = args.model in {"fovea_context", "full_object"}
    context_root = args.context_cache if needs_context else None
    training_cache = (
        args.full_object_cache if args.model == "full_object" else args.fovea_cache
    )
    train_dataset = CropDataset(
        training_cache,
        "training",
        context_root,
        args.train_limit,
    )
    validation_dataset = CropDataset(
        args.fovea_cache,
        "validation",
        context_root,
        args.validation_limit,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **common
    )
    if args.model == "full_object":
        matching_count = len(
            CropDataset(args.fovea_cache, "training", limit=args.train_limit)
        )
        steps = max(1, matching_count // args.batch_size)
    else:
        steps = len(train_loader)
    return train_loader, validation_loader, steps


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed, args.deterministic)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    }
    (output / "args.json").write_text(
        json.dumps(serializable_args, indent=2) + "\n"
    )

    train_loader, validation_loader, steps_per_epoch = make_loaders(args)
    model = build_model(
        args.model,
        pretrained=args.pretrained,
        fovea_size=PAPER_CONFIG.fovea_size,
    ).to(args.device)
    if isinstance(model, LRASPPSingleHead):
        model.build_head(args.device, torch.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=0.9)
    use_amp = args.amp and args.device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    best_score = -1.0
    history: list[dict[str, float | int]] = []

    print(
        f"model={args.model} device={args.device} train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        seen = 0
        batches = repeat_loader(train_loader)
        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}"):
            batch = next(batches)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, use_amp):
                if args.model == "baseline":
                    image = batch["image"].to(args.device)
                    target = batch["mask"].to(args.device)
                    logits = model(normalize_image(image))
                else:
                    logits, target, _ = forward_crop_model(
                        model, batch, args.device
                    )
                loss = F.cross_entropy(
                    logits, target, ignore_index=IGNORE_INDEX
                )
                loss = loss + args.dice_weight * soft_dice_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = target.shape[0]
            loss_sum += float(loss.item()) * count
            seen += count

        if args.model == "baseline":
            metrics = evaluate_baseline(
                model, validation_loader, args.device, use_amp, args.dice_weight
            )
            score = float(metrics["object_class_miou"])
        else:
            metrics = evaluate_crops(
                model, validation_loader, args.device, use_amp, args.dice_weight
            )
            score = float(metrics["mean_target_object_iou"]) + 0.25 * float(
                metrics["top1_object_accuracy"]
            )
        row = {
            "epoch": epoch,
            "training_loss": loss_sum / max(1, seen),
            "epoch_seconds": time.time() - started,
            **metrics,
        }
        history.append(row)
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": {
                "model_type": args.model,
                "image_size": PAPER_CONFIG.image_size,
                "fovea_size": PAPER_CONFIG.fovea_size,
                "context_size": PAPER_CONFIG.context_size,
                "seed": args.seed,
            },
        }
        torch.save(payload, output / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(payload, output / "best.pt")
            (output / "metrics_best.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
        (output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(PAPER_TRAINING),
        required=True,
        help="paper component to train",
    )
    parser.add_argument(
        "--ade-root",
        default="",
        help="ADEChallengeData2016 directory (required by baseline)",
    )
    parser.add_argument("--prepared-root", default="data/ade20k_object")
    parser.add_argument("--fovea-cache", default="")
    parser.add_argument("--context-cache", default="")
    parser.add_argument("--full-object-cache", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    defaults = PAPER_TRAINING[args.model]
    args.epochs = args.epochs or defaults["epochs"]
    args.batch_size = args.batch_size or defaults["batch_size"]
    prepared = Path(args.prepared_root)
    args.fovea_cache = args.fovea_cache or str(prepared / "fovea96_n3")
    args.context_cache = args.context_cache or str(prepared / "context128")
    args.full_object_cache = args.full_object_cache or str(
        prepared / "full_object96"
    )
    args.output = args.output or str(Path("outputs") / args.model)
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
