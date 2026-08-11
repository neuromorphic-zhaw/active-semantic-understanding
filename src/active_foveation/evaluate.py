"""Accuracy-only evaluation for the ADE20K-Object paper experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .attention import IttiKochSaliency, WinnerTakeAllIOR
from .canvas import SemanticCanvas, gaussian_write_mask
from .config import (
    IGNORE_INDEX,
    N_CLASSES,
    OBJECT_CLASS_IDS,
    PAPER_CONFIG,
)
from .data import (
    BILINEAR,
    NEAREST,
    crop_pad_tensor,
    crop_with_pad,
    image_to_tensor,
    manifest_paths,
    normalize_image,
    read_manifest,
    remap_object_mask,
)
from .metrics import CropMetricAccumulator, confusion_matrix, object_miou
from .models import ContextFoveaModel, load_checkpoint
from .train import amp_context, resolve_device

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints"


def load_crop_batch(
    files: list[Path],
    context: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, targets, class_ids, xs, ys = [], [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            images.append(
                torch.from_numpy(sample["image"].copy())
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            targets.append(
                torch.from_numpy(sample["mask"].astype(np.int64))
            )
            class_ids.append(int(sample["class_id"]))
            xs.append(int(sample["x"]))
            ys.append(int(sample["y"]))
    batch_size = len(files)
    contexts = context.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return (
        torch.stack(images).to(device),
        contexts.to(device),
        torch.stack(targets).to(device),
        torch.tensor(class_ids, device=device),
        torch.tensor(list(zip(xs, ys)), device=device),
    )


def context_for_image(
    image_id: str,
    image: Image.Image,
    context_cache: str | Path | None,
    split: str,
) -> torch.Tensor:
    if context_cache:
        path = Path(context_cache) / split / f"{image_id}.npz"
        with np.load(path, allow_pickle=False) as payload:
            context = payload["context"].copy()
        return (
            torch.from_numpy(context).permute(2, 0, 1).float() / 255.0
        )
    resized = image.resize(
        (PAPER_CONFIG.context_size, PAPER_CONFIG.context_size), BILINEAR
    )
    return image_to_tensor(resized)


@torch.no_grad()
def evaluate_single_fixation(args: argparse.Namespace) -> dict:
    device = args.device
    use_amp = args.amp and device.type == "cuda"
    models = {
        "fovea_only": load_checkpoint(
            "fovea_only", args.fovea_checkpoint, device
        ),
        "fovea_context": load_checkpoint(
            "fovea_context", args.context_checkpoint, device
        ),
        "full_object": load_checkpoint(
            "full_object", args.full_object_checkpoint, device
        ),
    }
    baseline = load_checkpoint(
        "baseline", args.baseline_checkpoint, device
    )
    accumulators = {
        name: CropMetricAccumulator(device)
        for name in ("baseline", *models.keys())
    }

    rows = read_manifest(args.prepared_root, "validation")
    if args.limit_images:
        rows = rows[: args.limit_images]
    rows_by_id = {row["image_id"]: row for row in rows}
    files_by_image: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((Path(args.fovea_cache) / "validation").glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            image_id = str(sample["image_id"])
        if image_id in rows_by_id:
            files_by_image[image_id].append(path)

    for image_id, files in tqdm(files_by_image.items(), desc="single fixation"):
        row = rows_by_id[image_id]
        image_path, _ = manifest_paths(args.ade_root, row)
        image = Image.open(image_path).convert("RGB").resize(
            (PAPER_CONFIG.image_size, PAPER_CONFIG.image_size), BILINEAR
        )
        full_image = image_to_tensor(image).unsqueeze(0).to(device)
        with amp_context(device, use_amp):
            full_logits = baseline(normalize_image(full_image))[0].float()
        context = context_for_image(
            image_id, image, args.context_cache, "validation"
        )

        for start in range(0, len(files), args.batch_size):
            batch_files = files[start : start + args.batch_size]
            crops, contexts, targets, class_ids, centers = load_crop_batch(
                batch_files, context, device
            )
            baseline_crops = torch.stack(
                [
                    crop_pad_tensor(
                        full_logits,
                        int(center[0]),
                        int(center[1]),
                        PAPER_CONFIG.fovea_size,
                    )
                    for center in centers
                ]
            )
            accumulators["baseline"].update(
                baseline_crops, targets, class_ids
            )
            with amp_context(device, use_amp):
                fovea_logits = models["fovea_only"](
                    normalize_image(crops)
                )
                context_logits = models["fovea_context"](
                    normalize_image(crops), normalize_image(contexts)
                )
                object_logits = models["full_object"](
                    normalize_image(crops), normalize_image(contexts)
                )
            accumulators["fovea_only"].update(
                fovea_logits.float(), targets, class_ids
            )
            accumulators["fovea_context"].update(
                context_logits.float(), targets, class_ids
            )
            accumulators["full_object"].update(
                object_logits.float(), targets, class_ids
            )

    results = {
        "protocol": {
            "dataset": "ADE20K-Object validation",
            "fovea_size": PAPER_CONFIG.fovea_size,
            "images": len(files_by_image),
            "metric_note": (
                "Paper Table 1 uses object_class_miou; the ablation table uses "
                "mean_target_object_iou."
            ),
        },
        "models": {
            name: accumulator.summary()
            for name, accumulator in accumulators.items()
        },
    }
    write_results(args.output, results)
    return results


def compute_saliency(
    image: Image.Image,
    model: IttiKochSaliency,
    device: torch.device,
) -> torch.Tensor:
    preview = image.resize(
        (PAPER_CONFIG.saliency_size, PAPER_CONFIG.saliency_size), BILINEAR
    )
    tensor = image_to_tensor(preview).unsqueeze(0).to(device)
    saliency = model(tensor)
    return F.interpolate(
        saliency,
        size=(PAPER_CONFIG.image_size, PAPER_CONFIG.image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


@dataclass
class SceneMetricAccumulator:
    device: torch.device
    matrix_full: torch.Tensor = field(init=False)
    matrix_observed: torch.Tensor = field(init=False)
    full_correct: int = 0
    full_total: int = 0
    observed_correct: int = 0
    observed_total: int = 0
    coverage_sum: float = 0.0
    images: int = 0
    objects_present: int = 0
    objects_observed: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    objects_found: int = 0

    def __post_init__(self) -> None:
        self.matrix_full = torch.zeros(
            N_CLASSES, N_CLASSES, dtype=torch.int64, device=self.device
        )
        self.matrix_observed = torch.zeros_like(self.matrix_full)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        min_discovery_fraction: float,
    ) -> None:
        prediction = logits.argmax(0)
        valid = target != IGNORE_INDEX
        observed_valid = valid & observed
        self.matrix_full += confusion_matrix(
            prediction.unsqueeze(0), target.unsqueeze(0)
        )
        if bool(observed_valid.any()):
            observed_target = target.clone()
            observed_target[~observed_valid] = IGNORE_INDEX
            self.matrix_observed += confusion_matrix(
                prediction.unsqueeze(0), observed_target.unsqueeze(0)
            )
            self.observed_correct += int(
                (prediction[observed_valid] == target[observed_valid]).sum().item()
            )
            self.observed_total += int(observed_valid.sum().item())
        self.full_correct += int(
            (prediction[valid] == target[valid]).sum().item()
        )
        self.full_total += int(valid.sum().item())
        self.coverage_sum += int(observed_valid.sum().item()) / max(
            1, int(valid.sum().item())
        )
        self.images += 1

        for class_id in OBJECT_CLASS_IDS:
            ground_truth = target == class_id
            object_pixels = int(ground_truth.sum().item())
            if object_pixels == 0:
                continue
            self.objects_present += 1
            visible = ground_truth & observed
            if bool(visible.any()):
                self.objects_observed += 1
                scores = logits[:, visible].mean(1)
                top3 = scores.topk(3).indices
                self.top1_correct += int(int(top3[0]) == class_id)
                self.top3_correct += int(bool((top3 == class_id).any()))
            hits = int(
                ((prediction == class_id) & ground_truth).sum().item()
            )
            self.objects_found += int(
                hits / object_pixels >= min_discovery_fraction
            )

    def summary(self, fixations: int | str) -> dict[str, float | int | str]:
        return {
            "fixations": fixations,
            "images": self.images,
            "full_object_miou": object_miou(self.matrix_full),
            "observed_object_miou": object_miou(self.matrix_observed),
            "full_pixel_accuracy": self.full_correct / max(1, self.full_total),
            "observed_pixel_accuracy": self.observed_correct
            / max(1, self.observed_total),
            "object_pixel_coverage": self.coverage_sum / max(1, self.images),
            "object_coverage": self.objects_observed
            / max(1, self.objects_present),
            "object_recall": self.objects_found / max(1, self.objects_present),
            "top1_when_observed": self.top1_correct
            / max(1, self.objects_observed),
            "top3_when_observed": self.top3_correct
            / max(1, self.objects_observed),
            "objects_found": self.objects_found,
            "objects_present": self.objects_present,
        }


@torch.no_grad()
def evaluate_canvas(args: argparse.Namespace) -> dict:
    device = args.device
    use_amp = args.amp and device.type == "cuda"
    fovea_model = load_checkpoint(
        "fovea_context", args.context_checkpoint, device
    )
    baseline = load_checkpoint(
        "baseline", args.baseline_checkpoint, device
    )
    budgets = tuple(sorted(set(args.budgets)))
    accumulators = {
        budget: SceneMetricAccumulator(device) for budget in budgets
    }
    baseline_accumulator = SceneMetricAccumulator(device)
    saliency_model = IttiKochSaliency(
        (PAPER_CONFIG.saliency_size, PAPER_CONFIG.saliency_size),
        PAPER_CONFIG.saliency_levels,
    ).to(device).eval()
    write_mask = gaussian_write_mask(
        PAPER_CONFIG.fovea_size, device=device
    )

    rows = read_manifest(args.prepared_root, "validation")
    if args.limit_images:
        rows = rows[: args.limit_images]
    for row in tqdm(rows, desc="canvas (Itti-Koch)"):
        image_path, annotation_path = manifest_paths(args.ade_root, row)
        image = Image.open(image_path).convert("RGB").resize(
            (PAPER_CONFIG.image_size, PAPER_CONFIG.image_size), BILINEAR
        )
        raw_mask = Image.open(annotation_path).resize(
            (PAPER_CONFIG.image_size, PAPER_CONFIG.image_size), NEAREST
        )
        target = torch.from_numpy(
            remap_object_mask(raw_mask).astype(np.int64)
        ).to(device)
        image_tensor = image_to_tensor(image).unsqueeze(0).to(device)
        with amp_context(device, use_amp):
            baseline_logits = baseline(normalize_image(image_tensor))[0].float()
        baseline_accumulator.update(
            baseline_logits,
            target,
            torch.ones_like(target, dtype=torch.bool),
            args.min_discovery_fraction,
        )

        context = context_for_image(
            row["image_id"], image, args.context_cache, "validation"
        ).unsqueeze(0).to(device)
        saliency = compute_saliency(image, saliency_model, device)
        selector = WinnerTakeAllIOR(
            saliency,
            margin=PAPER_CONFIG.fovea_size // 2,
            radius=PAPER_CONFIG.ior_radius,
        )
        canvas = SemanticCanvas(
            N_CLASSES,
            PAPER_CONFIG.image_size,
            PAPER_CONFIG.image_size,
            device,
        )
        for fixation in range(1, max(budgets) + 1):
            center_x, center_y = selector.next()
            crop = crop_with_pad(
                image,
                center_x,
                center_y,
                PAPER_CONFIG.fovea_size,
                (0, 0, 0),
            )
            crop_tensor = image_to_tensor(crop).unsqueeze(0).to(device)
            with amp_context(device, use_amp):
                logits = fovea_model(
                    normalize_image(crop_tensor), normalize_image(context)
                )[0].float()
            canvas.write(logits, center_x, center_y, write_mask)
            if fixation in accumulators:
                accumulators[fixation].update(
                    canvas.logits,
                    target,
                    canvas.coverage,
                    args.min_discovery_fraction,
                )

    results = {
        "protocol": {
            "dataset": "ADE20K-Object validation",
            "saliency": "itti_koch",
            "saliency_preview_size": PAPER_CONFIG.saliency_size,
            "saliency_pyramid_levels": PAPER_CONFIG.saliency_levels,
            "fovea_size": PAPER_CONFIG.fovea_size,
            "ior_radius": PAPER_CONFIG.ior_radius,
            "gaussian_write_sigma": PAPER_CONFIG.fovea_size / 4,
            "min_discovery_fraction": args.min_discovery_fraction,
        },
        "results": [
            accumulators[budget].summary(budget) for budget in budgets
        ]
        + [baseline_accumulator.summary("full_image")],
    }
    write_results(args.output, results)
    return results


def write_results(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {destination}", flush=True)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ade-root", required=True)
    parser.add_argument("--prepared-root", default="data/ade20k_object")
    parser.add_argument("--fovea-cache", default="")
    parser.add_argument("--context-cache", default="")
    parser.add_argument(
        "--baseline-checkpoint",
        default=str(CHECKPOINT_ROOT / "baseline.pt"),
    )
    parser.add_argument(
        "--context-checkpoint",
        default=str(CHECKPOINT_ROOT / "fovea_context.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit-images", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser(
        "single", help="single-fixation and training-ablation metrics"
    )
    add_shared_arguments(single)
    single.add_argument(
        "--fovea-checkpoint",
        default=str(CHECKPOINT_ROOT / "fovea_only.pt"),
    )
    single.add_argument(
        "--full-object-checkpoint",
        default=str(CHECKPOINT_ROOT / "full_object.pt"),
    )
    single.add_argument("--batch-size", type=int, default=64)
    single.add_argument(
        "--output", default="results/single_fixation.json"
    )

    canvas = commands.add_parser(
        "canvas", help="fixed-budget semantic-canvas metrics"
    )
    add_shared_arguments(canvas)
    canvas.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(PAPER_CONFIG.fixation_budgets),
    )
    canvas.add_argument(
        "--min-discovery-fraction",
        type=float,
        default=PAPER_CONFIG.min_discovery_fraction,
    )
    canvas.add_argument("--output", default="results/fixed_canvas.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepared = Path(args.prepared_root)
    args.fovea_cache = args.fovea_cache or str(prepared / "fovea96_n3")
    args.context_cache = args.context_cache or str(prepared / "context128")
    args.device = resolve_device(args.device)
    if args.command == "single":
        evaluate_single_fixation(args)
    elif args.command == "canvas":
        evaluate_canvas(args)


if __name__ == "__main__":
    main()
