"""Portable ADE20K-Object datasets and preprocessing primitives."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IGNORE_INDEX, OBJECT_CLASS_IDS

BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def resolve_ade_root(path: str | Path) -> Path:
    """Return the directory that directly contains ADE20K images/annotations."""
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "ADEChallengeData2016")
    for candidate in candidates:
        if (candidate / "images" / "training").is_dir() and (
            candidate / "annotations" / "training"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        f"{root} does not contain ADEChallengeData2016 images and annotations"
    )


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image - mean) / std


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def remap_object_mask(
    raw_mask: Image.Image | np.ndarray,
    keep_ids: Iterable[int] = OBJECT_CLASS_IDS,
) -> np.ndarray:
    """Map raw ADE labels (0..150) to zero-based IDs and ignore non-objects."""
    array = np.asarray(raw_mask, dtype=np.int16) - 1
    array[array < 0] = IGNORE_INDEX
    keep = np.asarray(tuple(keep_ids), dtype=np.int16)
    array[~np.isin(array, keep)] = IGNORE_INDEX
    return array.astype(np.uint8)


def crop_with_pad(image: Image.Image, cx: int, cy: int, size: int, fill) -> Image.Image:
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    output = Image.new(image.mode, (size, size), fill)
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(image.width, x1), min(image.height, y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output.paste(
            image.crop((source_x0, source_y0, source_x1, source_y1)),
            (source_x0 - x0, source_y0 - y0),
        )
    return output


def crop_pad_tensor(tensor: torch.Tensor, cx: int, cy: int, size: int) -> torch.Tensor:
    """Crop a CHW tensor, padding outside the image with zeros."""
    channels, height, width = tensor.shape
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    output = tensor.new_zeros(channels, size, size)
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(width, x1), min(height, y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[
            :, source_y0 - y0 : source_y1 - y0, source_x0 - x0 : source_x1 - x0
        ] = tensor[:, source_y0:source_y1, source_x0:source_x1]
    return output


def sample_class_points(
    mask: np.ndarray,
    class_id: int,
    count: int,
    min_distance: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample deterministic object pixels with a preferred spatial separation."""
    ys, xs = np.where(mask == class_id)
    if len(xs) == 0:
        return []
    order = rng.permutation(len(xs))
    points: list[tuple[int, int]] = []
    min_distance_sq = min_distance * min_distance
    for index in order:
        point = (int(xs[index]), int(ys[index]))
        if all(
            (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2
            >= min_distance_sq
            for previous in points
        ):
            points.append(point)
            if len(points) == count:
                return points
    for index in order:
        point = (int(xs[index]), int(ys[index]))
        if point not in points:
            points.append(point)
            if len(points) == count:
                break
    return points


def square_class_box(
    mask: np.ndarray,
    class_id: int,
    padding_fraction: float,
    image_size: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask == class_id)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    side = max(8, int(round(side * (1.0 + padding_fraction))))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
    clipped = (max(0, x0), max(0, y0), min(image_size, x1), min(image_size, y1))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def read_manifest(prepared_root: str | Path, split: str) -> list[dict[str, str]]:
    path = Path(prepared_root) / f"{split}_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run ade20k-prepare subset first")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_paths(ade_root: str | Path, row: dict[str, str]) -> tuple[Path, Path]:
    root = resolve_ade_root(ade_root)
    return root / row["image_path"], root / row["annotation_path"]


class FullImageDataset(Dataset):
    def __init__(
        self,
        ade_root: str | Path,
        prepared_root: str | Path,
        split: str,
        image_size: int = 512,
        limit: int = 0,
    ):
        self.ade_root = resolve_ade_root(ade_root)
        self.rows = read_manifest(prepared_root, split)
        self.image_size = image_size
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image_path, annotation_path = manifest_paths(self.ade_root, row)
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), BILINEAR
        )
        raw_mask = Image.open(annotation_path).resize(
            (self.image_size, self.image_size), NEAREST
        )
        return {
            "image": image_to_tensor(image),
            "mask": torch.from_numpy(remap_object_mask(raw_mask).astype(np.int64)),
            "image_id": row["image_id"],
        }


class CropDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        context_root: str | Path | None = None,
        limit: int = 0,
    ):
        self.files = sorted((Path(cache_root) / split).glob("*.npz"))
        self.context_root = Path(context_root) / split if context_root else None
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No crop files found in {Path(cache_root) / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        with np.load(self.files[index], allow_pickle=False) as sample:
            image_id = str(sample["image_id"])
            item: dict[str, torch.Tensor | str | int] = {
                "image": torch.from_numpy(sample["image"].copy())
                .permute(2, 0, 1)
                .float()
                / 255.0,
                "mask": torch.from_numpy(sample["mask"].astype(np.int64)),
                "class_id": int(sample["class_id"]),
                "image_id": image_id,
            }
            for key in ("x", "y"):
                if key in sample:
                    item[key] = int(sample[key])
        if self.context_root is not None:
            context_path = self.context_root / f"{image_id}.npz"
            with np.load(context_path, allow_pickle=False) as context:
                item["context"] = (
                    torch.from_numpy(context["context"].copy())
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
        return item


def group_crop_files(cache_root: str | Path, split: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((Path(cache_root) / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            grouped[str(sample["image_id"])].append(path)
    return dict(grouped)
