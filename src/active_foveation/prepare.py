"""Create the ADE20K-Object manifests and deterministic paper caches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import (
    ADE20K_CLASSES,
    IGNORE_INDEX,
    OBJECT_CLASS_IDS,
    OBJECT_CLASS_NAMES,
    PAPER_CONFIG,
)
from .data import (
    BILINEAR,
    NEAREST,
    crop_with_pad,
    manifest_paths,
    read_manifest,
    remap_object_mask,
    resolve_ade_root,
    sample_class_points,
    square_class_box,
)

SPLITS = ("training", "validation")


def write_class_metadata(output: Path) -> None:
    payload = {
        "label_convention": "zero-based ADE20K IDs; non-object labels use ignore_index",
        "ignore_index": IGNORE_INDEX,
        "object_class_ids": list(OBJECT_CLASS_IDS),
        "object_class_names": list(OBJECT_CLASS_NAMES),
    }
    (output / "object_classes.json").write_text(json.dumps(payload, indent=2) + "\n")


def prepare_subset(
    ade_root: str | Path,
    output: str | Path,
    training_limit: int = 0,
    validation_limit: int = 0,
) -> None:
    ade = resolve_ade_root(ade_root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    write_class_metadata(destination)
    limits = {"training": training_limit, "validation": validation_limit}
    for split in SPLITS:
        images = sorted((ade / "images" / split).glob("*.jpg"))
        if limits[split]:
            images = images[: limits[split]]
        rows: list[dict[str, str | int | float]] = []
        for image_path in tqdm(images, desc=f"subset {split}"):
            annotation_path = ade / "annotations" / split / f"{image_path.stem}.png"
            if not annotation_path.is_file():
                raise FileNotFoundError(annotation_path)
            object_mask = remap_object_mask(Image.open(annotation_path))
            valid = object_mask != IGNORE_INDEX
            if not bool(valid.any()):
                continue
            present = sorted(int(value) for value in np.unique(object_mask[valid]))
            rows.append(
                {
                    "split": split,
                    "image_id": image_path.stem,
                    "image_path": image_path.relative_to(ade).as_posix(),
                    "annotation_path": annotation_path.relative_to(ade).as_posix(),
                    "object_pixels": int(valid.sum()),
                    "object_fraction": float(valid.mean()),
                    "object_classes": "|".join(ADE20K_CLASSES[cid] for cid in present),
                    "n_object_classes": len(present),
                }
            )
        path = destination / f"{split}_manifest.csv"
        fields = (
            "split",
            "image_id",
            "image_path",
            "annotation_path",
            "object_pixels",
            "object_fraction",
            "object_classes",
            "n_object_classes",
        )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {len(rows)} images -> {path}", flush=True)


def prepare_fovea_cache(
    ade_root: str | Path,
    prepared_root: str | Path,
    output: str | Path,
    image_size: int = PAPER_CONFIG.image_size,
    fovea_size: int = PAPER_CONFIG.fovea_size,
    samples_per_class: int = PAPER_CONFIG.samples_per_class,
    min_center_distance: int = PAPER_CONFIG.min_center_distance,
    seed: int = PAPER_CONFIG.seed,
    training_limit: int = 0,
    validation_limit: int = 0,
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": "object_pixel_fovea",
        "image_size": image_size,
        "fovea_size": fovea_size,
        "samples_per_class": samples_per_class,
        "min_center_distance": min_center_distance,
        "seed": seed,
        "object_class_ids": list(OBJECT_CLASS_IDS),
        "object_class_names": list(OBJECT_CLASS_NAMES),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    limits = {"training": training_limit, "validation": validation_limit}
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        split_output = destination / split
        split_output.mkdir(exist_ok=True)
        rows = read_manifest(prepared_root, split)
        if limits[split]:
            rows = rows[: limits[split]]
        records: list[dict[str, str | int]] = []
        sample_index = 0
        for row in tqdm(rows, desc=f"fovea {split}"):
            image_path, annotation_path = manifest_paths(ade_root, row)
            image = Image.open(image_path).convert("RGB").resize(
                (image_size, image_size), BILINEAR
            )
            raw_mask = Image.open(annotation_path).resize(
                (image_size, image_size), NEAREST
            )
            mask = remap_object_mask(raw_mask)
            mask_image = Image.fromarray(mask, mode="L")
            present = sorted(
                int(value)
                for value in np.unique(mask)
                if int(value) != IGNORE_INDEX
            )
            for class_id in present:
                points = sample_class_points(
                    mask,
                    class_id,
                    samples_per_class,
                    min_center_distance,
                    rng,
                )
                for x, y in points:
                    sample_id = f"{sample_index:08d}"
                    np.savez_compressed(
                        split_output / f"{sample_id}.npz",
                        image=np.asarray(
                            crop_with_pad(image, x, y, fovea_size, (0, 0, 0)),
                            dtype=np.uint8,
                        ),
                        mask=np.asarray(
                            crop_with_pad(
                                mask_image, x, y, fovea_size, IGNORE_INDEX
                            ),
                            dtype=np.uint8,
                        ),
                        class_id=np.uint16(class_id),
                        image_id=row["image_id"],
                        x=np.uint16(x),
                        y=np.uint16(y),
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "image_id": row["image_id"],
                            "class_id": class_id,
                            "class_name": ADE20K_CLASSES[class_id],
                            "x": x,
                            "y": y,
                        }
                    )
                    sample_index += 1
        manifest_path = destination / f"{split}_samples.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("sample_id", "image_id", "class_id", "class_name", "x", "y"),
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"{split}: {len(records)} foveal samples -> {manifest_path}", flush=True)


def prepare_context_cache(
    ade_root: str | Path,
    prepared_root: str | Path,
    output: str | Path,
    context_size: int = PAPER_CONFIG.context_size,
    training_limit: int = 0,
    validation_limit: int = 0,
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    limits = {"training": training_limit, "validation": validation_limit}
    for split in SPLITS:
        split_output = destination / split
        split_output.mkdir(exist_ok=True)
        rows = read_manifest(prepared_root, split)
        if limits[split]:
            rows = rows[: limits[split]]
        for row in tqdm(rows, desc=f"context {split}"):
            image_path, _ = manifest_paths(ade_root, row)
            image = Image.open(image_path).convert("RGB").resize(
                (context_size, context_size), BILINEAR
            )
            np.savez_compressed(
                split_output / f"{row['image_id']}.npz",
                context=np.asarray(image, dtype=np.uint8),
            )
    meta = {"mode": "global_rgb_context", "context_size": context_size}
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def prepare_object_box_cache(
    ade_root: str | Path,
    prepared_root: str | Path,
    output: str | Path,
    image_size: int = PAPER_CONFIG.image_size,
    fovea_size: int = PAPER_CONFIG.fovea_size,
    padding_fraction: float = PAPER_CONFIG.object_box_padding,
    training_limit: int = 0,
    validation_limit: int = 0,
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": "class_mask_object_box",
        "image_size": image_size,
        "fovea_size": fovea_size,
        "padding_fraction": padding_fraction,
        "object_class_ids": list(OBJECT_CLASS_IDS),
        "object_class_names": list(OBJECT_CLASS_NAMES),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    limits = {"training": training_limit, "validation": validation_limit}
    for split in SPLITS:
        split_output = destination / split
        split_output.mkdir(exist_ok=True)
        rows = read_manifest(prepared_root, split)
        if limits[split]:
            rows = rows[: limits[split]]
        records: list[dict[str, str | int]] = []
        sample_index = 0
        for row in tqdm(rows, desc=f"full-object {split}"):
            image_path, annotation_path = manifest_paths(ade_root, row)
            image = Image.open(image_path).convert("RGB").resize(
                (image_size, image_size), BILINEAR
            )
            raw_mask = Image.open(annotation_path).resize(
                (image_size, image_size), NEAREST
            )
            mask = remap_object_mask(raw_mask)
            present = sorted(
                int(value)
                for value in np.unique(mask)
                if int(value) != IGNORE_INDEX
            )
            for class_id in present:
                box = square_class_box(
                    mask, class_id, padding_fraction, image_size
                )
                if box is None:
                    continue
                sample_id = f"{sample_index:08d}"
                crop_image = image.crop(box).resize((fovea_size, fovea_size), BILINEAR)
                crop_mask = (
                    Image.fromarray(mask, mode="L")
                    .crop(box)
                    .resize((fovea_size, fovea_size), NEAREST)
                )
                np.savez_compressed(
                    split_output / f"{sample_id}.npz",
                    image=np.asarray(crop_image, dtype=np.uint8),
                    mask=np.asarray(crop_mask, dtype=np.uint8),
                    class_id=np.uint16(class_id),
                    image_id=row["image_id"],
                    x0=np.uint16(box[0]),
                    y0=np.uint16(box[1]),
                    x1=np.uint16(box[2]),
                    y1=np.uint16(box[3]),
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "image_id": row["image_id"],
                        "class_id": class_id,
                        "class_name": ADE20K_CLASSES[class_id],
                        "x0": box[0],
                        "y0": box[1],
                        "x1": box[2],
                        "y1": box[3],
                    }
                )
                sample_index += 1
        manifest_path = destination / f"{split}_samples.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "image_id",
                    "class_id",
                    "class_name",
                    "x0",
                    "y0",
                    "x1",
                    "y1",
                ),
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"{split}: {len(records)} full-object samples -> {manifest_path}", flush=True)


def add_common_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    subset = commands.add_parser("subset", help="create portable 98-class manifests")
    subset.add_argument("--ade-root", required=True)
    subset.add_argument("--output", required=True)
    add_common_limits(subset)

    fovea = commands.add_parser("fovea", help="create object-balanced foveal crops")
    fovea.add_argument("--ade-root", required=True)
    fovea.add_argument("--prepared-root", required=True)
    fovea.add_argument("--output", required=True)
    fovea.add_argument("--image-size", type=int, default=PAPER_CONFIG.image_size)
    fovea.add_argument("--fovea-size", type=int, default=PAPER_CONFIG.fovea_size)
    fovea.add_argument(
        "--samples-per-class", type=int, default=PAPER_CONFIG.samples_per_class
    )
    fovea.add_argument(
        "--min-center-distance",
        type=int,
        default=PAPER_CONFIG.min_center_distance,
    )
    fovea.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    add_common_limits(fovea)

    context = commands.add_parser("context", help="create low-resolution contexts")
    context.add_argument("--ade-root", required=True)
    context.add_argument("--prepared-root", required=True)
    context.add_argument("--output", required=True)
    context.add_argument("--context-size", type=int, default=PAPER_CONFIG.context_size)
    add_common_limits(context)

    object_box = commands.add_parser(
        "full-object", help="create full-object control crops"
    )
    object_box.add_argument("--ade-root", required=True)
    object_box.add_argument("--prepared-root", required=True)
    object_box.add_argument("--output", required=True)
    object_box.add_argument("--image-size", type=int, default=PAPER_CONFIG.image_size)
    object_box.add_argument("--fovea-size", type=int, default=PAPER_CONFIG.fovea_size)
    object_box.add_argument(
        "--padding-fraction",
        type=float,
        default=PAPER_CONFIG.object_box_padding,
    )
    add_common_limits(object_box)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = {
        "training_limit": args.training_limit,
        "validation_limit": args.validation_limit,
    }
    if args.command == "subset":
        prepare_subset(args.ade_root, args.output, **common)
    elif args.command == "fovea":
        prepare_fovea_cache(
            args.ade_root,
            args.prepared_root,
            args.output,
            image_size=args.image_size,
            fovea_size=args.fovea_size,
            samples_per_class=args.samples_per_class,
            min_center_distance=args.min_center_distance,
            seed=args.seed,
            **common,
        )
    elif args.command == "context":
        prepare_context_cache(
            args.ade_root,
            args.prepared_root,
            args.output,
            context_size=args.context_size,
            **common,
        )
    elif args.command == "full-object":
        prepare_object_box_cache(
            args.ade_root,
            args.prepared_root,
            args.output,
            image_size=args.image_size,
            fovea_size=args.fovea_size,
            padding_fraction=args.padding_fraction,
            **common,
        )


if __name__ == "__main__":
    main()
