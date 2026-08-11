"""Paper constants and ADE20K-Object class selection."""

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = 255
N_CLASSES = 150

ADE20K_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug",
    "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "storage tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
)

# Zero-based ADE20K IDs retained by the object-centric benchmark.
OBJECT_CLASS_IDS = (
    7, 8, 10, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 27, 28, 30, 31,
    33, 35, 36, 37, 39, 41, 43, 44, 45, 47, 49, 50, 55, 56, 57, 58, 62,
    63, 64, 65, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 81, 82,
    83, 85, 87, 88, 89, 90, 92, 97, 98, 99, 100, 102, 103, 107, 108,
    110, 111, 112, 115, 116, 117, 118, 119, 120, 124, 125, 126, 127,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142,
    143, 145, 146, 147, 148, 149,
)
OBJECT_CLASS_NAMES = tuple(ADE20K_CLASSES[index] for index in OBJECT_CLASS_IDS)

if len(ADE20K_CLASSES) != N_CLASSES:
    raise RuntimeError("ADE20K class list must contain 150 entries")
if len(OBJECT_CLASS_IDS) != 98:
    raise RuntimeError("ADE20K-Object must contain 98 classes")


@dataclass(frozen=True)
class PaperConfig:
    seed: int = 123
    image_size: int = 512
    fovea_size: int = 96
    context_size: int = 128
    samples_per_class: int = 3
    min_center_distance: int = 72
    object_box_padding: float = 0.15
    ior_radius: int = 70
    saliency_size: int = 128
    saliency_levels: int = 3
    min_discovery_fraction: float = 0.05
    fixation_budgets: tuple[int, ...] = (4, 8, 12, 16, 20, 24)


PAPER_CONFIG = PaperConfig()
