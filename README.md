# Efficient Semantic Understanding from Digital Foveation

ADE20K-Object reproduction code for the ECCV Workshop 2026 paper.

## 1. Install Dependencies

Python 3.10 or newer is required. Install the PyTorch build appropriate for
your CPU or GPU, then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Train the Models

Download and extract the official `ADEChallengeData2016` archive. Set the data
and prepared-data locations:

```bash
export ADE_ROOT=/path/to/ADEChallengeData2016
export PREPARED_ROOT=data/ade20k_object
```

Create the 98-class ADE20K-Object manifests and deterministic training caches:

```bash
ade20k-prepare subset \
  --ade-root "${ADE_ROOT}" \
  --output "${PREPARED_ROOT}"

ade20k-prepare fovea \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output "${PREPARED_ROOT}/fovea96_n3"

ade20k-prepare context \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output "${PREPARED_ROOT}/context128"

ade20k-prepare full-object \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output "${PREPARED_ROOT}/full_object96"
```

Train the full-image baseline:

```bash
ade20k-train \
  --model baseline \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output outputs/baseline \
  --epochs 15 \
  --batch-size 12 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --dice-weight 0.5 \
  --seed 123
```

Train the fovea-only model:

```bash
ade20k-train \
  --model fovea_only \
  --prepared-root "${PREPARED_ROOT}" \
  --output outputs/fovea_only \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --dice-weight 0.5 \
  --seed 123
```

Train the fovea-plus-context model:

```bash
ade20k-train \
  --model fovea_context \
  --prepared-root "${PREPARED_ROOT}" \
  --output outputs/fovea_context \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --dice-weight 0.5 \
  --seed 123
```

Train the full-object control:

```bash
ade20k-train \
  --model full_object \
  --prepared-root "${PREPARED_ROOT}" \
  --output outputs/full_object \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --dice-weight 0.5 \
  --seed 123
```

All models use ImageNet initialization, AdamW, polynomial learning-rate decay,
and cross-entropy plus object Dice loss. The complete configuration is stored
in `configs/paper_ade20k.json`.

## 3. Test the Models

Evaluate the final full-image baseline, fovea-only model, fovea-plus-context
model, and full-object control on the same validation crops:

```bash
ade20k-evaluate single \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --baseline-checkpoint checkpoints/baseline.pt \
  --fovea-checkpoint checkpoints/fovea_only.pt \
  --context-checkpoint checkpoints/fovea_context.pt \
  --full-object-checkpoint checkpoints/full_object.pt \
  --output results/single_fixation.json
```

Evaluate the Itti-Koch fixed-budget pipeline and the final full-image baseline:

```bash
ade20k-evaluate canvas \
  --ade-root "${ADE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --baseline-checkpoint checkpoints/baseline.pt \
  --context-checkpoint checkpoints/fovea_context.pt \
  --budgets 4 8 12 16 20 24 \
  --output results/fixed_canvas.json
```
