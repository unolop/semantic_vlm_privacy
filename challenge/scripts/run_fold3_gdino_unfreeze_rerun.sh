#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${CONFIG:-$ROOT/challenge/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py}
SPLITS_DIR=${SPLITS_DIR:-$ROOT/results/base_cv_splits_3fold}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT/data/Biv-priv-seg/images}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-$ROOT/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/results/base_cv_train_unfreeze_lang}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_EPOCHS=${MAX_EPOCHS:-20}
FOLD=${FOLD:-3}

conda run --no-capture-output -n psi python "$ROOT/challenge/scripts/train_llm2seg_baseline.py"   --config "$CONFIG"   --train-ann "$SPLITS_DIR/fold$FOLD/train.json"   --train-img-root "$IMAGE_ROOT"   --val-ann "$SPLITS_DIR/fold$FOLD/val.json"   --val-img-root "$IMAGE_ROOT"   --pretrained-checkpoint "$PRETRAINED_CKPT"   --work-dir "$OUTPUT_ROOT/fold$FOLD"   --batch-size "$BATCH_SIZE"   --max-epochs "$MAX_EPOCHS"   --unfreeze-language-model
