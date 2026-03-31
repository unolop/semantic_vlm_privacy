# Challenge Project Overview

## Scope

This directory is for the VizWiz object localization challenge track built around BIV-Priv-Seg-related assets.

Canonical direction summary:

- [`DIRECTION.md`](/home/choheeseung/workspace/vlm-privacy/challenge/DIRECTION.md)

- Challenge reference: `https://vizwiz.org/tasks-and-datasets/object-localization/`
- Local data root: `/home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg`
- Baseline code snapshot: `/home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg`
- Reference papers:
  - `challenge/papers/Biv-priv-seg.pdf`
  - `challenge/papers/LLM2Seg LLM-Guided Few-Shot Object Localization with Visual Transformer.pdf`
  - `challenge/papers/신진연구_연구계획서_Final.pdf`

## Facts Driving the Project

From `BIV-Priv-Seg`:

- Dataset size reported in the paper: 1,028 images, 16 private categories, 967 instance segmentations.
- A meaningful fraction of images do not contain the target object; the paper reports about 18% to 19%.
- Failure cases emphasized by the paper are:
  - small objects
  - non-salient objects
  - objects without text
  - images with no target present

From `LLM2Seg`:

- Winning recipe is a cascade:
  - support augmentation
  - Grounding DINO fine-tuning
  - query image preprocessing
  - LLM category inference
  - SAM mask generation
- The paper attributes gains to image preprocessing plus fine-tuned detection, not to a single model swap.

From `신진연구`:

- The long-term framework is `privacy awareness -> localization/reasoning -> protection`.
- For this challenge, the closest directly testable slice is the `localization/reasoning` stage.
- Engineering should therefore improve:
  - target presence recognition
  - location precision
  - reasoning traceability for why a predicted region is privacy-sensitive

## Project Positioning

This project should not try to reproduce the full multi-year research plan inside the challenge setting. The challenge is narrower:

- input: support/query images
- output: object localization and segmentation
- evaluation: detection/segmentation metrics on challenge protocol

So the practical goal here is:

`Use the challenge as a controlled engineering testbed for the localization and privacy-reasoning components of the proposed research pipeline.`

## Fixed Project Constraint

Keep this distinction fixed throughout the project:

- The real target is not a generic local few-shot benchmark.
- The real target is `few-shot grounding under a non-private -> private split`.
- Concretely:
  - training prior comes from non-private categories
  - novel support comes from private categories
  - the model must discriminate and localize private objects at region level

Any local pseudo-novel split is only a restricted internal validation tool for iteration.
It is not the main result, not the official challenge evaluation, and should not replace the actual project objective above.

## Current Strategic Conclusion

The current direction is now:

`Qwen-first, hybrid fallback`

Meaning:

- the conceptually correct first path is direct `support + query -> bbox` grounding with `Qwen3-VL`
- this is the few-shot-faithful formulation of the challenge
- if VizWiz internal performance remains insufficient, the practical fallback is `Qwen -> semantic cue -> G-DINO -> SAM`

Current Track B evidence:

- mixed eval shows `positive grounding exists`, but `negative rejection` is weak
- positive-only eval confirms the model can do support-conditioned grounding
- prompt-only `existence-first` fails because the current checkpoint was trained on a `find-first` distribution

So the current interpretation is:

- `Track B` currently means `positive-only few-shot grounding`
- `Track A` remains the practical submission fallback if Track B is not strong enough

## Main Contribution Direction

At the current stage, the intended contribution is:

`diagnostic-driven category-type-aware hybrid few-shot grounding`

Concretely:

- diagnose which signal actually drives localization before overbuilding the method
- avoid one-prompt-for-all behavior
- keep detector prior where it is already strong
- spend semantic budget where it matters more, especially on `private-like` categories

This is the current practical bridge between the challenge work and the broader privacy-grounding research direction.

## Reproduction Priority

The first concrete target is not the original OpenAI-backed `LLM2Seg` reproduction.

The immediate reproduction target is:

- keep the `LLM2Seg` cascade structure
- replace GPT category inference with a local/open-source VLM
- start with `Qwen3-VL-8B`

So the first baseline to measure is effectively:

`Qwen3-VL-8B -> Grounding DINO -> SAM`

This is the right first step because it tests whether the winning pipeline still holds when the closed API dependency is removed.

## Recommended Engineering Focus

1. Reproducible baseline first
- Standardize data layout and evaluation entrypoints.
- Re-run `LLM2Seg` on the local dataset structure inside the `psi` environment.
- Save every prompt, prediction, and intermediate artifact.

2. Negative-image handling
- Explicitly model `target absent` rather than forcing a category guess.
- Separate:
  - category prediction confidence
  - box generation confidence
  - final keep/reject decision

3. OCR-aware and text-aware routing
- Many private categories are text-heavy, but not all.
- Add routing logic to distinguish:
  - text-dominant categories
  - package/form categories
  - visually subtle non-text objects

4. Support-set engineering
- Build stronger support prototypes from the provided one-shot data using deterministic augmentation and prompt templates.
- Keep this controlled and challenge-compliant.

5. Localization calibration
- Tune thresholds, candidate reranking, and box filtering for:
  - small objects
  - border-touching objects
  - low-saliency objects

6. Reasoning artifact collection
- Even if the final submission only needs masks/boxes, keep explanation traces for analysis:
  - predicted category
  - evidence phrases
  - rejection reason for no-target cases

## Current Track Positioning

### Track A

Main challenge path.

- stage 1:
  - optimize `Grounding DINO` on `VizWiz base 100`
  - use image-level `3-fold` CV for detector optimization
- stage 2:
  - infer the target class using `Qwen`
  - route prompts differently for `generic` vs `private-like`
  - localize with the base-trained detector
  - segment with `SAM`

Short version:

`base-trained detector + support-conditioned semantic routing`

### Track B

Scoped extension path, not the main challenge path.

- learn support-query conditioned grounding with `Qwen3-VL`
- use the curated support-query JSONL as the starting training format
- evaluate transfer on the internal pseudo-novel split first

Short version:

`support-conditioned VLM grounding`

Track B is worth continuing only if it shows value beyond detector prior, especially on `private-like` categories.

## Temporary Active Priority

`G-DINO` full training is currently the wall-clock bottleneck.

So the temporary execution priority is:

1. do not redesign `Track A` while the long detector runs are already in progress
2. move active experimentation to `Track B`
3. start from the curated support-query grounding data
4. get a small `Qwen` smoke fine-tuning result before expanding complexity

This changes the active work queue, not the long-term interpretation:

- `Track A` remains the main challenge path
- `Track B` is simply the faster next thing to execute while detector training is expensive

## Working Assumptions

- Use `conda run -n psi ...` or equivalent `psi` environment execution for all scripts.
- Do not rely on background jobs.
- Keep all experiment outputs in versioned, dated directories.
- Prefer objective records:
  - config
  - prompts
  - raw detections
  - processed submission JSON
  - dev-set evaluation summary

## Suggested Directory Expansion

Recommended additions under `challenge/` as implementation starts:

- `challenge/docs/`: experiment notes, metric summaries
- `challenge/configs/`: model and run configs
- `challenge/scripts/`: dataset prep, evaluation, submission builders
- `challenge/src/`: reusable pipeline modules
- `challenge/results/`: dated experiment outputs

## Immediate Next Step

Start with [`PROJECT_PLAN.md`](/home/choheeseung/workspace/vlm-privacy/challenge/PROJECT_PLAN.md) and implement Phase 0 to Phase 1 before adding any new model complexity.

## Baseline Execution

The current baseline path is:

`support augmentation -> Grounding DINO fine-tuning -> qwen-guided dev-158 inference`

1. Build augmented support data

```bash
conda activate psi
python challenge/scripts/build_llm2seg_augmented_support.py \
  --support-json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
  --support-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
  --output-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg \
  --augmentations-per-image 32 \
  --seed 42
```

2. Train Grounding DINO on the augmented support set

```bash
conda activate psi
python challenge/scripts/train_llm2seg_baseline.py \
  --config /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
  --augmented-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/augmented_images_v2 \
  --pretrained-checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  --work-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/llm2seg_baseline_train
```

3. Run the trained baseline on the 158-image dev metadata split

```bash
conda activate psi
python challenge/scripts/run_llm2seg_dev.py \
  --query-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/query_images \
  --support-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
  --support-json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
  --json-path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_set_images_info.json \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/llm2seg_baseline_dev \
  --config-path /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
  --checkpoint-path /path/to/trained_checkpoint.pth \
  --sam-checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/sam_vit_h_4b8939.pth \
  --llm-model /home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct
```

Notes:

- `dev_set_images_info.json` contains 158 images and no local GT annotations, so this stage produces inference outputs but not local mAP.
- The training data expected by the original `LLM2Seg` config was missing from the repo; the local augmentation builder fills that gap.
- Any local pseudo-novel split built from VizWiz base data should be treated only as internal validation for method debugging.

## VizWiz Base 3-Fold Optimization Setup

For `Track A`, optimize the base detector with image-level `3-fold` CV on the full VizWiz base 100 categories.

This is separate from the pseudo-novel split.

1. Build the 3-fold image-level splits

```bash
conda activate psi
python challenge/scripts/prepare_vizwiz_base_cv_splits.py \
  --base-annotations /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/base_annotations.json \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/base_cv_splits_3fold \
  --num-folds 3 \
  --seed 42
```

This writes:

- `fold1/train.json`, `fold1/val.json`, `fold1/meta.json`
- `fold2/train.json`, `fold2/val.json`, `fold2/meta.json`
- `fold3/train.json`, `fold3/val.json`, `fold3/meta.json`
- `summary.json`

Observed split summary with `seed=42`:

- `fold1`: train `2793` images / val `1436` images
- `fold2`: train `2848` images / val `1381` images
- `fold3`: train `2817` images / val `1412` images

All three folds keep all `100` categories in both train and val.

2. Train one fold

```bash
conda activate psi
python challenge/scripts/train_llm2seg_baseline.py \
  --config /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
  --train-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/base_cv_splits_3fold/fold1/train.json \
  --train-img-root /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --val-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/base_cv_splits_3fold/fold1/val.json \
  --val-img-root /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --pretrained-checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  --work-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/base_cv_train/fold1
```

After selecting hyperparameters on the 3 folds, retrain once on the full base-100 set for the final detector.

## Diagnostic Subgroup Analysis

To check whether semantic signal matters more for `private-like` categories than for generic ones:

```bash
conda activate psi
python challenge/scripts/analyze_diagnostic_subgroups.py \
  --records /home/choheeseung/workspace/vlm-privacy/challenge/results/diagnostic_signal/split1/20260317/143007/diagnostic_records.json \
  --split-config /home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json \
  --split-name split1 \
  --query-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/query_eval.json \
  --output-json /home/choheeseung/workspace/vlm-privacy/challenge/results/diagnostic_signal/split1/20260317/143007/subgroup_summary.json
```

This outputs subgroup-wise summaries for:

- `private_like`
- `generic`

with per-experiment IoU / Hit@50 / gap comparisons.

## Track B Data Preparation Preview

To prepare support-query grounding examples for `Qwen3-VL` fine-tuning:

```bash
conda activate psi
python challenge/scripts/prepare_qwen_support_query_grounding.py \
  --base-annotations /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/base_annotations.json \
  --image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_preview \
  --examples-per-category 4 \
  --split-config /home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json \
  --split-name split1 \
  --seed 42
```

Curated version with large targets filtered out:

```bash
conda activate psi
python challenge/scripts/prepare_qwen_support_query_grounding.py \
  --base-annotations /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/base_annotations.json \
  --image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview \
  --examples-per-category 4 \
  --max-area-ratio 0.6 \
  --split-config /home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json \
  --split-name split1 \
  --seed 42
```

This writes:

- `train.jsonl`
- `eval.jsonl`
- `meta.json`

Quality audit:

```bash
conda activate psi
python challenge/scripts/audit_qwen_support_query_data.py \
  --train-jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_preview/train.jsonl \
  --eval-jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_preview/eval.jsonl \
  --output-json /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_preview/audit_summary.json
```

Format:

- two-image input: `support image`, `query image`
- assistant target: Qwen native grounding text
  - positive: `<ref>category name</ref><box>(x1,y1),(x2,y2)</box>`
  - negative: `<ref>category name</ref><box>none</box>`
  - coordinates normalized to `[0,1000]`

Current preview audit warning:

- the positive bbox area ratios are large on average
  - train mean `0.6448`
  - eval mean `0.5564`
- many examples are near full-image targets
  - train `157` large examples with area ratio `>= 0.6`
  - eval `63` large examples with area ratio `>= 0.6`

So the current preview is good for format validation, but too easy for final Track B training.
Before actual Qwen fine-tuning, curate or filter overly large targets.

Curated preview check:

- output: [`qwen_support_query_curated_preview`](/home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview)
- with `--max-area-ratio 0.6`
  - train mean area ratio drops from `0.6448` to `0.3977`
  - eval mean area ratio drops from `0.5564` to `0.3623`
  - large bbox examples drop to `0` in both train and eval

Immediate next step for `Track B`:

- use `qwen_support_query_curated_preview` as the first smoke fine-tuning dataset
- keep the task minimal:
  - support image + query image
  - native grounding text output only
  - no extra reasoning target yet

Track B swift SFT launcher:

```bash
conda activate psi
MODEL=/home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct

CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model ${MODEL} \
  --model_name ${MODEL} \
  --train_type lora \
  --dataset /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview/train.jsonl \
  --val_dataset /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview/eval.jsonl \
  --torch_dtype bfloat16 \
  --max_steps 2000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 5 \
  --logging_steps 100 \
  --max_length 4096 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --output_dir /home/choheeseung/workspace/vlm-privacy/challenge/results/track_b_qwen_sft/smoke_qwen4b \
  --truncation_strategy right \
  --max_pixels 448 \
  --load_best_model_at_end true \
  --eval_strategy steps \
  --use_hf true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --save_only_model false \
  --report_to wandb \
  --run_name Qwen3-VL-4B-Instruct_trackb \
  --early_stop_interval 5
```

If you want the same parameter set through the local wrapper instead of writing the full CLI each time:

```bash
conda activate psi
python challenge/scripts/run_track_b_qwen_sft.py \
  --model /home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct \
  --train-jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview/train.jsonl \
  --eval-jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview/eval.jsonl \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/track_b_qwen_sft/smoke_qwen4b
```

Track B eval launcher:

```bash
conda activate psi
python challenge/scripts/eval_track_b_qwen_grounding.py \
  --model /home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct \
  --lora-path /path/to/track_b_lora_checkpoint \
  --eval-jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/qwen_support_query_curated_preview/eval.jsonl \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/track_b_qwen_eval/smoke_qwen4b
```

## Prompt Strategy

Because image-side changes are constrained, prompt design is now a primary optimization axis.

### Track A

Use category-type-aware prompting instead of one prompt for all categories.

- `generic` categories:
  - detector prior is already strong
  - prefer weak prompts such as `object` or category-only
- `private-like` categories:
  - semantic signal is more useful
  - compare:
    - category only
    - category + short visual cue
    - support-derived appearance description

Reusable templates:

- [`track_a_query_support_16way.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_a_query_support_16way.txt)
- [`track_a_generic_category_only.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_a_generic_category_only.txt)
- [`track_a_private_like_category_plus_cue.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_a_private_like_category_plus_cue.txt)
- [`track_a_private_like_support_description.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_a_private_like_support_description.txt)

Track A category-inference ablation:

```bash
conda activate psi
python challenge/scripts/run_track_a_category_ablation.py \
  --support-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/support_1shot.json \
  --support-image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --query-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/query_eval.json \
  --query-image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --llm-model /home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct \
  --split-config /home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json \
  --split-name split1 \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/track_a_category_ablation/split1
```

This compares:

- `query_only`
- `query_plus_support16`

on single-target query images only, to avoid ambiguous local labels.

### Track B

Keep the support-query grounding contract fixed.

- system prompt:
  - define support-conditioned grounding
  - force native grounding output
- user prompt:
  - two-image input only
  - no extra reasoning

Reusable templates:

- [`track_b_system.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_b_system.txt)
- [`track_b_user.txt`](/home/choheeseung/workspace/vlm-privacy/challenge/prompts/track_b_user.txt)

## Pseudo-Novel 3-Split Training Setup

The fixed local split file is:

- [`pseudo_novel_3split.json`](/home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json)

Use it only for internal validation, not as the main project result.

1. Prepare one split

```bash
conda activate psi
python challenge/scripts/prepare_pseudo_novel_split.py \
  --base-annotations /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/base_annotations.json \
  --split-config /home/choheeseung/workspace/vlm-privacy/challenge/folds/pseudo_novel_3split.json \
  --split-name split1 \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1
```

This writes:

- `train_base.json`
- `support_1shot.json`
- `query_eval.json`
- `meta.json`

2. Build augmented support for the split

```bash
conda activate psi
python challenge/scripts/build_llm2seg_augmented_support.py \
  --support-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/support_1shot.json \
  --support-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1 \
  --augmentations-per-image 32 \
  --seed 42
```

3. Merge train-base + augmented support into one training dataset

```bash
conda activate psi
python challenge/scripts/merge_coco_for_training.py \
  --base-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/train_base.json \
  --base-image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --support-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/augmented_images_v2/augmented_annotations.json \
  --support-image-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/augmented_images_v2 \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/train_merged
```

4. Fine-tune G-DINO on the merged split dataset

```bash
conda activate psi
python challenge/scripts/train_llm2seg_baseline.py \
  --config /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
  --train-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/train_merged/annotations.json \
  --train-img-root /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/train_merged/images \
  --val-ann /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/query_eval.json \
  --val-img-root /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --pretrained-checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  --work-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_train/split1
```

Notes:

- `person(39)` stays in train by construction because it is excluded from the pseudo-novel split file.
- The contaminated-image policy is fixed: any image containing a pseudo-novel category is removed from train.
- This setup is for local method iteration only.
