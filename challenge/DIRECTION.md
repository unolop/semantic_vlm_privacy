# Challenge Direction

## One-Line Goal

Use the challenge as a testbed for:

`few-shot grounding under a non-private -> private split`

This is the fixed interpretation.
Do not reduce the project to a generic local few-shot benchmark.

## Core Problem Interpretation

The intended transfer is:

- train on `VizWiz base` non-private categories
- condition on one-shot private support
- localize private novel objects at region level

So the real question is not just:

`Can the model find an unseen object from one support image?`

The actual question is:

`Can a model trained on non-private categories use one-shot private support to discriminate and localize private objects at region level?`

## Current Diagnostic Evidence

The benchmark-signal diagnostic remains the main signal-structure reference.

Observed on the local split1 benchmark-signal diagnostic:

- overall:
  - `exp1_oracle mean_iou = 0.8422`
  - `exp4_object mean_iou = 0.8412`
  - `exp1 - exp4 gap = 0.0010`
- subgroup:
  - `private_like`
    - oracle `0.9414`
    - object `0.9195`
    - oracle-object gap `+0.0219`
  - `generic`
    - oracle `0.8168`
    - object `0.8211`
    - oracle-object gap `-0.0043`

Working interpretation:

- `generic` categories are mostly detector-prior dominated
- `private-like` categories retain a meaningful semantic-guidance gap
- semantic signal importance is category-dependent

## Current Track B Evidence

The current `Qwen3-VL-8B` Track B result should be interpreted narrowly.

Mixed-setting eval (`100` examples):

- `parse_success_rate = 0.95`
- `exists_accuracy = 0.52`
- `positive_mean_iou = 0.504`
- `positive_hit50 = 0.481`

Positive-only eval (`100` examples):

- `parse_success_rate = 0.96`
- `exists_accuracy = 0.96`
- `positive_mean_iou = 0.452`
- `positive_hit50 = 0.40`

Existence-first prompt-only eval (`100` examples):

- `parse_success_rate = 0.27`
- `exists_accuracy = 0.16`
- `positive_mean_iou = 0.180`
- `positive_hit50 = 0.173`

Working interpretation:

- the model can perform `support-conditioned positive grounding`
- the main current weakness is `negative rejection` and `box calibration`
- prompt-only `existence-first` failed because the current checkpoint was not trained on that task distribution
- therefore Track B should currently be interpreted as `positive-only few-shot grounding`, not as a calibrated full detection system

## SFT vs Inference Prompt Interpretation

The current Track B checkpoint was trained on a `find-first` distribution.

Meaning:

- support image gives the category reference
- query image is mainly framed as a place to find a matching object
- the model is mainly rewarded for producing a native grounding box

So the current prompt relationship is:

- `SFT prompt`: `find-first`
- `baseline inference prompt`: still `find-first`
- `existence-first inference prompt`: `decision-first`

This means the failure of prompt-only `existence-first` should be interpreted as:

`task distribution mismatch`

not as proof that null-aware grounding is intrinsically impossible.

## SFT Format Interpretation

Track B currently follows Qwen's native grounding output style, but not a stock grounding training task.

What is aligned with Qwen native grounding:

- output schema uses native grounding text
  - positive: `<ref>...</ref><box>(x1,y1),(x2,y2)</box>`
  - negative: `<ref>...</ref><box>none</box>`
- coordinates are normalized to `[0,1000]`

What remains custom to this project:

- the training task is a support-conditioned few-shot grounding task
- the model receives `support image + query image`
- supervision is built from our VizWiz support-query construction, not from a stock public grounding benchmark format

So the correct statement is:

`Track B uses Qwen-native grounding output format, but the SFT task itself is a custom few-shot support-query grounding setup.`

## Main Contribution Direction

The current intended contribution is:

`VLM-first few-shot grounding with hybrid fallback under a non-private -> private split`

Meaning:

- start from the few-shot-faithful formulation: `support + query -> bbox`
- use the VLM directly as the primary hypothesis
- keep `G-DINO + SAM` as the practical fallback if direct VLM grounding is insufficient
- do not force one uniform story across all categories or all failure modes

## Track Positioning

### Track B

Primary hypothesis and first attempt.

Definition:

- support-query conditioned `Qwen3-VL` grounding
- direct bbox generation from support + query
- current interpretation: `positive-only few-shot grounding`

Practical summary:

`support-conditioned VLM grounding`

This is the conceptually correct first path because it matches the few-shot problem definition most directly.

### Track A

Fallback and practical submission path.

Definition:

- base-trained `Grounding DINO`
- semantic/category cue from `Qwen`
- `SAM` for segmentation

Practical summary:

`VLM cue -> detector -> SAM`

This is the path to emphasize if Track B remains weak on calibration, null handling, or final challenge-level robustness.

## Research Bridge

This challenge is not a full implementation of [`privacy_grounding_research.md`](/home/choheeseung/workspace/vlm-privacy/privacy_grounding_research.md).

The current bridge is narrower:

- the whole benchmark is not equally useful as a research bridge
- the bridge exists mainly through the `private-like` subset
- Track B is currently useful as evidence that `support-conditioned positive grounding` is possible
- `existence-first` or null-aware grounding remains a later research extension, not the current challenge setting

So the correct interpretation is:

`The challenge remains relevant to the research agenda only insofar as it exposes where semantic support matters for private-like region grounding and whether direct support-conditioned grounding is feasible at all.`

## What Not To Do

- do not present local pseudo-novel validation as the main result
- do not overclaim Track B as full detection when the current evidence is positive-only
- do not treat prompt-only `existence-first` failure as a final verdict on null-aware grounding
- do not overinvest in Track B if calibration remains weak after limited prompt/data refinement

## Current Execution Priority

The current order is:

1. `Track B` first, with `Qwen3-VL-8B`
2. interpret Track B as positive-only few-shot grounding unless retrained for null-aware behavior
3. if VizWiz internal performance remains insufficient, fall back to `Track A` hybrid
4. keep `G-DINO` runs alive, but do not block active reasoning on them

This is the current practical order:

`Qwen-first -> if insufficient, G-DINO hybrid`

## Decision Rule

Keep investing in Track B if:

- positive-only grounding remains stable enough to justify direct support-conditioned localization
- subgroup analysis suggests value beyond detector prior, especially on `private-like` categories

Switch practical emphasis to Track A if:

- Track B remains weak on calibration after limited prompt/data refinement
- Track B does not produce sufficiently competitive VizWiz internal performance
- the direct VLM path remains more useful as a research probe than as a submission path


## 2026-03-22 Support-Centric Summary

### What Has Been Established

The challenge is still interpreted as:

`few-shot grounding under a non-private -> private split`

Within that framing, the main distinction from `LLM2Seg` is now fixed as:

- `LLM2Seg` mainly reduces support to a category token or an augmentation source
- our Track B starts from `support as a direct visual reference`

So the key methodological question is:

`Can support images function as direct few-shot grounding references rather than being collapsed into label-level shortcuts?`

### What Track B Has Shown So Far

The current `Qwen3-VL-8B` Track B result supports the following narrower claim:

- direct support-conditioned positive grounding is feasible
- the current checkpoint is not yet a calibrated mixed-setting detector
- the main weakness is negative rejection and box calibration, not complete failure of support-query matching

This means the current Track B interpretation should stay fixed as:

`positive-only support-conditioned grounding capability`

### Why The Prompt Results Matter

The current checkpoint was trained on a `find-first` support-query grounding distribution.

So the prompt relationship is:

- `SFT prompt`: find the matching object in the query image
- `baseline inference prompt`: same family as training
- `existence-first inference prompt`: different task distribution

Therefore, the prompt-only collapse under `existence-first` should be interpreted as:

`distribution mismatch between SFT behavior and inference-time task framing`

not as evidence that null-aware grounding is impossible.

### Why This Differs From Detector-Style Pipelines

Detector-style systems are naturally closer to:

- `image -> multiple boxes`

The current Track B setup is instead:

- `support image + query image -> target box`

So Track B is structurally a `single-target grounding` setup, not a scene-wide multi-object detector.

This is why Track B is conceptually aligned with the few-shot problem definition, while Track A/Hybrid remains the practical fallback for full challenge submission.

### What The Next Step Means

For the next decision point, keep the following split fixed:

- `Track B`: support-centric, few-shot-faithful, positive-only grounding hypothesis
- `Track A`: practical fallback path using `Qwen -> semantic cue -> G-DINO -> SAM`

So after the current waiting period, the next question should be:

`Does direct support-conditioned grounding provide enough additional value to justify one more limited refinement step before we move practical emphasis to the hybrid detector path?`
