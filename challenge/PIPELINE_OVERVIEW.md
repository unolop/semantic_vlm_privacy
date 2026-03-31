# Pipeline Overview

## One-Line View

Treat the system as one controllable pipeline rather than a list of separate models.

`support/query -> controller -> localizer -> segmenter -> output`

For the current challenge work, the practical pipeline is:

`support/query -> Qwen cue -> Grounding DINO bbox -> SAM mask`

## Why This View Matters

The important unit is not `Qwen`, `G-DINO`, and `SAM` individually.
The important unit is the interface between them.

What we want to control is:

- what information from support is preserved
- how that information is converted into a localization cue
- how bbox prediction is refined into a final mask

This is the right level of abstraction for both implementation and documentation.

## Pipeline Roles

### 1. Controller

Current role:

- consume `support` and optionally `query`
- produce a semantic cue that can guide localization
- examples:
  - category name
  - short visual description
  - routing decision

Current model choice:

- `Qwen`

This module should be described as a controller, not just as a classifier.

### 2. Localizer

Current role:

- take the image and the controller cue
- produce one or more candidate bounding boxes

Current model choice:

- `Grounding DINO`

This module is responsible for practical multi-box localization and calibration.

### 3. Segmenter

Current role:

- refine a selected bbox into a segmentation mask

Current model choice:

- `SAM`

## Current Baseline Pipelines

### Baseline A: Public LLM2Seg-Style Reproduction

`support(16) -> augmentation -> G-DINO fine-tune -> local VLM cue -> G-DINO bbox -> SAM mask`

Interpretation:

- closest to the public LLM2Seg code path
- useful as a reproduction baseline
- not the main detector-prior baseline

### Baseline B: VizWiz Base Detector Baseline

`VizWiz base-100 train -> G-DINO detector prior -> local VLM cue -> G-DINO bbox -> SAM mask`

Interpretation:

- detector baseline that better matches the intended challenge prior
- should be treated as the main practical baseline for this week
- current plan:
  - 3-fold image-level CV
  - choose a stable epoch range
  - retrain on full base-100

## Research-Oriented Direct Path

The conceptually cleaner few-shot path is:

`support + query -> VLM grounding -> bbox -> mask`

Interpretation:

- this is the few-shot-faithful formulation
- support remains a direct visual reference
- current local evidence only supports this as `positive-only support-conditioned grounding`
- it is not yet a practical full detection pipeline

## Difference From A Model List

Do not write the system as:

- Qwen
- G-DINO
- SAM

Write it as:

- support-conditioned controller
- localizer
- segmenter
- final output

This makes the pipeline easier to reason about and makes prompt/interface design central.

## What Assets Should Be Shown

A pipeline-style note or slide should show one end-to-end example with:

- support image
- query image
- controller prompt
- controller output cue
- intermediate bbox
- final mask
- one failure case

Without these assets, the pipeline is hard to evaluate qualitatively.

## Current Weekly Priority

This week's priority is not to maximize the direct VLM path.
This week's priority is to establish the baseline pipeline cleanly.

So the current order is:

1. `Baseline A` reproduction
2. `Baseline B` VizWiz base detector baseline
3. after the baseline is stable, return to direct VLM grounding if needed

## Current Decision

Use the following wording as the default project view:

`We treat the challenge system as a support-conditioned controllable pipeline, where a VLM-derived controller provides localization cues, Grounding DINO proposes bounding boxes, and SAM refines the final mask.`
