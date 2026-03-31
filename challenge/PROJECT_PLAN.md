# Challenge Project Plan

## Objective

Build a challenge-focused engineering pipeline for private object localization that is:

- reproducible
- diagnosable
- competitive against the `LLM2Seg` baseline
- aligned with the `신진연구` direction on privacy-aware localization and reasoning

Canonical direction summary:

- [`DIRECTION.md`](/home/choheeseung/workspace/vlm-privacy/challenge/DIRECTION.md)

The current execution target is more specific:

- main path: `Track A` category-type-aware hybrid pipeline
- extension path: `Track B` support-conditioned VLM grounding
- research bridge: preserve evidence that semantic guidance matters on `private-like` categories

## Fixed Interpretation

Treat the challenge as:

- `few-shot grounding under a non-private/private split`

Do not collapse it into a generic local few-shot benchmark.

The intended transfer is:

- train on non-private prior
- condition on private one-shot support
- localize private novel objects at region level

Any pseudo-novel split built from base classes is only an internal validation mechanism for fast iteration.
It is not the official evaluation target and should not be presented as the main outcome.

## Current Diagnostic-Based Decision

The benchmark-signal diagnostic changed the project priority.

Current working conclusion:

- `generic` categories are mostly detector-prior dominated
- `private-like` categories retain a meaningful semantic-guidance gap
- therefore the project should not force a uniform single-model solution across all categories

Operational consequence:

- keep `Track A` as the main challenge path
- let `Track B` focus on whether support-conditioned VLM grounding adds value specifically where semantic signal matters more

## Non-Goals

- Do not attempt the full end-to-end privacy protection framework in this directory first.
- Do not add uncontrolled complexity before a reproducible baseline exists.
- Do not optimize only for qualitative examples; keep dev-set metrics central.

## Phase 0. Data and Evaluation Audit

Goal: remove ambiguity in dataset layout and scoring.

Deliverables:

- dataset inventory script
- category/id mapping validation
- support/query/dev split summary
- expected submission schema check

Checks:

- confirm all required files under `/data/Biv-priv-seg`
- verify image counts against JSON metadata
- verify category names exactly match challenge labels
- identify how target-absent images are represented in metadata

Exit criteria:

- one command produces a dataset audit report
- one command validates a submission JSON before upload

## Phase 1. Baseline Reproduction

Goal: run `LLM2Seg` reproducibly on the local layout.

Deliverables:

- wrapper script for local paths
- fixed config file for checkpoints and thresholds
- output schema:
  - `raw/`
  - `processed/`
  - `artifacts/`

Artifacts to save per run:

- resolved arguments
- prompt text
- category prediction
- raw detection candidates
- final bbox/mask outputs
- evaluation summary

Exit criteria:

- baseline runs end-to-end from one command
- dev-set metrics are recorded in a machine-readable summary

## Phase 2. Engineering Improvements

Goal: improve challenge performance without changing project scope.

The current practical principle is:

- do not spend semantic complexity uniformly
- use stronger semantic guidance where the diagnostic shows it matters
- avoid degrading detector-prior-dominated categories with unnecessary prompting

### Workstream A. Target-Absent Detection

Problem:

- BIV-Priv-Seg explicitly contains target-absent images.
- A forced positive prediction will create false positives early in the pipeline.

Plan:

- add a no-target decision stage before final box emission
- calibrate keep/reject thresholds on dev data
- log false-positive causes separately

Success signal:

- reduced false positives on target-absent samples without large recall collapse

### Workstream B. Query Preprocessing Ablation

Problem:

- `LLM2Seg` reports gains from preprocessing, but the effect size should be verified locally.

Plan:

- compare:
  - none
  - super-resolution only
  - contrast/sharpen only
  - combined preprocessing

Success signal:

- measurable gain on small-object and border-object subsets

### Workstream C. Prompt and Label Engineering

Problem:

- LLM category inference can fail when labels are visually similar or text-heavy.
- semantic prompting is not equally useful across all category types

Plan:

- compare prompt variants:
  - strict 16-way classification
  - shortlist then final pick
  - evidence-first reasoning then label pick
- force normalized output schema
- keep separate prompt policies for:
  - detector-prior-dominated `generic` categories
  - semantic-sensitive `private-like` categories

Success signal:

- improved category accuracy and lower prompt parsing failures
- improved `private-like` localization without harming `generic` categories

### Workstream D. Support Prototype Expansion

Problem:

- one-shot support is brittle.

Plan:

- deterministic augmentation bank
- category-specific text descriptions
- support-image OCR extraction where applicable

Success signal:

- better reranking and box selection for ambiguous categories

### Workstream E. Box and Mask Calibration

Problem:

- small, non-salient, border-touching objects are the documented failure region.

Plan:

- tune:
  - box threshold
  - text threshold
  - NMS or reranking policy
  - SAM candidate selection policy
- evaluate by object-size and border-contact subsets

Success signal:

- subset gains without broad regression

## Phase 3. Research-Aligned Extensions

Goal: add minimal research alignment without breaking challenge focus.

At the current stage, research alignment should be narrow and evidence-driven:

- do not claim the whole benchmark is a direct bridge to the research agenda
- treat the `private-like` subset as the main bridge region
- preserve artifacts that explain when semantic support helps and when detector prior is already enough

1. Reasoning trace logging
- store short evidence strings for each kept prediction
- store explicit rejection rationale for no-target predictions

2. Open-vocabulary auxiliary analysis
- compare challenge label prediction with a free-form privacy cue description
- use this only for analysis unless it improves dev metrics

3. Privacy sensitivity tagging
- attach a lightweight direct/indirect cue tag per prediction for later downstream protection studies

This phase is useful because it maps directly to the `신진연구` objective of privacy reasoning, while keeping the challenge pipeline measurable.

## Evaluation Plan

Primary:

- challenge dev metrics for detection and segmentation

Secondary:

- no-target false positive rate
- small-object subset performance
- text vs non-text subset performance
- border-touching object performance
- category prediction accuracy before detection

Diagnostics:

- per-category confusion
- prompt parse failure count
- empty prediction count
- oversized box rate
- duplicate prediction rate

Local-only internal validation:

- if a pseudo-novel split is used, keep it strictly as an internal debugging and model-selection tool
- do not treat it as the project's primary result
- use it only to sanity-check few-shot transfer behavior before challenge-side evaluation

## Proposed Implementation Order

1. Phase 0 dataset/eval audit
2. Phase 1 baseline reproduction
3. `Track A` base-detector optimization on VizWiz base 100
4. benchmark-signal diagnostics and subgroup analysis
5. Workstream C prompt engineering with category-type-aware routing
6. Workstream A target-absent detection
7. Workstream E box/mask calibration
8. Workstream B preprocessing ablation
9. Workstream D support prototype expansion
10. `Track B` support-conditioned VLM grounding smoke experiments
11. Phase 3 research-aligned logging

## Risks

- `challenge/LLM2Seg` currently assumes its own dependency stack and path layout.
- Some LLM2Seg gains may depend on external services or checkpoints not yet standardized locally.
- The challenge evaluation server may hide failure modes that must be approximated locally with dev-set analysis.

## First Implementation Batch

The first coding batch should create only these pieces:

- dataset audit script
- local baseline runner
- result directory convention
- submission validator
- dev evaluation summary writer

Anything beyond that should wait until the baseline is reproducible and measured.
