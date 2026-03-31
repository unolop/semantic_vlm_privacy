# privacy-challenge-code

Public code-only snapshot for the privacy challenge experiments.

## Main files
- `challenge/protocols/semantic_gdino_sam.py`
  - main semantic pipeline module
  - semantic controller
  - proposal prompt refinement
  - null policy
  - G-DINO proposal collection
  - optional reranking
  - optional SAM
- `challenge/scripts/run_semantic_gdino_sam_pipeline.py`
  - main execution entrypoint
- `challenge/protocols/qwen_gdino_sam.py`
  - baseline protocol module
- `challenge/scripts/eval_support_gdino_detector.py`
  - detector-only support-set evaluator
- `challenge/SEMANTIC_PIPELINE_STATUS.md`
  - concise status note for the semantic pipeline

## Included
- code
- prompts
- folds
- config overrides
- minimal public docs

## Excluded
- datasets
- model weights and checkpoints
- experiment outputs
- vendored third-party repositories
- internal planning documents

## External dependencies
See `THIRD_PARTY.md`.
