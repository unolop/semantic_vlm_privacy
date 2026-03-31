# privacy-challenge-code

Curated repo containing our challenge-specific code, prompts, protocols, configs, and docs.

This repo intentionally excludes:
- third-party code snapshots such as LLM2Seg
- model weights and checkpoints
- datasets
- experiment result artifacts

## Layout
- `challenge/protocols`: Qwen-GDINO-SAM protocol code
- `challenge/scripts`: data prep, training wrappers, evaluation, diagnostics
- `challenge/prompts`: prompt templates
- `challenge/configs`: our custom training config overrides
- `challenge/folds`: split definitions
- `challenge/*.md`: project docs

## External dependencies
This repo expects third-party code to be provided separately. See `THIRD_PARTY.md`.
