# Third-Party Dependencies

This repo excludes vendored third-party code. To run the detector/protocol stack, provide these separately:

- `third_party/LLM2Seg`
  - expected to contain the public LLM2Seg codebase and its config/model utilities
- model checkpoints and local model directories
- dataset files

The code prefers `LLM2SEG_DIR` from the environment. If unset, it falls back to `third_party/LLM2Seg`.
