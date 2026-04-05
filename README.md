semantic_vlm_privacy

Standalone code snapshot for the current privacy challenge pipeline.

Pipeline Summary
1. Stage 1 semantic split
2. Stage 2 Grounding DINO proposal generation
3. Stage 3 support-reference matching on top-k candidate crops
4. Final selection and optional visualization

Key point
- Few-shot support is used only at Stage 3.
- Stage 1 remains query-only in the current strongest no-train configuration.

Stage Details
Stage 1: Semantic split
- Input: query image only
- Output: coarse semantic family, detector-friendly cues, null prior
- Role: produce family prior and detector cues

Stage 2: Grounding DINO proposals
- Input: query image and Stage-1 cue list
- Output: candidate bounding boxes and detector scores
- Role: generate candidate object regions

Stage 3: Few-shot reference matching
- Input: support reference crops with exact labels, top-k candidate crops from Stage 2
- Output: best matching exact category and matching score
- Role: this is where few-shot support is used
- Role: asks which support category the crop is most similar to

Stage 4: Final selection
- Input: detector candidates and Stage-3 matches
- Output: final bbox/category result and optional overlay visualization
- Role: simple score-based final selection

Repository Structure
- semantic/: semantic pipeline and runner
- baseline/: detector-only evaluation and reusable Grounding DINO helpers
- common/: project-owned VLM caller, model loaders, overlay helpers, and text utils
- prompts/active/: active semantic and reference-matching prompts
- configs/: local Grounding DINO config files used by this repo

Main Files
- semantic/semantic_gdino_sam.py
- semantic/run_semantic_gdino_sam_pipeline.py
- baseline/qwen_gdino_sam.py
- baseline/eval_support_gdino_detector.py
- common/vlm.py
- common/model_loaders.py
- common/text_utils.py
- common/overlay_utils.py

Active Prompts
- prompts/active/semantic_query_only.txt
- prompts/active/semantic_support_query.txt
- prompts/active/semantic_rerank.txt
- prompts/active/semantic_document_text.txt
- prompts/active/semantic_transactional_text.txt
- prompts/active/semantic_reference_match.txt

Main Run Modes
1) Query-only semantic probe
python semantic/run_semantic_gdino_sam_pipeline.py --query-dir /path/to/images --json-path /path/to/annotations.json --output-dir /path/to/output/semantic_query_only --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --llm-model /path/to/Qwen3-VL-4B-Instruct --controller-mode query_only --device cuda --disable-sam --save-vis

2) Few-shot reference matching
This is the current main no-train configuration.
python semantic/run_semantic_gdino_sam_pipeline.py --query-dir /path/to/query_images --json-path /path/to/query_annotations.json --support-dir /path/to/support_images --support-json /path/to/support_set.json --output-dir /path/to/output/reference_match --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --llm-model /path/to/Qwen3-VL-4B-Instruct --controller-mode query_only --device cuda --disable-sam --calibration-mode reference_match --classification-top-k 2 --disable-document-text --save-vis

Notes
- This repo does not require an LLM2Seg checkout or runtime import path.
- Datasets, checkpoints, and experiment outputs are excluded.
- The current strongest no-train path is Stage-3 support-reference matching.

External Dependencies
See THIRD_PARTY.md.
