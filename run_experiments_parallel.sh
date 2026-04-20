#!/usr/bin/env bash
# Parallel experiment orchestrator for semantic VLM privacy pipeline.
#
# Dependency graph:
#   [s1_seed42, s1_seed99 already running] → merge → s2_ensemble → s3_ensemble_8B
#                                                                 → s3_ensemble_8B_thr025
#                                                                 → s3_ensemble_8B_ocr
#   [baseline_stage2 already done]          → s3_baseline_thr025  (was OOMed)
#
# Usage:
#   bash run_experiments_parallel.sh [--dry-run]
#
# GPU assignments:
#   GPU0: merge → s2_ensemble → s3_ensemble_8B (sequential chain)
#   GPU1: s3_baseline_thr025 (starts immediately), then s3_ensemble_8B_thr025, s3_ensemble_8B_ocr

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] No commands will be executed."
fi

REPO=/home/david/semantic_vlm_privacy
CONDA_ENV=sem_mmdet_pip
PYTHON=/home/david/miniconda3/envs/${CONDA_ENV}/bin/python
DATA=${REPO}/data/vizwiz_object_localization
OUT=${REPO}/outputs_postpull
LOG_DIR=${REPO}/outputs_postpull/orchestrator_logs
mkdir -p "${LOG_DIR}"

MODEL_4B=/home/david/Desktop/yuna/.cache/modelscope/models/Qwen/Qwen3-VL-4B-Instruct
MODEL_8B=/home/david/Desktop/yuna/.cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct
GDINO_CONFIG=${REPO}/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py
GDINO_CKPT=${REPO}/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth
SAM_CKPT=${REPO}/checkpoints/sam_vit_h_4b8939.pth
SUPPORT_DIR=${DATA}/support_images/support_images
SUPPORT_JSON=${DATA}/support_set.json
JSON_PATH=${DATA}/dev_pseudo_label_coco.json
DOC_REFINE_PROMPT=${REPO}/prompts/active/semantic_document_refine.txt
FAMILY_CONFIG=${REPO}/config/family_category_direct_v1.json
QUERY_DIR=${DATA}/query_images

# Paths for ensemble Stage 1 inputs
S1_BASELINE=${OUT}/baseline_stage1/stage1_out.json
S1_S1=${OUT}/exp_ensemble_stage1_s1/stage1_out.json
S1_S2=${OUT}/exp_ensemble_stage1_s2/stage1_out.json

# Output dirs
ENSEMBLE_MERGED=${OUT}/exp_ensemble_merged
ENSEMBLE_S2=${OUT}/exp_ensemble_stage2
S3_ENSEMBLE=${OUT}/exp_ensemble_stage3_8B
S3_ENSEMBLE_THR=${OUT}/exp_ensemble_stage3_8B_thr025
S3_ENSEMBLE_OCR=${OUT}/exp_ensemble_stage3_8B_ocr
S3_BASELINE_THR=${OUT}/exp_docthresh_stage3_8B

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_cmd() {
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label}.log"
    log "START  ${label} → ${logfile}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  CMD: $*"
        return 0
    fi
    "$@" >"${logfile}" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAILED ${label} (exit ${rc}). Check ${logfile}"
        exit $rc
    fi
    log "DONE   ${label}"
}

wait_for_pid() {
    local label="$1" pid="$2"
    log "WAIT   ${label} (pid=${pid})"
    while kill -0 "${pid}" 2>/dev/null; do sleep 5; done
    log "READY  ${label} finished (pid=${pid})"
}

wait_for_file() {
    local label="$1" fpath="$2"
    log "WAIT   ${label} → ${fpath}"
    while [[ ! -f "${fpath}" ]]; do sleep 10; done
    log "READY  ${fpath} exists"
}

# ── find PIDs of the two running Stage 1 jobs ────────────────────────────────

S1_S1_PID=$(pgrep -f "exp_ensemble_stage1_s1.*stage1_out" || true)
S1_S2_PID=$(pgrep -f "exp_ensemble_stage1_s2.*stage1_out" || true)

# Fallback: just wait for the output files
if [[ -z "${S1_S1_PID}" ]]; then
    log "WARN: seed=42 Stage 1 PID not found, will poll for output file."
fi
if [[ -z "${S1_S2_PID}" ]]; then
    log "WARN: seed=99 Stage 1 PID not found, will poll for output file."
fi

# ── GPU1 chain: doc-threshold Stage 3 on baseline (starts immediately) ───────

gpu1_chain() {
    log "=== GPU1 CHAIN START ==="

    mkdir -p "${S3_BASELINE_THR}"
    run_cmd "s3_baseline_thr025" \
        ${PYTHON} semantic/run_stage3_calibration.py \
            --json_path "${JSON_PATH}" \
            --stage1_path "${S1_BASELINE}" \
            --stage2_path "${OUT}/baseline_stage2/stage2_out.json" \
            --output_dir "${S3_BASELINE_THR}" \
            --sam_checkpoint "${SAM_CKPT}" \
            --llm_model "${MODEL_8B}" \
            --device cuda:1 \
            --llm_decoding_mode deterministic \
            --llm_max_pixels 448 \
            --calibration_mode reference_match \
            --reference_source crop \
            --support_dir "${SUPPORT_DIR}" \
            --support_json "${SUPPORT_JSON}" \
            --disable_sam \
            --proposal_score_threshold 0.35 \
            --final_score_threshold 0.35 \
            --classification_top_k 2 \
            --verbose_decisions \
            --decision_log_jsonl "${S3_BASELINE_THR}/decisions.jsonl" \
            --document_category_threshold 0.25

    log "=== GPU1: s3_baseline_thr025 done, waiting for ensemble Stage 1 outputs ==="

    # Now wait for both Stage 1 ensemble runs
    wait_for_file "s1_seed42" "${S1_S1}"
    wait_for_file "s1_seed99" "${S1_S2}"

    log "=== GPU1: both Stage 1 seeds done — waiting for GPU0 to finish merge+stage2 ==="

    # Wait for ensemble Stage 2 output (produced by GPU0 chain)
    wait_for_file "ensemble_stage2" "${ENSEMBLE_S2}/stage2_out.json"

    # Now run doc-threshold Stage 3 on ensemble
    mkdir -p "${S3_ENSEMBLE_THR}"
    run_cmd "s3_ensemble_thr025" \
        ${PYTHON} semantic/run_stage3_calibration.py \
            --json_path "${JSON_PATH}" \
            --stage1_path "${ENSEMBLE_MERGED}/stage1_out.json" \
            --stage2_path "${ENSEMBLE_S2}/stage2_out.json" \
            --output_dir "${S3_ENSEMBLE_THR}" \
            --sam_checkpoint "${SAM_CKPT}" \
            --llm_model "${MODEL_8B}" \
            --device cuda:1 \
            --llm_decoding_mode deterministic \
            --llm_max_pixels 448 \
            --calibration_mode reference_match \
            --reference_source crop \
            --support_dir "${SUPPORT_DIR}" \
            --support_json "${SUPPORT_JSON}" \
            --disable_sam \
            --proposal_score_threshold 0.35 \
            --final_score_threshold 0.35 \
            --classification_top_k 2 \
            --verbose_decisions \
            --decision_log_jsonl "${S3_ENSEMBLE_THR}/decisions.jsonl" \
            --document_category_threshold 0.25

    # After thr025 done, run OCR+docrefine on ensemble
    mkdir -p "${S3_ENSEMBLE_OCR}"
    run_cmd "s3_ensemble_ocr" \
        ${PYTHON} semantic/run_stage3_calibration.py \
            --json_path "${JSON_PATH}" \
            --stage1_path "${ENSEMBLE_MERGED}/stage1_out.json" \
            --stage2_path "${ENSEMBLE_S2}/stage2_out.json" \
            --output_dir "${S3_ENSEMBLE_OCR}" \
            --sam_checkpoint "${SAM_CKPT}" \
            --llm_model "${MODEL_8B}" \
            --device cuda:1 \
            --llm_decoding_mode deterministic \
            --llm_max_pixels 448 \
            --calibration_mode reference_match \
            --reference_source crop \
            --support_dir "${SUPPORT_DIR}" \
            --support_json "${SUPPORT_JSON}" \
            --disable_sam \
            --proposal_score_threshold 0.35 \
            --final_score_threshold 0.35 \
            --classification_top_k 2 \
            --verbose_decisions \
            --decision_log_jsonl "${S3_ENSEMBLE_OCR}/decisions.jsonl" \
            --enable_ocr_enrichment \
            --enable_document_refine \
            --document_refine_override_shortlist \
            --document_refine_prompt_path "${DOC_REFINE_PROMPT}"

    log "=== GPU1 CHAIN COMPLETE ==="
}

# ── GPU0 chain: merge → stage2 → stage3_ensemble ─────────────────────────────

gpu0_chain() {
    log "=== GPU0 CHAIN START ==="

    # Wait for both Stage 1 stochastic runs
    wait_for_file "s1_seed42" "${S1_S1}"
    wait_for_file "s1_seed99" "${S1_S2}"

    # Merge
    mkdir -p "${ENSEMBLE_MERGED}"
    run_cmd "merge_ensemble" \
        ${PYTHON} semantic/merge_stage1_ensemble.py \
            --inputs "${S1_BASELINE}" "${S1_S1}" "${S1_S2}" \
            --output "${ENSEMBLE_MERGED}/stage1_out.json"

    # Stage 2
    mkdir -p "${ENSEMBLE_S2}"
    run_cmd "s2_ensemble" \
        ${PYTHON} semantic/run_stage2_detection.py \
            --stage1_path "${ENSEMBLE_MERGED}/stage1_out.json" \
            --output_path "${ENSEMBLE_S2}/stage2_out.json" \
            --config_path "${GDINO_CONFIG}" \
            --checkpoint_path "${GDINO_CKPT}" \
            --device cuda:0 \
            --box_threshold 0.25 \
            --text_threshold 0.25 \
            --proposal_nms_iou 0.6 \
            --max_candidates 5

    # Stage 3 baseline (reference_match, no extras)
    mkdir -p "${S3_ENSEMBLE}"
    run_cmd "s3_ensemble_baseline" \
        ${PYTHON} semantic/run_stage3_calibration.py \
            --json_path "${JSON_PATH}" \
            --stage1_path "${ENSEMBLE_MERGED}/stage1_out.json" \
            --stage2_path "${ENSEMBLE_S2}/stage2_out.json" \
            --output_dir "${S3_ENSEMBLE}" \
            --sam_checkpoint "${SAM_CKPT}" \
            --llm_model "${MODEL_8B}" \
            --device cuda:0 \
            --llm_decoding_mode deterministic \
            --llm_max_pixels 448 \
            --calibration_mode reference_match \
            --reference_source crop \
            --support_dir "${SUPPORT_DIR}" \
            --support_json "${SUPPORT_JSON}" \
            --disable_sam \
            --proposal_score_threshold 0.35 \
            --final_score_threshold 0.35 \
            --classification_top_k 2 \
            --verbose_decisions \
            --decision_log_jsonl "${S3_ENSEMBLE}/decisions.jsonl"

    log "=== GPU0 CHAIN COMPLETE ==="
}

# ── launch both chains in parallel ───────────────────────────────────────────

log "Launching GPU0 and GPU1 chains in parallel..."
gpu0_chain &
GPU0_PID=$!
gpu1_chain &
GPU1_PID=$!

wait $GPU0_PID && log "GPU0 chain exited OK" || { log "GPU0 chain FAILED"; exit 1; }
wait $GPU1_PID && log "GPU1 chain exited OK" || { log "GPU1 chain FAILED"; exit 1; }

log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results:"
log "  Baseline thr=0.25  : ${S3_BASELINE_THR}"
log "  Ensemble baseline  : ${S3_ENSEMBLE}"
log "  Ensemble thr=0.25  : ${S3_ENSEMBLE_THR}"
log "  Ensemble OCR+refine: ${S3_ENSEMBLE_OCR}"
