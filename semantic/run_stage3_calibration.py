#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller
from semantic.family_config import set_active_family_config
from semantic.semantic_gdino_sam import (
    DetectionCandidate,
    ProposalCalibrator,
    SamSegmenter,
    SemanticCue,
    _extract_tag,
    _save_candidate_crop,
    finalize_candidate_results,
)

DOCUMENT_CATEGORIES = {
    'bills or receipt',
    'bank statement',
    'letters with address',
    'transcript',
    'mortgage or investment report',
    'medical record document',
    'doctors prescription',
    'local newspaper',
}

DOCUMENT_REFINE_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'active' / 'semantic_document_refine.txt'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 3 calibration/finalization only.')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--stage1_path', required=True)
    parser.add_argument('--stage2_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sam_checkpoint', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--family_config', default=None)
    parser.add_argument('--calibration_mode', choices=['legacy', 'reference_match'], default='legacy')
    parser.add_argument('--reference_source', choices=['crop', 'full_image'], default='crop')
    parser.add_argument('--support_dir', default=None)
    parser.add_argument('--support_json', default=None)
    parser.add_argument('--disable_sam', action='store_true')
    parser.add_argument('--proposal_score_threshold', type=float, default=None)
    parser.add_argument(
        '--final_score_threshold',
        type=float,
        default=None,
        help='Deprecated alias for --proposal_score_threshold.',
    )
    parser.add_argument('--classification_top_k', type=int, default=None)
    parser.add_argument('--skip_null_stage3', action='store_true')
    parser.add_argument('--verbose_decisions', action='store_true')
    parser.add_argument('--decision_log_jsonl', default=None)
    parser.add_argument('--save_calibration_raw_text', action='store_true')
    parser.add_argument('--enable_document_refine', action='store_true')
    parser.add_argument('--document_refine_prompt_path', default=str(DOCUMENT_REFINE_DEFAULT_PROMPT))
    parser.add_argument(
        '--enable_document_prompt_match_fallback',
        action='store_true',
        help='Allow document branch to prefer shortlist categories matched lexically from the candidate prompt.',
    )
    parser.add_argument('--save_document_refine_raw_text', action='store_true')
    return parser.parse_args()


def build_submission_records(outputs: list[dict[str, object]], category_name_to_id: dict[str, int]) -> list[dict[str, object]]:
    submission: list[dict[str, object]] = []
    for record in outputs:
        image_id = record['image_id']
        for result in record.get('results', []):
            category_name = result.get('matched_category')
            if not category_name:
                continue
            category_id = category_name_to_id.get(category_name)
            if category_id is None:
                continue
            segmentation = result.get('segmentation', [])
            bbox = result.get('bbox', [])
            area = result.get('area')
            if area is None and len(bbox) == 4:
                area = float(bbox[2]) * float(bbox[3])
            submission.append({
                'image_id': image_id,
                'score': float(result.get('proposal_score', result.get('score', 0.0))),
                'category_id': category_id,
                'area': float(area or 0.0),
                'bbox': bbox,
                'segmentation': segmentation,
            })
    return submission


def _normalize_category_name(text: str) -> str:
    return ' '.join((text or '').strip().replace('_', ' ').lower().split())


def _resolve_stage1_category_shortlist(stage1_categories: list[str], category_names: list[str]) -> list[str]:
    normalized_allowed = {
        _normalize_category_name(category): category
        for category in category_names
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for category in stage1_categories:
        normalized = _normalize_category_name(category)
        if not normalized:
            continue
        matched = normalized_allowed.get(normalized)
        if matched is None or matched in seen:
            continue
        resolved.append(matched)
        seen.add(matched)
    return resolved


def _candidate_prompt(candidate_record: dict[str, object]) -> str:
    prompt = candidate_record.get('source_prompt') or candidate_record.get('label_text') or 'candidate'
    return str(prompt)


def _reject_reasons(
    *,
    matched_category: str | None,
    proposal_score: float,
    proposal_score_threshold: float,
    decision: object,
    reference_match_mode: bool,
) -> list[str]:
    reasons: list[str] = []
    if not matched_category:
        reasons.append('no_allowed_category_match')
    if proposal_score < proposal_score_threshold:
        reasons.append('below_proposal_score_threshold')
    if reference_match_mode:
        return reasons
    if getattr(decision, 'decision', None) is False:
        reasons.append('decision_false')
    if getattr(decision, 'object_valid', None) is False:
        reasons.append('object_valid_false')
    if getattr(decision, 'family_match', None) is False:
        reasons.append('family_match_false')
    if getattr(decision, 'exact_match', None) is False:
        reasons.append('exact_match_false')
    return reasons


def _format_bbox(values: list[float]) -> str:
    return '[' + ', '.join(f'{value:.1f}' for value in values) + ']'


def _fallback_reference_category(candidate_prompt: str, allowed_categories: list[str]) -> tuple[str | None, str | None]:
    if len(allowed_categories) == 1:
        return allowed_categories[0], 'single_allowed_category'
    normalized_prompt = _normalize_category_name(candidate_prompt)
    if not normalized_prompt:
        return None, None
    matches = [
        category
        for category in allowed_categories
        if _normalize_category_name(category) in normalized_prompt
    ]
    if not matches:
        return None, None
    matches.sort(key=lambda category: len(_normalize_category_name(category)), reverse=True)
    return matches[0], 'candidate_prompt_match'


def _is_document_category(category: str | None) -> bool:
    return _normalize_category_name(category or '') in DOCUMENT_CATEGORIES


def _normalize_document_refine_category(text: str) -> str | None:
    normalized = _normalize_category_name(text)
    if not normalized or normalized == 'unknown':
        return None
    for category in DOCUMENT_CATEGORIES:
        if _normalize_category_name(category) == normalized:
            return category
    return None


def _normalize_yes_no(text: str) -> bool | None:
    normalized = _normalize_category_name(text)
    if normalized == 'yes':
        return True
    if normalized == 'no':
        return False
    return None


def _normalize_evidence(raw_text: str) -> list[str]:
    evidence: list[str] = []
    for item in raw_text.split(','):
        token = ' '.join(item.strip().lower().split())
        if not token or token in {'none', 'unknown', 'unreadable', 'n/a'}:
            continue
        if any(char.isdigit() for char in token) and len(token) > 8:
            continue
        if len(token) > 32:
            continue
        if token not in evidence:
            evidence.append(token)
    return evidence[:8]


def _should_refine_document_candidate(route_type: str, matched_category: str | None) -> bool:
    return _normalize_category_name(route_type) == 'document' and _is_document_category(matched_category)


def _is_document_route(route_type: str) -> bool:
    return _normalize_category_name(route_type) == 'document'


def _prune_document_candidates_top1(kept_candidates: list[dict[str, object]], route_type: str) -> list[dict[str, object]]:
    if not _is_document_route(route_type) or not kept_candidates:
        return kept_candidates
    top_item = max(kept_candidates, key=lambda item: float(item['proposal_score']))
    return [top_item]


def _refine_document_category(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
) -> dict[str, object]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        raw_text = client.generate(crop_path, instruction=prompt_text)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    document_present = _normalize_yes_no(_extract_tag(raw_text, 'document_present'))
    refined_category = _normalize_document_refine_category(_extract_tag(raw_text, 'category'))
    evidence = _normalize_evidence(_extract_tag(raw_text, 'evidence'))
    return {
        'raw_text': raw_text,
        'document_present': document_present,
        'category': refined_category,
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    if args.proposal_score_threshold is None:
        args.proposal_score_threshold = (
            args.final_score_threshold
            if args.final_score_threshold is not None
            else 0.30
        )
    args.final_score_threshold = args.proposal_score_threshold
    set_active_family_config(args.family_config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    category_name_to_id = {cat['name'].replace('_', ' '): cat['id'] for cat in dataset.get('categories', [])}
    category_names = list(category_name_to_id.keys())

    stage1_payload = json.loads(Path(args.stage1_path).read_text())
    stage2_payload = json.loads(Path(args.stage2_path).read_text())
    stage1_by_image = {int(record['image_id']): record for record in stage1_payload['records']}
    stage2_records = stage2_payload['records']
    document_refine_prompt = None
    if args.enable_document_refine:
        document_refine_prompt = Path(args.document_refine_prompt_path).read_text().strip()

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=128,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )
    if args.calibration_mode == 'reference_match' and (not args.support_json or not args.support_dir):
        raise ValueError('reference_match mode requires --support-json and --support-dir')
    calibrator = ProposalCalibrator(
        model_path=args.llm_model,
        max_new_tokens=128,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        calibration_mode=args.calibration_mode,
        support_json_path=args.support_json,
        support_dir=args.support_dir,
        reference_source=args.reference_source,
        client=shared_vlm,
    )
    segmenter = None if args.disable_sam else SamSegmenter(args.sam_checkpoint, device=args.device)
    decision_log_path = Path(args.decision_log_jsonl).resolve() if args.decision_log_jsonl else None
    if decision_log_path is not None:
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
    decision_log_fh = decision_log_path.open('w') if decision_log_path else None
    outputs: list[dict[str, object]] = []
    progress = tqdm(stage2_records, desc='stage3 calibration', unit='image')
    try:
        for detection_record in progress:
            image_id = int(detection_record['image_id'])
            semantic_record = stage1_by_image[image_id]
            semantic = SemanticCue(
                family=str(semantic_record['semantic_family']),
                route_type=str(semantic_record.get('route_type', '')),
                categories=list(semantic_record.get('semantic_categories', [])),
                proposal_prompts=list(semantic_record['proposal_prompts']),
                null_likely=bool(semantic_record['null_likely']),
            )
            category_shortlist = _resolve_stage1_category_shortlist(semantic.categories, category_names)
            if not category_shortlist and not semantic.null_likely:
                category_shortlist = list(category_names)
            kept_candidates: list[dict[str, object]] = []
            calibration_logs: list[dict[str, object]] = []

            raw_candidates = list(detection_record.get('proposal_candidates', []))
            if args.skip_null_stage3 and semantic.null_likely:
                outputs.append({
                    'image_id': image_id,
                    'query_image_path': detection_record['query_image_path'],
                    'support_image_paths': detection_record.get('support_image_paths', []),
                    'controller_mode': detection_record['controller_mode'],
                    'semantic_family': semantic.family,
                    'route_type': semantic.route_type,
                    'semantic_categories': semantic.categories,
                    'proposal_prompts': semantic.proposal_prompts,
                    'null_likely': semantic.null_likely,
                    'null_policy': detection_record['null_policy'],
                    'proposal_candidates': raw_candidates,
                    'category_shortlist': [],
                    'calibration_logs': [],
                    'rerank_logs': [],
                    'results': [],
                })
                progress.set_postfix({
                    'image_id': image_id,
                    'selected': 0,
                })
                message = (
                    f"Stage3 image_id={image_id} family={semantic.family} categories={semantic.categories} "
                    f"candidates={len(raw_candidates)} selected=0 skipped_null_stage3=1"
                )
                tqdm.write(message)
                if decision_log_fh is not None:
                    decision_log_fh.write(json.dumps({
                        'image_id': image_id,
                        'event': 'skip_null_stage3',
                        'semantic_family': semantic.family,
                        'route_type': semantic.route_type,
                        'semantic_categories': semantic.categories,
                        'proposal_prompts': semantic.proposal_prompts,
                        'candidate_count': len(raw_candidates),
                    }, ensure_ascii=False) + '\n')
                    decision_log_fh.flush()
                continue
            candidates_for_scoring = (
                raw_candidates[:args.classification_top_k]
                if args.classification_top_k
                else raw_candidates
            )
            for candidate_index, candidate_record in enumerate(candidates_for_scoring):
                candidate_prompt = _candidate_prompt(candidate_record)
                proposal_score = float(candidate_record['score'])
                candidate = DetectionCandidate(
                    score=proposal_score,
                    label_text=candidate_prompt,
                    category_id=-1,
                    xyxy=[float(value) for value in candidate_record['bbox_xyxy']],
                )
                matched_category = None
                stage3_branch = 'reference_match'
                decision = None
                decision_category = None
                decision_label = None
                decision_reason = None
                decision_score = None
                decision_decision = None
                decision_object_valid = None
                decision_family_match = None
                decision_exact_match = None
                category_fallback_reason = None
                document_refine_category = None
                document_refine_evidence: list[str] = []
                document_refine_override = False
                document_refine_raw_text = None
                document_present = None
                pre_refine_category = None

                if (
                    _is_document_route(semantic.route_type)
                    and args.enable_document_refine
                    and document_refine_prompt
                ):
                    stage3_branch = 'document_refine'
                    document_refine = _refine_document_category(
                        client=shared_vlm,
                        prompt_text=document_refine_prompt,
                        query_image_path=str(detection_record['query_image_path']),
                        candidate=candidate,
                    )
                    document_present = document_refine['document_present']
                    document_refine_category = document_refine['category']
                    document_refine_evidence = list(document_refine['evidence'])
                    document_refine_raw_text = str(document_refine['raw_text'])
                    normalized_shortlist = {
                        _normalize_category_name(name): name for name in category_shortlist
                    }
                    normalized_refine = _normalize_category_name(document_refine_category or '')
                    if category_shortlist:
                        matched_category = str(category_shortlist[0])
                        category_fallback_reason = 'document_route_prior'
                    prompt_matched_category = None
                    prompt_fallback_reason = None
                    if args.enable_document_prompt_match_fallback and category_shortlist:
                        prompt_matched_category, prompt_fallback_reason = _fallback_reference_category(
                            candidate_prompt=candidate_prompt,
                            allowed_categories=category_shortlist,
                        )
                    if (
                        document_refine_category
                        and document_refine_evidence
                        and normalized_refine in normalized_shortlist
                    ):
                        matched_category = normalized_shortlist[normalized_refine]
                        category_fallback_reason = None
                    elif prompt_matched_category is not None:
                        matched_category = prompt_matched_category
                        category_fallback_reason = f'document_{prompt_fallback_reason}'
                    elif category_shortlist and document_refine_category and document_refine_evidence:
                        category_fallback_reason = 'refine_out_of_shortlist'
                else:
                    decision = calibrator.score_candidate(
                        support_image_paths=detection_record.get('support_image_paths', []),
                        query_image_path=str(detection_record['query_image_path']),
                        candidate=candidate,
                        semantic=semantic,
                        allowed_categories=category_shortlist,
                    )
                    decision_category = decision.category
                    decision_label = decision.label
                    decision_reason = decision.reason
                    decision_score = decision.score
                    decision_decision = decision.decision
                    decision_object_valid = decision.object_valid
                    decision_family_match = decision.family_match
                    decision_exact_match = decision.exact_match
                    normalized_allowed = {
                        _normalize_category_name(name): name for name in category_shortlist
                    }
                    normalized_decision = _normalize_category_name(decision.category)
                    if normalized_decision in normalized_allowed:
                        matched_category = normalized_allowed[normalized_decision]
                    if matched_category is None and args.calibration_mode == 'reference_match':
                        matched_category, category_fallback_reason = _fallback_reference_category(
                            candidate_prompt=candidate_prompt,
                            allowed_categories=category_shortlist,
                        )
                    pre_refine_category = matched_category
                if stage3_branch == 'document_refine' or args.calibration_mode == 'reference_match':
                    accepted = bool(matched_category and proposal_score >= args.proposal_score_threshold)
                else:
                    accepted = bool(
                        matched_category
                        and proposal_score >= args.proposal_score_threshold
                        and (decision.decision is not False)
                        and (decision.object_valid is not False)
                        and (decision.family_match is not False)
                        and (decision.exact_match is not False)
                    )
                reject_reasons = [] if accepted else _reject_reasons(
                    matched_category=matched_category,
                    proposal_score=proposal_score,
                    proposal_score_threshold=args.proposal_score_threshold,
                    decision=decision,
                    reference_match_mode=(stage3_branch == 'document_refine' or args.calibration_mode == 'reference_match'),
                )
                log_entry = {
                    'stage3_branch': stage3_branch,
                    'candidate_index': candidate_index,
                    'proposal_score': proposal_score,
                    # Backward-compatible alias for old analysis scripts.
                    'candidate_score': proposal_score,
                    'candidate_label_text': str(candidate_record.get('label_text', '')),
                    'candidate_source_prompt': str(candidate_record.get('source_prompt', candidate_prompt)),
                    'candidate_bbox_xyxy': list(candidate.xyxy),
                    'calibration_decision': decision_decision,
                    'object_valid': decision_object_valid,
                    'family_match': decision_family_match,
                    'exact_match': decision_exact_match,
                    'calibration_score': decision_score,
                    'calibration_label': decision_label,
                    'calibration_category': decision_category,
                    'calibration_reason': decision_reason,
                    'pre_refine_category': pre_refine_category,
                    'document_refine_category': document_refine_category,
                    'document_present': document_present,
                    'document_refine_evidence': document_refine_evidence,
                    'document_refine_override': document_refine_override,
                    'matched_category': matched_category,
                    'category_fallback_reason': category_fallback_reason,
                    'accepted': accepted,
                    'reject_reasons': reject_reasons,
                    'proposal_score_threshold': args.proposal_score_threshold,
                    # Backward-compatible alias. This is not a semantic confidence.
                    'final_score': proposal_score,
                }
                if args.save_calibration_raw_text and decision is not None:
                    log_entry['calibration_raw_text'] = decision.raw_text
                if args.save_document_refine_raw_text and document_refine_raw_text is not None:
                    log_entry['document_refine_raw_text'] = document_refine_raw_text
                calibration_logs.append(log_entry)
                if args.verbose_decisions:
                    status = 'ACCEPT' if accepted else 'REJECT'
                    reason_text = ','.join(reject_reasons) if reject_reasons else '-'
                    refine_text = (
                        f" doc_refine={document_refine_category!r} evidence={document_refine_evidence}"
                        if document_refine_category or document_refine_evidence
                        else ''
                    )
                    tqdm.write(
                        f"  cand#{candidate_index} {status} proposal_score={proposal_score:.3f} "
                        f"branch={stage3_branch} "
                        f"prompt={candidate_prompt!r} bbox={_format_bbox(list(candidate.xyxy))} "
                        f"vlm_category={decision_category!r} matched={matched_category!r} "
                        f"fallback={category_fallback_reason or '-'} reasons={reason_text}{refine_text}"
                    )
                if decision_log_fh is not None:
                    jsonl_entry = {
                        'image_id': image_id,
                        'event': 'candidate_decision',
                        'semantic_family': semantic.family,
                        'route_type': semantic.route_type,
                        'semantic_categories': semantic.categories,
                        'proposal_prompts': semantic.proposal_prompts,
                        'category_shortlist': category_shortlist,
                        **log_entry,
                    }
                    decision_log_fh.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                    decision_log_fh.flush()
                if not accepted:
                    continue
                kept_candidates.append({
                    'detection': DetectionCandidate(
                        score=proposal_score,
                        label_text=matched_category,
                        category_id=category_name_to_id.get(matched_category, -1),
                        xyxy=list(candidate.xyxy),
                    ),
                    'proposal_score': proposal_score,
                    # Backward-compatible alias for old output readers.
                    'detector_score': proposal_score,
                    'calibration_category': decision_category,
                    'matched_category': matched_category,
                    # Backward-compatible alias. This is currently equal to proposal_score.
                    'final_score': proposal_score,
                })

            kept_candidates.sort(key=lambda item: item['proposal_score'], reverse=True)
            kept_candidates = _prune_document_candidates_top1(kept_candidates, semantic.route_type)
            finalized = finalize_candidate_results(
                segmenter=segmenter,
                query_image_path=str(detection_record['query_image_path']),
                candidates=[item['detection'] for item in kept_candidates],
                image_id=image_id,
                use_sam=not args.disable_sam,
            )
            for result, item in zip(finalized, kept_candidates):
                result['proposal_score'] = item['proposal_score']
                result['detector_score'] = item['detector_score']
                result['calibration_category'] = item['calibration_category']
                result['matched_category'] = item['matched_category']
                result['final_score'] = item['final_score']

            outputs.append({
                'image_id': image_id,
                'query_image_path': detection_record['query_image_path'],
                'support_image_paths': detection_record.get('support_image_paths', []),
                'controller_mode': detection_record['controller_mode'],
                'semantic_family': semantic.family,
                'route_type': semantic.route_type,
                'semantic_categories': semantic.categories,
                'proposal_prompts': semantic.proposal_prompts,
                'null_likely': semantic.null_likely,
                'null_policy': detection_record['null_policy'],
                'proposal_candidates': raw_candidates,
                'category_shortlist': category_shortlist,
                'calibration_logs': calibration_logs,
                'rerank_logs': calibration_logs,
                'results': finalized,
            })
            progress.set_postfix({
                'image_id': image_id,
                'selected': len(finalized),
            })
            rejected_count = len(calibration_logs) - len(kept_candidates)
            tqdm.write(
                f"Stage3 image_id={image_id} family={semantic.family} categories={semantic.categories} "
                f"shortlist={category_shortlist} candidates={len(raw_candidates)} "
                f"scored={len(candidates_for_scoring)} selected={len(finalized)} rejected={rejected_count}"
            )
    finally:
        progress.close()
        if decision_log_fh is not None:
            decision_log_fh.close()

    submission_records = build_submission_records(outputs, category_name_to_id)
    (output_dir / 'semantic_pipeline_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'query_submission.json').write_text(json.dumps(submission_records, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    tqdm.write(f"Saved Stage 3 outputs to: {output_dir / 'semantic_pipeline_results.json'}")
    tqdm.write(f"Saved Stage 3 submission to: {output_dir / 'query_submission.json'}")


if __name__ == '__main__':
    main()
