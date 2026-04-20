#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller, release_torch_runtime
from common.vlm import _resolve_device_map
from semantic.family_config import get_active_family_config_path, get_family_names, set_active_family_config
from semantic.semantic_gdino_sam import SemanticController, DOCUMENT_TEXT_PROMPT, _parse_semantic_cue, _extract_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 1 query-only semantic inference.')
    parser.add_argument('--query_dir', required=True)
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm_max_new_tokens', type=int, default=128)
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--family_config', '--family-config', dest='family_config', default=None)
    parser.add_argument('--query_prompt_path', default=None)
    parser.add_argument('--null_policy', choices=['strict', 'skip', 'ignore'], default='ignore')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image_id', type=int, default=None)
    parser.add_argument('--runtime_stats_jsonl', default=None)
    parser.add_argument('--save_raw_text', action='store_true')
    parser.add_argument('--cuda_cleanup_interval', type=int, default=0)
    parser.add_argument('--support_dir', default=None)
    parser.add_argument('--support_json', default=None)
    parser.add_argument('--enable_ocr_enrichment', action='store_true',
                        help='Run OCR on each image before Stage 1 VLM call and append extracted text to the prompt.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_active_family_config(args.family_config)
    query_prompt_text = None
    if args.query_prompt_path:
        query_prompt_text = Path(args.query_prompt_path).read_text().strip()

    # Build category → support image paths lookup
    support_paths_by_category: dict[str, list[str]] = {}
    if args.support_json and args.support_dir:
        support_data = json.loads(Path(args.support_json).read_text())
        support_cat_names = {c['id']: c['name'] for c in support_data['categories']}
        support_img_files = {img['id']: img['file_name'] for img in support_data['images']}
        for ann in support_data['annotations']:
            cat = support_cat_names[ann['category_id']]
            fpath = str((Path(args.support_dir) / support_img_files[ann['image_id']]).resolve())
            support_paths_by_category.setdefault(cat, [])
            if fpath not in support_paths_by_category[cat]:
                support_paths_by_category[cat].append(fpath)

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )
    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        query_only_instruction=query_prompt_text,
        client=shared_vlm,
    )

    request_config = getattr(shared_vlm, 'request_config', None)
    runtime_config = {
        'parsed_args': {
            'llm_model': args.llm_model,
            'device': args.device,
            'llm_max_new_tokens': args.llm_max_new_tokens,
            'llm_decoding_mode': args.llm_decoding_mode,
            'llm_seed': args.llm_seed,
            'llm_max_pixels': args.llm_max_pixels,
            'family_config': get_active_family_config_path(),
        },
        'vlm_caller': {
            'model_path': shared_vlm.model_path,
            'max_new_tokens': shared_vlm.max_new_tokens,
            'decoding_mode': shared_vlm.decoding_mode,
            'seed': shared_vlm.seed,
            'max_pixels': shared_vlm.max_pixels,
            'device': shared_vlm.device,
            'resolved_device_map': _resolve_device_map(shared_vlm.device),
        },
        'family_spec': {
            'config_path': get_active_family_config_path(),
            'family_names': get_family_names(),
        },
        'request_config': {
            'max_tokens': getattr(request_config, 'max_tokens', None),
            'temperature': getattr(request_config, 'temperature', None),
            'top_k': getattr(request_config, 'top_k', None),
            'top_p': getattr(request_config, 'top_p', None),
            'seed': getattr(request_config, 'seed', None),
            'repetition_penalty': getattr(request_config, 'repetition_penalty', None),
        },
    }
    print('[stage1] runtime_config')
    print(pformat(runtime_config, sort_dicts=False))

    outputs: list[dict[str, object]] = []
    runtime_stats_path = Path(args.runtime_stats_jsonl).resolve() if args.runtime_stats_jsonl else None
    runtime_stats_fh = runtime_stats_path.open('w') if runtime_stats_path else None
    progress = tqdm(images, desc='stage1 semantic', unit='image')
    try:
        for index, image_info in enumerate(progress, start=1):
            query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
            image_start = time.perf_counter()
            if args.enable_ocr_enrichment:
                # Run OCR pass first; append any extracted text to the Stage 1 prompt
                ocr_raw = shared_vlm.generate(query_image_path, instruction=DOCUMENT_TEXT_PROMPT)
                ocr_text    = _extract_tag(ocr_raw, 'text').strip()
                ocr_hint    = _extract_tag(ocr_raw, 'document_hint').strip()
                has_ocr = ocr_text and ocr_text.lower() not in ('none', '')
                if has_ocr:
                    ocr_context = (
                        f'\n\nOCR hint from image (use only if relevant to category selection):\n'
                        f'Type hint: {ocr_hint}\n'
                        f'Visible text: {ocr_text}'
                    )
                    enriched_instruction = controller.query_only_instruction + ocr_context
                    stage1_raw = shared_vlm.generate(query_image_path, instruction=enriched_instruction)
                    semantic = _parse_semantic_cue(stage1_raw)
                    raw_text = stage1_raw if args.save_raw_text else None
                else:
                    if args.save_raw_text:
                        semantic, raw_text = controller.infer_query_only_with_raw(query_image_path)
                    else:
                        semantic = controller.infer_query_only(query_image_path)
                        raw_text = None
            elif args.save_raw_text:
                semantic, raw_text = controller.infer_query_only_with_raw(query_image_path)
            else:
                semantic = controller.infer_query_only(query_image_path)
                raw_text = None
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed_sec = time.perf_counter() - image_start
            # Resolve support images for predicted categories
            support_image_paths: list[str] = []
            if support_paths_by_category:
                seen: set[str] = set()
                for cat in semantic.categories:
                    for path in support_paths_by_category.get(cat, []):
                        if path not in seen:
                            support_image_paths.append(path)
                            seen.add(path)

            record = {
                'image_id': image_info['id'],
                'query_image_path': query_image_path,
                'support_image_paths': support_image_paths,
                'controller_mode': 'query_only',
                'null_policy': args.null_policy,
                'semantic_family': semantic.family,
                'semantic_categories': semantic.categories,
                'proposal_prompts': semantic.proposal_prompts,
                'null_likely': semantic.null_likely,
            }
            if raw_text is not None:
                record['semantic_raw_text'] = raw_text
            runtime_stats = {
                'image_index': index,
                'image_id': image_info['id'],
                'elapsed_sec': round(elapsed_sec, 3),
                'controller_mode': 'query_only',
                'support_image_count': 0,
                'semantic_family': semantic.family,
                'semantic_categories': semantic.categories,
                'null_likely': semantic.null_likely,
                'prompt_count': len(semantic.proposal_prompts),
            }
            if raw_text is not None:
                runtime_stats['raw_text_char_len'] = len(raw_text)
                runtime_stats['raw_text_word_len'] = len(raw_text.split())
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                runtime_stats['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1)
                torch.cuda.reset_peak_memory_stats(args.device)
            outputs.append(record)
            progress.set_postfix({
                'image_id': image_info['id'],
                'categories': ','.join(semantic.categories[:2]) or semantic.family or '-',
                'null': semantic.null_likely,
                'sec': f'{elapsed_sec:.1f}',
            })
            message = (
                f"Stage1 image_id={image_info['id']} categories={semantic.categories or [semantic.family]} "
                f"null={semantic.null_likely} prompts={len(semantic.proposal_prompts)} "
                f"elapsed_sec={elapsed_sec:.1f} mode=query_only support_images=0"
            )
            if 'gpu_memory_reserved_mb' in runtime_stats:
                message += (
                    f" gpu_alloc_mb={runtime_stats['gpu_memory_allocated_mb']:.1f}"
                    f" gpu_reserved_mb={runtime_stats['gpu_memory_reserved_mb']:.1f}"
                    f" gpu_peak_mb={runtime_stats['gpu_max_memory_allocated_mb']:.1f}"
                )
            tqdm.write(message)
            if runtime_stats_fh is not None:
                runtime_stats_fh.write(json.dumps(runtime_stats) + '\n')
                runtime_stats_fh.flush()
            if args.cuda_cleanup_interval > 0 and index % args.cuda_cleanup_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        progress.close()
    finally:
        if runtime_stats_fh is not None:
            runtime_stats_fh.close()

    payload = {
        'config': vars(args),
        'records': outputs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tqdm.write(f'Saved Stage 1 outputs to: {output_path}')


if __name__ == '__main__':
    main()
