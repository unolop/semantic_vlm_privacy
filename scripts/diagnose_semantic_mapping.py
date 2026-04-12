"""
Semantic family diagnosis — runs only the VLM semantic stage (no GDino, no calibration).
For each image, records family/cue/null and scores alignment against GT category.

Usage:
    python scripts/diagnose_semantic_mapping.py \
        --query-dir data/vizwiz_object_localization/query_images \
        --json-path data/vizwiz_object_localization/dev_pseudo_label_coco.json \
        --llm-model /path/to/model \
        --output-json outputs_postpull/semantic_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic.semantic_gdino_sam import (
    FAMILY_CATEGORY_MAP,
    SemanticController,
    _build_category_shortlist,
    _normalize_phrase,
    SemanticCue,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--query-dir', required=True)
    p.add_argument('--json-path', required=True)
    p.add_argument('--llm-model', required=True)
    p.add_argument('--output-json', required=True)
    p.add_argument('--llm-max-new-tokens', type=int, default=256)
    p.add_argument('--llm-max-pixels', type=int, default=448)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = json.loads(Path(args.json_path).read_text())
    cats = {c['id']: c['name'].replace('_', ' ') for c in dataset['categories']}
    gt_by_image = {a['image_id']: cats[a['category_id']] for a in dataset.get('annotations', [])}
    images = dataset['images']
    if args.limit:
        images = images[:args.limit]

    all_categories = [c['name'].replace('_', ' ') for c in dataset['categories']]

    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        max_pixels=args.llm_max_pixels,
    )

    records = []
    per_category: dict[str, dict] = {c: {'total': 0, 'null': 0, 'map_hit': 0, 'correct_shortlist': 0, 'families': []} for c in all_categories}

    for img in images:
        query_path = str((Path(args.query_dir) / img['file_name']).resolve())
        gt_category = gt_by_image.get(img['id'])

        cue: SemanticCue = controller.infer_query_only(query_path)
        norm_family = _normalize_phrase(cue.family)
        shortlist = _build_category_shortlist(cue, all_categories)
        map_hit = norm_family in FAMILY_CATEGORY_MAP
        correct_shortlist = gt_category in shortlist if gt_category else None

        record = {
            'image_id': img['id'],
            'file_name': img['file_name'],
            'gt_category': gt_category,
            'semantic_family': cue.family,
            'norm_family': norm_family,
            'proposal_prompts': cue.proposal_prompts,
            'null_likely': cue.null_likely,
            'map_hit': map_hit,
            'shortlist': shortlist,
            'shortlist_size': len(shortlist),
            'correct_shortlist': correct_shortlist,
        }
        records.append(record)

        if gt_category:
            per_category[gt_category]['total'] += 1
            per_category[gt_category]['families'].append(cue.family)
            if cue.null_likely:
                per_category[gt_category]['null'] += 1
            if map_hit:
                per_category[gt_category]['map_hit'] += 1
            if correct_shortlist:
                per_category[gt_category]['correct_shortlist'] += 1

        status = 'NULL' if cue.null_likely else ('HIT' if map_hit else 'MISS')
        sl_info = f'shortlist={len(shortlist)}' if not map_hit else f'shortlist={shortlist}'
        correct_str = f'correct={correct_shortlist}' if gt_category else 'no-gt'
        print(f"[{status:4s}] {img['file_name']:<12} gt={gt_category or '?':<35} family={cue.family!r:<35} {sl_info} {correct_str}")

    # Summary
    total = len(records)
    with_gt = [r for r in records if r['gt_category']]
    null_count = sum(1 for r in records if r['null_likely'])
    map_hits = sum(1 for r in records if r['map_hit'])
    correct_shortlists = sum(1 for r in with_gt if r['correct_shortlist'])
    wrong_nulls = sum(1 for r in with_gt if r['null_likely'])

    print('\n' + '='*80)
    print(f'SUMMARY  ({total} images, {len(with_gt)} with GT)')
    print(f'  Family map hit rate:        {map_hits}/{total} ({map_hits/total:.1%})')
    print(f'  Correct shortlist (has GT): {correct_shortlists}/{len(with_gt)} ({correct_shortlists/len(with_gt):.1%})')
    print(f'  VLM null rate:              {null_count}/{total} ({null_count/total:.1%})')
    print(f'  Wrong nulls (has GT):       {wrong_nulls}/{len(with_gt)} ({wrong_nulls/len(with_gt):.1%})')
    print()
    print(f'{"GT Category":<40} {"total":>6} {"null":>6} {"map_hit":>8} {"correct_sl":>11} {"common_families"}')
    print('-'*120)
    for cat, stats in sorted(per_category.items(), key=lambda x: -x[1]['total']):
        if stats['total'] == 0:
            continue
        from collections import Counter
        top_families = Counter(stats['families']).most_common(3)
        fam_str = ' | '.join(f'{f!r}({c})' for f, c in top_families)
        print(f'{cat:<40} {stats["total"]:>6} {stats["null"]:>6} {stats["map_hit"]:>8} {stats["correct_shortlist"]:>11}   {fam_str}')

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'summary': {
            'total': total, 'with_gt': len(with_gt),
            'map_hit_rate': map_hits / total,
            'correct_shortlist_rate': correct_shortlists / len(with_gt) if with_gt else 0,
            'null_rate': null_count / total,
            'wrong_null_rate': wrong_nulls / len(with_gt) if with_gt else 0,
        },
        'per_category': per_category,
        'records': records,
    }, indent=2))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
