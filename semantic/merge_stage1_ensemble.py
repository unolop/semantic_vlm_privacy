#!/usr/bin/env python
"""Merge multiple Stage 1 outputs into an ensemble by unioning predictions.

For each image:
- null_likely = True only if ALL runs predict null (union of non-null wins)
- semantic_categories = union across runs (order: first run first, de-duplicated)
- proposal_prompts = union across runs (de-duplicated)
- support_image_paths = union across runs (de-duplicated)
- semantic_family = first non-empty value across runs
- semantic_raw_text = from first run (if present)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge Stage 1 ensemble outputs.')
    parser.add_argument('--inputs', nargs='+', required=True, help='Stage 1 output JSON paths')
    parser.add_argument('--output', required=True, help='Merged output JSON path')
    return parser.parse_args()


def merge_records(all_records: list[list[dict]]) -> list[dict]:
    # Index by image_id
    by_image: dict[int, list[dict]] = {}
    for records in all_records:
        for r in records:
            iid = r['image_id']
            by_image.setdefault(iid, [])
            by_image[iid].append(r)

    merged = []
    for iid, runs in by_image.items():
        # null only if all runs agree on null
        null_likely = all(r['null_likely'] for r in runs)

        # union categories preserving first-run order
        seen_cats: set[str] = set()
        categories: list[str] = []
        for r in runs:
            for c in r.get('semantic_categories', []):
                if c and c not in seen_cats:
                    categories.append(c)
                    seen_cats.add(c)

        # union proposal_prompts
        seen_prompts: set[str] = set()
        prompts: list[str] = []
        for r in runs:
            for p in r.get('proposal_prompts', []):
                if p and p not in seen_prompts:
                    prompts.append(p)
                    seen_prompts.add(p)

        # union support_image_paths
        seen_paths: set[str] = set()
        support_paths: list[str] = []
        for r in runs:
            for p in r.get('support_image_paths', []):
                if p and p not in seen_paths:
                    support_paths.append(p)
                    seen_paths.add(p)

        # first non-empty family
        family = next((r['semantic_family'] for r in runs if r.get('semantic_family')), '')

        record = {
            'image_id': iid,
            'query_image_path': runs[0]['query_image_path'],
            'support_image_paths': support_paths,
            'controller_mode': runs[0].get('controller_mode', 'query_only'),
            'null_policy': runs[0].get('null_policy', 'ignore'),
            'semantic_family': family,
            'semantic_categories': categories,
            'proposal_prompts': prompts,
            'null_likely': null_likely,
            'ensemble_run_count': len(runs),
            'ensemble_null_votes': sum(1 for r in runs if r['null_likely']),
        }
        merged.append(record)

    merged.sort(key=lambda r: r['image_id'])
    return merged


def main() -> None:
    args = parse_args()
    all_records = []
    configs = []
    for path in args.inputs:
        payload = json.loads(Path(path).read_text())
        all_records.append(payload['records'])
        configs.append(payload.get('config', {}))

    merged = merge_records(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'config': {'ensemble_inputs': args.inputs, 'base_config': configs[0]},
        'records': merged,
    }, ensure_ascii=False, indent=2))

    null_count = sum(1 for r in merged if r['null_likely'])
    total = len(merged)
    print(f'Merged {len(args.inputs)} runs into {total} images.')
    print(f'Null: {null_count}/{total}  Active: {total - null_count}/{total}')


if __name__ == '__main__':
    main()
