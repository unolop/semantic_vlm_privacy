#!/usr/bin/env python3
"""Expand Stage 1 semantic_categories shortlist before Stage 3.

Two modes:
  --mode confusion_pairs  : add known visually-confused counterparts
  --mode all_categories   : add all 16 GT categories to every non-null image

Usage:
  python expand_stage1_shortlist.py --input stage1_out.json --output stage1_expanded.json --mode confusion_pairs
"""
import argparse
import json
from pathlib import Path

# Known visually confused category pairs (underscore format matching Stage 1 output).
# Both directions are handled: if either side is in the shortlist, add the other.
CONFUSION_PAIRS = [
    ('bank_statement',              'bills_or_receipt'),
    ('medical_record_document',     'doctors_prescription'),
    ('mortgage_or_investment_report', 'bills_or_receipt'),
    ('tattoo_sleeve',               'condom_box'),
    ('pregnancy_test',              'pregnancy_test_box'),
]

ALL_CATEGORIES = [
    'bank_statement', 'bills_or_receipt', 'business_card', 'condom_box',
    'condom_with_plastic_bag', 'credit_or_debit_card', 'doctors_prescription',
    'empty_pill_bottle', 'letters_with_address', 'local_newspaper',
    'medical_record_document', 'mortgage_or_investment_report',
    'pregnancy_test', 'pregnancy_test_box', 'tattoo_sleeve', 'transcript',
]


def expand_confusion_pairs(categories: list[str]) -> list[str]:
    result = list(categories)
    seen = set(categories)
    for a, b in CONFUSION_PAIRS:
        if a in seen and b not in seen:
            result.append(b)
            seen.add(b)
        elif b in seen and a not in seen:
            result.append(a)
            seen.add(a)
    return result


def expand_all(categories: list[str]) -> list[str]:
    seen = set(categories)
    result = list(categories)
    for c in ALL_CATEGORIES:
        if c not in seen:
            result.append(c)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode',   choices=['confusion_pairs', 'all_categories'],
                        default='confusion_pairs')
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    records = payload['records']

    expand_fn = expand_confusion_pairs if args.mode == 'confusion_pairs' else expand_all

    expanded = 0
    for r in records:
        if r.get('null_likely'):
            continue
        original = r.get('semantic_categories', [])
        updated  = expand_fn(original)
        if updated != original:
            r['semantic_categories'] = updated
            expanded += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    total_non_null = sum(1 for r in records if not r.get('null_likely'))
    print(f'Mode    : {args.mode}')
    print(f'Records : {len(records)} total, {total_non_null} non-null')
    print(f'Expanded: {expanded} images had shortlist augmented')
    print(f'Output  : {args.output}')


if __name__ == '__main__':
    main()
