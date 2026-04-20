#!/usr/bin/env python3
"""Expand Stage 1 proposal_prompts for document images to include all document category prompts.

For any image where Stage 1 predicted at least one document category:
  - proposal_prompts  → union of ALL document category prompts (broad GDino search)
  - semantic_categories → UNCHANGED (Stage 3 still uses the narrow original shortlist)

This separates detection (broad) from classification (narrow):
  Stage 2 finds document boxes regardless of exact type.
  Stage 3 still maps to the original Stage 1 category prediction.

Usage:
  python expand_stage1_doc_prompts.py --input stage1_out.json --output stage1_doc_expanded.json
"""
import argparse
import json
from pathlib import Path

DOCUMENT_CATEGORIES = {
    'bank_statement', 'bills_or_receipt', 'business_card', 'credit_or_debit_card',
    'doctors_prescription', 'letters_with_address', 'local_newspaper',
    'medical_record_document', 'mortgage_or_investment_report', 'transcript',
}

# Canonical document prompts derived from Stage 1 outputs across the dataset
DOCUMENT_PROMPTS_BY_CAT = {
    'bank_statement':              ['bank statement', 'financial statement', 'transaction table'],
    'bills_or_receipt':            ['receipt', 'bill', 'invoice', 'financial document', 'paper with text and numbers'],
    'business_card':               ['business card', 'small rectangular card'],
    'credit_or_debit_card':        ['credit card', 'debit card'],
    'doctors_prescription':        ['prescription form', 'pharmacy order form', 'prescription bottle'],
    'letters_with_address':        ['letter with address', 'handwritten letter', 'typed address', 'folded paper'],
    'local_newspaper':             ['newspaper', 'newspaper with headlines'],
    'medical_record_document':     ['medical record', 'patient medical record', 'medical form', 'patient information'],
    'mortgage_or_investment_report': ['mortgage statement', 'quarterly mortgage statement', 'investment report'],
    'transcript':                  ['academic transcript', 'printed form', 'transcript'],
}

ALL_DOC_PROMPTS: list[str] = []
_seen: set[str] = set()
for prompts in DOCUMENT_PROMPTS_BY_CAT.values():
    for p in prompts:
        if p not in _seen:
            ALL_DOC_PROMPTS.append(p)
            _seen.add(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    records = payload['records']

    expanded = 0
    for r in records:
        if r.get('null_likely'):
            continue
        cats = set(r.get('semantic_categories', []))
        if not (cats & DOCUMENT_CATEGORIES):
            continue  # not a document image — leave unchanged

        original_prompts = r.get('proposal_prompts', [])
        seen = set(original_prompts)
        extra = [p for p in ALL_DOC_PROMPTS if p not in seen]
        if extra:
            r['proposal_prompts'] = original_prompts + extra
            expanded += 1
        # semantic_categories intentionally unchanged

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    doc_images = sum(1 for r in records
                     if not r.get('null_likely') and set(r.get('semantic_categories', [])) & DOCUMENT_CATEGORIES)
    print(f'Document images found : {doc_images}')
    print(f'Prompts expanded      : {expanded}')
    print(f'All-doc prompt count  : {len(ALL_DOC_PROMPTS)}')
    print(f'Output                : {args.output}')


if __name__ == '__main__':
    main()
