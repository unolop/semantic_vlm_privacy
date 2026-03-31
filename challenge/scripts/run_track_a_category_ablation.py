#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM2SEG_DIR = Path(os.environ.get('LLM2SEG_DIR', REPO_ROOT / 'third_party' / 'LLM2Seg'))
if str(LLM2SEG_DIR) not in sys.path:
    sys.path.insert(0, str(LLM2SEG_DIR))

from call_vlm import SwiftVLMCaller  # noqa: E402


QUERY_ONLY_TEMPLATE = """You are given one query image.

Task:
- Choose the single category that best matches the main target object in the query image.
- Choose only from the candidate list below.
- Return only one category name from the list.
- Do not explain your reasoning.

Candidate categories:
{category_list}

Output format:
<output>category name</output>
"""


QUERY_SUPPORT_TEMPLATE = """You are given 16 support images followed by one query image.

Image order:
- Images 1 to {num_support}: support images
- Image {query_index}: query image

Support set mapping:
{support_mapping}

Task:
- Compare the query image against the support images.
- Choose the single support category that best matches the target object in the query image.
- Use the support images as the primary reference, not world knowledge alone.
- Return only one category name from the provided support set.
- Do not explain your reasoning.

Output format:
<output>category name</output>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track A category-inference ablation: query-only vs query+support16."
    )
    parser.add_argument("--support-json", required=True, help="COCO support json")
    parser.add_argument("--support-image-dir", required=True, help="Support image directory")
    parser.add_argument("--query-json", required=True, help="COCO query json with GT")
    parser.add_argument("--query-image-dir", required=True, help="Query image directory")
    parser.add_argument("--llm-model", required=True, help="Local Qwen model path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--split-config", default=None, help="Optional pseudo_novel_3split.json")
    parser.add_argument("--split-name", default=None, help="Optional split name")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional cap for quick tests")
    parser.add_argument("--llm-max-new-tokens", type=int, default=128)
    parser.add_argument("--llm-decoding-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--llm-max-pixels", type=int, default=448)
    parser.add_argument("--date-tag", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--flat-output", action="store_true")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.flat_output:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.now()
        date_tag = args.date_tag or now.strftime("%Y%m%d")
        run_id = args.run_id or now.strftime("%H%M%S")
        out_dir = Path(args.output_dir) / date_tag / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_category_name(name: str) -> str:
    return name.replace("_", " ").strip()


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_output(raw_text: str, allowed_names: list[str]) -> tuple[str, bool]:
    match = re.search(r"<output>(.*?)</output>", raw_text, re.DOTALL | re.IGNORECASE)
    cleaned = match.group(1).strip() if match else raw_text.strip()
    lowered = cleaned.lower()
    allowed_lower = {name.lower(): name for name in allowed_names}
    if lowered in allowed_lower:
        return allowed_lower[lowered], True
    for name in allowed_names:
        if name.lower() in lowered:
            return name, True
    return cleaned, False


def build_single_target_queries(query_coco: dict[str, Any], allowed_ids: set[int]) -> list[dict[str, Any]]:
    images_by_id = {img["id"]: img for img in query_coco["images"]}
    category_name_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in query_coco["categories"]}
    img_to_allowed_cats: dict[int, set[int]] = defaultdict(set)
    for ann in query_coco["annotations"]:
        if ann["category_id"] in allowed_ids:
            img_to_allowed_cats[ann["image_id"]].add(ann["category_id"])

    rows: list[dict[str, Any]] = []
    for image_id, category_ids in img_to_allowed_cats.items():
        if len(category_ids) != 1:
            continue
        category_id = next(iter(category_ids))
        rows.append(
            {
                "image_id": image_id,
                "file_name": images_by_id[image_id]["file_name"],
                "category_id": category_id,
                "category_name": category_name_by_id[category_id],
            }
        )
    rows.sort(key=lambda row: (row["category_id"], row["image_id"]))
    return rows


def build_support_bundle(support_coco: dict[str, Any], support_image_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    images_by_id = {img["id"]: img for img in support_coco["images"]}
    category_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in support_coco["categories"]}
    chosen: list[dict[str, Any]] = []
    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in support_coco["annotations"]:
        anns_by_cat[ann["category_id"]].append(ann)
    for category_id in sorted(anns_by_cat):
        ann = sorted(anns_by_cat[category_id], key=lambda item: (-float(item.get("area", 0.0)), item["image_id"]))[0]
        image = images_by_id[ann["image_id"]]
        chosen.append(
            {
                "category_id": category_id,
                "category_name": category_by_id[category_id],
                "image_id": image["id"],
                "file_name": image["file_name"],
                "path": str((support_image_dir / image["file_name"]).resolve()),
            }
        )
    category_names = [item["category_name"] for item in chosen]
    return category_names, chosen


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"count": 0, "accuracy": 0.0, "valid_parse_rate": 0.0}
    return {
        "count": count,
        "accuracy": sum(1 for row in rows if row["correct"]) / count,
        "valid_parse_rate": sum(1 for row in rows if row["valid_parse"]) / count,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)

    support_coco = load_json(args.support_json)
    query_coco = load_json(args.query_json)
    category_names, support_bundle = build_support_bundle(support_coco, Path(args.support_image_dir))
    allowed_ids = {item["category_id"] for item in support_bundle}
    query_rows = build_single_target_queries(query_coco, allowed_ids)
    if args.query_limit is not None:
        query_rows = query_rows[: args.query_limit]

    private_like_names: set[str] = set()
    if args.split_config and args.split_name:
        split_cfg = load_json(args.split_config)
        split = split_cfg["folds"][args.split_name]
        support_cat_name_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in support_coco["categories"]}
        private_like_names = {support_cat_name_by_id[cat_id] for cat_id in split.get("private_like_ids", []) if cat_id in support_cat_name_by_id}

    caller = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )

    category_list = "\n".join(f"- {name}" for name in category_names)
    support_mapping = "\n".join(
        f"- Image {idx}: {item['category_name']}" for idx, item in enumerate(support_bundle, start=1)
    )
    query_only_prompt = QUERY_ONLY_TEMPLATE.format(category_list=category_list)
    query_support_prompt = QUERY_SUPPORT_TEMPLATE.format(
        num_support=len(support_bundle),
        query_index=len(support_bundle) + 1,
        support_mapping=support_mapping,
    )

    records: list[dict[str, Any]] = []
    support_paths = [item["path"] for item in support_bundle]
    for idx, row in enumerate(query_rows, start=1):
        query_path = str((Path(args.query_image_dir) / row["file_name"]).resolve())
        print(f"[{idx}/{len(query_rows)}] {row['file_name']} gt={row['category_name']}", flush=True)

        raw_query_only = caller.generate(query_path, instruction=query_only_prompt)
        pred_query_only, valid_query_only = parse_output(raw_query_only, category_names)
        records.append(
            {
                **row,
                "mode": "query_only",
                "raw_output": raw_query_only,
                "predicted_category": pred_query_only,
                "valid_parse": valid_query_only,
                "correct": pred_query_only.lower() == row["category_name"].lower(),
                "subgroup": "private_like" if row["category_name"] in private_like_names else "generic",
            }
        )

        raw_query_support = caller.generate_images(support_paths + [query_path], instruction=query_support_prompt)
        pred_query_support, valid_query_support = parse_output(raw_query_support, category_names)
        records.append(
            {
                **row,
                "mode": "query_plus_support16",
                "raw_output": raw_query_support,
                "predicted_category": pred_query_support,
                "valid_parse": valid_query_support,
                "correct": pred_query_support.lower() == row["category_name"].lower(),
                "subgroup": "private_like" if row["category_name"] in private_like_names else "generic",
            }
        )

    summary = {
        "support_json": args.support_json,
        "query_json": args.query_json,
        "single_target_queries": len(query_rows),
        "private_like_names": sorted(private_like_names),
        "overall": {},
        "subgroups": {},
    }

    for mode in ["query_only", "query_plus_support16"]:
        mode_rows = [row for row in records if row["mode"] == mode]
        summary["overall"][mode] = summarize(mode_rows)
        for subgroup in ["private_like", "generic"]:
            subgroup_rows = [row for row in mode_rows if row["subgroup"] == subgroup]
            summary["subgroups"].setdefault(subgroup, {})
            summary["subgroups"][subgroup][mode] = summarize(subgroup_rows)

    summary["gaps"] = {
        "overall_support_minus_query_only_accuracy": (
            summary["overall"]["query_plus_support16"]["accuracy"] - summary["overall"]["query_only"]["accuracy"]
        ),
        "private_like_support_minus_query_only_accuracy": (
            summary["subgroups"]["private_like"]["query_plus_support16"]["accuracy"]
            - summary["subgroups"]["private_like"]["query_only"]["accuracy"]
            if "private_like" in summary["subgroups"] else 0.0
        ),
        "generic_support_minus_query_only_accuracy": (
            summary["subgroups"]["generic"]["query_plus_support16"]["accuracy"]
            - summary["subgroups"]["generic"]["query_only"]["accuracy"]
            if "generic" in summary["subgroups"] else 0.0
        ),
    }

    (output_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "prompts.json").write_text(
        json.dumps(
            {
                "query_only_prompt": query_only_prompt,
                "query_plus_support16_prompt": query_support_prompt,
                "support_bundle": support_bundle,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
