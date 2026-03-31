#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze diagnostic records by subgroup (private-like vs generic) using the pseudo-novel split config."
    )
    parser.add_argument("--records", required=True, help="Path to diagnostic_records.json")
    parser.add_argument("--split-config", required=True, help="Path to pseudo_novel_3split.json")
    parser.add_argument("--split-name", required=True, help="Split name, e.g. split1")
    parser.add_argument("--query-json", required=True, help="COCO query json used for the run, to map category ids")
    parser.add_argument("--output-json", default=None, help="Optional path to save subgroup summary json")
    return parser.parse_args()


def normalize_category_name(name: str) -> str:
    return name.replace("_", " ").strip()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_iou": mean([row["best_iou"] for row in rows]),
        "hit_at_50": mean([1.0 if row["hit_at_50"] else 0.0 for row in rows]),
        "mean_top_score": mean([row["top_score"] for row in rows]),
        "mean_num_boxes": mean([row["num_boxes"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.records).read_text())
    split_cfg = json.loads(Path(args.split_config).read_text())
    query_coco = json.loads(Path(args.query_json).read_text())

    split = split_cfg["folds"][args.split_name]
    private_like_ids = set(split["private_like_ids"])
    categories_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in query_coco["categories"]}
    private_like_names = {categories_by_id[cat_id] for cat_id in private_like_ids if cat_id in categories_by_id}

    subgroup_rows: dict[str, list[dict[str, Any]]] = {
        "private_like": [],
        "generic": [],
    }
    subgroup_exp_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_rows: dict[str, dict[tuple[int, str], dict[str, dict[str, Any]]]] = {
        "private_like": defaultdict(dict),
        "generic": defaultdict(dict),
    }

    for row in records:
        subgroup = "private_like" if row["category_name"] in private_like_names else "generic"
        subgroup_rows[subgroup].append(row)
        subgroup_exp_rows[(subgroup, row["experiment"])].append(row)
        pair_rows[subgroup][(row["image_id"], row["category_name"])][row["experiment"]] = row

    summary: dict[str, Any] = {
        "split_name": args.split_name,
        "private_like_ids": sorted(private_like_ids),
        "private_like_names": sorted(private_like_names),
        "subgroups": {},
    }

    for subgroup, rows in subgroup_rows.items():
        exp_summary = {
            exp: summarize_rows(exp_rows)
            for (sg, exp), exp_rows in subgroup_exp_rows.items()
            if sg == subgroup
        }
        oracle = exp_summary.get("exp1_oracle", {}).get("mean_iou", 0.0)
        inferred = exp_summary.get("exp3_inferred", {}).get("mean_iou", 0.0)
        desc = exp_summary.get("exp2_description", {}).get("mean_iou", 0.0)
        obj = exp_summary.get("exp4_object", {}).get("mean_iou", 0.0)

        object_comp = 0
        inferred_comp = 0
        desc_comp = 0
        pair_count = 0
        for pair_key, exp_map in pair_rows[subgroup].items():
            if len(exp_map) != 4:
                continue
            pair_count += 1
            o = exp_map["exp1_oracle"]["best_iou"]
            if exp_map["exp4_object"]["best_iou"] >= o - 0.05:
                object_comp += 1
            if exp_map["exp3_inferred"]["best_iou"] >= o - 0.05:
                inferred_comp += 1
            if exp_map["exp2_description"]["best_iou"] >= o - 0.05:
                desc_comp += 1

        summary["subgroups"][subgroup] = {
            "experiments": exp_summary,
            "gaps": {
                "exp1_minus_exp3_mean_iou": oracle - inferred,
                "exp1_minus_exp2_mean_iou": oracle - desc,
                "exp1_minus_exp4_mean_iou": oracle - obj,
            },
            "competitive_rates": {
                "object_vs_oracle_within_0.05": object_comp / pair_count if pair_count else 0.0,
                "inferred_vs_oracle_within_0.05": inferred_comp / pair_count if pair_count else 0.0,
                "description_vs_oracle_within_0.05": desc_comp / pair_count if pair_count else 0.0,
            },
            "pair_count": pair_count,
        }

    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
